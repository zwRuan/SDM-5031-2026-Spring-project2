"""Quick evaluation script to test augmentation strategies on the validation set.

Usage:
    cd project2
    python eval_quick.py                      # Test baseline (standard 8-fold dihedral)
    python eval_quick.py --best               # Test best discovered augmentation strategy
    python eval_quick.py --aug-factor 32      # Test baseline with more augmentations
    python eval_quick.py --best --aug-factor 64  # Test best strategy with 64 augmentations
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POMO_DIR = os.path.join(PROJECT_ROOT, "TSP", "POMO")
TSP_DIR = os.path.join(PROJECT_ROOT, "TSP")

sys.path.insert(0, POMO_DIR)
sys.path.insert(0, TSP_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from evaluation import (
    MODEL_PARAMS, CHECKPOINT_PATH, VAL_DATA_PATH,
    standard_8fold_augment, evaluate_with_custom_aug, normalize_to_unit_square,
)
from TSPModel import TSPModel as Model
from tsplib_utils import TSPLIBReader, tsplib_cost


def baseline_augment(coords: torch.Tensor, aug_factor: int = 8) -> torch.Tensor:
    """Baseline: standard 8-fold dihedral + extra evenly-spaced rotations."""
    dihedral = standard_8fold_augment(coords)  # always 8

    if aug_factor <= 8:
        return dihedral[:aug_factor]

    # Extra rotations at evenly spaced angles (avoiding dihedral angles)
    num_extra = aug_factor - 8
    angles = np.linspace(np.pi / 16, 2 * np.pi - np.pi / 16, num_extra)
    extras = []
    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        centered = coords - 0.5
        x_r = centered[:, :, 0] * cos_a - centered[:, :, 1] * sin_a
        y_r = centered[:, :, 0] * sin_a + centered[:, :, 1] * cos_a
        rotated = torch.stack([x_r, y_r], dim=2)
        r_min = rotated.min(dim=1, keepdim=True).values
        r_max = rotated.max(dim=1, keepdim=True).values
        scale = (r_max - r_min).max(dim=2, keepdim=True).values.clamp(min=1e-8)
        extras.append((rotated - r_min) / scale)

    return torch.cat([dihedral] + extras, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Quick eval of augmentation strategy on val set")
    parser.add_argument("--best", action="store_true", help="Use the best discovered augmentation strategy")
    parser.add_argument("--aug-factor", type=int, default=32, help="Number of augmentations (default: 32)")
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs best side-by-side")
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
        torch.cuda.set_device(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    # Load model
    model = Model(**MODEL_PARAMS)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Determine augmentation function
    if args.best or args.compare:
        try:
            from best_algorithm import generate_augmentations as best_aug_func
            print(f"[Loaded] best_algorithm.py augmentation strategy")
        except ImportError:
            print("ERROR: best_algorithm.py not found or does not have generate_augmentations!")
            return
    
    if not args.compare:
        if args.best:
            aug_func = best_aug_func
            mode_name = "Best LLM strategy"
        else:
            aug_func = baseline_augment
            mode_name = "Baseline (dihedral + rotations)"
        
        print(f"[Mode] {mode_name}")
        print(f"[Aug factor] {args.aug_factor}")

    print(f"\nEvaluating on: {VAL_DATA_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("-" * 80)

    start_time = time.time()

    if args.compare:
        # Side-by-side comparison
        print(f"{'Instance':<14} | {'Baseline-8':>10} | {'Best-'+str(args.aug_factor):>10} | {'Delta':>8}")
        print("-" * 60)
        gaps_base = []
        gaps_best = []
        
        for root, _, files in os.walk(VAL_DATA_PATH):
            for file in sorted(files):
                if not file.endswith('.tsp'):
                    continue
                full_path = os.path.join(root, file)
                name, dimension, locs, ew_type = TSPLIBReader(full_path)
                if name is None:
                    continue
                optimal = tsplib_cost.get(name, None)
                if optimal is None:
                    continue

                coords_orig_np = np.array(locs, dtype=np.float32)
                coords_orig = torch.from_numpy(coords_orig_np).to(device)
                node_coord = coords_orig[None, :, :]
                nodes_xy = normalize_to_unit_square(node_coord)

                # Baseline with standard 8-fold
                score_base = evaluate_with_custom_aug(
                    model, device, nodes_xy, coords_orig, ew_type, standard_8fold_augment, 8)
                gap_base = (score_base - optimal) / optimal * 100
                gaps_base.append(gap_base)

                # Best strategy
                score_best = evaluate_with_custom_aug(
                    model, device, nodes_xy, coords_orig, ew_type, best_aug_func, args.aug_factor)
                gap_best = (score_best - optimal) / optimal * 100
                gaps_best.append(gap_best)

                delta = gap_best - gap_base
                marker = "✓" if delta < -0.001 else ("✗" if delta > 0.001 else "=")
                print(f"  {name:12s} | {gap_base:>9.4f}% | {gap_best:>9.4f}% | {delta:>+7.4f}% {marker}")

        print("-" * 60)
        avg_base = np.mean(gaps_base)
        avg_best = np.mean(gaps_best)
        improved = sum(1 for b, s in zip(gaps_base, gaps_best) if s < b - 1e-6)
        print(f"  {'AVG':<12} | {avg_base:>9.4f}% | {avg_best:>9.4f}% | {avg_best-avg_base:>+7.4f}%")
        print(f"\nImproved: {improved}/{len(gaps_base)} instances ({100*improved/len(gaps_base):.0f}%)")
        print(f"Relative improvement: {(avg_base-avg_best)/avg_base*100:.1f}%")
    else:
        # Single evaluation
        aug_gaps = []
        for root, _, files in os.walk(VAL_DATA_PATH):
            for file in sorted(files):
                if not file.endswith('.tsp'):
                    continue
                full_path = os.path.join(root, file)
                name, dimension, locs, ew_type = TSPLIBReader(full_path)
                if name is None:
                    continue
                optimal = tsplib_cost.get(name, None)
                if optimal is None:
                    continue

                coords_orig_np = np.array(locs, dtype=np.float32)
                coords_orig = torch.from_numpy(coords_orig_np).to(device)
                node_coord = coords_orig[None, :, :]
                nodes_xy = normalize_to_unit_square(node_coord)

                try:
                    score = evaluate_with_custom_aug(
                        model, device, nodes_xy, coords_orig, ew_type, aug_func, args.aug_factor)
                    gap = (score - optimal) / optimal * 100
                    aug_gaps.append(gap)
                    print(f"  {name:12s} (N={dimension:4d}): aug_gap={gap:6.4f}%")
                except Exception as e:
                    print(f"  {name:12s}: FAILED - {e}")

        print("-" * 80)
        if aug_gaps:
            print(f"avg_aug_gap: {np.mean(aug_gaps):.4f}%")

    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
