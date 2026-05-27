"""Final evaluation: Compare Project 1 (standard 8-fold aug) vs Project 2 (LLM-designed augmentation).

Usage:
    cd project2
    python run_final_eval.py
    python run_final_eval.py --data-dir ../TSP/data/test  # Use test set
    python run_final_eval.py --aug-factor 128             # Custom aug factor

Outputs a comparison table showing per-instance gaps and improvement statistics.
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


def main():
    parser = argparse.ArgumentParser(description="Final eval: Project 1 vs Project 2")
    parser.add_argument("--data-dir", type=str, default=VAL_DATA_PATH,
                        help="Path to TSP instance directory")
    parser.add_argument("--aug-factor", type=int, default=None,
                        help="Aug factor for Project 2 (default: min(N*8, 800))")
    args = parser.parse_args()

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

    # Load best augmentation strategy
    try:
        from best_algorithm import generate_augmentations as best_aug_func
    except ImportError:
        print("ERROR: best_algorithm.py not found or missing generate_augmentations!")
        return

    # Collect instances
    instances = []
    data_dir = args.data_dir
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            if not file.endswith('.tsp'):
                continue
            full_path = os.path.join(root, file)
            name, dimension, locs, ew_type = TSPLIBReader(full_path)
            if name is None:
                continue
            optimal = tsplib_cost.get(name, None)
            if optimal is None:
                print(f"  SKIP {name} (no known optimal)")
                continue
            instances.append((name, dimension, locs, ew_type, optimal, full_path))

    if not instances:
        print(f"No valid instances found in {data_dir}")
        return

    # ---- Evaluate ----
    print("=" * 80)
    print("FINAL EVALUATION: Project 1 (8-fold dihedral) vs Project 2 (Golden-Dihedral)")
    print("=" * 80)
    print(f"Data: {data_dir} ({len(instances)} instances)")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Project 2 aug_factor: {'adaptive (min(N*8, 800))' if args.aug_factor is None else args.aug_factor}")
    print()
    print(f"{'Instance':<14} {'N':>4} {'Aug':>4} {'P1 (8-fold)':>12} {'P2 (ours)':>12} {'Delta':>8} {'Status':>7}")
    print("-" * 70)

    gaps_p1 = []
    gaps_p2 = []
    start_time = time.time()

    for name, dimension, locs, ew_type, optimal, _ in instances:
        coords_orig_np = np.array(locs, dtype=np.float32)
        coords_orig = torch.from_numpy(coords_orig_np).to(device)
        node_coord = coords_orig[None, :, :]
        nodes_xy = normalize_to_unit_square(node_coord)

        # Project 1: standard 8-fold dihedral
        score_p1 = evaluate_with_custom_aug(
            model, device, nodes_xy, coords_orig, ew_type,
            standard_8fold_augment, 8
        )
        gap_p1 = (score_p1 - optimal) / optimal * 100
        gaps_p1.append(gap_p1)

        # Project 2: LLM-designed augmentation
        if args.aug_factor is not None:
            aug_f = args.aug_factor
        else:
            aug_f = min(dimension * 8, 800)

        score_p2 = evaluate_with_custom_aug(
            model, device, nodes_xy, coords_orig, ew_type,
            best_aug_func, aug_f
        )
        gap_p2 = (score_p2 - optimal) / optimal * 100
        gaps_p2.append(gap_p2)

        delta = gap_p2 - gap_p1
        if delta < -1e-6:
            status = "✓ BETTER"
        elif delta > 1e-6:
            status = "✗ WORSE"
        else:
            status = "= SAME"

        print(f"{name:<14} {dimension:>4} {aug_f:>4} {gap_p1:>11.4f}% {gap_p2:>11.4f}% {delta:>+7.4f}% {status}")

    elapsed = time.time() - start_time
    print("-" * 70)

    avg_p1 = np.mean(gaps_p1)
    avg_p2 = np.mean(gaps_p2)
    improved = sum(1 for a, b in zip(gaps_p1, gaps_p2) if b < a - 1e-6)
    same = sum(1 for a, b in zip(gaps_p1, gaps_p2) if abs(b - a) < 1e-6)
    worse = len(gaps_p1) - improved - same
    total = len(gaps_p1)

    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  {'Metric':<30} {'Project 1':>10} {'Project 2':>10}")
    print(f"  {'-'*50}")
    print(f"  {'avg_aug_gap (%)':<30} {avg_p1:>10.4f} {avg_p2:>10.4f}")
    print(f"  {'Gap reduction':<30} {'':>10} {avg_p1-avg_p2:>+10.4f}")
    print(f"  {'Relative improvement':<30} {'':>10} {(avg_p1-avg_p2)/max(avg_p1,1e-6)*100:>9.1f}%")
    print()
    print(f"  Instances improved: {improved}/{total} ({100*improved/total:.0f}%)")
    print(f"  Instances same:     {same}/{total} ({100*same/total:.0f}%)")
    print(f"  Instances worse:    {worse}/{total} ({100*worse/total:.0f}%)")
    print()
    requirement_met = improved / total >= 0.6
    print(f"  Requirement (>=60% improved): {'PASS ✓' if requirement_met else 'FAIL ✗'}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()