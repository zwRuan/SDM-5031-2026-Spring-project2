"""Project 2 Test Script: LLM-Designed Augmentation Strategy for POMO TSP.

This is the main entry point for the teaching assistant to evaluate Project 2.
It loads the same POMO model from Project 1 and applies our LLM-designed
augmentation strategy (Halton-Sequence Quasi-Random Rotations) at test time.

Usage:
    cd project2
    python test.py --data-dir <path_to_test_instances>
    python test.py --data-dir ../TSP/data/val          # validation set
    python test.py --data-dir ../TSP/data/test         # test set
    python test.py --data-dir ../TSP/data/val --compare  # compare with Project 1

The augmentation strategy:
  - Standard POMO uses 8 dihedral-group transformations
  - Our method uses Halton quasi-random sequence (base 2) to generate rotation
    angles, providing low-discrepancy angular coverage far superior to the
    fixed 8-fold dihedral group
  - aug_factor = min(problem_size * 8, 800)
  - Compliant with rule: aug_factor <= problem_size * 8
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

from TSPModel import TSPModel as Model
from TSPEnv import TSPEnv as Env
from tsplib_utils import TSPLIBReader, tsplib_cost
from best_algorithm import generate_augmentations


# ---- Model Configuration (same as Project 1) ----
MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}

CHECKPOINT_PATH = os.path.join(POMO_DIR, "result", "best_ckpt_2", "checkpoint-best.pt")


def normalize_to_unit_square(node_xy: torch.Tensor) -> torch.Tensor:
    """Normalize coordinates to [0,1] with uniform scaling (preserves aspect ratio)."""
    xy_max = torch.max(node_xy, dim=1, keepdim=True).values
    xy_min = torch.min(node_xy, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1
    return (node_xy - xy_min) / ratio.expand(-1, 1, 2)


def solve_instance(model, device, nodes_xy_normalized, coords_orig, ew_type, aug_factor):
    """Solve a single TSP instance using our augmentation strategy.

    Returns the best tour length found (in original coordinates).
    """
    # Generate augmented versions using our LLM-designed strategy
    problems = generate_augmentations(nodes_xy_normalized, aug_factor)
    problems = problems.clamp(0.0, 1.0)

    effective_batch = problems.size(0)
    problem_size = problems.size(1)

    env = Env(problem_size=problem_size, pomo_size=problem_size)
    env.batch_size = effective_batch
    env.problems = problems.to(device)
    env.BATCH_IDX = torch.arange(effective_batch, device=device)[:, None].expand(
        effective_batch, env.pomo_size
    )
    env.POMO_IDX = torch.arange(env.pomo_size, device=device)[None, :].expand(
        effective_batch, env.pomo_size
    )
    env.original_node_xy_lib = coords_orig[None, :, :]
    env.edge_weight_type = ew_type

    model.eval()
    with torch.no_grad():
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected, lib_mode=True)

    tour_lens = (-reward).detach()
    return float(tour_lens.min().item())


def main():
    parser = argparse.ArgumentParser(
        description="Project 2: LLM-Designed Augmentation for POMO TSP"
    )
    # Data path (support both --data-dir and --data_path)
    parser.add_argument(
        "--data-dir", "--data_path", type=str,
        default=os.path.join(TSP_DIR, "data", "val"),
        help="Directory containing .tsp files",
        dest="data_dir",
    )
    parser.add_argument(
        "--aug-factor", type=int, default=None,
        help="Augmentation factor (default: min(N*8, 800))"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run Project 1 baseline for comparison"
    )
    # Checkpoint path (support both --checkpoint and --checkpoint_path)
    parser.add_argument(
        "--checkpoint", "--checkpoint_path", type=str,
        default=CHECKPOINT_PATH,
        help="Path to model checkpoint",
        dest="checkpoint",
    )
    # JSON output for automated evaluation
    parser.add_argument(
        "--output_json", "--output-json", type=str, default=None,
        help="Path to write JSON evaluation results",
        dest="output_json",
    )
    # CUDA control (TA standard interface)
    parser.add_argument(
        "--use_cuda", type=int, default=1,
        help="Whether to use CUDA (1=yes, 0=no)"
    )
    parser.add_argument(
        "--cuda_device_num", type=int, default=0,
        help="CUDA device index"
    )
    # Augmentation toggle (always enabled for Project 2)
    parser.add_argument(
        "--augmentation_enable", type=int, default=1,
        help="Enable augmentation (1=yes, 0=no; always 1 for Project 2)"
    )
    # Detailed logging
    parser.add_argument(
        "--detailed_log", action="store_true",
        help="Enable detailed per-instance logging"
    )
    args = parser.parse_args()

    # Device setup
    use_cuda = bool(args.use_cuda) and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda', args.cuda_device_num)
        torch.cuda.set_device(args.cuda_device_num)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    # Load model
    model = Model(**MODEL_PARAMS)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Collect instances
    data_dir = args.data_dir
    instances = []
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            if not file.endswith('.tsp'):
                continue
            full_path = os.path.join(root, file)
            name, dimension, locs, ew_type = TSPLIBReader(full_path)
            if name is None:
                continue
            optimal = tsplib_cost.get(name, None)
            instances.append((name, dimension, locs, ew_type, optimal, full_path))

    if not instances:
        print(f"No .tsp files found in {data_dir}")
        return

    # Header
    print("=" * 80)
    print("Project 2: LLM-Designed Augmentation Strategy (Halton-Sequence Rotations)")
    print("=" * 80)
    print(f"  Data: {data_dir} ({len(instances)} instances)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Aug factor: {'adaptive min(N*8, 800)' if args.aug_factor is None else args.aug_factor}")
    print(f"  Device: {device}")
    print()

    if args.compare:
        from TSProblemDef import augment_xy_data_by_8_fold
        print(f"{'Instance':<14} {'N':>4} {'Optimal':>10} {'P1 (8-fold)':>12} {'P2 (ours)':>12} {'Gap_P1':>8} {'Gap_P2':>8} {'Status':>7}")
    else:
        print(f"{'Instance':<14} {'N':>4} {'Optimal':>10} {'Score':>10} {'Gap (%)':>8}")
    print("-" * 80)

    gaps = []
    gaps_p1 = []
    results = []  # For JSON output
    start_time = time.time()

    for name, dimension, locs, ew_type, optimal, filepath in instances:
        coords_orig_np = np.array(locs, dtype=np.float32)
        coords_orig = torch.from_numpy(coords_orig_np).to(device)
        node_coord = coords_orig[None, :, :]
        nodes_xy = normalize_to_unit_square(node_coord)

        # Determine aug factor
        aug_f = args.aug_factor if args.aug_factor is not None else min(dimension * 8, 800)

        # If augmentation is disabled, fall back to standard 8-fold
        if not args.augmentation_enable:
            aug_f = 8

        # Project 2: our augmentation strategy
        score_p2 = solve_instance(model, device, nodes_xy, coords_orig, ew_type, aug_f)

        result_entry = {
            "instance": name,
            "dimension": dimension,
            "score": score_p2,
            "aug_factor": aug_f,
        }

        if optimal is not None:
            gap_p2 = (score_p2 - optimal) / optimal * 100
            gaps.append(gap_p2)
            result_entry["optimal"] = optimal
            result_entry["gap_percent"] = round(gap_p2, 4)
        else:
            gap_p2 = None

        if args.compare and optimal is not None:
            # Project 1 baseline: standard 8-fold
            from evaluation import standard_8fold_augment, evaluate_with_custom_aug
            score_p1 = evaluate_with_custom_aug(
                model, device, nodes_xy, coords_orig, ew_type, standard_8fold_augment, 8
            )
            gap_p1 = (score_p1 - optimal) / optimal * 100
            gaps_p1.append(gap_p1)
            result_entry["p1_score"] = score_p1
            result_entry["p1_gap_percent"] = round(gap_p1, 4)

            status = "✓" if gap_p2 < gap_p1 - 1e-6 else ("=" if abs(gap_p2 - gap_p1) < 1e-6 else "✗")
            print(f"{name:<14} {dimension:>4} {optimal:>10.1f} {score_p1:>11.1f} {score_p2:>11.1f} {gap_p1:>7.4f}% {gap_p2:>7.4f}% {status:>7}")
        elif optimal is not None:
            print(f"{name:<14} {dimension:>4} {optimal:>10.1f} {score_p2:>10.1f} {gap_p2:>7.4f}%")
        else:
            print(f"{name:<14} {dimension:>4} {'N/A':>10} {score_p2:>10.1f} {'N/A':>8}")

        results.append(result_entry)

    elapsed = time.time() - start_time
    print("-" * 80)

    summary = {}
    if gaps:
        avg_gap = float(np.mean(gaps))
        print(f"\n  avg_aug_gap: {avg_gap:.4f}%")
        summary["avg_aug_gap_percent"] = round(avg_gap, 4)
        summary["num_instances"] = len(gaps)
        if args.compare and gaps_p1:
            avg_p1 = float(np.mean(gaps_p1))
            print(f"  P1 baseline: {avg_p1:.4f}%")
            improved = sum(1 for a, b in zip(gaps_p1, gaps) if b < a - 1e-6)
            print(f"  Improved:    {improved}/{len(gaps)} ({100*improved/len(gaps):.0f}%)")
            summary["p1_avg_gap_percent"] = round(avg_p1, 4)
            summary["improved_count"] = improved
    print(f"  Time: {elapsed:.1f}s")
    summary["elapsed_seconds"] = round(elapsed, 1)

    # Write JSON output if requested
    if args.output_json:
        import json
        output = {
            "summary": summary,
            "method": "Halton-Sequence Quasi-Random Rotations",
            "instances": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results written to: {args.output_json}")


if __name__ == "__main__":
    main()
