#!/usr/bin/env python
"""Full validation of a discovered projection function on the entire val set.

Usage:
    cd TSP/POMO/projection_search
    python validate_projection.py [--projection_path ../best_projection.py]
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POMO_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _POMO_DIR)
sys.path.insert(0, os.path.join(_POMO_DIR, ".."))

from pomo_fast_eval import POMOFastEvaluator


def main():
    parser = argparse.ArgumentParser(description="Validate a projection function")
    parser.add_argument(
        "--projection_path",
        type=str,
        default=os.path.join(_POMO_DIR, "best_projection.py"),
        help="Path to projection module (default: ../best_projection.py)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_POMO_DIR, "result", "best_ckpt_2", "checkpoint-best.pt"),
        help="Path to POMO checkpoint",
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0, help="CUDA device number",
    )
    parser.add_argument(
        "--num_instances", type=int, default=0,
        help="Number of instances (0 = all)",
    )
    args = parser.parse_args()

    # Load projection function
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "candidate_projection", args.projection_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    projection_fn = mod.projection

    print(f"Loaded projection from: {args.projection_path}")

    # Full evaluation
    evaluator = POMOFastEvaluator(
        checkpoint_path=args.checkpoint,
        device=f"cuda:{args.cuda_device}",
        num_fast_eval=32,
        use_aug=True,
    )

    num = args.num_instances if args.num_instances > 0 else len(evaluator.val_instances)
    print(f"Evaluating on {num} instances with x8 augmentation...")

    result = evaluator.full_evaluate(projection_fn)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Avg gap (with projection):    {result['avg_gap']:.4f}%")
    print(f"  Avg gap (baseline, no proj):  {result['avg_baseline_gap']:.4f}%")
    print(f"  Improved instances:           {result['improved_instances']}/{result['total_instances']}")
    print(f"  Improvement rate:             {result['improve_rate']*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
