#!/usr/bin/env python
"""LLM-guided search for TSP coordinate projection functions.

Uses gpt-4o-mini + EoH (Evolution of Heuristics) to discover a ``normalize()``
function that preprocesses TSP coordinates before the POMO neural solver.

Usage:
    cd TSP/POMO/projection_search
    python run_search.py

Output:
    - best_projection.py          The best discovered function (ready for test-time use)
    - logs/<timestamp>_POMO*/    Full search logs: every generated function,
                                  fitness history, and final population.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from datetime import datetime

# ---------------------------------------------------------------------------
# Path bootstrap — the LLM4AD core is bundled inside the TTPL repo.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_TTPL_DIR = os.path.join(_PROJECT_ROOT, "src", "TTPL", "TTPL")
_POMO_DIR = os.path.join(_PROJECT_ROOT, "TSP", "POMO")

for _p in [_TTPL_DIR, _POMO_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytz
import torch

# LLM4AD framework (from TTPL bundled copy)
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.task import import_all_evaluation_classes
from llm4ad.method import import_all_method_classes_from_subfolders
from llm4ad.tools.llm import import_all_llm_classes_from_subfolders

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_HOST = "https://api.ai-gaochao.cn/v1"
API_KEY = "sk-VwEyJuZZKoiLgJvTF549C9D131Ae4b98A110Ca706250Bb0e"
API_MODEL = "gpt-4o-mini"

DEFAULT_CHECKPOINT = os.path.join(
    _POMO_DIR, "result", "best_ckpt_2", "checkpoint-best.pt"
)

# ---------------------------------------------------------------------------
# Dynamic import of discoverable classes (same pattern as run_TTPL.py)
# ---------------------------------------------------------------------------
_llm4ad_base = os.path.join(_TTPL_DIR, "llm4ad")
import_all_evaluation_classes(os.path.join(_llm4ad_base, "task"))
import_all_method_classes_from_subfolders(os.path.join(_llm4ad_base, "method"))
import_all_llm_classes_from_subfolders(os.path.join(_llm4ad_base, "tools", "llm"))


def build_parser():
    p = argparse.ArgumentParser(
        description="Search for TSP coordinate projection functions via LLM+EoH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to POMO checkpoint (default: best_ckpt_2/checkpoint-best.pt)",
    )
    p.add_argument(
        "--max_sample_nums",
        type=int,
        default=1000,
        help="Total LLM API calls budget (default: 1000)",
    )
    p.add_argument(
        "--max_generations",
        type=int,
        default=50,
        help="Max EoH generations (default: 50)",
    )
    p.add_argument(
        "--pop_size",
        type=int,
        default=10,
        help="Population size (default: 10)",
    )
    p.add_argument(
        "--num_samplers",
        type=int,
        default=4,
        help="Parallel samplers (default: 4)",
    )
    p.add_argument(
        "--num_evaluators",
        type=int,
        default=4,
        help="Parallel evaluators (default: 4)",
    )
    p.add_argument(
        "--num_fast_eval",
        type=int,
        default=32,
        help="Number of validation instances per fitness eval (default: 32)",
    )
    p.add_argument(
        "--use_aug",
        action="store_true",
        help="Use x8 augmentation during fitness evaluation (slower but more accurate)",
    )
    p.add_argument(
        "--timeout_seconds",
        type=int,
        default=60,
        help="Timeout per evaluation (default: 60s)",
    )
    p.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device number (default: 0)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (prints generated code to stdout)",
    )
    return p


def main():
    args = build_parser().parse_args()

    # --- LLM ---
    llm = HttpsApi(
        host=API_HOST,
        key=API_KEY,
        model=API_MODEL,
        timeout=60,
    )
    print(f"[run_search] LLM: {API_MODEL} @ {API_HOST}")

    # --- Evaluation ---
    evaluation = POMOProjectionEvaluation(
        timeout_seconds=args.timeout_seconds,
        checkpoint_path=args.checkpoint,
        num_fast_eval=args.num_fast_eval,
        use_aug=args.use_aug,
        device=f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu",
    )

    # --- Log directory ---
    ts = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        _SCRIPT_DIR,
        "logs",
        f"{ts}_POMOProjection_EoH",
    )
    os.makedirs(log_dir, exist_ok=True)

    # --- Profiler ---
    profiler = EoHProfiler(
        evaluation_name="POMOProjectionEvaluation",
        method_name="EoH",
        log_dir=log_dir,
        log_style="complex",
        create_random_path=False,
        final_log_dir=log_dir,
    )

    # --- Method (EoH) ---
    method = EoH(
        llm=llm,
        profiler=profiler,
        evaluation=evaluation,
        max_sample_nums=args.max_sample_nums,
        max_generations=args.max_generations,
        pop_size=args.pop_size,
        num_samplers=args.num_samplers,
        num_evaluators=args.num_evaluators,
        selection=2,
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,
        debug_mode=args.debug,
    )

    print(f"[run_search] EoH: pop_size={args.pop_size}, "
          f"max_generations={args.max_generations}, "
          f"max_samples={args.max_sample_nums}")
    print(f"[run_search] Fast eval on {args.num_fast_eval} instances "
          f"(aug={args.use_aug})")
    print(f"[run_search] Logs: {log_dir}")
    print(f"[run_search] Starting search...\n")

    # --- Run ---
    method.run()

    print(f"\n[run_search] Search complete. Logs: {log_dir}")

    # --- Post-process: find and save best function ---
    _save_best_from_logs(log_dir)


def _save_best_from_logs(log_dir: str):
    """Extract the best function from EoH logs and save as best_projection.py."""
    import json
    import glob

    # Try to find the population file
    pop_files = glob.glob(os.path.join(log_dir, "*population*.json"))
    best_files = glob.glob(os.path.join(log_dir, "*best*.json"))
    all_json = glob.glob(os.path.join(log_dir, "*.json"))

    best_code = None
    best_score = float("-inf")

    for fpath in pop_files + best_files + all_json:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        # Handle different log formats
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                score = item.get("score") or item.get("fitness") or float("-inf")
                code = item.get("code") or item.get("program") or item.get("function")
                if code and float(score) > best_score:
                    best_score = float(score)
                    best_code = str(code)

    if best_code is None:
        print("[run_search] WARNING: Could not extract best function from logs.")
        print("[run_search] Check the log directory for generated functions.")
        return

    # Clean the code and save
    best_code = _clean_generated_code(best_code)
    dest = os.path.join(_POMO_DIR, "best_projection.py")

    header = textwrap.dedent("""\
    # Auto-discovered coordinate projection function for POMO TSP solver.
    # Generated by LLM4AD + EoH search (gpt-4o-mini).
    # This function is applied to TSP coordinates BEFORE the POMO model.
    #
    # To use: place this file in TSP/POMO/ alongside test.py.
    # The test pipeline auto-detects and applies it.

    import torch

    """)

    with open(dest, "w") as f:
        f.write(header)
        f.write(best_code)
        f.write("\n")

    print(f"[run_search] Best function saved to: {dest}")
    print(f"[run_search] Best fitness (neg gap): {best_score:.4f}")


def _clean_generated_code(code: str) -> str:
    """Strip markdown fences and leading/trailing whitespace from LLM output."""
    code = code.strip()
    # Remove ```python ... ``` wrappers
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first line if it's a fence
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's a fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    return code


if __name__ == "__main__":
    # The dynamic imports above already added POMOProjectionEvaluation to globals.
    main()
