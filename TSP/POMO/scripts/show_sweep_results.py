#!/usr/bin/env python3
"""Aggregate and display results from all phased fine-tune runs.

Scans ``result/`` for directories that contain a ``finetune_phased_config.json``
(i.e. runs launched by ``run_sweep.sh`` / ``run_finetune_phased.sh``), then
reads:
  - ``finetune_phased_config.json``  → hyperparameters
  - ``eval_*.json``                  → avg_aug_gap  (from validate_phased.sh)
  - ``log.txt``                      → phase_best_score (proxy when eval not run)

Prints a sorted table to the terminal and optionally saves ``sweep_summary.csv``
next to the result directories.

Usage::

    # From TSP/POMO/:
    python scripts/show_sweep_results.py
    python scripts/show_sweep_results.py --result_dir result --sort avg_aug_gap
    python scripts/show_sweep_results.py --csv sweep_summary.csv
    python scripts/show_sweep_results.py --filter "ablation=all_three"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ────────────────────────────────────────────────────────────────────────────
# Column definitions — (display_name, config_key_or_None, width)
# ────────────────────────────────────────────────────────────────────────────
DISPLAY_COLS = [
    ("run_dir",       None,                      36),
    ("ablation",      "ablation",                12),
    ("B",             "total_finetune_epochs",    4),
    ("knn_k",         "knn_k",                    5),
    ("knn_v",         "knn_bias_value",           5),
    ("backbone_lr",   "backbone_lr",             10),
    ("phase3_lr",     "phase3_lr",               10),
    ("alpha",         "leader_alpha",             5),
    ("alpha_f",       "leader_alpha_final",       6),
    ("bias",          "enable_bias",              4),
    ("msc",           "enable_msc",               4),
    ("ldr",           "enable_leader",            4),
    ("p1_best",       None,                      10),   # from log.txt
    ("p2_best",       None,                      10),
    ("p3_best",       None,                      10),
    ("aug_gap",       None,                      10),   # from eval_*.json (best)
    ("no_aug_gap",    None,                      10),
]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def parse_log_phase_bests(log_path: Path) -> dict[str, float]:
    """Extract phase_best_score lines from log.txt."""
    bests: dict[str, float] = {}
    if not log_path.exists():
        return bests
    # Format logged by run_phase:
    # [PHASE 1_bias] phase_best_score=0.0624  ...
    pattern = re.compile(
        r"\[PHASE\s+(?P<name>\S+)\].*phase_best_score=(?P<score>[0-9.eE+\-]+)"
    )
    # Also match the final summary line:
    # 1_bias: epochs=... phase_best_score=0.0624 -> ...
    summary_pattern = re.compile(
        r"(?P<name>\S+):\s+epochs=.*phase_best_score=(?P<score>[0-9.eE+\-]+)"
    )
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            for pat in (pattern, summary_pattern):
                m = pat.search(line)
                if m:
                    bests[m.group("name")] = float(m.group("score"))
    return bests


def parse_eval_jsons(run_dir: Path) -> dict[str, dict]:
    """Return {checkpoint_name: {avg_aug_gap, avg_no_aug_gap}} for all eval_*.json."""
    results: dict[str, dict] = {}
    for p in sorted(run_dir.glob("eval_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        results[p.stem] = {
            "avg_aug_gap":    data.get("avg_aug_gap"),
            "avg_no_aug_gap": data.get("avg_no_aug_gap"),
        }
    return results


def best_eval(eval_results: dict[str, dict]) -> tuple[float | None, float | None]:
    """Return (best_avg_aug_gap, corresponding_avg_no_aug_gap) across all checkpoints."""
    best_gap = None
    best_no_aug = None
    for v in eval_results.values():
        g = v.get("avg_aug_gap")
        if g is None:
            continue
        if best_gap is None or g < best_gap:
            best_gap = g
            best_no_aug = v.get("avg_no_aug_gap")
    return best_gap, best_no_aug


def collect_runs(result_dir: Path) -> list[dict[str, Any]]:
    """Walk result_dir and collect metadata for every phased run."""
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(result_dir.iterdir()):
        cfg_path = run_dir / "finetune_phased_config.json"
        if not cfg_path.exists():
            continue

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

        log_path = run_dir / "log.txt"
        bests = parse_log_phase_bests(log_path)
        eval_results = parse_eval_jsons(run_dir)
        aug_gap, no_aug_gap = best_eval(eval_results)

        row: dict[str, Any] = {
            "run_dir":   run_dir.name,
            "_run_path": str(run_dir),
            "p1_best":   bests.get("1_bias"),
            "p2_best":   bests.get("2_msc"),
            "p3_best":   bests.get("3_leader"),
            "aug_gap":   aug_gap,
            "no_aug_gap": no_aug_gap,
        }
        for _, key, _ in DISPLAY_COLS:
            if key and key not in row:
                row[key] = cfg.get(key)

        rows.append(row)

    return rows


def apply_filter(rows: list[dict], filter_str: str) -> list[dict]:
    """Filter rows by 'key=value' predicates (comma-separated)."""
    if not filter_str:
        return rows
    predicates: list[tuple[str, str]] = []
    for part in filter_str.split(","):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            predicates.append((k.strip(), v.strip()))
    filtered = []
    for row in rows:
        ok = True
        for k, v in predicates:
            cell = str(row.get(k, ""))
            if cell != v:
                ok = False
                break
        if ok:
            filtered.append(row)
    return filtered


def fmt(val: Any, width: int) -> str:
    """Format a cell value for the table."""
    if val is None:
        s = "-"
    elif isinstance(val, float):
        s = f"{val:.4f}"
    elif isinstance(val, bool):
        s = "Y" if val else "N"
    else:
        s = str(val)
    # Truncate if too long.
    if len(s) > width:
        s = s[: width - 1] + "…"
    return s.ljust(width)


def print_table(rows: list[dict], sort_key: str) -> None:
    """Print a terminal table sorted by sort_key (ascending, None last)."""
    def sort_val(row: dict):
        v = row.get(sort_key)
        return (1, 0.0) if v is None else (0, float(v))

    rows = sorted(rows, key=sort_val)

    # Header
    header = "  ".join(fmt(col[0], col[2]) for col in DISPLAY_COLS)
    sep = "  ".join("-" * col[2] for col in DISPLAY_COLS)
    print(header)
    print(sep)

    for row in rows:
        cells = []
        for name, key, width in DISPLAY_COLS:
            if key:
                val = row.get(key)
            else:
                val = row.get(name)
            cells.append(fmt(val, width))
        print("  ".join(cells))

    print(sep)
    print(f"  {len(rows)} run(s)")


def save_csv(rows: list[dict], csv_path: Path) -> None:
    col_names = [col[0] for col in DISPLAY_COLS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=col_names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Normalise keys to display names.
            out: dict[str, Any] = {}
            for name, key, _ in DISPLAY_COLS:
                out[name] = row.get(key if key else name)
            writer.writerow(out)
    print(f"CSV saved → {csv_path}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Show aggregated results of all phased fine-tune runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--result_dir", default="result",
        help="Top-level results directory to scan.",
    )
    p.add_argument(
        "--sort", default="aug_gap",
        help=(
            "Column to sort by (ascending). "
            "Useful values: aug_gap, p3_best, p2_best, ablation."
        ),
    )
    p.add_argument(
        "--filter", default="",
        help=(
            "Comma-separated 'key=value' predicates to filter rows. "
            "Example: --filter 'ablation=all_three,knn_k=30'"
        ),
    )
    p.add_argument(
        "--csv", default="",
        help="If set, also write results to this CSV file.",
    )
    p.add_argument(
        "--validate_missing", action="store_true",
        help=(
            "Print validate_phased.sh commands for runs that don't yet have "
            "eval_*.json (i.e. avg_aug_gap not yet measured)."
        ),
    )
    return p


def main() -> None:
    # Auto-locate TSP/POMO from the script's own location so you can call
    # the script from any working directory.
    script_dir = Path(__file__).resolve().parent
    pomo_dir   = script_dir.parent
    os.chdir(pomo_dir)

    args = build_parser().parse_args()
    result_dir = Path(args.result_dir)

    if not result_dir.exists():
        print(f"[show_sweep] result dir '{result_dir}' not found; no runs yet.", file=sys.stderr)
        sys.exit(0)

    rows = collect_runs(result_dir)
    if not rows:
        print("[show_sweep] No phased runs found (looking for finetune_phased_config.json).", file=sys.stderr)
        sys.exit(0)

    rows = apply_filter(rows, args.filter)
    if not rows:
        print("[show_sweep] No runs match the filter.", file=sys.stderr)
        sys.exit(0)

    print(f"\n[show_sweep] {len(rows)} run(s) in '{result_dir}'  (sorted by {args.sort})\n")
    print_table(rows, args.sort)

    if args.csv:
        save_csv(rows, Path(args.csv))

    if args.validate_missing:
        missing = [r for r in rows if r.get("aug_gap") is None]
        if missing:
            print("\n[show_sweep] Runs without avg_aug_gap (validate_phased.sh not run yet):")
            for r in missing:
                print(f"  bash scripts/validate_phased.sh {r['_run_path']}")
        else:
            print("\n[show_sweep] All runs already have avg_aug_gap.")


if __name__ == "__main__":
    main()
