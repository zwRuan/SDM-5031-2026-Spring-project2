"""Ablation summary writer.

Writes / appends to ``<summary_dir>/summary.csv`` and ``<summary_dir>/summary.json``
plus a per-instance JSON dump at ``<summary_dir>/per_instance_<run_name>.json``.

Schema of each summary row:
    run_name, avg_aug_gap, avg_no_aug_gap, solved, total,
    aug_factor, rerank_enabled, two_opt_enabled, distance_bias_enabled,
    knn_bias_enabled, leader_reward_enabled,
    total_runtime_ms, win_rate_vs_baseline, lose_rate_vs_baseline, tie_rate_vs_baseline
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional


CSV_COLUMNS = [
    "run_name", "timestamp", "avg_aug_gap", "avg_no_aug_gap", "solved", "total",
    "aug_factor", "rerank_enabled", "two_opt_enabled",
    "distance_bias_enabled", "knn_bias_enabled",
    "leader_reward_enabled",
    "total_runtime_ms",
    "win_rate_vs_baseline", "lose_rate_vs_baseline", "tie_rate_vs_baseline",
    "checkpoint_path", "data_path",
]


def _safe_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default)


def _compute_winrate(
    payload: Dict[str, Any], baseline_ref: Optional[Dict[str, Any]]
) -> Dict[str, Optional[float]]:
    if baseline_ref is None:
        return {"win_rate_vs_baseline": None, "lose_rate_vs_baseline": None, "tie_rate_vs_baseline": None}
    ours = {name: score for name, score in zip(
        payload.get("instances", []), payload.get("aug_score", [])
    )}
    base = {name: score for name, score in zip(
        baseline_ref.get("instances", []), baseline_ref.get("aug_score", [])
    )}
    common = [n for n in ours if n in base]
    if not common:
        return {"win_rate_vs_baseline": None, "lose_rate_vs_baseline": None, "tie_rate_vs_baseline": None}
    wins = sum(1 for n in common if ours[n] < base[n] - 1e-9)
    losses = sum(1 for n in common if ours[n] > base[n] + 1e-9)
    ties = len(common) - wins - losses
    total = float(len(common))
    return {
        "win_rate_vs_baseline": wins / total,
        "lose_rate_vs_baseline": losses / total,
        "tie_rate_vs_baseline": ties / total,
    }


def _runtime_ms_from_per_instance(payload: Dict[str, Any]) -> Optional[float]:
    rows: List[Dict[str, Any]] = payload.get("per_instance", []) or []
    if not rows:
        return None
    total = 0.0
    has_any = False
    for r in rows:
        if "baseline_ms" in r:
            total += float(r.get("baseline_ms", 0.0))
            has_any = True
        ri = r.get("rerank_info") or {}
        if "rerank_ms" in ri:
            total += float(ri.get("rerank_ms", 0.0))
            has_any = True
        for sec_key in ("no_aug_pool", "aug_pool"):
            sec = r.get(sec_key) or {}
            two_opt = sec.get("two_opt") or []
            if isinstance(two_opt, list):
                for t in two_opt:
                    if isinstance(t, dict) and "time_ms" in t:
                        total += float(t["time_ms"])
                        has_any = True
    return total if has_any else None


def write_summary_row(
    summary_dir: str,
    run_name: str,
    payload: Dict[str, Any],
    config: Dict[str, Any],
    baseline_ref_json: Optional[str] = None,
) -> str:
    os.makedirs(summary_dir, exist_ok=True)
    baseline_ref: Optional[Dict[str, Any]] = None
    if baseline_ref_json is not None and os.path.exists(baseline_ref_json):
        with open(baseline_ref_json, "r", encoding="utf-8") as f:
            baseline_ref = json.load(f)

    winrates = _compute_winrate(payload, baseline_ref)
    runtime_ms = _runtime_ms_from_per_instance(payload)

    row = {
        "run_name": run_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "avg_aug_gap": payload.get("avg_aug_gap"),
        "avg_no_aug_gap": payload.get("avg_no_aug_gap"),
        "solved": payload.get("solved_instance_num"),
        "total": payload.get("total_instance_num"),
        "aug_factor": payload.get("aug_factor"),
        "rerank_enabled": _safe_get(config, "rerank_enabled", False),
        "two_opt_enabled": _safe_get(config, "two_opt_enabled", False),
        "distance_bias_enabled": _safe_get(config, "distance_bias_enabled", False),
        "knn_bias_enabled": _safe_get(config, "knn_bias_enabled", False),
        "leader_reward_enabled": _safe_get(config, "leader_reward_enabled", False),
        "total_runtime_ms": runtime_ms,
        **winrates,
        "checkpoint_path": payload.get("checkpoint_path"),
        "data_path": payload.get("data_path"),
    }

    csv_path = os.path.join(summary_dir, "summary.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in CSV_COLUMNS})

    json_path = os.path.join(summary_dir, "summary.json")
    all_rows: List[Dict[str, Any]] = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                all_rows = json.load(f)
            if not isinstance(all_rows, list):
                all_rows = []
        except Exception:
            all_rows = []
    # Attach full config + per-instance details in JSON form (CSV stays flat).
    full_row = dict(row)
    full_row["config"] = {k: v for k, v in config.items() if _is_jsonable(v)}
    all_rows.append(full_row)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    per_inst_path = os.path.join(summary_dir, f"per_instance_{run_name}.json")
    with open(per_inst_path, "w", encoding="utf-8") as f:
        json.dump(payload.get("per_instance", []), f, ensure_ascii=False, indent=2, default=_default_json)

    return csv_path


def _is_jsonable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _default_json(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)
