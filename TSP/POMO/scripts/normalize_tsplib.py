#!/usr/bin/env python3
"""Normalize TSPLIB-style instances to the unit square.

Mirrors the per-instance, aspect-ratio-preserving min-max scaling used by
``TSPTester_LIB._normalize_to_unit_square``:

    xy_min = min over nodes
    xy_max = max over nodes
    ratio  = max(x_range, y_range)        # uniform scaling -> keeps shape
    coords = (coords - xy_min) / ratio    # in [0, 1]

For each input ``.tsp`` file the script can produce three artefacts (all on
by default; opt out per-output via flags):

  1. A normalized ``.tsp`` file in ``--out_dir`` (TSPLIB-formatted, parseable
     by ``tsplib_utils.TSPLIBReader``).
  2. A per-file ``<name>.meta.json`` sidecar with ``xy_min`` / ``ratio`` /
     ``scale`` so the transformation can be inverted exactly.
  3. A combined ``all.npz`` archive containing every instance's coords and
     metadata, suitable for feeding straight into a torch/numpy training
     pipeline.

Usage:
    cd TSP/POMO
    python scripts/normalize_tsplib.py                    # ../data/val -> ../data/val_normalized
    python scripts/normalize_tsplib.py --in_file ../data/val/kroB150.tsp --out_dir /tmp/out
    python scripts/normalize_tsplib.py --no_npz --no_meta # only emit .tsp files
    python scripts/normalize_tsplib.py --scale 1000.0     # rescale to [0, 1000]

Importable API:
    from normalize_tsplib import normalize_xy
    norm_xy, meta = normalize_xy(coords)  # coords: (N, 2) float ndarray
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POMO_DIR = os.path.dirname(SCRIPT_DIR)
TSP_ROOT = os.path.dirname(POMO_DIR)
if POMO_DIR not in sys.path:
    sys.path.insert(0, POMO_DIR)

from tsplib_utils import TSPLIBReader  # noqa: E402

DEFAULT_IN_DIR = os.path.join(TSP_ROOT, "data", "val")
DEFAULT_OUT_DIR = os.path.join(TSP_ROOT, "data", "val_normalized")


def normalize_xy(coords: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, Dict[str, object]]:
    """Per-instance uniform min-max scaling, mirroring the tester.

    Args:
        coords: (N, 2) float array.
        scale: target side length. Default 1.0 -> output in [0, 1].

    Returns:
        (normalized_coords, meta) where meta contains the inversion params.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"expected (N, 2) array, got shape {coords.shape}")

    xy_min = coords.min(axis=0)
    xy_max = coords.max(axis=0)
    span = xy_max - xy_min
    ratio = float(span.max())
    if ratio == 0.0:
        ratio = 1.0

    normalized = (coords - xy_min) / ratio * scale

    meta: Dict[str, object] = {
        "xy_min": [float(xy_min[0]), float(xy_min[1])],
        "xy_max": [float(xy_max[0]), float(xy_max[1])],
        "ratio": ratio,
        "scale": float(scale),
        "n": int(coords.shape[0]),
    }
    return normalized.astype(np.float64), meta


def _format_coord(value: float) -> str:
    return f"{value:.10g}"


def write_normalized_tsp(
    out_path: str,
    name: str,
    dimension: int,
    edge_weight_type: str,
    locs_norm: np.ndarray,
    meta: Dict[str, object],
    source_file: str,
) -> None:
    """Emit a TSPLIB-formatted file with normalized coordinates."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    xy_min = meta["xy_min"]
    ratio = meta["ratio"]
    scale = meta["scale"]
    lines: List[str] = [
        f"NAME: {name}",
        "TYPE: TSP",
        f"COMMENT: Normalized from {os.path.basename(source_file)} via normalize_tsplib.py",
        f"COMMENT: scale={scale}  ratio={ratio:.10g}  xy_min=[{xy_min[0]:.10g}, {xy_min[1]:.10g}]",
        f"DIMENSION: {dimension}",
        f"EDGE_WEIGHT_TYPE: {edge_weight_type}",
        "NODE_COORD_SECTION",
    ]
    for idx in range(dimension):
        x = _format_coord(float(locs_norm[idx, 0]))
        y = _format_coord(float(locs_norm[idx, 1]))
        lines.append(f"{idx + 1} {x} {y}")
    lines.append("EOF")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_meta_sidecar(out_path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def process_file(
    in_path: str,
    out_dir: str,
    scale: float,
    write_tsp: bool,
    write_meta: bool,
) -> Optional[Dict[str, object]]:
    """Returns the per-instance record (or None if skipped)."""
    name, dimension, locs, edge_weight_type = TSPLIBReader(in_path)
    if name is None:
        return None

    coords = np.asarray(locs, dtype=np.float64)
    norm_coords, meta = normalize_xy(coords, scale=scale)

    record: Dict[str, object] = {
        "name": name,
        "source_file": os.path.abspath(in_path),
        "dimension": int(dimension),
        "edge_weight_type": edge_weight_type,
        **meta,
    }

    if write_tsp:
        out_tsp = os.path.join(out_dir, f"{name}.tsp")
        write_normalized_tsp(
            out_tsp, name, dimension, edge_weight_type,
            norm_coords, meta, in_path,
        )
        record["normalized_tsp"] = os.path.abspath(out_tsp)

    if write_meta:
        out_json = os.path.join(out_dir, f"{name}.meta.json")
        write_meta_sidecar(out_json, record)
        record["meta_json"] = os.path.abspath(out_json)

    record["coords_normalized"] = norm_coords
    record["coords_original"] = coords
    return record


def gather_inputs(in_dir: Optional[str], in_file: Optional[str]) -> List[str]:
    if in_file is not None:
        if not os.path.isfile(in_file):
            raise FileNotFoundError(f"--in_file not found: {in_file}")
        return [os.path.abspath(in_file)]
    assert in_dir is not None
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"--in_dir not found: {in_dir}")
    paths = []
    for root, _, files in os.walk(in_dir):
        for fname in files:
            if fname.lower().endswith(".tsp"):
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def write_combined_npz(
    out_path: str,
    records: List[Dict[str, object]],
) -> None:
    """Bundle all normalized coords + metadata into a single .npz archive.

    Layout (loaded via np.load(..., allow_pickle=True)):
        names         : (M,) object array of instance names
        dimensions    : (M,) int64 array
        edge_weight_types : (M,) object array
        ratios        : (M,) float64 array
        xy_mins       : (M, 2) float64 array
        scale         : scalar float64
        coords        : (M,) object array, each entry is (Ni, 2) float64
                        (kept as object array because Ni varies per instance)
    """
    if not records:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = np.array([r["name"] for r in records], dtype=object)
    dimensions = np.array([r["dimension"] for r in records], dtype=np.int64)
    ewt = np.array([r["edge_weight_type"] for r in records], dtype=object)
    ratios = np.array([r["ratio"] for r in records], dtype=np.float64)
    xy_mins = np.array([r["xy_min"] for r in records], dtype=np.float64)
    coords_obj = np.empty(len(records), dtype=object)
    for i, r in enumerate(records):
        coords_obj[i] = np.asarray(r["coords_normalized"], dtype=np.float64)
    scales = np.array([r["scale"] for r in records], dtype=np.float64)
    scale_uniform = scales[0] if np.allclose(scales, scales[0]) else scales

    np.savez(
        out_path,
        names=names,
        dimensions=dimensions,
        edge_weight_types=ewt,
        ratios=ratios,
        xy_mins=xy_mins,
        scale=scale_uniform,
        coords=coords_obj,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize TSPLIB .tsp files to the unit square (uniform "
                    "min-max, preserves aspect ratio). Mirrors the tester's "
                    "_normalize_to_unit_square exactly.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--in_dir", default=DEFAULT_IN_DIR,
                     help=f"Directory of .tsp files (default: {DEFAULT_IN_DIR})")
    src.add_argument("--in_file", default=None,
                     help="Single .tsp file (overrides --in_dir).")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Target side length (default 1.0 -> [0, 1]).")
    parser.add_argument("--no_tsp", action="store_true",
                        help="Skip writing normalized .tsp files.")
    parser.add_argument("--no_meta", action="store_true",
                        help="Skip per-file .meta.json sidecars.")
    parser.add_argument("--no_npz", action="store_true",
                        help="Skip the combined all.npz archive.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-file progress prints.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.scale <= 0.0:
        raise SystemExit(f"--scale must be > 0, got {args.scale}")

    inputs = gather_inputs(args.in_dir, args.in_file)
    if not inputs:
        print(f"[normalize] no .tsp files found under: "
              f"{args.in_file or args.in_dir}", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out_dir)
    write_tsp = not args.no_tsp
    write_meta = not args.no_meta
    write_npz = not args.no_npz

    records: List[Dict[str, object]] = []
    skipped: List[str] = []
    for in_path in inputs:
        record = process_file(
            in_path=in_path,
            out_dir=out_dir,
            scale=args.scale,
            write_tsp=write_tsp,
            write_meta=write_meta,
        )
        if record is None:
            skipped.append(in_path)
            if not args.quiet:
                print(f"[normalize] SKIP (unsupported TSPLIB): {in_path}")
            continue
        records.append(record)
        if not args.quiet:
            xmin, ymin = record["xy_min"]
            print(f"[normalize] {record['name']:<12} "
                  f"N={record['dimension']:<5} "
                  f"ewt={record['edge_weight_type']:<7} "
                  f"ratio={record['ratio']:.4g}  "
                  f"xy_min=({xmin:.4g}, {ymin:.4g})")

    if not records:
        print("[normalize] nothing produced.", file=sys.stderr)
        return 1

    if write_npz:
        npz_path = os.path.join(out_dir, "all.npz")
        write_combined_npz(npz_path, records)
        if not args.quiet:
            print(f"[normalize] wrote {npz_path} ({len(records)} instances)")

    print(f"[normalize] done: {len(records)} normalized, "
          f"{len(skipped)} skipped, scale={args.scale}, out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
