"""M4: 2-opt post-processing for TSP tours.

This module implements a vectorized 2-opt local search that operates directly
on the TSPLIB-style integer distance metric so that improvements are measured
against the official evaluation metric.

Public API:
    two_opt_refine(tour, coords, ew_type, cfg) -> (new_tour, new_length, info)
    tour_length_lib(tour, coords, ew_type) -> float

The ``cfg`` is a dictionary with keys:
    two_opt_max_iters:        int (default 50)
    two_opt_first_improvement: bool (default True)
    two_opt_time_budget_ms:   Optional[float] (default None)
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


DEFAULT_CFG: Dict[str, Any] = {
    "two_opt_enabled": False,
    "two_opt_target": "final_best",  # or "topk_candidates"
    "two_opt_topk": 3,
    "two_opt_max_iters": 50,
    "two_opt_first_improvement": True,
    "two_opt_time_budget_ms": None,
}


def _tsplib_round(x: np.ndarray, ew_type: str) -> np.ndarray:
    if ew_type == "CEIL_2D":
        return np.ceil(x)
    if ew_type == "EUC_2D":
        # TSPLIB NINT = floor(x + 0.5); matches TSPEnv._get_travel_distance.
        return np.floor(x + 0.5)
    return x


def _pairwise_distance(coords: np.ndarray, ew_type: str) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff * diff).sum(-1))
    return _tsplib_round(d, ew_type)


def tour_length_lib(tour: np.ndarray, coords: np.ndarray, ew_type: str) -> float:
    """Closed-cycle TSPLIB length for a single tour."""
    if isinstance(tour, torch.Tensor):
        tour = tour.detach().cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    tour = np.asarray(tour, dtype=np.int64).reshape(-1)
    pts = coords[tour]
    rolled = np.roll(pts, -1, axis=0)
    seg = np.sqrt(((pts - rolled) ** 2).sum(-1))
    seg = _tsplib_round(seg, ew_type)
    return float(seg.sum())


def _validate_tour(tour: np.ndarray, n: int) -> bool:
    if tour.shape[0] != n:
        return False
    return np.array_equal(np.sort(tour), np.arange(n))


def two_opt_refine(
    tour,
    coords,
    ew_type: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Apply 2-opt until no improvement / budget exhausted.

    Args:
        tour: (N,) integer permutation.
        coords: (N, 2) float coordinates (ORIGINAL TSPLIB coordinates).
        ew_type: "EUC_2D" | "CEIL_2D" | other (continuous fallback).
        cfg: see DEFAULT_CFG.

    Returns:
        new_tour: (N,) int numpy array. Guaranteed permutation.
        new_length: float tour length (same rounding as TSPEnv lib_mode).
        info: dict with keys
              {iters, improvements, delta_total, time_ms, improved}.
    """
    config = dict(DEFAULT_CFG)
    if cfg:
        config.update(cfg)

    if isinstance(tour, torch.Tensor):
        tour = tour.detach().cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()

    tour = np.asarray(tour, dtype=np.int64).reshape(-1).copy()
    coords = np.asarray(coords, dtype=np.float64)
    n = tour.shape[0]
    assert _validate_tour(tour, n), "Input tour is not a valid permutation."

    dmat = _pairwise_distance(coords, ew_type)
    original_len = float(_segment_length_from_dmat(tour, dmat))

    max_iters = int(config["two_opt_max_iters"])
    first_improvement = bool(config["two_opt_first_improvement"])
    time_budget_ms = config["two_opt_time_budget_ms"]
    t0 = time.time()

    improvements = 0
    iters = 0
    current = tour.copy()
    current_len = original_len

    while iters < max_iters:
        iters += 1
        if time_budget_ms is not None:
            elapsed_ms = (time.time() - t0) * 1000.0
            if elapsed_ms > time_budget_ms:
                break

        if first_improvement:
            improved, current, current_len = _first_improvement_sweep(
                current, current_len, dmat
            )
        else:
            improved, current, current_len = _best_improvement_sweep(
                current, current_len, dmat
            )

        if not improved:
            break
        improvements += 1

    if current_len > original_len:
        # Never return a worse tour.
        current = tour.copy()
        current_len = original_len

    assert _validate_tour(current, n), "2-opt produced invalid tour."

    info = {
        "iters": iters,
        "improvements": improvements,
        "delta_total": float(original_len - current_len),
        "time_ms": float((time.time() - t0) * 1000.0),
        "improved": bool(current_len < original_len),
        "original_length": float(original_len),
        "final_length": float(current_len),
    }
    return current, float(current_len), info


def _segment_length_from_dmat(tour: np.ndarray, dmat: np.ndarray) -> float:
    rolled = np.roll(tour, -1)
    return float(dmat[tour, rolled].sum())


def _compute_delta_matrix(tour: np.ndarray, dmat: np.ndarray) -> np.ndarray:
    """Return delta[i, j] = length(new) - length(old) for reversing tour[i+1 .. j].

    Only i < j are considered meaningful; we mask the rest to +inf.
    Edges involved:
        old: tour[i] -> tour[i+1]    and    tour[j] -> tour[j+1 mod n]
        new: tour[i] -> tour[j]      and    tour[i+1] -> tour[j+1 mod n]
    """
    n = tour.shape[0]
    t = tour
    t_next = np.roll(t, -1)
    # d_edge[k] = dmat[tour[k], tour[k+1]]
    d_edge = dmat[t, t_next]

    # old_ij[i, j] = d_edge[i] + d_edge[j]
    old_sum = d_edge[:, None] + d_edge[None, :]

    # new_ij[i, j] = dmat[tour[i], tour[j]] + dmat[tour[i+1], tour[j+1]]
    new_first = dmat[t[:, None], t[None, :]]
    new_second = dmat[t_next[:, None], t_next[None, :]]
    new_sum = new_first + new_second

    delta = new_sum - old_sum
    # Only consider j >= i+2 and (i,j) != (0, n-1) -- the latter is a no-op swap.
    mask = np.ones((n, n), dtype=bool)
    idx = np.arange(n)
    # require j >= i + 2
    mask[idx[:, None] + 1 >= idx[None, :]] = False
    # forbid the wrap-around swap (i=0, j=n-1) because edges are identical
    mask[0, n - 1] = False
    delta = np.where(mask, delta, np.inf)
    return delta


def _best_improvement_sweep(
    tour: np.ndarray, cur_len: float, dmat: np.ndarray
) -> Tuple[bool, np.ndarray, float]:
    delta = _compute_delta_matrix(tour, dmat)
    flat_idx = int(np.argmin(delta))
    best_delta = float(delta.flat[flat_idx])
    if best_delta >= -1e-12:
        return False, tour, cur_len
    n = tour.shape[0]
    i, j = divmod(flat_idx, n)
    new_tour = tour.copy()
    new_tour[i + 1 : j + 1] = new_tour[i + 1 : j + 1][::-1]
    new_len = cur_len + best_delta
    return True, new_tour, new_len


def _first_improvement_sweep(
    tour: np.ndarray, cur_len: float, dmat: np.ndarray
) -> Tuple[bool, np.ndarray, float]:
    n = tour.shape[0]
    t = tour
    t_next = np.roll(t, -1)
    d_edge = dmat[t, t_next]
    for i in range(n - 2):
        d_i = d_edge[i]
        a = t[i]
        b = t[i + 1]
        # j = n-1 combined with i=0 gives a trivial swap; skip it.
        j_max = n - 1 if i > 0 else n - 2
        for j in range(i + 2, j_max + 1):
            c = t[j]
            d = t_next[j]
            delta = dmat[a, c] + dmat[b, d] - d_i - d_edge[j]
            if delta < -1e-12:
                new_tour = tour.copy()
                new_tour[i + 1 : j + 1] = new_tour[i + 1 : j + 1][::-1]
                return True, new_tour, cur_len + float(delta)
    return False, tour, cur_len
