"""Lightweight sanity / regression tests.

Run with either:
    python -m pytest tests/test_sanity.py -q
or:
    python tests/test_sanity.py        (uses plain assert / runs all checks)

Tests are intentionally CPU-only and use a small synthetic problem so they
finish in seconds without any external data.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Make TSP/POMO modules importable regardless of where pytest runs from.
HERE = os.path.dirname(os.path.abspath(__file__))
POMO_DIR = os.path.dirname(HERE)
sys.path.insert(0, POMO_DIR)
sys.path.insert(0, os.path.dirname(POMO_DIR))           # for TSProblemDef
sys.path.insert(0, os.path.dirname(os.path.dirname(POMO_DIR)))  # for utils.utils

from search.two_opt import two_opt_refine, tour_length_lib  # noqa: E402
from search.sgbs_lite import _canonicalize_tour, pool_and_select_best  # noqa: E402
from model_ext.distance_bias import DistanceBiasModule  # noqa: E402
from train_ext.leader_reward import compute_leader_loss  # noqa: E402


def _rand_coords(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n, 2)).astype(np.float64)


def test_two_opt_returns_valid_permutation_and_not_worse():
    coords = _rand_coords(20, seed=1)
    n = coords.shape[0]
    tour = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(123)
    rng.shuffle(tour)
    orig_len = tour_length_lib(tour, coords, "EUC_2D")

    new_tour, new_len, info = two_opt_refine(
        tour, coords, "EUC_2D",
        cfg={"two_opt_max_iters": 50, "two_opt_first_improvement": True},
    )

    assert new_tour.shape == (n,), "tour shape must be (N,)"
    assert sorted(new_tour.tolist()) == list(range(n)), "tour must be a permutation"
    assert new_len <= orig_len + 1e-9, "2-opt must not worsen the tour"
    assert info["improved"] == (new_len < orig_len - 1e-9)


def test_two_opt_best_vs_first_improvement_consistent():
    coords = _rand_coords(15, seed=42)
    n = coords.shape[0]
    tour = np.arange(n, dtype=np.int64)
    np.random.default_rng(7).shuffle(tour)

    _, len_first, _ = two_opt_refine(
        tour, coords, "EUC_2D",
        cfg={"two_opt_max_iters": 100, "two_opt_first_improvement": True},
    )
    _, len_best, _ = two_opt_refine(
        tour, coords, "EUC_2D",
        cfg={"two_opt_max_iters": 100, "two_opt_first_improvement": False},
    )
    orig = tour_length_lib(tour, coords, "EUC_2D")
    assert len_first <= orig + 1e-9
    assert len_best <= orig + 1e-9


def test_canonicalize_tour_invariants():
    a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)  # not a TSP tour, just sequence
    # Take a real permutation:
    a = np.array([2, 3, 0, 4, 1], dtype=np.int64)
    rotated = np.array([4, 1, 2, 3, 0], dtype=np.int64)
    reversed_ = a[::-1].copy()
    assert _canonicalize_tour(a) == _canonicalize_tour(rotated)
    assert _canonicalize_tour(a) == _canonicalize_tour(reversed_)


def test_pool_and_select_best_baseline_only():
    aug, pomo, n = 2, 3, 5
    rng = np.random.default_rng(0)
    tours = np.stack(
        [np.stack([rng.permutation(n) for _ in range(pomo)]) for _ in range(aug)]
    ).astype(np.int64)
    lens = rng.uniform(1, 10, size=(aug, pomo)).astype(np.float64)
    src, tour, length, info = pool_and_select_best(
        tours, lens, None, None, pool_across_augs=True, deduplicate=True,
    )
    assert src == 0
    assert length == float(lens.min())
    assert tour.shape == (n,)


def test_distance_bias_module_no_nan_and_logit_only():
    cfg = {
        "distance_bias_enabled": True,
        "distance_bias_scale": 1.0,
        "distance_bias_mode": "logit",
        "distance_norm_mode": "mean",
        "knn_bias_enabled": True,
        "knn_k": 3,
        "knn_bias_value": 0.5,
    }
    mod = DistanceBiasModule(cfg)
    coords = torch.rand(2, 8, 2)  # batch=2, N=8
    mod.prepare(coords)
    current = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)  # (batch=2, pomo=3)
    bias = mod(current)
    assert bias.shape == (2, 3, 8)
    assert torch.isfinite(bias).all().item()


def test_distance_bias_attn_mode_raises():
    try:
        DistanceBiasModule({"distance_bias_enabled": True, "distance_bias_mode": "attn"})
    except NotImplementedError:
        return
    raise AssertionError("attn mode should raise NotImplementedError")


def test_leader_loss_bonus_adv_gradient_flow():
    torch.manual_seed(0)
    reward = torch.tensor([[-5.0, -7.0, -3.0, -8.0]])  # leader = idx 2 (least negative)
    log_prob = torch.randn(1, 4, requires_grad=True)
    loss, stats = compute_leader_loss(
        reward, log_prob,
        cfg={"leader_reward_enabled": True, "leader_mode": "bonus_adv", "leader_gamma": 0.5},
    )
    loss.backward()
    assert log_prob.grad is not None
    assert torch.isfinite(loss).item()
    assert not stats["nan_detected"]
    assert stats["leader_mode"] == "bonus_adv"


def test_leader_loss_aux_imitation_matches_argmax():
    reward = torch.tensor([[-5.0, -1.0, -3.0]])  # leader at idx 1
    log_prob = torch.zeros(1, 3, requires_grad=True)
    loss, stats = compute_leader_loss(
        reward, log_prob,
        cfg={"leader_reward_enabled": True, "leader_mode": "aux_imitation", "leader_aux_weight": 1.0},
    )
    loss.backward()
    # grad on leader index should be the most negative (push log_prob up)
    grad = log_prob.grad.detach().squeeze().numpy()
    assert int(np.argmin(grad)) == 1
    assert not stats["nan_detected"]


def test_two_opt_handles_already_optimal_tour():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    tour = np.array([0, 1, 2, 3], dtype=np.int64)
    new_tour, new_len, info = two_opt_refine(
        tour, coords, "EUC_2D",
        cfg={"two_opt_max_iters": 50, "two_opt_first_improvement": True},
    )
    assert sorted(new_tour.tolist()) == [0, 1, 2, 3]
    # Square tour length under EUC_2D rounding = 4.
    assert new_len == 4.0
    assert not info["improved"]


def _all_tests():
    return [v for k, v in globals().items() if k.startswith("test_") and callable(v)]


def main() -> int:
    failures = 0
    for fn in _all_tests():
        name = fn.__name__
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}: {exc!r}")
    print("---")
    print(f"{len(_all_tests()) - failures}/{len(_all_tests())} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
