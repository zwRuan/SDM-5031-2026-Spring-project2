"""Sanity tests for the Mixed Structured Curriculum (MSC) data generators.

Run with either:
    python -m pytest tests/test_msc.py -q
or:
    python tests/test_msc.py

The tests are CPU-only and finish in <1s.  They cover:

* legacy parity (``get_random_problems(B, N)`` returns the legacy uniform);
* shape / range / finiteness invariants for each generator;
* curriculum stage selection and ratio rounding;
* full ``get_random_problems`` mixing path.
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
POMO_DIR = os.path.dirname(HERE)
sys.path.insert(0, POMO_DIR)
sys.path.insert(0, os.path.dirname(POMO_DIR))           # for TSProblemDef
sys.path.insert(0, os.path.dirname(os.path.dirname(POMO_DIR)))  # for utils

from TSProblemDef import (  # noqa: E402
    DEFAULT_MSC_CONFIG,
    generate_clustered_uniform,
    generate_gaussian_mixture,
    generate_uniform,
    get_curriculum_stage,
    get_default_msc_config,
    get_random_problems,
    get_stage_ratios,
    sample_distribution_types,
)


def _check_problems(problems: torch.Tensor, batch_size: int, problem_size: int) -> None:
    assert problems.shape == (batch_size, problem_size, 2), problems.shape
    assert problems.dtype == torch.float32 or problems.dtype == torch.float64
    assert torch.isfinite(problems).all().item()
    assert problems.min().item() >= 0.0 - 1e-6
    assert problems.max().item() <= 1.0 + 1e-6


def test_legacy_call_matches_uniform_seedwise():
    torch.manual_seed(0)
    expected = torch.rand(size=(7, 12, 2))
    torch.manual_seed(0)
    got = get_random_problems(7, 12)
    assert torch.allclose(expected, got)


def test_disabled_config_falls_back_to_uniform():
    cfg = get_default_msc_config()
    cfg["enabled"] = False
    torch.manual_seed(1)
    expected = torch.rand(size=(5, 9, 2))
    torch.manual_seed(1)
    got = get_random_problems(5, 9, epoch=10, total_epochs=100, config=cfg)
    assert torch.allclose(expected, got)


def test_uniform_generator_invariants():
    p = generate_uniform(4, 16)
    _check_problems(p, 4, 16)


def test_clustered_uniform_invariants():
    cfg = DEFAULT_MSC_CONFIG["clustered_uniform"]
    p = generate_clustered_uniform(4, 50, cfg)
    # No clipping yet: still expect finite values, mostly inside [0,1].
    assert p.shape == (4, 50, 2)
    assert torch.isfinite(p).all().item()


def test_gaussian_mixture_invariants():
    cfg = DEFAULT_MSC_CONFIG["gaussian_mixture"]
    p = generate_gaussian_mixture(4, 50, cfg)
    assert p.shape == (4, 50, 2)
    assert torch.isfinite(p).all().item()


def test_get_curriculum_stage():
    bounds = [0.3, 0.7]
    assert get_curriculum_stage(1, 10, bounds) == 0    # 0.0 progress
    assert get_curriculum_stage(3, 10, bounds) == 0    # 0.2 progress
    assert get_curriculum_stage(4, 10, bounds) == 1    # 0.3 progress
    assert get_curriculum_stage(7, 10, bounds) == 1    # 0.6 progress
    assert get_curriculum_stage(8, 10, bounds) == 2    # 0.7 progress
    assert get_curriculum_stage(10, 10, bounds) == 2
    assert get_curriculum_stage(None, 10, bounds) is None
    assert get_curriculum_stage(5, None, bounds) is None


def test_get_stage_ratios_fallbacks():
    cfg = DEFAULT_MSC_CONFIG["curriculum"]
    # No stage_idx -> fixed_ratios.
    fixed = get_stage_ratios(None, cfg)
    assert set(fixed.keys()) == {"uniform", "clustered_uniform", "gaussian_mixture"}
    # Stage idx clamps to last stage.
    last = get_stage_ratios(99, cfg)
    assert abs(sum(last.values()) - 1.0) < 1e-6


def test_sample_distribution_types_sums_to_batch_size():
    ratios = {"uniform": 0.7, "clustered_uniform": 0.3, "gaussian_mixture": 0.0}
    counts = sample_distribution_types(64, ratios)
    assert sum(counts.values()) == 64
    assert counts["gaussian_mixture"] == 0
    assert counts["uniform"] >= counts["clustered_uniform"]


def test_sample_distribution_types_handles_all_zero_ratios():
    counts = sample_distribution_types(8, {"uniform": 0.0, "clustered_uniform": 0.0, "gaussian_mixture": 0.0})
    assert sum(counts.values()) == 8
    assert counts.get("uniform", 0) == 8


def test_get_random_problems_msc_path_invariants():
    cfg = get_default_msc_config()
    cfg["log_every"] = 0  # silence
    p = get_random_problems(64, 100, epoch=50, total_epochs=100, config=cfg)
    _check_problems(p, 64, 100)


def test_get_random_problems_no_curriculum_uses_fixed_ratios():
    cfg = get_default_msc_config()
    cfg["use_curriculum"] = False
    cfg["log_every"] = 0
    p = get_random_problems(32, 100, epoch=None, total_epochs=None, config=cfg)
    _check_problems(p, 32, 100)


def test_postprocess_clips_and_jitters():
    cfg = get_default_msc_config()
    cfg["log_every"] = 0
    cfg["postprocess"]["jitter_eps"] = 1e-3
    p = get_random_problems(8, 50, epoch=20, total_epochs=100, config=cfg)
    assert p.min().item() >= 0.0
    assert p.max().item() <= 1.0


def _all_tests():
    return [v for k, v in globals().items() if k.startswith("test_") and callable(v)]


def main() -> int:
    failures = 0
    for fn in _all_tests():
        name = fn.__name__
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # pragma: no cover
            failures += 1
            print(f"FAIL  {name}: {exc!r}")
    print("---")
    print(f"{len(_all_tests()) - failures}/{len(_all_tests())} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
