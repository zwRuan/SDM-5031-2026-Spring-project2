"""Fast POMO evaluator for projection function search.

Loads the POMO model once, then evaluates candidate projection functions on a
subset of validation instances without augmentation (for speed during evolution).

Usage:
    evaluator = POMOFastEvaluator(checkpoint_path, device='cuda:0')
    gap = evaluator.evaluate(projection_fn, num_instances=32)
    # returns avg_gap (lower is better) on original coordinates
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Bootstrap paths to match the POMO test environment.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POMO_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _POMO_DIR)
sys.path.insert(0, os.path.join(_POMO_DIR, ".."))

from TSPEnv import TSPEnv
from TSPModel import TSPModel
from TSProblemDef import augment_xy_data_by_8_fold
from tsplib_utils import TSPLIBReader, tsplib_cost

try:
    from model_ext.distance_bias import DistanceBiasModule
except Exception:
    DistanceBiasModule = None


_MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}


def _normalize_to_unit_square(node_xy: torch.Tensor) -> torch.Tensor:
    xy_max = torch.max(node_xy, dim=1, keepdim=True).values
    xy_min = torch.min(node_xy, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1
    return (node_xy - xy_min) / ratio.expand(-1, 1, 2)


def _load_val_instances(data_dir: str) -> List[Tuple[torch.Tensor, Optional[float], str, str]]:
    """Load TSPLIB validation instances.

    Returns list of (coords_tensor, optimal_tour_length, name, ew_type).
    """
    instances = []
    fnames = sorted(f for f in os.listdir(data_dir) if f.endswith(".tsp"))
    for fname in fnames:
        fpath = os.path.join(data_dir, fname)
        try:
            name, dimension, locs, ew_type = TSPLIBReader(fpath)
            if name is None or dimension is None:
                continue
            optimal = tsplib_cost.get(name, None)
            coords = torch.from_numpy(np.array(locs, dtype=np.float32))
            instances.append((coords, optimal, name, ew_type))
        except Exception:
            continue
    return instances


class POMOFastEvaluator:
    """Fast POMO evaluator for projection function scoring during LLM search."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        val_data_dir: str | None = None,
        num_fast_eval: int = 32,
        use_aug: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_fast_eval = num_fast_eval
        self.use_aug = use_aug

        # Set default device/dtype so POMO internals (torch.arange etc.) land on GPU.
        if self.device.type == "cuda":
            torch.set_default_device(self.device)
            torch.set_default_dtype(torch.float32)
            torch.cuda.set_device(self.device)

        # --- Load model ---
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = TSPModel(**_MODEL_PARAMS)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # --- Attach bias module ---
        bias_cfg = checkpoint.get("bias_cfg") or {}
        if bias_cfg.get("distance_bias_enabled") or bias_cfg.get("knn_bias_enabled"):
            if DistanceBiasModule is not None:
                bias_module = DistanceBiasModule(bias_cfg)
                self.model.attach_distance_bias(bias_module)
                print(f"[POMOFastEval] Attached bias module: {bias_cfg}")

        # --- Load validation instances ---
        if val_data_dir is None:
            val_data_dir = os.path.join(_POMO_DIR, "..", "data", "val")
        self.val_instances = _load_val_instances(val_data_dir)
        print(f"[POMOFastEval] Loaded {len(self.val_instances)} validation instances")

        self._eval_count = 0

    def evaluate(self, projection_fn: Callable, num_instances: int | None = None) -> float:
        """Evaluate a projection function.

        Args:
            projection_fn: Callable that takes (batch, N, 2) tensor and returns
                           projected (batch, N, 2) tensor.
            num_instances: Number of instances to evaluate on (default: self.num_fast_eval).

        Returns:
            avg_gap (float): Average gap in percent. Lower is better.
        """
        if num_instances is None:
            num_instances = self.num_fast_eval

        instances = self.val_instances[: min(num_instances, len(self.val_instances))]
        gaps = []
        failures = 0

        for coords, optimal, name, ew_type in instances:
            try:
                gap = self._eval_one(projection_fn, coords, optimal, ew_type)
                gaps.append(gap)
            except Exception:
                failures += 1
                continue

        self._eval_count += 1

        if not gaps:
            return 100.0  # worst possible

        avg_gap = float(np.mean(gaps))
        return avg_gap

    def _eval_one(
        self,
        projection_fn: Callable,
        coords_orig: torch.Tensor,
        optimal: Optional[float],
        ew_type: str,
    ) -> float:
        """Evaluate projection on a single instance."""
        N = coords_orig.size(0)
        coords = coords_orig.unsqueeze(0).to(self.device)  # (1, N, 2)

        # Apply projection function
        with torch.no_grad():
            projected = projection_fn(coords)
            if projected.shape != coords.shape:
                projected = coords  # fallback on shape mismatch

        # Compute model score
        if self.use_aug:
            score = self._pomo_infer_aug(projected, coords, ew_type)
        else:
            score = self._pomo_infer_single(projected, coords, ew_type)

        if optimal is not None and optimal > 0:
            gap = (score - optimal) / optimal * 100.0
        else:
            gap = 0.0
        return gap

    def _pomo_infer_single(
        self, node_xy: torch.Tensor, coords_orig: torch.Tensor, ew_type: str
    ) -> float:
        """Single-pass POMO inference (no augmentation)."""
        N = node_xy.size(1)
        env = TSPEnv(problem_size=N, pomo_size=N)

        env.batch_size = 1
        env.problems = node_xy.to(self.device)
        env.BATCH_IDX = torch.arange(1, device=self.device)[:, None].expand(1, N)
        env.POMO_IDX = torch.arange(N, device=self.device)[None, :].expand(1, N)

        # For cost computation on original coordinates
        env.original_node_xy_lib = coords_orig.to(self.device)
        env.edge_weight_type = ew_type

        with torch.no_grad():
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected, lib_mode=True)

            tour_lens = -reward  # (1, N)
            return float(tour_lens.min().item())

    def _pomo_infer_aug(
        self, node_xy: torch.Tensor, coords_orig: torch.Tensor, ew_type: str
    ) -> float:
        """POMO inference with x8 augmentation."""
        N = node_xy.size(1)

        # Augment
        aug_problems = augment_xy_data_by_8_fold(node_xy)  # (8, N, 2)
        effective_batch = aug_problems.size(0)

        env = TSPEnv(problem_size=N, pomo_size=N)
        env.batch_size = effective_batch
        env.problems = aug_problems.to(self.device)
        env.BATCH_IDX = torch.arange(effective_batch, device=self.device)[:, None].expand(
            effective_batch, N
        )
        env.POMO_IDX = torch.arange(N, device=self.device)[None, :].expand(effective_batch, N)

        # Original coords for cost computation
        env.original_node_xy_lib = coords_orig.to(self.device)
        env.edge_weight_type = ew_type

        with torch.no_grad():
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected, lib_mode=True)

            tour_lens = -reward  # (8, N)
            return float(tour_lens.min().item())

    def full_evaluate(
        self, projection_fn: Callable
    ) -> Dict[str, Any]:
        """Full evaluation on all validation instances with augmentation."""
        gaps = []
        no_aug_gaps = []
        improved = 0
        baseline_gaps = []

        for coords, optimal, name, ew_type in self.val_instances:
            try:
                # With projection + augmentation
                gap = self._eval_one(projection_fn, coords, optimal, ew_type)
                gaps.append(gap)

                # Without projection (baseline), single pass
                N = coords.size(0)
                coords_batch = coords.unsqueeze(0).to(self.device)
                norm = _normalize_to_unit_square(coords_batch)
                bl_gap = self._eval_one_baseline(norm, coords_batch, optimal, ew_type)
                baseline_gaps.append(bl_gap)

                if gap < bl_gap:
                    improved += 1
            except Exception:
                continue

        result = {
            "avg_gap": float(np.mean(gaps)) if gaps else 100.0,
            "avg_baseline_gap": float(np.mean(baseline_gaps)) if baseline_gaps else 100.0,
            "improved_instances": improved,
            "total_instances": len(gaps),
            "improve_rate": improved / len(gaps) if gaps else 0.0,
            "gaps": gaps,
            "baseline_gaps": baseline_gaps,
        }
        return result

    def _eval_one_baseline(
        self, node_xy: torch.Tensor, coords_orig: torch.Tensor, optimal, ew_type: str
    ) -> float:
        """Baseline evaluation without projection."""
        return self._eval_one(lambda x: x, coords_orig.squeeze(0), optimal, ew_type)
