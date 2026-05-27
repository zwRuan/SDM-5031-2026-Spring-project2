"""Evaluation module for Project 2: LLM-designed augmentation strategy for POMO TSP.

This module wraps the POMO model evaluation pipeline so that EoH can evaluate
LLM-generated augmentation functions by using them during inference and measuring
avg_aug_gap on the validation set.
"""
from __future__ import annotations

import os
import sys
import logging
import traceback
from typing import Any, Optional

# Suppress verbose logging during evaluation
logging.disable(logging.WARNING)

# Path setup - resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POMO_DIR = os.path.join(PROJECT_ROOT, "TSP", "POMO")
TSP_DIR = os.path.join(PROJECT_ROOT, "TSP")

sys.path.insert(0, POMO_DIR)
sys.path.insert(0, TSP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from template import get_template_and_description

# ---- Configuration ----
CHECKPOINT_PATH = os.path.join(POMO_DIR, "result", "best_ckpt_2", "checkpoint-best.pt")
VAL_DATA_PATH = os.path.join(TSP_DIR, "data", "val")

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

# Augmentation factor for LLM-designed strategy
# Rule: aug <= problem_size * 8. We use a moderate value for fast eval during search.
EVAL_AUG_FACTOR = 32  # 32 augmentations during EoH search (fast yet effective)
FINAL_AUG_FACTOR = 64  # 64 for final evaluation


class LLMBiasModule:
    """Wraps an LLM-generated bias function as a bias module compatible with POMO."""

    def __init__(self, bias_func):
        self.bias_func = bias_func
        self.enabled = True
        self._problems = None

    def prepare(self, problems: torch.Tensor):
        """Store the problem coordinates for use in forward."""
        self._problems = problems

    def __call__(self, current_node: torch.Tensor) -> torch.Tensor:
        """Compute bias using the LLM-generated function."""
        if self._problems is None:
            batch, pomo = current_node.shape
            return torch.zeros(batch, pomo, 0, device=current_node.device)
        return self.bias_func(self._problems, current_node)


def normalize_to_unit_square(node_xy: torch.Tensor) -> torch.Tensor:
    """Normalize coordinates to [0,1] with uniform scaling."""
    xy_max = torch.max(node_xy, dim=1, keepdim=True).values
    xy_min = torch.min(node_xy, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1
    return (node_xy - xy_min) / ratio.expand(-1, 1, 2)


def standard_8fold_augment(xy_data: torch.Tensor) -> torch.Tensor:
    """Standard POMO 8-fold dihedral augmentation (baseline)."""
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    return torch.cat([
        torch.cat((x, y), dim=2),
        torch.cat((1 - x, y), dim=2),
        torch.cat((x, 1 - y), dim=2),
        torch.cat((1 - x, 1 - y), dim=2),
        torch.cat((y, x), dim=2),
        torch.cat((1 - y, x), dim=2),
        torch.cat((y, 1 - x), dim=2),
        torch.cat((1 - y, 1 - x), dim=2),
    ], dim=0)


def evaluate_single_instance(model, device, nodes_xy_normalized, coords_orig, ew_type, aug_factor=8):
    """Evaluate a single TSPLIB instance with the model using standard 8-fold aug."""
    from TSPEnv import TSPEnv as Env

    problems = standard_8fold_augment(nodes_xy_normalized) if aug_factor > 1 else nodes_xy_normalized

    effective_batch = problems.size(0)
    problem_size = problems.size(1)

    env = Env(problem_size=problem_size, pomo_size=problem_size)
    env.batch_size = effective_batch
    env.problems = problems.to(device)
    env.BATCH_IDX = torch.arange(effective_batch, device=device)[:, None].expand(effective_batch, env.pomo_size)
    env.POMO_IDX = torch.arange(env.pomo_size, device=device)[None, :].expand(effective_batch, env.pomo_size)
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
    no_aug_score = float(tour_lens[0].min().item())
    aug_score = float(tour_lens.min().item())
    return no_aug_score, aug_score


def evaluate_with_custom_aug(model, device, nodes_xy_normalized, coords_orig, ew_type, aug_func, aug_factor):
    """Evaluate using a custom augmentation function."""
    from TSPEnv import TSPEnv as Env

    # Generate augmented problems using LLM-designed function
    try:
        problems = aug_func(nodes_xy_normalized, aug_factor)
    except TypeError:
        # Fallback: function might not accept aug_factor
        problems = aug_func(nodes_xy_normalized)

    # Validate output shape
    if problems is None or problems.dim() != 3 or problems.size(2) != 2:
        return float('inf')

    # Clamp to [0,1] for safety
    problems = problems.clamp(0.0, 1.0)

    effective_batch = problems.size(0)
    problem_size = problems.size(1)

    env = Env(problem_size=problem_size, pomo_size=problem_size)
    env.batch_size = effective_batch
    env.problems = problems.to(device)
    env.BATCH_IDX = torch.arange(effective_batch, device=device)[:, None].expand(effective_batch, env.pomo_size)
    env.POMO_IDX = torch.arange(env.pomo_size, device=device)[None, :].expand(effective_batch, env.pomo_size)
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


def run_evaluation(aug_func, aug_factor=EVAL_AUG_FACTOR) -> float:
    """Run POMO evaluation with the given augmentation function on the validation set.

    Args:
        aug_func: A callable with signature (coords, aug_factor) -> augmented coords
        aug_factor: Number of augmentations to generate

    Returns:
        Negative avg_aug_gap (higher is better for EoH maximization).
        Returns a large negative number on failure.
    """
    from TSPModel import TSPModel as Model
    from tsplib_utils import TSPLIBReader, tsplib_cost

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
        torch.cuda.set_device(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    # Load model
    model = Model(**MODEL_PARAMS)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Evaluate on validation set
    aug_gaps = []

    for root, _, files in os.walk(VAL_DATA_PATH):
        for file in sorted(files):
            if not file.endswith('.tsp'):
                continue

            full_path = os.path.join(root, file)
            name, dimension, locs, ew_type = TSPLIBReader(full_path)
            if name is None:
                continue

            optimal = tsplib_cost.get(name, None)
            if optimal is None:
                continue

            coords_orig_np = np.array(locs, dtype=np.float32)
            coords_orig = torch.from_numpy(coords_orig_np).to(device)
            node_coord = coords_orig[None, :, :]
            nodes_xy_normalized = normalize_to_unit_square(node_coord)

            try:
                aug_score = evaluate_with_custom_aug(
                    model, device, nodes_xy_normalized, coords_orig, ew_type,
                    aug_func, aug_factor
                )
                aug_gap = (aug_score - optimal) / optimal * 100
                aug_gaps.append(aug_gap)
            except Exception:
                aug_gaps.append(100.0)  # penalty for failure

    if not aug_gaps:
        return -100.0

    avg_aug_gap = float(np.mean(aug_gaps))
    # Return negative gap (EoH maximizes fitness)
    return -avg_aug_gap


# ---- LLM4AD Evaluation Interface ----

# Add LLM4AD to path
LLM4AD_DIR = os.path.join(PROJECT_ROOT, "src", "LLM4AD")
sys.path.insert(0, LLM4AD_DIR)

from llm4ad.base import Evaluation


class POMOBiasEvaluation(Evaluation):
    """LLM4AD-compatible evaluation class for POMO augmentation strategy search."""

    def __init__(self, timeout_seconds=300, **kwargs):
        template_program, task_description_str = get_template_and_description()
        super().__init__(
            template_program=template_program,
            task_description=task_description_str,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        """Evaluate an LLM-generated augmentation function.

        Args:
            program_str: The source code of the generated function.
            callable_func: The compiled callable function.

        Returns:
            Negative avg_aug_gap (float), or None on failure.
        """
        try:
            fitness = run_evaluation(callable_func, aug_factor=EVAL_AUG_FACTOR)
            print(f"  [Eval] avg_aug_gap = {-fitness:.4f}%, fitness = {fitness:.4f}")
            return fitness
        except Exception as e:
            print(f"  [Eval] FAILED: {e}")
            traceback.print_exc()
            return None
