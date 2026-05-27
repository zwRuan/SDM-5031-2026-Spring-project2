"""Template and task description for LLM-designed augmentation strategy.

The LLM will generate a `generate_augmentations` function that takes normalized
node coordinates and returns multiple augmented versions (coordinate transformations)
to be used during POMO inference. The model runs on each augmented version independently
and takes the best solution across all augmentations.

This is the core of Project 2: LLM-designed test-time augmentation strategy.
"""

template_program = '''
import torch
import numpy as np

def generate_augmentations(coords: torch.Tensor, aug_factor: int = 8) -> torch.Tensor:
    """Generate augmented versions of TSP instance coordinates for test-time augmentation.

    The POMO model runs independently on each augmented copy and picks the shortest tour.
    More diverse yet valid augmentations lead to better solutions.

    Args:
        coords: normalized coordinates of nodes, shape (1, num_nodes, 2), values in [0, 1].
        aug_factor: number of augmented copies to generate (default 8).

    Returns:
        augmented: tensor of shape (aug_factor, num_nodes, 2), each slice is a valid
                   transformation of the input coordinates, with values in [0, 1].

    Notes:
        - Standard POMO uses the 8 dihedral group transformations (rotations + reflections).
        - You should design transformations that provide DIVERSE viewpoints of the same problem.
        - Valid transformations include: rotations, reflections, scaling, translations
          (as long as output stays in [0,1] after re-normalization).
        - The key insight: different rotations can lead to different greedy solutions;
          more angular diversity = higher chance of finding the optimal tour.
        - Use vectorized PyTorch operations. Avoid for-loops over nodes.
        - After any transformation, re-normalize to [0,1] using min-max scaling.

    # Example: standard dihedral 8-fold (baseline to beat)
    x = coords[:, :, [0]]
    y = coords[:, :, [1]]
    augmented = torch.cat([
        torch.cat((x, y), dim=2),
        torch.cat((1-x, y), dim=2),
        torch.cat((x, 1-y), dim=2),
        torch.cat((1-x, 1-y), dim=2),
        torch.cat((y, x), dim=2),
        torch.cat((1-y, x), dim=2),
        torch.cat((y, 1-x), dim=2),
        torch.cat((1-y, 1-x), dim=2),
    ], dim=0)
    """
    return augmented
'''

task_description = (
    "I need help designing an innovative test-time augmentation strategy for a POMO-based "
    "neural TSP solver. The function receives normalized node coordinates (1, num_nodes, 2) "
    "in [0,1] and must return aug_factor augmented copies (aug_factor, num_nodes, 2). "
    "The solver runs independently on each copy and picks the best (shortest) tour. "
    "Standard POMO uses 8 dihedral group transformations (4 rotations × 2 reflections). "
    "Your goal is to design a BETTER set of augmentations that produces MORE DIVERSE "
    "viewpoints of the problem, leading to shorter tours. "
    "Consider: (1) rotations at non-standard angles (not just 0/90/180/270), "
    "(2) combining rotations with reflections in novel ways, "
    "(3) using the golden angle (≈137.5°) for maximum angular diversity, "
    "(4) slight non-linear warps that preserve topology, "
    "(5) anisotropic scaling before rotation. "
    "After each transformation, re-normalize coordinates to [0,1] using min-max scaling. "
    "Use vectorized PyTorch ops (no for-loops over nodes). "
    "The goal is to maximize negative gap (minimize optimality gap)."
)


def get_template_and_description():
    return template_program, task_description
