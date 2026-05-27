"""Template and task description for POMO coordinate projection search.

The LLM generates a ``normalize()`` function that transforms TSP node coordinates
before they are fed to the pre-trained POMO neural solver.  The goal is to discover
a projection that minimizes the optimality gap.
"""

# ---------------------------------------------------------------------------
# Template program — the LLM fills in the body
# ---------------------------------------------------------------------------

template_program = '''
import torch

def normalize(coords: torch.Tensor) -> torch.Tensor:
    """
    Project TSP node coordinates to improve a neural solver's tour quality.

    Args:
        coords: Raw node coordinates, shape (batch, N, 2), dtype float32.

    Returns:
        Projected coordinates, same shape (batch, N, 2), dtype float32.

    The neural solver was trained on [0,1]^2 uniform instances.  Your projection
    should transform arbitrary input distributions into a representation that
    helps the model select better tours.  Keep all operations differentiable
    (or in torch.no_grad()) and fully vectorised over the batch dimension.

    Tips (the model is attention-based, so relative geometry matters):
      - Translation (centering) can help with spatial biases.
      - Scaling (by range, std, or inter-quartile range) controls "zoom".
      - Non-linear warping (tanh, sigmoid, sqrt, log1p, pow) changes density.
      - Per-dimension transforms can break symmetries that confuse attention.
      - Avoid hard clipping unless followed by a rescaling step.
      - Numerical safety: guard against div-by-zero and NaN.
    """
    batch_size = coords.shape[0]
    n_nodes = coords.shape[1]

    # ---- Basic min-max normalization (template — REPLACE / MODIFY) ----
    xy_max = torch.max(coords, dim=1, keepdim=True).values
    xy_min = torch.min(coords, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1.0
    normalized = (coords - xy_min) / ratio.expand(-1, 1, 2)
    normalized = torch.clip(normalized, 0.0, 1.0)
    # ------------------------------------------------------------------

    return normalized
'''

# ---------------------------------------------------------------------------
# Task description — guides the LLM's search direction
# ---------------------------------------------------------------------------

task_description = """
I need to design an innovative coordinate projection (normalization) function
implemented in PyTorch that preprocesses TSP node coordinates before a neural
solver (trained on uniform [0,1]^2 instances) constructs a tour.

**Goal**: Minimize the optimality gap (tour_length / optimal - 1) * 100.
Lower gap = better.  The solver sees ONLY the projected coordinates; the true
tour cost is always computed on the original coordinates.

**Constraints**:
- Input shape: (batch, N, 2).  Output shape: (batch, N, 2).
- Fully vectorised (no Python for-loops over batch).
- Return a finite tensor without NaN/Inf.
- The projection must be deterministic (same input → same output).

**Design space** (non-exhaustive):
1. Centering: mean, median, geometric-median, min, or first-node anchor.
2. Scaling: global range, per-axis range, standard deviation, MAD, quantile.
3. Non-linearities: tanh, sigmoid, softsign, sqrt, log1p, pow(exponent).
4. Composite: e.g. center → tanh → scale → clip.
5. Per-axis: apply different transforms to x vs y coordinates.
6. Distance-based: scale by distance to centroid or farthest pair.

**Hint**: The model has a distance/kNN bias module that encourages nearby
nodes to be selected consecutively.  Preserving or enhancing local proximity
structure in the projected space can be beneficial.
"""
