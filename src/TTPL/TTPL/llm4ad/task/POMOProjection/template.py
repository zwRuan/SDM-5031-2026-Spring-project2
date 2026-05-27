"""Template and task description for POMO coordinate projection search.

The LLM generates a ``normalize()`` function that transforms TSP node coordinates
before they are fed to the pre-trained POMO neural solver.
"""

# Template — LLM fills in the function body
template_program = '''
import torch

def normalize(coords: torch.Tensor) -> torch.Tensor:
    """
    Project TSP node coordinates to improve a POMO neural solver's tour quality.

    Args:
        coords: Raw node coordinates, shape (batch, N, 2), dtype float32.

    Returns:
        Projected coordinates, same shape (batch, N, 2), dtype float32.

    The solver was trained on [0,1]^2 uniform data and uses attention over
    node embeddings, so relative geometry (distances, angles, density) matters
    more than absolute positions.

    Design ideas (modify the template below):
      - Center at mean / median / min / first-node before scaling.
      - Scale by range, std, quantile, or distance-to-centroid.
      - Apply tanh, sigmoid, or softsign for soft saturation.
      - Use per-axis (x vs y) transforms to break symmetry.
      - Preserve or enhance local kNN structure (the model has a kNN bias).
    """
    batch_size = coords.shape[0]
    n_nodes = coords.shape[1]

    # ---- Template: min-max normalization (REPLACE / IMPROVE) ----
    xy_max = torch.max(coords, dim=1, keepdim=True).values
    xy_min = torch.min(coords, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1.0
    normalized = (coords - xy_min) / ratio.expand(-1, 1, 2)
    normalized = torch.clip(normalized, 0.0, 1.0)
    # ---------------------------------------------------------------

    return normalized
'''

task_description = """
I need to design an innovative coordinate projection (normalization) function
implemented in PyTorch that preprocesses TSP node coordinates before a neural
solver (trained on uniform [0,1]^2 instances) constructs a tour.

**Goal**: Minimize the optimality gap = (tour_length / optimal - 1) * 100.
Lower gap = better.  The solver sees only the projected coordinates; the true
tour cost is always computed on the original (unprojected) coordinates.

**Constraints**:
- Input shape: (batch, N, 2).  Output shape: (batch, N, 2), same dtype/device.
- Fully vectorised (no Python for-loops over the batch dimension).
- Return a finite tensor without NaN or Inf.
- Deterministic: same input always produces the same output.

**Design dimensions** (non-exhaustive, be creative):
1. Centering: mean, median, geometric-median, min-corner, centroid-of-extremes.
2. Scaling: global range, per-axis range, standard deviation, MAD, quantile range.
3. Non-linear warping: tanh, sigmoid, softsign, sqrt, log1p, pow(exponent), exp(-x).
4. Composition: e.g. center -> tanh -> scale -> clip, or scale -> sqrt -> center.
5. Per-axis: apply different strategies to x and y coordinates.
6. Distance-aware: scale by distance to centroid or farthest-pair distance.

**Hint**: The model has a distance/kNN bias module that encourages nearby nodes
to be selected consecutively.  Preserving local proximity structure in the
projected space can be beneficial.  Overly aggressive scaling that collapses
nodes together will destroy the spatial signal.
"""
