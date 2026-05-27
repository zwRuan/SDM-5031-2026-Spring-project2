"""Test-time coordinate projection for POMO TSP solver.

This module is auto-discovered by TSPTester_LIB at inference time.
If present, the ``projection()`` function is applied to coordinates
before the POMO model runs.

The default projection is identity (standard min-max normalization).
After running the search (run_search.py), this file is overwritten with
the LLM-discovered best function.
"""

import torch


def projection(coords: torch.Tensor) -> torch.Tensor:
    """Project TSP coordinates to improve POMO model tour quality.

    Args:
        coords: Node coordinates, shape (batch, N, 2), dtype float32.
                These are already min-max normalized to [0,1] by the tester.

    Returns:
        Projected coordinates, same shape (batch, N, 2), dtype float32.
    """
    # ---- Identity projection (no change) ----
    # Replace this body with the LLM-discovered function after search.
    return coords
