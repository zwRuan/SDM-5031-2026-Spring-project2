"""M2: distance-aware / kNN logit bias.

The bias is applied as an additive term on the decoder's single-head
logit *before* the ninf_mask is added, so masked (visited) nodes stay
``-inf`` and the numerics are stable.

Usage:
    module = DistanceBiasModule(cfg)
    model.attach_distance_bias(module)
    # training / inference loop proceeds unchanged; the model calls
    # ``module.prepare(problems)`` inside ``pre_forward`` and
    # ``module(current_node) -> bias (batch, pomo, N)`` inside the decoder.

Configuration keys (all optional; defaults OFF):
    distance_bias_enabled:   bool
    distance_bias_scale:     float
    distance_bias_mode:      {"logit", "attn"} - only "logit" is implemented.
    distance_norm_mode:      {"none", "mean", "max", "std"}
    knn_bias_enabled:        bool
    knn_k:                   int
    knn_bias_value:          float
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


DEFAULT_CFG: Dict[str, Any] = {
    "distance_bias_enabled": False,
    "distance_bias_scale": 1.0,
    "distance_bias_mode": "logit",
    "distance_norm_mode": "mean",
    "knn_bias_enabled": False,
    "knn_k": 10,
    "knn_bias_value": 0.5,
}


class DistanceBiasModule(nn.Module):
    """Non-learnable (by default) logit-bias module for POMO decoder."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        merged = dict(DEFAULT_CFG)
        if cfg:
            merged.update(cfg)
        self.cfg = merged
        if merged["distance_bias_mode"] not in {"logit", "attn"}:
            raise ValueError("distance_bias_mode must be 'logit' or 'attn'.")
        if merged["distance_bias_mode"] == "attn":
            # We expose the flag for forward compatibility but the implementation
            # scope (per plan) is logit-only. Falling back silently is confusing,
            # so we refuse early.
            raise NotImplementedError(
                "distance_bias_mode=attn is intentionally not implemented; "
                "see README_modified.md for the rationale. Use 'logit'."
            )
        if merged["distance_norm_mode"] not in {"none", "mean", "max", "std"}:
            raise ValueError("distance_norm_mode must be one of none/mean/max/std.")

        self._norm_dist: Optional[torch.Tensor] = None
        self._knn_mask: Optional[torch.Tensor] = None
        # Cache to avoid recomputing on the same problem tensor during a single forward.
        self._problem_version: int = 0

    # --- convenience flags ---------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.cfg["distance_bias_enabled"] or self.cfg["knn_bias_enabled"])

    @property
    def distance_enabled(self) -> bool:
        return bool(self.cfg["distance_bias_enabled"])

    @property
    def knn_enabled(self) -> bool:
        return bool(self.cfg["knn_bias_enabled"])

    # --- preparation (called once per encoder pass) --------------------------

    def prepare(self, problems: torch.Tensor) -> None:
        """Compute pairwise distances and kNN mask for the current batch.

        Args:
            problems: (batch, N, 2) float tensor (normalized or raw coords).
        """
        if not self.enabled:
            self._norm_dist = None
            self._knn_mask = None
            return

        with torch.no_grad():
            diff = problems[:, :, None, :] - problems[:, None, :, :]
            dist = (diff * diff).sum(-1).clamp(min=0.0).sqrt()  # (batch, N, N)

            # Exclude self-distance (which is 0) from statistics.
            B, N, _ = dist.shape
            eye = torch.eye(N, dtype=torch.bool, device=dist.device)[None]
            dist_nodiag = dist.masked_fill(eye, float("nan"))

            mode = self.cfg["distance_norm_mode"]
            if mode == "none":
                norm = dist
            elif mode == "mean":
                mean = torch.nanmean(dist_nodiag, dim=(1, 2), keepdim=True).clamp(min=1e-8)
                norm = dist / mean
            elif mode == "max":
                flat = dist_nodiag.reshape(B, -1)
                mx = flat.nan_to_num(nan=-float("inf")).max(dim=1, keepdim=True).values
                mx = mx.clamp(min=1e-8).view(B, 1, 1)
                norm = dist / mx
            elif mode == "std":
                valid = ~eye
                n_valid = valid.sum().item() // B if B > 0 else 1
                mean = torch.nanmean(dist_nodiag, dim=(1, 2), keepdim=True)
                var = ((dist - mean) ** 2).masked_fill(eye, 0.0).sum(dim=(1, 2), keepdim=True) / max(n_valid, 1)
                std = var.clamp(min=1e-12).sqrt()
                norm = (dist - mean) / std
            else:  # pragma: no cover
                norm = dist

            self._norm_dist = norm  # (batch, N, N)

            if self.knn_enabled:
                k = int(self.cfg["knn_k"])
                k = min(max(k, 1), N - 1)
                # For kNN we use the raw distances (not normalized) for correctness
                # of neighborhood definition.
                sorted_dist = dist.masked_fill(eye, float("inf"))
                _, nn_idx = sorted_dist.topk(k, dim=-1, largest=False)  # (batch, N, k)
                mask = torch.zeros_like(dist, dtype=torch.bool)
                batch_idx = torch.arange(B, device=dist.device)[:, None, None].expand(B, N, k)
                row_idx = torch.arange(N, device=dist.device)[None, :, None].expand(B, N, k)
                mask[batch_idx, row_idx, nn_idx] = True
                self._knn_mask = mask  # (batch, N, N)
            else:
                self._knn_mask = None

    # --- forward -------------------------------------------------------------

    def forward(self, current_node: torch.Tensor) -> torch.Tensor:
        """Return a bias of shape (batch, pomo, N) to add to decoder logits.

        Args:
            current_node: (batch, pomo) int64 tensor of currently-selected nodes.
        """
        if not self.enabled or self._norm_dist is None:
            batch, pomo = current_node.shape
            n = 0
            # Caller should only use bias if enabled. Return zeros of best shape.
            return torch.zeros(batch, pomo, n, device=current_node.device)

        batch, pomo = current_node.shape
        N = self._norm_dist.size(1)

        # Gather rows from the (batch, N, N) distance matrix at indices current_node.
        idx_expanded = current_node.unsqueeze(-1).expand(batch, pomo, N)  # (batch, pomo, N)
        dist_rows = self._norm_dist.gather(
            dim=1,
            index=current_node.unsqueeze(-1).expand(batch, pomo, N),
        )  # (batch, pomo, N)

        bias = torch.zeros(batch, pomo, N, device=self._norm_dist.device)
        if self.distance_enabled:
            scale = float(self.cfg["distance_bias_scale"])
            bias = bias - scale * dist_rows

        if self.knn_enabled and self._knn_mask is not None:
            knn_rows = self._knn_mask.gather(
                dim=1,
                index=current_node.unsqueeze(-1).expand(batch, pomo, N),
            ).to(bias.dtype)
            bias = bias + float(self.cfg["knn_bias_value"]) * knn_rows

        return bias
