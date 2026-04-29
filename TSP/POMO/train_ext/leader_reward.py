"""M3: leader-focused reward / training objective.

Two variants are supported (per spec):

    leader_mode = "bonus_adv":
        A_i = (r_i - b) + gamma * (r_leader - b) * 1[i == argmax r_i]
        loss = -(A * log_prob).mean()
        (Baseline POMO advantage + extra bonus targeting the best rollout.)

    leader_mode = "aux_imitation":
        loss_pg  = -(A_baseline * log_prob).mean()
        loss_aux = -log_prob[leader].mean()
        loss    = loss_pg + lambda_leader * loss_aux

Both modes are numerically safe: NaN detection, configurable grad clipping
is expected to be applied by the trainer (we just return the loss). A
telemetry dict is returned so the trainer can log leader statistics.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


DEFAULT_CFG: Dict[str, Any] = {
    "leader_reward_enabled": False,
    "leader_mode": "bonus_adv",  # or "aux_imitation"
    "leader_gamma": 0.5,
    "leader_aux_weight": 0.1,
}


def compute_leader_loss(
    reward: torch.Tensor,
    log_prob: torch.Tensor,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute leader-focused loss.

    Args:
        reward: (batch, pomo) tensor of rollout rewards (negative tour lengths).
        log_prob: (batch, pomo) cumulative log probability of each rollout.
        cfg: config dict (see DEFAULT_CFG).

    Returns:
        loss: scalar tensor.
        stats: dict of scalar telemetry (advantage mean/std, leader advantage,
               reward std, nan_detected flag, leader fraction, loss components).
    """
    config = dict(DEFAULT_CFG)
    if cfg:
        config.update(cfg)

    assert reward.dim() == 2 and log_prob.dim() == 2, "shapes must be (batch, pomo)"
    baseline = reward.float().mean(dim=1, keepdim=True)  # (batch, 1)
    base_adv = reward - baseline  # (batch, pomo)

    # Leader = argmax reward (for negative-length rewards the max is the shortest tour).
    leader_idx = reward.float().argmax(dim=1)  # (batch,)
    batch_size = reward.size(0)
    batch_arange = torch.arange(batch_size, device=reward.device)
    leader_mask = torch.zeros_like(reward, dtype=torch.bool)
    leader_mask[batch_arange, leader_idx] = True

    mode = config["leader_mode"]
    nan_detected = False

    if mode == "bonus_adv":
        gamma = float(config["leader_gamma"])
        leader_adv = (reward[batch_arange, leader_idx] - baseline.squeeze(1))  # (batch,)
        bonus = torch.zeros_like(reward)
        bonus[batch_arange, leader_idx] = gamma * leader_adv
        advantage = base_adv + bonus
        loss = -(advantage * log_prob).mean()
        pg_component = loss.detach()
        aux_component = torch.tensor(0.0, device=loss.device)
    elif mode == "aux_imitation":
        lambda_leader = float(config["leader_aux_weight"])
        pg_loss = -(base_adv * log_prob).mean()
        leader_lp = log_prob[batch_arange, leader_idx]
        aux_loss = -leader_lp.mean()
        loss = pg_loss + lambda_leader * aux_loss
        pg_component = pg_loss.detach()
        aux_component = (lambda_leader * aux_loss).detach()
    else:
        raise ValueError(f"Unknown leader_mode: {mode}")

    if not torch.isfinite(loss):
        nan_detected = True

    stats = {
        "leader_mode": mode,
        "base_adv_mean": float(base_adv.float().mean().item()),
        "base_adv_std": float(base_adv.float().std().item()),
        "leader_reward_mean": float(reward[batch_arange, leader_idx].float().mean().item()),
        "reward_std": float(reward.float().std().item()),
        "loss_pg": float(pg_component.item()),
        "loss_aux": float(aux_component.item()),
        "nan_detected": bool(nan_detected),
    }
    return loss, stats
