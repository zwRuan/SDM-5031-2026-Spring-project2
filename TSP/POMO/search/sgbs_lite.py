"""M1: Augmentation-aware SGBS-lite reranking.

This module runs a light-weight beam search on top of the POMO decoder to
produce additional candidate tours at inference time. It is intentionally
kept simple and stable:

    1. Initialise ``pomo_new = N * beam_width`` rollouts. Slot k has
       start-node ``k // beam_width``.
    2. At the first decoding step (choosing the 2nd node after the start)
       pick the top-``beam_width`` distinct actions per starting node; no
       reindexing is needed because the ``beam_width`` rollouts in a group
       share the same state.
    3. For subsequent steps 2..``rerank_depth``: each live beam proposes
       top-``rerank_topk_per_step`` children, the full set is grouped by
       starting node (size ``beam_width * topk`` per group), and pruned
       back to ``beam_width`` by cumulative log-probability. The env state
       (``selected_node_list``, ``ninf_mask``, ``current_node``) is
       re-gathered along the ``pomo`` axis accordingly.
    4. From step ``rerank_depth + 1`` onwards rollouts complete greedily
       (argmax).
    5. All candidate tours are scored with the official TSPLIB rounding
       and returned. The caller decides how to pool with baseline tours.

This does NOT modify the model or env in-place beyond creating a fresh
env instance sized for the enlarged pomo dimension.

Entropy gate: when ``rerank_use_entropy_gate=True``, if the mean per-step
entropy is below ``rerank_entropy_threshold`` we skip expansion at that
step and take argmax (but still carry the full ``pomo_new`` beam).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Local env/model imports are deferred to keep this file usable when run as
# a script, but TSPEnv / TSPModel are resolved via sys.path by the caller.


DEFAULT_CFG: Dict[str, Any] = {
    "rerank_enabled": False,
    "rerank_beam_width": 4,
    "rerank_depth": 5,
    "rerank_topk_per_step": 4,
    "rerank_use_entropy_gate": False,
    "rerank_entropy_threshold": 1.0,
    "rerank_pool_across_augs": True,
    "rerank_deduplicate": True,
}


def _gather_pomo(tensor: torch.Tensor, parent_idx: torch.Tensor) -> torch.Tensor:
    """Reindex tensor along the pomo axis (dim=1).

    Args:
        tensor: shape (aug, pomo_old, *rest).
        parent_idx: shape (aug, pomo_new) with values in [0, pomo_old).
    Returns:
        shape (aug, pomo_new, *rest).
    """
    aug = parent_idx.size(0)
    pomo_new = parent_idx.size(1)
    rest = tensor.shape[2:]
    view_shape = (aug, pomo_new) + tuple(1 for _ in rest)
    expand_shape = (aug, pomo_new) + tuple(rest)
    idx = parent_idx.view(*view_shape).expand(*expand_shape)
    return tensor.gather(1, idx)


def _tsplib_round(x: torch.Tensor, ew_type: str) -> torch.Tensor:
    if ew_type == "CEIL_2D":
        return torch.ceil(x)
    if ew_type == "EUC_2D":
        return torch.floor(x + 0.5)
    return x


def _tour_lengths_lib(tours: torch.Tensor, coords_orig: torch.Tensor, ew_type: str) -> torch.Tensor:
    """Compute TSPLIB-rounded closed-cycle lengths.

    Args:
        tours: (aug, pomo, N) int indices.
        coords_orig: (1, N, 2) or (N, 2) original (un-normalized) coords.
        ew_type: distance rounding mode.
    Returns:
        lengths: (aug, pomo) float.
    """
    if coords_orig.dim() == 2:
        coords_orig = coords_orig[None]  # (1, N, 2)
    aug, pomo, n = tours.shape
    base = coords_orig
    if base.size(0) == 1 and aug != 1:
        base = base.expand(aug, -1, -1)
    gather_idx = tours.unsqueeze(-1).expand(aug, pomo, n, 2)
    seq_expanded = base[:, None, :, :].expand(aug, pomo, n, 2)
    ordered = seq_expanded.gather(dim=2, index=gather_idx)
    rolled = ordered.roll(dims=2, shifts=-1)
    seg_raw = ((ordered - rolled) ** 2).sum(-1).sqrt()
    seg = _tsplib_round(seg_raw, ew_type)
    return seg.sum(-1)


def _canonicalize_tour(tour: np.ndarray) -> Tuple[int, ...]:
    """Rotate so smallest node is first, flip to the lexicographically smaller direction."""
    n = len(tour)
    start = int(np.argmin(tour))
    rotated = np.concatenate([tour[start:], tour[:start]])
    reversed_ = np.concatenate([rotated[:1], rotated[1:][::-1]])
    fwd = tuple(rotated.tolist())
    bwd = tuple(reversed_.tolist())
    return fwd if fwd <= bwd else bwd


def rerank_sgbs_lite(
    model,
    env_cls,
    problems: torch.Tensor,
    coords_orig_lib: torch.Tensor,
    ew_type: str,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run SGBS-lite and return candidate tours + their lib lengths.

    Args:
        model: trained POMO model (already in eval()).
        env_cls: the TSPEnv class.
        problems: (aug, N, 2) already augmented (normalized) coords.
        coords_orig_lib: (N, 2) or (1, N, 2) ORIGINAL TSPLIB coords for scoring.
        ew_type: TSPLIB edge weight type.
        cfg: see DEFAULT_CFG.
        device: target device.

    Returns:
        tours_np: (aug, pomo_new, N) int numpy (cpu).
        lengths_np: (aug, pomo_new) float numpy (cpu).
        info: dict with
              {rerank_ms, n_candidates, beam, depth,
               gate_skipped_steps, expansions_done}.
    """
    config = dict(DEFAULT_CFG)
    config.update(cfg or {})
    assert config["rerank_enabled"], "rerank_sgbs_lite called with rerank_enabled=False"

    beam = int(config["rerank_beam_width"])
    depth = int(config["rerank_depth"])
    topk = int(config["rerank_topk_per_step"])
    use_gate = bool(config["rerank_use_entropy_gate"])
    ent_thresh = float(config["rerank_entropy_threshold"])

    t0 = time.time()
    aug, N, _ = problems.shape
    pomo_new = N * beam

    env = env_cls(problem_size=N, pomo_size=pomo_new)
    env.batch_size = aug
    env.problems = problems.to(device)
    env.BATCH_IDX = torch.arange(aug, device=device)[:, None].expand(aug, pomo_new)
    env.POMO_IDX = torch.arange(pomo_new, device=device)[None, :].expand(aug, pomo_new)
    if coords_orig_lib.dim() == 2:
        coords_orig_lib_b = coords_orig_lib[None].to(device)
    else:
        coords_orig_lib_b = coords_orig_lib.to(device)
    env.original_node_xy_lib = coords_orig_lib_b
    env.edge_weight_type = ew_type

    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    # Step 0: select start nodes manually. Slot k starts at k // beam.
    start_nodes = (
        torch.arange(N, device=device).repeat_interleave(beam)[None].expand(aug, pomo_new)
    )
    # Manually set q_first using the real start nodes (mirrors TSPModel.forward).
    from TSPModel import _get_encoding  # lazy import to avoid circulars

    encoded_first = _get_encoding(model.encoded_nodes, start_nodes)
    model.decoder.set_q1(encoded_first)

    state, reward, done = env.step(start_nodes, lib_mode=False)
    cum_logp = torch.zeros(aug, pomo_new, device=device)

    gate_skipped_steps = 0
    expansions_done = 0
    step_idx = 0  # count of decoding steps taken since start selection

    eps = 1e-20

    while not done:
        step_idx += 1
        encoded_last = _get_encoding(model.encoded_nodes, state.current_node)
        probs = model.decoder(encoded_last, ninf_mask=state.ninf_mask, current_node=state.current_node)
        # shape: (aug, pomo_new, N)

        do_expand = step_idx <= depth
        if do_expand and use_gate:
            # Average entropy across alive beams.
            safe_probs = probs.clamp(min=eps)
            entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)  # (aug, pomo_new)
            if entropy.mean().item() < ent_thresh:
                do_expand = False
                gate_skipped_steps += 1

        if not do_expand:
            # Greedy rollout for this step.
            # Must guard against zero-prob: pick argmax (matches baseline eval).
            selected = probs.argmax(dim=2)
            chosen_prob = probs.gather(2, selected[:, :, None]).squeeze(-1)
            cum_logp = cum_logp + chosen_prob.clamp(min=eps).log()
            state, reward, done = env.step(selected, lib_mode=False)
            continue

        # --- Beam expansion ---
        expansions_done += 1
        if step_idx == 1:
            # All beams within a starting-node group share state.
            # Directly take top-`beam` distinct next nodes per group.
            # Pick top-beam actions from the first rollout in each group.
            group_probs = probs.view(aug, N, beam, N)[:, :, 0, :]  # (aug, N, N)
            k_actual = min(beam, N - 1)
            top_vals, top_idx = group_probs.topk(k_actual, dim=-1)  # (aug, N, k)
            if k_actual < beam:
                # pad by repeating the top-1 (rare: N=1 shouldn't occur)
                pad = beam - k_actual
                top_vals = torch.cat([top_vals, top_vals[..., :1].expand(-1, -1, pad)], dim=-1)
                top_idx = torch.cat([top_idx, top_idx[..., :1].expand(-1, -1, pad)], dim=-1)
            selected = top_idx.reshape(aug, pomo_new)
            sel_probs = top_vals.reshape(aug, pomo_new)
            cum_logp = cum_logp + sel_probs.clamp(min=eps).log()
            state, reward, done = env.step(selected, lib_mode=False)
            continue

        # step_idx > 1: real beam expansion with reindexing.
        top_vals, top_idx = probs.topk(topk, dim=-1)  # (aug, pomo_new, topk)
        # child cumulative log-prob
        child_logp = cum_logp[:, :, None] + top_vals.clamp(min=eps).log()  # (aug, pomo_new, topk)
        # Reshape (aug, pomo_new, topk) = (aug, N, beam, topk) -> (aug, N, beam*topk)
        child_logp_g = child_logp.view(aug, N, beam * topk)
        child_act_g = top_idx.view(aug, N, beam * topk)
        # parent beam within group (0..beam-1) for each of the beam*topk children
        parent_in_group = torch.arange(beam * topk, device=device) // topk  # (beam*topk,)
        parent_in_group = parent_in_group.view(1, 1, -1).expand(aug, N, beam * topk)

        top_vals_p, top_idx_p = child_logp_g.topk(beam, dim=-1)  # (aug, N, beam)
        pruned_actions = child_act_g.gather(-1, top_idx_p)  # (aug, N, beam)
        pruned_parent_in_group = parent_in_group.gather(-1, top_idx_p)  # (aug, N, beam)
        group_offset = torch.arange(N, device=device)[None, :, None] * beam  # (1, N, 1)
        pruned_parent_global = (group_offset + pruned_parent_in_group).reshape(aug, pomo_new)
        pruned_actions_flat = pruned_actions.reshape(aug, pomo_new)
        cum_logp = top_vals_p.reshape(aug, pomo_new)

        # Reindex env state along the pomo axis to match pruned parents.
        env.selected_node_list = _gather_pomo(env.selected_node_list, pruned_parent_global)
        env.step_state.ninf_mask = _gather_pomo(env.step_state.ninf_mask, pruned_parent_global)
        env.current_node = _gather_pomo(env.current_node.unsqueeze(-1), pruned_parent_global).squeeze(-1)
        env.step_state.current_node = env.current_node

        state, reward, done = env.step(pruned_actions_flat, lib_mode=False)

    # Compute lib lengths on original coords.
    tours = env.selected_node_list  # (aug, pomo_new, N)
    lib_lengths = _tour_lengths_lib(tours, coords_orig_lib_b, ew_type)

    info = {
        "rerank_ms": float((time.time() - t0) * 1000.0),
        "n_candidates": int(aug * pomo_new),
        "beam": beam,
        "depth": depth,
        "topk": topk,
        "gate_skipped_steps": int(gate_skipped_steps),
        "expansions_done": int(expansions_done),
    }
    tours_np = tours.detach().cpu().numpy()
    lengths_np = lib_lengths.detach().cpu().numpy()
    return tours_np, lengths_np, info


def pool_and_select_best(
    baseline_tours: np.ndarray,
    baseline_lengths: np.ndarray,
    rerank_tours: Optional[np.ndarray],
    rerank_lengths: Optional[np.ndarray],
    pool_across_augs: bool,
    deduplicate: bool,
) -> Tuple[int, np.ndarray, float, Dict[str, Any]]:
    """Combine baseline and rerank candidate tours; return the best.

    Args:
        baseline_tours: (aug, pomo, N) int array.
        baseline_lengths: (aug, pomo) float array (lib length).
        rerank_tours: optional (aug, pomo_new, N).
        rerank_lengths: optional (aug, pomo_new).
        pool_across_augs: if False, take min per-aug then min across augs
            separately for baseline vs rerank, pool only within each aug.
        deduplicate: canonicalize and drop duplicate tours before selecting.

    Returns:
        best_source: 0 = baseline, 1 = rerank.
        best_tour: (N,) int numpy.
        best_length: float.
        info: dict {baseline_best, rerank_best, n_unique, improved}.
    """
    def _flatten(tours: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        aug, pomo, n = tours.shape
        return tours.reshape(aug * pomo, n), lengths.reshape(aug * pomo)

    baseline_best = float(baseline_lengths.min())
    if rerank_tours is None:
        best_tour = baseline_tours.reshape(-1, baseline_tours.shape[-1])[
            int(np.argmin(baseline_lengths))
        ]
        info = {
            "baseline_best": baseline_best,
            "rerank_best": None,
            "improved": False,
            "n_unique": None,
        }
        return 0, best_tour.copy(), baseline_best, info

    rerank_best = float(rerank_lengths.min())

    # Flatten candidate pools.
    b_flat_tours, b_flat_len = _flatten(baseline_tours, baseline_lengths)
    r_flat_tours, r_flat_len = _flatten(rerank_tours, rerank_lengths)

    if not pool_across_augs:
        # Keep per-aug best; then take min across augs. (We still pool both
        # baseline and rerank within the same aug.)
        aug = baseline_tours.shape[0]
        best_len_per_aug = np.empty(aug, dtype=np.float64)
        best_tour_per_aug = np.empty((aug, baseline_tours.shape[-1]), dtype=np.int64)
        best_src_per_aug = np.empty(aug, dtype=np.int64)
        for a in range(aug):
            cand_tours = np.concatenate([baseline_tours[a], rerank_tours[a]], axis=0)
            cand_len = np.concatenate([baseline_lengths[a], rerank_lengths[a]], axis=0)
            if deduplicate:
                cand_tours, cand_len = _dedup_tours(cand_tours, cand_len)
            idx = int(np.argmin(cand_len))
            best_len_per_aug[a] = cand_len[idx]
            best_tour_per_aug[a] = cand_tours[idx]
            # Recover source: rerank block starts at baseline_tours.shape[1]
            src = 1 if (idx >= baseline_tours.shape[1] - 0 and not deduplicate) else -1
            best_src_per_aug[a] = src  # source ambiguous after dedup; best-effort
        chosen = int(np.argmin(best_len_per_aug))
        best_tour = best_tour_per_aug[chosen]
        best_length = float(best_len_per_aug[chosen])
        info = {
            "baseline_best": baseline_best,
            "rerank_best": rerank_best,
            "improved": bool(best_length < baseline_best - 1e-9),
            "n_unique": None,
        }
        return int(best_length < baseline_best - 1e-9), best_tour.copy(), best_length, info

    # Global pool across augs.
    cand_tours = np.concatenate([b_flat_tours, r_flat_tours], axis=0)
    cand_len = np.concatenate([b_flat_len, r_flat_len], axis=0)
    baseline_count = b_flat_tours.shape[0]
    n_unique = len(cand_len)
    if deduplicate:
        cand_tours, cand_len, baseline_mask = _dedup_tours_with_origin(
            cand_tours, cand_len, baseline_count
        )
        n_unique = len(cand_len)
    idx = int(np.argmin(cand_len))
    best_tour = cand_tours[idx].copy()
    best_length = float(cand_len[idx])
    if deduplicate:
        best_source = 0 if baseline_mask[idx] else 1
    else:
        best_source = 0 if idx < baseline_count else 1
    info = {
        "baseline_best": baseline_best,
        "rerank_best": rerank_best,
        "improved": bool(best_length < baseline_best - 1e-9),
        "n_unique": int(n_unique),
    }
    return best_source, best_tour, best_length, info


def _dedup_tours(tours: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seen: Dict[Tuple[int, ...], int] = {}
    for i, t in enumerate(tours):
        key = _canonicalize_tour(t)
        if key not in seen or lengths[i] < lengths[seen[key]]:
            seen[key] = i
    keep = sorted(seen.values())
    return tours[keep], lengths[keep]


def _dedup_tours_with_origin(
    tours: np.ndarray, lengths: np.ndarray, baseline_count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seen: Dict[Tuple[int, ...], int] = {}
    for i, t in enumerate(tours):
        key = _canonicalize_tour(t)
        if key not in seen or lengths[i] < lengths[seen[key]]:
            seen[key] = i
    keep = sorted(seen.values())
    keep_arr = np.asarray(keep)
    baseline_mask = keep_arr < baseline_count
    return tours[keep_arr], lengths[keep_arr], baseline_mask
