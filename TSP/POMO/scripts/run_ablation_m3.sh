#!/usr/bin/env bash
# M3 leader-focused reward ablations (TRAINING-SIDE).
#
# This script launches SHORT finetune runs from the baseline checkpoint.
# It then evaluates each resulting checkpoint with `test.py`. Adjust
# `EPOCHS` if you want longer training.
#
# Required env vars (with defaults):
#   EPOCHS           : finetune epochs (default 5)
#   EPISODES         : train_episodes   (default 10000)
#   BATCH_SIZE       : train_batch_size (default 64)

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

EPOCHS="${EPOCHS:-5}"
EPISODES="${EPISODES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

FT_BASE="${CHECKPOINT_PATH}"

finetune_and_eval() {
    local run_name="$1"
    local train_extra="$2"
    local desc="ft_${run_name}"

    # NOTE: train.py writes to ./result/<timestamp>__<desc>. We locate the
    # most recent matching directory after finishing.
    python train.py \
        --finetune_from "$FT_BASE" \
        --epochs "$EPOCHS" \
        --train_episodes "$EPISODES" \
        --train_batch_size "$BATCH_SIZE" \
        --lr 1e-5 \
        --desc "$desc" \
        $train_extra

    local new_ckpt
    new_ckpt=$(ls -td ./result/*"$desc" 2>/dev/null | head -n 1)/"checkpoint-${EPOCHS}.pt"
    if [[ ! -f "$new_ckpt" ]]; then
        echo "Finetune checkpoint not found for $run_name (expected $new_ckpt). Skipping eval." >&2
        return
    fi

    # Eval with this checkpoint.
    CHECKPOINT_PATH="$new_ckpt" run_eval "$run_name" ""
}

# M3-A: baseline already produced elsewhere; here we add leader variants.
finetune_and_eval "m3_bonus_gamma0.5" "--leader_reward_enabled true --leader_mode bonus_adv --leader_gamma 0.5"
finetune_and_eval "m3_aux_lambda0.1"  "--leader_reward_enabled true --leader_mode aux_imitation --leader_aux_weight 0.1"

# M3-B: gamma / aux_weight sweeps
finetune_and_eval "m3_bonus_gamma0.1" "--leader_reward_enabled true --leader_mode bonus_adv --leader_gamma 0.1"
finetune_and_eval "m3_bonus_gamma1.0" "--leader_reward_enabled true --leader_mode bonus_adv --leader_gamma 1.0"
finetune_and_eval "m3_aux_lambda0.01" "--leader_reward_enabled true --leader_mode aux_imitation --leader_aux_weight 0.01"
finetune_and_eval "m3_aux_lambda1.0"  "--leader_reward_enabled true --leader_mode aux_imitation --leader_aux_weight 1.0"

# M3-C: leader + distance bias combination
finetune_and_eval "m3_bonus_plus_dist" "--leader_reward_enabled true --leader_mode bonus_adv --leader_gamma 0.5 --distance_bias_enabled true --distance_bias_scale 1.0"
