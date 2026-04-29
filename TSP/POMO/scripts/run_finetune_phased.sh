#!/usr/bin/env bash
# Phased fine-tune launcher.
#
# Activates the drl_tsp conda env, cd's into TSP/POMO, then invokes
# `python finetune_phased.py` with sane defaults. Each tunable maps 1:1
# to a flag on `finetune_phased.py`; passing CLI flags to this script
# forwards them through unchanged.
#
# Usage:
#   bash scripts/run_finetune_phased.sh                          # default 400-epoch recipe
#   bash scripts/run_finetune_phased.sh --total_finetune_epochs 200
#   bash scripts/run_finetune_phased.sh --ablation bias_only
#   CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_finetune_phased.sh \
#       --desc B400_all_three > logs/phased_B400.log 2>&1 &
#
# Common env vars (also accepted as CLI flags - CLI wins):
#   B            -> --total_finetune_epochs
#   ABLATION     -> --ablation
#   RESUME_CKPT  -> --resume_checkpoint
#   DESC         -> --desc
#   CONFIG       -> --config
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
mkdir -p logs

B="${B:-400}"
ABLATION="${ABLATION:-all_three}"
RESUME_CKPT="${RESUME_CKPT:-./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt}"
DESC="${DESC:-phased}"
CONFIG="${CONFIG:-./configs/finetune_phased.json}"

EXTRA_ARGS=("$@")

echo "=================================================================="
echo " [phased]   B=$B  ablation=$ABLATION"
echo " [phased]   resume=$RESUME_CKPT"
echo " [phased]   config=$CONFIG"
echo " [phased]   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " [phased]   start=$(date '+%F %T')"
echo "=================================================================="

python finetune_phased.py \
    --config "$CONFIG" \
    --total_finetune_epochs "$B" \
    --resume_checkpoint "$RESUME_CKPT" \
    --ablation "$ABLATION" \
    --desc "$DESC" \
    "${EXTRA_ARGS[@]}"

# Locate the run directory we just produced.
RUN_DIR="$(ls -td result/*phased_${DESC}_B${B}_${ABLATION}* 2>/dev/null | head -1)"
if [[ -n "${RUN_DIR}" ]]; then
    echo "[phased] result_dir=$RUN_DIR"
    echo "[phased] phase checkpoints:"
    ls -1 "${RUN_DIR}"/checkpoint-phase_*.pt 2>/dev/null || true
fi

echo "=================================================================="
echo " [phased] DONE  $(date '+%F %T')"
echo "=================================================================="
