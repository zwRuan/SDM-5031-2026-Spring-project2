#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Hyperparameter sweep launcher for finetune_phased.py.
#
# Usage:
#   bash scripts/run_sweep.sh                   # run all configs below
#   DRY_RUN=1 bash scripts/run_sweep.sh         # print commands only
#   MAX_JOBS=2 bash scripts/run_sweep.sh        # at most 2 runs in parallel
#   CUDA_LIST="0,1" bash scripts/run_sweep.sh   # round-robin across GPU 0 and 1
#
# After all runs finish, view results:
#   python scripts/show_sweep_results.py
#
# ===== HOW TO DEFINE RUNS ================================================
#
# Edit the SWEEP_CONFIGS array below.
# Each element is one string of CLI flags for finetune_phased.py.
# --desc is required and becomes the result-directory tag.
# ---------------------------------------------------------------------------

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# --------------------------------------------------------------------------
# ★  EDIT THIS SECTION to define your sweep  ★
# --------------------------------------------------------------------------
SWEEP_CONFIGS=(
    # ---- ablation presets (uncomment / add more as needed) ---------------
    "--ablation all_three   --desc abl_all_three"
    "--ablation bias_only   --desc abl_bias_only"
    "--ablation msc_only    --desc abl_msc_only"
    "--ablation leader_only --desc abl_leader_only"
    "--ablation bias_msc    --desc abl_bias_msc"
    "--ablation msc_leader  --desc abl_msc_leader"
    "--ablation bias_leader --desc abl_bias_leader"

    # ---- knn_k sweep (uncomment to enable) --------------------------------
    # "--ablation all_three --knn_k 20 --desc knn20"
    # "--ablation all_three --knn_k 30 --desc knn30"
    # "--ablation all_three --knn_k 40 --desc knn40"

    # ---- backbone_lr sweep ------------------------------------------------
    # "--ablation all_three --backbone_lr 1e-5 --desc lr1e5"
    # "--ablation all_three --backbone_lr 3e-5 --desc lr3e5"
    # "--ablation all_three --backbone_lr 5e-5 --desc lr5e5"

    # ---- leader_alpha sweep -----------------------------------------------
    # "--ablation all_three --leader_alpha 10 --leader_alpha_final 20 --desc alpha10"
    # "--ablation all_three --leader_alpha 20 --leader_alpha_final 40 --desc alpha20"
    # "--ablation all_three --leader_alpha 40 --leader_alpha_final 60 --desc alpha40"

    # ---- bias_strength × knn_k 2D grid ------------------------------------
    # "--ablation all_three --knn_k 20 --knn_bias_value 0.5 --desc k20_v05"
    # "--ablation all_three --knn_k 20 --knn_bias_value 1.0 --desc k20_v10"
    # "--ablation all_three --knn_k 30 --knn_bias_value 0.5 --desc k30_v05"
    # "--ablation all_three --knn_k 30 --knn_bias_value 1.0 --desc k30_v10"
)
# --------------------------------------------------------------------------
# Common settings shared by all runs (override with env vars before calling).
B="${B:-400}"
RESUME_CKPT="${RESUME_CKPT:-./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt}"
MAX_JOBS="${MAX_JOBS:-1}"      # how many runs in parallel; set = num GPUs for multi-GPU
DRY_RUN="${DRY_RUN:-0}"       # 1 = print commands only, do not execute
CUDA_LIST="${CUDA_LIST:-0}"   # comma-separated GPU IDs; assigned round-robin per run
# --------------------------------------------------------------------------

IFS=',' read -ra GPUS <<< "$CUDA_LIST"
NUM_GPUS="${#GPUS[@]}"

mkdir -p logs

declare -A PID_DESC=()
declare -A PID_GPU=()
declare -A PID_LOG=()
sweep_idx=0
active_jobs=0

# Block until fewer than MAX_JOBS child processes are alive.
wait_for_slot() {
    while (( active_jobs >= MAX_JOBS )); do
        for pid in "${!PID_DESC[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[sweep] DONE  run='${PID_DESC[$pid]}'  gpu=${PID_GPU[$pid]}  log=${PID_LOG[$pid]}"
                unset "PID_DESC[$pid]" "PID_GPU[$pid]" "PID_LOG[$pid]"
                (( active_jobs-- )) || true
            fi
        done
        (( active_jobs >= MAX_JOBS )) && sleep 5
    done
}

echo "============================================================"
echo " [sweep] total configs : ${#SWEEP_CONFIGS[@]}"
echo " [sweep] B             : $B"
echo " [sweep] MAX_JOBS      : $MAX_JOBS"
echo " [sweep] CUDA_LIST     : $CUDA_LIST"
echo " [sweep] DRY_RUN       : $DRY_RUN"
echo " [sweep] start         : $(date '+%F %T')"
echo "============================================================"

for flags in "${SWEEP_CONFIGS[@]}"; do
    wait_for_slot

    GPU_IDX=$(( sweep_idx % NUM_GPUS ))
    GPU="${GPUS[$GPU_IDX]}"

    # Extract --desc value for the log filename.
    desc=""
    prev=""
    for word in $flags; do
        [[ "$prev" == "--desc" ]] && { desc="$word"; break; }
        prev="$word"
    done
    [[ -z "$desc" ]] && desc="run${sweep_idx}"

    LOG_FILE="logs/sweep_${desc}.log"

    echo "------------------------------------------------------------"
    echo " [sweep] launching   : $desc"
    echo " [sweep] GPU         : $GPU"
    echo " [sweep] flags       : $flags"
    echo " [sweep] log         : $LOG_FILE"

    # Build the command.  eval is needed to expand embedded quotes/vars.
    CMD="CUDA_VISIBLE_DEVICES=${GPU} python finetune_phased.py \
        --total_finetune_epochs ${B} \
        --resume_checkpoint '${RESUME_CKPT}' \
        ${flags}"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry_run] $CMD"
    else
        eval "$CMD" > "$LOG_FILE" 2>&1 &
        pid=$!
        PID_DESC[$pid]="$desc"
        PID_GPU[$pid]="$GPU"
        PID_LOG[$pid]="$LOG_FILE"
        (( active_jobs++ )) || true
    fi

    (( sweep_idx++ )) || true
done

# Wait for all remaining children.
if [[ "$DRY_RUN" != "1" && ${#PID_DESC[@]} -gt 0 ]]; then
    echo "============================================================"
    echo " [sweep] waiting for ${#PID_DESC[@]} remaining jobs..."
    for pid in "${!PID_DESC[@]}"; do
        wait "$pid" || true
        echo "[sweep] DONE  run='${PID_DESC[$pid]}'  gpu=${PID_GPU[$pid]}  log=${PID_LOG[$pid]}"
    done
fi

echo "============================================================"
echo " [sweep] ALL DONE  $(date '+%F %T')"
echo ""
echo " Summary table:"
echo "   python scripts/show_sweep_results.py"
echo ""
echo " Validate a specific run's avg_aug_gap:"
echo "   bash scripts/validate_phased.sh result/<run_dir>"
echo "============================================================"
