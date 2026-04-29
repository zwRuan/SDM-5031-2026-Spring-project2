#!/usr/bin/env bash
# Shared environment preamble for all ablation scripts.
# Activates the `drl_tsp` conda env and cd's into TSP/POMO.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POMO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate drl_tsp
fi

cd "$POMO_DIR"

# Common paths
DATA_PATH="${DATA_PATH:-../data/val}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt}"
SUMMARY_DIR="${SUMMARY_DIR:-./results/ablation}"
BASELINE_REF_JSON="${BASELINE_REF_JSON:-$SUMMARY_DIR/baseline.json}"

mkdir -p "$SUMMARY_DIR"

# Helper: run test.py with a given run_name and output file.
# Usage: run_eval <run_name> <extra_flags_string>
run_eval() {
    local run_name="$1"
    local extra="$2"
    local out_json="$SUMMARY_DIR/${run_name}.json"
    echo ">>> Running $run_name"
    # shellcheck disable=SC2086
    python test.py \
        --data_path "$DATA_PATH" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --augmentation_enable true \
        --aug_factor 8 \
        --detailed_log false \
        --output_json "$out_json" \
        --summary_dir "$SUMMARY_DIR" \
        --baseline_ref_json "$BASELINE_REF_JSON" \
        --run_name "$run_name" \
        $extra
}
