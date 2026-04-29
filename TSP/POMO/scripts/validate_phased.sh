#!/usr/bin/env bash
# Run the standard test.py (avg_aug_gap on TSPLIB val set) on each phase's
# best checkpoint produced by finetune_phased.py. The test pipeline must
# match what was used during training: bias flags ON if you trained with
# bias, otherwise leave them OFF (test.py default).
#
# Usage:
#   bash scripts/validate_phased.sh <result_dir>                   # auto-detects bias flags from finetune_phased_config.json
#   RUN_DIR=result/2026..._phased_B400_all_three bash scripts/validate_phased.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

RUN_DIR="${1:-${RUN_DIR:-}}"
if [[ -z "${RUN_DIR}" ]]; then
    echo "ERROR: pass result dir as arg, e.g. bash scripts/validate_phased.sh result/<timestamp>_..."
    exit 2
fi
RUN_DIR="$(readlink -f "$RUN_DIR")"
echo "[validate] run_dir=$RUN_DIR"

CFG_JSON="$RUN_DIR/finetune_phased_config.json"
EXTRA=""
if [[ -f "$CFG_JSON" ]]; then
    # Auto-extract bias config matching the training side.
    EXTRA="$(python - "$CFG_JSON" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
flags = []
if cfg.get("enable_bias"):
    flags += [
        "--distance_bias_enabled true",
        f"--distance_bias_scale {float(cfg.get('distance_bias_scale', 1.0))}",
        f"--distance_norm_mode {cfg.get('distance_norm_mode', 'mean')}",
    ]
    if cfg.get("knn_bias_enabled"):
        flags += [
            "--knn_bias_enabled true",
            f"--knn_k {int(cfg.get('knn_k', 30))}",
            f"--knn_bias_value {float(cfg.get('knn_bias_value', 1.0))}",
        ]
print(" ".join(flags))
PY
)"
fi
echo "[validate] inference flags: $EXTRA"

for CKPT in "$RUN_DIR"/checkpoint-phase_*_best.pt "$RUN_DIR"/checkpoint-latest.pt; do
    [[ -f "$CKPT" ]] || continue
    NAME="$(basename "$CKPT" .pt)"
    OUT_JSON="$RUN_DIR/eval_${NAME}.json"
    echo "------------------------------------------------------------------"
    echo "[validate] testing $CKPT"
    # shellcheck disable=SC2086
    python test.py \
        --data_path "$DATA_PATH" \
        --checkpoint_path "$CKPT" \
        --augmentation_enable true \
        --aug_factor 8 \
        --detailed_log false \
        --output_json "$OUT_JSON" \
        --run_name "phased_${NAME}" \
        $EXTRA
done

echo "------------------------------------------------------------------"
echo "[validate] summary:"
for J in "$RUN_DIR"/eval_*.json; do
    [[ -f "$J" ]] || continue
    python - "$J" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))
print(f"  {os.path.basename(sys.argv[1])}: avg_aug_gap={d.get('avg_aug_gap'):.4f} avg_no_aug_gap={d.get('avg_no_aug_gap'):.4f}")
PY
done
