#!/usr/bin/env bash
# Run the baseline (all features off). Its output JSON is used as the
# reference for win-rate computation by every other ablation script.

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

run_eval "baseline" ""

# Copy the just-produced baseline.json as the canonical reference.
cp "$SUMMARY_DIR/baseline.json" "$BASELINE_REF_JSON"
echo "Baseline reference written to: $BASELINE_REF_JSON"
