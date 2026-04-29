#!/usr/bin/env bash
# M2 distance / kNN bias ablations (INFERENCE-SIDE ONLY).
#
# These scripts reuse the baseline checkpoint and apply the bias purely at
# inference time. For a proper train-side M2 experiment, first finetune a
# model with `train.py --finetune_from ... --distance_bias_enabled true`,
# then point CHECKPOINT_PATH to that new checkpoint before re-running.
#
#   M2-A: off / dist / knn / dist+knn
#   M2-B: mode = logit only (attn intentionally not implemented; see README_modified)
#   M2-C: knn_k sweep
#   M2-D: distance_bias_scale sweep

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# M2-A
run_eval "m2_off"       ""
run_eval "m2_dist"      "--distance_bias_enabled true --distance_bias_scale 1.0"
run_eval "m2_knn"       "--knn_bias_enabled true --knn_k 10 --knn_bias_value 0.5"
run_eval "m2_dist_knn"  "--distance_bias_enabled true --distance_bias_scale 1.0 --knn_bias_enabled true --knn_k 10 --knn_bias_value 0.5"

# M2-B (logit only; attn intentionally unimplemented)
run_eval "m2_mode_logit" "--distance_bias_enabled true --distance_bias_mode logit --distance_bias_scale 1.0"

# M2-C
run_eval "m2_knn_k5"  "--knn_bias_enabled true --knn_k 5"
run_eval "m2_knn_k10" "--knn_bias_enabled true --knn_k 10"
run_eval "m2_knn_k20" "--knn_bias_enabled true --knn_k 20"

# M2-D
run_eval "m2_dist_scale_small"  "--distance_bias_enabled true --distance_bias_scale 0.25"
run_eval "m2_dist_scale_medium" "--distance_bias_enabled true --distance_bias_scale 1.0"
run_eval "m2_dist_scale_large"  "--distance_bias_enabled true --distance_bias_scale 4.0"
