#!/usr/bin/env bash
# Combination ablations (version B and leave-one-out).
#
# NOTE: M2/M3 variants load their finetuned checkpoint from CHECKPOINT_VERB
# (set by the user after running run_ablation_m3.sh or a dedicated finetune).
# If CHECKPOINT_VERB is unset, the baseline checkpoint is used and the M2
# bias is applied purely at inference-time (less effective but still runs).

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CHECKPOINT_VERB="${CHECKPOINT_VERB:-$CHECKPOINT_PATH}"

# Single-module baseline++ (re-using one representative config each)
run_eval "baseline"        ""
run_eval "plus_m1"         "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5"
run_eval "plus_m2"         "--distance_bias_enabled true --distance_bias_scale 1.0 --knn_bias_enabled true --knn_k 10 --knn_bias_value 0.5"
run_eval "plus_m4"         "--two_opt_enabled true --two_opt_target final_best --two_opt_max_iters 50"

# Two-module combos
run_eval "plus_m1_m2" "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 --distance_bias_enabled true --knn_bias_enabled true --knn_k 10"
run_eval "plus_m1_m4" "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 --two_opt_enabled true --two_opt_target topk_candidates --two_opt_topk 3"
run_eval "plus_m2_m4" "--distance_bias_enabled true --knn_bias_enabled true --knn_k 10 --two_opt_enabled true --two_opt_target final_best"

# Three-module combo
run_eval "plus_m1_m2_m4" "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 --distance_bias_enabled true --knn_bias_enabled true --knn_k 10 --two_opt_enabled true --two_opt_target topk_candidates --two_opt_topk 3"

# Full version B (requires CHECKPOINT_VERB if you trained with M3)
CHECKPOINT_PATH="$CHECKPOINT_VERB" run_eval "verB_full" \
    "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 \
     --distance_bias_enabled true --distance_bias_scale 1.0 \
     --knn_bias_enabled true --knn_k 10 --knn_bias_value 0.5 \
     --two_opt_enabled true --two_opt_target topk_candidates --two_opt_topk 3"

# Leave-one-out ablations against version B
CHECKPOINT_PATH="$CHECKPOINT_VERB" run_eval "verB_minus_m1" \
    "--distance_bias_enabled true --distance_bias_scale 1.0 \
     --knn_bias_enabled true --knn_k 10 --knn_bias_value 0.5 \
     --two_opt_enabled true --two_opt_target final_best"
CHECKPOINT_PATH="$CHECKPOINT_VERB" run_eval "verB_minus_m2" \
    "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 \
     --two_opt_enabled true --two_opt_target topk_candidates --two_opt_topk 3"
# verB_minus_m3 means "evaluate the same pipeline but with the baseline checkpoint"
run_eval "verB_minus_m3" \
    "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 \
     --distance_bias_enabled true --distance_bias_scale 1.0 \
     --knn_bias_enabled true --knn_k 10 --knn_bias_value 0.5 \
     --two_opt_enabled true --two_opt_target topk_candidates --two_opt_topk 3"
CHECKPOINT_PATH="$CHECKPOINT_VERB" run_eval "verB_minus_m4" \
    "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 \
     --distance_bias_enabled true --distance_bias_scale 1.0 \
     --knn_bias_enabled true --knn_k 10 --knn_bias_value 0.5"
