#!/usr/bin/env bash
# M1 SGBS-lite reranking ablations.
#   M1-A: beam width sweep (2, 4, 8)
#   M1-B: entropy gate on/off
#   M1-C: per-aug vs global pool
#   M1-D: rerank_depth 3 / 5 / 10

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# M1-A
run_eval "m1_rerank_b2_d5" "--rerank_enabled true --rerank_beam_width 2 --rerank_depth 5"
run_eval "m1_rerank_b4_d5" "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5"
run_eval "m1_rerank_b8_d5" "--rerank_enabled true --rerank_beam_width 8 --rerank_depth 5"

# M1-B
run_eval "m1_gate_off" "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 --rerank_use_entropy_gate false"
run_eval "m1_gate_on"  "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 --rerank_use_entropy_gate true --rerank_entropy_threshold 1.0"

# M1-C
run_eval "m1_pool_within_aug" "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 --rerank_pool_across_augs false"
run_eval "m1_pool_global"     "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5 --rerank_pool_across_augs true"

# M1-D
run_eval "m1_depth_3"  "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 3"
run_eval "m1_depth_5"  "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 5"
run_eval "m1_depth_10" "--rerank_enabled true --rerank_beam_width 4 --rerank_depth 10"
