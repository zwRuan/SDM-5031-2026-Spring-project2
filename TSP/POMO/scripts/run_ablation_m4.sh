#!/usr/bin/env bash
# M4 2-opt post-processing ablations.
#   M4-A: off / top1 / top3 / top5
#   M4-B: first_improvement True vs False
#   M4-C: max_iters sweep

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# M4-A
run_eval "m4_off"   ""
run_eval "m4_top1"  "--two_opt_enabled true --two_opt_target final_best"
run_eval "m4_top3"  "--two_opt_enabled true --two_opt_target topk_candidates --two_opt_topk 3 --rerank_enabled true --rerank_beam_width 4 --rerank_depth 5"
run_eval "m4_top5"  "--two_opt_enabled true --two_opt_target topk_candidates --two_opt_topk 5 --rerank_enabled true --rerank_beam_width 4 --rerank_depth 5"

# M4-B
run_eval "m4_first_improve" "--two_opt_enabled true --two_opt_first_improvement true"
run_eval "m4_best_improve"  "--two_opt_enabled true --two_opt_first_improvement false"

# M4-C
run_eval "m4_iters_10"  "--two_opt_enabled true --two_opt_max_iters 10"
run_eval "m4_iters_50"  "--two_opt_enabled true --two_opt_max_iters 50"
run_eval "m4_iters_200" "--two_opt_enabled true --two_opt_max_iters 200"
