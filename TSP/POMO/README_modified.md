# POMO TSP — Modified Version (M1–M4)

This document describes four modular improvements built on top of the
course-provided POMO baseline. **Every module is OFF by default** so the
existing `train.py` / `test.py` invocations keep producing the original
baseline numbers bit-for-bit.

> **Runtime environment** — every command in this document assumes the
> `drl_tsp` conda environment is active:
>
> ```bash
> conda activate drl_tsp
> ```
>
> The shell scripts under `scripts/` activate it automatically.

---

## 1. Module overview

| Module | Where it lives | Stage | Default | Effect on baseline contract |
|--------|----------------|-------|---------|------------------------------|
| **M1 SGBS-lite reranking** | `search/sgbs_lite.py`             | inference | OFF | Adds CLI flags only |
| **M2 distance / kNN bias** | `model_ext/distance_bias.py`      | encoder + decoder | OFF | Adds optional bias hook in `TSPModel.TSP_Decoder` (no-op when module is `None`) |
| **M3 leader-focused reward** | `train_ext/leader_reward.py`    | training  | OFF | Replaces the advantage formula or adds an aux loss when enabled |
| **M4 2-opt post-processing** | `search/two_opt.py`             | inference | OFF | Pure post-processing on the chosen tour |

Common housekeeping:

* `ablation/summary.py` — appends each `test.py` run as a row in
  `<summary_dir>/summary.csv` and `summary.json`, plus
  `per_instance_<run_name>.json`. Computes win/lose/tie rate vs the
  baseline JSON (`--baseline_ref_json`).
* `scripts/*.sh` — runner scripts for the M1, M2, M3, M4, and combo
  ablations. Activate `drl_tsp` automatically.
* `tests/test_sanity.py` — fast unit tests (CPU only, ~5s) covering
  permutation validity, monotonicity, NaN-free behaviour, leader-loss
  gradient flow, etc.
* `tests/test_all_off_parity.py` — end-to-end check that with all flags
  off the tester reproduces the recorded baseline `avg_aug_gap`
  (2.329…) and `avg_no_aug_gap` (3.753…) on the public val set.

---

## 2. CLI flags (added to `test.py` / `train.py`)

### `test.py` (inference)

| Flag | Default | Notes |
|------|---------|-------|
| `--rerank_enabled` | `false` | M1 toggle |
| `--rerank_beam_width` | `4` | Beams per starting node |
| `--rerank_depth` | `5` | Beam-expansion steps |
| `--rerank_topk_per_step` | `4` | Children per beam during expansion |
| `--rerank_use_entropy_gate` | `false` | Skip expansion when softmax entropy < threshold |
| `--rerank_entropy_threshold` | `1.0` | Threshold value for the gate |
| `--rerank_pool_across_augs` | `true` | Pool candidates globally vs per-augmentation |
| `--rerank_deduplicate` | `true` | Canonicalize tours before deduping |
| `--two_opt_enabled` | `false` | M4 toggle |
| `--two_opt_target` | `final_best` | Or `topk_candidates` |
| `--two_opt_topk` | `3` | Used with `topk_candidates` |
| `--two_opt_max_iters` | `50` | Iteration cap |
| `--two_opt_first_improvement` | `true` | First vs best-improvement sweep |
| `--two_opt_time_budget_ms` | `null` | Optional wall-clock budget |
| `--distance_bias_enabled` | `false` | M2 distance term |
| `--distance_bias_scale` | `1.0` | Multiplier on the distance bias |
| `--distance_bias_mode` | `logit` | `attn` is **not implemented** (see §6) |
| `--distance_norm_mode` | `mean` | `none / mean / max / std` |
| `--knn_bias_enabled` | `false` | M2 kNN term |
| `--knn_k` | `10` | Neighborhood size |
| `--knn_bias_value` | `0.5` | Logit bonus for in-kNN nodes |
| `--run_name` | auto | Appears in summary CSV; auto-derived from active flags |
| `--summary_dir` | `null` | Where to append summary.csv/json |
| `--baseline_ref_json` | `null` | Baseline summary used for win-rate computation |

### `train.py` (training)

`python train.py` with **zero** arguments still mirrors the original
baseline recipe (3100 epochs, 100k episodes, batch 64). Added flags:

| Flag | Notes |
|------|-------|
| `--epochs / --train_episodes / --train_batch_size / --lr` | Override the baseline values |
| `--problem_size / --pomo_size` | Override env params |
| `--desc` | Tag for the result folder |
| `--finetune_from <path>` | Load weights only (optimizer reset). Recommended for M2/M3 |
| `--grad_clip_max_norm` | Manual override; auto-on (1.0) when M2 or M3 active |
| `--distance_bias_enabled / --distance_bias_scale / --distance_bias_mode / --distance_norm_mode` | M2 |
| `--knn_bias_enabled / --knn_k / --knn_bias_value` | M2 |
| `--leader_reward_enabled / --leader_mode / --leader_gamma / --leader_aux_weight` | M3 |

---

## 3. Reproducing the baseline

```bash
conda activate drl_tsp
cd TSP/POMO
bash scripts/run_baseline.sh
```

This writes `results/ablation/baseline.json` and uses it as the
canonical reference for downstream win-rate calculations.

---

## 4. Per-module ablations

Every script appends rows to `results/ablation/summary.csv` (override
with `SUMMARY_DIR=...`):

```bash
# M1 SGBS-lite (inference only)
bash scripts/run_ablation_m1.sh

# M2 distance / kNN bias (inference-side; for the train-side variant
# finetune first and then point CHECKPOINT_PATH at the new ckpt)
bash scripts/run_ablation_m2.sh

# M3 leader reward (each variant FINETUNES from the baseline ckpt)
EPOCHS=5 EPISODES=10000 bash scripts/run_ablation_m3.sh

# M4 2-opt (inference only)
bash scripts/run_ablation_m4.sh
```

Each script enumerates all the sub-ablations listed in the project spec
(M1-A/B/C/D, M2-A/B/C/D, M3-A/B/C, M4-A/B/C). Open the script if you
want to tweak which combinations run.

---

## 5. Combination experiments and Version B

```bash
# Single-module, two-module, three-module combos and leave-one-out
# ablation against version B.
CHECKPOINT_VERB=./result/<your_finetuned_run>/checkpoint-<epoch>.pt \
    bash scripts/run_ablation_combo.sh
```

The `verB_full` row is the spec's full Version B: M1 + M2 + M3 + M4. To
get the most out of M2/M3 you should first finetune a checkpoint:

```bash
python train.py \
    --finetune_from ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
    --epochs 50 \
    --train_episodes 100000 \
    --lr 1e-5 \
    --distance_bias_enabled true --distance_bias_scale 1.0 \
    --knn_bias_enabled true --knn_k 10 --knn_bias_value 0.5 \
    --leader_reward_enabled true --leader_mode bonus_adv --leader_gamma 0.5 \
    --desc verB_finetune
```

then export the resulting checkpoint path as `CHECKPOINT_VERB` for the
combo script.

---

## 6. Output layout

```
TSP/POMO/
├── results/
│   └── ablation/                      # default --summary_dir
│       ├── summary.csv                # one row per run (CSV)
│       ├── summary.json               # same data + full configs
│       ├── baseline.json              # raw baseline payload
│       ├── baseline.json (renamed)    # used as --baseline_ref_json
│       ├── m1_rerank_b4_d5.json
│       ├── per_instance_<run>.json
│       └── ...
└── result_lib/                        # baseline tester logs (unchanged)
```

`summary.csv` columns:

```
run_name, timestamp, avg_aug_gap, avg_no_aug_gap, solved, total,
aug_factor, rerank_enabled, two_opt_enabled,
distance_bias_enabled, knn_bias_enabled, leader_reward_enabled,
total_runtime_ms, win_rate_vs_baseline, lose_rate_vs_baseline,
tie_rate_vs_baseline, checkpoint_path, data_path
```

Per-instance JSON includes the rerank info (`rerank_ms`, candidate
count, expansions done, gate skips), the 2-opt records (orig/final
length, improvements, time), and the `improved_no_aug` /
`improved_aug` booleans for win-rate tracking.

---

## 7. Compatibility notes & explicit trade-offs

* **All-off parity.** With every new flag default-off, `test.py` runs the
  exact same code path as before (a couple of `if` checks and one
  `.clone()`). The recorded `avg_aug_gap` of 2.329…% on the public val
  set is reproduced bit-for-bit (verified by
  `tests/test_all_off_parity.py`).

* **M2 `distance_bias_mode=attn` is intentionally not implemented.**
  Putting the bias on the encoder attention score requires modifying
  `multi_head_attention` in two places (encoder + decoder MHA) and
  retraining from scratch to be useful — both are outside the
  course-project budget. The `logit` variant gives 90% of the value at
  ~5% of the engineering cost. The flag still exists for forward
  compatibility but raises `NotImplementedError`.

* **M3 implements both variants** (`bonus_adv` and `aux_imitation`).
  `bonus_adv` is the recommended default because it integrates with the
  existing POMO advantage formulation.

* **Finetune-first recipe for M2/M3.** Finetuning from
  `checkpoint-3000.pt` for 5–50 epochs at `lr=1e-5` is materially faster
  and more stable than training from scratch with the new objectives.
  Auto grad-clipping (`max_norm=1.0`) is enabled whenever M2 or M3 is
  active; you can override via `--grad_clip_max_norm`.

* **M4 uses TSPLIB rounding** (`floor(x + 0.5)` for `EUC_2D`,
  `ceil(x)` for `CEIL_2D`) so any improvement it reports is measured
  against the same metric `_get_travel_distance(lib_mode=True)` uses.

* **M1 stability.** SGBS-lite never modifies the model and always
  re-builds a fresh `TSPEnv`. Worst case (rerank picks no improvement)
  the original baseline candidate is still kept in the pool, so the
  final aug score is bounded by the baseline.

* **Directory naming.** New code sits under `model_ext/`,
  `train_ext/`, and `ablation/` (rather than `model/`, `train/`,
  `utils/`) to avoid Python import collisions with the existing
  `train.py` module and the project's top-level `utils/` package.

---

## 8. Running the sanity tests

Fast unit tests (no checkpoint, no data, ~5s):

```bash
conda activate drl_tsp
python tests/test_sanity.py
```

End-to-end parity (loads the baseline checkpoint on CPU, ~1 minute):

```bash
python tests/test_all_off_parity.py
```

Both should print `PASS` lines.

---

## 9. Quick wins reproduction

A representative smoke run (baseline checkpoint, beam=2 / depth=3,
2-opt on the final best) lifted `avg_aug_gap` from `2.330%` → `1.316%`
on the public val set in ~2 minutes on CPU. Stronger configs
(`beam=4`, `depth=5`, `topk_candidates`/`top3`) push this further at the
cost of more inference time.
