# scripts/ — runner scripts overview

All shell scripts in this directory are thin wrappers around
`train.py` / `test.py` / `finetune_phased.py` that activate the `drl_tsp`
conda environment and pass the right flags for one specific experiment
type. They `cd` into `TSP/POMO/` via `_env.sh`, so you can launch them
from anywhere.

## Quick reference

| Script | What it does | Needs training? |
|---|---|---|
| `run_finetune_phased.sh` | **Phased fine-tune** (bias -> MSC -> leader) — primary training entry | Yes |
| `run_sweep.sh` | **Grid sweep** — parallel launcher for multiple hyperparameter configs | Yes |
| `show_sweep_results.py` | **Sweep viewer** — aggregates and sorts all run results into a table | No |
| `validate_phased.sh` | Run `test.py` on each phase checkpoint of a phased run | No |
| `run_baseline.sh` | Baseline POMO + aug8; writes the canonical `baseline.json` reference | No |
| `run_ablation_m1.sh` | SGBS-lite rerank ablations (A/B/C/D) | No |
| `run_ablation_m2.sh` | Distance + kNN bias ablations (inference-side only) | No |
| `run_ablation_m3.sh` | Leader reward inference-side ablations | No |
| `run_ablation_m4.sh` | 2-opt ablations | No |
| `run_ablation_combo.sh` | Two/three/four-module combos + Version B | Optional |
| `normalize_tsplib.py` | Normalize TSPLIB instances to the unit square (utility) | (utility) |
| `_env.sh` | Shared preamble: activates conda, sets paths, defines `run_eval` | (lib) |

## Phased fine-tune — `run_finetune_phased.sh`

Resumes from the baseline 3000-epoch checkpoint and runs three phases in
the spec order: **Phase 1 bias adapter (15% of B) -> Phase 2 MSC main
adaptation (50%) -> Phase 3 leader-focused reward (35%)**.

Quick-start:
```bash
cd /home/zhaorj/Code/SDM-5031-2026-Spring/TSP/POMO
bash scripts/run_finetune_phased.sh                                # default B=400
B=200 bash scripts/run_finetune_phased.sh                          # smaller budget
ABLATION=bias_only bash scripts/run_finetune_phased.sh --desc abl_bias_only
ABLATION=msc_leader bash scripts/run_finetune_phased.sh --desc abl_msc_leader
```

Available `ABLATION` presets (also accepted as `--ablation`): `all_three`
(default), `bias_only`, `msc_only`, `leader_only`, `bias_msc`,
`msc_leader`, `bias_leader`.

Common environment variables:

| Var          | Default                                               | Maps to                       |
|--------------|-------------------------------------------------------|-------------------------------|
| `B`          | `400`                                                 | `--total_finetune_epochs`     |
| `ABLATION`   | `all_three`                                           | `--ablation`                  |
| `RESUME_CKPT`| `./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt` | `--resume_checkpoint`   |
| `DESC`       | `phased`                                              | `--desc` (result dir tag)     |
| `CONFIG`     | `./configs/finetune_phased.json`                      | `--config`                    |

Any extra args are forwarded to `finetune_phased.py`, so any flag from
the spec table can be overridden:
```bash
bash scripts/run_finetune_phased.sh --knn_k 40 --leader_alpha 10
```

Result directory contains:
- `log.txt`                            — full per-epoch trace with phase tags
- `checkpoint-latest.pt`               — most recent model
- `checkpoint-phase_<N>_best.pt`       — best train_score within each phase
- `checkpoint-<E>.pt`                  — periodic snapshots (every `model_save_interval`)
- `finetune_phased_config.json`        — resolved config for reproducibility

Evaluate avg_aug_gap on the validation set after training:
```bash
bash scripts/validate_phased.sh result/<timestamp>_phased_phased_B400_all_three
```
This auto-runs `test.py` against every phase checkpoint and writes one
`eval_<name>.json` per checkpoint, picking up matching bias inference
flags from the saved `finetune_phased_config.json`.

## Hyperparameter sweep — `run_sweep.sh` + `show_sweep_results.py`

Define which runs to launch by editing the `SWEEP_CONFIGS` array inside
`run_sweep.sh` (one string of CLI flags per run), then:

```bash
cd TSP/POMO

# 1. Launch the sweep (serial, single GPU):
bash scripts/run_sweep.sh

# 2. Launch in parallel across two GPUs:
MAX_JOBS=2 CUDA_LIST="0,1" bash scripts/run_sweep.sh

# 3. Dry-run — print commands without executing:
DRY_RUN=1 bash scripts/run_sweep.sh

# 4. View results table after runs finish:
python scripts/show_sweep_results.py

# 5. Sort by a different column or filter:
python scripts/show_sweep_results.py --sort p3_best --filter "ablation=all_three"

# 6. Export to CSV for plotting:
python scripts/show_sweep_results.py --csv sweep_summary.csv

# 7. Find runs that still need avg_aug_gap measured:
python scripts/show_sweep_results.py --validate_missing
```

The result table columns are:

| Column | Source | Meaning |
|---|---|---|
| `run_dir` | directory name | timestamp + desc tag |
| `ablation` | config JSON | which modules were enabled |
| `B`, `knn_k`, `knn_v`, `backbone_lr`, `phase3_lr`, `alpha`, `alpha_f` | config JSON | key hyperparams |
| `bias` / `msc` / `ldr` | config JSON | Y/N module switches |
| `p1_best`, `p2_best`, `p3_best` | `log.txt` | best train score per phase (proxy metric) |
| `aug_gap` | `eval_*.json` | **avg_aug_gap** — official metric (needs `validate_phased.sh`) |
| `no_aug_gap` | `eval_*.json` | avg_no_aug_gap |

Rows with no `aug_gap` yet (training done, validation not run) still show
`p3_best` as a proxy. Use `--validate_missing` to get the exact commands
needed to fill them in.

## Inference-side ablations

The `run_ablation_m{1,2,3,4}.sh` and `run_ablation_combo.sh` scripts run
`test.py` on the **baseline** checkpoint with various inference-time
flags. They are independent of training and are useful for ablations
that don't need to retrain (e.g. SGBS-lite rerank, 2-opt, inference-only
distance bias).

```bash
cd TSP/POMO
bash scripts/run_baseline.sh            # generates baseline.json
bash scripts/run_ablation_m1.sh         # inference-only rerank ablations
bash scripts/run_ablation_m2.sh         # inference-only bias ablations
```

To run inference ablations against a phased fine-tune checkpoint
instead of the baseline, set `CHECKPOINT_PATH`:

```bash
CHECKPOINT_PATH=result/<phased_run>/checkpoint-phase_3_leader_best.pt \
    bash scripts/run_ablation_m1.sh
```
