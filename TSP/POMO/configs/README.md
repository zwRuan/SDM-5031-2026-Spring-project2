# Named experiment configurations

The shell scripts in `../scripts/` are the canonical way to launch
experiments — they assemble the CLI flags directly. The JSON snapshots in
this directory document each named run so you can inspect the exact flag
set without reading shell.

| Name                          | Purpose                                                              | Needs training? |
|-------------------------------|----------------------------------------------------------------------|-----------------|
| `baseline.json`               | Baseline POMO + aug8. Win-rate reference.                            | No              |
| `finetune_phased.json`        | **Phased fine-tune** (bias -> MSC -> leader) starting from the baseline 3000-epoch checkpoint. Default 400-epoch budget split as 15% / 50% / 35%. | Yes |

`baseline.json` is documentation-only (the baseline test is launched via
`scripts/run_baseline.sh`). `finetune_phased.json` is consumed by
`finetune_phased.py --config configs/finetune_phased.json` and contains
every spec hyperparameter as well as a `param_name_mapping` block that
records how spec semantic names map onto the existing implementation
keys (`distance_bias_cfg`, `msc_cfg`, `leader_cfg`).

To launch the default 400-epoch recipe:

```bash
cd TSP/POMO
bash scripts/run_finetune_phased.sh                    # B=400 (default)
B=200 bash scripts/run_finetune_phased.sh              # smaller budget
ABLATION=bias_only bash scripts/run_finetune_phased.sh # ablation preset
```
