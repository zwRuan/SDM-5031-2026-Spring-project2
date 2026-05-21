# Phased Fine-Tuning of POMO with Three Inductive-Bias Modules + Size Curriculum

> 本项目对 SDM-5031 课程提供的 POMO baseline 进行 fine-tune，引入三个独立模块、一套三阶段训练框架，以及一个 **Phase 4 size curriculum**（多 N 训练）。在 validation 集上将 `avg_aug_gap` 从 baseline 的 **2.330%** 降低到 **0.624%**（相对降幅 **73.2%**）。
>
> **最终提交模型**：`TSP/POMO/result/20260512_012500_phased_PHASE4_A_from_alpha10_20_280_B400_all_three/checkpoint-3580.pt`

## 1. 改动总览

### 1.1 新增模块（三个独立创新点）

| 模块 | 作用层 | 类型 | 位置 |
|---|---|---|---|
| **Distance/kNN Logit Bias** | Forward path（decoder logit）| 归纳偏置（无可学参数）| `TSP/POMO/model_ext/distance_bias.py` |
| **Mixed Structured Curriculu (MSC)** | 数据生成 | 训练分布混合（uniform / clustered / gaussian）| `TSP/TSProblemDef.py`（generators 部分）|
| **Leader-Focused Reward** | Loss 函数 | 优化目标（强化 best-of-pomo 信号）| `TSP/POMO/train_ext/leader_reward.py` |

### 1.2 新增训练框架

**Phased Fine-Tune（三阶段 + Phase 4 size curriculum）**：[TSP/POMO/finetune_phased.py](TSP/POMO/finetune_phased.py)

| Phase | epochs | 主任务 | learning rate |
|---|---|---|---|
| Phase 1 (`1_bias`) | 60 (15%) | bias 模块线性 warm-up | 1e-5 |
| Phase 2 (`2_msc`) | 200 (50%) | MSC 分布主适应 | 1e-5 |
| Phase 3 (`3_leader`) | 140–280 (35–70%) | leader-focused reward + lr decay | 5e-5 → 5e-6 |
| **Phase 4 (size curriculum)** | 150（在 Phase 3 best ckpt 上继续训练）| per-batch 从 $N\in\{100,150,200,250\}$（权重 1:3:3:2）抽样；leader 冻结；MSC + bias 继承 | 1e-5 → 1e-6 |

Phase 4 通过 `--n_values` / `--n_weights` CLI flag 启用；不传则退化为标准三阶段训练。

配置文件：[TSP/POMO/configs/finetune_phased.json](TSP/POMO/configs/finetune_phased.json)

### 1.3 关键工程改动

| 改动 | 原因 | 涉及文件 |
|---|---|---|
| **Checkpoint 自带 `bias_cfg`** | 课程组用标准 `test.py` 命令评测时不传额外 flag，bias 模块需自动从 ckpt 激活 | `TSPTrainer.py`, `TSPTester_LIB.py` |
| **`--phase_epochs` CLI 覆盖** | 支持只跑 phase 3（从 phase 2 ckpt resume），用于快速 hyperparameter sweep | `finetune_phased.py` |
| **`control` ablation preset** | 全模块关闭的 fine-tune，作为 ablation 表的 zero-row | `finetune_phased.py` |


### 1.4 新增脚本

| 脚本 | 用途 |
|---|---|
| [TSP/POMO/scripts/run_sweep.sh](TSP/POMO/scripts/run_sweep.sh) | 多任务 GPU 共享 sweep 启动器 |
| [TSP/POMO/scripts/validate_phased.sh](TSP/POMO/scripts/validate_phased.sh) | 在 val 集上自动测试一个 result 目录的所有 phase ckpt |
| [TSP/POMO/scripts/show_sweep_results.py](TSP/POMO/scripts/show_sweep_results.py) | 把多个 run 的结果汇总成 CSV |

---

## 2. 训练命令

### 2.1 默认配置（B=400, all_three）

复现本项目 winner 模型（`winner_k20`）：

```bash
cd TSP/POMO
python finetune_phased.py \
    --config ./configs/finetune_phased.json \
    --total_finetune_epochs 400 \
    --resume_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
    --ablation all_three --knn_k 20 \
    --desc winner_k20
```

输出目录：`result/<timestamp>_phased_winner_k20_B400_all_three/`，包含：
- `checkpoint-phase_3_leader_best.pt` — 推荐用于评测的 winner ckpt
- `checkpoint-{epoch}.pt` — periodic checkpoints
- `log.txt`, `finetune_phased_config.json` — 完整训练日志和参数快照

### 2.2 Ablation 配置

```bash
# control: 全模块关闭（用于排除"多训 400 epoch"混淆变量）
python finetune_phased.py --config ./configs/finetune_phased.json \
    --total_finetune_epochs 400 \
    --resume_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
    --ablation control --desc abl_control

# 单模块 ablation
python finetune_phased.py --config ./configs/finetune_phased.json \
    --total_finetune_epochs 400 \
    --resume_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
    --ablation bias_only --knn_k 20 --desc abl_bias_only
# 同理可用 --ablation msc_only / leader_only
```

可选 `--ablation` 取值：`all_three / bias_only / msc_only / leader_only / bias_msc / msc_leader / bias_leader / control`。

### 2.3 超参 sweep（仅 phase 3，从 phase 2 ckpt resume）

跳过 phase 1+2，只跑 phase 3，用于快速 leader_alpha / phase 3 length 探索：

```bash
# 例: leader_alpha=10→20 + 280 epoch（phase 3 加倍）
CKPT_P2=./result/<winner_k20 dir>/checkpoint-3260.pt
python finetune_phased.py --config ./configs/finetune_phased.json \
    --phase_epochs 0,0,280 \
    --resume_checkpoint "$CKPT_P2" \
    --ablation all_three --knn_k 20 \
    --leader_alpha 10 --leader_alpha_final 20 \
    --desc tune_alpha10_20_280
```

### 2.4 Phase 4：在 Phase 3 best ckpt 上加 size curriculum（⭐ 最终模型）

从已经训好的 `alpha10_20_280` ckpt 起步，多 N 训练 150 epoch；leader α 冻结（rampup=0），learning rate 线性降到 1/10：

```bash
cd TSP/POMO
python finetune_phased.py \
    --config ./configs/finetune_phased.json \
    --resume_checkpoint ./result/best_ckpt_1/checkpoint-alpha10_20_280.pt \
    --continue_epoch_counter true \
    --phase_epochs 0,0,150 \
    --phase3_snapshot_epochs 45,90,120,140 \
    --train_episodes 50000 --train_batch_size 48 \
    --phase3_lr 1e-5 --phase3_final_lr 1e-6 \
    --knn_k 20 \
    --leader_alpha 20 --leader_alpha_final 20 --leader_rampup_portion 0 \
    --n_values 100,150,200,250 --n_weights 1,3,3,2 \
    --model_save_interval 20 \
    --ablation all_three \
    --desc PHASE4_A_from_alpha10_20_280
```

要点：
- `--continue_epoch_counter true`：epoch 计数从 anchor 的 3400 继续递增，便于按全局 epoch 取 ckpt（最终选的是 `checkpoint-3580.pt`，即 in-phase epoch 42）。
- `--knn_k 20`：**必须显式传**，否则会用 DEFAULT_HPARAMS 的 30，与 anchor 的 `bias_cfg` 不一致。
- `--n_values / --n_weights`：N 在 batch 之间变化（per-batch 固定单 N，因为 tensor shape 在 batch 内必须一致）。权重 1:3:3:2 故意降低 N=100 的比例，因为 anchor 已在 N=100 上过训练。
- `--train_batch_size 48`：在 48GB VRAM 上跑 N=250 的上限（batch=64 在 N=250 会 OOM）。
- 单卡 A100-48G 上耗时约 18 小时（150 epoch × 50k episodes × ~7 min/epoch）。

输出目录：`result/<timestamp>_phased_PHASE4_A_from_alpha10_20_280_B400_all_three/`

### 2.5 多任务并行（在单 GPU 上）

在 24GB 4090 上可同时跑两个任务：

```bash
# Task 1 后台启动
CUDA_VISIBLE_DEVICES=0 nohup python finetune_phased.py \
    --config ./configs/finetune_phased.json --total_finetune_epochs 400 \
    --resume_checkpoint .../checkpoint-3000.pt \
    --ablation all_three --knn_k 20 --desc run1 > logs/run1.log 2>&1 &

sleep 15  # 错开启动，避免初始化撞 OOM

# Task 2 后台启动
CUDA_VISIBLE_DEVICES=0 nohup python finetune_phased.py \
    --config ./configs/finetune_phased.json --total_finetune_epochs 400 \
    --resume_checkpoint .../checkpoint-3000.pt \
    --ablation bias_only --knn_k 20 --desc run2 > logs/run2.log 2>&1 &
```

每个任务约占 5-6 GB 显存，dual 共享 GPU-Util ~97% 时每个 epoch 约 6.8 分钟（vs solo ~5 min）。

---

## 3. 测试命令

### 3.1 ⭐ 课程组官方评测命令（提交时使用）

直接使用项目原始 README 中规定的标准接口。**Bias 配置已嵌入 ckpt，无需额外 flag**：

```bash
cd TSP/POMO
python test.py \
    --data_path /path/to/hidden_test_set \
    --checkpoint_path /path/to/checkpoint.pt \
    --use_cuda true \
    --cuda_device_num 0 \
    --augmentation_enable true \
    --aug_factor 8 \
    --detailed_log false \
    --output_json /path/to/eval_result.json
```

`--checkpoint_path` 使用最终提交模型 `./result/20260512_012500_phased_PHASE4_A_from_alpha10_20_280_B400_all_three/checkpoint-3580.pt`。`bias_cfg` 会从 ckpt 自动加载并 attach 到模型，与训练时配置完全一致。

### 3.2 在 validation 集上评测

```bash
cd TSP/POMO
python test.py \
    --data_path ../data/val \
    --checkpoint_path ./result/<run_dir>/checkpoint-phase_3_leader_best.pt \
    --use_cuda true --cuda_device_num 0 \
    --augmentation_enable true --aug_factor 8 \
    --detailed_log false \
    --output_json /tmp/eval.json
```

或用脚本一次性测一个 run 目录里所有 phase 的 best ckpt：

```bash
bash scripts/validate_phased.sh result/<run_dir>
```

## 4. 主要实验结果

### 4.1 Validation 集（10 个实例，N=100-300）

| 配置 | **avg_aug_gap** | Δ vs baseline |
|---|---|---|
| baseline (3000 ep) | 2.330% | — |
| control (3000+400 ep, all off) | 1.962% | −0.368% |
| winner_k20 (k=20, α=20→40, 140 ep) | 1.122% | −1.208% |
| alpha=10→20 (140 ep) | 1.181% | −1.149% |
| alpha=40→60 (140 ep) | 1.184% | −1.146% |
| p3double (α=20→40, 280 ep) | 1.117% | −1.213% |
| alpha10\_20\_280 (α=10→20, 280 ep, **Phase 4 anchor**) | 1.170% | −1.160% |
| ⭐ **Phase 4 winner: `checkpoint-3580.pt`** | **0.624%** | **−1.706%** (**−73.2%** rel) |

Phase 4 size curriculum 同时改善 in-distribution 和 OOD 桶：
- N∈[100,200]（8 实例）：0.347 → 0.303
- N∈[201,300]（2 实例）：4.465 → 1.906（相对降 −57%）

---

## 5. 项目结构

```
SDM-5031-2026-Spring/
├── README.md                         # 本文件
├── README_bak.md                     # 课程原始 README（保留）
├── slides/
│   └── sdm5031_project_pre_new.tex   # 答辩 slides
├── report/
│   └── sdm5031_project_report.tex    # 最终报告
├── TSP/
│   ├── TSProblemDef.py               # +MSC 三个 generator
│   └── POMO/
│       ├── finetune_phased.py        # 四阶段 fine-tune 主入口（含 Phase 4 size curriculum）
│       ├── test.py                   # 标准评测入口（保留原接口）
│       ├── TSPTrainer.py             # +bias_cfg 嵌入 ckpt; +per-batch n_sampler
│       ├── TSPTester_LIB.py          # +bias_cfg 自动 attach
│       ├── TSPModel.py               # +distance_bias_module hook
│       ├── model_ext/
│       │   └── distance_bias.py
│       ├── train_ext/
│       │   └── leader_reward.py
│       ├── configs/
│       │   └── finetune_phased.json  # 默认配置
│       ├── scripts/
│       │   ├── run_sweep.sh
│       │   ├── validate_phased.sh
│       │   ├── test_on_external.sh
│       │   └── show_sweep_results.py
│       └── result/
│           ├── saved_tsp100_model2_longTrain/  # baseline ckpt（保留）
│           └── <timestamp>_phased_*_B400_*/    # 各次 fine-tune 输出（gitignore）
└── utils/
```

---

## 6. 复现最终模型的最快路径

最终模型由两段训练拼接而成：先得到 Phase 3 anchor，再在其上跑 Phase 4 size curriculum。

```bash
cd TSP/POMO

# 1. Phase 1-3：得到 anchor ckpt（α=10→20，Phase 3 长度 280 epoch）
#    单卡 RTX 4090 约 20h
python finetune_phased.py --config ./configs/finetune_phased.json \
    --total_finetune_epochs 400 \
    --phase_epochs 60,200,280 \
    --resume_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
    --ablation all_three --knn_k 20 \
    --leader_alpha 10 --leader_alpha_final 20 \
    --desc anchor_alpha10_20_280
# 取 ./result/<timestamp>_anchor_alpha10_20_280_*/checkpoint-phase_3_leader_best.pt
# 重命名/拷到 ./result/best_ckpt_1/checkpoint-alpha10_20_280.pt（方便下一步引用）

# 2. Phase 4：size curriculum 150 epoch（A100-48G 约 18h）
python finetune_phased.py --config ./configs/finetune_phased.json \
    --resume_checkpoint ./result/best_ckpt_1/checkpoint-alpha10_20_280.pt \
    --continue_epoch_counter true \
    --phase_epochs 0,0,150 \
    --train_episodes 50000 --train_batch_size 48 \
    --phase3_lr 1e-5 --phase3_final_lr 1e-6 \
    --knn_k 20 \
    --leader_alpha 20 --leader_alpha_final 20 --leader_rampup_portion 0 \
    --n_values 100,150,200,250 --n_weights 1,3,3,2 \
    --model_save_interval 20 \
    --ablation all_three \
    --desc PHASE4_A_from_alpha10_20_280

# 3. 评测：用全局 epoch 3580 的 ckpt（in-phase epoch 42）
python test.py \
    --data_path ../data/val \
    --checkpoint_path ./result/<PHASE4_A run dir>/checkpoint-3580.pt \
    --use_cuda true --cuda_device_num 0 \
    --augmentation_enable true --aug_factor 8 \
    --detailed_log false \
    --output_json /tmp/final_eval.json
# 期望 avg_aug_gap ≈ 0.624%
```

