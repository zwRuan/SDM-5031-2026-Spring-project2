# Project 2: Test-Time Augmentation Improvement for POMO TSP

## 方法简介

我们在 Project 1 训练好的 POMO 模型基础上，通过 **LLM-guided algorithm design**（使用 gpt-4o-mini + LLM4AD 框架）搜索到了更优的 test-time augmentation 策略：**Halton-Sequence Quasi-Random Rotations**。

**核心思路**：标准 POMO 使用 8 个 dihedral 变换（固定的旋转+反射），我们用 Halton 准随机序列生成更多旋转角度，提供低差异（low-discrepancy）的角度覆盖，让模型从更多样的几何视角搜索更优路径。

**验证集结果**：
- Baseline (Project 1, 8-fold dihedral): avg_gap = 0.6237%
- Our method: avg_gap = **0.3848%** (相对改善 38.3%)
- 改善实例数: **7/10 (70%)**，变差实例数: 0/10

---

## 环境配置

### Python 环境

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy
```

或使用 conda：

```bash
conda create -n pomo python=3.10
conda activate pomo
pip install torch numpy
```

### 依赖

仅需要：
- **Python** >= 3.10
- **PyTorch** >= 2.0 (需要 CUDA 支持)
- **NumPy**

无其他第三方依赖。

---

## 测试方法

### 快速测试命令

```bash
cd project2
python test.py --data_path <测试集路径>
```

### 标准接口（TA 推荐）

```bash
cd project2

# 基本评测
python test.py --data_path /path/to/test_instances --output_json /path/to/eval_result.json

# 完整参数示例
python test.py \
    --data_path /path/to/test_instances \
    --checkpoint_path ../TSP/POMO/result/best_ckpt_2/checkpoint-best.pt \
    --output_json /path/to/eval_result.json \
    --use_cuda 1 \
    --cuda_device_num 0 \
    --augmentation_enable 1
```

### 其他用法

```bash
# 在验证集上运行并与 Project 1 对比
python test.py --data_path ../TSP/data/val --compare

# 指定自定义 aug_factor（默认为 min(N*8, 800)）
python test.py --data_path /path/to/test --aug-factor 256
```

### 支持的参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_path` / `--data-dir` | str | `../TSP/data/val` | 测试集目录 |
| `--checkpoint_path` / `--checkpoint` | str | 自动定位 | 模型权重路径 |
| `--output_json` | str | 无 | JSON 结果输出路径 |
| `--use_cuda` | int | 1 | 是否使用 GPU |
| `--cuda_device_num` | int | 0 | GPU 编号 |
| `--augmentation_enable` | int | 1 | 启用增强 |
| `--compare` | flag | - | 对比 Project 1 |
| `--aug-factor` | int | 自适应 | 自定义 aug 数 |

### 输入格式

测试集目录下应包含 `.tsp` 格式文件（标准 TSPLIB 格式）。脚本会自动扫描目录中所有 `.tsp` 文件。

### 输出格式

```
================================================================================
Project 2: LLM-Designed Augmentation Strategy (Halton-Sequence Rotations)
================================================================================
  Data: /path/to/test (N instances)
  Checkpoint: .../TSP/POMO/result/best_ckpt_2/checkpoint-best.pt
  Aug factor: adaptive min(N*8, 800)
  Device: cuda:0

Instance          N    Optimal  P1 (8-fold)    P2 (ours)   Gap_P1   Gap_P2  Status
--------------------------------------------------------------------------------
...
```

使用 `--compare` 参数会同时输出 Project 1 (8-fold dihedral) 和 Project 2 (ours) 的结果。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `test.py` | **测试入口脚本** — 助教直接运行此脚本 |
| `best_algorithm.py` | **核心算法** — `generate_augmentations()` 函数 |
| `evaluation.py` | 评估模块（被 test.py 调用） |
| `found_methods.json` | 10 种方案的完整记录（名称、思路、指标） |
| `methods/` | 10 种方案的独立 Python 文件 |
| `methods_readme.md` | 各方案的详细技术文档 |

### 关键依赖路径

```
项目根目录/
├── TSP/
│   ├── POMO/
│   │   ├── TSPModel.py          # POMO 模型定义
│   │   ├── TSPEnv.py            # TSP 环境
│   │   ├── tsplib_utils.py      # TSPLIB 读取 & 最优解
│   │   └── result/best_ckpt_2/
│   │       └── checkpoint-best.pt  # Project 1 训练的模型权重
│   └── TSProblemDef.py
└── project2/
    ├── test.py                  # ← 助教运行这个
    └── best_algorithm.py        # ← 核心 augmentation 函数
```

---

## 规则合规性

| 规则 | 状态 | 说明 |
|------|------|------|
| aug ≤ problem_size × 8 | ✅ | 使用 `min(N*8, 800)` |
| 不允许 2-opt/LKH3 后处理 | ✅ | 仅做坐标几何变换，不修改模型输出的解 |
| 推理时改进，不重新训练 | ✅ | 使用 Project 1 的 checkpoint，不修改模型参数 |

---

## 算法原理（简述）

```python
def generate_augmentations(coords, aug_factor):
    """
    1. 生成 8 个标准 dihedral 变换 (与 POMO 源码一致)
    2. 用 Halton 准随机序列 (base=2) 生成额外旋转角度:
       angle_i = Halton(i) × 2π
    3. 对每个旋转结果做 min-max 归一化回 [0,1]
    4. 返回 (aug_factor, N, 2) 张量
    """
```

Halton 序列是一种低差异序列，通过二进制数字反转产生 [0,1) 中的准随机点，保证前 N 个角度的分布比均匀间隔或随机采样更均匀。
