# Project 2: LLM-Guided Test-Time Augmentation for POMO TSP

## 概述

本项目在 Project 1 训练好的 POMO (Policy Optimization with Multiple Optima) 模型基础上，通过 **LLM-guided algorithm design** 搜索更优的 **test-time augmentation 策略**，在不重新训练模型的前提下显著提升 TSP 求解质量。

### 核心成果

| 指标 | Project 1 (Baseline) | Project 2 (Best) |
|------|---------------------|------------------|
| **Avg Gap** | 0.6237% | **0.3848%** |
| **相对改善** | — | **38.3%** |
| **改善实例数** | — | **7/10 (70%)** |
| **变差实例数** | — | **0/10** |

### 方法论

- **搜索框架**: 基于 [LLM4AD](https://github.com/Optima-CityU/LLM4AD) 的 Evolution of Heuristics (EoH) 搜索
- **LLM 模型**: gpt-4o-mini (通过阿里内部 API 调用)
- **搜索空间**: `generate_augmentations(coords, aug_factor)` 函数设计
- **评估指标**: 验证集平均 optimality gap + 改善实例比例

---

## 快速开始

### 环境要求

```bash
# Python 3.10+, PyTorch 2.x with CUDA
pip install torch numpy matplotlib
```

### 运行测试（助教入口）

```bash
cd project2

# 在验证集上评估并与 Project 1 对比
python test.py --data-dir ../TSP/data/val --compare

# 在测试集上评估
python test.py --data-dir ../TSP/data/test --compare

# 指定自定义 aug_factor
python test.py --data-dir ../TSP/data/val --aug-factor 128
```

### 运行 LLM 搜索（复现 10 种方案）

```bash
cd project2
python search_10_methods.py
```

---

## 文件结构

```
project2/
├── README.md                    # 本文档
├── test.py                      # 🔑 助教测试入口
├── best_algorithm.py            # 🔑 核心算法 (Halton-Sequence Rotations)
├── evaluation.py                # 评估模块 (LLM4AD 兼容)
├── template.py                  # LLM 搜索的函数模板
├── search_10_methods.py         # 自动搜索 10 种有效方案的脚本
├── run_search_simple.py         # 简化版 LLM 迭代搜索
├── run_eoh.py                   # LLM4AD/EoH 搜索入口
├── run_final_eval.py            # 最终全集评测对比
├── eval_quick.py                # 快速子集评估
├── extract_best.py              # 从 EoH 日志提取最优函数
├── found_methods.json           # 🔑 10 种方案的完整 JSON 记录
└── methods/                     # 10 种方案的独立代码文件
    ├── method_01_kronecker.py
    ├── method_02_weyl_sqrt2.py
    ├── method_03_rotation_shear.py
    ├── method_04_power_law.py
    ├── method_05_halton_base3.py
    ├── method_06_vdc_dihedral.py
    ├── method_07_halton_base2.py
    ├── method_08_prime_sqrt.py
    ├── method_09_interleaved_halton.py
    └── method_10_rotation_scale.py
```

---

## 测试流程详解

### 1. 模型加载

使用 Project 1 训练好的 POMO 模型 checkpoint：

```python
CHECKPOINT_PATH = "TSP/POMO/result/best_ckpt_2/checkpoint-best.pt"

MODEL_PARAMS = {
    "embedding_dim": 128,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",  # 贪心解码
}
```

### 2. 坐标预处理

将 TSPLIB 实例坐标归一化到 `[0, 1]²`（保持宽高比）：

```python
def normalize_to_unit_square(node_xy):
    xy_max = torch.max(node_xy, dim=1, keepdim=True).values
    xy_min = torch.min(node_xy, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    return (node_xy - xy_min) / ratio
```

### 3. Augmentation 生成

核心函数 `generate_augmentations(coords, aug_factor)`：

1. **输入**: 归一化坐标 `(1, N, 2)`，增强倍数 `aug_factor`
2. **第一步**: 生成 8 个标准 dihedral 变换（旋转 + 反射）
3. **第二步**: 用 Halton 准随机序列生成额外旋转角度
4. **第三步**: 对每个旋转结果做 min-max 归一化回 `[0, 1]`
5. **输出**: `(aug_factor, N, 2)` 增强坐标张量

```python
# aug_factor 自适应设置（遵循助教规则：aug ≤ problem_size × 8）
# 实际实现中我们 cap 在 800 以平衡 GPU 显存和计算时间
aug_factor = min(problem_size * 8, 800)
# 例: N=100 → aug=800, N=150 → aug=800, N=200 → aug=800 (capped at 800)
# 助教规则允许 N=100 最多 800, N=200 最多 1600, 但 800 已足够覆盖角度空间
```

### 4. POMO 推理

对每个增强副本独立运行 POMO（N 个起始节点 × aug 个变换 = N×aug 条候选路径）：

```python
# 对 aug_factor 个增强副本并行推理
env.problems = augmented_coords  # (aug_factor, N, 2)
# POMO 为每个副本生成 N 条路径 (每个节点作为起始点)
# 最终取所有 aug_factor × N 条路径中的最优解
best_tour_length = tour_lengths.min()
```

### 5. 距离计算

使用原始坐标计算路径长度（尊重 TSPLIB 的 edge weight type）：
- `EUC_2D`: 欧几里得距离取整
- `CEIL_2D`: 欧几里得距离向上取整

### 6. 评估指标

```python
gap = (model_tour_length - optimal_tour_length) / optimal_tour_length × 100%
```

---

## 算法核心：Halton-Sequence Rotation Augmentation

### 动机

标准 POMO 只使用 8 个 dihedral 变换（4 个旋转 + 4 个反射），对于复杂大规模实例，这 8 个视角远远不够。增加更多旋转角度能让 argmax 解码从不同几何视角发现更优路径。

**关键问题**: 如何选择额外的旋转角度？

### Halton 序列的优势

[Halton 序列](https://en.wikipedia.org/wiki/Halton_sequence) 是一种**低差异准随机序列**：

```
Halton(1, base=2) = 0.5
Halton(2, base=2) = 0.25
Halton(3, base=2) = 0.75
Halton(4, base=2) = 0.125
Halton(5, base=2) = 0.625
...
```

对比其他角度选择策略：

| 策略 | 数学性质 | aug=32 gap | aug=256 gap | aug=800 gap |
|------|---------|-----------|------------|------------|
| 均匀间隔 | 周期性，有系统性间隙 | ~0.55% | — | — |
| Golden Angle | 单一无理数递推 | 0.5357% | 0.4103% | 0.3922% |
| Fibonacci | φ 的倒数 | 0.4605% | — | — |
| **Halton (base 2)** | **低差异准随机** | **0.4298%** | **0.3948%** | **0.3848%** |
| **Kronecker (φ frac)** | **三距离定理最优** | — | **0.3848%** | **0.3848%** |

> 注：Kronecker 在 aug=256 时即达到 0.3848%，而 Halton Base-2 需要 aug=800 才达到同一水平。Kronecker 的收敛速度更快。

### 算法伪代码

```
function generate_augmentations(coords[1, N, 2], aug_factor):
    # Phase 1: Standard dihedral group (8 transforms)
    results = [identity, flip_x, flip_y, flip_xy, 
               transpose, flip_x∘transpose, flip_y∘transpose, flip_xy∘transpose]
    
    # Phase 2: Halton quasi-random rotations
    for i in 1...(aug_factor - 8):
        θ = Halton(i, base=2) × 2π    # quasi-random angle in [0, 2π)
        rotated = rotate(coords - 0.5, θ)
        normalized = min_max_normalize(rotated)
        results.append(normalized)
    
    return stack(results)  # shape: (aug_factor, N, 2)
```

---

## 10 种方案详细介绍

以下所有方案均在验证集（10 个 TSPLIB 实例，N=100~299）上通过 ≥60% 实例改善的要求。

### Method 1: Kronecker Sequence (Golden Ratio) ⭐ 最佳

- **Avg Gap**: 0.3848% (改善 38.3%)
- **改善实例**: 7/10 (70%)
- **思路**: 角度由黄金比例小数部分的加法递推生成：`angle_i = (i × (√5-1)/2 mod 1) × 2π`
- **Novelty**: 三距离定理保证 Kronecker 序列在任何前 N 项中只产生至多 3 种间距，是理论上 1D 等分布的最优序列
- **为何不同于 Golden Angle**: Golden angle 使用 `i × 2π/(φ²)` 即乘以黄金角度，而 Kronecker 使用 `i × (φ-1) mod 1`，产生不同的角度分布模式

### Method 2: Weyl Sequence (sqrt(2))

- **Avg Gap**: 0.3931% (改善 37.0%)
- **改善实例**: 7/10 (70%)
- **思路**: Weyl 等分布定理：`angle_i = (i × √2 mod 1) × 2π`
- **Novelty**: 使用 √2 而非黄金比例，Weyl 定理保证等分布但产生与黄金比例不同的相关结构，对某些实例几何更有利

### Method 3: Rotation + Shear Transformation

- **Avg Gap**: 0.3940% (改善 36.8%)
- **改善实例**: 7/10 (70%)
- **思路**: Halton 旋转角度 + 小幅确定性剪切变换 (shear factor ∈ [-0.1, 0.1])
- **Novelty**: 超越纯旋转，加入非正交仿射变换（剪切），在保持拓扑不变的前提下提供更丰富的几何多样性

### Method 4: Power-Law Angle Distribution

- **Avg Gap**: 0.3943% (改善 36.8%)
- **改善实例**: 7/10 (70%)
- **思路**: 非均匀角度采样，在基数方向 (0°/90°/180°/270°) 附近集中更多采样点
- **Novelty**: 信息论启发 —— 8 个 dihedral 变换在 45° 倍数处有覆盖，但 0°/90° 方向间有空隙。Power-law 分配更多预算填补这些间隙

### Method 5: Halton Base-3 Rotations

- **Avg Gap**: 0.3944% (改善 36.8%)
- **改善实例**: 7/10 (70%)
- **思路**: Halton 序列使用 base=3 而非 base=2
- **Novelty**: 不同素数底产生完全不同的数字反转模式（base-2 按二进制位翻转，base-3 按三进制位翻转），覆盖 base-2 遗留的角度间隙

### Method 6: VDC-Dihedral Expansion

- **Avg Gap**: 0.3945% (改善 36.7%)
- **改善实例**: 7/10 (70%)
- **思路**: Van der Corput 角度 + 转置扩展：每个旋转同时生成 `(x,y)` 和 `(y,x)` 版本
- **Novelty**: 将 axis-swap (坐标置换) 与准随机旋转结合，每个角度的有效多样性翻倍

### Method 7: Halton Base-2 Rotations

- **Avg Gap**: 0.3948% (改善 36.7%)
- **改善实例**: 7/10 (70%)
- **思路**: 标准 Halton 准随机序列 (base 2) 确定旋转角度
- **Novelty**: 将数论中的低差异序列首次应用于 neural TSP solver 的 test-time augmentation，保证角度分布比任何周期性序列更均匀

### Method 8: Prime-Sqrt Multi-Base Rotations

- **Avg Gap**: 0.3954% (改善 36.6%)
- **改善实例**: 7/10 (70%)
- **思路**: 循环使用 15 个素数的平方根 (√2, √3, √5, √7, ..., √47) 作为无理数乘子
- **Novelty**: 多基底无理数旋转 —— 每 15 个角度切换一次无理数基底，最大化角度间的去相关性

### Method 9: Interleaved Halton Base-2&3

- **Avg Gap**: 0.4019% (改善 35.6%)
- **改善实例**: 7/10 (70%)
- **思路**: 交替使用 Halton base-2 和 base-3 序列（奇数位用 base-2，偶数位用 base-3）
- **Novelty**: 两种不同的低差异序列交织，产生兼具两者优点的混合分布

### Method 10: Rotation + Anisotropic Scaling

- **Avg Gap**: 0.4058% (改善 34.9%)
- **改善实例**: 7/10 (70%)
- **思路**: Halton 旋转 + 温和的 x 轴各向异性缩放 (scale ∈ [0.9, 1.1])
- **Novelty**: 通过轻微拉伸/压缩打破纯旋转对称性，让模型看到不同长宽比下的实例，揭示不同的贪心路径结构

---

## 验证集逐实例结果

以部署方案 (Halton Base-2, 与 Kronecker 结果完全一致) 运行 `python test.py --compare` 的输出：

```
Instance          N    Optimal  P1 (8-fold)    P2 (ours)   Gap_P1   Gap_P2  Status
--------------------------------------------------------------------------------
ch150           150     6528.0      6559.0      6557.0  0.4749%  0.4442%       ✓
eil101          101      629.0       629.0       629.0  0.0000%  0.0000%       =
kroA100         100    21282.0     21295.0     21282.0  0.0611%  0.0000%       ✓
kroA200         200    29368.0     29547.0     29464.0  0.6095%  0.3269%       ✓
kroB150         150    26130.0     26257.0     26153.0  0.4860%  0.0880%       ✓
kroC100         100    20749.0     20749.0     20749.0  0.0000%  0.0000%       =
kroE100         100    22068.0     22243.0     22173.0  0.7930%  0.4758%       ✓
pr124           124    59030.0     59030.0     59030.0  0.0000%  0.0000%       =
pr226           226    80369.0     81507.0     81143.0  1.4160%  0.9631%       ✓
pr299           299    48191.0     49346.0     48938.0  2.3967%  1.5501%       ✓
--------------------------------------------------------------------------------
avg_aug_gap: 0.3848%   P1 baseline: 0.6237%   Improved: 7/10 (70%)
```

**分析**:
- 3 个小实例 (eil101, kroC100, pr124) 已被 baseline 解决至最优 → 无法再改善
- 大实例 (kroA200, pr226, pr299) 改善最显著 → 更多旋转角度对大规模实例更有价值
- 没有任何实例变差 → 策略稳定可靠

---

## 规则合规性

| 规则 | 合规 |
|------|------|
| aug ≤ problem_size × 8 | ✅ 使用 `min(N*8, 800)` |
| 不允许 2-opt/LKH3 后处理 | ✅ 无任何后处理 |
| 推理时改进，不重新训练 | ✅ 仅修改 augmentation 函数 |
| 使用 Project 1 的 checkpoint | ✅ 同一 checkpoint |

---

## LLM 搜索方法论

### 搜索框架

```
┌─────────────┐     ┌───────────┐     ┌──────────────┐
│  gpt-4o-mini │────>│ 函数生成   │────>│ 编译 & 验证  │
│  (API call)  │     │ (Python)   │     │ (exec + test)│
└─────────────┘     └───────────┘     └──────┬───────┘
       ▲                                       │
       │                                       ▼
┌──────┴──────┐                      ┌──────────────┐
│ Prompt 进化  │<────────────────────│ POMO 评估    │
│ (历史反馈)   │                      │ (val set)    │
└─────────────┘                      └──────────────┘
```

### 搜索配置

```python
LLM_HOST = "pre-openai-keys.alibaba-inc.com"
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.9  # 高创造性
MAX_TOKENS = 3000
AUG_FACTOR = 256   # 搜索时用 256，最终用 min(N*8, 800)
```

### EoH 进化策略

1. **初始种群**: 5 个手工设计的策略（Halton、Golden-angle、Weyl 等）
2. **变异**: LLM 根据当前已有方案 + 失败历史生成新方案
3. **选择**: 保留 avg_gap < baseline 且改善 ≥ 60% 实例的方案
4. **终止**: 找到 10 个有效方案即停止

---

## 数学背景

### Halton 序列

Halton 序列是 Van der Corput 序列的高维推广。对于 base $b$，第 $i$ 个元素为：

\[
H_b(i) = \sum_{k=0}^{\infty} d_k(i) \cdot b^{-(k+1)}
\]

其中 $d_k(i)$ 是 $i$ 的 $b$ 进制展开的第 $k$ 位。

**性质**: 前 $N$ 个点的星差异 (star discrepancy) 为 $O(\log N / N)$，远优于随机序列的 $O(1/\sqrt{N})$。

### Kronecker 序列 (三距离定理)

对于无理数 $\alpha$，序列 $\{i \cdot \alpha \mod 1\}_{i=1}^N$ 将 $[0,1)$ 分为至多 3 种不同长度的间隔。

当 $\alpha = (\sqrt{5}-1)/2$（黄金比例小数部分）时，这 3 种间隔长度比最接近 1:1:1，给出理论上最均匀的 1D 准随机分布。

### Weyl 等分布定理

对任何无理数 $\alpha$，序列 $\{n\alpha\}_{n=1}^{\infty}$ 在 $[0,1)$ 上等分布。

不同的 $\alpha$ 产生不同的收敛速率和相关结构：
- $\alpha = \sqrt{2}$: 二次无理数，良好的等分布性
- $\alpha = (\sqrt{5}-1)/2$: 最慢收敛但最均匀

---

## 复现指南

### 完整复现 10 种方案

```bash
cd project2
python search_10_methods.py
# 输出: found_methods.json + methods/ 目录
# 耗时: ~3-5 分钟 (含 GPU 评估)
```

### 评估单个方案

```python
import torch
import importlib.util

# 加载方案
spec = importlib.util.spec_from_file_location("m", "methods/method_01_kronecker.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 使用
coords = torch.rand(1, 100, 2)
augmented = mod.generate_augmentations(coords, aug_factor=800)
print(augmented.shape)  # torch.Size([800, 100, 2])
```

### 切换部署方案

修改 `best_algorithm.py` 中的 `generate_augmentations` 函数即可。当前部署的是 **Halton Base-2** 方案。

> **注**: 实验发现在 aug=800 时，Kronecker (Method 1) 和 Halton Base-2 (Method 7) 在所有 10 个验证集实例上产生完全相同的最优解。这是因为 800 个旋转角度已经足够密集，两种准随机序列都能覆盖到相同的最优视角。选择 Halton Base-2 作为部署方案是因为其实现更直观、代码更简洁。

---

## 依赖

- Python >= 3.10
- PyTorch >= 2.0 (with CUDA)
- NumPy
- (可选) LLM4AD framework (`src/LLM4AD/`)

---

## 致谢

- POMO 模型: [Kwon et al., 2020](https://arxiv.org/abs/2010.16011)
- LLM4AD 框架: [Optima-CityU/LLM4AD](https://github.com/Optima-CityU/LLM4AD)
- LLM API: 阿里内部 gpt-4o-mini 预发环境
