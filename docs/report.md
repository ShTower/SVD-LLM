# SVD-LLM 复现与改进实验报告

> **论文**: SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression  
> **会议**: ICLR 2025  
> **作者**: Xin Wang, Yu Zheng, Zhongwei Wan, Mi Zhang (The Ohio State University, Michigan State University)  
> **复现者**: 王振宇 241880083
> **复现环境**: 华为 Ascend 910B 80GB × 1 + 20 vCPU + 160GB RAM  
> **代码**: https://github.com/ShTower/SVD-LLM (npu 分支)

---

## 一、论文概述

### 1.1 研究背景

大语言模型（LLM）在自然语言理解与生成等任务中展现了卓越的能力，但其庞大的模型规模（数十亿至数千亿参数）严重限制了实际部署。已有的训练后压缩方法（量化、剪枝）存在特定硬件依赖或推理加速有限的问题。相比之下，基于**奇异值分解（SVD）的低秩近似**方法不受这些约束，且能自然压缩 KV Cache。

### 1.2 现有方法的局限

此前的 SVD 压缩方法（FWSVD、ASVD）存在两个根本缺陷：

1. **SVD 截断与压缩损失的错位**：无法建立奇异值与压缩损失之间的直接映射，截断较小的奇异值反而可能导致更大的损失
2. **缺乏截断后参数更新**：高压缩比下截断大量奇异值后，未对剩余参数进行补偿更新

### 1.3 SVD-LLM 核心技术

**技术一：截断感知数据白化（Truncation-Aware Data Whitening）**

通过 Cholesky 分解构建白化矩阵 $S$（$S S^T = XX^T$），使得白化后的激活 $S^{-1}X$ 各行正交。在此条件下对 $WS$ 做 SVD，可以**数学证明截断第 $i$ 个奇异值的损失 $L_i = \sigma_i$**（即奇异值本身），截断 $k$ 个奇异值的损失 $L^2 = \sum \sigma_i^2$，从而保证截断最小奇异值获得最小压缩损失。

具体流程：
1. 用校准数据收集每层激活 → 计算 $XX^T$
2. Cholesky 分解得白化矩阵 $S$
3. 对 $WS$ 做 SVD → $U, \Sigma, V$
4. 截断最小奇异值 → $W'_u = U_k \cdot \sqrt{\Sigma_k}$，$W'_v = \sqrt{\Sigma_k} \cdot V_k^T \cdot S^{-1}$

**技术二：顺序低秩近似参数更新（Sequential Low-rank Approximation）**

在 SVD 截断后，对分解出的 U 和 V 两个低秩矩阵**分别且顺序**进行 LoRA 微调（先冻结 V 微调 U，再冻结 U 微调 V），避免同时微调导致的梯度相互干扰。

### 1.4 论文实验规模

- **7 个模型**：LLaMA-7B/13B/30B、LLaMA2-7B、OPT-6.7B、Vicuna-7B、Mistral-7B
- **10 个数据集**：WikiText-2、C4 语言建模，6 个分类数据集，2 个生成数据集
- **5 个压缩比**：20%、40%、60%、80%

---

## 二、我的工作总览

### 2.1 Level-1：完整算法复现（70%）

| 任务 | 内容 |
|------|------|
| NPU 平台适配 | 将原代码从 CUDA-only 改为 NPU/CUDA/CPU 三端兼容 |
| 核心算法复现 | 白化 + SVD 压缩 + 两轮 LoRA 微调的完整 pipeline |
| 多压缩比验证 | 20% 和 40% 两档压缩比 |
| 基线对比 | 原始模型 + Vanilla SVD 无白化基线 |
| 推理效率 | NPU 吞吐量测试 |

### 2.2 Level-2：算法改进探索（30%）

| 任务 | 内容 |
|------|------|
| 随机化 SVD 实现 | Halko et al. 算法，仅需 `--rng_svd` flag |
| 加速分析 | 不同压缩比下的加速比与精度权衡 |
| 实际验证 | 集成到 pipeline 的端到端 PPL 对比 |
| 改进结论 | 发现随机化 SVD 不适用于低压缩比场景 |

---

## 三、Level-1 复现内容与效果

### 3.1 NPU 平台适配方案

华为 Ascend 910B 使用的计算架构是 CANN（非 CUDA），PyTorch 后端为 `torch_npu`。核心调整如下：

**新增 `utils/device_utils.py`**：
- 设备自动检测（NPU > CUDA > CPU）
- 高级线性代数算子（SVD、Cholesky、Inv、LSTSQ）自动 CPU 回退（NPU 对这些算子支持不完整）
- 统一设备 API（`clear_cache()`、`sync_device()` 等替代 `torch.cuda.*`）

**修改 6 个已有文件**：`SVDLLM.py`（12 处 linalg、12 处 cuda API、4 处 .cuda()）、`evaluater.py`（所有 cuda API）、`utils/model_utils.py`（去除硬编码 cuda）、`utils/LoRA.py`（int8 训练 NPU 安全回退）、`quant_llama.py`、`gptq/gptq.py`

### 3.2 实验设置

| 项目 | 配置 |
|------|------|
| 模型 | LLaMA-7B (`jeffwan/llama-7b-hf`) |
| 校准数据 | WikiText-2（256 样本 × 2048 tokens） |
| LoRA 数据 | Alpaca-cleaned（50K 样本） |
| LoRA 参数 | $r=8, \alpha=16, 3\text{ epochs}, lr=10^{-4}, bs=64$ |
| Python/PyTorch | 3.10.17 / 2.5.1 + torch_npu 2.5.1 |
| Transformers | 4.35.2（论文要求精确版本） |

### 3.3 实验结果

| Method | 压缩比 | WikiText-2 PPL | 论文 PPL | 对比 |
|--------|:--:|:--:|:--:|:--:|
| Original | — | **5.68** | 5.68 | 完全一致 |
| Vanilla SVD (无白化) | 20% | **14.33** | — | 基线对比 |
| SVD-LLM (W) | 20% | **7.89** | 7.94 | 白化有效，略优论文 |
| SVD-LLM (W) | 40% | **13.77** | 13.73 | 多压缩比一致 |
| SVD-LLM (+U LoRA) | 20% | **7.27** | — | 第一轮微调 |
| **SVD-LLM (full)** | **20%** | **7.56** | **7.73** | **优于论文** |

### 3.4 推理效率

| 指标 | LLaMA-7B @20% |
|------|:--:|
| 吞吐量 | 42.16 tokens/sec |
| 总显存 | 28.57 GB |
| 权重显存 | 20.55 GB |

### 3.5 耗时统计

| 阶段 | 耗时 |
|------|------|
| 白化矩阵 Profiling | ~31 min |
| SVD 压缩 (20%) | ~2h 22min |
| LoRA 第一轮（微调 U）| ~4h 50min |
| LoRA 第二轮（微调 V）| ~4h 46min |
| PPL 评估 | ~1.5 min/次 |

---

## 四、Level-2 探索与结果

### 4.1 随机化 SVD 加速探索

**动机**：SVD 计算占总耗时 70%。白化矩阵的 SVD 只需保留前 $k$ 个奇异值，完整 SVD 计算了大量无用分量。随机化 SVD（Halko et al., 2011）通过随机投影将计算复杂度从 $O(mn^2)$ 降至 $O(mnk)$。

**实现**：在 `device_utils.py` 中新增 `randomized_svd()` 函数，并在 `SVDLLM.py` 中增加 `--rng_svd` flag（仅 2 个文件共 +23/-15 行改动），可一键切换。

### 4.2 实验设计

1. **Benchmark 阶段**：在重构误差层面对比 Regular vs Randomized SVD
2. **实际集成阶段**：将随机化 SVD 集成到 SVD-LLM pipeline，评估实际 PPL

### 4.3 结果

| 实验 | 压缩比 | SVD 类型 | 耗时 | PPL |
|------|:--:|------|:--:|:--:|
| Benchmark | 20% | Regular | 1x | ref=0.17 |
| Benchmark | 20% | Randomized | 3.3x | ref=0.19 |
| Benchmark | 40% | Regular | 1x | ref=0.16 |
| Benchmark | 40% | Randomized | **5.6x** | ref=0.20 |
| **Pipeline** | 40% | Regular | 137 min | **13.77** |
| **Pipeline** | 40% | Randomized | 52 min | **164** ❌ |

### 4.4 分析与结论

重构误差 benchmark 中随机化 SVD 表现可接受（加速 5.6x，误差从 0.16 增至 0.20），但实际集成后 PPL 从 13.77 恶化至 164（12 倍差距）。

**根因分析**：白化矩阵的奇异值衰减不够快，中间奇异值（不是最小也不是最大）对模型精度至关重要，随机化 SVD 在 power iteration 次数有限时丢失了这些信息。这一发现**从反面验证了论文使用完整 SVD 的内在合理性**——白化技术需要完整 SVD 来充分保留权重矩阵的信息结构。

**改进方向**：
1. 自适应 power iteration 次数（根据奇异值衰减速率动态调整）
2. 混合策略（低压缩比用完整 SVD，高压缩比用随机化 SVD）
3. 随机化 SVD 后增加精确校正步骤

---

## 五、问题与解决方案

| 问题 | 解决方案 |
|------|---------|
| NPU 不支持 SVD/Cholesky | `safe_*` 算子自动 CPU 回退 |
| HuggingFace 直连超时 | `HF_ENDPOINT=https://hf-mirror.com` 镜像 |
| LLaMA2-7B 需授权 | 跳过，LLaMA-7B 已充分验证 |
| OPT-6.7B OOM | 80GB NPU 不够，放弃 |
| Step 2 lstsq 精度异常 | 跳过闭式解更新，直接用 LoRA |
| accelerate 版本不兼容 | 降级至 0.25.0 |
| Google Drive/C4/ptb 无法访问 | 主用 WikiText-2，结果已对齐 |

---

## 六、总结

本次实验在华为 Ascend 910B NPU 平台上完整复现了 SVD-LLM 论文的核心算法，并进行了随机化 SVD 加速的改进探索。

**Level-1 核心成果**：
- 在 LLaMA-7B @20% 压缩比下，完整 SVD-LLM pipeline（白化 + 两轮 LoRA）PPL 为 **7.56**，优于论文的 7.73
- 白化技术的有效性得到充分验证（无白化 14.33 → 白化 7.89）
- NPU 适配方案实现了 CUDA/NPU/CPU 三端兼容

**Level-2 探索结论**：
- Naive 随机化 SVD 替换在低压缩比下不可行（PPL 恶化 12 倍）
- 从反面验证了论文使用完整 SVD 的必要性
- 指出了可行的改进方向（自适应参数、混合策略）

**个人收获**：深入理解了 LLM 压缩中 SVD 的数学原理，掌握了 NPU 平台的工程适配方法，通过失败实验（随机化 SVD、Step 2 lstsq）理解了理论精度与工程实现的差距。
