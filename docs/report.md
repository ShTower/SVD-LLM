# SVD-LLM 复现实验报告

> **复现论文**: SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression (ICLR 2025)  
> **复现日期**: 2026-07-10 ~ 2026-07-11  
> **实验环境**: 华为 Ascend 910B 80GB NPU × 1 + 20 vCPU + 160GB RAM

---

## 1. 引言

SVD-LLM 是一种基于奇异值分解（SVD）的大语言模型训练后压缩方法。其核心创新包括：

- **截断感知数据白化**（Truncation-Aware Data Whitening）：通过 Cholesky 分解构建白化矩阵，确保奇异值与压缩损失直接映射，截断最小奇异值即获得最小损失
- **顺序低秩近似参数更新**（Sequential Low-rank Approximation）：对 SVD 分解后的 U、V 矩阵依次 LoRA 微调

本次复现的目标是在华为昇腾 NPU 910B 平台上验证 SVD-LLM 的核心压缩效果。

## 2. 实验环境

| 项目 | 配置 |
|------|------|
| NPU 型号 | Ascend 910B1 × 1（80GB）|
| 计算架构 | CANN 8.1.RC1 |
| CPU | 20 vCPU (aarch64) |
| 系统内存 | 160 GB |
| 操作系统 | openEuler 22.03 LTS-SP4 (aarch64) |
| Python | 3.10.17 |
| PyTorch | 2.5.1 |
| torch_npu | 2.5.1.post1 |
| Transformers | 4.35.2（论文要求精确版本）|

## 3. NPU 适配方案

### 3.1 核心问题

PyTorch 的华为 NPU 后端（torch_npu）不完整支持高级线性代数算子（SVD、Cholesky、Inv、LSTSQ），而 SVD-LLM 的核心压缩流程强依赖这些算子。

### 3.2 解决方案

新增 `utils/device_utils.py` 统一设备管理模块，实现两层策略：

1. **设备自动检测**：按 NPU > CUDA > CPU 优先级自动选择计算设备
2. **线性代数 CPU 回退**：SVD、Cholesky、Inv、LSTSQ、eigvalsh 等算子在 CPU 上执行后结果搬运回 NPU，绕过 NPU 算子缺失问题
3. **统一设备 API**：`clear_cache()`、`sync_device()`、`allocated_memory()` 等替代 `torch.cuda.*`

### 3.3 修改文件清单

| 文件 | 改动内容 |
|------|---------|
| `utils/device_utils.py` (新增) | 设备管理、CPU 回退线性代数算子 |
| `SVDLLM.py` | 12处 torch.linalg.\* → safe\_\*，torch.cuda → device_utils，--DEV 自动检测 |
| `evaluater.py` | 所有 torch.cuda.\* → device_utils，.cuda() → .to(device) |
| `utils/model_utils.py` | 去除硬编码 torch.device("cuda") |
| `utils/LoRA.py` | 设备自动检测，int8训练 → NPU 安全回退 |
| `quant_llama.py`, `gptq/gptq.py` | torch.cuda.* 替换 |

---

## 4. 实验设置

### 4.1 实验模型

- **模型**: LLaMA-7B (`jeffwan/llama-7b-hf`)
- **压缩比**: 20%（保留 80% 参数）
- **校准数据集**: WikiText-2（256 样本 × 2048 tokens）
- **方法变体**: SVD-LLM (W) — 仅白化压缩，无 LoRA 微调

### 4.2 执行命令

```bash
# Step 1: 白化 + SVD 压缩
./run.sh SVDLLM.py --step 1 --model jeffwan/llama-7b-hf --ratio 0.2 \
    --whitening_nsamples 256 --dataset wikitext2 --seed 3 \
    --model_seq_len 2048 --save_path ./output

# Step 4: WikiText-2 PPL 评估
./run.sh SVDLLM.py --step 4 --model_path ./output/jeffwan_llama_7b_hf_whitening_only_0.8.pt

# Step 5: 推理效率测试
./run.sh SVDLLM.py --step 5 --model_path ./output/jeffwan_llama_7b_hf_whitening_only_0.8.pt
```

---

## 5. 实验结果

### 5.1 压缩效果（WikiText-2 Perplexity）

| Method | PPL (WikiText-2) | 论文 PPL | 相对论文 |
|--------|:--:|:--:|:--:|
| Original LLaMA-7B | **5.68** | 5.68 | 完全一致 |
| SVD-LLM (W) @20% | **7.89** | 7.94 | 优于论文 (+0.05) |

**分析**：白化+SVD 压缩后的 PPL 为 7.89，与论文的 7.94 高度一致，甚至略优。这验证了截断感知数据白化技术的有效性——与不做白化的 Vanilla SVD（PPL 暴增至 20000+）相比，白化将精度损失控制在极低水平。

### 5.2 参数更新尝试（Step 2）

| Method | PPL | 状态 |
|--------|:--:|:--:|
| SVD-LLM @20% | 16.96 | ❌ 异常 |

Step 2 使用的闭式解更新（`torch.linalg.lstsq`）在 CPU 回退过程中引入了数值精度问题，导致 PPL 反而上升。论文原版的 LoRA 两轮微调因 NPU 环境限制未能在本次复现中验证。**建议后续在 NPU 环境验证完整的 LoRA 训练流程。**

### 5.3 推理效率（OPT-6.7B 未完成）

| 指标 | 压缩后模型 (LLaMA-7B @20%) |
|------|---------------------------|
| 总显存 | 28.57 GB |
| 权重显存 | 20.55 GB |
| 激活显存 | 8.02 GB |
| 吞吐量 | **42.16 tokens/sec** |

### 5.4 压缩时间分析

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 模型下载 | ~30 min | 13GB 从 HF 镜像 |
| Profiling（白化矩阵）| ~31 min | 32层激活收集 |
| SVD 压缩 | ~2h 22min | 16线程 aarch64 CPU |
| Step 1 总计 | **~3.5h** | |

**注意**：SVD 计算在 CPU 上执行（SVD 4096×11008 矩阵约 39秒），这是复现的主要瓶颈。可选未来通过 `scipy.linalg.svd` 或其他 BLAS 加速库进一步提升速度。

---

## 6. 遇到的问题与解决方案

| 问题 | 解决方案 |
|------|---------|
| NPU 不支持 SVD/Cholesky | `safe_*` 自动回退 CPU |
| `torch.cuda.*` 在 NPU 不存在 | `device_utils` 统一 API |
| HuggingFace 直连失败 | `HF_ENDPOINT=https://hf-mirror.com` 镜像 |
| 模型下载中断 | 自动断点续传 |
| Google Drive 无法访问 | 跳过 C4 数据集，使用 WikiText-2 |
| ptb 数据集无法下载 | raw.githubusercontent.com 被墙 |
| LLaMA2-7B 需授权 | 跳过，使用 LLaMA-7B |
| OPT-6.7B OOM Kill | 80GB NPU 不够（bias层额外显存）|
| Step 2 lstsq 结果异常 | NPU→CPU 数值精度问题，待解决 |
| 硬编码 `torch.device("cuda")` | 改为 `detect_device()` 自动检测 |

---

## 7. 结论

本次复现在华为昇腾 910B NPU 平台上成功验证了 SVD-LLM 的核心机制：

1. **截断感知数据白化技术有效**：LLaMA-7B @20% 压缩比下，PPL 从原始 5.68 仅升至 7.89，与论文的 7.94 高度一致，证明白化技术显著优于直接 SVD（PPL >20000）
2. **NPU 适配方案可行**：通过 SVD/Cholesky 自动 CPU 回退策略，在不修改 CANN 内核的前提下完成了实验
3. **推理效率可接受**：42 tok/s 的吞吐量验证了压缩模型在 NPU 上的推理可用性

**后续工作建议**：
- 在 NPU 上完成完整的 LoRA 两轮顺序微调
- 测试更多压缩比（40%、60%）
- 使用 scipy BLAS 加速 SVD 计算
- 验证 Step 2 lstsq 的数值精度问题

---

## 参考

- SVD-LLM 论文: https://openreview.net/forum?id=LNYIUouhdt
- 代码仓库: https://github.com/ShTower/SVD-LLM (npu 分支)
