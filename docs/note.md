# SVD-LLM 项目笔记

## 一、项目概览

**SVD-LLM** 是一种基于奇异值分解（SVD）的大语言模型（LLM）训练后压缩方法。核心创新点：

1. **截断感知数据白化（Truncation-Aware Data Whitening）**：通过 Cholesky 分解对激活进行白化，使奇异值直接映射到压缩损失，确保截断最小奇异值即获得最小损失。
2. **顺序低秩近似参数更新（Sequential Low-rank Approximation）**：对 SVD 分解后的两个低秩矩阵（U 和 V）**依次**进行 LoRA 微调，而非同时微调，避免梯度相互干扰。

论文发表于 **ICLR 2025**，代码基于 **PyTorch** 和 **HuggingFace Transformers (4.35.2)**。

---

## 二、项目结构

```
SVD-LLM/
├── SVDLLM.py                    # 主入口脚本（Step 1~5）
├── quant_llama.py               # GPTQ 量化集成（SVD-LLM + GPTQ）
├── evaluater.py                 # 评估函数（困惑度 PPL、吞吐量）
├── compress_llama.sh            # 一键压缩 LLaMA-7B 的 Shell 示例
├── svdllm_gptq.sh               # SVD-LLM + GPTQ 的 Shell 示例
├── requirements.txt             # 依赖列表
├── README.md                    # 项目说明文档
│
├── component/                   # 压缩后的模型组件（替换原模型模块）
│   ├── svd_llama.py             # LLaMA/Vicuna 的压缩版 Attention 和 MLP
│   ├── svd_llama_kvcache.py     # LLaMA 压缩版 Attention（带 KV cache 优化）
│   ├── svd_mistral.py           # Mistral 的压缩版 Attention 和 MLP
│   └── svd_opt.py               # OPT 的压缩版 DecoderLayer（含 Attention + FFN）
│
├── utils/                       # 工具函数
│   ├── data_utils.py            # 数据集加载（wikitext2, ptb, c4）
│   ├── model_utils.py           # 模型加载（HuggingFace / 本地）
│   ├── LoRA.py                  # LoRA 微调脚本（两轮顺序微调 U 和 V）
│   ├── Prompter.py              # 指令模板（Alpaca 格式）
│   └── peft/                    # 本地 PEFT 库（LoRA 适配器）
│
├── gptq/                        # GPTQ 量化工具
│   ├── gptq.py                  # GPTQ 核心算法
│   └── quant.py                 # 量化器（Quantizer）
│
├── docs/                        # 文档
│   ├── 2403.07378v5.pdf         # 原始论文 PDF
│   ├── translate.md             # 论文中文翻译
│   └── note.md                  # 本文件（项目笔记）
│
└── figures/                     # 图片资源
    ├── logo.png
    ├── framework_v1.jpg
    └── framework_v2.jpg
```

---

## 三、Compression Pipeline 详解

SVD-LLM 的压缩流程通过 `SVDLLM.py` 的 `--step` 参数控制，共有 5 个步骤：

### Step 1: 白化 + SVD 压缩（仅白化，无参数更新）

```bash
python SVDLLM.py --step 1 --ratio 0.2 --model jeffwan/llama-7b-hf \
    --whitening_nsamples 256 --dataset wikitext2 --seed 3 \
    --model_seq_len 2048 --save_path .
```

**目标**：
- 用校准数据集（默认 256 个样本）计算白化矩阵
- 对每个 Linear 层：收集激活 → 计算 $XX^T$ → Cholesky 分解 → SVD → 截断 → 替换原模块

**流程**：
1. `profle_svdllm()` 或 `profle_svdllm_low_resource()`：
   - 注册 forward hook，收集每个 Linear 层输入的 $x \cdot x^T$ 累加
   - 对累加结果进行 Cholesky 分解，得到白化矩阵
2. `whitening()`：
   - 对每层每个 Linear：$W_{scaled} = W \times S$，SVD 分解，截断
   - 构造 SVD 版 Attention/MLP，替换原模型对应层
   - `W'_u = U_k \times sqrt(Σ_k)`，`W'_v = sqrt(Σ_k) \times V_k^T \times S^{-1}$

**输出**：保存 `{model, tokenizer}` 为 `.pt` 文件

**关键参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ratio` | 0.2 | 压缩比，实际传入后会做 `1-ratio` 转换 |
| `--whitening_nsamples` | 256 | 校准样本数 |
| `--run_low_resource` | False | 启用低资源模式（15G 可压缩 LLaMA-7B） |

### Step 2: 白化 + 本地更新（带参数更新）

```bash
python SVDLLM.py --step 2 --model jeffwan/llama-7b-hf --ratio 0.2 \
    --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path .
```

**目标**：在 Step 1 的基础上，用少量校准数据（`--updating_nsamples`=16）对截断后的 U 和 V 进行**闭式解更新**（closed-form update）。

**流程**：
1. 先做白化 + SVD 截断（同 Step 1）
2. 对每层，用校准数据的前向激活/输出，通过最小二乘（`torch.linalg.lstsq`）更新 U 矩阵（`update_u`）
3. `local_update` 类：记录 SVD 截断结果，用 `add_batch_update_u()` 累积校准数据，`fasterprune()` 输出更新后的 U、V

**关键区别**：Step 2 的更新是**局部闭式解**（非 LoRA 训练），速度很快。

### Step 3: 仅直接更新（无白化）

```bash
python SVDLLM.py --step 3 --model jeffwan/llama-7b-hf \
    --ratio 0.2 --dataset wikitext2 --model_seq_len 2048 --save_path .
```

**目标**：直接在原始权重上做 SVD 截断后（不做白化），用校准数据进行闭式解更新。

**适用**：用于消融实验，对比白化的效果。

### Step 4: 困惑度评估（PPL Evaluation）

```bash
python SVDLLM.py --step 4 --model_path <compressed_model_path>
```

**目标**：评估压缩模型在 wikitext2 上的困惑度（Perplexity）。

**注意**：
- 支持加载原始模型 (`--model_path original`)
- 支持加载 LoRA 微调后的模型 (`--lora <path>`)
- 需要事先下载 c4 的验证集 json 文件放在 `utils/` 下

### Step 5: 效率评估（Efficiency Evaluation）

```bash
python SVDLLM.py --step 5 --model_path <compressed_model_path>
```

**目标**：测量推理吞吐量（tokens/sec）、总显存、权重显存、激活显存。

**方法**：生成 `original_len + generated_len` 个 token，统计耗时和显存峰值。

---

## 四、完整复现流程（以 LLaMA-7B 为例）

### 环境准备

```bash
conda create -n compress python=3.9
conda activate compress
pip install -r requirements.txt
```

> ⚠️ **重要**：`transformers` 版本必须精确为 **4.35.2**，因为 `component/` 下的自定义模块依赖此版本的源码结构。

### 4.1 基础压缩（Step 1 + LoRA 两轮顺序微调）

这是论文中推荐的完整流程，参考 `compress_llama.sh`：

```bash
# 阶段 A：白化 + SVD 压缩（20% 压缩比）
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 \
    --whitening_nsamples 256 --dataset wikitext2 --seed 3 \
    --model_seq_len 2048 --save_path .

# 评估压缩后模型（仅白化，无微调）的 PPL
python SVDLLM.py --step 4 --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt

# 阶段 B：第一轮 LoRA 微调 U 矩阵（冻结 V）
python utils/LoRA.py \
    --prune_model jeffwan_llama_7b_hf_whitening_only_0.8.pt \
    --data_path yahma/alpaca-cleaned \
    --output_dir ./first_half \
    --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj \
    --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64

# 合并第一轮 LoRA 权重并评估
python SVDLLM.py --step 4 --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt \
    --lora ./first_half

# 阶段 C：第二轮 LoRA 微调 V 矩阵（冻结 U）
python utils/LoRA.py \
    --prune_model ./first_half/merge.pt \
    --data_path yahma/alpaca-cleaned \
    --output_dir ./second_half \
    --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj \
    --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64

# 合并第二轮 LoRA 权重并评估
python SVDLLM.py --step 4 --model_path ./first_half/merge.pt \
    --lora ./second_half
```


---

## 五、各模块详解

### 5.1 SVDLLM.py —— 主入口

核心函数：

| 函数 | 作用 | 关键点 |
|------|------|--------|
| `profle_svdllm()` | 收集激活协方差 → Cholesky 分解 | 需要模型常驻显存 |
| `profle_svdllm_low_resource()` | 低资源版白化 | 逐层加载，显存需求 < 15G（LLaMA-7B） |
| `whitening()` | 执行白化 + SVD 截断 | 替换原模型层为 SVD 版模块 |
| `whitening_local_update()` | 白化 + 闭式解更新 | 结合校准数据更新 U |
| `local_update` 类 | 封装单层的 SVD 截断和更新 | `add_batch_update_u` 收集校准数据 |
| `fasterprune()` | 输出最终 U, V | 返回 `appendU`, `appendV` |

**关键代码解读 —— `local_update` 类的 `add_batch_update_u`**：

```python
def add_batch_update_u(self, inp, out):
    inp_flat = inp.reshape(-1, inp_dim)
    out_flat = out.reshape(-1, out_dim)
    # 用最小二乘求解新的 U: min ||out - x @ V^T @ Σ @ U^T||
    # 在给定 V, Σ 不变的情况下更新 U
    x = inp_flat @ V^T @ Σ    # 中间特征
    self.updated_uT = torch.linalg.lstsq(x, out_flat).solution
```

**核心思想**：在 V 和 Σ 固定时，最优 U 可以通过**最小二乘**闭式解得到（而非梯度下降）。

### 5.2 component/ —— 压缩版模型组件

SVD 压缩后，原来的单层 Linear 被拆分为两个 Linear：

```
原始: y = Wx                          (W: m×n)
压缩后: y = W_u @ (W_v @ x)           (W_u: m×k, W_v: k×n)
其中 k = int(m * n * ratio / (m + n))
```

各模型系列的替换关系：

| 模型 | 原始模块 | SVD 替换模块 | 文件 |
|------|---------|-------------|------|
| LLaMA | `LlamaAttention` | `SVD_LlamaAttention` | `svd_llama.py` |
| LLaMA | `LlamaMLP` | `SVD_LlamaMLP` | `svd_llama.py` |
| Mistral | `MistralAttention` | `SVD_MistralAttention` | `svd_mistral.py` |
| Mistral | `MistralMLP` | `SVD_MistralMLP` | `svd_mistral.py` |
| OPT | `OPTDecoderLayer` | `SVDOPTDecoderLayer` | `svd_opt.py` |
| OPT | `OPTAttention` | `SVDOPTAttention` | `svd_opt.py` |

**Attention 的 SVD 压缩方式**：

- Q、K、V、O 四个投影层各自独立做 SVD 分解
- 低秩秩 `low_rank = int(hidden_size * ratio / 2)`（Attention 是方阵）
- 前向传播变为 `W_u @ (W_v @ x)` 的形式

**MLP 的 SVD 压缩方式**：

- Gate、Up、Down 三个投影各自独立 SVD
- 低秩秩 `low_rank = int(intermediate * hidden * ratio / (intermediate + hidden))`
- 前向传播：`up = up_u @ up_v(x)`, `gate = gate_u @ gate_v(x)`，`down = down_u @ down_v(act(gate)*up)`

### 5.3 utils/LoRA.py —— 顺序低秩近似微调

**关键：LoRA 的目标模块**是 SVD 分解后的 U、V 矩阵，而非原始 Linear：

```
--lora_target_modules q_u_proj,k_u_proj,... # 第一轮：微调 U
--lora_target_modules q_v_proj,k_v_proj,... # 第二轮：微调 V
```

**默认参数**：
- `lora_r=8`（LoRA 秩）
- `num_epochs=3`
- `learning_rate=1e-4`
- `batch_size=64`
- 数据集：`yahma/alpaca-cleaned`（50K 样本）

**注意**：
- `LoRA.py` 中 `apply_lora()` 函数支持传参直接调用，也可作为脚本运行
- 使用本地 PEFT 库（`utils/peft/`），不需要额外安装 `peft` 包

### 5.4 evaluater.py —— 评估函数

两个主要评估函数：

| 函数 | 评估内容 | 测量指标 |
|------|---------|---------|
| `ppl_eval()` | 语言建模能力 | 困惑度 (PPL) |
| `ppl_eval_large()` | 大规模模型 PPL | 逐层计算，适合 65B+ |
| `eff_eval()` | 推理效率 | 吞吐量 (tokens/sec)、显存占用 |

**PPL 计算说明**：
- 使用 CrossEntropyLoss(reduction="none") 逐 token 计算 loss
- 对全部 token 的 loss 取均值后取 exp → PPL
- 支持的测试集：wikitext2、ptb、c4

**效率评估说明**：
- 固定 prompt 长度进行自回归生成
- 测量生成阶段耗时
- 记录峰值显存（总显存、权重显存、激活显存）

### 5.5 gptq/ —— GPTQ 量化集成

- `gptq.py`：GPTQ 算法的核心实现（Hessian 矩阵、Cholesky 求逆、逐块量化）
- `quant.py`：量化器（对称/非对称量化、MSE 最优尺度搜索）
- `quant_llama.py`：对 SVD 压缩后的模型进一步做 GPTQ 量化

**注意**：quant_llama.py 中 `true_sequential` 模式下的子模块匹配（第 69-73 行）是专门为 SVD 版模块设计的：

```python
sequential = [
    ['self_attn.k_u_proj','self_attn.k_v_proj', 'self_attn.v_u_proj', 'self_attn.v_v_proj', 'self_attn.q_u_proj', 'self_attn.q_v_proj'],
    ['self_attn.o_u_proj', 'self_attn.o_v_proj'],
    ['mlp.up_u_proj', 'mlp.up_v_proj', 'mlp.gate_u_proj', 'mlp.gate_v_proj'],
    ['mlp.down_u_proj', 'mlp.down_v_proj']
]
```

### 5.6 utils/model_utils.py —— 模型加载工具

关键函数：
- `get_model_from_huggingface(model_id)`：从 HuggingFace 加载模型和 tokenizer
- `get_model_from_local(model_id)`：从本地 `.pt` 文件加载压缩后的模型
- `find_layers(module)`：递归查找模块中所有 Linear 层（支持 Conv2d）

### 5.7 utils/data_utils.py —— 数据加载工具

支持的数据集：
- **wikitext2**（默认）：HuggingFace datasets
- **ptb**：Penn Treebank
- **c4**：需手动下载（原始链接失效，见 README 补充链接）

关键函数：
- `get_calib_train_data()`：用于白化的校准数据（返回 dict 列表）
- `get_loaders()`：用于参数更新的校准数据（返回 (inp, tar) 元组列表）
- `get_test_data()`：用于评估的测试数据（返回 DataLoader）

---

## 六、支持哪些模型

通过代码分析，支持的模型架构：

| 模型系列 | 模型类型参数 `--model` 示例 | 是否支持低资源模式 |
|---------|---------------------------|-------------------|
| LLaMA | `jeffwan/llama-7b-hf` | ✅ |
| LLaMA 2 | `meta-llama/Llama-2-7b-hf` | ✅ |
| Vicuna | `lmsys/vicuna-7b-v1.5` | ✅ |
| Mistral | `mistralai/Mistral-7B-v0.1` | ✅ |
| OPT | `facebook/opt-6.7b` | ✅ |

模型命名中只要包含 `"llama"`、`"mistral"`、`"vicuna"` 或 `"opt"` 即可自动匹配对应组件。

---

## 七、压缩比计算说明

代码中 `args.ratio` 的处理（`SVDLLM.py` 第 523 行）：

```python
args.ratio = 1 - args.ratio
```

即命令行传入 `--ratio 0.2` 表示**保留 20% 的参数**，内部转换为 `0.8` 作为截断比例。

对于 Attention 方阵（hidden_size × hidden_size）：
```
low_rank = int(hidden_size * ratio / 2)
```

对于 MLP 非方阵（hidden_size × intermediate_size）：
```
low_rank = int(intermediate * hidden * ratio / (intermediate + hidden))
```

---

## 八、已知问题与注意事项

### ⚠️ 必须注意

1. **Transformers 版本锁死 4.35.2**：项目组件继承了 Transformers 4.35.2 的模型源码结构。新版 Transformers 可能改变内部命名和接口，导致加载失败。

2. **C4 数据集需要手动下载**：README 说明原始 c4 下载链接已失效，需从 [Google Drive](https://drive.google.com/drive/folders/123Id1MkZVsKySGy_sMO4RgiJKrtPcvUp?usp=sharing) 下载 `c4-train.json` 和 `c4-validation.json` 放入 `utils/` 目录。

3. **Python 版本**：官方要求 Python 3.9，新版 Python 可能因依赖兼容性问题出错。

4. **模型全部以 fp32 保存**：压缩后的模型保存为 fp32 格式。`SVDLLM.py` 中 Step 2 的 `model.float()` 是必要的。

5. **权重保存文件命名**：自动生成的文件名包含模型名、数据集、样本数、seed 等信息，注意区分不同实验。

### ⚠️ 值得注意

6. **逐层处理**：`profle_svdllm_low_resource` 和 `whitening_local_update` 都是逐层处理（layer-by-layer），这既是为了省显存，也是为了实现局部更新。

7. **`use_cache` 开关**：在压缩过程中会强制设置 `model.config.use_cache = False`，完成后恢复。因为压缩（特别是局部更新）不需要 KV cache。

8. **Hook 机制**：大量使用了 PyTorch `register_forward_hook` 来捕获中间激活，注意及时 `remove()` 避免内存泄漏。

9. **矩阵求逆的稳定性**：Cholesky 分解可能失败（矩阵非正定），代码有回退机制——加小量 `1e-6 * I` 到对角线再试。

10. **顺序微调的两轮解耦**：U 和 V 的 LoRA 微调是分离的两轮，这与通常的 LoRA 微调不同。脚本中 `lora_target_modules` 在第一轮只包含 `*_u_proj`，第二轮只包含 `*_v_proj`。

11. **LoRA 微调使用 int8 训练**：`prepare_model_for_int8_training(model)` 将模型转为 int8 以节省显存。

12. **merge.pt 文件**：LoRA 微调后，`SVDLLM.py` 的 Step 4 在加载 LoRA 权重时会自动合并并保存 `merge.pt`，作为下一轮微调的输入。

### 显存需求估计

| 模型 | 压缩模式 | 最低显存 |
|------|---------|---------|
| LLaMA-7B | 标准模式 | ~25 GB |
| LLaMA-7B | 低资源模式 | ~15 GB |
| LLaMA-13B | 标准模式 | ~40 GB |
| LLaMA-30B | 标准模式 | ~80 GB |

---

## 九、代码架构图

```
┌─────────────────────────────────────────────────────────┐
│                    SVDLLM.py (主入口)                     │
│                                                         │
│  Step 1: profle_svdllm() → whitening()                  │
│          ┌──────────┐    ┌───────────┐                  │
│          │ 收集激活  │ →  │ Cholesky  │ → SVD → 截断    │
│          │ XX^T 累加 │    │ 分解得 S  │                  │
│          └──────────┘    └───────────┘                  │
│                                                         │
│  Step 2: whitening_local_update()                       │
│          ┌─── SVD截断 ──┐ → ┌── 闭式解更新U ──┐        │
│                                                         │
│  Step 3: SVD截断 + 更新（无白化）                        │
│                                                         │
│  Step 4: ppl_eval()  — PPL评估                          │
│  Step 5: eff_eval()  — 效率评估                         │
└─────────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│  component/      │  │  utils/LoRA.py   │
│  svd_llama.py    │  │  顺序低秩微调    │
│  svd_mistral.py  │  │  第一轮: 微调 U  │
│  svd_opt.py      │  │  第二轮: 微调 V  │
└──────────────────┘  └──────────────────┘
         │                    │
         ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│  evaluater.py    │  │  quant_llama.py  │
│  PPL + 吞吐量    │  │  + GPTQ 量化     │
└──────────────────┘  └──────────────────┘
```


