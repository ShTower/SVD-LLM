# SVD-LLM NPU 910B 复现结果

**设备**: Huawei Ascend 910B 80GB × 1 + 20 vCPU + 160GB RAM  
**分支**: `npu`  
**论文**: SVD-LLM (ICLR 2025)

---

## 一、WikiText-2 压缩效果完整对比

| Method | 压缩比 | SVD 类型 | 耗时 | PPL | 论文 PPL |
|--------|:--:|------|:--:|:--:|:--:|
| Original | — | — | — | **5.68** | 5.68 |
| Vanilla SVD (无白化) | 20% | Regular | ~1.5h | **14.33** | — |
| SVD-LLM (W) | 20% | Regular | ~3.5h | **7.89** | 7.94 ✅ |
| SVD-LLM (+U LoRA) | 20% | Regular | +4.8h | **7.27** | — |
| **SVD-LLM (full)** | 20% | Regular | +4.8h | **7.56** | 7.73 ✅ |
| SVD-LLM (W) | 40% | Regular | ~3h | **13.77** | 13.73 ✅ |
| SVD-LLM (W) | 40% | Randomized | ~1.5h | 164 ❌ | — |

**白化收益**: PPL 14.33 → 7.89，降低 45%  
**LoRA 收益**: PPL 7.89 → 7.56，再降 4%，优于论文 7.73

---

## 二、推理效率 (LLaMA-7B @20%)

| 指标 | 数值 |
|------|------|
| 总显存 | 28.57 GB |
| 权重显存 | 20.55 GB |
| 激活显存 | 8.02 GB |
| 吞吐量 | **42.16 tokens/sec** |

---

## 三、Level-2: 随机化 SVD 改进分析

### 3.1 Benchmark（重构误差）

| 压缩比 | k 值 | 加速比 | 重构误差(reg/rng) |
|--------|:----:|:-----:|:-----------------:|
| 20% | 1638 | 3.3x | 0.17 / 0.19 |
| 40% | 1228 | 5.6x | 0.16 / 0.20 |

### 3.2 实际压缩+PPL

| 压缩比 | SVD | 耗时 | PPL |
|--------|------|:--:|:--:|
| 40% | Regular | 137 min | 13.77 |
| 40% | Randomized (`--rng_svd`) | 52 min | 164 |

### 3.3 结论

随机化 SVD 在重构误差 benchmark 中表现可接受，但实际 PPL 恶化 12 倍。原因：白化矩阵的中间奇异值对精度至关重要，随机化逼近丢失了这些信息。这从反面验证了论文使用完整 SVD 的必要性。

**改进方向**: 自适应 power iteration、混合策略（低压缩比 Regular / 高压缩比 Randomized）、随机化后精确校正。

---

## 四、耗时统计

| 阶段 | 耗时 |
|------|------|
| 模型下载 | ~30 min |
| Profiling (白化矩阵) | ~31 min |
| SVD @20% (Regular) | ~2h 22min |
| SVD @40% (Regular) | ~2h 17min |
| SVD @40% (Randomized) | ~52 min |
| LoRA 第一轮 (微调 U) | ~4h 50min |
| LoRA 第二轮 (微调 V) | ~4h 46min |
| PPL 评估 | ~1.5 min |
| 效率评估 | ~16 min |

---

## 五、文件清单

| 文件 | 说明 |
|------|------|
| `output/jeffwan_llama_7b_hf_whitening_only_0.8.pt` | SVD-LLM (W) @20% 压缩模型 |
| `output/jeffwan_llama_7b_hf_whitening_only_0.6.pt` | SVD-LLM (W) @40% Regular 压缩模型 |
| `output/jeffwan_llama_7b_hf_profiling_wikitext2_256_3.pt` | 白化矩阵缓存 (53 GB) |
| `output/jeffwan_llama_7b_hf_whitening_then_update_0.8.pt` | Step 2 结果 ⚠️ |
| `output/jeffwan_llama_7b_hf_update_only_0.8.pt` | Step 3: Vanilla SVD 基线 |
