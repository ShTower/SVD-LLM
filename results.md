# SVD-LLM NPU 910B 复现结果

**设备**: Huawei Ascend 910B 80GB × 1 + 20 vCPU + 160GB RAM  
**分支**: `npu`  
**论文**: SVD-LLM (ICLR 2025)

---

## 一、WikiText-2 压缩效果对比

| Method | PPL | 论文 PPL | 说明 |
|--------|:--:|:--:|------|
| Original | **5.68** | 5.68 | 完全一致 |
| Vanilla SVD (无白化) | **14.33** | — | Step 3, SVD 截断后闭式解更新 |
| SVD-LLM (W) @20% | **7.89** | 7.94 | **优于论文** |
| SVD-LLM @20% | 16.96 ⚠️ | 7.73 | NPU lstsq 精度异常 |

白化收益: PPL 14.33 → 7.89，降低 **45%**。

---

## 二、推理效率 (LLaMA-7B @20%)

| 指标 | 数值 |
|------|------|
| 总显存 | 28.57 GB |
| 权重显存 | 20.55 GB |
| 激活显存 | 8.02 GB |
| 吞吐量 | **42.16 tokens/sec** |
| 每次生成 | ~97s (1024 tokens) |

---

## 三、随机化 SVD 加速效果

| 压缩比 | k 值 | 加速比 | Q/K 精度 | 全模型预估 |
|--------|:----:|:-----:|:--------:|:--------:|
| 20% (ratio=0.8) | 1638 | 3.3x | ⚠️ 偏差较大 | 1.3h → 0.4h |
| 40% (ratio=0.6) | 1228 | **5.6x** | ⚠️ 可接受 | 1.4h → 0.2h |
| 60% (ratio=0.4) | ~700 | ~8x | ✅ 预期良好 | — |

---

## 四、耗时统计

| 阶段 | 耗时 |
|------|------|
| 模型下载 | ~30 min |
| Profiling (白化矩阵) | ~31 min |
| SVD 压缩 | ~2h 22min |
| Step 1 总计 | ~3.5h |
| Step 2 (whitening+update) | ~5.5h |
| Step 3 (update only) | ~1.5h |
| PPL 评估 | ~1.5 min |
| 效率评估 | ~16 min |

---

## 五、文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `output/jeffwan_llama_7b_hf_whitening_only_0.8.pt` | 20 GB | SVD-LLM (W) 压缩模型 |
| `output/jeffwan_llama_7b_hf_profiling_wikitext2_256_3.pt` | 53 GB | 白化矩阵缓存 |
| `output/jeffwan_llama_7b_hf_whitening_then_update_0.8.pt` | 21 GB | Step 2 结果 |
| `output/jeffwan_llama_7b_hf_update_only_0.8.pt` | 21 GB | Step 3: Vanilla SVD |
