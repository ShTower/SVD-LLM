"""
随机化 SVD 速度/精度 Benchmark（独立脚本，不修改原代码）

用法：
    python bench_rng_svd.py --model jeffwan/llama-7b-hf --ratio 0.8 --device cpu
"""

import argparse
import time
import torch
import numpy as np

def randomized_svd(A, k, n_oversamples=10, n_power_iter=2):
    """跟 device_utils.randomized_svd 一样的实现，脚本自包含"""
    m, n = A.shape
    p = k + n_oversamples
    Omega = torch.randn(n, p, dtype=A.dtype, device=A.device)
    Y = A @ Omega
    for _ in range(n_power_iter):
        Y = A @ (A.T @ Y)
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ A
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], S[:k], Vh[:k, :]


def benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="jeffwan/llama-7b-hf")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="SVDLLM 内部压缩比（0.8 表示保留 80%）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="测试几个矩阵")
    args = parser.parse_args()

    # 加载模型，提取 Linear 权重
    print(f"Loading {args.model} ...")
    from utils.model_utils import get_model_from_huggingface
    model, _ = get_model_from_huggingface(args.model)
    from utils.model_utils import find_layers

    layers = model.model.layers
    weights = []
    names = []
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name, module in subset.items():
            w = module.weight.data.float().clone()
            weights.append(w)
            names.append(f"layer{i}.{name}")

    print(f"Found {len(weights)} weight matrices")
    print(f"Testing first {args.n_samples} ...\n")

    regular_times = []
    rng_times = []
    errors = []

    for idx in range(min(args.n_samples, len(weights))):
        W = weights[idx]
        m, n = W.shape

        # 计算 SVDLLM 使用的 k
        k = int(m * n * args.ratio / (m + n))
        print(f"  [{idx}] {names[idx]:30s} shape=({m:5d},{n:5d}) k={k:4d}", end="")

        # --- Regular SVD ---
        t0 = time.time()
        U1, S1, Vh1 = torch.linalg.svd(W, full_matrices=False)
        t1 = time.time()

        # --- Randomized SVD ---
        t2 = time.time()
        U2, S2, Vh2 = randomized_svd(W, k, n_oversamples=10, n_power_iter=2)
        t3 = time.time()

        # --- 精度比较 ---
        # 重构误差: ||W - U*diag(S)*Vh|| / ||W||
        W_hat1 = (U1[:, :k] * S1[:k]) @ Vh1[:k, :]
        W_hat2 = (U2 * S2) @ Vh2
        err1 = torch.norm(W - W_hat1) / torch.norm(W)
        err2 = torch.norm(W - W_hat2) / torch.norm(W)

        regular_times.append(t1 - t0)
        rng_times.append(t3 - t2)
        errors.append((err1.item(), err2.item()))

        speedup = (t1 - t0) / (t3 - t2)
        print(f"  regular={t1-t0:5.1f}s  rng={t3-t2:5.1f}s  speedup={speedup:4.1f}x  "
              f"err(regular)={err1:.6f}  err(rng)={err2:.6f}", end="")
        if err2 < 1.2 * err1:
            print(" ✅")
        else:
            print(" ⚠️ 精度略低但可接受")

    # --- 汇总 ---
    print(f"\n{'='*60}")
    print(f"Summary ({args.n_samples} matrices, ratio={args.ratio})")
    print(f"  Regular SVD:  total={sum(regular_times):.0f}s  avg={np.mean(regular_times):.1f}s")
    print(f"  Randomized:   total={sum(rng_times):.0f}s  avg={np.mean(rng_times):.1f}s")
    print(f"  Overall speedup: {sum(regular_times)/sum(rng_times):.1f}x")
    print(f"  Avg reconstruction error: regular={np.mean([e[0] for e in errors]):.6f}  "
          f"rng={np.mean([e[1] for e in errors]):.6f}")

    # 估算全模型 SVD 时间
    est_regular = sum(regular_times) / len(regular_times) * len(weights) / 3600
    est_rng = sum(rng_times) / len(rng_times) * len(weights) / 3600
    print(f"  Estimated full model: regular={est_regular:.1f}h  rng={est_rng:.1f}h")


if __name__ == "__main__":
    benchmark()
