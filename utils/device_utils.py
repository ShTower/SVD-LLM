"""
设备管理工具：统一 CPU / CUDA / NPU (Ascend) 接口。

用法:
    from utils.device_utils import get_device, sync_device, clear_cache, \
        safe_cholesky, safe_inv, safe_svd, safe_lstsq, safe_eigvalsh

    dev = get_device("auto")   # 或 "npu:0" / "cuda:0" / "cpu"
    model = model.to(dev)
"""

import torch
import os

_HAS_NPU = False
try:
    import torch_npu
    _HAS_NPU = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════
# 设备检测
# ═══════════════════════════════════════════════════════════

def detect_device():
    """按优先级自动检测可用设备: NPU > CUDA > CPU"""
    if _HAS_NPU and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def detect_device_str():
    """返回 'npu' / 'cuda' / 'cpu' 字符串"""
    if _HAS_NPU and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device(device_str=None):
    """解析设备字符串，None 或 'auto' 时自动检测"""
    if device_str is None or device_str.lower() == "auto":
        return detect_device()
    return torch.device(device_str)


def is_npu(device):
    """判断是否为 NPU 设备"""
    if isinstance(device, torch.device):
        return device.type == "npu" or "npu" in str(device)
    return "npu" in str(device)


# ═══════════════════════════════════════════════════════════
# 统一设备 API (替代 torch.cuda.*)
# ═══════════════════════════════════════════════════════════

def _resolve_device(device):
    """将 None 或字符串解析为 torch.device，用于统一设备 API。"""
    if device is None:
        return detect_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def sync_device(device=None):
    """同步设备（CUDA: synchronize, NPU: synchronize）"""
    dev = _resolve_device(device)
    if dev.type == "npu":
        torch.npu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def clear_cache(device=None):
    """清空设备缓存"""
    dev = _resolve_device(device)
    if dev.type == "npu":
        torch.npu.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def allocated_memory(device=None):
    """当前已分配显存 (bytes)"""
    dev = _resolve_device(device)
    if dev.type == "npu":
        return torch.npu.memory_allocated()
    elif dev.type == "cuda":
        return torch.cuda.memory_allocated()
    return 0


def max_allocated_memory(device=None):
    """峰值显存 (bytes)"""
    dev = _resolve_device(device)
    if dev.type == "npu":
        return torch.npu.max_memory_allocated()
    elif dev.type == "cuda":
        return torch.cuda.max_memory_allocated()
    return 0


def reset_peak_memory(device=None):
    """重置峰值显存统计"""
    dev = _resolve_device(device)
    if dev.type == "npu":
        torch.npu.reset_peak_memory_stats()
    elif dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


# ═══════════════════════════════════════════════════════════
# 安全线性代数算子（NPU 不支持时回退 CPU）
# ═══════════════════════════════════════════════════════════

def _cpu_fallback(x, op_fn):
    """
    对 NPU/CUDA 张量：先搬 CPU 计算，再搬回原设备。
    对 CPU 张量：直接计算。
    这样避免了 NPU 对 SVD/Cholesky 等高级 linalg 算子可能的不支持问题。
    """
    orig_device = x.device
    if orig_device.type in ("npu", "cuda"):
        x_cpu = x.cpu()
        result = op_fn(x_cpu)
        # result 可能是 tuple 或单个张量
        if isinstance(result, tuple):
            return tuple(r.to(orig_device) for r in result)
        return result.to(orig_device)
    return op_fn(x)


def safe_cholesky(x):
    """Cholesky 分解，NPU 回退 CPU"""
    return _cpu_fallback(x, lambda t: torch.linalg.cholesky(t))


def safe_inv(x):
    """矩阵求逆，NPU 回退 CPU"""
    return _cpu_fallback(x, lambda t: torch.linalg.inv(t))


def safe_svd(x, full_matrices=False):
    """SVD 分解，NPU 回退 CPU"""
    return _cpu_fallback(x, lambda t: torch.linalg.svd(t, full_matrices=full_matrices))


def safe_eigvalsh(x):
    """对称矩阵特征值，NPU 回退 CPU"""
    return _cpu_fallback(x, lambda t: torch.linalg.eigvalsh(t))


def safe_lstsq(A, B):
    """最小二乘求解，NPU 回退 CPU"""
    orig_device = A.device
    if orig_device.type in ("npu", "cuda"):
        A_cpu, B_cpu = A.cpu(), B.cpu()
        result = torch.linalg.lstsq(A_cpu, B_cpu)
        return type('LstsqResult', (), {
            'solution': result.solution.to(orig_device),
            'residuals': result.residuals.to(orig_device) if result.residuals is not None else None,
        })
    return torch.linalg.lstsq(A, B)


# ═══════════════════════════════════════════════════════════
# 信息输出
# ═══════════════════════════════════════════════════════════

def print_device_info():
    """打印当前设备信息"""
    dev = detect_device()
    print(f"[Device] Detected: {dev}")
    if is_npu(dev):
        print(f"[Device] NPU memory allocated: {allocated_memory(dev)/1024**3:.2f} GB")
    elif dev.type == "cuda":
        print(f"[Device] CUDA memory allocated: {allocated_memory(dev)/1024**3:.2f} GB")
    else:
        print("[Device] Running on CPU")
