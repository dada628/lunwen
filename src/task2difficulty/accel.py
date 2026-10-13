"""GPU 加速的统一入口。

全项目"能上 GPU 的都上 GPU"这条原则落在这个模块里，避免每个模型各写一套。
它集中四件事：

1. torch 全局开关：TF32、cuDNN benchmark、显存分配策略；
2. scipy 稀疏矩阵 → torch 稀疏张量（一次搬运，之后全在显存里算）；
3. 两个 GPU 原语：随机化截断 SVD、RBF Nyström 特征映射；
4. 设备信息汇报，供 sysinfo 与实验日志记录"这次实验跑在什么硬件上"。

为什么必须有这两个原语
----------------------
传统侧有两个模型的接口口径要求文本通道（`tfidf+struct`），但在 8864×20045
的高维稀疏表示上它们都跑不动，且原因不是调参能解决的：

- **SVR（RBF 核）**：libsvm 要先算全核矩阵再做 QP，实测 2.3 小时未收敛。
  高维稀疏下 RBF 核本身还会退化（距离集中），全维核矩阵没有统计意义。
- **有序回归（累积链接模型）**：参数维度等于特征维度，若不提供解析梯度，
  L-BFGS 的有限差分每步要 20046 次函数求值，实测 >300 秒未完成。

两个原语都只做**训练集上拟合的常量变换**：SVD 的投影方向、Nyström 的地标点
都取自训练集，再套用到验证/测试集，因此不构成泄漏，可以写进正文的特征口径。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# ---------------------------------------------------------------- 全局开关

_CONFIGURED = False
_CONFIG_REPORT: dict[str, Any] = {}


def configure_torch() -> dict[str, Any]:
    """打开 torch 的加速开关。幂等，可重复调用。

    - `set_float32_matmul_precision("high")`：在 sm_120 上启用 TF32，
      矩阵乘与卷积走张量核，深度模型前向/反向免费提速；
    - `cudnn.benchmark = True`：输入形状固定（BERT 的 max_length 恒定），
      让 cuDNN 自动挑最快卷积算法；
    - `PYTORCH_ALLOC_CONF=expandable_segments:True`：8 GB 显存下
      反复建/销不同形状的中间张量容易碎片化，开可扩展段能显著减少 OOM。
      （torch 2.9 起旧名 `PYTORCH_CUDA_ALLOC_CONF` 已弃用，会打警告。）

    注意必须在**第一次 CUDA 调用之前**设置分配器环境变量。
    """
    global _CONFIGURED, _CONFIG_REPORT
    if _CONFIGURED:
        return dict(_CONFIG_REPORT)

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    rep: dict[str, Any] = {}
    try:
        import torch

        torch.set_float32_matmul_precision("high")
        rep["matmul_precision"] = torch.get_float32_matmul_precision()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            rep["cudnn_benchmark"] = True
            rep["tf32"] = True
        else:
            rep["cudnn_benchmark"] = False
            rep["tf32"] = False
    except Exception as exc:                      # pragma: no cover
        rep["error"] = f"{type(exc).__name__}: {exc}"
    rep["alloc_conf"] = os.environ.get("PYTORCH_ALLOC_CONF", "")
    _CONFIGURED = True
    _CONFIG_REPORT = rep
    return dict(rep)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def get_device(prefer: str = "cuda"):
    """返回可用设备：优先 CUDA，失败则回 CPU（CPU 路径结果一致，只是慢）。"""
    configure_torch()
    import torch

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_label(device=None) -> str:
    """给日志/结果表用的设备短标签，例如 'cuda:NVIDIA GeForce RTX 5070 Laptop GPU'。"""
    if device is None:
        device = get_device()
    if getattr(device, "type", "cpu") != "cuda":
        return str(device)
    import torch

    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{torch.cuda.get_device_name(idx)}"
    except Exception:                             # pragma: no cover
        return "cuda"


def gpu_report() -> dict[str, Any]:
    """硬件与加速配置快照，写进实验元数据（复现性要求）。"""
    rep: dict[str, Any] = {"configured": _CONFIG_REPORT or configure_torch()}
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            rep.update({
                "gpu": p.name,
                "compute_capability": f"{p.major}.{p.minor}",
                "vram_gb": round(p.total_memory / 1024 ** 3, 2),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "bf16": bool(torch.cuda.is_bf16_supported()),
            })
            free, total = torch.cuda.mem_get_info()
            rep["vram_free_gb"] = round(free / 1024 ** 3, 2)
        else:
            rep["gpu"] = None
            rep["torch"] = torch.__version__
    except Exception as exc:                      # pragma: no cover
        rep["gpu"] = None
        rep["error"] = f"{type(exc).__name__}: {exc}"
    return rep


# ---------------------------------------------------------------- 稀疏搬运


def to_torch_sparse(X, device=None, dtype=None):
    """scipy 稀疏矩阵 → torch 稀疏 CSR 张量（默认搬到 GPU）。

    为什么用 CSR 而不是 COO：`torch.sparse.mm(csr, dense)` 在 CUDA 上走
    稀疏-稠密乘的专用内核，实测 (8864×20045, nnz=2.1M) 做一次
    "A @ Ω(k=256)" 只要 30 毫秒，COO 的同类操作用时长一个量级。
    """
    import torch
    from scipy import sparse

    configure_torch()
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = torch.float32

    if sparse.issparse(X):
        A = X.tocsr()
        return torch.sparse_csr_tensor(
            torch.from_numpy(np.ascontiguousarray(A.indptr, dtype=np.int64)),
            torch.from_numpy(np.ascontiguousarray(A.indices, dtype=np.int64)),
            torch.from_numpy(np.ascontiguousarray(A.data, dtype=np.float32)),
            size=tuple(A.shape), device=device, dtype=dtype,
        )
    arr = np.asarray(X, dtype=np.float32)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def to_torch_dense(X, device=None, dtype=None):
    import torch

    configure_torch()
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = torch.float32
    from scipy import sparse

    if sparse.issparse(X):
        X = np.asarray(X.todense(), dtype=np.float32)
    return torch.as_tensor(np.asarray(X, dtype=np.float32), device=device, dtype=dtype)


def _as_dense_np(X) -> np.ndarray:
    from scipy import sparse

    if sparse.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    if hasattr(X, "to_numpy"):
        return X.to_numpy(dtype=np.float64, na_value=np.nan)
    return np.asarray(X, dtype=np.float64)


# ---------------------------------------------------------------- 随机化 SVD


class RandomizedSvd:
    """训练集上拟合的随机化截断 SVD，全部矩阵乘在 GPU 上做。

    为什么不用 `sklearn.TruncatedSVD`：它只有 CPU 实现，在 8864×20045 上
    单次拟合要几十秒，而实验链里每个条件都要拟合一次。随机化 SVD 的
    计算主体是两三次稀疏-稠密乘，正好是 GPU 的强项（实测单个条件 <0.5 秒）。

    算法（Halko et al. 2011 的 randomized range finder）：
        Y = A Ω,  Q = qr(Y)
        B = Qᵀ A,  B = U_b Σ Vᵀ  （k×d 的小矩阵做精确 SVD）
    为提高精度做 `n_iter` 次幂迭代（A ← A Aᵀ A 的等价形式）。

    ⚠ 只允许在**训练集**上 fit：投影方向若用了验证/测试集就是泄漏。
    """

    def __init__(self, k: int = 200, n_iter: int = 2, seed: int = 42,
                 device=None) -> None:
        self.k = int(k)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.device = device
        self.components_: np.ndarray | None = None      # (k, d)
        self.singular_values_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None            # 未使用，保留接口一致性

    def fit(self, X) -> "RandomizedSvd":
        import torch

        device = self.device if self.device is not None else get_device()
        n, d = X.shape
        k = min(self.k, min(n, d))
        A = to_torch_sparse(X, device=device)
        g = torch.Generator(device=device).manual_seed(self.seed)
        Omega = torch.randn(d, k, device=device, generator=g, dtype=torch.float32)

        Y = torch.sparse.mm(A, Omega)
        # 幂迭代：Y ← A (Aᵀ Y)，把奇异值谱衰减压平，提高尾部精度
        for _ in range(self.n_iter):
            Y, _ = torch.linalg.qr(Y)
            Y = torch.sparse.mm(A, torch.sparse.mm(A.transpose(0, 1), Y))
        Q, _ = torch.linalg.qr(Y)

        # B = Qᵀ A（k×d）。A 是稀疏的，转置后用稀疏-稠密乘一次算出 Bᵀ。
        Bt = torch.sparse.mm(A.transpose(0, 1), Q)          # d×k
        B = Bt.transpose(0, 1).to(torch.float64)            # k×d
        # k×k 的小矩阵做精确特征分解，比直接对 k×d 做 SVD 稳定
        BtB = B @ B.transpose(0, 1)
        evals, evecs = torch.linalg.eigh(BtB)
        order = torch.argsort(evals, descending=True)
        evals = evals[order].clamp_min(0.0)
        evecs = evecs[:, order]
        sing = torch.sqrt(evals)
        keep = sing > 1e-10
        sing = sing[keep]
        evecs = evecs[:, keep]
        V = (B.transpose(0, 1) @ evecs) / sing.unsqueeze(0)  # d×k
        self.components_ = V.transpose(0, 1).cpu().numpy().astype(np.float64)
        self.singular_values_ = sing.cpu().numpy().astype(np.float64)
        return self

    def transform(self, X) -> np.ndarray:
        """把（稀疏）特征矩阵投影到 SVD 子空间。可用 GPU 分批做。"""
        import torch

        device = self.device if self.device is not None else get_device()
        V = torch.as_tensor(self.components_.T.astype(np.float32), device=device)
        A = to_torch_sparse(X, device=device)
        Z = torch.sparse.mm(A, V)
        return Z.cpu().numpy().astype(np.float64)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------- Nyström RBF


class NystroemRbf:
    """RBF 核的低秩显式展开（Nyström 方法），地标点取自训练集。

    为什么 SVR 需要它：libsvm 的核 SVR 在 n≈8900 时要解一个 n×n 的 QP，
    实测 2.3 小时不收敛；而 Nyström 把 RBF 核写成
    `z(x) = exp(-γ‖x − l‖²)` 的 m 维显式特征（m≪n），
    于是核 SVR 退化成**线性** SVR 在 m 维特征上的标准问题，
    求解成本从 O(n²)–O(n³) 降到 O(nm²)，而且与全维核 SVR 同族
    （都是"核方法、非线性、非集成"这一格）。

    ⚠ 地标点只在训练集里抽，γ 也只用训练集估计，不引入泄漏。
    """

    def __init__(self, n_components: int = 800, gamma: float | str = "scale",
                 seed: int = 42, device=None) -> None:
        self.n_components = int(n_components)
        self.gamma = gamma
        self.seed = int(seed)
        self.device = device
        self.landmarks_: np.ndarray | None = None
        self.gamma_: float = float("nan")

    def _resolve_gamma(self, X) -> float:
        if isinstance(self.gamma, str):
            if self.gamma != "scale":
                raise ValueError(f"未知 gamma 口径: {self.gamma!r}")
            Xd = _as_dense_np(X)
            var = float(np.var(Xd))
            d = max(Xd.shape[1], 1)
            self.gamma_ = 1.0 / (d * var) if var > 0 else 1.0
        else:
            self.gamma_ = float(self.gamma)
        return self.gamma_

    def fit(self, X) -> "NystroemRbf":
        self._resolve_gamma(X)
        Xd = _as_dense_np(X)
        n = len(Xd)
        m = min(self.n_components, n)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(n, size=m, replace=False)
        idx.sort()
        self.landmarks_ = Xd[idx]
        return self

    def _kernel_block(self, X, block: int = 4096) -> np.ndarray:
        """K(X, L) = exp(-γ(‖x‖² + ‖l‖² − 2xᵀl))，分块避免显存峰值。"""
        import torch

        device = self.device if self.device is not None else get_device()
        L = torch.as_tensor(self.landmarks_.astype(np.float32), device=device)
        ln = (L ** 2).sum(dim=1)
        Xd = _as_dense_np(X)
        out = np.empty((len(Xd), len(self.landmarks_)), dtype=np.float64)
        for s in range(0, len(Xd), block):
            xb = torch.as_tensor(Xd[s:s + block].astype(np.float32), device=device)
            d2 = (xb ** 2).sum(dim=1, keepdim=True) + ln.unsqueeze(0) - 2.0 * (xb @ L.T)
            d2 = d2.clamp_min_(0.0)
            out[s:s + block] = torch.exp(-self.gamma_ * d2).cpu().numpy().astype(np.float64)
        return out

    def transform(self, X) -> np.ndarray:
        return self._kernel_block(X)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
