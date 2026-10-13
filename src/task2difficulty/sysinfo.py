"""环境采集：生成论文"运行环境表"（四·（一）要求）与显存预算依据。

设计说明
--------
本模块只做只读探测，不依赖 torch 之外的第三方库；torch 导入失败时也能降级运行，
因为论文的"运行环境表"必须在最终成稿的同一个环境里采一次并原样写入正文。
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GPUInfo:
    name: str = ""
    total_vram_gb: float = 0.0
    free_vram_gb_at_detect: float = 0.0
    compute_capability: str = ""
    torch_arch_list: list[str] = field(default_factory=list)
    driver_version: str = ""
    cuda_umd_version: str = ""


@dataclass
class EnvReport:
    os_caption: str = ""
    os_version: str = ""
    python_version: str = ""
    cpu_name: str = ""
    cpu_physical_cores: int = 0
    cpu_logical_cores: int = 0
    total_ram_gb: float = 0.0
    torch_version: str = ""
    torch_cuda_version: str = ""
    cudnn_version: str = ""
    cuda_available: bool = False
    gpus: list[GPUInfo] = field(default_factory=list)
    packages: dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """输出可直接粘进论文的运行环境表。"""
        lines: list[str] = []
        lines.append("| 项目 | 配置 |")
        lines.append("|---|---|")
        lines.append(f"| 操作系统 | {self.os_caption} ({self.os_version}) |")
        lines.append(f"| CPU | {self.cpu_name}，{self.cpu_physical_cores} 核 "
                     f"{self.cpu_logical_cores} 线程 |")
        lines.append(f"| 内存 | {self.total_ram_gb:.1f} GB |")
        lines.append(f"| Python | {self.python_version} |")
        lines.append(f"| 深度学习框架 | PyTorch {self.torch_version} "
                     f"(CUDA {self.torch_cuda_version}, cuDNN {self.cudnn_version}) |")
        if self.gpus:
            for i, g in enumerate(self.gpus):
                lines.append(
                    f"| GPU{i} | {g.name}，显存 {g.total_vram_gb:.1f} GB，"
                    f"算力 {g.compute_capability}，驱动 {g.driver_version} |"
                )
        else:
            lines.append("| GPU | 无（纯 CPU 训练） |")
        key_pkgs = ["numpy", "pandas", "scikit-learn", "lightgbm", "scipy",
                    "transformers", "tokenizers", "matplotlib", "seaborn"]
        present = {k: v for k, v in self.packages.items() if k in key_pkgs}
        if present:
            pkgs = "；".join(f"{k} {v}" for k, v in present.items())
            lines.append(f"| 主要依赖 | {pkgs} |")
        return "\n".join(lines)


def _pkg_versions() -> dict[str, str]:
    from importlib import metadata

    names = [
        "numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "scipy",
        "torch", "transformers", "tokenizers", "datasets", "matplotlib",
        "seaborn", "beautifulsoup4", "lxml", "requests", "tqdm", "mord",
    ]
    out: dict[str, str] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:
            out[n] = "NOT INSTALLED"
    return out


def _nvidia_smi() -> tuple[str, str]:
    """返回 (driver_version, cuda_umd_version)；失败则空串。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "", ""
    try:
        cp = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        driver = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else ""
    except Exception:
        driver = ""
    cuda = ""
    try:
        cp = subprocess.run([exe], capture_output=True, text=True, timeout=20)
        for ln in cp.stdout.splitlines():
            if "CUDA UMD Version" in ln:
                cuda = ln.split("CUDA UMD Version")[-1].strip()
                break
    except Exception:
        pass
    return driver, cuda


def _run_ps(command: str, timeout: int = 30) -> str:
    """执行 PowerShell 命令并返回 stdout。

    ⚠ 必须显式指定 `encoding="utf-8"` 与 `errors="replace"`：
    Windows 上 PowerShell 的输出受控制台代码页影响（本机为 GBK），
    而系统的首选编码（locale.getpreferredencoding）可能与子进程不一致。
    实测踩坑：不指定编码时 `subprocess` 的解码线程抛 UnicodeDecodeError，
    **stdout 直接返回 None**，于是 `.strip()` 抛异常、整段环境信息静默回退——
    表现为"内存 0.0 GB、操作系统变成一串原始 platform 字符串"，
    而调用方完全看不出是采集失败（环境表会被原样写进论文）。
    """
    try:
        cp = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        return (cp.stdout or "").strip()
    except Exception:
        return ""


def collect() -> EnvReport:
    rep = EnvReport()
    rep.python_version = platform.python_version()
    rep.packages = _pkg_versions()

    if sys.platform == "win32":
        # 只用标准库 + PowerShell，避免依赖 wmi
        raw = _run_ps(
            "Get-CimInstance Win32_OperatingSystem | "
            "Select-Object Caption,Version,TotalVisibleMemorySize | ConvertTo-Json -Compress")
        try:
            d = json.loads(raw or "{}")
            rep.os_caption = d.get("Caption", "") or platform.platform()
            rep.os_version = str(d.get("Version", ""))
            rep.total_ram_gb = float(d.get("TotalVisibleMemorySize", 0) or 0) / 1024**2
        except Exception:
            rep.os_caption = rep.os_caption or platform.platform()

        raw = _run_ps(
            "Get-CimInstance Win32_Processor | "
            "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress")
        try:
            d = json.loads(raw or "{}")
            if isinstance(d, list):
                d = d[0] if d else {}
            rep.cpu_name = str(d.get("Name", "")).strip()
            rep.cpu_physical_cores = int(d.get("NumberOfCores", 0) or 0)
            rep.cpu_logical_cores = int(d.get("NumberOfLogicalProcessors", 0) or 0)
        except Exception:
            pass
    else:
        rep.os_caption = platform.platform()
        rep.cpu_name = platform.processor()
        rep.cpu_logical_cores = os.cpu_count() or 0

    # 内存兜底：PowerShell 取值失败时用 sysconf / 物理内存近似
    if rep.total_ram_gb <= 0:
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                rep.total_ram_gb = st.ullTotalPhys / 1024**3
        except Exception:
            pass

    driver, cuda_umd = _nvidia_smi()

    try:
        import torch
        rep.torch_version = torch.__version__
        rep.torch_cuda_version = torch.version.cuda or ""
        try:
            rep.cudnn_version = str(torch.backends.cudnn.version() or "")
        except Exception:
            rep.cudnn_version = ""
        rep.cuda_available = bool(torch.cuda.is_available())
        if rep.cuda_available:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                try:
                    free, _total = torch.cuda.mem_get_info(i)
                    free_gb = free / 1024**3
                except Exception:
                    free_gb = 0.0
                rep.gpus.append(GPUInfo(
                    name=p.name,
                    total_vram_gb=p.total_memory / 1024**3,
                    free_vram_gb_at_detect=free_gb,
                    compute_capability=f"sm_{p.major}{p.minor}",
                    torch_arch_list=list(torch.cuda.get_arch_list()),
                    driver_version=driver,
                    cuda_umd_version=cuda_umd,
                ))
    except Exception as e:  # torch 未装也要能出报告
        rep.torch_version = f"IMPORT FAILED: {type(e).__name__}"

    return rep


def assert_gpu_supported() -> tuple[bool, str]:
    """检查 GPU 算力是否在 torch 编译的架构列表内。

    这是 RTX 50 系（Blackwell, sm_120）最容易踩的坑：旧版 torch 编译时不含 sm_120，
    会静默回落到 CPU 或直接报 kernel 不可用。
    """
    try:
        import torch
    except Exception as e:
        return False, f"torch 不可用: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA 不可用（torch.cuda.is_available() == False）"
    p = torch.cuda.get_device_properties(0)
    cc = f"sm_{p.major}{p.minor}"
    archs = torch.cuda.get_arch_list()
    if cc not in archs:
        return False, (f"算力 {cc} 不在 torch 编译架构列表 {archs} 中，"
                       f"GPU 内核不可用，需升级 PyTorch 到较新 CUDA 版本")
    # 真实做一次前向，确认内核能跑（比只比对字符串更硬）
    try:
        x = torch.randn(64, 64, device="cuda")
        _ = (x @ x).sum().item()
    except Exception as e:
        return False, f"{cc} 在架构列表内但计算失败: {type(e).__name__}: {e}"
    return True, f"{p.name} ({cc}) 可用，显存 {p.total_memory/1024**3:.2f} GB"


def acceleration_table() -> list[tuple[str, str, str]]:
    """返回（环节, 设备, 说明）三元组，供环境报告与正文四·（一）引用。

    为什么要把"哪些不在 GPU 上"也列出来：只说"已启用 GPU 加速"会让读者
    以为全流程都在 GPU 上；实测受限的恰恰是随机森林（sklearn 无 GPU 实现）
    与 LightGBM（PyPI 轮子未编译 GPU 支持），如实列出才是可复现的写法。
    """
    from . import accel as A

    gpu = A.device_label() if A.gpu_available() else "CPU（无可用 GPU）"
    rows = [
        ("BERT / 融合模型", gpu, "主开销；fp16 + AMP，TF32 与 cuDNN benchmark 已开"),
        ("MLP / TextCNN", gpu, "结构化/文本单通道，单次秒级至数十秒"),
        ("有序回归（累积链接模型）", gpu,
         "torch 稀疏-稠密乘 + autograd 解析梯度，单次约 8 秒"),
        ("SVR（核方法）", gpu,
         "文本随机化 SVD 与 Nyström RBF 展开在 GPU；末段线性 SVR 在 CPU"),
        ("岭回归", "CPU", "稀疏最小二乘，单次 0.7 秒，上 GPU 无收益"),
        ("随机森林", "CPU", "sklearn 无 GPU 实现（RAPIDS/cuML 不支持 Windows）"),
        ("LightGBM", "CPU",
         "PyPI 轮子未含 GPU 支持，实测报 'GPU Tree Learner was not enabled in this build'"),
        ("TF-IDF 向量化", "CPU", "词表统计为 CPU 密集；每条件只拟合一次并缓存"),
    ]
    return rows


if __name__ == "__main__":
    r = collect()
    print(json.dumps(asdict(r), ensure_ascii=False, indent=2))
    print()
    print(r.to_markdown())
    print()
    ok, msg = assert_gpu_supported()
    print(f"[GPU 自检] {'PASS' if ok else 'FAIL'}: {msg}")
    print()
    print("| 环节 | 设备 | 说明 |")
    print("|---|---|---|")
    for name, dev, note in acceleration_table():
        print(f"| {name} | {dev} | {note} |")
