from __future__ import annotations

import gc
import os
from typing import Callable

LogFn = Callable[[str], None]


def resolve_torch_device(requested: str = "auto") -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def cuda_total_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
    except Exception:
        return None


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    needles = (
        "cuda out of memory",
        "out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "failed to allocate",
    )
    return any(n in text for n in needles)


def release_accelerator_memory() -> None:
    """Best-effort release of Python/PyTorch accelerator allocations."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()


def memory_snapshot() -> str:
    parts: list[str] = []

    # psutil is optional. v0.6 installs it, but keep the app usable if an old venv
    # is reused before requirements are refreshed.
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / (1024 ** 3)
        vm = psutil.virtual_memory()
        parts.append(f"RAM process {rss:.2f} GB")
        parts.append(f"RAM system {vm.percent:.0f}% ({vm.available / (1024 ** 3):.1f} GB free)")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            parts.append(
                "VRAM "
                f"{(total-free)/(1024**3):.2f}/{total/(1024**3):.2f} GB used; "
                f"torch alloc {allocated/(1024**3):.2f}, reserved {reserved/(1024**3):.2f}"
            )
        else:
            parts.append("VRAM: CUDA unavailable")
    except Exception:
        pass

    return " | ".join(parts) if parts else "memory telemetry unavailable"


def log_memory(log: LogFn, label: str) -> None:
    log(f"[MEM] {label}: {memory_snapshot()}")


def resolve_diarization_window(base_seconds: int, requested_device: str = "auto") -> int:
    """Cap external pyannote chunk size on small GPUs.

    The diarization pipeline internally chunks audio too, but feeding it a huge
    in-memory waveform still raises host/GPU pressure. Small external chunks also
    make resume/cache behavior much friendlier.
    """
    device = resolve_torch_device(requested_device)
    if device != "cuda":
        return min(base_seconds, 300)
    total = cuda_total_gb()
    if total is None:
        return min(base_seconds, 300)
    if total <= 6.5:
        return min(base_seconds, 300)
    if total <= 10.0:
        return min(base_seconds, 600)
    return base_seconds


def resolve_demucs_segment(requested_device: str = "auto", model_name: str = "htdemucs") -> int | None:
    """Choose a conservative Demucs segment length for available VRAM.

    Hybrid Transformer Demucs models (htdemucs/htdemucs_ft) have a maximum
    supported segment length of about 7.8 seconds, so memory-safe values stay
    below that ceiling.
    """
    device = resolve_torch_device(requested_device)
    if device != "cuda":
        return None
    total = cuda_total_gb()
    if model_name.startswith("htdemucs"):
        if total is None or total <= 6.5:
            return 6
        if total <= 10.0:
            return 7
        return None
    if total is None:
        return 8
    if total <= 4.5:
        return 6
    if total <= 6.5:
        return 8
    if total <= 8.5:
        return 12
    return None
