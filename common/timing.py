"""Training time, inference latency and peak GPU memory -- measured, never estimated."""

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer(label: str = ""):
    """Wall-clock timer. Yields a dict that gets a 'seconds' key on exit."""
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        if label:
            print(f"[timing] {label}: {result['seconds']:.1f}s")


def reset_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_vram_mb() -> float | None:
    """Peak allocated VRAM since the last reset, in MB. None on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except ImportError:
        pass
    return None


def measure_latency(predict_one, sample, warmup: int = 50, iters: int = 200) -> dict:
    """Single-image inference latency.

    Batch size 1 on purpose: that is the deployment case for an endoscopy tool,
    and it is the only number that is comparable across four models with
    different natural batch behaviour. CUDA is synchronized around every call --
    without that you time the kernel launch, not the inference.
    """
    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except ImportError:
        sync = lambda: None  # noqa: E731

    for _ in range(warmup):
        predict_one(sample)
    sync()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        predict_one(sample)
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    return {
        "inference_ms_mean": round(mean, 3),
        "inference_ms_median": round(samples[n // 2], 3),
        "inference_ms_p95": round(samples[int(n * 0.95)], 3),
        "fps": round(1000.0 / mean, 1) if mean > 0 else None,
        "latency_iters": iters,
        "latency_warmup": warmup,
    }


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable),
            "params_millions": round(total / 1e6, 2)}


def file_size_mb(path) -> float | None:
    from pathlib import Path
    p = Path(path)
    return round(p.stat().st_size / 1024**2, 2) if p.exists() else None
