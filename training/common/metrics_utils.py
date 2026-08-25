"""
common/metrics_utils.py
Shared helpers used by all 3 training scripts so the final comparison
(compare_results.py) reads a consistent schema regardless of which
model/framework produced it.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable, Sequence


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark_fps(
    predict_fn: Callable,
    sample_inputs: Sequence,
    device,
    n_warmup: int = 10,
    n_iters: int = 50,
) -> tuple[float, float]:
    """Measure single-image (batch=1) inference FPS/latency on `device`.
    `predict_fn(x)` should run one forward pass with no_grad already handled
    by the caller (each training script wraps its own model-specific call).
    """
    import torch

    for i in range(n_warmup):
        predict_fn(sample_inputs[i % len(sample_inputs)])
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(n_iters):
        predict_fn(sample_inputs[i % len(sample_inputs)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    fps = n_iters / elapsed
    latency_ms = (elapsed / n_iters) * 1000
    return fps, latency_ms


def save_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"wrote {path}")
