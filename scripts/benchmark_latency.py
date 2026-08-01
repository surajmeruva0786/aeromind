#!/usr/bin/env python
"""Measures end-to-end streaming inference latency per prediction window
(README §13, roadmap step 101) on this machine, for all three models.

For each model, replays a synthetic recording (non-realtime, i.e. as fast
as the CPU can go) through `src.inference.stream.StreamingEngine` and times
each `push_sample` call that produces a `PredictionEvent` — i.e. the cost
of one sliding-window inference, which is what actually gates real-time
usability at a given hop size.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.inference.replay import replay_synthetic
from src.inference.stream import StreamingEngine
from src.models.registry import build_model
from src.utils.config import DataConfig, ModelConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def benchmark_model(
    model_name: str,
    duration_s: float,
    sequence_length: int,
    window_s: float,
    hop_s: float,
    sfreq: float,
) -> dict:
    data_config = DataConfig(sfreq=sfreq, epoch_seconds=window_s, sequence_length=sequence_length)
    model = build_model(ModelConfig(name=model_name), data_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    engine = StreamingEngine(
        model,
        sequence_length=sequence_length,
        window_samples=int(window_s * sfreq),
        hop_samples=int(hop_s * sfreq),
        n_channels=data_config.n_channels,
        device=device,
    )

    latencies_ms: list[float] = []
    for sample, t in replay_synthetic(
        subject_id=0, duration_s=duration_s, sfreq=sfreq, realtime=False
    ):
        start = time.perf_counter()
        event = engine.push_sample(sample, t)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if event is not None:
            latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "model": model_name,
        "device": device,
        "n_predictions": n,
        "mean_ms": sum(latencies_ms) / n if n else None,
        "p50_ms": latencies_ms[n // 2] if n else None,
        "p95_ms": latencies_ms[int(n * 0.95)] if n else None,
        "max_ms": latencies_ms[-1] if n else None,
        "hop_s": hop_s,
        "realtime_budget_ms": hop_s * 1000.0,
        "realtime_capable": (latencies_ms[int(n * 0.95)] < hop_s * 1000.0) if n else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["aeromind_capsnet", "aeromind_cnn_lstm", "aeromind_eegnet"]
    )
    parser.add_argument("--duration_s", type=float, default=120.0)
    parser.add_argument("--sequence_length", type=int, default=15)
    parser.add_argument("--window_s", type=float, default=2.0)
    parser.add_argument("--hop_s", type=float, default=0.5)
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--output", default="results/latency_benchmark.json")
    args = parser.parse_args()

    results = []
    for model_name in args.models:
        logger.info("Benchmarking %s...", model_name)
        result = benchmark_model(
            model_name, args.duration_s, args.sequence_length, args.window_s, args.hop_s, args.sfreq
        )
        results.append(result)
        logger.info(
            "%s: p50=%.1fms p95=%.1fms (budget=%.0fms)",
            model_name,
            result["p50_ms"],
            result["p95_ms"],
            result["realtime_budget_ms"],
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
