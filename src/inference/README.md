# Real-time inference notes

Phase 9 implements the streaming engine and its input/output sources
described in README §13.

## Modules

- **`stream.py`** — `SlidingWindowBuffer` (raw per-sample readings ->
  epochs, at the configured window/hop), `EWMASmoother` (per-task
  exponentially-weighted probability smoothing), and `StreamingEngine`
  (ties both together with a rolling `sequence_length`-epoch buffer for
  the Bi-LSTM context window and a loaded model). `push_sample(sample,
  timestamp)` is the single entry point — call it once per incoming raw
  sample; it returns a `PredictionEvent` whenever a new window completes
  *and* enough sequence context has accumulated, else `None`.
- **`replay.py`** — offline sources. `replay_synthetic(...)` needs zero
  external setup (generates a fresh synthetic recording); `replay_file(path)`
  replays a real `.edf`/`.bdf`/`.fif` via MNE. Both support `realtime=True`
  (paced to `sfreq`, for demos) or `realtime=False` (as fast as possible,
  for benchmarking/tests).
- **`lsl_source.py`** — optional live source via `pylsl`. The import is
  guarded (`LSL_AVAILABLE` flag) since `pylsl` requires the native
  `liblsl` library, which most dev/CI machines without EEG hardware won't
  have; `stream_lsl()` raises a clear `RuntimeError` rather than crashing
  at import time when it's unavailable.
- **`websocket_server.py`** — `PredictionBroadcastServer`, a minimal
  one-way `websockets`-based pub/sub server: every connected client
  receives each `PredictionEvent` pushed via `broadcast()`. This is what
  the Streamlit dashboard (Phase 10) would consume in a fully decoupled
  deployment; the app itself calls `StreamingEngine` directly for
  simplicity, so this server is for external/multi-client consumers.

## Usage sketch

```python
import asyncio
from src.inference.replay import replay_synthetic
from src.inference.stream import StreamingEngine
from src.inference.websocket_server import PredictionBroadcastServer
from src.models.registry import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import DataConfig, ModelConfig

model = build_model(ModelConfig(name="aeromind_capsnet"), DataConfig())
load_checkpoint("runs/my_run/subject_dependent/best.ckpt", model)

engine = StreamingEngine(model, sequence_length=15, window_samples=512, hop_samples=128)

async def main():
    server = PredictionBroadcastServer()
    await server.start()
    for sample, t in replay_synthetic(duration_s=60.0, realtime=True):
        event = engine.push_sample(sample, t)
        if event is not None:
            await server.broadcast(event)

asyncio.run(main())
```

## Measured latency (this machine, CPU)

```bash
python scripts/benchmark_latency.py --duration_s 60 --output results/latency_benchmark.json
```

| Model | p50 | p95 | max | Real-time capable? (0.5s hop budget) |
|---|---|---|---|---|
| AeroMind-CapsNet | 13.5 ms | 16.1 ms | 20.3 ms | Yes |
| AeroMind-CNN-LSTM | 9.6 ms | 11.1 ms | 16.5 ms | Yes |
| AeroMind-EEGNet | 6.0 ms | 6.9 ms | 57.0 ms | Yes |

All three models finish a single-window inference in well under the 500ms
hop budget (2s window / 0.5s hop, README §13) on CPU alone — no GPU is
required for real-time operation at this window/hop configuration. This
is **measured (synthetic smoke test)**, not the README's aspirational
"~120ms on an RTX 3060" figure (that number describes different hardware
and isn't reproduced here); see `results/latency_benchmark.json` for the
raw numbers. EEGNet's outlier max (57ms vs. a 6.9ms p95) is consistent
with an OS scheduling jitter spike on a single CPU run, not a systematic
cost — re-run the benchmark for a fresh sample if that number matters to you.

## Honesty note

`replay_file` (real `.edf`/`.bdf`/`.fif` playback) and `lsl_source.py`
(live hardware) are implemented but not exercised by the measured numbers
above — no real EEG hardware or recording file was available in this
environment. `tests/test_inference.py` covers the buffer/smoother/engine
logic and the synthetic replay + WebSocket broadcast paths only.
