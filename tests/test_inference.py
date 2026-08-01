"""Phase 9 tests: sliding-window buffer logic, EWMA smoothing, the
streaming engine end-to-end, replay pacing, and the WebSocket broadcast
server (README roadmap step 100)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import websockets

from src.inference.replay import replay_synthetic
from src.inference.stream import EWMASmoother, PredictionEvent, SlidingWindowBuffer, StreamingEngine
from src.inference.websocket_server import PredictionBroadcastServer, event_to_json
from src.models import AeroMindEEGNet


def test_sliding_window_buffer_emits_first_window_as_soon_as_full():
    buf = SlidingWindowBuffer(n_channels=2, window_samples=4, hop_samples=2)
    windows = []
    for i in range(4):
        w = buf.push(np.array([float(i), float(i)]))
        windows.append(w)
    assert windows[:3] == [None, None, None]
    assert windows[3] is not None
    assert windows[3].shape == (2, 4)


def test_sliding_window_buffer_respects_hop():
    buf = SlidingWindowBuffer(n_channels=1, window_samples=4, hop_samples=2)
    emitted = []
    for i in range(10):
        w = buf.push(np.array([float(i)]))
        if w is not None:
            emitted.append(w[0].tolist())
    # first window at sample index 3 (0-based), then every 2 samples after
    assert emitted[0] == [0.0, 1.0, 2.0, 3.0]
    assert emitted[1] == [2.0, 3.0, 4.0, 5.0]
    assert emitted[2] == [4.0, 5.0, 6.0, 7.0]


def test_sliding_window_buffer_rejects_wrong_shape():
    buf = SlidingWindowBuffer(n_channels=3, window_samples=4, hop_samples=2)
    with pytest.raises(ValueError):
        buf.push(np.array([1.0, 2.0]))


def test_ewma_smoother_converges_toward_constant_input():
    smoother = EWMASmoother(alpha=0.5)
    target = np.array([0.1, 0.9])
    out = None
    for _ in range(20):
        out = smoother.update(target)
    assert np.allclose(out, target, atol=1e-4)


def test_ewma_smoother_reset_clears_state():
    smoother = EWMASmoother(alpha=0.3)
    smoother.update(np.array([1.0, 0.0]))
    smoother.reset()
    first = smoother.update(np.array([0.0, 1.0]))
    assert np.allclose(first, [0.0, 1.0])


def test_streaming_engine_produces_predictions_once_context_fills():
    model = AeroMindEEGNet()
    engine = StreamingEngine(model, sequence_length=3, window_samples=32, hop_samples=32, n_channels=7)

    events = []
    for sample, t in replay_synthetic(subject_id=0, duration_s=5.0, sfreq=256.0, realtime=False):
        event = engine.push_sample(sample, t)
        if event is not None:
            events.append(event)

    assert len(events) > 0
    first = events[0]
    assert isinstance(first, PredictionEvent)
    assert first.workload_probs.shape == (3,)
    assert first.fatigue_probs.shape == (2,)
    assert np.isclose(first.workload_probs.sum(), 1.0, atol=1e-4)


def test_replay_synthetic_non_realtime_is_fast_and_ordered():
    samples = list(replay_synthetic(subject_id=1, duration_s=1.0, sfreq=256.0, realtime=False))
    assert len(samples) == 256
    timestamps = [t for _, t in samples]
    assert timestamps == sorted(timestamps)
    assert samples[0][0].shape == (7,)


def test_prediction_broadcast_server_delivers_events():
    async def _run():
        server = PredictionBroadcastServer(host="localhost", port=8799)
        await server.start()
        try:
            async with websockets.connect("ws://localhost:8799") as client:
                event = PredictionEvent(
                    timestamp=1.0,
                    workload_probs=np.array([0.2, 0.5, 0.3]),
                    fatigue_probs=np.array([0.6, 0.4]),
                    smoothed_workload_probs=np.array([0.2, 0.5, 0.3]),
                    smoothed_fatigue_probs=np.array([0.6, 0.4]),
                )
                # give the server a moment to register the connection
                await asyncio.sleep(0.05)
                await server.broadcast(event)
                message = await asyncio.wait_for(client.recv(), timeout=2.0)
                assert message == event_to_json(event)
        finally:
            await server.stop()

    asyncio.run(_run())
