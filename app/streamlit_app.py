"""AeroMind Streamlit demo (README §14): live EEG plot, workload/fatigue
probability bars, an on-demand SHAP channel-attribution topomap, and CSV
session export.

Usage:
    streamlit run app/streamlit_app.py

Zero-setup: with no checkpoint path given, the app runs on a freshly
initialized (untrained) model and prominently labels predictions as such
— this is a UI-demonstration path, not a claim of working predictions.
Point the sidebar "Checkpoint path" field at a real `runs/.../best.ckpt`
(see `results/synthetic_smoke_test.md`) for real model output.

All top-level Streamlit calls live inside `main()`, guarded by
`if __name__ == "__main__"` — `import app.streamlit_app` (used by the
headless smoke test) never touches the Streamlit runtime.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data.synthetic import CHANNEL_NAMES, FATIGUE_LABELS, WORKLOAD_LABELS
from src.inference.replay import replay_file, replay_synthetic
from src.inference.stream import PredictionEvent, StreamingEngine
from src.models.registry import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import DataConfig, ModelConfig
from src.xai.shap_channel import compute_channel_attributions
from src.xai.topomap import plot_channel_topomap

WINDOW_S = 2.0
HOP_S = 0.5
SFREQ = 256.0
SEQUENCE_LENGTH = 15
EEG_PLOT_SECONDS = 10.0
RERUN_PACING_SECONDS = 0.05  # avoids a zero-delay busy loop while auto-playing


def event_to_history_row(event: PredictionEvent) -> dict:
    """Pure (no Streamlit dependency) so it's unit-testable directly."""
    return {
        "timestamp": event.timestamp,
        **{f"workload_{lbl}": p for lbl, p in zip(WORKLOAD_LABELS, event.smoothed_workload_probs)},
        **{f"fatigue_{lbl}": p for lbl, p in zip(FATIGUE_LABELS, event.smoothed_fatigue_probs)},
    }


def _init_state() -> None:
    defaults = {
        "engine": None,
        "replay_iter": None,
        "history": [],
        "eeg_buffer": [],
        "running": False,
        "checkpoint_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_engine(
    model_name: str, checkpoint_path: str | None, sequence_length: int
) -> StreamingEngine:
    data_config = DataConfig(sfreq=SFREQ, epoch_seconds=WINDOW_S, sequence_length=sequence_length)
    model = build_model(ModelConfig(name=model_name), data_config)

    checkpoint_loaded = False
    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model)
        checkpoint_loaded = True
    st.session_state.checkpoint_loaded = checkpoint_loaded

    return StreamingEngine(
        model,
        sequence_length=sequence_length,
        window_samples=int(WINDOW_S * SFREQ),
        hop_samples=int(HOP_S * SFREQ),
        n_channels=data_config.n_channels,
    )


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Source & model")
        source = st.radio("Replay source", ["Synthetic (zero setup)", "Upload .edf/.bdf/.fif"])
        model_name = st.selectbox(
            "Model", ["aeromind_capsnet", "aeromind_cnn_lstm", "aeromind_eegnet"]
        )
        checkpoint_path = st.text_input("Checkpoint path (optional)", value="")
        n_steps_per_tick = st.slider("Samples processed per refresh", 32, 512, 128, step=32)

        uploaded_file = None
        if source == "Upload .edf/.bdf/.fif":
            uploaded_file = st.file_uploader("EEG file", type=["edf", "bdf", "fif"])

        start = st.button("Start / restart session", type="primary")
        stop = st.button("Stop")

    return {
        "source": source,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "n_steps_per_tick": n_steps_per_tick,
        "uploaded_file": uploaded_file,
        "start": start,
        "stop": stop,
    }


def handle_start(controls: dict) -> None:
    st.session_state.engine = build_engine(
        controls["model_name"], controls["checkpoint_path"] or None, SEQUENCE_LENGTH
    )
    st.session_state.history = []
    st.session_state.eeg_buffer = []

    if controls["source"] == "Synthetic (zero setup)":
        st.session_state.replay_iter = replay_synthetic(
            duration_s=600.0, sfreq=SFREQ, realtime=False
        )
        st.session_state.running = True
    elif controls["uploaded_file"] is not None:
        tmp_path = (
            Path(tempfile.gettempdir()) / "aeromind_streamlit" / controls["uploaded_file"].name
        )
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(controls["uploaded_file"].read())
        st.session_state.replay_iter = replay_file(str(tmp_path), realtime=False)
        st.session_state.running = True
    else:
        st.warning("Upload a file first, or switch to the synthetic source.")
        st.session_state.replay_iter = None
        st.session_state.running = False


def step_replay(n_steps: int):
    engine = st.session_state.engine
    latest_event = None
    for _ in range(n_steps):
        try:
            sample, t = next(st.session_state.replay_iter)
        except StopIteration:
            st.session_state.running = False
            break
        st.session_state.eeg_buffer.append(sample)
        max_buffer = int(SFREQ * EEG_PLOT_SECONDS)
        if len(st.session_state.eeg_buffer) > max_buffer:
            st.session_state.eeg_buffer.pop(0)

        event = engine.push_sample(sample, t)
        if event is not None:
            latest_event = event
            st.session_state.history.append(event_to_history_row(event))
    return latest_event


def render_live_panels(latest_event) -> None:
    if st.session_state.eeg_buffer:
        arr = np.stack(st.session_state.eeg_buffer, axis=0)
        st.line_chart(pd.DataFrame(arr, columns=CHANNEL_NAMES))
    else:
        st.caption("Waiting for samples...")

    if latest_event is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Workload")
            st.bar_chart(
                pd.DataFrame(
                    {"probability": latest_event.smoothed_workload_probs},
                    index=list(WORKLOAD_LABELS),
                )
            )
        with col2:
            st.subheader("Fatigue")
            fatigued_prob = float(
                latest_event.smoothed_fatigue_probs[FATIGUE_LABELS.index("fatigued")]
            )
            st.metric("Fatigue probability", f"{fatigued_prob:.0%}")
            st.progress(min(max(fatigued_prob, 0.0), 1.0))


def render_explainability() -> None:
    st.subheader("Explainability")
    st.caption(
        "SHAP channel attribution is computed on demand (not every frame) — "
        "it's too expensive to run per-tick in a live loop."
    )
    if st.button("Explain current prediction (SHAP topomap)"):
        engine = st.session_state.engine
        if engine is None or len(engine.epoch_sequence) < SEQUENCE_LENGTH:
            st.info(
                "Not enough streaming context yet — start a session and let it run for a bit first."
            )
            return
        seq = np.stack(engine.epoch_sequence, axis=0)
        x = torch.from_numpy(seq).unsqueeze(0)
        background = x.repeat(4, 1, 1, 1) + torch.randn(4, *seq.shape) * 0.01
        attribution = compute_channel_attributions(engine.model, background, x, CHANNEL_NAMES)
        fig = plot_channel_topomap(
            attribution.channel_values[0],
            CHANNEL_NAMES,
            title="Channel attribution (current window)",
        )
        st.pyplot(fig)


def render_session_export() -> None:
    st.subheader("Session export")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df.tail(20))
        st.download_button(
            "Download session CSV",
            df.to_csv(index=False),
            file_name="aeromind_session.csv",
            mime="text/csv",
        )
    else:
        st.caption("No predictions recorded yet this session.")


def main() -> None:
    st.set_page_config(page_title="AeroMind", layout="wide")
    _init_state()

    st.title("AeroMind — EEG Cognitive Workload & Fatigue Monitor")
    st.caption(
        "Research prototype — decision support only, not autonomous decision-making (README §20)."
    )

    controls = render_sidebar()

    if controls["start"]:
        handle_start(controls)
    if controls["stop"]:
        st.session_state.running = False

    if st.session_state.engine is not None and not st.session_state.checkpoint_loaded:
        st.warning(
            "No trained checkpoint loaded — predictions come from a freshly initialized "
            "(untrained) model. This is a UI-demonstration path only; point 'Checkpoint "
            "path' at a real runs/.../best.ckpt for real output."
        )

    latest_event = None
    if (
        st.session_state.running
        and st.session_state.engine is not None
        and st.session_state.replay_iter is not None
    ):
        latest_event = step_replay(controls["n_steps_per_tick"])

    render_live_panels(latest_event)
    render_explainability()
    render_session_export()

    if st.session_state.running:
        time.sleep(RERUN_PACING_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
