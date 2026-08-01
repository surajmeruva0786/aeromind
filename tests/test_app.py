"""Phase 10 smoke test (README roadmap step 107): headless import check
plus a Streamlit `AppTest` run verifying the demo app renders without
exceptions and responds to sidebar/control interactions.

Deliberately does NOT click "Start / restart session": that path enters an
intentional auto-refreshing `st.rerun()` loop (see the module docstring in
`app/streamlit_app.py`) meant for a live browser session, not a bounded
test run. `event_to_history_row` and `build_engine` are pure/session-state
helpers exercised directly instead.
"""

from __future__ import annotations

import numpy as np
from streamlit.testing.v1 import AppTest

from app.streamlit_app import WORKLOAD_LABELS, event_to_history_row
from src.inference.stream import PredictionEvent


def test_app_module_imports_cleanly():
    import app.streamlit_app as mod

    assert hasattr(mod, "main")
    assert hasattr(mod, "build_engine")


def test_event_to_history_row_is_pure_and_correct():
    event = PredictionEvent(
        timestamp=1.5,
        workload_probs=np.array([0.1, 0.6, 0.3]),
        fatigue_probs=np.array([0.7, 0.3]),
        smoothed_workload_probs=np.array([0.15, 0.55, 0.3]),
        smoothed_fatigue_probs=np.array([0.65, 0.35]),
    )
    row = event_to_history_row(event)
    assert row["timestamp"] == 1.5
    assert row[f"workload_{WORKLOAD_LABELS[1]}"] == 0.55
    assert row["fatigue_fatigued"] == 0.35


def test_app_initial_render_has_no_exceptions():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value.startswith("AeroMind")


def test_app_sidebar_controls_respond_without_error():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=30)
    at.run()
    at.selectbox[0].set_value("aeromind_eegnet").run()
    assert not at.exception
    at.radio[0].set_value("Upload .edf/.bdf/.fif").run()
    assert not at.exception


def test_explain_button_without_session_shows_info_not_crash():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=30)
    at.run()
    explain_button = next(b for b in at.button if "Explain" in b.label)
    explain_button.click().run()
    assert not at.exception
    assert any("start a session" in i.value for i in at.info)
