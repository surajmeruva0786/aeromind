# AeroMind Streamlit demo

Phase 10 implements the demo dashboard described in README §14.

## Run it

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. Zero setup required — the default
"Synthetic (zero setup)" source generates a fresh synthetic recording, so
you can click **Start / restart session** immediately with no data
download.

## What's in the UI

- **Sidebar** — replay source (synthetic or upload a `.edf`/`.bdf`/`.fif`
  file), model architecture, an optional checkpoint path, and how many raw
  samples to process per screen refresh.
- **Live EEG plot** — the last 10 seconds of the 7-channel signal.
- **Workload / fatigue panels** — EWMA-smoothed probability bars and a
  fatigue-probability meter, updated every time a new 2s/0.5s-hop window
  completes.
- **Explainability** — an on-demand "Explain current prediction" button
  runs `src.xai.shap_channel` + `src.xai.topomap` on the model's current
  streaming context and renders a scalp topomap. This is a button, not a
  per-frame computation, because `shap.GradientExplainer` is too expensive
  to run on every refresh tick.
- **Session export** — every smoothed prediction this session is
  accumulated into a table, downloadable as CSV.

## No-checkpoint behavior

If you don't provide a checkpoint path (or the path doesn't resolve to a
real file), the app runs on a freshly initialized, **untrained** model and
shows a persistent warning banner saying so. This keeps the "zero setup"
promise (you can explore the whole UI immediately) while never silently
presenting untrained-model output as if it were a real prediction. Point
"Checkpoint path" at a real `runs/<output_dir>/<fold_name>/best.ckpt` —
e.g. the one referenced in `results/synthetic_smoke_test.md` — for actual
model output. The model architecture selected in the sidebar must match
the checkpoint's architecture (the app doesn't infer this automatically).

## How live updates work

`Start / restart session` kicks off an auto-refreshing loop: each script
run processes `n_steps_per_tick` raw samples, updates the panels, then
(if still "running") sleeps briefly and calls `st.rerun()` to process the
next batch. This is a standard lightweight pattern for Streamlit live
dashboards — no extra polling libraries needed — but it does mean the
browser tab keeps re-rendering while a session is running; click **Stop**
to end it.

## Testing

`app/streamlit_app.py` keeps every `st.*` call inside `main()`
(`if __name__ == "__main__":` guarded), so `import app.streamlit_app` is
always safe — no Streamlit runtime is touched at import time.
`tests/test_app.py` uses `streamlit.testing.v1.AppTest` to actually render
the app and click sidebar/explain controls headlessly, without exercising
the intentional live auto-refresh loop (see that file's docstring for
why). The app was also manually smoke-tested with a real
`streamlit run` server (HTTP 200, functioning UI) during development.
