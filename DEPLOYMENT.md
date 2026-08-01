# Deployment guide

Three supported deployment paths for the Streamlit demo (`app/streamlit_app.py`),
in order of setup effort. All three run the same zero-setup synthetic
replay source out of the box (README §14, `app/README.md`).

---

## 1. Docker (local or any container host)

```bash
docker build -t aeromind .
docker run -p 8501:8501 aeromind
```

or via Compose (also mounts `runs/`, `results/`, `data/` so trained
checkpoints and generated results persist on the host):

```bash
docker compose up --build
```

Open `http://localhost:8501`.

**Image notes**: `Dockerfile` is CPU-only — it installs the CPU-only
PyTorch wheel (`--index-url https://download.pytorch.org/whl/cpu`)
explicitly, so the default CUDA-bundled PyPI `torch` package is never
pulled in. This is intentional and matches the rest of the repo: nothing
here requires a GPU (see `results/latency_benchmark.json` — all three
models clear the real-time budget on CPU alone). If you do have a CUDA
host and want GPU inference, install a CUDA-matched `torch` build instead
before `pip install -r requirements.txt` in a customized image.

**Build-health CI**: `.github/workflows/docker.yml` builds this image
(without pushing anywhere) on every push/PR, catching Dockerfile
breakage before it reaches a release. Honesty note: the Dockerfile
authored here could **not** be build-verified locally (no reachable
Docker daemon in that sandboxed environment) — but `.github/workflows/docker.yml`
**has now run and passed on GitHub Actions**, confirming the image
actually builds; this was verified by checking the real run result via
the GitHub API, not assumed. See `ROADMAP.md`'s Phase 12 addendum for
the full story (a separate, real bug in `ci.yml` — unrelated to Docker —
was also caught and fixed the same way).

**Running other CLIs in the container**: the entrypoint launches
Streamlit by default, but the image has the full repo installed — override
the entrypoint to run training/evaluation/XAI/inference instead:

```bash
docker run --entrypoint python aeromind -m src.training.train --model aeromind_capsnet --protocol subject_dependent
```

---

## 2. Streamlit Community Cloud

1. Push this repository to GitHub (already the case if you're reading this
   from `github.com/surajmeruva0786/aeromind`).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at this repo, branch `main`, entrypoint file `app/streamlit_app.py`.
3. Streamlit Cloud reads `requirements.txt` (repo root) and `runtime.txt`
   (pins `python-3.11`) automatically — no extra config needed for a
   basic deploy.
4. `.streamlit/config.toml` (committed) sets headless mode and disables
   usage-stats gathering; it applies automatically.
5. Secrets: the app needs **no secrets to run** in its default zero-setup
   mode. `.streamlit/secrets.toml.example` documents the one optional key
   it would read if you set it (a default checkpoint URL) — copy it to
   `.streamlit/secrets.toml` locally (gitignored) or use the Community
   Cloud "Secrets" dashboard for a deployed app; neither is required to
   get the demo running.

**Resource note**: Streamlit Community Cloud's free tier has limited
CPU/RAM. The synthetic replay + streaming inference path is lightweight
(see the latency benchmark), but loading `shap`/`mne`/`torch` together at
import time is the main startup-time cost — expect a slower cold start
than a local run.

---

## 3. Hugging Face Spaces (Docker SDK)

HF Spaces can run this repo's own `Dockerfile` directly (Docker SDK,
not the Streamlit SDK) so no separate Spaces-specific config is needed:

1. Create a new Space, SDK = **Docker**.
2. Push this repo's contents (or link the GitHub repo) to the Space.
3. HF Spaces builds `Dockerfile` as-is and exposes port `8501` (already
   `EXPOSE`d) — the platform's own reverse proxy handles the public URL.
4. No secrets required for the default synthetic demo, same as above; add
   any via the Space's "Settings → Repository secrets" if you later wire
   up a hosted checkpoint.

---

## Which one should I actually use?

- **Just trying it out / no GitHub repo yet**: Docker locally — fastest
  path, identical to what CI build-checks.
- **Sharing a public demo link with zero infra to manage**: Streamlit
  Community Cloud — free, GitHub-integrated, but the lower-effort option
  in exchange for less control over resources.
- **Need more compute/RAM than Community Cloud's free tier, or want a
  persistent container you fully control**: HF Spaces (Docker SDK) — reuses
  the exact same `Dockerfile`.

None of these paths require a GPU, gated dataset access, or any manual
data download — the zero-setup synthetic path (`app/README.md`) works
identically in all three environments.
