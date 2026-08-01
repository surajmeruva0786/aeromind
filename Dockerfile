# AeroMind — CPU-only container image (README §15, ROADMAP Phase 12).
# Runs the Streamlit demo by default; the same image can run any CLI
# (training/evaluation/XAI/inference) via `docker run ... python -m ...`.

FROM python:3.11-slim

# build-essential: a few scientific-Python deps fall back to source builds
# on some platforms if no prebuilt wheel matches; libgomp1: OpenMP runtime
# needed by scikit-learn/scipy; curl: container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

# CPU-only torch wheel first, so the default PyPI (CUDA-bundled) torch
# package is never pulled in — keeps the image well under a GPU image's
# size and avoids requiring a CUDA-capable host.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
