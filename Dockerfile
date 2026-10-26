# Builder: Debian 13 + Python 3.13, the same pair as the distroless runtime so the
# compiled wheels (duckdb, pydantic-core) are ABI-compatible across stages.
FROM python:3.13-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv
# No bytecode: for scipy/sklearn/pandas the .pyc files are ~100 MB and buy ~1 s of cold start.
ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
WORKDIR /build
# Dependencies first so code changes don't invalidate this layer; the cache mount keeps
# downloaded wheels across builds so a lockfile change doesn't re-download everything.
COPY pyproject.toml uv.lock ./
# Scientific packages ship their own test suites (~100 MB); the service never runs them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project \
 && find /opt/venv/lib/python3.13/site-packages -depth -type d \( -name tests -o -name test \) \
      -exec rm -rf {} + \
 && find /opt/venv/lib/python3.13/site-packages -depth -type d -name __pycache__ -exec rm -rf {} +
# Our code goes to its own directory, not the venv: a source edit then rebuilds a <1 MB layer
# instead of re-snapshotting the ~430 MB dependency layer.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps --target /opt/project .
ENV PYTHONPATH=/opt/project
# Data product, built at image-build time: the DQ report first (standalone, raw CSVs only),
# then the warehouse. A failed pipeline invariant fails the build, which is the point.
COPY data/raw ./data/raw
RUN /opt/venv/bin/python -m dq check --input data/raw --output reports/dq_report.json \
 && /opt/venv/bin/python -m pipeline --input data/raw --db data/warehouse.duckdb

# Runtime base: distroless = no shell, no package manager, no pip (and so none of pip's vendored
# urllib3/msgpack/setuptools CVEs), runs as uid 65532.
FROM gcr.io/distroless/python3-debian13:nonroot AS runtime
WORKDIR /app
ENV PYTHONPATH=/opt/site-packages PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    # One BLAS/OpenMP thread per request thread: under load, per-thread pools oversubscribe the
    # CPU (load test: one predict call at 700% CPU, 0.4-2.9 s; pinned: ~30 ms).
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1

# --- target: api (default) — the data API with the warehouse, DQ report and model baked in ---
FROM runtime AS api
COPY --from=builder /opt/venv/lib/python3.13/site-packages /opt/site-packages
COPY --from=builder /opt/project /opt/site-packages
COPY --from=builder /build/data/warehouse.duckdb ./data/warehouse.duckdb
COPY --from=builder /build/reports/dq_report.json ./reports/dq_report.json
COPY models ./models
ENV DB_PATH=/app/data/warehouse.duckdb DQ_REPORT_PATH=/app/reports/dq_report.json \
    MODEL_PATH=/app/models/delay_model.joblib
EXPOSE 8000
# exec form only: there is no /bin/sh in this image.
# Generous limits: startup is ~1 s idle but 8 s on a loaded laptop, where the probe itself
# (a fresh python process) took 4.7 s and a 3 s timeout failed compose --wait.
HEALTHCHECK --interval=10s --timeout=10s --start-period=60s --retries=3 \
  CMD ["/usr/bin/python3.13", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=8)"]
# Worker processes, not threads: the GIL caps one process at ~67 rps (mix @50); 4 workers gave
# ~95 rps and /health p50 545 → 110 ms under load, at ~150 MB each. Override with WEB_CONCURRENCY.
ENV WEB_CONCURRENCY=4
ENTRYPOINT ["/usr/bin/python3.13", "-m", "uvicorn", "app.main:app"]
# --limit-concurrency (per worker, 4 × 32 = 128) sheds load with a 503 instead of queueing for
# 10+ s. --timeout-keep-alive outlives Caddy's 2 min upstream idle reuse; uvicorn's 5 s default
# reset reused connections and Caddy turned them into 502s.
CMD ["--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "32", "--timeout-keep-alive", "130"]

# --- target: assistant — the same code plus the google-adk extra; needs GEMINI_API_KEY at run time ---
FROM builder AS builder-assistant
# The assistant only calls the data API over HTTP, so the ML/DuckDB stack (~340 MB) stays out.
# Its own venv: syncing over /opt/venv would start from one that already has that stack.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv-assistant
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra assistant \
      --no-install-package scikit-learn --no-install-package scipy --no-install-package numpy \
      --no-install-package pandas --no-install-package joblib --no-install-package threadpoolctl \
      --no-install-package duckdb \
 && find /opt/venv-assistant/lib/python3.13/site-packages -depth -type d \( -name tests -o -name test \) \
      -exec rm -rf {} + \
 && find /opt/venv-assistant/lib/python3.13/site-packages -depth -type d -name __pycache__ -exec rm -rf {} + \
 && mkdir /build/var

FROM runtime AS assistant
COPY --from=builder-assistant /opt/venv-assistant/lib/python3.13/site-packages /opt/site-packages
COPY --from=builder /opt/project /opt/site-packages
# Owned by nonroot so the named volume mounted here inherits a writable owner on first use.
COPY --from=builder-assistant --chown=65532:65532 /build/var /app/var
# Sessions and logs go to a mounted directory; the root filesystem is read-only.
ENV ASSISTANT_API_URL=http://api:8000 ASSISTANT_ROOT_PATH=/assistant \
    ASSISTANT_SESSION_DB=/app/var/assistant_sessions.db ASSISTANT_LOG_PATH=/app/var/assistant.jsonl
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=10s --start-period=60s --retries=3 \
  CMD ["/usr/bin/python3.13", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health', timeout=8)"]
ENTRYPOINT ["/usr/bin/python3.13", "-m", "uvicorn", "assistant.api:app"]
CMD ["--host", "0.0.0.0", "--port", "8001"]
