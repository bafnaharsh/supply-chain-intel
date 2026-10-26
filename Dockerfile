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
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable
# Scientific packages ship their own test suites (~100 MB); the service never runs them.
RUN find /opt/venv/lib/python3.13/site-packages -depth -type d \( -name tests -o -name test \) \
      -exec rm -rf {} + \
 && find /opt/venv/lib/python3.13/site-packages -depth -type d -name __pycache__ -exec rm -rf {} +
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
COPY --from=builder /build/data/warehouse.duckdb ./data/warehouse.duckdb
COPY --from=builder /build/reports/dq_report.json ./reports/dq_report.json
COPY models ./models
ENV DB_PATH=/app/data/warehouse.duckdb DQ_REPORT_PATH=/app/reports/dq_report.json \
    MODEL_PATH=/app/models/delay_model.joblib
EXPOSE 8000
# exec form only: there is no /bin/sh in this image.
# start-period 20 s: startup is ~1 s idle but was measured at 6.6 s on a loaded laptop.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD ["/usr/bin/python3.13", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]
ENTRYPOINT ["/usr/bin/python3.13", "-m", "uvicorn", "app.main:app"]
# --limit-concurrency sheds load with a 503 instead of queueing requests for 10+ s (load test).
CMD ["--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "128"]

# --- target: assistant — the same code plus the google-adk extra; needs GEMINI_API_KEY at run time ---
FROM builder AS builder-assistant
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra assistant \
 && find /opt/venv/lib/python3.13/site-packages -depth -type d \( -name tests -o -name test \) \
      -exec rm -rf {} + \
 && find /opt/venv/lib/python3.13/site-packages -depth -type d -name __pycache__ -exec rm -rf {} +

FROM runtime AS assistant
COPY --from=builder-assistant /opt/venv/lib/python3.13/site-packages /opt/site-packages
# Sessions and logs go to a mounted directory; the root filesystem is read-only.
ENV ASSISTANT_API_URL=http://api:8000 ASSISTANT_ROOT_PATH=/assistant \
    ASSISTANT_SESSION_DB=/app/var/assistant_sessions.db ASSISTANT_LOG_PATH=/app/var/assistant.jsonl
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD ["/usr/bin/python3.13", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health', timeout=2)"]
ENTRYPOINT ["/usr/bin/python3.13", "-m", "uvicorn", "assistant.api:app"]
CMD ["--host", "0.0.0.0", "--port", "8001"]
