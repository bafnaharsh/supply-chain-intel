# Supply Chain Intelligence Mini-Platform

[![CI](https://github.com/bafnaharsh/supply-chain-intel/actions/workflows/ci.yml/badge.svg)](https://github.com/bafnaharsh/supply-chain-intel/actions/workflows/ci.yml)

A small end-to-end slice of a supply-chain platform: data-quality checks on the raw shipping CSVs, a curated DuckDB warehouse, a REST API over it, a delay-risk model, and an assistant that answers questions by calling the API. One `docker compose up` brings it all up on one port.

The brief is in [`docs/assignment/`](docs/assignment/Take_Home_Assignment.pdf); the data is described in [`data/DATA_DETAILS.md`](data/DATA_DETAILS.md). Longer write-ups live in `docs/`: [engineering](docs/engineering.md) (API contract, image size, security and load tests), [data](docs/data.md) (every DQ finding, the data model), [ml](docs/ml.md) (the full evaluation) and [assistant](docs/assistant.md) (tools, guardrails, logging, evaluation).

## Getting started

```bash
git clone https://github.com/bafnaharsh/supply-chain-intel.git && cd supply-chain-intel
docker compose up --build --wait                     # data API on http://localhost:8000
curl localhost:8000/health                           # {"status":"ok","version":"0.1.0","checks":{...}}

cp .env.example .env                                 # optional: add GEMINI_API_KEY for the assistant
docker compose --profile assistant up --build --wait # same port: /assistant/chat and /assistant/docs
```

Interactive docs: http://localhost:8000/docs and http://localhost:8000/assistant/docs. Needs Docker Compose 2.24 or newer. Without a key the assistant container runs in a degraded state (`/assistant/health` says so, `/assistant/chat` returns 503) and nothing else is affected.

Working on it locally needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python dq_check.py --input ./data/ --markdown        # standalone data-quality check
uv run python -m pipeline --input data/raw                  # raw → curated warehouse
uv run uvicorn app.main:app --reload                        # the API without Docker
uv run pytest tests/unit --cov                              # 187 tests, no infrastructure
uv run pytest tests/integration                             # against the running stack (skips if it is down)
uv run python -m assistant "What is the delay risk for shipment 421?"   # needs the key; --offline does not
```

Framework: Python 3.13 with FastAPI and Pydantic v2. Endpoints: `/health`, `/shipments`, `/shipments/{id}`, `/routes/{origin}/{destination}/stats`, `/routes/stats`, `/ports`, `/predict-delay`, `/data-quality/report`. Every error uses the same `{"error": {code, message, request_id, details}}` shape, and every request is logged as one JSON line with method, path, status and latency. Details in [docs/engineering.md](docs/engineering.md).

```
src/app         FastAPI service, DuckDB access, error handling, JSON logging
src/dq          Standalone DQ checks and column profiler (raw CSVs only)
src/pipeline    raw → staging → curated (+ quarantine) → serving views
src/ml          Features, training with model comparison, the predictor the API loads
src/assistant   ADK agent: tools, guardrails, sessions, CLI, HTTP API, offline mode
src/core        One config module: settings from .env plus every tunable constant
tests/          187 unit tests (fixture warehouse, fake LLM) and 11 integration tests
scripts/, evals/  Load test, security test, 25-question assistant evaluation
models/, reports/ Committed model artefact and card; DQ report and data profile
```

## Architecture diagram

```mermaid
flowchart LR
    subgraph Data
        RAW[(data/raw/*.csv)]
        DQ[dq check<br/>standalone]
        REPORT[/dq_report.json/]
        PIPE[pipeline<br/>raw → staging → curated]
        STORE[(warehouse.duckdb)]
    end
    subgraph ML
        TRAIN[ml.train<br/>time split, 7 candidates, gate]
        MODEL[/delay_model.joblib/]
    end
    subgraph API["Data API (FastAPI)"]
        MW[request id + JSON logs<br/>structured errors]
        EP[/health /shipments /routes<br/>/ports /predict-delay /data-quality]
    end
    subgraph Assistant["Assistant (Google ADK)"]
        ASSIST[8 tools · guardrails<br/>CLI · HTTP · offline]
        SESS[(sessions.db)]
        ALOG[/assistant.jsonl/]
    end
    LLM{{Gemini}}
    GW[Caddy gateway :8000]
    USER([user])

    USER --> GW
    GW --> MW --> EP
    GW -- /assistant/* --> ASSIST
    RAW --> DQ --> REPORT
    RAW --> PIPE --> STORE
    STORE --> TRAIN --> MODEL
    EP --> STORE
    EP --> MODEL
    EP --> REPORT
    ASSIST <--> LLM
    ASSIST --> SESS
    ASSIST --> ALOG
    ASSIST -- tools over HTTP --> MW
```

Three containers behind one port. The data API has the warehouse, the DQ report and the model baked in at image build, so it is stateless. The assistant is a separate container under a compose profile because it needs the Gemini key and the ADK stack, neither of which belongs in the data image. The CLI is the same agent run on the host.

## Tech choices

| Area | Choice | Why |
|---|---|---|
| API | FastAPI + Pydantic v2, uvicorn | Typed models, free OpenAPI docs, same language as the data and ML work |
| Data store | DuckDB, one file with `raw`/`staging`/`curated` schemas | Columnar, so route aggregates and DQ checks are fast group-bys; reads CSV directly; embedded, no server; the API opens it read-only |
| DQ | Plain SQL, one `Check` dataclass per rule | 37 checks, each a `SELECT … WHERE`; a framework would add more config than code at this size |
| ML | scikit-learn Pipeline, joblib artefact | The whole transform is inside the Pipeline so training and serving cannot drift; seven candidates compared on one time split |
| LLM | Gemini `3.5-flash-lite` via Google ADK | ADK gives the tool loop, callbacks (where guardrails live) and SQLite sessions; about $0.002 per answered question |
| Tests | pytest, FastAPI TestClient, httpx | Unit tests in-process over a fixture warehouse; integration tests against the real container |
| Packaging, lint | uv with a lockfile, ruff | One lockfile for local, Docker and CI; one tool for lint and format |
| Containers | Multi-stage build, distroless runtime, Caddy gateway | 492 MB with the ML stack; no shell, no pip, non-root, read-only; one entry point |
| CI | GitHub Actions | Lint → unit tests with a coverage gate → image build → Trivy → compose up → integration tests |

**At 100× the volume** (half a million shipments, a few million events) I would keep DuckDB and change what is around it: Parquet partitions by booking month, an incremental pipeline keyed on `shipment_id` and `event_key`, DQ on the changed partitions only, and the warehouse in object storage instead of inside the image so data and code ship separately. The real switch point is not volume, it is concurrent writers and 24/7 serving — the moment ingestion runs while the API serves, point lookups and pagination should move to Postgres, with DuckDB and Parquet kept for analytics and training.

**What I chose not to do.** No dbt, Great Expectations or orchestrator: three tables and one DAG step do not need them. No incremental loads or history: a full rebuild of 30k rows takes under a second and is trivially idempotent. No join from port events to shipments: there is no shipment key, and vessel + port + ±3 days matches only about 680 of 5,000 shipments, so events stay a port-level feed and I say so rather than fabricate a join. No CI gate on the DQ report: the data is known-dirty, so it would always fail; the pipeline's invariants are the gate. No guessing at free-text notes or missing event types.

## Data quality summary

I profiled every column first, then wrote one check per finding. The checks run against the raw CSVs only (`python dq_check.py --input ./data/`), write `reports/dq_report.json`, and that same file is served by `GET /data-quality/report`. Each entry has the check name, rows affected, percentage, severity, the action taken and the reasoning. 21 of 37 checks fail on this data; the biggest ones:

| Issue | Rows | Severity | What I did |
|---|---|---|---|
| Placeholder port codes (`XXTST`, `ZZZZZ`, `UNKNW`) | 42 origin, 55 destination | critical | Quarantined with a reason; no real port behind them |
| Same `shipment_id`, different values | 40 (20 ids) | critical | Kept the most complete row, flagged `conflicting_duplicate` |
| Exact duplicate rows | 8 | warning | Dropped |
| Status spelled five ways (`COMPLETED`, `Complete`, `delivered`, `N/A`, blank) | 146 | warning | Mapped to canonical; unknown → NULL |
| Cargo type case and whitespace variants | 261 | warning | Normalised |
| Negative weight; container count 0 or 510–996 | 36; 23 + 17 | warning | NULL and flagged rather than guessed |
| Actual departure before booking | 39 | warning | Departure nulled, arrival and label kept |
| Reused `event_id` in the event feed | 206 | warning | Keyed events on a hash of the row |
| Placeholder vessels `VSL-0000`/`VSL-9999` | 207 | warning | NULL and flagged |
| Missing notes; notes contradicting the measured delay | 1,913; 671 | info | Left alone; the measurement wins |

Two findings shaped the rest: `status` is a perfect leak (`DELAYED` is exactly "more than 24 h late" on every row), so it is never a model feature; and every oddly-spelled status row has no actual dates, so those rows are unlabelled. The full table, the 16 passing guard checks and the data model (four layers, keys, relationships, idempotency) are in [docs/data.md](docs/data.md).

## ML: delay prediction

The honest headline: booking-time features carry almost no signal in this data. The shipped model, a calibrated HistGradientBoosting, reaches test AUC **0.535 [0.502, 0.567]** and PR-AUC 0.431 against a 0.399 base rate — measurably better than chance, and nothing more. Precision/recall/F1 are 0.398 / 0.982 / 0.566 at the tuned threshold and 0.667 / 0.004 / 0.008 at 0.5; the always-late baseline gets F1 0.57, which is why I lead with AUC and calibration rather than F1.

Features are only what is known at booking: ports, cargo type, container count, weight, planned departure and arrival, and what derives from those. `status`, the actual dates and the DQ flags are excluded as leaks; customer, vessel and route ids are excluded because their late rates do not persist out of time (train→validation correlation about zero). Evaluation is a time split (train < April 2025, validate April–May, test from June) with hyper-parameters chosen on validation and test scored once. Seven candidates were compared; a written selection rule picked the base model on validation, calibrated it there, and gated it on test. The first version of the rule shipped the base-rate prior; I revised it and say so. Monitoring: every prediction is logged; retrain on drift, calibration error above 0.05, or a 5-point base-rate shift. Everything, including the full comparison table and provenance, is in [docs/ml.md](docs/ml.md) and [models/model_card.md](models/model_card.md).

## GenAI assistant

Gemini through Google ADK, with eight tools that call the data API over HTTP: `query_shipments`, `get_shipment`, `get_route_stats`, `top_routes`, `predict_delay`, `list_ports`, `resolve_port` and `data_period`. It resolves "Shanghai" to `CNSHA`, normalises "shp 421" to `SHP-00421`, anchors "this quarter" to the data's last booking date rather than today, and asks one clarifying question when something is missing instead of guessing. Sessions persist in SQLite so a conversation can be resumed. Guardrails in the ADK callbacks: a cap on LLM and tool calls per turn, argument validation before any HTTP call, a grounding check that refuses to state numbers no tool produced, and an off-domain refusal. Every prompt, model response, tool call and final answer is logged as JSON. Cost is about $0.002 per answered question. A 25-question evaluation with SQL ground truth scores 24/25. The CLI also has an offline mode for reviewers without a key. All of it in [docs/assistant.md](docs/assistant.md).

## Production support runbook

**After a deploy:** `curl -fsS http://<host>:8000/health` must return 200 with the expected version and all `checks` ok; `docker compose ps` shows `gateway` and `api` (and `assistant`) healthy. Logs are JSON on stdout; filter on `status >= 500` or on the `request_id` the client received. The image is distroless, so there is no shell to exec into — debug with logs and `/health`.

**Likely failures:**

| Symptom | Likely cause | Diagnose |
|---|---|---|
| `api` restarts in a loop | Warehouse or model file missing; the app fails fast | `docker compose logs api` shows the startup error; check `DB_PATH` / `MODEL_PATH` |
| Image build fails in the pipeline step | A data invariant failed after a data change (duplicate key, unknown port, row counts off) | The build log names the invariant; run `python dq_check.py --markdown` on the new CSVs |
| `/assistant/*` returns 502 or 503 | Assistant profile off, or no `GEMINI_API_KEY` | `docker compose --profile assistant ps`; `/assistant/health` reports `unconfigured` |
| Latency spikes on `/shipments` | Wide windows or deep `offset` scan the whole view | Sort log lines by `latency_ms`; narrow the query or precompute aggregates |

**Rollback:** redeploy the previous image tag. The service is stateless — data and model are inside the image — so there is nothing to migrate back; `/health` showing the old version confirms it.

**Next for production:** Prometheus metrics from the request middleware with alerts on p95 and 5xx rate; error tracking keyed by `request_id`; the LLM key in a secret store; central log shipping; model drift monitoring as described in the model card. The load and security tests that informed this are in [docs/engineering.md](docs/engineering.md).

## CI/CD

[`ci.yml`](.github/workflows/ci.yml) runs on every push and PR: install with the lockfile, ruff lint and format check, unit tests with an 80% coverage gate, then build the image, scan it with Trivy (fails on any fixable CRITICAL/HIGH), start the stack and run the integration tests over real HTTP. In CI a missing stack is an error, not a skip.

Environment-dependent tests, and why: the live Gemini tests need a key, so they are skipped on PRs and run in [`nightly-live.yml`](.github/workflows/nightly-live.yml) behind a repository variable and secret; the integration tests need the compose stack and skip locally when it is absent; the unit suite takes about 30 seconds rather than milliseconds, mostly importing scikit-learn and ADK.

Deployment, once there is a target: the same workflow tags the image with the git SHA and pushes it; a deploy job pulls that tag, waits for `/health` to show the new version, then shifts traffic. Rollback is the previous tag.

## If I had more time

The model is the weak point, and the fix is a feature, not a bigger model: the port-event feed as a *trailing* congestion measure (events and delay minutes at both ports in the 14 days before booking), which is the only column family I did not use and is legitimately known at booking time. I would also refit the calibrator on the most recent window, add incremental loads to the pipeline, give the assistant streaming output and a retry policy for rate limits, and put authentication in front of both APIs.

## Scaling to production

At millions of shipments and 24/7 use, the first thing that breaks is the deployment shape, not the database: today new data means a new image. The warehouse would move to object storage with an incremental pipeline on a scheduler, and serving reads — point lookups and pagination — would move to Postgres with route aggregates precomputed, keeping DuckDB and Parquet for analytics and training. That is the decision I would revisit first.

The model would need a feature store once port-event features exist, because they are time-dependent and point-in-time correctness is the only thing standing between "trailing congestion" and the leakage I avoided here. Serving would move from "load joblib at startup" to a registry with the champion/challenger gate run on every retrain. The assistant's session store becomes Postgres, and the real work there is cost and abuse control: per-user budgets, prompt caching, and the eval set run on every prompt change. Operationally: replicas behind a load balancer with `/health` as the readiness probe, central logs joined on `request_id`, and scheduled image rebuilds so base-image patches land without a code change. None of it needs Kubernetes to start.

## Assumptions

- Timestamps carry no timezone and are treated as UTC.
- The API serves the curated layer only: duplicate ids resolve to one row, and the 94 quarantined shipments return 404.
- `/shipments` date filters apply to `booking_date`, the only date every shipment has.
- Impossible numbers (negative weight, container counts of 0 or above 100) are nulled and flagged rather than corrected, because any correction would be a guess.
- Route stats are computed over completed shipments; the shipment count includes cancelled ones.
- "This quarter" means the quarter of the latest booking in the data, not today's date.

## AI usage

I used Google's Antigravity as my coding agent throughout, as the brief invites. It was most useful for the mechanical parts: scaffolding the FastAPI service and the test suites, turning my DQ rules into SQL and fixtures, generating the model comparison harness from my evaluation design, and drafting documentation that I then rewrote. I also used it as a second pair of eyes — a strict review pass found the calendar-day bug in `transit_days` and a request-id header missing on unhandled errors, both of which I fixed.

The design and the judgment calls are mine. I chose DuckDB over SQLite and Postgres and wrote the reasoning about when that choice stops being right. I chose a distroless image after looking at the CVE and wheel evidence for Alpine and Chainguard, and decided to build the warehouse at image-build time so the service is stateless. I decided to quarantine placeholder ports rather than repair them, and to null impossible numbers rather than guess. I designed the time-split evaluation and the ship/no-ship gate, and when the first run shipped the base rate I made the call to revise the rule and ship a calibrated model that is only marginally better than chance — and to say exactly that in the write-up. I chose Gemini with ADK for the assistant, required it to ask back rather than guess, and kept it in its own container behind a gateway so the data API never holds a secret. Where the agent's output disagreed with one of those decisions, I changed the output.
