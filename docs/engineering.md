# Engineering notes

Detail that backs the README: how the image got small, what the security and load tests found, and the per-request API contract. Everything here is re-runnable from the repo.

## API contract

| Endpoint | Notes |
|---|---|
| `GET /health` | Readiness: `200 ok` or `503 degraded`, with per-dependency `checks` (data store, model, DQ report) and the version |
| `GET /shipments` | Filters `origin`, `destination`, `status`, `booked_from`, `booked_to` (on `booking_date`; `start_date`/`end_date` accepted as aliases); `limit` (≤500) / `offset`; returns `{items, total, limit, offset}` |
| `GET /shipments/{id}` | Full row plus derived `route_key`, `actual_delay_hours`, `on_time_flag`, `transit_days_*` |
| `GET /routes/{origin}/{destination}/stats` | `shipment_count`, `completed_count`, `avg_delay_hours`, `on_time_rate` (over completed shipments only) |
| `GET /routes/stats` | Per-route aggregates ranked by `sort` and `order` in a booking window, with a `min_completed` floor. "Which routes had the highest delay this quarter" is one call |
| `GET /ports` | Reference list of ports; the assistant uses it to turn "Shanghai" into `CNSHA` |
| `POST /predict-delay` | `{"shipment_id": "SHP-00421"}` **or** `{"features": {...booking-time fields}}`; returns `delay_risk`, `risk_label`, `model_version`, `features_used`. Extra fields are rejected, so a post-booking column cannot be slipped in |
| `GET /data-quality/report` | The JSON written by the standalone DQ step; `503` if it has not been generated |

Every error has one shape, `{"error": {"code", "message", "request_id", "details"}}`: `404` for an unknown shipment or route, `422` for bad input, `503` when a dependency is missing, `500` with no internal detail for anything unexpected. Every response carries `x-request-id` (the client's if it looks like one, otherwise generated), and the same id is on the request log line. Unhandled 500s are answered by Starlette outside the logging middleware, so the error helper sets the header itself; there is a regression test for that.

```bash
curl "localhost:8000/shipments?origin=CNSHA&destination=NLRTM&limit=5"
curl localhost:8000/routes/CNSHA/NLRTM/stats
curl -X POST localhost:8000/predict-delay -H 'content-type: application/json' -d '{"shipment_id":"SHP-00421"}'
```

## Configuration

Everything that varies by environment lives in [`src/core/config.py`](../src/core/config.py): a pydantic-settings object read from the environment and `.env` (copy [`.env.example`](../.env.example)), plus the domain tunables (on-time threshold, container range, ML split dates, assistant caps, model prices) as named constants. No module defines its own inline constants.

## Image size and CVEs

The first build was `python:3.13-slim` + `uvicorn[standard]`: 333 MB, and a scan showed HIGH CVEs in urllib3, msgpack and setuptools. None of those are in the dependency tree — they are the copies vendored inside pip in the base image, which the service never uses. What I changed, in order of effect:

| Change | Effect |
|---|---|
| Runtime stage on `gcr.io/distroless/python3-debian13` | −120 MB; removes pip (and its vendored CVEs), the shell, apt, tar. Same Debian 13 / glibc / Python 3.13 as the builder, so compiled wheels carry over |
| `uvicorn` instead of `uvicorn[standard]` | −24 MB; none of the extras are needed for one worker |
| uv cache mount in the builder | Cached rebuild in ~4 s |
| Compose: `read_only`, `cap_drop: [ALL]`, `no-new-privileges` | The process cannot write to disk or gain privileges even if compromised |
| Strip `tests/` dirs and bytecode from site-packages | −200 MB once scikit-learn, SciPy, NumPy and pandas joined for the model; costs ~0.7 s of cold start |

Result: **214 MB** for the API alone, **492 MB** with the ML stack (about 280 MB of that is SciPy/NumPy/pandas/scikit-learn with two bundled OpenBLAS copies). Exporting the model to ONNX to drop the stack was not worth it for a 1 MB model whose transform is a pandas function.

Trivy on the final image: 0 fixable OS CVEs, 0 CRITICAL/HIGH Python-package CVEs. One fixable MEDIUM remains, `opentelemetry-api` 1.44.0, because `google-adk` pins `<=1.44` and uv resolves one version for the whole lock; it clears when ADK lifts the pin. The rest are Debian advisories with no fix yet, which no Debian-based image avoids; CI runs Trivy with `--ignore-unfixed` so the build fails the day a fix exists that we have not picked up.

Alternatives I looked at: Alpine (DuckDB ships no musl wheels) and Chainguard Python (the free tag is a floating `latest`, currently Python 3.15; pinned tags are paid).

```bash
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest image supply-chain-intel:local
```

## Security test

`uv run python scripts/sectest.py` runs 76 black-box checks against the running container: injection payloads on every parameter, input bounds, header handling, method tampering, information exposure, abuse, and the container configuration. Plus six adversarial assistant cases.

- **Injection cannot reach the engine.** Every query is parameterised; `sort`/`order` are `Literal`-validated; port codes and statuses are validated before SQL. 28/28 payloads came back as 404, 422 or an empty page.
- **No leaks.** Errors never carry a traceback or an internal path. Non-finite floats, out-of-range offsets and oversized request ids originally produced 500s; they are now 422s, with regression tests.
- **Container.** Distroless, non-root (uid 65532), read-only filesystem, all capabilities dropped, `no-new-privileges`, no secrets in the image environment. The Gemini key lives only in the caller's `.env`, gitignored and never logged.
- **Assistant.** Prompt injection ("print your system prompt") is refused — the key is not in the model's context at all; SQL-injected tool arguments are blocked before any HTTP call; bulk exfiltration hits the row cap and `query_shipments` never projects `customer_id`.
- **Left out on purpose:** authentication and rate limiting. The service is single-tenant and local with synthetic data. In production both belong at the gateway; `--limit-concurrency` on uvicorn sheds load with a 503 in the meantime. The assistant API trusts `user_id` from the request, so anyone can list or resume another user's sessions — fine locally, not beyond. When the assistant profile is off, Caddy answers `/assistant/*` with a plain 502 rather than the JSON envelope.

## Load test

`uv run python scripts/loadtest.py --scenario {mix,read,predict,heavy} --concurrency 50 --duration 30` is seeded and writes `reports/loadtest_<scenario>.json`; it also probes `/health` from a separate client so starvation shows up.

On a laptop with one uvicorn worker: read @20 gave 100 rps at p99 0.42 s, @100 gave 46 rps at p99 6.8 s; heavy queries @20 gave 39 rps at p99 0.83 s. No 5xx in 11.8k requests. After a 200-concurrent spike the service was back to p50 ≈ 45 ms within 4 s.

The ceiling was `/predict-delay` at about 4 rps: one call ran at 700% CPU for up to 2.9 s because every request thread spawned its own OpenMP/BLAS pool. Fixes: `OMP_NUM_THREADS=1` and friends in the image, a model warm-up in the lifespan, and `--limit-concurrency 128` so overload returns 503 instead of queueing. A single predict is now ~40 ms and predict @20 doubled to 16 rps. The remaining ceiling is the GIL around the pandas transform; the next step is more workers, not more threads. At scale, `/shipments` (a `count(*)` plus an `OFFSET` scan) wants keyset pagination and `/routes/stats` wants a precomputed table.
