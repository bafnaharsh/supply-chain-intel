# GenAI assistant

A CLI and HTTP agent built on Google ADK with Gemini (`gemini-3.5-flash-lite` by default). It answers questions about shipments, routes, ports and delay risk by calling the running data API over HTTP — it never touches the warehouse directly, so it sees exactly what any API client sees.

```bash
uv sync --all-extras                                   # pulls the optional google-adk extra
docker compose up -d --wait                            # the data API
uv run python -m assistant "Which routes have had the highest average delay this quarter?"
uv run python -m assistant                             # REPL; prints the session id to resume later
uv run python -m assistant --session <id> "and for Rotterdam?"
uv run python -m assistant --list-sessions
uv run python -m assistant --offline "Which ports are there?"   # no key, no LLM: fixed shapes only
docker compose --profile assistant up --wait           # the same agent as an HTTP API behind the gateway
```

## Tools

The model decides when to call them: `list_ports`, `resolve_port(name_or_code)`, `query_shipments(filters)`, `get_shipment(id)`, `get_route_stats(origin, destination)`, `top_routes(window, min_completed, sort, order, limit)`, `predict_delay(shipment_id)` (which calls `POST /predict-delay`), and `data_period()`.

## Normalising and asking back

People say "Shanghai", "shp 421", "this quarter". The agent resolves port names through `resolve_port` — an exact match gives the code, one obvious typo like "Rotterdm" is corrected and the correction is said out loud, several candidates come back as ambiguous and the model must ask. It normalises ids ("421" → `SHP-00421`) and anchors "this quarter" to the data's latest booking date (2025-10-14, so Q4 2025), never the wall clock. Before quoting a delay risk it calls `get_shipment`, so a shipment that is already delivered or cancelled gets its actual outcome alongside the forecast.

When information is missing — a route with no destination, no shipment id, a window outside the data, "worst route" with no metric — the instruction requires one concise clarifying question with real options, and forbids calling data tools until it is answered. Observed live: "What is the on-time rate from Shanghai?" → "Shanghai (CNSHA) to which destination? Rotterdam (NLRTM), Los Angeles (USLAX), …"; "Rotterdam" in the resumed session → 75.0% (n=12, `get_route_stats`).

## Sessions

ADK's `DatabaseSessionService` persists every event — user turns, model text, tool calls and results — and the session state to SQLite (`data/assistant_sessions.db` on the host, the `assistant-var` volume in compose). `--session <id>` resumes with full history; `--no-persist` keeps it in memory. The data period is cached in session state so "this quarter" cannot drift mid-conversation.

## Guardrails

All live in ADK callbacks and are unit-tested without a key.

| Guardrail | Where | Behaviour |
|---|---|---|
| LLM-call cap (8 per turn) | `before_model` + `RunConfig.max_llm_calls` | Fixed message, `guardrail_triggered` logged |
| Tool-call cap (6 per turn) | `before_tool` | An error dict goes back to the model instead of an HTTP call |
| Argument validation | `before_tool` | Port codes, ISO dates, `booked_from < booked_to`; a bad call never reaches the API and the model is told to `resolve_port` |
| Grounding | `after_model` | A final answer with a number, in a turn where no tool succeeded, is replaced with "I can only answer from shipment data…". Shipment ids, quarter labels, the 24 h threshold and clarifying questions are exempt — the evaluation showed the first version blocked "could you give the id, e.g. SHP-00421?" |
| Off-domain refusal and citation | instruction | One-sentence refusal; every figure names the tool that produced it and the window and n behind it |
| Tools never raise | tools | API and network errors become `{status: "error", hint}` so the model can recover or ask |

## Logging

`logs/assistant.jsonl` plus stderr, through the same JSON formatter as the API; every record carries `session_id` and `user_id`. Record types: `llm_prompt` once per turn (the complete prompt as sent — system instruction with the port table, tool declarations, the conversation so far), `llm_call` per model call (what is new since the model last spoke), `llm_response` (parts, prompt and output tokens, estimated USD, latency), `tool_call` (tool, args, status, latency, result size), `final_answer` (what the user actually receives, after any guardrail), and `guardrail_triggered`. A fake-LLM unit test drives the real ADK runner against the fixture API and asserts the records are emitted.

## Cost per query

Measured on live runs at $0.30 / $2.50 per million input/output tokens: the instruction, port table and eight tool schemas are about 1.9k input tokens per call. A clarifying question or a refusal is one call, about **$0.0007**; a data answer is three calls (plan → tool → answer), about **$0.0023**; the 8-call cap bounds a turn at about $0.006. `gemini-3.8-flash` is roughly 2.5× that.

## As an HTTP API

`assistant.api` wraps the same agent, sessions, guardrails and logs in a small FastAPI app. In compose it is its own container, built from the `assistant` Dockerfile target, behind the gateway at `/assistant/*`. The key comes from `.env` at run time and is never in an image; the container keeps the same hardening as the data API, with the `assistant-var` named volume for the session store and the log. Without a key it starts in a degraded state: `/assistant/health` says so and `/assistant/chat` answers `503 llm_unconfigured`.

`POST /assistant/chat {"question": "...", "session_id": "<optional>"}` returns the answer, the session id to continue with, the tools called, token usage and estimated cost, and which guardrail fired if any. `GET /assistant/sessions` lists a user's sessions; `GET /assistant/health` reports whether the data API is reachable and which model is configured.

Why a gateway rather than one process: the data API must run without secrets and without the ADK stack, and the assistant must have the key. Two processes with different trust, one entry point in front — the same shape an API gateway gives in production, which is also where auth and rate limiting would go.

## Offline mode

`--offline` needs no key and does not load ADK; the CLI also drops into it by itself, with one line on stderr, when `.env` has no key. A small router answers a fixed set of question shapes with the same tools: route stats ("on-time rate from X to Y"), rankings ("highest average delay this quarter", "worst on-time rate", "busiest"), delay risk and status of a shipment, the port list, and counts ("how many shipments from X to Y delivered in 2024-06"). There are no regular expressions in it: the question is tokenized once, port names are a dictionary lookup against the API's own port list (a lone misspelt word after "from"/"to" goes through `resolve_port`), windows are parsed as dates, and the intent is a keyword-set test. Anything else prints the list of supported shapes. It is deterministic, stores no sessions, and every footer starts with `[offline mode: no LLM]`. It is not the assistant; it is a demo path so a reviewer without a key can still try the CLI.

## Evaluation

`uv run python evals/run_eval.py` runs 25 questions with SQL/API ground truth ([`evals/questions.json`](../evals/questions.json)) through the real agent, each in a fresh session at temperature 0 (the CLI and API use 0.1), and scores tool choice, values (±0.5 pp / ±0.5 h), behaviour (answer / clarify / refuse), groundedness and tool citation.

| | Route stats | Rankings | Shipment lookups | Clarify | Guardrails | **All** |
|---|---|---|---|---|---|---|
| Passed | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 | **24/25** |

Right tools 25/25, values 20/20, grounded 25/25, cited 13/13; about $0.046 for the run (about $0.002 per question) and 2.2 s per question. The first run scored 17/25 and found three systematic gaps: no tool fetched a shipment by id (so the model said "not found" for a shipment that existed), `top_routes` had no ascending sort (it paged five times hunting for the worst route and hit the call cap), and the grounding guard blocked clarifying questions that quoted an example id. All three were fixed. The remaining miss is the scorer's: "Please provide your shipment ID (for example, SHP-00421)…" is the right clarification, phrased as an imperative, which the question-mark rule did not count; it now does. Results are recorded as scored at the time ([`evals/results_latest.json`](../evals/results_latest.json)).

## Tests

30 assistant unit tests with no network and no key: the tools against the in-process fixture API, id and port normalisation, each guardrail, the full loop through a fake LLM, session persistence across two service instances on one SQLite file, and the degraded no-key mode. `tests/integration/test_assistant_live.py` runs the three brief questions plus the clarifying case against real Gemini; it is skipped unless a key and `BASE_URL` are set and runs only in the nightly workflow.
