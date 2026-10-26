# Docs

| Path | What it is |
|---|---|
| [`assignment/Take_Home_Assignment.pdf`](assignment/Take_Home_Assignment.pdf) | The original brief as received by email: scope, required endpoints, README sections, rubric |
| [`../data/DATA_DETAILS.md`](../data/DATA_DETAILS.md) | Column-level description of the supplied CSVs and first-pass quality findings |

## Brief at a glance

Three parts of equal weight, time-boxed to about 14 hours:

1. **Engineering:** API (`/health`, `/shipments`, `/shipments/{id}`, `/routes/{o}/{d}/stats`, `/predict-delay`, `/data-quality/report`), structured errors, JSON logs, unit and integration tests, Docker, CI, and a runbook.
2. **Data:** a standalone DQ check against raw CSVs, then a layered and idempotent pipeline into a store, a curated data model, and the tech-choice write-up.
3. **ML + GenAI:** a booking-time delay model (no leakage, honest metrics, committed artefact) and a tool-calling assistant CLI with guardrails and full LLM logging.

Note: the brief asks for "all twelve DQ issues described in Section 2.1", but Section 2.1 does not list them. We find and document them ourselves.
