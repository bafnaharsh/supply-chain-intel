# Docs

| File | What it covers |
|---|---|
| [`assignment/Take_Home_Assignment.pdf`](assignment/Take_Home_Assignment.pdf) | The brief as received |
| [`engineering.md`](engineering.md) | API contract and errors, configuration, image size and CVEs, security test, load test |
| [`data.md`](data.md) | Every data-quality finding with the decision, the passing guard checks, the four-layer data model |
| [`ml.md`](ml.md) | Features and leakage reasoning, the candidate comparison, selection rule and provenance, monitoring |
| [`assistant.md`](assistant.md) | Tools, clarifying behaviour, sessions, guardrails, logging, cost, HTTP API, offline mode, evaluation |
| [`../data/DATA_DETAILS.md`](../data/DATA_DETAILS.md) | Column-level description of the supplied CSVs |

The brief, in one paragraph: three equal parts, about fourteen hours. Engineering — an API with six endpoints, structured errors and logs, tests, Docker, CI and a runbook. Data — a standalone DQ check on the raw CSVs, a layered idempotent pipeline into a store, a documented data model. ML and GenAI — a booking-time delay model with honest metrics and a committed artefact, and a tool-calling assistant with a guardrail and full LLM logging. The brief refers to "twelve" seeded data issues without listing them; finding them is part of the exercise.
