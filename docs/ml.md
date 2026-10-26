# ML: delay prediction

The short version is in the README. This is the full evaluation.

## Task and data

Predict at booking time whether a shipment arrives more than 24 hours late. Training reads the curated warehouse (`python -m ml.train`, about three minutes). Rows with no actual arrival (cancelled or missing) have no label and are excluded.

Time split on `booking_date`: train before 2025-04-01 (2,741 rows, 39.9% late), validation April–May 2025 (570 rows, 44.2%), test from 2025-06-01 (1,253 rows, 39.9%). No random K-fold: it would mix future bookings into training and inflate every number. Hyper-parameters are chosen on validation only; test is scored once; AUC confidence intervals come from a 1,000-resample bootstrap.

## Features

All known at booking — the list is `BOOKING_TIME_FEATURES` in [`src/ml/predictor.py`](../src/ml/predictor.py): origin and destination port, cargo type, container count, weight, planned departure and planned arrival. From these the pipeline derives planned transit days, departure month and weekday. The whole transform (cargo normalisation, nulling of impossible numerics, one-hot with an "infrequent" bucket so unseen ports do not crash, median imputation with missing indicators) lives inside the scikit-learn Pipeline, so the API passes raw dicts and training and serving cannot drift.

**Excluded, and why.** `status` is a perfect leak (`DELAYED` ⇔ late on every row). `actual_departure`, `actual_arrival` and `transit_days_actual` are post-booking. `dq_flags` are post-hoc. `customer_id`, `vessel_id` and `route_key` looked informative in a static profile — customer late rates range from 10% to 74% — but their train→validation late-rate correlation is about zero (customer −0.04, route −0.16, vessel +0.02): the spread is binomial noise from 5–34 shipments per group. Even the ports are unstable on a 570-row validation window (origin +0.24, destination −0.44 across 25 groups), which is why the lift is as small as it is. A target-encoding ablation confirmed it (test AUC 0.488 → 0.489).

## Candidates

All metrics on the test set. Precision, recall and F1 are given at the default 0.5 cut-off and at each model's validation-tuned threshold (in brackets): at a 40% base rate the 0.5 cut-off makes every model look useless and the tuned one makes every model look like "always late", so both are shown. Log loss, CV stability, artefact size and latency are in [`models/model_card.md`](../models/model_card.md).

| Candidate | Val AUC | Test AUC [95% CI] | PR-AUC (base 0.399) | Brier | ECE | P / R / F1 @0.5 | P / R / F1 @tuned |
|---|---|---|---|---|---|---|---|
| Prior (train base rate) | 0.500 | 0.500 | 0.399 | **0.2398** | **0.000** | 0 / 0 / 0 | 0.399 / 1.000 / 0.570 (0.399) |
| Route-history baseline | 0.465 | 0.494 [0.460, 0.525] | 0.397 | 0.2434 | 0.053 | 0.407 / 0.044 / 0.079 | 0.399 / 1.000 / 0.570 (0.236) |
| Logistic regression (C=0.01) | 0.498 | 0.488 [0.456, 0.519] | 0.391 | 0.2411 | 0.026 | 0 / 0 / 0 | 0.397 / 0.988 / 0.567 (0.316) |
| HistGradientBoosting (uncalibrated) | **0.527** | **0.535 [0.502, 0.567]** | **0.431** | 0.2529 | 0.090 | 0.421 / 0.254 / 0.317 | 0.398 / 0.982 / 0.566 (0.133) |
| **HGB + sigmoid calibration (shipped)** | **0.527** | **0.535 [0.502, 0.567]** | **0.431** | 0.2407 | 0.043 | 0.667 / 0.004 / 0.008 | 0.398 / 0.982 / 0.566 (0.398) |
| Random forest | 0.500 | 0.505 [0.476, 0.537] | 0.405 | 0.2426 | 0.055 | 0.423 / 0.044 / 0.080 | 0.401 / 0.888 / 0.553 (0.333) |
| LR + target encoding (ablation) | 0.512 | 0.489 [0.457, 0.521] | 0.390 | 0.2411 | 0.022 | 0 / 0 / 0 | 0.398 / 0.990 / 0.568 (0.311) |

## Selection and provenance

The rule: among the *uncalibrated* learned models, pick the best validation AUC (tie-break on validation log loss) — HGB at 0.527. Calibrate only that model on validation. Ship it only if, on test, the AUC confidence interval's lower bound is above 0.50 **and** its Brier score is no worse than the prior's within 0.003; otherwise ship the prior. Nothing fitted on validation labels takes part in choosing the model, and the test set never chooses between models, only whether to ship. With 570 validation rows no candidate can clear a strict significance bar, so the gate is "measurably better than chance and honestly calibrated". Raw HGB would fail on calibration (ECE 0.09); calibrated, it passes — barely.

Provenance, recorded in the bundle, `metrics.json` and the card: base HGB fit on train; sigmoid calibrator fit on validation; threshold 0.398 chosen on validation (max F1); test scored once for the gate.

An honest caveat: the first version of this rule shipped the prior, and I revised the rule after seeing that result, so it is not fully pre-registered. With the gate passed by 0.002 of AUC-CI and 0.0009 of Brier, read the shipped model as "indistinguishable from a calibrated base rate, with a small ranking signal". Logistic regression, which the brief rightly favours by default, is the best-calibrated learned model but is not better than chance here; a well-evaluated LR in this case means reporting that and moving on.

## What the product does with it

Scores on test run from 0.38 to 0.51 (median 0.44): almost every shipment is `medium`, and that is the honest shape of the signal, not a bug. Read the score as "the calibrated base rate, nudged by port and cargo". One known defect: the calibrator learned the validation window's higher late rate, so test scores run about 4 points high (mean 0.439 versus observed 0.399). Refitting the calibrator on the most recent window is the first thing to do before a real deployment. The always-late baseline has F1 0.57, so F1 alone would make any model look useful; lead with AUC, PR-AUC and Brier.

## Monitoring and retraining

Every `/predict-delay` call logs `request_id`, `model_version`, the score, the risk label and the raw features as one JSON line. From that: weekly PSI on the score distribution and on the port/cargo mix versus training, plus the share of unseen categories; once arrivals land (lag up to about 50 days), rolling 30-day AUC, Brier and calibration error.

Retrain when any of these trip: the AUC interval falls below the route baseline, calibration error exceeds 0.05, PSI exceeds 0.2, the base rate shifts by more than 5 points — or monthly, always gated by the same time-split champion/challenger rule. The committed artefact is the exact model that was evaluated, not a refit on all data, so the metrics describe what runs.
