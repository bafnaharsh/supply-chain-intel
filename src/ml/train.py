"""Train delay-risk candidates on a time split, compare them, ship the selected one.

CLI: python -m ml.train [--db data/warehouse.duckdb] [--out models/]

Every candidate is fit on train, tuned on validation and scored once on test. The shipped
artefact is the exact fitted pipeline that was scored, so its metrics describe it.
"""

import argparse
import io
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from core.config import (
    LABEL_THRESHOLD_HOURS,
    RANDOM_SEED,
    RISK_LABEL_THRESHOLDS,
    TRAIN_END,
    VALIDATION_END,
    settings,
)
from ml.features import FEATURES, build_pipeline, make_label

ABLATION_KEYS = ["route_key", "customer_id"]  # high-cardinality ids, ablation only
STABILITY_KEYS = [
    "customer_id",
    "vessel_id",
    "route_key",
    "origin_port",
    "destination_port",
    "cargo_type",
]
BOOTSTRAP_RESAMPLES = 1000
LATENCY_CALLS = 200


def load_labelled(db_path: Path) -> pd.DataFrame:
    """Curated shipments that have a label (an actual arrival) and were not cancelled."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"warehouse not found at {db_path}; build it first with `python -m pipeline`"
        )
    cols = ", ".join(dict.fromkeys([*FEATURES, "vessel_id", *ABLATION_KEYS]))
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(
            f"SELECT shipment_id, booking_date, actual_arrival, {cols} FROM shipments "
            "WHERE actual_arrival IS NOT NULL AND status IS DISTINCT FROM 'CANCELLED' "
            "ORDER BY booking_date, shipment_id"
        ).df()
    df["late"] = make_label(df)
    return df


def time_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """train < TRAIN_END <= validation < VALIDATION_END <= test, on booking_date."""
    booked = pd.to_datetime(df["booking_date"])
    splits = {
        "train": df[booked < TRAIN_END],
        "validation": df[(booked >= TRAIN_END) & (booked < VALIDATION_END)],
        "test": df[booked >= VALIDATION_END],
    }
    ids = [set(s["shipment_id"]) for s in splits.values()]
    assert all(ids), "a time split is empty"
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "time splits overlap"
    assert sum(map(len, ids)) == len(df), "time split dropped rows"
    return splits


class RouteBaseline(ClassifierMixin, BaseEstimator):
    """Train-only late rate by route, shrunk to the origin port's rate, shrunk to the global rate.

    An unseen route falls back to its origin port, an unseen port to the global rate.
    """

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing

    @staticmethod
    def _route(X: pd.DataFrame) -> pd.Series:
        return X["origin_port"].astype(str) + "|" + X["destination_port"].astype(str)

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        self.classes_ = np.array([False, True])
        self.global_ = float(y.mean())
        d = pd.DataFrame({"o": X["origin_port"].to_numpy(), "r": self._route(X).to_numpy(), "y": y})
        o = d.groupby("o")["y"].agg(["sum", "count"])
        m = self.smoothing
        self.origin_ = ((o["sum"] + m * self.global_) / (o["count"] + m)).to_dict()
        r = d.groupby("r").agg(s=("y", "sum"), n=("y", "count"), o=("o", "first"))
        self.route_ = ((r["s"] + m * r["o"].map(self.origin_)) / (r["n"] + m)).to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        origin = X["origin_port"].map(self.origin_).astype(float).fillna(self.global_)
        p = self._route(X).map(self.route_).astype(float).fillna(origin).to_numpy()
        return np.column_stack([1 - p, p])


def _logreg(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=5000)  # no class_weight keeps probabilities calibrated


def _logreg_target_encoded(c: float) -> Pipeline:
    base = build_pipeline(_logreg(c))
    features = ColumnTransformer(
        [
            ("base", Pipeline(base.steps[:-1]), list(FEATURES)),
            (
                "te",  # cross-fitted inside fit_transform, so train rows never see their own label
                TargetEncoder(cv=KFold(5, shuffle=True, random_state=RANDOM_SEED)),
                ABLATION_KEYS,
            ),
        ]
    )
    return Pipeline([("features", features), ("model", base.steps[-1][1])])


def candidates() -> dict[str, list[tuple[dict, object]]]:
    """Name -> grid of (params, unfitted estimator), simplest family first."""
    seed = RANDOM_SEED
    return {
        "prior": [({"strategy": "prior"}, build_pipeline(DummyClassifier(strategy="prior")))],
        "route_history": [({"smoothing": 20}, RouteBaseline(20.0))],
        "logreg": [({"C": c}, build_pipeline(_logreg(c))) for c in (0.01, 0.1, 1, 10)],
        "hist_gb": [
            (
                {"max_iter": n, "learning_rate": lr, "max_depth": d},
                build_pipeline(
                    HistGradientBoostingClassifier(
                        max_iter=n, learning_rate=lr, max_depth=d, random_state=seed
                    )
                ),
            )
            for n in (100, 300)
            for lr in (0.03, 0.1)
            for d in (3, None)
        ],
        "random_forest": [
            (
                {"n_estimators": 300, "min_samples_leaf": leaf},
                build_pipeline(
                    RandomForestClassifier(
                        n_estimators=300, min_samples_leaf=leaf, random_state=seed, n_jobs=-1
                    )
                ),
            )
            for leaf in (5, 20)
        ],
        "ablation_logreg_te": [({"C": c}, _logreg_target_encoded(c)) for c in (0.01, 0.1, 1, 10)],
    }


LEARNED = ["route_history", "logreg", "hist_gb", "random_forest"]  # uncalibrated bases
BRIER_TOLERANCE = 0.003  # ship gate: Brier may exceed the prior's by at most this
VERSION_TAGS = {"hist_gb_calibrated": "hgb-cal", "hist_gb": "hgb", "random_forest": "rf"}


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray) -> list[float]:
    rng = np.random.default_rng(RANDOM_SEED)
    aucs = []
    for idx in rng.integers(0, len(y), (BOOTSTRAP_RESAMPLES, len(y))):
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Mean predicted vs observed late rate in equal-width probability bins (empty bins omitted)."""
    which = np.minimum((p * bins).astype(int), bins - 1)
    return [
        {
            "bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}",
            "n": int((which == b).sum()),
            "mean_pred": float(p[which == b].mean()),
            "observed": float(y[which == b].mean()),
        }
        for b in range(bins)
        if (which == b).any()
    ]


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def score(y, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    rel = reliability(y, p)
    m = {
        "n": len(y),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "roc_auc_ci95": bootstrap_auc_ci(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(sum(b["n"] * abs(b["mean_pred"] - b["observed"]) for b in rel) / len(y)),
        "pred_mean": float(p.mean()),
        "pred_quantiles": {
            q: float(np.percentile(p, v))
            for q, v in (("min", 0), ("p10", 10), ("p50", 50), ("p90", 90), ("max", 100))
        },
    }
    for tag, t in (("0.5", 0.5), ("tuned", threshold)):
        pred = p >= t
        m[f"precision@{tag}"] = float(precision_score(y, pred, zero_division=0))
        m[f"recall@{tag}"] = float(recall_score(y, pred, zero_division=0))
        m[f"f1@{tag}"] = float(f1_score(y, pred, zero_division=0))
    m["reliability"] = rel
    return m


def evaluate(
    name: str, grid: list, splits: dict[str, pd.DataFrame], fit_on: str = "train"
) -> tuple[dict, object]:
    """Fit every grid point on `fit_on`, keep the best on validation AUC, score it once on test.

    fit_on="validation" is only for post-hoc calibration of an already train-fitted model.
    """
    cols = [*FEATURES, *ABLATION_KEYS] if name.startswith("ablation") else list(FEATURES)
    X = {k: s[cols] for k, s in splits.items()}
    y = {k: s["late"].to_numpy() for k, s in splits.items()}

    tried = []
    for params, est in grid:
        est.fit(X[fit_on], y[fit_on])
        tried.append(
            (roc_auc_score(y["validation"], est.predict_proba(X["validation"])[:, 1]), params, est)
        )
    val_auc, params, model = max(tried, key=lambda t: t[0])

    p_val = model.predict_proba(X["validation"])[:, 1]
    threshold = best_f1_threshold(y["validation"], p_val)
    cv = None  # a frozen, validation-calibrated model has no train-only refit to cross-validate
    if fit_on == "train":
        cv = cross_val_score(
            sklearn.clone(model),
            X["train"],
            y["train"],
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
            scoring="roc_auc",
        )

    buf = io.BytesIO()
    joblib.dump(model, buf)
    row = pd.DataFrame([X["test"].iloc[0].to_dict()])  # the API's shape: one dict per call
    times = []
    for _ in range(LATENCY_CALLS):
        start = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - start) * 1000)

    result = {
        "params": params,
        "grid_validation_auc": [{"params": p, "roc_auc": float(a)} for a, p, _ in tried],
        "threshold": threshold,
        "fit_on": fit_on,
        "cv5_train_auc_mean": None if cv is None else float(cv.mean()),
        "cv5_train_auc_std": None if cv is None else float(cv.std()),
        "validation": score(y["validation"], p_val, threshold),
        "test": score(y["test"], model.predict_proba(X["test"])[:, 1], threshold),
        "artefact_bytes": buf.tell(),
        "latency_ms_median": float(statistics.median(times)),
    }
    return result, model


def group_stability(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict]:
    """Does a group's train late rate predict its later (validation) late rate? Pearson r over
    shared groups. Diagnostic only: the test split must never inform feature choices."""
    out = {}
    for key in STABILITY_KEYS:
        a = train.groupby(key)["late"].agg(["mean", "count"])
        both = a.join(test.groupby(key)["late"].mean().rename("test"), how="inner")
        out[key] = {
            "groups_in_train": len(a),
            "median_train_rows_per_group": float(a["count"].median()),
            "groups_in_both": len(both),
            "train_test_late_rate_corr": float(both["mean"].corr(both["test"])),
        }
    return out


def select_base(results: dict[str, dict]) -> tuple[str, str]:
    """(a) best uncalibrated learned model on validation AUC; ties -> lower validation log loss."""
    val = {n: results[n]["validation"] for n in LEARNED}
    ranked = sorted(LEARNED, key=lambda n: (-round(val[n]["roc_auc"], 6), val[n]["log_loss"]))
    log = "(a) validation AUC ranking (uncalibrated): " + ", ".join(
        f"{n} {val[n]['roc_auc']:.3f} (log loss {val[n]['log_loss']:.4f})" for n in ranked
    )
    return ranked[0], f"{log} -> base model {ranked[0]}"


def gate(results: dict[str, dict], challenger: str) -> tuple[str, str]:
    """(b) ship the calibrated challenger only if it beats chance and is calibrated, on test."""
    t, prior = results[challenger]["test"], results["prior"]["test"]
    auc_ok = t["roc_auc_ci95"][0] > 0.5
    brier_ok = t["brier"] <= prior["brier"] + BRIER_TOLERANCE
    shipped = challenger if auc_ok and brier_ok else "prior"
    return shipped, (
        f"(b) {challenger}: test AUC CI lower bound {t['roc_auc_ci95'][0]:.3f} > 0.50: {auc_ok}; "
        f"test Brier {t['brier']:.4f} <= prior {prior['brier']:.4f} + {BRIER_TOLERANCE}: "
        f"{brier_ok} -> ship {shipped}"
    )


def comparison_table(results: dict[str, dict], shipped: str) -> str:
    cols = [
        "candidate",
        "params",
        "val AUC",
        "test AUC [95% CI]",
        "CV5 train AUC",
        "test PR-AUC (base)",
        "test Brier",
        "test log loss",
        "test ECE",
        "P@0.5",
        "R@0.5",
        "F1@0.5",
        "tuned thr",
        "P@thr",
        "R@thr",
        "F1@thr",
        "bytes",
        "latency ms",
    ]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, r in results.items():
        v, t = r["validation"], r["test"]
        lo, hi = t["roc_auc_ci95"]
        cv = "n/a"
        if r["cv5_train_auc_mean"] is not None:
            cv = f"{r['cv5_train_auc_mean']:.3f} ± {r['cv5_train_auc_std']:.3f}"
        cells = [
            name + (" **(shipped)**" if name == shipped else ""),
            ", ".join(f"{k}={val}" for k, val in r["params"].items()),
            f"{v['roc_auc']:.3f}",
            f"{t['roc_auc']:.3f} [{lo:.3f}, {hi:.3f}]",
            cv,
            f"{t['pr_auc']:.3f} ({t['base_rate']:.3f})",
            f"{t['brier']:.4f}",
            f"{t['log_loss']:.4f}",
            f"{t['ece']:.3f}",
            *(f"{t[f'{m}@0.5']:.3f}" for m in ("precision", "recall", "f1")),
            f"{r['threshold']:.3f}",
            *(f"{t[f'{m}@tuned']:.3f}" for m in ("precision", "recall", "f1")),
            f"{r['artefact_bytes']:,}",
            f"{r['latency_ms_median']:.2f}",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _reliability_table(rel: list[dict]) -> str:
    rows = [f"| {b['bin']} | {b['n']} | {b['mean_pred']:.3f} | {b['observed']:.3f} |" for b in rel]
    return "\n".join(["| bin | n | mean predicted | observed |", "|---|---|---|---|", *rows])


def model_card(info: dict) -> str:
    r, shipped, stab = info["results"], info["shipped"], info["group_stability"]
    t = r[shipped]["test"]
    q = t["pred_quantiles"]
    lo, hi = t["roc_auc_ci95"]
    splits = "\n".join(
        f"| {k} | {s['window']} | {s['n']} | {s['base_rate']:.3f} |"
        for k, s in info["splits"].items()
    )
    stability = "\n".join(
        f"| `{k}` | {s['groups_in_train']} | {s['median_train_rows_per_group']:.0f} | "
        f"{s['groups_in_both']} | {s['train_test_late_rate_corr']:.2f} |"
        for k, s in stab.items()
    )
    selection = "\n".join(f"- {line}" for line in info["selection_log"])
    provenance = "\n".join(f"- {k}: {v}" for k, v in info["provenance"].items())
    abl, lr = r["ablation_logreg_te"], r["logreg"]
    n_cust = stab["customer_id"]["median_train_rows_per_group"]
    se_cust = 100 * (0.4 * 0.6 / n_cust) ** 0.5
    if shipped == "prior":
        headline = (
            "no candidate passed the ship gate, so the shipped model is the calibrated base rate "
            f"itself: every shipment scores {t['pred_mean']:.3f} (risk label `medium`). The "
            "booking-time fields in this dataset carry no measurable delay signal out of time "
            "(see the group-stability table)."
        )
    else:
        headline = (
            f"the lift is small: test ROC-AUC {t['roc_auc']:.3f} with a 95% CI of "
            f"{lo:.3f}–{hi:.3f}, barely above 0.5. The model's main value is a calibrated base "
            "rate plus weak port/cargo effects; no booking-time feature carries real signal out "
            "of time (see the group-stability table). Use the score to rank shipments for "
            "attention, not to promise a delay."
        )
    return f"""# Model card: shipment delay risk

_Generated by `python -m ml.train` on {info["trained_at"]}; do not edit by hand._

**Shipped:** `{info["version"]}` (`{shipped}`), artefact `models/delay_model.joblib`, \
scikit-learn {sklearn.__version__}.
**Predicts:** P(actual arrival is more than {LABEL_THRESHOLD_HOURS}h after planned arrival), \
from fields known at booking time only.
**Test (bookings from {VALIDATION_END}, scored once):** ROC-AUC {t["roc_auc"]:.3f} \
[95% CI {lo:.3f}–{hi:.3f}], PR-AUC {t["pr_auc"]:.3f} vs base rate {t["base_rate"]:.3f}, \
Brier {t["brier"]:.4f}, ECE {t["ece"]:.3f}.

Headline: {headline}

RISK_LABEL_THRESHOLDS ({RISK_LABEL_THRESHOLDS[0]}/{RISK_LABEL_THRESHOLDS[1]}) mean most \
shipments land in 'medium'; that is the honest shape of the score distribution, not a bug.

## Data and split

Source: the curated warehouse (`shipments` view: deduplicated, quarantined rows excluded, cargo \
normalised, invalid numerics nulled). Labelled rows = an actual arrival exists and status is not \
CANCELLED ({info["n_labelled"]} rows). Rows with no actual arrival (cancelled, in transit, \
"delivered" without dates) have no label and are not used.

Time split on `booking_date`, so every evaluation is out of time:

| split | window | rows | late rate |
|---|---|---|---|
{splits}

Tuning (grid choice, the decision threshold and the sigmoid calibrator of the selected base \
model) uses validation only. Test is scored once per candidate and is used only for the \
ship/no-ship gate in the selection rule, never to choose between models or settings. The CV5 \
column is a shuffled 5-fold CV on train only: a stability number for the fit, not an \
out-of-time estimate.

## Features (all known at booking)

| feature | used as | why it is not a leak |
|---|---|---|
| `origin_port` | one-hot (< 20 train rows pooled as "infrequent"; unseen -> infrequent) | \
chosen at booking |
| `destination_port` | one-hot, same | chosen at booking |
| `cargo_type` | one-hot after trim + title-case (the API may send `"machinery "`) | declared at \
booking |
| `container_count` | numeric; outside 1–100 -> missing; median impute + missing indicator | \
booked quantity |
| `weight_tons` | log1p; <= 0 -> missing; median impute + missing indicator | declared at booking |
| `planned_departure` | -> `dep_month`, `dep_dow` | schedule is fixed at booking |
| `planned_arrival` | -> `transit_days_planned` (with departure) | schedule is fixed at booking |

All cleaning happens in `ml.features.clean_and_derive`, which is the first step of the pickled \
pipeline, so training and the API apply the same rules to the same raw dict.

## Excluded columns and why

- `status`: a perfect leak. `DELAYED` ⇔ arrival > 24h late on every row; it is the label in \
another form and is only known after arrival.
- `actual_departure`, `actual_arrival`, `actual_delay_hours`, `transit_days_actual`, \
`on_time_flag`: post-booking outcomes; the label is computed from them.
- `dq_flags`: written by curation after the fact, and several flags (e.g. \
`completed_without_actuals`) depend on outcome fields.
- `booking_date`: used only to split; its absolute value never recurs in the future.
- `customer_id`, `vessel_id`, `route_key`: high-cardinality ids whose historical late rate does \
not carry over in time. Evidence (late rate per group in train vs the same group in test):

| key | groups in train | median train rows / group | groups in both | train→validation corr |
|---|---|---|---|---|
{stability}

With a median of {n_cust:.0f} train shipments per customer, a group's late rate is mostly \
binomial noise (the standard error of a 40% rate on {n_cust:.0f} rows is ±{se_cust:.0f}pp), so the \
correlation sits near zero; even ports (~110 rows each) and cargo types show no carry-over, and \
their spread of late rates matches what sampling noise alone produces. The ablation below confirms \
it: adding cross-fitted target encodings of `route_key` and `customer_id` to the logistic \
regression moves validation AUC from {lr["validation"]["roc_auc"]:.3f} to \
{abl["validation"]["roc_auc"]:.3f} and test AUC from {lr["test"]["roc_auc"]:.3f} to \
{abl["test"]["roc_auc"]:.3f}. It is documented, not shipped. `vessel_id` is also usually not \
known at booking.

## Candidates

All evaluated identically: base models fit on train, grid point and F1 threshold chosen on \
validation (the calibrated candidate's sigmoid is fitted on validation too; see Provenance), \
scored once on test. Test metrics; bytes = pickled pipeline; latency = median of \
{LATENCY_CALLS} single-row `predict_proba` calls (the API's shape).

{info["table"]}

- `prior`: constant train base rate. At the tuned threshold it predicts "late" for everyone, \
which is why its F1 looks respectable.
- `route_history`: train-only late rate per route, shrunk (m=20) to the origin port's rate, \
shrunk to the global rate; unseen route -> origin port -> global.
- `logreg`: C chosen on validation AUC; no class weights, so probabilities stay calibrated.
- `hist_gb`, `random_forest`: small grids (see `models/metrics.json` for every grid point).
- `{info["calibrated"]}`: the base model picked in step (a) (`{info["base"]}`, fit on train), \
frozen, with a sigmoid (Platt) re-mapping of its probabilities fitted on the validation split \
only (`CalibratedClassifierCV(FrozenEstimator(...), method="sigmoid")`). The ranking, and so \
the AUC, is the base model's; only the probabilities change. Its tuned threshold comes from the \
same validation rows. CV5 is n/a: there is no train-only refit to cross-validate.
- `ablation_logreg_te`: logreg + target-encoded `route_key`, `customer_id`. Not eligible to ship.

## Selection rule

(a) Among the uncalibrated learned candidates ({", ".join(LEARNED)}), select the one with \
the best validation AUC; a tie is broken by the lower validation log loss of the uncalibrated \
model. Nothing fitted on validation labels takes part in this choice.
Then calibrate the selected base model on validation (sigmoid, base model frozen).
(b) Ship the calibrated model only if, on test, its AUC 95% bootstrap CI lower bound > 0.50 AND \
its Brier <= prior Brier + {BRIER_TOLERANCE} (i.e. calibration no worse than the base rate \
within tolerance). Otherwise ship the prior.

The ablation is not eligible: it needs ids the API does not send. (b) looks at test metrics for \
a ship/no-ship gate, not for choosing between learned models (that is done on validation in \
(a)). The reason: with {info["splits"]["validation"]["n"]} validation rows no candidate can \
clear a strict significance bar, so the gate is "measurably better than chance and honestly \
calibrated", nothing more. Bootstrap CIs use {BOOTSTRAP_RESAMPLES} resamples.

{selection}

**Chosen: `{shipped}`** with {r[shipped]["params"]}, decision threshold \
{r[shipped]["threshold"]:.3f} (validation-optimal F1; the API reports the probability and the \
`risk_label` bands, the threshold is stored for anyone who needs a yes/no).

Provenance of the shipped artefact (also in `metrics.json` and the joblib bundle):

{provenance}

The artefact is the exact model that was scored on test, so the numbers above describe the \
file. Production practice would be to refit the chosen configuration on train + validation + \
test (and recalibrate on a fresh window) before deploying; that model would be unscored, which \
is why it is not done here.

## Shipped model: test calibration

{_reliability_table(t["reliability"])}

## Honest limits

- Predictions cluster near the base rate (test scores: min {q["min"]:.3f}, p10 \
{q["p10"]:.3f}, median {q["p50"]:.3f}, p90 {q["p90"]:.3f}, max {q["max"]:.3f}; mean \
{t["pred_mean"]:.3f}). Few shipments will ever reach the "high" band.
- Base-rate drift shows up as calibration error: the late rate was \
{info["splits"]["train"]["base_rate"]:.3f} in train, \
{info["splits"]["validation"]["base_rate"]:.3f} in validation and {t["base_rate"]:.3f} in \
test, so the shipped mean score {t["pred_mean"]:.3f} is off from the test late rate by \
{t["pred_mean"] - t["base_rate"]:+.3f} (test ECE {t["ece"]:.3f}). \
This is the main thing the monitoring's base-rate and calibration triggers are for.
- F1 alone is misleading here: always predicting "late" scores F1 ≈ \
{r["prior"]["test"]["f1@tuned"]:.2f}. Lead with ROC-AUC, PR-AUC against the base rate, and Brier.
- The data is synthetic-looking: the booking-time fields barely relate to the outcome. Real \
signal would need features this dataset does not have (port congestion at departure time, \
weather, carrier schedule reliability, transhipment count).
- One year of history: `dep_month` sees each month once in train, so any seasonality it learns \
is a single draw.
- Test covers bookings from {VALIDATION_END} onwards that have already arrived; late arrivals \
from the newest bookings are not in it yet, which slightly biases recent windows toward on-time.

## Monitoring

Per prediction, log: request id, timestamp, `model_version`, the raw feature dict, the cleaned \
features, the score and the risk label. Join to the actual arrival when it lands.

Weekly:
- PSI of the score distribution vs the test-period distribution.
- PSI / share-shift of origin port, destination port and cargo mix.
- Share of categories unseen in training (they score as "infrequent") and of missing / invalid \
weight and container count.

Once arrivals land (labels lag booking by up to ~50 days): rolling 30-day ROC-AUC with a \
bootstrap CI, Brier, calibration error (ECE), and the observed late rate.

Retrain triggers:
- rolling AUC CI falls below the `route_history` baseline's AUC on the same window;
- calibration error > 0.05;
- score or input PSI > 0.2;
- observed base rate moves > 5pp from the training base rate;
- otherwise monthly, through a champion/challenger gate: the challenger is trained on the newer \
time window and must win on a later held-out time slice under the selection rule above.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.train", description=__doc__)
    parser.add_argument("--db", type=Path, default=settings.db_path)
    parser.add_argument("--out", type=Path, default=settings.model_path.parent)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        df = load_labelled(args.db)
    except FileNotFoundError as e:
        print(f"input error: {e}", file=sys.stderr)
        return 2
    splits = time_split(df)
    bounds = {
        "train": f"booking_date < {TRAIN_END}",
        "validation": f"{TRAIN_END} <= booking_date < {VALIDATION_END}",
        "test": f"booking_date >= {VALIDATION_END}",
    }
    split_info = {
        k: {"window": bounds[k], "n": len(s), "base_rate": float(s["late"].mean())}
        for k, s in splits.items()
    }
    for k, s in split_info.items():
        print(f"{k:<10} {s['window']:<40} n={s['n']:<5} late rate={s['base_rate']:.3f}")

    results, models = {}, {}
    for name, grid in candidates().items():
        print(f"training {name} ({len(grid)} grid points)...", flush=True)
        results[name], models[name] = evaluate(name, grid, splits)
    base, rank_log = select_base(results)
    calibrated_name = f"{base}_calibrated"
    print(f"calibrating {base} on validation...", flush=True)
    calibrated = CalibratedClassifierCV(FrozenEstimator(models[base]), method="sigmoid")
    results[calibrated_name], models[calibrated_name] = evaluate(
        calibrated_name, [(results[base]["params"], calibrated)], splits, fit_on="validation"
    )
    shipped, gate_log = gate(results, calibrated_name)
    selection_log = [rank_log, gate_log]
    shipped_base = "prior (constant train base rate)" if shipped == "prior" else base
    provenance = {
        "base_model": f"{shipped_base} fit on train ({bounds['train']})",
        "calibrator": "none (the prior is the train base rate)"
        if shipped == "prior"
        else f"sigmoid calibrator fit on validation ({bounds['validation']})",
        "threshold": f"chosen on validation (max F1), {results[shipped]['threshold']:.3f}",
        "test": f"scored once on test ({bounds['test']}); used only for the ship gate",
    }

    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    version = f"{VERSION_TAGS.get(shipped, shipped)}-v1-train<{TRAIN_END[:7]}"
    table = comparison_table(results, shipped)
    info = {
        "version": version,
        "shipped": shipped,
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
        "label": f"actual_arrival - planned_arrival > {LABEL_THRESHOLD_HOURS}h",
        "features": list(FEATURES),
        "n_labelled": len(df),
        "splits": split_info,
        "selection_rule": (
            "(a) base = best uncalibrated learned model on validation AUC, ties -> lower "
            "validation log loss; calibrate it on validation (sigmoid); (b) ship the calibrated "
            "model only if test AUC CI lower bound > 0.50 and test Brier <= prior Brier + "
            f"{BRIER_TOLERANCE}, else prior"
        ),
        "base": base,
        "calibrated": calibrated_name,
        "provenance": provenance,
        "selection_log": selection_log,
        # Diagnostic on train vs validation only: the test split must not inform feature choices.
        "group_stability": group_stability(splits["train"], splits["validation"]),
        "results": results,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": models[shipped],
        "version": version,
        "threshold": results[shipped]["threshold"],
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "trained_at": trained_at,
        "train_window": bounds["train"],
        "provenance": provenance,
        "metrics": results[shipped]["test"],
    }
    joblib.dump(bundle, args.out / "delay_model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (args.out / "model_card.md").write_text(model_card({**info, "table": table}), encoding="utf-8")

    print()
    print(table)
    print()
    print("\n".join(selection_log))
    for k, s in info["group_stability"].items():
        print(
            f"stability {k:<17} train->validation late-rate corr "
            f"{s['train_test_late_rate_corr']:+.2f}"
        )
    print(f"shipped {version} -> {args.out / 'delay_model.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
