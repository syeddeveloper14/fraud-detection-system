"""Training pipeline: data -> features -> time split -> train -> calibrate -> thresholds ->
test evaluation -> explanations -> drift reference -> versioned bundle."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .config import Config, ModelConfig
from .data import load_transactions, prepare_transactions, simulate_transactions, time_split
from .features import FEATURE_DESCRIPTIONS, FeatureSpec, build_offline_features
from .models import (
    FraudModel,
    ModelBundle,
    ReasonCoder,
    calibration_table,
    confusion_by_action,
    evaluate,
    legit_baseline,
    optimize_thresholds,
    permutation_importance,
    save_bundle,
)
from .models.registry import make_version
from .monitoring import reference_profile
from .schema import LABEL

log = logging.getLogger(__name__)


@dataclass
class Splits:
    train: pd.Index
    calib: pd.Index
    tune: pd.Index
    test: pd.Index


@dataclass
class TrainResult:
    bundle: ModelBundle
    bundle_dir: Path
    metrics: dict[str, Any]


def load_data(cfg: Config) -> pd.DataFrame:
    if cfg.data.source == "synthetic":
        s = cfg.data.synthetic
        log.info("Simulating %d cards over %d days", s.n_cards, s.days)
        raw = simulate_transactions(
            n_cards=s.n_cards,
            days=s.days,
            start=s.start,
            n_merchants=s.n_merchants,
            fraud_card_share=s.fraud_card_share,
            seed=s.seed,
        )
        return prepare_transactions(raw)
    if cfg.data.source == "file":
        return load_transactions(cfg.data.paths)
    raise ValueError(f"Unknown data.source {cfg.data.source!r}")


def make_splits(df: pd.DataFrame, cfg: Config) -> Splits:
    """Time-ordered splits; the validation period is halved into calibration and tuning."""
    train, valid, test = time_split(df, cfg.split.valid_fraction, cfg.split.test_fraction)
    mid = valid["ts"].quantile(0.5)
    calib = valid[valid["ts"] < mid]
    tune = valid[valid["ts"] >= mid]
    return Splits(train.index, calib.index, tune.index, test.index)


def fingerprint(df: pd.DataFrame, cfg: Config) -> str:
    h = hashlib.sha1()
    h.update(str(len(df)).encode())
    h.update(str(df["ts"].min()).encode() + str(df["ts"].max()).encode())
    h.update(str(int(df[LABEL].sum())).encode())
    h.update(json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode())
    return h.hexdigest()


def train_pipeline(cfg: Config, df: pd.DataFrame | None = None) -> TrainResult:
    df = load_data(cfg) if df is None else df
    log.info("Building point-in-time features for %d transactions", len(df))
    feats = build_offline_features(df)
    y = df[LABEL]
    amt = df["amt"]
    sp = make_splits(df, cfg)
    for name in ("train", "calib", "tune", "test"):
        idx = getattr(sp, name)
        log.info("%-5s %8d rows  fraud rate %.4f", name, len(idx), y.loc[idx].mean())

    spec = FeatureSpec()
    model = FraudModel(config=cfg.model, spec=spec, seed=cfg.seed)
    model.fit(feats.loc[sp.train], y.loc[sp.train], feats.loc[sp.calib], y.loc[sp.calib])
    model.calibrate(feats.loc[sp.calib], y.loc[sp.calib])

    p_tune = model.predict_proba(feats.loc[sp.tune])
    thresholds, tune_cost = optimize_thresholds(
        y.loc[sp.tune].to_numpy(), amt.loc[sp.tune].to_numpy(), p_tune, cfg.costs
    )
    log.info("Thresholds: review >= %.4f, decline >= %.4f", thresholds.review, thresholds.decline)

    p_test = model.predict_proba(feats.loc[sp.test])
    y_test = y.loc[sp.test].to_numpy()
    test_metrics = evaluate(y_test, p_test, amt.loc[sp.test].to_numpy(), thresholds, cfg.costs)

    # Baseline for comparison, same features, same splits, same thresholding procedure.
    baseline_metrics = None
    if cfg.model.type != "logistic":
        base = FraudModel(
            config=ModelConfig(type="logistic", pos_weight=cfg.model.pos_weight),
            spec=spec,
            seed=cfg.seed,
        )
        base.fit(feats.loc[sp.train], y.loc[sp.train]).calibrate(
            feats.loc[sp.calib], y.loc[sp.calib]
        )
        base_t, _ = optimize_thresholds(
            y.loc[sp.tune].to_numpy(),
            amt.loc[sp.tune].to_numpy(),
            base.predict_proba(feats.loc[sp.tune]),
            cfg.costs,
        )
        baseline_metrics = evaluate(
            y_test,
            base.predict_proba(feats.loc[sp.test]),
            amt.loc[sp.test].to_numpy(),
            base_t,
            cfg.costs,
        )

    X_tune = model.model_frame(feats.loc[sp.tune])
    importance = permutation_importance(model, X_tune, y.loc[sp.tune].to_numpy(), seed=cfg.seed)
    X_train = model.model_frame(feats.loc[sp.train])
    coder = ReasonCoder(
        baseline=legit_baseline(X_train, y.loc[sp.train].to_numpy(), spec.categorical),
        candidates=importance["feature"].head(10).tolist(),
        category_levels={c: spec.category_levels for c in spec.categorical},
    )
    sample = X_train.sample(n=min(len(X_train), 200_000), random_state=cfg.seed)
    profile = reference_profile(
        sample,
        model.calibrated(model.predict_raw_matrix(sample)),
        cfg.monitoring.n_bins,
        categorical=spec.categorical,
    )

    metadata = {
        "version": make_version(fingerprint(df, cfg)),
        "package_version": __version__,
        "model_type": cfg.model.type,
        "features": spec.columns,
        "thresholds": thresholds.to_dict(),
        "costs": cfg.to_dict()["costs"],
        "data": {
            "rows": int(len(df)),
            "fraud_rate": float(y.mean()),
            **{
                f"{name}_window": [str(df.loc[idx, "ts"].min()), str(df.loc[idx, "ts"].max())]
                for name, idx in vars(sp).items()
            },
            **{f"{name}_rows": int(len(idx)) for name, idx in vars(sp).items()},
        },
        "tuning_cost": tune_cost,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "feature_importance": importance.head(15).to_dict(orient="records"),
        "config": cfg.to_dict(),
    }
    bundle = ModelBundle(
        model=model,
        thresholds=thresholds,
        reason_coder=coder,
        reference_profile=profile,
        metadata=metadata,
    )
    bundle_dir = save_bundle(bundle, cfg.artifacts_dir)
    write_model_card(
        bundle,
        bundle_dir / "model_card.md",
        confusion_by_action(y_test, p_test, thresholds),
        calibration_table(y_test, p_test),
    )
    log.info("Saved bundle %s", bundle_dir)
    return TrainResult(bundle=bundle, bundle_dir=bundle_dir, metrics=test_metrics)


def _table(df: pd.DataFrame, fmt: str = "{:,.4f}") -> str:
    frame = df.reset_index()
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| "
        + " | ".join(fmt.format(v) if isinstance(v, float | np.floating) else str(v) for v in row)
        + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([head, sep, *body])


def write_model_card(
    bundle: ModelBundle, path: Path, confusion: pd.DataFrame, calibration: pd.DataFrame
) -> Path:
    m = bundle.metadata
    t = m["test_metrics"]
    b = m.get("baseline_test_metrics")
    rows = [
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Recall @ 0.1% FPR", "recall_at_fpr_0.1%"),
        ("Recall @ 1% FPR", "recall_at_fpr_1%"),
        ("Precision in top 0.5%", "precision_at_top_0.5%"),
        ("Fraud recall (review + decline)", "fraud_recall"),
        ("Fraud $ recall", "fraud_dollar_recall"),
        ("False positive rate", "false_positive_rate"),
        ("Cost savings vs approve-all", "cost_savings_pct"),
    ]
    metric_lines = ["| Metric | Model | Logistic baseline |", "|---|---|---|"]
    for label, key in rows:
        base = f"{b[key]:.4f}" if b else "n/a"
        metric_lines.append(f"| {label} | {t[key]:.4f} | {base} |")
    importance = "\n".join(
        f"| {r['feature']} | {FEATURE_DESCRIPTIONS.get(r['feature'], '')} | {r['importance']:.4f} |"
        for r in m["feature_importance"]
    )
    d = m["data"]
    text = f"""# Model card: card fraud detection ({bundle.version})

## Intended use
Real-time scoring of card-not-present and card-present transactions. Each transaction gets a
calibrated fraud probability and an action (approve / review / decline). The model supports
analysts; declines above the decline threshold are automatic.

## Data
- {d["rows"]:,} transactions, overall fraud rate {d["fraud_rate"]:.4%}
- Time-ordered splits: train {d["train_window"][0]} to {d["train_window"][1]},
  calibration {d["calib_window"][0]} to {d["calib_window"][1]},
  threshold tuning {d["tune_window"][0]} to {d["tune_window"][1]},
  test {d["test_window"][0]} to {d["test_window"][1]}

## Model
- Type: `{m["model_type"]}` with smoothed out-of-fold target encoding for merchant and category,
  positive-class weight {m["config"]["model"]["pos_weight"]}, isotonic calibration on the
  calibration period
- Features: {len(m["features"])} point-in-time features (see docs/feature_catalog.md)

## Decision thresholds (tuned for minimum expected cost)
- Decline if score >= {m["thresholds"]["decline"]:.4f}
- Review if score >= {m["thresholds"]["review"]:.4f}
- Cost assumptions: review {m["costs"]["review_cost"]}, false decline {m["costs"]["decline_fp_cost"]},
  chargeback fee {m["costs"]["chargeback_fee"]}, review catch rate {m["costs"]["review_catch_rate"]},
  review capacity {m["costs"]["max_review_rate"]:.2%} of traffic

## Test-period performance
{chr(10).join(metric_lines)}

Action rates on the test period: {", ".join(f"{k} {v:.3%}" for k, v in t["action_rates"].items())}

Confusion by action:

{_table(confusion, "{:,.0f}")}

Calibration (test period, fixed score bands):

{_table(calibration)}

## Most important features (permutation importance, drop in PR-AUC)
| Feature | Description | Importance |
|---|---|---|
{importance}

## Limitations
- Labels arrive late (chargebacks take weeks); the most recent weeks of any training window
  are under-labelled. Leave a label-maturity gap before the end of the training data.
- Thresholds depend on the cost assumptions above; revisit them when review staffing or
  chargeback fees change.
- Target encodings are fixed at training time; new merchants fall back to the global rate.
- Velocity features assume transactions arrive in timestamp order per card.
"""
    path.write_text(text, encoding="utf-8")
    return path
