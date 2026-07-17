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
