"""Typed configuration loaded from YAML (all keys optional)."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SyntheticConfig:
    n_cards: int = 1000
    days: int = 180
    start: str = "2023-01-01"
    n_merchants: int = 400
    fraud_card_share: float = 0.12
    seed: int = 7


@dataclass
class DataConfig:
    source: str = "synthetic"  # "synthetic" or "file"
    paths: list[str] = field(
        default_factory=lambda: ["data/raw/fraudTrain.csv", "data/raw/fraudTest.csv"]
    )
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


@dataclass
class SplitConfig:
    valid_fraction: float = 0.15
    test_fraction: float = 0.20


@dataclass
class ModelConfig:
    type: str = "hgb"  # "hgb", "lightgbm" or "logistic"
    pos_weight: float = 10.0
    learning_rate: float = 0.05
    max_iter: int = 600
    max_leaf_nodes: int = 63
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    calibration: str = "sigmoid"  # "sigmoid", "isotonic" or "none"


@dataclass
class CostConfig:
    review_cost: float = 5.0
    decline_fp_cost: float = 15.0
    chargeback_fee: float = 20.0
