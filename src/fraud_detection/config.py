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
    review_catch_rate: float = 0.9
    max_review_rate: float = 0.01


@dataclass
class MonitoringConfig:
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    n_bins: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    rules_path: str | None = "configs/rules.yaml"
    artifacts_dir: str = "artifacts"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, values: dict[str, Any] | None) -> Any:
    values = values or {}
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in values:
            ftype = hints[f.name]
            kwargs[f.name] = (
                _build(ftype, values[f.name]) if dataclasses.is_dataclass(ftype) else values[f.name]
            )
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return _build(Config, raw)
