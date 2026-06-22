"""The fraud model: target encoders + gradient boosting (or logistic baseline) + calibrator.

Kept as one object so training, offline evaluation, replay and the API all apply exactly the
same transformations in the same order. Estimators are trained on a plain float matrix
(categoricals as integer codes), so the online path can score a NumPy row without pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import ModelConfig
from ..features.encoders import TargetEncoder
from ..features.spec import FeatureSpec


class SigmoidCalibrator:
    """Platt scaling on the logit of the raw score.

    Strictly monotonic, so it fixes the probability scale without creating ties (isotonic
    regression on a small calibration window collapses scores into a few steps, which hurts
    ranking metrics and makes threshold tuning coarse).
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e4, max_iter=1000)

    @staticmethod
    def _logit(raw: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw, dtype=float), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> SigmoidCalibrator:
        self.model.fit(self._logit(raw), np.asarray(y))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(raw))[:, 1]


def build_estimator(cfg: ModelConfig, spec: FeatureSpec, seed: int) -> Any:
    cat_mask = spec.categorical_mask
    if cfg.type == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=cfg.learning_rate,
            max_iter=cfg.max_iter,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization,
            categorical_features=cat_mask,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=seed,
        )
    if cfg.type == "lightgbm":
        try:
