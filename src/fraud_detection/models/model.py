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
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pip install 'fraud-detection-system[lightgbm]'") from exc
        return lgb.LGBMClassifier(
            n_estimators=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.max_leaf_nodes,
            min_child_samples=cfg.min_samples_leaf,
            reg_lambda=cfg.l2_regularization,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
    if cfg.type == "logistic":
        numeric_idx = np.flatnonzero(~cat_mask).tolist()
        cat_idx = np.flatnonzero(cat_mask).tolist()
        pre = ColumnTransformer(
            [
                (
                    "num",
                    make_pipeline(
                        SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                    ),
                    numeric_idx,
                ),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx),
            ]
        )
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, C=0.5))])
    raise ValueError(f"Unknown model type {cfg.type!r}")


@dataclass
class FraudModel:
    config: ModelConfig = field(default_factory=ModelConfig)
    spec: FeatureSpec = field(default_factory=FeatureSpec)
    seed: int = 42
    merchant_encoder: TargetEncoder | None = None
    category_encoder: TargetEncoder | None = None
    estimator: Any = None
    calibrator: IsotonicRegression | SigmoidCalibrator | None = None

    # ------------------------------------------------------------ encodings
    def _check_encoders(self) -> None:
        if self.merchant_encoder is None or self.category_encoder is None:
            raise RuntimeError("Encoders are not fitted")

    def model_frame(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Raw feature frame -> encoded float matrix (as a DataFrame with named columns)."""
        self._check_encoders()
        encoded = feats.copy()
        encoded["merchant_risk"] = self.merchant_encoder.transform(feats["merchant"])
        encoded["category_risk"] = self.category_encoder.transform(feats["category"])
        return self.spec.to_model_frame(encoded)

    def row_vector(self, row: dict[str, Any]) -> np.ndarray:
        """Online equivalent of ``model_frame`` for one transaction."""
        self._check_encoders()
        values = dict(row)
        values["merchant_risk"] = self.merchant_encoder.transform_one(row["merchant"])
        values["category_risk"] = self.category_encoder.transform_one(row["category"])
        return self.spec.row_vector(values)

    # -------------------------------------------------------------- training
    def fit(
        self,
        feats: pd.DataFrame,
        y: pd.Series,
        eval_feats: pd.DataFrame | None = None,
        eval_y: pd.Series | None = None,
    ) -> FraudModel:
        y = pd.Series(np.asarray(y), index=feats.index)
        self.merchant_encoder = TargetEncoder(seed=self.seed)
        self.category_encoder = TargetEncoder(seed=self.seed)
        encoded = feats.copy()
        encoded["merchant_risk"] = self.merchant_encoder.fit_transform_oof(feats["merchant"], y)
        encoded["category_risk"] = self.category_encoder.fit_transform_oof(feats["category"], y)
        X = self.spec.to_model_frame(encoded).to_numpy()
        weights = np.where(y.to_numpy() == 1, self.config.pos_weight, 1.0)

        self.estimator = build_estimator(self.config, self.spec, self.seed)
        if self.config.type == "logistic":
            self.estimator.fit(X, y.to_numpy(), clf__sample_weight=weights)
        elif self.config.type == "lightgbm":
            import lightgbm as lgb

            fit_kwargs: dict[str, Any] = {
                "sample_weight": weights,
                "categorical_feature": np.flatnonzero(self.spec.categorical_mask).tolist(),
            }
            if eval_feats is not None:
                fit_kwargs["eval_set"] = [
                    (self.model_frame(eval_feats).to_numpy(), np.asarray(eval_y))
                ]
                fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
            self.estimator.fit(X, y.to_numpy(), **fit_kwargs)
        else:
            self.estimator.fit(X, y.to_numpy(), sample_weight=weights)
        self.calibrator = None
        return self

    def calibrate(self, feats: pd.DataFrame, y: pd.Series) -> FraudModel:
        """Fit a calibrator on a held-out, later time period.

        Class weighting during training inflates raw scores; calibration maps them back to
        probabilities so cost-based thresholds mean what they say.
        """
        method = self.config.calibration
        if method == "none":
            return self
        raw = self.predict_raw(feats)
        if method == "sigmoid":
            self.calibrator = SigmoidCalibrator().fit(raw, np.asarray(y))
        elif method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw, np.asarray(y))
        else:
            raise ValueError(f"Unknown calibration method {method!r}")
        return self

    # ------------------------------------------------------------ inference
    def predict_raw_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model is not fitted")
        return self.estimator.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict_raw(self, feats: pd.DataFrame) -> np.ndarray:
        return self.predict_raw_matrix(self.model_frame(feats))

    def calibrated(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        return self.calibrated(self.predict_raw(feats))
