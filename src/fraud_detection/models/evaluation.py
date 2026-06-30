"""Offline evaluation for a highly imbalanced, cost-sensitive problem."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from ..config import CostConfig
from .thresholds import ACTIONS, Thresholds, total_cost


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def precision_at_top(y: np.ndarray, scores: np.ndarray, share: float) -> float:
    k = max(1, int(round(share * len(scores))))
    top = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return float(np.asarray(y)[top].mean())


def evaluate(
    y: np.ndarray,
    scores: np.ndarray,
    amt: np.ndarray,
    thresholds: Thresholds,
    costs: CostConfig,
) -> dict[str, object]:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    amt = np.asarray(amt, dtype=float)
    actions = thresholds.actions(scores)
    fraud = y == 1
    caught = fraud & (actions != "approve")

    out: dict[str, object] = {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "recall_at_fpr_0.1%": recall_at_fpr(y, scores, 0.001),
        "recall_at_fpr_1%": recall_at_fpr(y, scores, 0.01),
