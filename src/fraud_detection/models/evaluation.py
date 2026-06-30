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
        "precision_at_top_0.5%": precision_at_top(y, scores, 0.005),
        "precision_at_top_1%": precision_at_top(y, scores, 0.01),
        "thresholds": thresholds.to_dict(),
        "action_rates": {a: float((actions == a).mean()) for a in ACTIONS},
        "fraud_recall": float(caught.sum() / max(fraud.sum(), 1)),
        "fraud_dollar_recall": float(amt[caught].sum() / max(amt[fraud].sum(), 1e-9)),
        "false_positive_rate": float(
            ((actions != "approve") & ~fraud).sum() / max((~fraud).sum(), 1)
        ),
    }
    model_cost = total_cost(y, amt, scores, thresholds, costs)
    approve_all = total_cost(
        y, amt, np.zeros_like(scores), Thresholds(review=2.0, decline=2.0), costs
    )
    out["cost"] = model_cost
    out["cost_approve_all"] = approve_all["total"]
    out["cost_savings"] = approve_all["total"] - model_cost["total"]
    out["cost_savings_pct"] = out["cost_savings"] / max(approve_all["total"], 1e-9)
    return out


def confusion_by_action(y: np.ndarray, scores: np.ndarray, thresholds: Thresholds) -> pd.DataFrame:
    actions = thresholds.actions(scores)
    table = pd.crosstab(
        pd.Series(np.where(np.asarray(y) == 1, "fraud", "legit"), name="truth"),
        pd.Series(actions, name="action"),
    )
    return table.reindex(columns=list(ACTIONS), fill_value=0)


def calibration_table(
    y: np.ndarray,
    scores: np.ndarray,
    edges: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    """Observed vs predicted fraud rate in fixed score bands.

    Fixed bands (rather than deciles) keep the high-score region visible when fraud is rare.
    """
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(scores)})
    frame["band"] = pd.cut(frame["p"], bins=list(edges), include_lowest=True)
    table = frame.groupby("band", observed=False).agg(
        n=("y", "size"), mean_score=("p", "mean"), fraud_rate=("y", "mean")
    )
    table.index = table.index.astype(str)
    return table
