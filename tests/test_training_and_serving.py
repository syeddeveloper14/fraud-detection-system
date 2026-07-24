import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import create_app
from fraud_detection.config import ModelConfig, load_config
from fraud_detection.decision import DecisionEngine
from fraud_detection.features import build_offline_features
from fraud_detection.models import FraudModel, load_bundle
from fraud_detection.monitoring import drift_report, psi
from fraud_detection.pipeline import make_splits
from fraud_detection.replay import replay
from fraud_detection.rules import RuleEngine
from tests.conftest import make_txn


def test_training_outputs(trained):
    cfg, result = trained
    m = result.metrics
    assert m["pr_auc"] > 0.5 and m["roc_auc"] > 0.9
    assert result.bundle.thresholds.review <= result.bundle.thresholds.decline
    assert (result.bundle_dir / "model_card.md").exists()
    assert (result.bundle_dir / "metadata.json").exists()
    assert result.bundle.metadata["baseline_test_metrics"] is not None


def test_bundle_roundtrip(trained, small_df):
    cfg, result = trained
    loaded = load_bundle(cfg.artifacts_dir)  # resolves LATEST
    assert loaded.version == result.bundle.version
    feats = build_offline_features(small_df.iloc[:500])
    np.testing.assert_allclose(
        loaded.model.predict_proba(feats), result.bundle.model.predict_proba(feats)
    )


def test_online_scores_match_offline(trained, small_df):
    _, result = trained
    model = result.bundle.model
    feats = build_offline_features(small_df)
    tail = small_df.index[-300:]
    offline = model.predict_proba(feats.loc[tail])
    engine = DecisionEngine(result.bundle, explain=False)
    engine.store.warm_up(small_df.loc[small_df.index[:-300]])
    online = [engine.decide(t).score for t in small_df.loc[tail].to_dict("records")]
    np.testing.assert_allclose(online, offline, rtol=1e-9, atol=1e-12)


def test_decision_reasons_for_flagged_transactions(trained):
    _, result = trained
    engine = DecisionEngine(result.bundle, rules=RuleEngine.from_yaml("configs/rules.yaml"))
    # a burst of tiny online charges followed by a large one, at night
    for i in range(5):
        engine.decide(
            make_txn(
                trans_num=f"b{i}",
                ts=pd.Timestamp("2023-03-01 02:00:00") + pd.Timedelta(minutes=i),
                amt=2.0,
                category="misc_net",
            )
        )
    d = engine.decide(
        make_txn(
            trans_num="big", ts=pd.Timestamp("2023-03-01 02:10:00"), amt=1.5, category="misc_net"
        )
    )
    assert d.action in {"review", "decline"}
    assert any(h["name"] == "card_testing_burst" for h in d.rule_hits)


def test_replay_parity_and_costs(trained, small_df):
    cfg, result = trained
    sp = make_splits(small_df, cfg)
    history = small_df.loc[sp.train.append(sp.calib).append(sp.tune)]
    stream = small_df.loc[sp.test].iloc[:400]
    out = replay(
