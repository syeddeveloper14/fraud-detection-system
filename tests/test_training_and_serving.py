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
        result.bundle,
        history,
        stream,
        offline_features=build_offline_features(small_df),
        costs=cfg.costs,
    )
    assert out.summary["feature_parity_ok"]
    assert out.summary["n"] == 400
    assert out.summary["cost_total"] <= out.summary["cost_approve_all"]


def test_drift_detection(trained, small_df):
    _, result = trained
    model = result.bundle.model
    feats = build_offline_features(small_df)
    X = model.model_frame(feats)
    stable = drift_report(result.bundle.reference_profile, X, model.predict_proba(feats))
    assert (stable.set_index("feature").loc[["hour", "category"], "status"] == "ok").all()

    shifted = feats.copy()
    shifted["amt"] = shifted["amt"] * 5
    report = drift_report(result.bundle.reference_profile, model.model_frame(shifted))
    assert report.set_index("feature").loc["amt", "status"] == "alert"


def test_psi_properties():
    p = np.array([0.2, 0.3, 0.5])
    assert psi(p, p) == pytest.approx(0.0)
    assert psi(p, np.array([0.5, 0.3, 0.2])) > 0.25


def test_api(trained):
    cfg, result = trained
    client = TestClient(create_app(result.bundle, RuleEngine.from_yaml("configs/rules.yaml")))
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model_version"] == result.bundle.version

    payload = make_txn()
    payload["ts"] = payload["ts"].isoformat()
    payload["dob"] = payload["dob"].date().isoformat()
    r = client.post("/v1/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["action"] in {"approve", "review", "decline"}

    bad = dict(payload, amt=-5)
    assert client.post("/v1/score", json=bad).status_code == 422

    batch = [dict(payload, trans_num=f"x{i}", ts=f"2023-03-02T10:0{i}:00") for i in range(3)]
    r = client.post("/v1/score/batch", json=list(reversed(batch)))
    assert [x["trans_num"] for x in r.json()] == ["x0", "x1", "x2"]

    metrics = client.get("/metrics").text
    assert "fraud_decisions_total" in metrics


@pytest.mark.parametrize("model_type", ["logistic", "lightgbm"])
def test_alternative_model_types(small_df, model_type):
    if model_type == "lightgbm":
        pytest.importorskip("lightgbm")
    cfg = load_config()
    sp = make_splits(small_df, cfg)
    feats = build_offline_features(small_df)
    y = small_df["is_fraud"]
    model = FraudModel(config=ModelConfig(type=model_type, max_iter=100))
    model.fit(feats.loc[sp.train], y.loc[sp.train], feats.loc[sp.calib], y.loc[sp.calib])
    model.calibrate(feats.loc[sp.calib], y.loc[sp.calib])
    p = model.predict_proba(feats.loc[sp.test])
    assert p.shape == (len(sp.test),) and ((p >= 0) & (p <= 1)).all()


def test_unknown_model_type():
    with pytest.raises(ValueError):
        FraudModel(config=ModelConfig(type="svm")).fit(
            pd.DataFrame({"merchant": ["a"], "category": ["home"]}), pd.Series([0])
        )
