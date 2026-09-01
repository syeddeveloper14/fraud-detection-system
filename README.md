# Fraud Detection System

[![CI](https://github.com/YOUR_GITHUB_USERNAME/fraud-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/fraud-detection-system/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end card fraud detection, from raw transactions to a real-time scoring API. The parts that make fraud ML hard in production are treated as first-class code rather than notebook steps:

- **Point-in-time features** that never see the future, computed in batch for training and by an **online feature store** at request time, with tests that prove the two paths produce identical values.
- **Time-based evaluation** (train → calibrate → tune thresholds → test, all in chronological order).
- **Calibrated probabilities** and **cost-based decision thresholds** (approve / review / decline) that respect analyst capacity.
- A **rules engine** in YAML for patterns the model has not learned yet.
- **Reason codes** for every flagged transaction.
- **Drift monitoring** (PSI) against a reference profile stored inside the model bundle.
- A **FastAPI service** with batch scoring and Prometheus metrics, plus a **stream replay** tool that measures latency and business cost before deployment.

Works on the public [Kaggle credit card transactions dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Sparkov simulator, ~1.85M transactions) or on the built-in generator, which produces the same schema with card-testing, account-takeover and impossible-travel fraud patterns.

## Architecture

```mermaid
flowchart TB
    subgraph Offline training
        A[Transactions] --> B[Static + point-in-time<br/>history features]
        B --> C[Time split<br/>train / calib / tune / test]
        C --> D[Target encoding OOF<br/>+ gradient boosting]
        D --> E[Platt calibration]
        E --> F[Cost-optimal thresholds]
        F --> G[Test evaluation<br/>+ logistic baseline]
        G --> H[(Model bundle<br/>model, thresholds,<br/>reason coder, drift profile)]
    end
    subgraph Online scoring
        R[POST /v1/score] --> S[Online feature store<br/>per-card state]
        S --> T[Model + calibrator]
        H --> T
        T --> U[Thresholds]
        R --> V[Rules engine]
        U & V --> W[Decision + reason codes]
        W --> S
    end
```

## Quickstart

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/fraud-detection-system.git
cd fraud-detection-system
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,lightgbm]"

make train      # synthetic data -> artifacts/<version>/ (bundle, metadata, model card)
make replay     # stream the test period through the online path; checks feature parity
make serve      # FastAPI on :8000, docs at http://localhost:8000/docs
make test
```

Score a transaction:

```bash
curl -s localhost:8000/v1/score -H 'content-type: application/json' -d '{
  "trans_num": "0b242abb623afc578575680df30655b9",
  "cc_num": "4613314721966",
  "ts": "2023-06-21T23:14:05",
  "amt": 812.47,
  "merchant": "fraud_Merchant_0042",
  "category": "shopping_net",
  "lat": 40.4931, "long": -91.8912,
  "merch_lat": 41.1012, "merch_long": -92.2210,
  "city_pop": 1306,
  "dob": "1968-03-19"
}'
```

```json
{
  "trans_num": "0b242abb623afc578575680df30655b9",
  "score": 0.91,
  "action": "decline",
  "model_action": "decline",
  "reasons": [
    {"feature": "amt", "description": "Transaction amount", "value": 812.47, "impact": 0.42},
    {"feature": "is_night", "description": "1 if hour is 22:00-05:59", "value": 1.0, "impact": 0.21}
  ],
  "rule_hits": [],
  "model_version": "20260115-101502-8c41d2",
  "latency_ms": 4.1
}
```
(The response above is illustrative; values depend on the trained model and the card's history.)

### On the Kaggle dataset

```bash
# put fraudTrain.csv and fraudTest.csv in data/raw/ (see data/README.md)
fraud-detect train --config configs/kaggle_sparkov.yaml
fraud-detect replay --config configs/kaggle_sparkov.yaml --limit 50000 --check-parity
```

Results from the synthetic generator are much easier than real fraud data and should not be quoted as model performance. Run on the Kaggle data (or your own) for meaningful numbers; every training run writes them to `artifacts/<version>/model_card.md`.

## How it works

### Features without leakage

All history features for a transaction at time *t* are computed from the card's events strictly before *t*: counts and sums over 1h / 24h / 7d, 30-day amount baseline (mean, std, z-score, ratio), time and distance since the previous transaction, implied travel speed, first use of a merchant, and transaction count so far. Static features cover amount, hour, night flag, cardholder age, home-to-merchant distance and city size. Merchant and category get smoothed target encodings; training rows receive out-of-fold values so their own label never leaks into their encoding.

The batch implementation (`features/history.py`) uses pandas time-window rolling; the online store (`features/online.py`) keeps a 30-day deque per card. `tests/test_feature_parity.py` streams thousands of transactions through both and requires identical results, including tricky cases like two transactions in the same second. Full catalogue: [docs/feature_catalog.md](docs/feature_catalog.md).

### Decisions are about cost, not accuracy

The validation period is split in two (both after the training period):

1. **Calibration**: Platt scaling on the logit of the raw score turns class-weighted outputs back into probabilities. It is strictly monotonic, so ranking is unchanged.
2. **Threshold tuning**: two cut-offs are chosen jointly to minimise expected cost, where a missed fraud costs its amount plus a chargeback fee, a review costs analyst time, and a false decline costs customer friction. A cap on the review rate models analyst capacity. The search uses prefix sums over the sorted scores, so the full grid of (review, decline) pairs is evaluated in one vectorised pass; a unit test checks it against brute force.

The test period is touched once, at the end, and reported alongside a logistic-regression baseline trained on the same features and splits.

### Rules next to the model

`configs/rules.yaml` holds declarative rules (`all` / `any` / `not` over any feature). The final action is the stricter of the model's action and any rule hit. Rules are validated on load and evaluated with an operator table, never `eval`. Missing values never trigger a rule.

### Monitoring

Each bundle stores a reference profile: quantile bin edges and proportions for every feature plus the score distribution. `fraud-detect monitor --data new_batch.csv` computes PSI per feature and flags `warn` (≥ 0.1) or `alert` (≥ 0.25); `--fail-on-alert` makes it usable as a scheduled job gate.

## Project structure

```
fraud-detection-system/
├── configs/
│   ├── default.yaml          # synthetic data, model, costs, monitoring
│   ├── kaggle_sparkov.yaml   # real dataset
│   └── rules.yaml            # decision rules
├── docs/
│   ├── architecture.md       # design decisions and trade-offs
│   └── feature_catalog.md    # every feature, definition and window
├── src/fraud_detection/
│   ├── data/                 # loader, simulator, time split
│   ├── features/             # static, history (batch), online store, encoders, spec
│   ├── models/               # model, thresholds, evaluation, explanations, registry
│   ├── rules/                # rules engine
│   ├── monitoring/           # PSI drift
│   ├── api/                  # FastAPI app + schemas
│   ├── decision.py           # features -> score -> rules -> action + reasons
│   ├── replay.py             # stream replay, parity check, latency, cost
│   ├── pipeline.py           # training pipeline + model card
│   └── cli.py                # `fraud-detect`
├── tests/                    # 41 tests: features, parity, thresholds, rules, training, API
├── Dockerfile
└── docker-compose.yml
```

## Deployment

```bash
make train
docker compose up --build        # serves the latest bundle from ./artifacts on :8000
```

The container reads `FRAUD_BUNDLE`, `FRAUD_RULES` and optionally `FRAUD_WARMUP` (a CSV of recent transactions to rebuild card history on start-up). The in-memory feature store is per process; `features/online.py` defines a `StateBackend` protocol for moving card state to a shared store when running several replicas.

## License

MIT, see [LICENSE](LICENSE).
