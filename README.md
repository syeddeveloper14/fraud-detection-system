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
