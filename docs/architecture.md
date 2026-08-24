# Architecture and design decisions

## Goals

1. Decisions in real time (single-digit milliseconds of model work per transaction).
2. No training/serving skew: the features a model is trained on are exactly the features it sees in production.
3. Offline numbers that predict online behaviour: evaluation mirrors deployment in time order.
4. Decisions tied to money, not to an abstract metric.
5. Operations can react to a new attack the same day, without retraining.

## Data flow

```
raw transactions ──► prepare (parse, dedupe, sort into stream order)
                    │
                    ├─► static_features  ─┐
                    └─► history_features ─┴─► feature frame ─► FraudModel.model_frame ─► estimator
                                                              (target encoding, codes)
online request ──► static_row + OnlineFeatureStore.compute ─► FraudModel.row_vector ─► estimator
```

Both paths share the feature spec (`features/spec.py`), which fixes the column order and encoding. Categoricals are stored as integer codes so the online path can build a NumPy row without pandas. Building pandas frames per request was the dominant cost; switching the online path to plain dicts and a NumPy row cut replay latency by roughly 4x. Most of what remains is the estimator's own input validation.

## Decision 1: point-in-time windows defined as [t − W, t)

Window features count events with timestamps strictly earlier than the current one. Two events in the same second do not see each other in window features, but "previous transaction" features (gap, distance, speed) use the previous event in stream order (timestamp, then transaction id). Both conventions are easy to reproduce online and are pinned down by `test_window_features_exclude_current_and_simultaneous_events`.

## Decision 2: chronological splits with two validation halves

```
|------------- train -------------|-- calib --|-- tune --|------ test ------|
```

* **Train**: fit target encoders (out-of-fold) and the estimator.
* **Calib**: fit the calibrator. Using later data than training corrects both the class-weight distortion and some temporal drift.
* **Tune**: choose thresholds on calibrated scores. Separate from calibration so the threshold search does not reuse the calibrator's data.
* **Test**: report once.

Random splits put a card's future transactions in the training set, which leaks behaviour and inflates every metric.

## Decision 3: Platt scaling instead of isotonic by default

Isotonic regression fitted on a short calibration window with few positives produces a coarse step function. Many transactions end up with identical scores, which lowers PR-AUC and makes thresholds jump between steps. Platt scaling on the logit of the raw score has two parameters, is strictly monotonic and works with few positives. Isotonic remains available (`model.calibration: isotonic`) for large datasets.

## Decision 4: three actions with explicit costs

| truth \ action | approve | review | decline |
|---|---|---|---|
| fraud | amount + chargeback fee | review cost + (1 − catch rate)·(amount + fee) | 0 |
| legit | 0 | review cost | false-decline friction |

Amount-weighted costs push the model's decisions toward high-value fraud. The review band only exists if reviewing is cheaper than both alternatives somewhere in the score range; with some cost settings the optimum is approve/decline only, and that is a legitimate result. Analyst capacity enters as a hard cap on the review rate.

## Decision 5: rules as data

Rules live in YAML, are validated at load time and are evaluated by a small interpreter. They can only make a decision stricter. Typical uses: a burst pattern seen in a new attack, a merchant under investigation, a temporary block on a BIN range. Every rule hit is returned in the API response and counted in replay.

## Decision 6: occlusion-based reason codes

Per-transaction explanations replace each of the ten globally most important features with a typical legitimate value and re-score the transaction in one batched call. Features whose replacement lowers the score the most are reported. This is model-agnostic (works for gradient boosting, LightGBM and logistic regression) and costs one extra predict on flagged transactions only.

## Decision 7: the bundle carries everything needed to serve and monitor

`bundle.joblib` holds the fitted encoders, estimator, calibrator, thresholds, reason coder and the drift reference profile; `metadata.json` and `model_card.md` describe the run. Serving and monitoring never need the training data.

## Scaling notes

* **State**: `OnlineFeatureStore` takes a `StateBackend`. The in-memory backend serves one process; a Redis or DynamoDB backend keyed by card number lets several replicas share state. Per-card writes must be serialised (a per-key lock or a single partition per card in a stream processor) to keep stream order.
* **Throughput**: the API serialises access to the store with a lock. For high volume, partition traffic by card so each worker owns a disjoint set of cards and no lock is needed.
* **Label delay**: chargebacks arrive weeks later. Retraining jobs should exclude the most recent weeks or treat them as unlabelled.

## Known limitations

* Target encodings are frozen at training time; a merchant that becomes risky after training is only caught through rules or other features until the next retrain.
* The simulator's fraud patterns are cleaner than real fraud; treat synthetic metrics as a smoke test only.
* PSI flags distribution change, not performance change. Pair it with delayed-label performance tracking once chargebacks arrive.
