# Feature catalogue

All features are computed at decision time from information available before the transaction completes. History features use only the card's earlier transactions (window definition: [t − W, t)). The same definitions are implemented in `features/history.py` (batch) and `features/online.py` (real time); `tests/test_feature_parity.py` checks they agree.

## Transaction

| Feature | Definition |
|---|---|
| `amt` | Transaction amount |
| `log_amt` | log(1 + amount) |
| `hour` | Hour of day (0-23) |
| `day_of_week` | Day of week (Mon=0) |
| `is_night` | 1 if hour is 22:00-05:59 |
| `age_years` | Cardholder age at transaction time |
| `distance_home_km` | Distance from cardholder home to merchant |
| `log_city_pop` | log(1 + population of cardholder city) |

## Card history (point-in-time)

| Feature | Definition |
|---|---|
| `txn_count_1h` | Card transactions in the previous hour |
| `txn_count_24h` | Card transactions in the previous 24 hours |
| `txn_count_7d` | Card transactions in the previous 7 days |
| `amt_sum_1h` | Card spend in the previous hour |
| `amt_sum_24h` | Card spend in the previous 24 hours |
| `amt_sum_7d` | Card spend in the previous 7 days |
| `amt_mean_30d` | Mean card transaction amount over the previous 30 days |
| `amt_std_30d` | Std of card transaction amounts over the previous 30 days |
| `amt_zscore_30d` | (amt - mean_30d) / (std_30d + 1) |
| `amt_ratio_30d` | amt / (mean_30d + 1) |
| `secs_since_last` | Seconds since the card's previous transaction |
| `km_from_last` | Distance between this merchant and the previous one |
| `speed_kmh` | Implied travel speed from the previous transaction |
| `is_new_merchant` | 1 if the card has never used this merchant before |
| `n_prior_txns` | Number of earlier transactions on the card |

## Target encodings

| Feature | Definition |
|---|---|
| `merchant_risk` | Smoothed historical fraud rate of the merchant |
| `category_risk` | Smoothed historical fraud rate of the category |

## Categorical

| Feature | Definition |
|---|---|
| `category` | Merchant category (categorical) |

## Missing values

History features are missing (NaN) when the card has no qualifying earlier transaction: no previous transaction for `secs_since_last`, `km_from_last` and `speed_kmh`; fewer than two transactions in 30 days for `amt_std_30d`. Gradient boosting handles NaN natively; the logistic baseline uses median imputation with missing-value indicators.

## Fields never used as model inputs

Cardholder name, street address, job and zip code are present in the source data but are not features. Using them risks learning proxies for protected characteristics and adds little signal beyond location and age, which are already captured.
