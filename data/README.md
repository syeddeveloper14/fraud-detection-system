# Data

Raw data is not committed; place files in `data/raw/` (git-ignored).

## Kaggle: Credit Card Transactions Fraud Detection Dataset

Simulated with the Sparkov data generator: ~1.3M training and ~0.56M test transactions from 1,000 cards and 800 merchants, January 2019 to December 2020.

1. Download from https://www.kaggle.com/datasets/kartik2112/fraud-detection (Kaggle account required), or with the Kaggle CLI:
   ```bash
   kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
   ```
2. You should have `data/raw/fraudTrain.csv` and `data/raw/fraudTest.csv`.
3. `fraud-detect train --config configs/kaggle_sparkov.yaml`

The two files are concatenated and re-split by time; the original train/test boundary is not used, because history features need the full stream and the split should be chronological anyway.

## Synthetic data

`fraud-detect simulate --n-cards 1000 --days 180 --out data/raw/synthetic.csv` writes a dataset with the same 22 columns. See `src/fraud_detection/data/simulator.py` for the fraud patterns it injects.

## Columns used

| Column | Used for |
|---|---|
| `trans_date_trans_time` | Event time (stream order, windows, hour) |
| `cc_num` | Card key for history features |
| `merchant`, `category` | Target encodings, categorical feature, new-merchant flag |
| `amt` | Amount features and cost-weighted evaluation |
| `lat`, `long`, `merch_lat`, `merch_long` | Home-to-merchant distance, travel speed |
| `city_pop`, `dob` | City size, cardholder age |
| `trans_num` | Transaction id (tie-break for stream order) |
| `is_fraud` | Label |

Name, street, job and zip are deliberately ignored.
