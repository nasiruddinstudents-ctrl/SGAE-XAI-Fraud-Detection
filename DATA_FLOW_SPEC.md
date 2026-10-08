# SGAE v2 — Data-Flow Specification

Controlling question: *How reliable are the explanations used by a SHAP-guided fraud ensemble, and does using them improve detection over appropriate baselines?*

Rule for every step: a quantity used to make a prediction for a test transaction may depend only on (a) training-partition labels, (b) validation-partition labels for calibration and threshold selection, and (c) unlabeled features of transactions strictly earlier in time. Test labels are touched once, by the final evaluation script, after every model, threshold and SGAE parameter is frozen.

## 1. Data and ordering
IEEE-CIS `train_transaction.csv` left-joined with `train_identity.csv` on TransactionID (590,540 rows). Rows are sorted by (TransactionDT, TransactionID); TransactionDT is seconds from an unknown reference and covers about 182 days.

## 2. Partitions (step 1)
The data is split chronologically by row count: 60% train, 20% validation, 20% test. Each boundary is moved forward so that no timestamp appears in two partitions. An optional `--gap_days` drops a buffer between partitions to mimic label delay. There is no shuffling and no stratification. Exact counts, prevalence, day ranges and hashes are written to `split_report.json`.

| Partition | Used for |
|---|---|
| Train | Model fitting, preprocessing fits, SMOTE/Tomek (train only), SHAP background |
| Validation | Early stopping, hyperparameter choice, decision thresholds, SGAE calibration (σ_A, τ_a, K, negative-agreement weight), probability calibration |
| Test | One final evaluation of frozen models; no tuning of any kind |

Cross-validation, if reported, uses expanding-window time-series folds inside the train partition only.

## 3. Entity key
IEEE-CIS has no account identifier. The documented proxy is `card1|card2|card3|card4|card5|card6`, with missing parts kept as "NA". It is used for grouping, sequences and cluster bootstrap, never as a feature. `keys_seen_in_train_frac_of_test_rows` is reported to show how often test cards have training history.

## 4. Engineered features (all past-only, no labels)
For transaction *i* at time *t*, each feature uses only rows with TransactionDT < *t*. Same-timestamp rows do not see each other.

| Feature | Definition | No-history value |
|---|---|---|
| card_is_first_tx | 1 if the card has no earlier transaction | — |
| card_n_prior_tx_all / _24h / _7d | Count of the card's earlier transactions (all / in [t−24h, t) / in [t−7d, t)) | 0 |
| card_amt_ratio_prior_7d_mean | Amount ÷ mean amount of the card's transactions in [t−7d, t) | 1.0 |
| card_secs_since_prev_tx | *t* − time of the card's previous transaction | train maximum |
| card_abs_dist1_change_vs_prev | abs(dist1 − dist1 of the previous transaction) | 0 |
| card_n_distinct_productcd_prior_7d | Distinct ProductCD values in [t−7d, t). This is a product code, not a merchant. | 0 |
| {card1, card2, addr1}_n_prior_tx_all / _7d | Earlier transactions sharing that value | NaN if the value is missing, then median-imputed |

Raw IEEE-CIS columns are kept, except TransactionID, TransactionDT and isFraud.

## 5. Preprocessing (fit on train only)
- **Categorical codes:** categories are learned from train. Unseen or missing values become −1.
- **Imputation:** median imputation, using train medians.
- **Scaling:** Min-Max scaling, using train min and max. Validation and test values are not clipped, and the out-of-range fraction is reported.
- **Resampling:** SMOTE+Tomek is applied only to training rows, and only inside model fitting.

## 6. Leakage audit (automated, `leakage_audit.json`)
1. **Truncation test.** Features are recomputed on a copy of the data cut at the end of train, and again at the end of validation. Pre-cutoff values must match the full run (relative tolerance 1e-7, for float rounding). A future-dependent feature fails this test. Checked: v1's `card1_tx_count` changes for 100% of rows under truncation.
2. **Time ordering.** max DT(train) < min DT(val) and max DT(val) < min DT(test).
3. **Provenance.** Preprocessing parameters are saved and fit only on train.

Step 1 exits with an error if any check fails.

## 7. Later steps (to be implemented against this spec)
- **Step 2, sequence models.** The LSTM input for transaction *i* is the card's last L transactions strictly before *t* plus *i* itself. L is chosen on validation. Inputs are left-padded with a mask, and first transactions have length 1. The Transformer is described honestly: either a history of transactions, or a TabTransformer over the fields of one row, not called "sequential."
- **Step 3, historical graph.** An edge j→i exists only if *j* shares card1 or addr1 with *i* and TransactionDT(j) < TransactionDT(i). The graph for a partition includes only that partition's nodes and their earlier history, never later nodes. Edge counts per partition are reported. Message passing never sees test labels.
- **Step 4, SGAE.**
  - The SHAP background is drawn from train. Attributions are computed for every validation and test row, in a common feature space.
  - σ_A, K, τ_a, the tanh scale and the negative-agreement weight are chosen on validation by a declared grid and criterion (PR-AUC).
  - The frozen rule is applied to all test rows.
  - Baselines are equal weight, a validation-tuned constant weight, permuted-SHAP agreement, SGAE without the 0.60 rule, and a confidence gate. Stacking and a learned gate are added if time allows.
- **Step 5, evaluation.**
  - PR-AUC is primary. Secondary metrics are ROC-AUC, F1, precision, recall and MCC at validation-chosen thresholds, plus Brier score, ECE and reliability plots.
  - Paired bootstrap confidence intervals resample by card key, with Holm correction across the declared comparison family.
  - The mechanism test stratifies test cases by local LSTM explanation reliability.
- **Step 6, explanation quality.** Formal sufficiency and comprehensiveness definitions on one scale, with bootstrap confidence intervals. Kendall's W is reported with CIs and background and sample-size sensitivity, plus local agreement. GNN results go in a separate table labeled "frozen embedding dimensions."

## 8. v1 → v2 corrections (for the response letter)
| v1 (`sgae_complete_experiments.py`, `gnn_graphsage_pipeline.py`) | v2 |
|---|---|
| MinMaxScaler fit on all 590k rows before the split | Fit on train only |
| Random stratified 80/20 split, stratified shuffled CV | Chronological 60/20/20; time-series CV inside train |
| "tx_count_24h" = running count with no window | True 24h and 7d windows, strictly prior |
| "amt_vs_7d_mean" = 7-transaction rolling mean including the current row | Mean over the prior 7 days, excluding the current row |
| "unique_merchant_7d" = expanding distinct ProductCD | Distinct ProductCD in the prior 7 days, named as such |
| card1/card2/addr1 counts over the full dataset (future and test rows) | Counts of strictly earlier rows |
| LSTM with SEQ_LEN = 1 | Real per-card history, or described as tabular |
| GNN graph built on all nodes before the split | Historical, time-respecting edges per partition |
| SGAE σ_A and SHAP background from the test set; τ_a hard-coded at 0.60 | Background from train; all parameters chosen on validation |
| SGAE evaluated on 500 test rows | Applied to every test row |
| F1-optimal threshold chosen on the test set (`compute_metrics(y_test, …)` with no threshold) | Threshold chosen on validation, applied unchanged to test |
