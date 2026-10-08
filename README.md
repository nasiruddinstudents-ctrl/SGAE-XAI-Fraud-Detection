# SGAE: Shapley-Guided Adaptive Ensemble for Explainable Financial Fraud Detection

[![Status](https://img.shields.io/badge/Status-Under%20Review%20(CAAI%20TIT)-yellow)](https://arxiv.org/abs/2604.14231)
[![arXiv](https://img.shields.io/badge/arXiv-2604.14231-b31b1b)](https://arxiv.org/abs/2604.14231)

Code for the paper:

> **When Does Shapley-Guided Ensembling Help? A Leakage-Controlled Chronological Evaluation of SGAE for Explainable Financial Fraud Detection**<br>
> Mohammad Nasir Uddin, Westcliff University<br>
> *Under review at CAAI Transactions on Intelligence Technology, 2026 (revision R1)*

## Repository structure
- **Root:** leakage-controlled chronological pipeline used in the revised manuscript (R1), run in order `step01` → `step07`. Chronological train/validation/test split; scalers, velocity features and graph edges fit on training-window data only; SGAE calibrated on validation and applied to every test row. See `DATA_FLOW_SPEC.md`, `leakage_audit.json` and `split_report.json`.
- **`logs/`:** run logs for each step.
- **`results/`:** final test metrics, explanation-quality results and ablation ladder reported in the manuscript.
- **`manifest.txt`:** SHA-256 checksums of the scripts and outputs as run.
- **`legacy/`:** original experiment code from the first submission, kept for transparency. It contains known issues (pre-split scaling, random split, full-graph construction, test-set tuning) that the revision corrects; do not use it to reproduce the revised results.

## Data
Uses the IEEE-CIS Fraud Detection dataset, available from Kaggle under its own terms (not redistributed here).
