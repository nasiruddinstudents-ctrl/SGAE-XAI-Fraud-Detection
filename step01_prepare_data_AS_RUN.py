#!/usr/bin/env python3
"""
SGAE v2 — STEP 1: chronological split, past-only features, train-fitted preprocessing.

Replaces CELLS 2-3 of sgae_complete_experiments.py (v1). Fixes v1 issues:
  - scaler fit on all 590k rows before splitting        -> fit on TRAIN only
  - random stratified split                             -> chronological train / val / test
  - "24h count" was an unbounded running count          -> true 24h and 7d windows
  - "7-day mean" was a 7-transaction rolling mean       -> mean over prior 7 days
  - "merchant" count used ProductCD                     -> renamed honestly (distinct ProductCD)
  - card1/card2/addr1 counts used ALL rows (future+test)-> counts of STRICTLY EARLIER rows only
  - categorical codes fit on full data                  -> categories learned from TRAIN only

Every engineered feature for transaction i uses only transactions with
TransactionDT strictly earlier than i's (same-timestamp rows never see each other),
and never uses labels. This is verified automatically by a truncation test
(features recomputed on a time-truncated copy must be identical).

Usage (on Vast.ai):
  python step01_prepare_data.py --data_dir /workspace/ieee-cis --out_dir /workspace/sgae_v2/data

Inputs : train_transaction.csv, train_identity.csv (Kaggle IEEE-CIS)
Outputs: X_{train,val,test}.npy (float32, scaled), y_{train,val,test}.npy,
         meta.csv.gz (TransactionID, TransactionDT, split, key_id, isFraud),
         feature_names.json, preprocessing.json, split_report.json, leakage_audit.json
"""
import argparse, json, os, sys, time, hashlib
from collections import defaultdict, deque

import numpy as np
import pandas as pd

BIG = 10**9          # key offset; must exceed max TransactionDT (~1.6e7) + largest window
DAY = 86_400
WINDOWS = {"24h": DAY, "7d": 7 * DAY}
DEFAULT_KEY_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]


# ─────────────────────────────────────────────────────────────────────────────
# Loading and ordering
# ─────────────────────────────────────────────────────────────────────────────
def load(data_dir):
    tx = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    idf = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
    df = tx.merge(idf, on="TransactionID", how="left")
    # deterministic global order: time, then ID
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    return df


def add_key(df, key_cols):
    """Card-level grouping key. IEEE-CIS has no account ID; this is a documented proxy.
    Missing parts are kept as the literal 'NA' so the key is always defined.
    factorize() is label-free; key_id is used for grouping only, never as a feature."""
    parts = [df[c].astype(str).where(df[c].notna(), "NA") for c in key_cols]
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    df["key_id"], _ = pd.factorize(key, sort=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Chronological split
# ─────────────────────────────────────────────────────────────────────────────
def chrono_split(dt, train_frac, val_frac, gap_days):
    """Split a time-sorted array into train/val/test by row quantiles, moving each
    boundary forward so that no timestamp is shared across partitions. Optional gap
    (days) drops rows at the start of val and test to mimic label/processing delay."""
    n = len(dt)
    def boundary(frac):
        b = int(round(frac * n))
        b = min(max(b, 1), n - 1)
        # advance until timestamp changes
        return int(np.searchsorted(dt, dt[b - 1], side="right"))
    b1 = boundary(train_frac)
    b2 = boundary(train_frac + val_frac)
    split = np.full(n, "test", dtype=object)
    split[:b1] = "train"
    split[b1:b2] = "val"
    if gap_days > 0:
        g = gap_days * DAY
        split[(dt > dt[b1 - 1]) & (dt <= dt[b1 - 1] + g)] = "gap"
        split[(dt > dt[b2 - 1]) & (dt <= dt[b2 - 1] + g)] = "gap"
    return split


# ─────────────────────────────────────────────────────────────────────────────
# Past-only engineered features
# ─────────────────────────────────────────────────────────────────────────────
def _prior_windows(group, t):
    """For arrays already sorted by (group, t): return group_start, and for each row the
    index range [.., right) of rows in the same group with time STRICTLY < t."""
    T = group.astype(np.int64) * BIG + t.astype(np.int64)
    right = np.searchsorted(T, T, side="left")                   # excludes same-time rows
    group_start = np.searchsorted(T, group.astype(np.int64) * BIG, side="left")
    return T, right, group_start


def _prior_count_by(group_values, t, windows):
    """Counts of strictly-earlier rows sharing group_values (NaN group -> NaN count)."""
    valid = ~pd.isna(group_values)
    codes = np.full(len(t), -1, dtype=np.int64)
    codes[valid] = pd.factorize(pd.Series(group_values[valid]), sort=True)[0]
    order = np.lexsort((t, codes))
    g, tt = codes[order], t[order]
    T, right, gs = _prior_windows(g, tt)
    out = {"all": (right - gs).astype(float)}
    for name, w in windows.items():
        left = np.maximum(np.searchsorted(T, T - w, side="left"), gs)
        out[name] = (right - left).astype(float)
    res = {}
    for k, v in out.items():
        full = np.empty(len(t)); full[order] = v
        full[~valid] = np.nan
        res[k] = full
    return res


def _prior_distinct(group, t, cat, w):
    """Number of distinct `cat` values among same-group rows with time in [t-w, t).
    Arrays must be sorted by (group, t). Single O(n) pass."""
    n = len(t)
    out = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and group[j] == group[i]:
            j += 1
        counts = defaultdict(int)
        lo = hi = i                  # window = rows [lo, hi) of this group
        for r in range(i, j):
            while hi < r and t[hi] < t[r]:      # add strictly-earlier rows
                counts[cat[hi]] += 1; hi += 1
            while lo < hi and t[lo] < t[r] - w:  # evict rows older than window
                c = cat[lo]; counts[c] -= 1
                if counts[c] == 0: del counts[c]
                lo += 1
            out[r] = len(counts)
        i = j
    return out


def build_engineered(df):
    """All engineered features. Uses only TransactionDT, key_id, TransactionAmt,
    ProductCD, dist1, addr1, card1, card2 of strictly-earlier rows. No labels."""
    n = len(df)
    t_all = df["TransactionDT"].to_numpy(np.int64)
    order = np.lexsort((df["TransactionID"].to_numpy(), t_all, df["key_id"].to_numpy()))
    k = df["key_id"].to_numpy(np.int64)[order]
    t = t_all[order]
    amt = df["TransactionAmt"].to_numpy(float)[order]
    d1 = df["dist1"].to_numpy(float)[order]
    prod = pd.factorize(df["ProductCD"].fillna("NA"))[0][order]

    T, right, gs = _prior_windows(k, t)
    has_prev = right > gs
    prev = np.clip(right - 1, 0, None)

    f = {}
    f["card_is_first_tx"] = (~has_prev).astype(float)
    f["card_n_prior_tx_all"] = (right - gs).astype(float)
    for name, w in WINDOWS.items():
        left = np.maximum(np.searchsorted(T, T - w, side="left"), gs)
        f[f"card_n_prior_tx_{name}"] = (right - left).astype(float)
    # amount relative to mean amount of the prior 7 days
    S = np.concatenate([[0.0], np.cumsum(amt)])
    left7 = np.maximum(np.searchsorted(T, T - WINDOWS["7d"], side="left"), gs)
    cnt7 = right - left7
    mean7 = (S[right] - S[left7]) / np.where(cnt7 > 0, cnt7, 1)
    f["card_amt_ratio_prior_7d_mean"] = np.where(cnt7 > 0, amt / (mean7 + 1e-6), np.nan)
    f["card_secs_since_prev_tx"] = np.where(has_prev, (t - t[prev]).astype(float), np.nan)
    f["card_abs_dist1_change_vs_prev"] = np.where(has_prev, np.abs(d1 - d1[prev]), np.nan)
    f["card_n_distinct_productcd_prior_7d"] = _prior_distinct(k, t, prod, WINDOWS["7d"])

    out = pd.DataFrame(index=df.index)
    for name, v in f.items():
        full = np.empty(n); full[order] = v
        out[name] = full

    # entity-level prior counts (replace v1's full-dataset frequency counts)
    for col in ["card1", "card2", "addr1"]:
        r = _prior_count_by(df[col].to_numpy(object), t_all, {"7d": WINDOWS["7d"]})
        out[f"{col}_n_prior_tx_all"] = r["all"]
        out[f"{col}_n_prior_tx_7d"] = r["7d"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Train-fitted preprocessing
# ─────────────────────────────────────────────────────────────────────────────
EXPLICIT_FILLS = {                       # documented semantics for "no prior history"
    "card_amt_ratio_prior_7d_mean": 1.0,  # no deviation from (non-existent) baseline
    "card_abs_dist1_change_vs_prev": 0.0,
}

def fit_preprocessing(Xdf, is_train, cat_cols):
    tr = Xdf[is_train]
    prep = {"categories": {}, "median": {}, "min": {}, "max": {}, "explicit_fills": EXPLICIT_FILLS}
    for c in cat_cols:
        prep["categories"][c] = sorted(tr[c].dropna().astype(str).unique().tolist())
    return prep


def apply_preprocessing(Xdf, prep, is_train, cat_cols, fit_numeric=False):
    X = Xdf.copy()
    for c in cat_cols:   # unseen / missing categories -> -1
        X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna(), None),
                              categories=prep["categories"][c]).codes.astype(float)
    for c, v in EXPLICIT_FILLS.items():
        if c in X: X[c] = X[c].fillna(v)
    if "card_secs_since_prev_tx" in X:   # first tx: treat as "long ago" (train max)
        m = X.loc[is_train, "card_secs_since_prev_tx"].max()
        prep["explicit_fills"]["card_secs_since_prev_tx"] = float(m)
        X["card_secs_since_prev_tx"] = X["card_secs_since_prev_tx"].fillna(m)
    if fit_numeric:
        med = X[is_train].median()
        prep["median"] = {c: (float(v) if pd.notna(v) else 0.0) for c, v in med.items()}
    X = X.fillna(pd.Series(prep["median"]))
    if fit_numeric:
        prep["min"] = X[is_train].min().astype(float).to_dict()
        prep["max"] = X[is_train].max().astype(float).to_dict()
    mn, mx = pd.Series(prep["min"]), pd.Series(prep["max"])
    rng = (mx - mn).replace(0, 1.0)
    X = (X - mn) / rng                   # NOT clipped: out-of-range rate is reported
    return X.astype(np.float32), prep


# ─────────────────────────────────────────────────────────────────────────────
# Leakage audit
# ─────────────────────────────────────────────────────────────────────────────
def truncation_test(df, cutoff_dt, eng_full):
    """Recompute engineered features using ONLY rows with TransactionDT <= cutoff.
    If any feature used future rows, values for pre-cutoff rows would differ."""
    sub = df[df["TransactionDT"] <= cutoff_dt].copy()
    eng_sub = build_engineered(sub)
    a = eng_full.loc[sub.index].to_numpy()
    b = eng_sub.to_numpy()
    # rtol covers float64 rounding in cumulative sums (window sums are computed as
    # differences of a running cumsum, whose rounding depends on array length)
    same = np.isclose(a, b, equal_nan=True, rtol=1e-7, atol=1e-9)
    bad = {c: int((~same[:, i]).sum()) for i, c in enumerate(eng_sub.columns) if (~same[:, i]).any()}
    diff = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
    return {"cutoff_TransactionDT": int(cutoff_dt), "rows_checked": int(len(sub)),
            "max_abs_difference": float(diff.max()) if diff.size else 0.0,
            "features_checked": list(eng_sub.columns), "mismatching_features": bad,
            "passed": len(bad) == 0}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_frac", type=float, default=0.60)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--gap_days", type=float, default=0.0)
    ap.add_argument("--key_cols", default=",".join(DEFAULT_KEY_COLS))
    ap.add_argument("--skip_truncation_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    df = load(a.data_dir)
    key_cols = a.key_cols.split(",")
    df = add_key(df, key_cols)
    print(f"Loaded {len(df):,} rows | fraud rate {df.isFraud.mean():.4f} | keys {df.key_id.nunique():,}")

    dt = df["TransactionDT"].to_numpy()
    df["split"] = chrono_split(dt, a.train_frac, a.val_frac, a.gap_days)
    is_tr = (df["split"] == "train").to_numpy()

    eng = build_engineered(df)
    print(f"Engineered {eng.shape[1]} past-only features ({time.time()-t0:.0f}s)")

    audit = {}
    if not a.skip_truncation_test:
        for name in ["train", "val"]:
            cutoff = df.loc[df.split == name, "TransactionDT"].max()
            audit[f"truncation_test_at_end_of_{name}"] = truncation_test(df, cutoff, eng)
            print(f"Truncation test @ end of {name}: "
                  f"{'PASSED' if audit[f'truncation_test_at_end_of_{name}']['passed'] else 'FAILED'}")

    drop = {"TransactionID", "TransactionDT", "isFraud", "split", "key_id"}
    raw_cols = [c for c in df.columns if c not in drop]
    Xdf = pd.concat([df[raw_cols], eng], axis=1)
    cat_cols = [c for c in raw_cols if not pd.api.types.is_numeric_dtype(df[c])]

    prep = fit_preprocessing(Xdf, is_tr, cat_cols)
    X, prep = apply_preprocessing(Xdf, prep, is_tr, cat_cols, fit_numeric=True)
    feat = list(X.columns)

    y = df["isFraud"].to_numpy(np.int8)
    report = {"n_total": int(len(df)), "key_cols": key_cols, "gap_days": a.gap_days,
              "n_features": len(feat), "n_raw_features": len(raw_cols),
              "n_engineered_features": int(eng.shape[1]), "partitions": {}}
    for s in ["train", "val", "gap", "test"]:
        m = (df["split"] == s).to_numpy()
        if not m.any(): continue
        Xs = X.to_numpy()[m]
        report["partitions"][s] = {
            "n": int(m.sum()), "n_fraud": int(y[m].sum()), "fraud_rate": float(y[m].mean()),
            "TransactionDT_min": int(dt[m].min()), "TransactionDT_max": int(dt[m].max()),
            "day_min": round(float(dt[m].min() / DAY), 2), "day_max": round(float(dt[m].max() / DAY), 2),
            "n_keys": int(df.loc[m, "key_id"].nunique()),
            "frac_values_outside_train_range": float(((Xs < -1e-6) | (Xs > 1 + 1e-6)).mean()),
        }
        if s == "gap": continue
        np.save(os.path.join(a.out_dir, f"X_{s}.npy"), Xs)
        np.save(os.path.join(a.out_dir, f"y_{s}.npy"), y[m])
        report["partitions"][s]["sha_X"] = sha(Xs)

    P = report["partitions"]
    audit["partitions_time_ordered"] = bool(P["train"]["TransactionDT_max"] < P["val"]["TransactionDT_min"]
                                            and P["val"]["TransactionDT_max"] < P["test"]["TransactionDT_min"])
    audit["preprocessing_fit_on"] = "train only (categories, medians, min/max)"
    audit["labels_used_in_features"] = False
    audit["keys_seen_in_train_frac_of_test_rows"] = float(
        df.loc[df.split == "test", "key_id"].isin(set(df.loc[is_tr, "key_id"])).mean())
    audit["passed"] = audit["partitions_time_ordered"] and all(
        v["passed"] for k, v in audit.items() if k.startswith("truncation_test"))

    df[["TransactionID", "TransactionDT", "split", "key_id", "isFraud"]].to_csv(
        os.path.join(a.out_dir, "meta.csv.gz"), index=False)
    json.dump(feat, open(os.path.join(a.out_dir, "feature_names.json"), "w"), indent=1)
    json.dump(prep, open(os.path.join(a.out_dir, "preprocessing.json"), "w"), default=float)
    json.dump(report, open(os.path.join(a.out_dir, "split_report.json"), "w"), indent=2)
    json.dump(audit, open(os.path.join(a.out_dir, "leakage_audit.json"), "w"), indent=2)

    print(json.dumps({s: {k: v for k, v in d.items() if k in ("n", "n_fraud", "fraud_rate", "day_min", "day_max")}
                      for s, d in P.items()}, indent=1))
    print(f"Leakage audit passed: {audit['passed']}  ({time.time()-t0:.0f}s total)")
    if not audit["passed"]:
        sys.exit("LEAKAGE AUDIT FAILED — do not proceed to step 2.")


if __name__ == "__main__":
    main()
