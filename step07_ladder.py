#!/usr/bin/env python3
"""
SGAE v2 — STEP 7: REPRODUCTION LADDER (explains why v1 and v2 results differ).

Rung 0  ORIGINAL v1 RESULT: metrics recomputed from the saved v1 GNN held-out predictions
        (gnn_test_probs.csv in the v1 repository). No training.
Rung 1  v1 DESIGN, re-implemented: v1 preprocessing fitted on ALL rows, v1 features, v1
        random stratified 80/20 split (the exact v1 test set, verified against the saved
        v1 TransactionIDs), v1 graph (undirected; card1 groups of 2-50 rows, first 10;
        addr1 x ProductCD groups of <=30, first 5; built over all rows), BatchNorm,
        transductive training over all nodes.
          XGBoost uses the v1 SGAE-script features (incl. full-dataset card/addr counts,
          -999 fill); the GNN uses the v1 GNN-script features (label encoding, median
          imputation and Min-Max over all rows). Also reports F1 at the TEST-optimal
          threshold (v1 practice) vs the validation-optimal threshold.
Rung 2  + CHRONOLOGICAL SPLIT (the v2 60/20/20 split). Everything else as rung 1.
Rung 3  + v2 FEATURES AND TRAIN-ONLY PREPROCESSING (step-1 arrays). GNN still uses the v1
        undirected graph over all rows with BatchNorm: this is also the RETROSPECTIVE
        (transductive) setting, where a closed period is audited after the fact and later
        transactions' unlabelled features may legitimately be used.
          XGBoost at rung 3 = the v2 XGBoost (step 2/5), so it is taken from step 5.
Rung 4  + FORWARD-ONLY HISTORICAL GRAPH, LayerNorm, test nodes unseen in training = v2 GNN
        (taken from step 5). This is the real-time setting.

Fixed across rungs (declared): XGBoost max_depth 8, colsample_bytree 0.5 (the v2 selection),
lr 0.05, early stopping on the rung's validation set; GNN = v1 architecture (SAGE 128 -> 64,
MLP head 256-128-64), Adam 1e-3, full-batch, early stopping on validation PR-AUC. Seeds 42,
43, 44. Differences from the original v1 GNN run (NeighborLoader mini-batches, 2-class
softmax, 5-fold CV) mean rung 1 is a re-implementation, compared against rung 0.
Metrics: test PR-AUC (primary) and ROC-AUC, mean over seeds; seed-42 95% CI by card-key
cluster bootstrap. Note the test set itself changes between rung 1 (random) and rung 2
(chronological): that change is part of the split effect.

Usage
  python step07_ladder.py --raw_dir /workspace/ieee-cis --data_dir /workspace/sgae_v2/data \
     --step05_dir /workspace/sgae_v2/step05 --v1_gnn_probs /workspace/gnn_test_probs.csv \
     --out_dir /workspace/sgae_v2/step07
"""
import argparse, itertools, json, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step03_gnn as s3
import step05_locked_test as s5

SEEDS = [42, 43, 44]


# ─────────────────────────────────────────────────────────────────────────────
# v1 preprocessing (faithful to the two v1 scripts)
# ─────────────────────────────────────────────────────────────────────────────
def nonnum(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def v1_sgae_features(df):
    """v1 sgae_complete_experiments.py CELL 2 (all computed on ALL rows)."""
    d = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    g = d.groupby("card1")
    d["time_since_last_tx"] = g["TransactionDT"].diff().fillna(0)
    d["tx_count_24h"] = g.cumcount() + 1                                   # expanding count
    d["amt_vs_7d_mean"] = d["TransactionAmt"] / (g["TransactionAmt"].transform(lambda x: x.rolling(7, min_periods=1).mean()) + 1e-6)
    first = (~d.duplicated(["card1", "ProductCD"])).astype(int)
    d["unique_merchant_7d"] = first.groupby(d["card1"]).cumsum()          # expanding distinct ProductCD
    d["geo_dist_proxy"] = g["dist1"].diff().abs().fillna(0)
    d["card1_tx_count"] = g["card1"].transform("count")
    d["card2_tx_count"] = d.groupby("card2")["card2"].transform("count").fillna(0)
    d["addr1_tx_count"] = d.groupby("addr1")["addr1"].transform("count").fillna(0)
    for c in nonnum(d):
        d[c] = pd.Categorical(d[c]).codes
    cols = [c for c in d.columns if c not in ("TransactionID", "isFraud", "TransactionDT")]
    X = MinMaxScaler().fit_transform(d[cols].fillna(-999).to_numpy(np.float64)).astype(np.float32)
    return pd.DataFrame(X, index=d["TransactionID"]).loc[df["TransactionID"]].to_numpy(np.float32)


def v1_gnn_features(df):
    """v1 gnn_graphsage_pipeline.py step 2 (all fitted on ALL rows)."""
    f = df.drop(columns=["TransactionID", "TransactionDT", "isFraud"]).copy()
    for c in nonnum(f):
        f[c] = LabelEncoder().fit_transform(f[c].fillna("MISSING").astype(str))
    for c in f.columns:
        if f[c].isna().any(): f[c] = f[c].fillna(f[c].median())
    return MinMaxScaler().fit_transform(f.to_numpy(np.float64)).astype(np.float32)


def v1_graph(df):
    """v1 graph: undirected, over all rows in file order."""
    src, dst = [], []
    def add(groups, max_group, max_nb):
        for trans in groups.values():
            if 2 <= len(trans) <= max_group:
                for a, b in itertools.combinations(trans[:max_nb], 2):
                    src.extend([a, b]); dst.extend([b, a])
    card = defaultdict(list)
    for i, c in enumerate(df["card1"].fillna(-1).astype(int).to_numpy()): card[c].append(i)
    add(card, 50, 10)
    mk = (df["addr1"].fillna(-1).astype(str) + "_" + df["ProductCD"].fillna("UNK").astype(str)).to_numpy()
    merch = defaultdict(list)
    for i, m in enumerate(mk): merch[m].append(i)
    add(merch, 30, 5)
    return np.array([src, dst], dtype=np.int64)


def adjacency_any(e, n, dev):
    deg = np.bincount(e[1], minlength=n).astype(np.float32)
    val = 1.0 / np.maximum(deg[e[1]], 1)
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack([e[1], e[0]])), torch.from_numpy(val), (n, n)).coalesce().to(dev)


class LadderGNN(nn.Module):
    """v1 architecture; BatchNorm (v1) by default."""
    def __init__(self, din, hidden=128, emb=64, drop=0.25):
        super().__init__()
        self.c1, self.n1 = s3.SAGE(din, hidden), nn.BatchNorm1d(hidden)
        self.c2, self.n2 = s3.SAGE(hidden, emb), nn.BatchNorm1d(emb)
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(emb, 256), nn.ReLU(), nn.Dropout(drop), nn.Linear(256, 128), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))
    def forward(self, x, A):
        h = self.drop(F.relu(self.n1(self.c1(x, A))))
        h = self.drop(F.relu(self.n2(self.c2(h, A))))
        return self.head(h).squeeze(-1)


def train_gnn(X, y, A, tr, va, seed, dev, a):
    torch.manual_seed(seed); np.random.seed(seed)
    Xg, yg = torch.from_numpy(X).to(dev), torch.from_numpy(y.astype(np.float32)).to(dev)
    trt = torch.from_numpy(tr).to(dev)
    model = LadderGNN(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = float((y[tr] == 0).sum() / y[tr].sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
    best, state, bad = -1, None, 0
    for step in range(1, a.max_steps + 1):
        model.train()
        loss = lossf(model(Xg, A)[trt], yg[trt])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % a.eval_every == 0:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xg, A)).cpu().numpy()
            ap = average_precision_score(y[va], p[va])
            if ap > best + 1e-4: best, bad, state = ap, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            else: bad += 1
            if bad >= a.patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xg, A)).cpu().numpy(), best


def train_xgb(X, y, tr, va, seed, rounds):
    import xgboost as xgb
    p = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
         "device": "cuda" if torch.cuda.is_available() else "cpu", "learning_rate": 0.05, "subsample": 0.8,
         "max_depth": 8, "colsample_bytree": 0.5, "seed": seed}
    bst = xgb.train(p, xgb.DMatrix(X[tr], label=y[tr]), rounds, evals=[(xgb.DMatrix(X[va], label=y[va]), "v")],
                    early_stopping_rounds=100, verbose_eval=False)
    bst = bst[: bst.best_iteration + 1]
    return bst.predict(xgb.DMatrix(X)), None


def f1_at(y, p, th):
    yh = p >= th; tp = (yh & (y == 1)).sum()
    return float(2 * tp / (yh.sum() + (y == 1).sum()))


def summarize(y, P, va, te, keys, n_boot, seed=0):
    """P: list of full-length probability arrays (one per seed)."""
    prs = [average_precision_score(y[te], p[te]) for p in P]
    rocs = [roc_auc_score(y[te], p[te]) for p in P]
    rk = s5.Ranker(P[0][te], y[te])
    kc = pd.factorize(keys[te])[0]; nk = kc.max() + 1; rng = np.random.default_rng(seed)
    bs = [rk.ap(np.bincount(rng.integers(0, nk, nk), minlength=nk)[kc].astype(float)) for _ in range(n_boot)]
    th_val = s5.f1_threshold(y[va], P[0][va])
    pr_, rc_, th_ = precision_recall_curve(y[te], P[0][te])
    f1s = 2 * pr_[:-1] * rc_[:-1] / np.maximum(pr_[:-1] + rc_[:-1], 1e-12)
    return {"pr_auc_mean": float(np.mean(prs)), "pr_auc_by_seed": [float(v) for v in prs],
            "roc_auc_mean": float(np.mean(rocs)), "pr_auc_seed42_ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
            "f1_val_threshold_seed42": f1_at(y[te], P[0][te], th_val),
            "f1_test_optimal_threshold_seed42": float(f1s.max())}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    for k in ["raw_dir", "data_dir", "step05_dir", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--v1_gnn_probs", default=None)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--xgb_rounds", type=int, default=3000)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.max_steps, a.eval_every, a.patience, a.xgb_rounds, a.n_boot = 60, 5, 4, 50, 50
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    tx = pd.read_csv(os.path.join(a.raw_dir, "train_transaction.csv"))
    idf = pd.read_csv(os.path.join(a.raw_dir, "train_identity.csv"))
    df = tx.merge(idf, on="TransactionID", how="left")                     # v1 file order
    y = df["isFraud"].to_numpy(int); n = len(df); ids = df["TransactionID"].to_numpy()
    meta = pd.read_csv(os.path.join(a.data_dir, "meta.csv.gz")).set_index("TransactionID").loc[ids]
    keys = meta["key_id"].to_numpy()
    out = {"rungs": {}, "notes": {}}
    print(f"Loaded {n:,} rows in v1 file order | device {dev}", flush=True)

    # ── splits ───────────────────────────────────────────────────────────────
    trv, te_r = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=42)       # v1 exact
    tr_r, va_r = train_test_split(trv, test_size=0.1, stratify=y[trv], random_state=42)
    if a.v1_gnn_probs:
        v1 = pd.read_csv(a.v1_gnn_probs)
        same = set(v1["TransactionID"]) == set(ids[te_r])
        out["notes"]["v1_test_set_reproduced_exactly"] = bool(same)
        assert same, "could not reproduce the v1 random test set"
        # v1 saved TransactionID as trans_ids[test_idx_arr] (shuffled order) but labels and
        # probabilities from out[mask] (ascending row order). Realign: row k of the file
        # belongs to the k-th smallest test index. Labels must then match exactly.
        lab_as_saved = pd.Series(y, index=ids).loc[v1["TransactionID"]].to_numpy()
        ids_aligned = ids[np.sort(te_r)]
        lab_aligned = y[np.sort(te_r)]
        out["notes"]["v1_saved_file_id_column_misordered"] = bool((lab_as_saved != v1["true_label"].to_numpy()).any())
        out["notes"]["v1_labels_match_after_realigning_ids"] = bool((lab_aligned == v1["true_label"].to_numpy()).all())
        assert out["notes"]["v1_labels_match_after_realigning_ids"], "v1 labels do not match even after realignment"
        v1["TransactionID"] = ids_aligned
        out["rungs"]["0_original_v1_saved_predictions"] = {
            "gnn": {"pr_auc": float(average_precision_score(v1["true_label"], v1["gnn_prob"])),
                    "roc_auc": float(roc_auc_score(v1["true_label"], v1["gnn_prob"]))}}
        print(f"Rung 0 (saved v1 GNN): {out['rungs']['0_original_v1_saved_predictions']['gnn']}", flush=True)
    sp = meta["split"].to_numpy()
    tr_c, va_c, te_c = (np.where(sp == s)[0] for s in ["train", "val", "test"])

    # ── features and graphs ──────────────────────────────────────────────────
    X_sg = v1_sgae_features(df); X_gn = v1_gnn_features(df)
    parts = [np.load(os.path.join(a.data_dir, f"X_{s}.npy")) for s in ["train", "val", "test"]]
    m_all = pd.read_csv(os.path.join(a.data_dir, "meta.csv.gz"))
    m_all = m_all[m_all["split"] != "gap"]
    X_v2 = pd.DataFrame(np.concatenate(parts), index=m_all["TransactionID"].to_numpy()).loc[ids].to_numpy(np.float32)
    e1 = v1_graph(df)
    A1 = adjacency_any(e1, n, dev)
    out["notes"]["v1_graph_edges_directed_pairs"] = int(e1.shape[1])
    part = np.empty(n, dtype=object); part[tr_c], part[va_c], part[te_c] = "train", "val", "test"
    out["notes"]["v1_graph_edges_by_partition_under_chrono_split"] = pd.Series(
        [f"{p}->{q}" for p, q in zip(part[e1[0]], part[e1[1]])]).value_counts().to_dict()
    out["notes"]["v1_graph_edges_from_later_to_earlier_transaction"] = int(
        (df["TransactionDT"].to_numpy()[e1[0]] > df["TransactionDT"].to_numpy()[e1[1]]).sum())
    print(f"Features and v1 graph built ({e1.shape[1]:,} directed pairs, {time.time()-t0:.0f}s)", flush=True)

    def rung(name, Xx, Xg_, tr, va, te, do_xgb=True):
        res = {}
        if do_xgb:
            P = [train_xgb(Xx, y, tr, va, s, a.xgb_rounds)[0] for s in SEEDS]
            res["xgb"] = summarize(y, P, va, te, keys, a.n_boot)
        P = [train_gnn(Xg_, y, A1, tr, va, s, dev, a)[0] for s in SEEDS]
        res["gnn"] = summarize(y, P, va, te, keys, a.n_boot)
        out["rungs"][name] = res
        print(f"{name}: " + " | ".join(f"{m} PR-AUC {r['pr_auc_mean']:.4f} ROC {r['roc_auc_mean']:.4f}" for m, r in res.items())
              + f" ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(a.out_dir, "step07_ladder.json"), "w"), indent=2, default=float)

    rung("1_v1_design_random_split", X_sg, X_gn, tr_r, va_r, te_r)
    rung("2_plus_chronological_split", X_sg, X_gn, tr_c, va_c, te_c)
    rung("3_plus_v2_features_train_only_preprocessing__retrospective_GNN", X_v2, X_v2, tr_c, va_c, te_c, do_xgb=False)

    # rung 3 XGBoost and rung 4 GNN = v2 locked results (step 5)
    sv = np.load(os.path.join(a.step05_dir, "test_predictions.npz"))
    order = pd.Series(np.arange(len(sv["TransactionID"])), index=sv["TransactionID"]).loc[ids[te_c]].to_numpy()
    yte = y[te_c]
    assert (np.load(os.path.join(a.data_dir, "y_test.npy"))[order] == yte).all(), "step-5 predictions misaligned"
    def v2(prefix):
        prs = [average_precision_score(yte, sv[f"{prefix}{s}"][order]) for s in SEEDS]
        return {"pr_auc_mean": float(np.mean(prs)), "pr_auc_by_seed": [float(v) for v in prs],
                "roc_auc_mean": float(np.mean([roc_auc_score(yte, sv[f"{prefix}{s}"][order]) for s in SEEDS])),
                "source": "step 5 locked test"}
    out["rungs"]["3_plus_v2_features_train_only_preprocessing__retrospective_GNN"]["xgb"] = v2("xgb_seed")
    out["rungs"]["4_plus_forward_only_graph__v2_real_time"] = {"xgb": v2("xgb_seed"), "gnn": v2("gnn_seed")}
    out["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out_dir, "step07_ladder.json"), "w"), indent=2, default=float)

    print("\nREPRODUCTION LADDER — test PR-AUC (mean of 3 seeds) / ROC-AUC")
    for k, r in out["rungs"].items():
        print(f"  {k}")
        for m, v in r.items():
            pr = v.get("pr_auc", v.get("pr_auc_mean")); rc = v.get("roc_auc", v.get("roc_auc_mean"))
            extra = ""
            if "f1_test_optimal_threshold_seed42" in v:
                extra = f" | F1 val-thr {v['f1_val_threshold_seed42']:.3f} vs test-thr {v['f1_test_optimal_threshold_seed42']:.3f}"
            print(f"      {m:4s} PR-AUC {pr:.4f}  ROC-AUC {rc:.4f}{extra}")
    if "v1_saved_file_id_column_misordered" in out["notes"]:
        print(f"  v1 file ID column misordered: {out['notes']['v1_saved_file_id_column_misordered']} | "
              f"labels match after realignment: {out['notes']['v1_labels_match_after_realigning_ids']}")
    print(f"  v1 graph edges pointing backward in time: {out['notes']['v1_graph_edges_from_later_to_earlier_transaction']:,}")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
