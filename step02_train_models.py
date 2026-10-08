#!/usr/bin/env python3
"""
SGAE v2 — STEP 2: train base models on TRAIN, select on VALIDATION, save per-row
VALIDATION probabilities and the final model files. This step never runs inference on the
test partition: test predictions are produced only by the final locked-inference step,
after every model, threshold and SGAE parameter has been frozen.

Models
  xgb          XGBoost on the current transaction's features (tabular).
  lstm         LSTM over the card's history: the L most recent transactions strictly
               before t (same card key) + the current transaction, oldest first.
               L = 0 reproduces a tabular (single-row) model, i.e. v1's SEQ_LEN = 1.
  transformer  Transformer encoder over the same transaction sequence (each token is one
               transaction), read out at the current transaction. This is a sequential
               model of transaction histories, NOT a TabTransformer over fields.

Selection protocol (declared in advance)
  - Selection metric: validation PR-AUC (average precision). Test is never used.
  - XGBoost grid: max_depth {6, 8, 10} x colsample_bytree {0.5, 0.8}; lr 0.05, up to
    3000 rounds with early stopping (100) on validation aucpr. Then seeds {42,43,44}
    at the selected config.
  - LSTM / Transformer: history length L in {0, 8, 16} selected with seed 42; then
    seeds {43, 44} at the selected L. Early stopping on validation PR-AUC (patience 4,
    max 30 epochs). Primary seed for downstream steps: 42 (declared, not selected).
  - Imbalance: NN models use BCE with pos_weight = n_neg/n_pos on TRAIN; XGBoost uses
    scale_pos_weight = 1. No SMOTE (not applicable to sequences). Probabilities are
    calibrated later on VALIDATION (step 5).

History construction and inference policy (forward-time deployment simulation)
  History rows are earlier transactions of the same card key (step 1 proxy) with
  TransactionDT strictly less than the current one; same-timestamp rows are excluded.
  History uses only features, never labels. A validation or test transaction may use the
  unlabeled features of any earlier transaction of the same card, including earlier
  validation/test transactions, as it would in live scoring. Gap rows (only if
  --gap_days > 0 in step 1) are not in the feature arrays and so are not used as history.
  Every row's history is verified EXHAUSTIVELY (all rows, all positions) against an
  independent pandas ranking before training. Each history token also carries
  log1p(t_current - t_history) scaled by log1p(183 days).

Usage
  python step02_train_models.py --data_dir /workspace/sgae_v2/data --out_dir /workspace/sgae_v2/step02
  (optional) --models xgb,lstm,transformer   --quick  (tiny smoke test)
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

DAY = 86_400
BIG = 10**9
SEEDS = [42, 43, 44]
L_GRID = [0, 8, 16]
DT_SCALE = np.log1p(183 * DAY)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load(data_dir):
    meta = pd.read_csv(os.path.join(data_dir, "meta.csv.gz"))
    meta = meta[meta["split"] != "gap"].reset_index(drop=True)
    parts = {s: np.load(os.path.join(data_dir, f"X_{s}.npy")) for s in ["train", "val", "test"]}
    ys = {s: np.load(os.path.join(data_dir, f"y_{s}.npy")) for s in ["train", "val", "test"]}
    X = np.concatenate([parts["train"], parts["val"], parts["test"]]).astype(np.float32)
    y = np.concatenate([ys["train"], ys["val"], ys["test"]]).astype(np.float32)
    # step 1 writes partitions in time order; confirm alignment with meta
    assert len(X) == len(meta), "X rows do not match meta rows"
    assert (y == meta["isFraud"].to_numpy()).all(), "labels misaligned with meta"
    assert (meta["split"].to_numpy() == np.repeat(["train", "val", "test"],
            [len(parts["train"]), len(parts["val"]), len(parts["test"])])).all()
    idx = {s: np.where(meta["split"].to_numpy() == s)[0] for s in ["train", "val", "test"]}
    # test labels are blanked in memory: nothing in this step can read them
    y[idx["test"]] = np.nan
    meta.loc[idx["test"], "isFraud"] = np.nan
    return meta, X, y, idx


def build_sequences(meta, L):
    """Return seq_idx (n, L+1) of global row indices, oldest first, current transaction
    at position len-1, padded with -1 on the right; and lengths (n,)."""
    n = len(meta)
    key = meta["key_id"].to_numpy(np.int64)
    t = meta["TransactionDT"].to_numpy(np.int64)
    assert key.min() >= 0 and t.min() >= 0 and t.max() + 183 * DAY < BIG, \
        "composite key*BIG+time would not be collision-free"
    order = np.lexsort((meta["TransactionID"].to_numpy(), t, key))
    T = key[order] * BIG + t[order]
    assert (np.diff(T) >= 0).all()
    right = np.searchsorted(T, T, side="left")                     # strictly earlier
    gs = np.searchsorted(T, key[order] * BIG, side="left")
    pos = np.empty(n, dtype=np.int64); pos[order] = np.arange(n)   # global -> sorted pos
    r, g = right[pos], gs[pos]
    if L == 0:
        return np.arange(n, dtype=np.int64)[:, None], np.ones(n, dtype=np.int64)
    offs = np.arange(L, 0, -1)                                     # L..1  (oldest first)
    hp = r[:, None] - offs[None, :]                                # sorted positions
    valid = hp >= g[:, None]
    hist = np.where(valid, order[np.clip(hp, 0, n - 1)], -1)      # global indices
    c = valid.sum(1)                                               # history count
    full = np.concatenate([hist, np.arange(n)[:, None]], axis=1)   # (n, L+1), invalid first
    col = np.arange(L + 1)[None, :] + (L - c)[:, None]             # left-shift valid block
    seq = np.where(col <= L, np.take_along_axis(full, np.clip(col, 0, L), axis=1), -1)
    return seq.astype(np.int64), (c + 1).astype(np.int64)


def verify_sequences(meta, seq, lens):
    """EXHAUSTIVE check over every row and position, using an independent pandas ranking
    (not the searchsorted logic above). For row i with n_prior(i) = number of same-card
    rows with TransactionDT strictly < t_i, the history must be exactly the
    min(L, n_prior) most recent of those rows, oldest first, followed by i, then -1s.
    Returns the number of rows violating any condition."""
    n, S = seq.shape
    L = S - 1
    d = meta[["key_id", "TransactionDT", "TransactionID"]].copy()
    d["i"] = np.arange(n)
    d = d.sort_values(["key_id", "TransactionDT", "TransactionID"], kind="mergesort")
    d["rank"] = d.groupby("key_id").cumcount()                       # 0-based within card
    d["n_prior"] = d.groupby(["key_id", "TransactionDT"])["rank"].transform("min")
    d = d.sort_values("i")
    rank, n_prior = d["rank"].to_numpy(), d["n_prior"].to_numpy()
    key = meta["key_id"].to_numpy()
    rows = np.arange(n)
    exp_len = np.minimum(L, n_prior) + 1
    ok = lens == exp_len
    ok &= seq[rows, np.clip(lens - 1, 0, L)] == rows                 # current is last
    for j in range(S):
        is_hist = j < lens - 1
        is_pad = j >= lens
        v = seq[:, j]
        vh = np.clip(v, 0, n - 1)
        exp_rank = n_prior - (lens - 1) + j                           # most recent, in order
        ok &= ~is_hist | ((v >= 0) & (key[vh] == key) & (rank[vh] == exp_rank))
        ok &= ~is_pad | (v == -1)
    return int((~ok).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, d_in, hidden=128, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x, lens):
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1]).squeeze(-1)              # h = state after current transaction


class SeqTransformer(nn.Module):
    def __init__(self, d_in, max_len, d_model=128, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x, lens):
        B, S, _ = x.shape
        pad = torch.arange(S, device=x.device)[None, :] >= lens[:, None]
        h = self.enc(self.proj(x) + self.pos[:, :S], src_key_padding_mask=pad)
        cur = h[torch.arange(B, device=x.device), lens - 1]  # current transaction token
        return self.head(cur).squeeze(-1)


def make_batch(Xg, tg, seq, lens, rows):
    """Gather sequences for a batch of global row ids. Adds a scaled time-gap channel."""
    s = seq[rows]                                        # (B, S)
    m = s >= 0
    si = s.clamp(min=0)
    x = Xg[si]                                           # (B, S, F)
    cur_t = tg[rows][:, None]
    gap = torch.log1p((cur_t - tg[si]).clamp(min=0).float()) / DT_SCALE
    x = torch.cat([x, gap.unsqueeze(-1)], dim=-1) * m.unsqueeze(-1)
    return x, lens[rows]


@torch.no_grad()
def predict_nn(model, Xg, tg, seq, lens, rows, bs):
    model.eval(); out = []
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        x, l = make_batch(Xg, tg, seq, lens, r)
        out.append(torch.sigmoid(model(x, l)).float().cpu())
    return torch.cat(out).numpy()


def train_nn(kind, X, y, idx, meta, L, seed, dev, a):
    torch.manual_seed(seed); np.random.seed(seed)
    seq_np, lens_np = build_sequences(meta, L)
    Xg = torch.from_numpy(X).to(dev)
    tg = torch.from_numpy(meta["TransactionDT"].to_numpy(np.int64).copy()).to(dev)
    seq = torch.from_numpy(seq_np).to(dev); lens = torch.from_numpy(lens_np).to(dev)
    yg = torch.from_numpy(y).to(dev)
    d_in = X.shape[1] + 1
    model = (LSTMModel(d_in) if kind == "lstm" else SeqTransformer(d_in, L + 1)).to(dev)
    tr = torch.from_numpy(idx["train"]).to(dev); va = torch.from_numpy(idx["val"]).to(dev)
    npos = float(y[idx["train"]].sum()); pw = (len(idx["train"]) - npos) / npos
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad, hist = -1.0, None, 0, []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(a.max_epochs):
        model.train(); t0 = time.time()
        perm = tr[torch.randperm(len(tr), generator=g).to(dev)]
        for i in range(0, len(perm), a.batch):
            r = perm[i:i + a.batch]
            x, l = make_batch(Xg, tg, seq, lens, r)
            loss = lossf(model(x, l), yg[r])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        pv = predict_nn(model, Xg, tg, seq, lens, va, a.batch * 4)
        ap = average_precision_score(y[idx["val"]], pv)
        hist.append({"epoch": ep + 1, "val_pr_auc": ap, "secs": round(time.time() - t0, 1)})
        print(f"  {kind} L={L} seed={seed} ep{ep+1}: val PR-AUC {ap:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ap > best + 1e-4:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(best_state)
    p = {"val": predict_nn(model, Xg, tg, seq, lens, va, a.batch * 4)}   # validation ONLY
    if dev == "cuda":
        hist.append({"peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        torch.cuda.reset_peak_memory_stats()
    del Xg, tg, seq, lens, yg
    return model, p, hist


def train_xgb(X, y, idx, params, seed, a):
    import xgboost as xgb
    dtr = xgb.DMatrix(X[idx["train"]], label=y[idx["train"]])
    dva = xgb.DMatrix(X[idx["val"]], label=y[idx["val"]])
    p = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
         "device": "cuda" if torch.cuda.is_available() else "cpu", "learning_rate": 0.05,
         "subsample": 0.8, "min_child_weight": 1, "scale_pos_weight": 1.0, "seed": seed, **params}
    bst = xgb.train(p, dtr, num_boost_round=a.xgb_rounds, evals=[(dva, "val")],
                    early_stopping_rounds=100, verbose_eval=False)
    # validation ONLY; saved model keeps only the early-stopped trees
    bst = bst[: bst.best_iteration + 1]
    out = {"val": bst.predict(dva)}
    return bst, out, int(bst.num_boosted_rounds())


# ─────────────────────────────────────────────────────────────────────────────
def val_metrics(y_val, p):
    return {"val_pr_auc": float(average_precision_score(y_val, p)),
            "val_roc_auc": float(roc_auc_score(y_val, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default="xgb,lstm,transformer")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--xgb_rounds", type=int, default=3000)
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    a = ap.parse_args()
    if a.quick:
        a.max_epochs, a.patience, a.xgb_rounds, a.batch = 2, 1, 50, 512
    for d in ["preds", "models"]:
        os.makedirs(os.path.join(a.out_dir, d), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta, X, y, idx = load(a.data_dir)
    print(f"Device {dev} | X {X.shape} | train/val/test {[len(idx[s]) for s in ['train','val','test']]}")
    log_path = os.path.join(a.out_dir, "step02_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    log["protocol"] = {"selection_metric": "validation PR-AUC", "seeds": SEEDS, "primary_seed": 42,
                       "L_grid": L_GRID, "xgb_grid": {"max_depth": [6, 8, 10], "colsample_bytree": [0.5, 0.8]},
                       "test_inference_in_this_step": False, "quick": a.quick,
                       "note": "seed spread = training variability after selection, not a CI"}

    # sequence sanity check (brute force) for every L used
    log["sequence_checks"] = {}
    for L in L_GRID:
        s, l = build_sequences(meta, L)
        v = verify_sequences(meta, s, l)
        log["sequence_checks"][str(L)] = {"violations": int(v),
                                         "mean_history_len": float((l - 1).mean()),
                                         "frac_first_tx": float((l == 1).mean())}
        print(f"Sequence check L={L}: {v} violations | mean history {float((l-1).mean()):.2f}")
        assert v == 0, "sequence construction failed brute-force check"
        np.save(os.path.join(a.out_dir, f"seq_idx_L{L}.npy"), s)
        np.save(os.path.join(a.out_dir, f"seq_len_L{L}.npy"), l)

    def save_preds(name, p):
        assert set(p) == {"val"}, "step 2 must not produce test predictions"
        np.save(os.path.join(a.out_dir, "preds", f"{name}_val.npy"), p["val"].astype(np.float32))

    models = a.models.split(",")
    if "xgb" in models:
        import xgboost as xgb
        grid = [{"max_depth": d, "colsample_bytree": c} for d in [6, 8, 10] for c in [0.5, 0.8]]
        res = []
        for g in grid:
            bst, p, nit = train_xgb(X, y, idx, g, 42, a)
            m = val_metrics(y[idx["val"]], p["val"]); m.update(g); m["n_rounds"] = nit; res.append(m)
            print(f"  xgb {g} -> val PR-AUC {m['val_pr_auc']:.4f} ({nit} rounds)", flush=True)
        best = max(res, key=lambda r: r["val_pr_auc"])
        cfg = {"max_depth": best["max_depth"], "colsample_bytree": best["colsample_bytree"]}
        log["xgb"] = {"grid": res, "selected": cfg, "seeds": {}}
        for sd in SEEDS:
            bst, p, nit = train_xgb(X, y, idx, cfg, sd, a)
            save_preds(f"xgb_seed{sd}", p)
            bst.save_model(os.path.join(a.out_dir, "models", f"xgb_seed{sd}.json"))
            log["xgb"]["seeds"][sd] = {**val_metrics(y[idx["val"]], p["val"]), "n_rounds": nit}
        json.dump(log, open(log_path, "w"), indent=2, default=float)

    for kind in [m for m in models if m in ("lstm", "transformer")]:
        log[kind] = {"L_search": {}, "seeds": {}}
        best_L, best_ap = None, -1
        for L in L_GRID:
            model, p, hist = train_nn(kind, X, y, idx, meta, L, 42, dev, a)
            m = val_metrics(y[idx["val"]], p["val"]); m["epochs"] = hist
            log[kind]["L_search"][str(L)] = m
            save_preds(f"{kind}_L{L}_seed42", p)
            torch.save(model.state_dict(), os.path.join(a.out_dir, "models", f"{kind}_L{L}_seed42.pt"))
            if m["val_pr_auc"] > best_ap: best_L, best_ap = L, m["val_pr_auc"]
            json.dump(log, open(log_path, "w"), indent=2, default=float)
        log[kind]["selected_L"] = best_L
        log[kind]["seeds"]["42"] = {k: v for k, v in log[kind]["L_search"][str(best_L)].items() if k != "epochs"}
        for sd in SEEDS[1:]:
            model, p, hist = train_nn(kind, X, y, idx, meta, best_L, sd, dev, a)
            save_preds(f"{kind}_L{best_L}_seed{sd}", p)
            torch.save(model.state_dict(), os.path.join(a.out_dir, "models", f"{kind}_L{best_L}_seed{sd}.pt"))
            log[kind]["seeds"][str(sd)] = {**val_metrics(y[idx["val"]], p["val"]), "epochs": hist}
            json.dump(log, open(log_path, "w"), indent=2, default=float)
        print(f"{kind}: selected L={best_L} (val PR-AUC {best_ap:.4f})")

    # per-row ledger of validation IDs so every prediction file is traceable
    for s in ["val"]:
        meta.iloc[idx[s]][["TransactionID", "TransactionDT", "key_id", "isFraud"]].to_csv(
            os.path.join(a.out_dir, "preds", f"rows_{s}.csv.gz"), index=False)
    json.dump(log, open(log_path, "w"), indent=2, default=float)
    print("\nVALIDATION summary (no test inference in this step; ± = seed spread, not a CI):")
    for k in ["xgb", "lstm", "transformer"]:
        if k in log:
            v = [s["val_pr_auc"] for s in log[k]["seeds"].values()]
            print(f"  {k:12s} val PR-AUC {np.mean(v):.4f} ± {np.std(v, ddof=1) if len(v)>1 else 0:.4f}"
                  + (f"  (L={log[k]['selected_L']})" if "selected_L" in log[k] else ""))


if __name__ == "__main__":
    main()
