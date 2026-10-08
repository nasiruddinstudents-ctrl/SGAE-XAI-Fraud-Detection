#!/usr/bin/env python3
"""
SGAE v2 — STEP 3: historical (time-respecting) GraphSAGE.

Replaces gnn_graphsage_pipeline.py (v1). v1 problems fixed:
  - v1 built ONE undirected graph over all 590k rows before splitting, so a transaction
    received messages from FUTURE transactions, including test ones.
  - v1 linked only card1 groups of size 2-50 (first 10 rows) and addr1+ProductCD groups
    of size <=30 (first 5 rows); large groups got no edges at all.
  - v1 used BatchNorm over all nodes (feature statistics of val/test nodes entered
    training) and a random stratified split.

Graph (directed, past -> present only)
  For each transaction i, incoming edges come from:
    (a) the K_CARD = 10 most recent transactions of the same card key (step 1 proxy),
    (b) the K_ADDR = 5 most recent transactions with the same addr1 x ProductCD value
        (a billing-region x product proxy; NOT a merchant ID; rows with missing addr1
        get no (b) edges),
  where "earlier" means TransactionDT strictly less than i's. Duplicate (src, dst) pairs
  are merged. No self-loops (the SAGE root weight carries the node's own features).
  Because every edge points forward in time, a node's output depends only on its own
  features and those of earlier transactions -- never on later ones and never on labels.

Protocol (declared in advance; mirrors step 2)
  - Train graph = train nodes only. Validation inference uses train+val nodes.
    Test nodes are never loaded into any graph in this step.
  - Full-batch training on the train graph, BCE with pos_weight = n_neg/n_pos (train),
    Adam lr 1e-3, wd 1e-4, eval on validation PR-AUC every 10 steps, early stopping
    patience 30 evaluations, max 3000 steps. Seeds 42 (primary), 43, 44.
  - Architecture kept as in v1 (SAGE 445->128->64, MLP head 64->256->128->64->1,
    dropout 0.25) except LayerNorm replaces BatchNorm (per-node, so no statistics are
    shared across nodes or partitions).
  - Ablation: identical model with all edges removed (a per-transaction MLP), seed 42,
    to measure what the graph contributes.

Audits written to step03_log.json
  - Exhaustive edge check (every edge, every node) against an independent pandas ranking.
  - Causality check: train-node outputs computed on the train-only graph must equal those
    computed after adding all validation nodes (eval mode).
  - Edge counts by (source partition -> destination partition).

Usage
  python step03_gnn.py --data_dir /workspace/sgae_v2/data --raw_dir /workspace/ieee-cis \
                       --out_dir /workspace/sgae_v2/step03
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

BIG = 10**9
K_CARD, K_ADDR = 10, 5
SEEDS = [42, 43, 44]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load(data_dir, raw_dir):
    meta = pd.read_csv(os.path.join(data_dir, "meta.csv.gz"))
    meta = meta[meta["split"] != "gap"].reset_index(drop=True)
    Xs = {s: np.load(os.path.join(data_dir, f"X_{s}.npy")) for s in ["train", "val"]}
    ys = {s: np.load(os.path.join(data_dir, f"y_{s}.npy")) for s in ["train", "val"]}
    n_tr, n_va = len(Xs["train"]), len(Xs["val"])
    assert (meta["split"].to_numpy()[:n_tr] == "train").all()
    assert (meta["split"].to_numpy()[n_tr:n_tr + n_va] == "val").all()
    meta = meta.iloc[:n_tr + n_va].copy()          # test rows are not loaded at all
    X = np.concatenate([Xs["train"], Xs["val"]]).astype(np.float32)
    y = np.concatenate([ys["train"], ys["val"]]).astype(np.float32)
    assert (y == meta["isFraud"].to_numpy()).all()
    raw = pd.read_csv(os.path.join(raw_dir, "train_transaction.csv"),
                      usecols=["TransactionID", "addr1", "ProductCD"])
    meta = meta.merge(raw, on="TransactionID", how="left", validate="one_to_one")
    assert len(meta) == n_tr + n_va
    return meta, X, y, n_tr, n_va


def addr_key(meta):
    k = np.full(len(meta), -1, dtype=np.int64)
    ok = meta["addr1"].notna().to_numpy()
    s = meta["addr1"].astype("Int64").astype(str) + "|" + meta["ProductCD"].fillna("NA").astype(str)
    k[ok] = pd.factorize(s[ok], sort=True)[0]
    return k


def recent_prior_edges(key, t, tid, K):
    """Edges j -> i for the K most recent same-key rows with t_j < t_i. key < 0 = none."""
    n = len(key)
    assert t.min() >= 0 and t.max() < BIG
    kk = np.where(key >= 0, key, key.max() + 1).astype(np.int64)   # missing = own group, never a dst
    order = np.lexsort((tid, t, kk))
    T = kk[order] * BIG + t[order]
    right = np.searchsorted(T, T, side="left")
    gs = np.searchsorted(T, kk[order] * BIG, side="left")
    pos = np.empty(n, dtype=np.int64); pos[order] = np.arange(n)
    r, g = right[pos], gs[pos]
    hp = r[:, None] - np.arange(1, K + 1)[None, :]
    valid = (hp >= g[:, None]) & (key[:, None] >= 0)
    dst = np.repeat(np.arange(n), K).reshape(n, K)[valid]
    src = order[np.clip(hp, 0, n - 1)][valid]
    return src, dst


def verify_edges(key, t, tid, K, src, dst):
    """Exhaustive, independent check: for every node i, its in-edges of this type are
    exactly the min(K, n_prior(i)) most recent same-key rows strictly before t_i."""
    d = pd.DataFrame({"key": key, "t": t, "tid": tid, "i": np.arange(len(key))})
    d = d.sort_values(["key", "t", "tid"], kind="mergesort")
    d["rank"] = d.groupby("key").cumcount()
    d["n_prior"] = d.groupby(["key", "t"])["rank"].transform("min")
    d = d.sort_values("i")
    rank, n_prior = d["rank"].to_numpy(), d["n_prior"].to_numpy()
    n_prior = np.where(key >= 0, n_prior, 0)
    bad = 0
    bad += int((key[src] != key[dst]).sum())
    bad += int((t[src] >= t[dst]).sum())
    # each src must be among the K most recent prior rows of dst
    bad += int(((rank[src] < n_prior[dst] - K) | (rank[src] >= n_prior[dst])).sum())
    cnt = np.bincount(dst, minlength=len(key))
    bad += int((cnt != np.minimum(K, n_prior)).sum())
    return bad


def build_graph(meta):
    t = meta["TransactionDT"].to_numpy(np.int64)
    tid = meta["TransactionID"].to_numpy(np.int64)
    kc = meta["key_id"].to_numpy(np.int64)
    ka = addr_key(meta)
    s1, d1 = recent_prior_edges(kc, t, tid, K_CARD)
    s2, d2 = recent_prior_edges(ka, t, tid, K_ADDR)
    checks = {"card_edges_violations": verify_edges(kc, t, tid, K_CARD, s1, d1),
              "addr_product_edges_violations": verify_edges(ka, t, tid, K_ADDR, s2, d2),
              "n_card_edges": int(len(s1)), "n_addr_product_edges": int(len(s2))}
    e = np.unique(np.stack([np.concatenate([s1, s2]), np.concatenate([d1, d2])]), axis=1)
    checks["n_edges_after_dedup"] = int(e.shape[1])
    checks["all_edges_point_forward_in_time"] = bool((t[e[0]] < t[e[1]]).all())
    checks["all_src_index_lt_dst_index"] = bool((e[0] < e[1]).all())   # rows are time-sorted
    return e, checks


def adjacency(e, m, dev):
    """Row-normalised (mean) adjacency over the first m nodes: A[i, j] = 1/indeg(i)."""
    keep = e[1] < m                       # src < dst, so src < m too
    src, dst = e[0][keep], e[1][keep]
    deg = np.bincount(dst, minlength=m).astype(np.float32)
    val = 1.0 / deg[dst]
    A = torch.sparse_coo_tensor(torch.from_numpy(np.stack([dst, src])),
                                torch.from_numpy(val), (m, m)).coalesce().to(dev)
    return A, int(keep.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class SAGE(nn.Module):
    """h_i' = W_self h_i + W_nb mean_{j -> i} h_j   (same as PyG SAGEConv, mean aggr)."""
    def __init__(self, din, dout):
        super().__init__()
        self.self_lin = nn.Linear(din, dout)
        self.nb_lin = nn.Linear(din, dout, bias=False)
    def forward(self, h, A):
        out = self.self_lin(h)
        if A is not None:
            out = out + torch.sparse.mm(A, self.nb_lin(h))
        return out


class GNN(nn.Module):
    def __init__(self, din, hidden=128, emb=64, drop=0.25):
        super().__init__()
        self.c1, self.n1 = SAGE(din, hidden), nn.LayerNorm(hidden)
        self.c2, self.n2 = SAGE(hidden, emb), nn.LayerNorm(emb)
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(emb, 256), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(256, 128), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(64, 1))
    def embed(self, x, A):
        h = self.drop(F.relu(self.n1(self.c1(x, A))))
        return self.drop(F.relu(self.n2(self.c2(h, A))))
    def forward(self, x, A):
        return self.head(self.embed(x, A)).squeeze(-1)


def run(X, y, n_tr, n_va, A_tr, A_all, seed, dev, a, use_edges=True, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    Xg = torch.from_numpy(X).to(dev)
    ytr = torch.from_numpy(y[:n_tr]).to(dev)
    yva = y[n_tr:n_tr + n_va]
    model = GNN(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = float((ytr == 0).sum() / ytr.sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
    At = A_tr if use_edges else None
    Aa = A_all if use_edges else None
    best, best_state, bad, hist, t0 = -1.0, None, 0, [], time.time()
    for step in range(1, a.max_steps + 1):
        model.train()
        loss = lossf(model(Xg[:n_tr], At), ytr)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % a.eval_every == 0:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xg, Aa)[n_tr:]).cpu().numpy()
            ap = average_precision_score(yva, pv)
            hist.append({"step": step, "loss": float(loss.detach()), "val_pr_auc": float(ap)})
            if ap > best + 1e-4:
                best, bad = ap, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if step % (a.eval_every * 10) == 0:
                print(f"  {tag} seed={seed} step {step}: loss {float(loss.detach()):.4f} | val PR-AUC {ap:.4f} "
                      f"(best {best:.4f}, {time.time()-t0:.0f}s)", flush=True)
            if bad >= a.patience: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xg, Aa)[n_tr:]).cpu().numpy()
    info = {"val_pr_auc": float(average_precision_score(yva, pv)),
            "val_roc_auc": float(roc_auc_score(yva, pv)),
            "best_step": hist[int(np.argmax([h["val_pr_auc"] for h in hist]))]["step"],
            "secs": round(time.time() - t0, 1), "curve": hist}
    if dev == "cuda":
        info["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        torch.cuda.reset_peak_memory_stats()
    return model, pv, info


def causality_check(model, X, n_tr, A_tr, A_all, dev):
    """Train-node outputs must not change when all validation nodes are added."""
    model.eval()
    Xg = torch.from_numpy(X).to(dev)
    with torch.no_grad():
        a = model(Xg[:n_tr], A_tr).cpu().numpy()
        b = model(Xg, A_all)[:n_tr].cpu().numpy()
    d = float(np.abs(a - b).max())
    return {"max_abs_logit_difference": d, "passed": d < 1e-3}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.max_steps, a.eval_every, a.patience = 60, 5, 4
    for d in ["preds", "models"]:
        os.makedirs(os.path.join(a.out_dir, d), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta, X, y, n_tr, n_va = load(a.data_dir, a.raw_dir)
    print(f"Device {dev} | nodes train {n_tr:,} + val {n_va:,} (test not loaded) | features {X.shape[1]}")
    e, checks = build_graph(meta)
    part = np.where(np.arange(len(meta)) < n_tr, "train", "val")
    checks["edges_by_partition"] = pd.Series(
        [f"{p}->{q}" for p, q in zip(part[e[0]], part[e[1]])]).value_counts().to_dict()
    indeg = np.bincount(e[1], minlength=len(meta))
    checks["frac_nodes_without_in_edges"] = {"train": float((indeg[:n_tr] == 0).mean()),
                                             "val": float((indeg[n_tr:] == 0).mean())}
    print(json.dumps(checks, indent=1))
    assert checks["card_edges_violations"] == 0 and checks["addr_product_edges_violations"] == 0
    assert checks["all_edges_point_forward_in_time"] and checks["all_src_index_lt_dst_index"]

    A_tr, ne_tr = adjacency(e, n_tr, dev)
    A_all, ne_all = adjacency(e, n_tr + n_va, dev)
    checks["edges_in_train_graph"], checks["edges_in_train_val_graph"] = ne_tr, ne_all

    log = {"protocol": {"K_card": K_CARD, "K_addr_product": K_ADDR, "seeds": SEEDS, "primary_seed": 42,
                        "selection_metric": "validation PR-AUC", "test_nodes_loaded": False,
                        "norm": "LayerNorm", "quick": a.quick,
                        "note": "seed spread = training variability, not a CI"},
           "graph_checks": checks, "runs": {}}
    save = lambda: json.dump(log, open(os.path.join(a.out_dir, "step03_log.json"), "w"), indent=2, default=float)

    for sd in SEEDS:
        model, pv, info = run(X, y, n_tr, n_va, A_tr, A_all, sd, dev, a, True, "gnn")
        if sd == 42:
            log["causality_check"] = causality_check(model, X, n_tr, A_tr, A_all, dev)
            print(f"Causality check: {log['causality_check']}")
            assert log["causality_check"]["passed"], "train outputs changed when val nodes were added"
        np.save(os.path.join(a.out_dir, "preds", f"gnn_seed{sd}_val.npy"), pv.astype(np.float32))
        torch.save(model.state_dict(), os.path.join(a.out_dir, "models", f"gnn_seed{sd}.pt"))
        log["runs"][f"gnn_seed{sd}"] = info; save()
        print(f"gnn seed {sd}: val PR-AUC {info['val_pr_auc']:.4f} | ROC-AUC {info['val_roc_auc']:.4f}")

    model, pv, info = run(X, y, n_tr, n_va, A_tr, A_all, 42, dev, a, False, "no-edges")
    np.save(os.path.join(a.out_dir, "preds", "gnn_noedges_seed42_val.npy"), pv.astype(np.float32))
    torch.save(model.state_dict(), os.path.join(a.out_dir, "models", "gnn_noedges_seed42.pt"))
    log["runs"]["gnn_noedges_seed42"] = info
    np.save(os.path.join(a.out_dir, "edges_train_val.npy"), e)
    save()

    v = [log["runs"][f"gnn_seed{s}"]["val_pr_auc"] for s in SEEDS]
    print(f"\nVALIDATION summary (no test nodes loaded; ± = seed spread, not a CI):")
    print(f"  GNN (historical graph)  val PR-AUC {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}")
    print(f"  Same model, no edges    val PR-AUC {info['val_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
