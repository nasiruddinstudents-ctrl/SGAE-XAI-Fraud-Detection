#!/usr/bin/env python3
"""
SGAE v2 — STEP 5: LOCKED TEST EVALUATION.

Phase A (no file containing test labels is read: metadata is read without the isFraud
column, the raw-transaction read is restricted to TransactionID/addr1/ProductCD, and
only y_train / y_val are loaded)
  1. Verify the step-4 config fingerprint (sha256) — no parameter may change.
  2. Train the two remaining no-edge GNN control seeds (43, 44) on train/validation only,
     so the graph ablation has three seeds like every other model.
  3. Produce test predictions for every frozen model and ensemble:
       XGBoost, LSTM, Transformer (3 seeds each, plus L=0 history-free controls, seed 42),
       GNN and no-edge GNN (3 seeds each), and all step-4 ensembles from frozen parameters.
     Test-time attributions use the frozen settings (train background, EG samples,
     global feature ranking, K). Two independent LSTM runs give test repeatability for
     the preregistered test.
  4. Consistency checks: re-applying the frozen ensemble code to VALIDATION must reproduce
     step 4's stored validation predictions; GNN validation outputs must not change when
     test nodes are added to the graph; TreeSHAP must reproduce XGBoost test probabilities.
  5. Write test_predictions.npz and record its sha256 BEFORE any label is read.

Phase B (labels loaded once)
  - Primary metric PR-AUC; secondary ROC-AUC. Precision, recall, F1 and MCC at the
    F1-optimal threshold chosen on VALIDATION predictions and applied unchanged.
  - Calibration: Brier score and ECE (15 equal-mass bins), raw and after the frozen
    validation Platt maps; reliability-curve data saved.
  - Uncertainty: cluster bootstrap over card keys (default 2,000 resamples), percentile
    95% intervals for PR-AUC, ROC-AUC and (thresholds held fixed at their validation
    values) precision, recall, F1 and MCC. Paired differences for a DECLARED comparison
    family on PR-AUC: percentile interval plus a null-centred bootstrap p-value
    (p = share of |d* - d_obs| >= |d_obs|), Holm-corrected. Seed spread reported
    separately (training variability, not a CI).
  - The mechanism test exactly as prespecified and frozen in the step-4 config before any
    test evaluation in this revision (described as "prespecified within the revision",
    not as an external preregistration).
  - Time-block (5 contiguous blocks) PR-AUC for drift.

Usage
  python step05_locked_test.py --data_dir /workspace/sgae_v2/data --raw_dir /workspace/ieee-cis \
      --step02_dir /workspace/sgae_v2/step02 --step03_dir /workspace/sgae_v2/step03 \
      --step04_dir /workspace/sgae_v2/step04 --out_dir /workspace/sgae_v2/step05
"""
import argparse, json, os, sys, time, hashlib, types
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             matthews_corrcoef, f1_score, precision_score, recall_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step02_train_models as s2
import step03_gnn as s3
import step04_sgae as s4

SEEDS = [42, 43, 44]
FAMILY = [  # declared comparison family (PR-AUC differences, Holm-corrected)
    ("sgae_tuned", "xgb_only"), ("sgae_tuned", "tuned_constant"), ("sgae_tuned", "learned_gate"),
    ("sgae_tuned", "stacking"), ("sgae_tuned", "sgae_v1_weighting_rule"),
    ("gnn_seed42", "gnn_noedges_seed42"), ("gnn_seed42", "xgb_seed42"),
    ("lstm_selL_seed42", "lstm_L0_seed42"), ("transformer_selL_seed42", "transformer_L0_seed42")]


_trapz = getattr(np, "trapezoid", None) or np.trapz
def logit(p): return s4.logit(p)
def sigm(z): return s4.sigm(z)


# ─────────────────────────────────────────────────────────────────────────────
# Fast weighted AP / ROC-AUC for the bootstrap (validated against sklearn below)
# ─────────────────────────────────────────────────────────────────────────────
class Ranker:
    def __init__(self, p, y):
        o = np.argsort(-p, kind="mergesort")
        self.o, self.ys = o, y[o].astype(float)
        ps = p[o]
        self.last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]   # end of each tie group
    def ap(self, w):
        ws = w[self.o]
        tp = np.cumsum(ws * self.ys)[self.last]; fp = np.cumsum(ws * (1 - self.ys))[self.last]
        P = tp[-1]
        if P <= 0: return np.nan
        rec = tp / P; prec = tp / np.maximum(tp + fp, 1e-12)
        return float(np.sum(np.diff(np.r_[0, rec]) * prec))
    def auc(self, w):
        ws = w[self.o]
        tp = np.r_[0, np.cumsum(ws * self.ys)[self.last]]; fp = np.r_[0, np.cumsum(ws * (1 - self.ys))[self.last]]
        if tp[-1] <= 0 or fp[-1] <= 0: return np.nan
        return float(_trapz(tp / tp[-1], fp / fp[-1]))


def ece(y, p, bins=15):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    e, curve = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
            curve.append([float(p[m].mean()), float(y[m].mean()), int(m.sum())])
    return float(e), curve


def f1_threshold(y_val, p_val):
    pr_, rc_, th = precision_recall_curve(y_val, p_val)
    f1 = 2 * pr_[:-1] * rc_[:-1] / np.maximum(pr_[:-1] + rc_[:-1], 1e-12)
    return float(th[int(np.argmax(f1))])


def apply_methods(pl, px, A10, AK, M, base_z):
    """Frozen step-4 methods applied to calibrated base probabilities."""
    zl, zx = logit(pl), logit(px)
    cl, cx = np.abs(zl - base_z), np.abs(zx - base_z)
    s = M["sgae_tuned"]
    return {
        "xgb_only": px, "lstm_only": pl,
        "equal_weight": s4.mix(0.5, pl, px), "static_v1_0.6": s4.mix(0.6, pl, px),
        "tuned_constant": s4.mix(M["tuned_constant"]["w"], pl, px),
        "confidence_gate": s4.mix(cl / (cl + cx + s4.EPS), pl, px),
        "stacking": sigm(M["stacking"]["intercept"] + np.stack([zl, zx], 1) @ np.array(M["stacking"]["coef"])),
        "learned_gate": s4.apply_gate(M["learned_gate"], zl, zx, pl, px),
        "learned_gate_plus_A": s4.apply_gate(M["learned_gate_plus_A"], zl, zx, pl, px, A10),
        "sgae_v1_weighting_rule": s4.mix(s4.sgae_w(A10, M["sgae_v1_weighting_rule"]["sigma_A"], 0.5, 0.2, 0.6, 0.3, 0.7), pl, px),
        "sgae_tuned": s4.mix(s4.sgae_w(AK, s["sigma_A"], s["w0"], s["c"], s["neg"]), pl, px),
    }


def platt(p, prm): return sigm(prm["coef"] * logit(p) + prm["intercept"])


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    for k in ["data_dir", "raw_dir", "step02_dir", "step03_dir", "step04_dir", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.n_boot = 50
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log = {"phase_A_started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ── frozen config ────────────────────────────────────────────────────────
    cfg = json.load(open(os.path.join(a.step04_dir, "step04_config.json")))
    c2 = {k: v for k, v in cfg.items() if k != "sha256"}
    sha = hashlib.sha256(json.dumps(c2, sort_keys=True, default=str).encode()).hexdigest()
    assert sha == cfg["sha256"], "step-4 config was modified after freezing"
    log["config_sha256_verified"] = sha
    M, base = cfg["methods"], cfg["base"]
    # a negative Platt slope would silently reverse a model's ranking inside the ensembles
    log["platt_slopes"] = {m: base["platt"][m]["coef"] for m in base["platt"]}
    if not a.quick:
        assert all(v > 0 for v in log["platt_slopes"].values()), f"non-positive Platt slope: {log['platt_slopes']}"
    seed, L = base["seed"], base["lstm_L"]
    log2 = json.load(open(os.path.join(a.step02_dir, "step02_log.json")))
    L_tf = log2["transformer"]["selected_L"]

    # ── data (NO test labels) ────────────────────────────────────────────────
    meta = pd.read_csv(os.path.join(a.data_dir, "meta.csv.gz"),
                       usecols=["TransactionID", "TransactionDT", "split", "key_id"])   # no isFraud
    assert "isFraud" not in meta.columns
    meta = meta[meta["split"] != "gap"].reset_index(drop=True)
    parts = {s: np.load(os.path.join(a.data_dir, f"X_{s}.npy")) for s in ["train", "val", "test"]}
    n_tr, n_va, n_te = (len(parts[s]) for s in ["train", "val", "test"])
    X = np.concatenate([parts["train"], parts["val"], parts["test"]]).astype(np.float32)
    ytr, yva = (np.load(os.path.join(a.data_dir, f"y_{s}.npy")).astype(float) for s in ["train", "val"])
    test_ids = meta["TransactionID"].to_numpy()[n_tr + n_va:]
    te = np.arange(n_tr + n_va, n_tr + n_va + n_te); va = np.arange(n_tr, n_tr + n_va)
    print(f"Phase A | train {n_tr:,} val {n_va:,} test {n_te:,} | no test-label file read", flush=True)
    Xg = torch.from_numpy(X).to(dev)
    tg = torch.from_numpy(meta["TransactionDT"].to_numpy(np.int64).copy()).to(dev)
    P_val, P_test = {}, {}

    need = [os.path.join(a.step02_dir, "models", f) for f in
            [f"xgb_seed{s}.json" for s in SEEDS] + [f"lstm_L{L}_seed{s}.pt" for s in SEEDS] +
            [f"transformer_L{L_tf}_seed{s}.pt" for s in SEEDS] + ["lstm_L0_seed42.pt", "transformer_L0_seed42.pt"]]
    need += [os.path.join(a.step03_dir, "models", f) for f in [f"gnn_seed{s}.pt" for s in SEEDS] + ["gnn_noedges_seed42.pt"]]
    missing = [f for f in need if not os.path.exists(f)]
    assert not missing, f"missing model files (partial earlier run?): {missing}"

    # ── no-edge GNN control seeds 43, 44 (train/val only) ────────────────────
    raw = pd.read_csv(os.path.join(a.raw_dir, "train_transaction.csv"), usecols=["TransactionID", "addr1", "ProductCD"])
    meta_g = meta.merge(raw, on="TransactionID", how="left", validate="one_to_one")
    ga = types.SimpleNamespace(max_steps=60 if a.quick else 3000, eval_every=5 if a.quick else 10,
                               patience=4 if a.quick else 30)
    y_trva = np.concatenate([ytr, yva]).astype(np.float32)
    for sd in [43, 44]:
        mdl, pv, info = s3.run(X[:n_tr + n_va], y_trva, n_tr, n_va, None, None, sd, dev, ga, False, "no-edges")
        torch.save(mdl.state_dict(), os.path.join(a.out_dir, f"gnn_noedges_seed{sd}.pt"))
        log.setdefault("noedge_controls", {})[sd] = {k: v for k, v in info.items() if k != "curve"}
        print(f"  no-edge GNN seed {sd}: val PR-AUC {info['val_pr_auc']:.4f}", flush=True)

    # ── single-model test predictions ────────────────────────────────────────
    import xgboost as xgb
    for sd in SEEDS:
        bst = xgb.Booster(); bst.load_model(os.path.join(a.step02_dir, "models", f"xgb_seed{sd}.json"))
        # step 2 saved the booster truncated to its early-stopped trees and produced its
        # validation predictions from that same truncated booster; verify, then reuse
        it = (0, bst.num_boosted_rounds())
        saved = np.load(os.path.join(a.step02_dir, "preds", f"xgb_seed{sd}_val.npy"))
        rv = bst.predict(xgb.DMatrix(X[va]), iteration_range=it)
        d = float(np.abs(rv - saved).max())
        log.setdefault("xgb_reproduces_saved_val", {})[sd] = {"max_abs_diff": d, "n_trees": it[1]}
        assert d < 1e-5, f"xgb seed {sd}: reloaded booster does not reproduce saved validation predictions"
        P_test[f"xgb_seed{sd}"] = bst.predict(xgb.DMatrix(X[te]), iteration_range=it)
        P_val[f"xgb_seed{sd}"] = saved

    def seq_model(kind, LL, sd):
        seq = torch.from_numpy(np.load(os.path.join(a.step02_dir, f"seq_idx_L{LL}.npy"))).to(dev)
        lens = torch.from_numpy(np.load(os.path.join(a.step02_dir, f"seq_len_L{LL}.npy"))).to(dev)
        m = (s2.LSTMModel(X.shape[1] + 1) if kind == "lstm" else s2.SeqTransformer(X.shape[1] + 1, LL + 1)).to(dev)
        m.load_state_dict(torch.load(os.path.join(a.step02_dir, "models", f"{kind}_L{LL}_seed{sd}.pt"), map_location=dev))
        m.eval()
        pv = s2.predict_nn(m, Xg, tg, seq, lens, torch.from_numpy(va).to(dev), 4096)
        saved = np.load(os.path.join(a.step02_dir, "preds", f"{kind}_L{LL}_seed{sd}_val.npy"))
        assert np.abs(pv - saved).max() < 1e-4, f"{kind} L{LL} seed{sd} does not reproduce saved val preds"
        return saved, s2.predict_nn(m, Xg, tg, seq, lens, torch.from_numpy(te).to(dev), 4096)

    for kind, LL in [("lstm", L), ("transformer", L_tf)]:
        for sd in SEEDS:
            P_val[f"{kind}_selL_seed{sd}"], P_test[f"{kind}_selL_seed{sd}"] = seq_model(kind, LL, sd)
        P_val[f"{kind}_L0_seed42"], P_test[f"{kind}_L0_seed42"] = seq_model(kind, 0, 42)
    print(f"  sequence/tabular test predictions done ({time.time()-t0:.0f}s)", flush=True)

    # GNN on the full forward-time graph (test nodes receive edges only from earlier rows)
    e, gchk = s3.build_graph(meta_g)
    assert gchk["card_edges_violations"] == 0 and gchk["addr_product_edges_violations"] == 0
    part = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * n_te)
    gchk["edges_by_partition"] = pd.Series([f"{p}->{q}" for p, q in zip(part[e[0]], part[e[1]])]).value_counts().to_dict()
    pidx = np.array([0] * n_tr + [1] * n_va + [2] * n_te)
    tt = meta_g["TransactionDT"].to_numpy()
    gchk["all_sources_strictly_earlier"] = bool((tt[e[0]] < tt[e[1]]).all())
    gchk["no_edges_from_later_partition"] = bool((pidx[e[0]] <= pidx[e[1]]).all())
    assert gchk["all_sources_strictly_earlier"] and gchk["no_edges_from_later_partition"]
    log["graph_checks_full"] = {k: v for k, v in gchk.items()}
    A_all, _ = s3.adjacency(e, len(meta_g), dev)
    for tag, use_edges in [("gnn", True), ("gnn_noedges", False)]:
        for sd in SEEDS:
            path = (os.path.join(a.step03_dir, "models", f"{tag}_seed{sd}.pt") if (tag == "gnn" or sd == 42)
                    else os.path.join(a.out_dir, f"gnn_noedges_seed{sd}.pt"))
            m = s3.GNN(X.shape[1]).to(dev); m.load_state_dict(torch.load(path, map_location=dev)); m.eval()
            with torch.no_grad():
                pall = torch.sigmoid(m(Xg, A_all if use_edges else None)).cpu().numpy()
            if tag == "gnn" or sd == 42:
                saved = np.load(os.path.join(a.step03_dir, "preds", f"{tag}_seed{sd}_val.npy"))
                d = float(np.abs(pall[va] - saved).max())
                log.setdefault("gnn_val_unchanged_when_test_nodes_added", {})[f"{tag}_seed{sd}"] = d
                assert d < 1e-3, "GNN validation outputs changed when test nodes were added"
                P_val[f"{tag}_seed{sd}"] = saved
            else:
                P_val[f"{tag}_seed{sd}"] = pall[va]
            P_test[f"{tag}_seed{sd}"] = pall[te]
    print(f"  GNN test predictions done ({time.time()-t0:.0f}s)", flush=True)

    # ── ensembles from frozen parameters ─────────────────────────────────────
    pr_te = {"lstm": P_test[f"lstm_selL_seed{seed}"], "xgb": P_test[f"xgb_seed{seed}"]}
    pr_va = {"lstm": P_val[f"lstm_selL_seed{seed}"], "xgb": P_val[f"xgb_seed{seed}"]}
    pa = types.SimpleNamespace(step02_dir=a.step02_dir, seed=seed)
    phi_x_te, xinfo = s4.xgb_contribs(pa, X, te, saved_prob=pr_te["xgb"])
    lstm = s2.LSTMModel(X.shape[1] + 1).to(dev)
    lstm.load_state_dict(torch.load(os.path.join(a.step02_dir, "models", f"lstm_L{L}_seed{seed}.pt"), map_location=dev))
    seqL = np.load(os.path.join(a.step02_dir, f"seq_idx_L{L}.npy")); lenL = np.load(os.path.join(a.step02_dir, f"seq_len_L{L}.npy"))
    bg = np.array(cfg["attribution"]["bg_train_rows"]); Mg = cfg["attribution"]["eg_samples"]
    bg_b = np.sort(np.random.default_rng(seed + 7).choice(n_tr, len(bg), replace=False))
    phi_l_te, gapA = s4.lstm_eg(lstm, Xg, tg, seqL, lenL, te, bg, Mg, 4000 + seed, dev, a.batch)
    phi_l_te_b, _ = s4.lstm_eg(lstm, Xg, tg, seqL, lenL, te, bg_b, Mg, 5000 + seed, dev, a.batch)
    rank = np.array(cfg["attribution"]["global_rank"])
    K = M["sgae_tuned"]["K"]
    A10_te, AK_te = s4.agreement(phi_l_te, phi_x_te, rank, 10), s4.agreement(phi_l_te, phi_x_te, rank, K)
    rel_te = s4.rowwise_spearman(phi_l_te[:, rank[:20]], phi_l_te_b[:, rank[:20]])
    log["test_attribution_checks"] = {"xgb_treeshap": xinfo, "lstm_eg_gap_median": float(np.median(gapA)),
                                      "lstm_eg_gap_p95": float(np.percentile(gapA, 95)),
                                      "eg_seeds_test": [4000 + seed, 5000 + seed]}
    pl_te, px_te = platt(pr_te["lstm"], base["platt"]["lstm"]), platt(pr_te["xgb"], base["platt"]["xgb"])
    ens_te = apply_methods(pl_te, px_te, A10_te, AK_te, M, base["train_base_rate_logit"])

    # consistency: frozen code on VALIDATION must reproduce step-4 stored in-sample predictions
    phi_x_va = np.load(os.path.join(a.step04_dir, "shap_val_xgb.npy"))
    phi_l_va = np.load(os.path.join(a.step04_dir, "shap_val_lstm_A.npy"))
    A10_va, AK_va = s4.agreement(phi_l_va, phi_x_va, rank, 10), s4.agreement(phi_l_va, phi_x_va, rank, K)
    ens_va = apply_methods(platt(pr_va["lstm"], base["platt"]["lstm"]), platt(pr_va["xgb"], base["platt"]["xgb"]),
                           A10_va, AK_va, M, base["train_base_rate_logit"])
    stored = np.load(os.path.join(a.step04_dir, "val_ensemble_preds.npz"))
    diffs = {k: float(np.abs(ens_va[k] - stored[f"in_{k}"]).max()) for k in ens_va}
    log["frozen_ensemble_code_reproduces_step4_val"] = diffs
    assert max(diffs.values()) < 1e-4, f"ensemble code mismatch: {diffs}"
    for k in ens_te:
        P_test[k], P_val[k] = ens_te[k], ens_va[k]

    # ── seal predictions ─────────────────────────────────────────────────────
    pred_path = os.path.join(a.out_dir, "test_predictions.npz")
    np.savez_compressed(pred_path, TransactionID=test_ids, repeatability=rel_te.astype(np.float32),
                        agreement_K=AK_te.astype(np.float32), **{k: np.asarray(v, np.float32) for k, v in P_test.items()})
    log["test_predictions_sha256"] = hashlib.sha256(open(pred_path, "rb").read()).hexdigest()
    log["phase_A_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(log, open(os.path.join(a.out_dir, "step05_log.json"), "w"), indent=2, default=str)
    print(f"Phase A sealed: test_predictions.npz sha256 {log['test_predictions_sha256'][:16]} ({time.time()-t0:.0f}s)", flush=True)

    # ═════════════════════ PHASE B: labels read once ═════════════════════════
    y = np.load(os.path.join(a.data_dir, "y_test.npy")).astype(float)
    assert len(y) == n_te
    log["phase_B_labels_loaded"] = time.strftime("%Y-%m-%d %H:%M:%S")
    keys = meta["key_id"].to_numpy()[te]
    kcode = pd.factorize(keys)[0]; nk = kcode.max() + 1
    names = list(P_test)

    # validate fast metrics against sklearn
    rk = Ranker(P_test["sgae_tuned"], y); wtest = np.random.default_rng(0).integers(0, 3, n_te).astype(float)
    log["fast_metric_check"] = {
        "ap_unweighted": abs(rk.ap(np.ones(n_te)) - average_precision_score(y, P_test["sgae_tuned"])),
        "ap_weighted": abs(rk.ap(wtest) - average_precision_score(y, P_test["sgae_tuned"], sample_weight=wtest)),
        "auc_weighted": abs(rk.auc(wtest) - roc_auc_score(y, P_test["sgae_tuned"], sample_weight=wtest))}
    assert max(log["fast_metric_check"].values()) < 1e-8, log["fast_metric_check"]

    table = {}
    for k in names:
        p, pv = np.asarray(P_test[k], float), np.asarray(P_val[k], float)
        th = f1_threshold(yva, pv); yh = (p >= th).astype(int)
        e_raw, curve = ece(y, p)
        prm = s4.Platt().fit(pv, yva); pc = prm(p)
        e_cal, curve_cal = ece(y, pc)
        table[k] = {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
                    "val_threshold": th, "precision": float(precision_score(y, yh, zero_division=0)),
                    "recall": float(recall_score(y, yh)), "f1": float(f1_score(y, yh)),
                    "mcc": float(matthews_corrcoef(y, yh)), "brier_raw": float(np.mean((p - y) ** 2)),
                    "ece_raw": e_raw, "brier_val_platt": float(np.mean((pc - y) ** 2)), "ece_val_platt": e_cal,
                    "reliability_raw": curve, "reliability_val_platt": curve_cal}

    # cluster bootstrap over card keys
    rng = np.random.default_rng(12345)
    rankers = {k: Ranker(np.asarray(P_test[k], float), y) for k in names}
    boot_pr = {k: np.empty(a.n_boot) for k in names}; boot_roc = {k: np.empty(a.n_boot) for k in names}
    yh = {k: (np.asarray(P_test[k], float) >= table[k]["val_threshold"]).astype(float) for k in names}
    boot_thr = {k: np.empty((a.n_boot, 4)) for k in names}          # precision, recall, F1, MCC

    def thr_metrics(w, yy, hh):
        tp = np.sum(w * yy * hh); fp = np.sum(w * (1 - yy) * hh)
        fn = np.sum(w * yy * (1 - hh)); tn = np.sum(w * (1 - yy) * (1 - hh))
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else np.nan
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return prec, rec, f1, ((tp * tn - fp * fn) / den if den > 0 else 0.0)
    cuts = cfg["preregistered_mechanism_test"]["tertile_cutoffs_from_validation"]
    grp = np.digitize(rel_te, cuts)
    mech_r = {g: {m: Ranker(np.asarray(P_test[m], float)[grp == g], y[grp == g]) for m in ["sgae_tuned", "tuned_constant"]}
              for g in [0, 2]}
    D = np.empty(a.n_boot)
    for b in range(a.n_boot):
        w = np.bincount(rng.integers(0, nk, nk), minlength=nk)[kcode].astype(float)
        for k in names:
            boot_pr[k][b] = rankers[k].ap(w); boot_roc[k][b] = rankers[k].auc(w)
            boot_thr[k][b] = thr_metrics(w, y, yh[k])
        dd = {g: mech_r[g]["sgae_tuned"].ap(w[grp == g]) - mech_r[g]["tuned_constant"].ap(w[grp == g]) for g in [0, 2]}
        D[b] = dd[2] - dd[0]
        if b % 500 == 0: print(f"  bootstrap {b}/{a.n_boot}", flush=True)
    for k in names:
        table[k]["pr_auc_ci95"] = np.nanpercentile(boot_pr[k], [2.5, 97.5]).tolist()
        table[k]["roc_auc_ci95"] = np.nanpercentile(boot_roc[k], [2.5, 97.5]).tolist()
        for j, nm in enumerate(["precision", "recall", "f1", "mcc"]):
            table[k][f"{nm}_ci95"] = np.nanpercentile(boot_thr[k][:, j], [2.5, 97.5]).tolist()

    comps = []
    for x1, x2 in FAMILY:
        d = boot_pr[x1] - boot_pr[x2]
        d_obs = table[x1]["pr_auc"] - table[x2]["pr_auc"]
        p2 = float(np.mean(np.abs(d - d_obs) >= abs(d_obs)))        # null-centred bootstrap
        comps.append({"comparison": f"{x1} - {x2}", "pr_auc_diff": table[x1]["pr_auc"] - table[x2]["pr_auc"],
                      "ci95": np.nanpercentile(d, [2.5, 97.5]).tolist(), "p_boot": p2,
                      "roc_auc_diff": table[x1]["roc_auc"] - table[x2]["roc_auc"],
                      "roc_ci95": np.nanpercentile(boot_roc[x1] - boot_roc[x2], [2.5, 97.5]).tolist()})
    order = np.argsort([c["p_boot"] for c in comps]); m = len(comps); run_max = 0.0
    for r, i in enumerate(order):                                   # Holm step-down
        run_max = max(run_max, min(1.0, (m - r) * comps[i]["p_boot"]))
        comps[i]["p_holm"] = run_max

    d_obs = {g: average_precision_score(y[grp == g], P_test["sgae_tuned"][grp == g]) -
                average_precision_score(y[grp == g], P_test["tuned_constant"][grp == g]) for g in [0, 2]}
    mech = {"preregistration": cfg["preregistered_mechanism_test"], "D_observed": d_obs[2] - d_obs[0],
            "diff_high_tertile": d_obs[2], "diff_low_tertile": d_obs[0],
            "one_sided_95_lower_bound": float(np.nanpercentile(D, 5)),
            "tertile_sizes": {n: int((grp == g).sum()) for g, n in [(0, "low"), (1, "mid"), (2, "high")]}}
    mech["H1_supported"] = bool(mech["one_sided_95_lower_bound"] > 0)

    seeds_tbl = {}
    for fam in ["xgb_seed", "lstm_selL_seed", "transformer_selL_seed", "gnn_seed", "gnn_noedges_seed"]:
        v = [table[f"{fam}{s}"]["pr_auc"] for s in SEEDS]
        seeds_tbl[fam[:-5]] = {"pr_auc_by_seed": v, "mean": float(np.mean(v)), "sd_seed": float(np.std(v, ddof=1))}
    blocks = np.array_split(np.arange(n_te), 5)
    drift = {k: [float(average_precision_score(y[bk], np.asarray(P_test[k])[bk])) for bk in blocks]
             for k in ["xgb_only", "sgae_tuned", "tuned_constant", "gnn_seed42", "lstm_only"]}

    out = {"n_test": n_te, "n_fraud_test": int(y.sum()), "n_boot": a.n_boot, "bootstrap_unit": "card key",
           "n_card_keys_test": int(nk), "metrics": table, "comparisons_holm": comps,
           "preregistered_mechanism_test": mech, "seed_spread": seeds_tbl, "time_block_pr_auc": drift}
    json.dump(out, open(os.path.join(a.out_dir, "step05_results.json"), "w"), indent=2, default=float)
    log["phase_B_finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); log["secs"] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(a.out_dir, "step05_log.json"), "w"), indent=2, default=str)

    rows = [{"method": k, **{c: v for c, v in table[k].items() if not c.startswith("reliability")}} for k in names]
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "test_metrics.csv"), index=False)
    print("\nTEST RESULTS (primary: PR-AUC; 95% CI = cluster bootstrap over card keys)")
    for k in sorted(names, key=lambda k: -table[k]["pr_auc"]):
        t = table[k]
        print(f"  {k:26s} PR-AUC {t['pr_auc']:.4f} [{t['pr_auc_ci95'][0]:.4f}, {t['pr_auc_ci95'][1]:.4f}]  "
              f"ROC {t['roc_auc']:.4f}  F1 {t['f1']:.3f}  ECE(platt) {t['ece_val_platt']:.4f}")
    print("\nDECLARED COMPARISONS (PR-AUC difference, Holm-adjusted p):")
    for c in comps:
        print(f"  {c['comparison']:48s} {c['pr_auc_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  p_holm {c['p_holm']:.4f}")
    print(f"\nPRESPECIFIED MECHANISM TEST (frozen before test evaluation): D = {mech['D_observed']:+.4f}, one-sided 95% lower bound "
          f"{mech['one_sided_95_lower_bound']:+.4f} -> H1 {'SUPPORTED' if mech['H1_supported'] else 'NOT supported'}")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
