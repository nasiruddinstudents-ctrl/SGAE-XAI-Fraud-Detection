#!/usr/bin/env python3
"""
SGAE v2 — STEP 6: EXPLANATION QUALITY (replaces v1 Table 5 and the Kendall's W analysis).

Runs after the locked test evaluation; nothing here selects or tunes any model.

WHAT IS EXPLAINED (the manuscript uses these exact scopes)
  All four primary models (seed 42) are explained on the same 445 input columns, on the logit
  scale, as CONDITIONAL CURRENT-TRANSACTION attributions:
    XGBoost      TreeSHAP via xgboost pred_contribs. This is PATH-DEPENDENT TreeSHAP: its
                 reference distribution is the training data's tree cover, not the background
                 rows used below. (Interventional TreeSHAP with a shared background was not
                 run; stated as a limitation.)
    LSTM, Transformer
                 Expected Gradients (own implementation of the shap.GradientExplainer
                 algorithm) over the current transaction's columns, the card history held
                 fixed. Background: the 1,000 training rows frozen in the step-4 config.
    GNN          Expected Gradients over the transaction's own columns with its incoming
                 neighbour messages held fixed. This is a current-node-feature,
                 fixed-neighbourhood explanation: it does NOT assess how card or address
                 relationships (graph structure) drove the prediction.
                 Exactness rests on the step-3 graph: every edge goes from a strictly earlier
                 transaction (no self-loops), so a node's own features cannot reach its own
                 incoming messages. Checked here twice: closed form == full graph forward on
                 the unmodified rows, and again after perturbing single rows one at a time.
  Scores are therefore comparable as conditional current-feature perturbation scores; they
  are not a ranking of overall explanation quality across architectures.

EXPLAINED SET
  2,000 TEST transactions, stratified: 1,000 fraud + 1,000 legitimate (fixed seed), same rows
  for every model. Results are reported per class (primary) and pooled; pooled summaries are
  also given re-weighted to the natural test fraud rate. Intervals use a bootstrap over CARD
  KEYS (rows sharing a card are resampled together).

FAITHFULNESS PROXIES (perturbation-based; not evidence of causal mechanism)
  Replacement = values of a random training background row (same 1,000 rows for all models),
  averaged over 16 draws; k in {5, 10, 20, 50}; s = sign(Delta):
    Delta = f(x) - E_b f(b);  C_k = f(x) - E_b f(x, top-k replaced);  S_k = f(x) - E_b f(x, all
    but top-k replaced);  random-k controls C_k^rnd, S_k^rnd.
    NC_k = mean(s C_k)/mean|Delta| (higher better), NS_k = mean(s S_k)/mean|Delta| (lower better),
    gains over random: NC_k - NC_k^rnd, NS_k^rnd - NS_k.
  Background-row replacement can create implausible feature combinations; stated.

STABILITY (each source of variation labelled)
  (a) Retraining: attributions from the three independently trained seeds (42, 43, 44):
      Kendall's W of global importance across seeds, and per-row top-20 Spearman vs seed 42.
  (b) Explainer randomness (EG models only): 10 runs with new training backgrounds and seeds.
      TreeSHAP is deterministic, so its run-to-run repeatability is a computational property
      and is reported as "not applicable", not as evidence of robustness.
  (c) Instance resampling: Kendall's W across bootstrap resamples of the explained set,
      n in {100, 500, 2000}. The reported spread describes variability under this stated
      resampling procedure, not a confidence interval for a population quantity.
  (d) Input robustness: top-10 Jaccard between attributions of x and of x with Gaussian noise
      (1% of the training s.d.) added ONLY to continuous columns (not categorical codes,
      identifier-like columns or binary columns; rule logged), same explainer seed.
  No reliability cutoff (such as W < 0.5) is used.

Usage
  python step06_explanation_quality.py --data_dir /workspace/sgae_v2/data --raw_dir /workspace/ieee-cis \
     --step02_dir /workspace/sgae_v2/step02 --step03_dir /workspace/sgae_v2/step03 \
     --step04_dir /workspace/sgae_v2/step04 --step05_dir /workspace/sgae_v2/step05 \
     --out_dir /workspace/sgae_v2/step06
"""
import argparse, json, os, sys, time, types
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step02_train_models as s2
import step03_gnn as s3
import step04_sgae as s4

KS = [5, 10, 20, 50]
SEEDS = [42, 43, 44]
MODELS = ["xgb", "lstm", "transformer", "gnn"]
ID_LIKE = {"card1", "card2", "card3", "card5", "addr1", "addr2"}
R_FAITH, N_RUNS, TOP_M, N_REP, N_WREP, N_BOOT = 16, 10, 30, 20, 200, 1000


# ─────────────────────────────────────────────────────────────────────────────
# Model wrappers: logit(local_idx, x_current), everything else held fixed
# ─────────────────────────────────────────────────────────────────────────────
def make_seq_fn(model, Xg, tg, seq, lens, rows_g, dev):
    seq_t, len_t, rows_t = (torch.from_numpy(v).to(dev) for v in (seq, lens, rows_g))
    Fd = Xg.shape[1]
    def fn(j, xcur):
        r = rows_t[j]
        xs, l = s2.make_batch(Xg, tg, seq_t, len_t, r)
        xs = xs.clone()
        xs[torch.arange(len(r), device=Xg.device), l - 1, :Fd] = xcur
        return model(xs, l)
    return fn


def make_gnn_fn(model, Xg, A, rows_g, dev):
    model.eval()
    with torch.no_grad():
        agg1 = torch.sparse.mm(A, model.c1.nb_lin(Xg))
        h1 = F.relu(model.n1(model.c1(Xg, A)))
        agg2 = torch.sparse.mm(A, model.c2.nb_lin(h1))
        full = model(Xg, A)
    r = torch.from_numpy(rows_g).to(dev)
    a1, a2, ref = agg1[r], agg2[r], full[r]
    def fn(j, xcur):
        h = F.relu(model.n1(model.c1.self_lin(xcur) + a1[j]))
        z = F.relu(model.n2(model.c2.self_lin(h) + a2[j]))
        return model.head(z).squeeze(-1)
    return fn, ref


def eg(fn, xcur, bg_X, M, seed, batch, dev, cudnn_off=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, Fd = xcur.shape
    out, gaps = [], []
    with torch.backends.cudnn.flags(enabled=not cudnn_off):
        for i in range(0, n, batch):
            j = torch.arange(i, min(i + batch, n), device=dev)
            B = len(j); x = xcur[j]
            b = bg_X[torch.randint(len(bg_X), (B, M), generator=g).to(dev)]
            al = torch.rand((B, M, 1), generator=g).to(dev)
            pts = (b + al * (x[:, None] - b)).reshape(B * M, Fd).requires_grad_(True)
            jj = j.repeat_interleave(M)
            gr, = torch.autograd.grad(fn(jj, pts).sum(), pts)
            att = ((x[:, None] - b) * gr.reshape(B, M, Fd)).mean(1)
            with torch.no_grad():
                gap = att.sum(1) - (fn(j, x) - fn(jj, b.reshape(B * M, Fd)).reshape(B, M).mean(1))
            out.append(att.detach().cpu().numpy()); gaps.append(gap.abs().cpu().numpy())
    return np.concatenate(out).astype(np.float32), np.concatenate(gaps)


def torch_to_np_fn(f, n, dev):
    def npf(Xm):
        out = []
        with torch.no_grad():
            for i in range(0, n, 1024):
                j = torch.arange(i, min(i + 1024, n), device=dev)
                out.append(f(j, torch.from_numpy(np.ascontiguousarray(Xm[i:i + 1024])).to(dev)).cpu().numpy())
        return np.concatenate(out)
    return npf


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────
def cluster_draws(keys, n_boot, seed):
    """Card-key cluster bootstrap: returns a list of row-index arrays."""
    codes = pd.factorize(keys)[0]
    groups = [np.where(codes == c)[0] for c in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[c] for c in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]


def faithfulness(f_np, Xc, phi, bg_np, seed, y, draws_idx, prev):
    n, Fd = Xc.shape
    rng = np.random.default_rng(seed)
    order = np.argsort(-np.abs(phi), axis=1)
    fx = f_np(Xc)
    bgs = [bg_np[rng.integers(0, len(bg_np), n)] for _ in range(R_FAITH)]
    delta = fx - np.mean([f_np(b) for b in bgs], 0)
    s = np.sign(delta)
    per = {}
    for k in KS:
        top = np.zeros((n, Fd), bool); np.put_along_axis(top, order[:, :k], True, 1)
        rnd = np.zeros((n, Fd), bool); np.put_along_axis(rnd, np.argsort(rng.random((n, Fd)), 1)[:, :k], True, 1)
        per[k] = {"C": fx - np.mean([f_np(np.where(top, b, Xc)) for b in bgs], 0),
                  "S": fx - np.mean([f_np(np.where(top, Xc, b)) for b in bgs], 0),
                  "Cr": fx - np.mean([f_np(np.where(rnd, b, Xc)) for b in bgs], 0),
                  "Sr": fx - np.mean([f_np(np.where(rnd, Xc, b)) for b in bgs], 0)}
    w_pop = np.where(y == 1, prev / y.mean(), (1 - prev) / (1 - y.mean()))

    def summ(idx, w=None):
        w = np.ones(len(idx)) if w is None else w[idx]
        d = np.sum(w * np.abs(delta[idx])) / w.sum()
        o = {}
        for k in KS:
            p = per[k]
            m = lambda v: np.sum(w * s[idx] * v[idx]) / w.sum() / d
            nc, ns, ncr, nsr = m(p["C"]), m(p["S"]), m(p["Cr"]), m(p["Sr"])
            o[k] = [nc, ns, nc - ncr, nsr - ns]
        return o

    names = ["norm_comprehensiveness", "norm_sufficiency", "comp_gain_over_random", "suff_gain_over_random"]
    res = {}
    for label, sel, w in [("fraud", y == 1, None), ("legit", y == 0, None),
                          ("pooled_balanced", np.ones(n, bool), None), ("pooled_reweighted_to_test_prevalence", np.ones(n, bool), w_pop)]:
        base = np.where(sel)[0]
        full = summ(base, w)
        boots = [summ(d[sel[d]], w) for d in draws_idx]
        res[label] = {f"k{k}": {nm: {"value": float(full[k][i]),
                                     "ci95_card_bootstrap": np.nanpercentile([b[k][i] for b in boots], [2.5, 97.5]).tolist()}
                                for i, nm in enumerate(names)} for k in KS}
    res["mean_abs_total_effect_logit"] = float(np.abs(delta).mean())
    return res


def kendall_w(ranks):
    m, M = ranks.shape
    Rj = ranks.sum(0)
    return float(12 * ((Rj - Rj.mean()) ** 2).sum() / (m ** 2 * (M ** 3 - M)))


def w_resampling(runs, top, n_sub, instances_only, seed):
    rng = np.random.default_rng(seed)
    n = runs[0].shape[0]
    ws = []
    for _ in range(N_WREP):
        rk = []
        for _ in range(N_REP):
            phi = runs[0] if instances_only else runs[rng.integers(0, len(runs))]
            imp = np.abs(phi[rng.integers(0, n, n_sub)][:, top]).mean(0)
            rk.append(np.argsort(np.argsort(-imp)) + 1)
        ws.append(kendall_w(np.array(rk)))
    return {"mean": float(np.mean(ws)), "spread_2.5_97.5": np.percentile(ws, [2.5, 97.5]).tolist(),
            "note": "variability under the stated resampling procedure, not a population CI"}


def cluster_mean_ci(v, draws_idx):
    bs = [np.nanmean(v[d]) for d in draws_idx]
    return {"mean": float(np.nanmean(v)), "ci95_card_bootstrap": np.nanpercentile(bs, [2.5, 97.5]).tolist()}


def jaccard_topk(a, b, k=10):
    ta = np.argsort(-np.abs(a), 1)[:, :k]; tb = np.argsort(-np.abs(b), 1)[:, :k]
    return np.array([len(set(x) & set(z)) / len(set(x) | set(z)) for x, z in zip(ta, tb)])


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    for k in ["data_dir", "raw_dir", "step02_dir", "step03_dir", "step04_dir", "step05_dir", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--eg_samples", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    global R_FAITH, N_RUNS, N_WREP, N_BOOT
    if a.quick: a.n_per_class, a.eg_samples, R_FAITH, N_RUNS, N_WREP, N_BOOT = 60, 16, 4, 3, 20, 50
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    cfg = json.load(open(os.path.join(a.step04_dir, "step04_config.json")))
    L = cfg["base"]["lstm_L"]
    L_tf = json.load(open(os.path.join(a.step02_dir, "step02_log.json")))["transformer"]["selected_L"]
    feat = json.load(open(os.path.join(a.data_dir, "feature_names.json")))
    prep = json.load(open(os.path.join(a.data_dir, "preprocessing.json")))

    meta = pd.read_csv(os.path.join(a.data_dir, "meta.csv.gz"))
    meta = meta[meta["split"] != "gap"].reset_index(drop=True)
    parts = {s: np.load(os.path.join(a.data_dir, f"X_{s}.npy")) for s in ["train", "val", "test"]}
    n_tr, n_va = len(parts["train"]), len(parts["val"])
    X = np.concatenate([parts["train"], parts["val"], parts["test"]]).astype(np.float32)
    y_te = np.load(os.path.join(a.data_dir, "y_test.npy")); prev = float(y_te.mean())
    te0 = n_tr + n_va
    rng = np.random.default_rng(2026)
    loc = np.sort(np.concatenate([rng.choice(np.where(y_te == 1)[0], a.n_per_class, replace=False),
                                  rng.choice(np.where(y_te == 0)[0], a.n_per_class, replace=False)]))
    rows = te0 + loc; yex = y_te[loc].astype(int); n = len(rows)
    keys = meta["key_id"].to_numpy()[rows]
    draws = cluster_draws(keys, N_BOOT, 11)
    Xg = torch.from_numpy(X).to(dev)
    tg = torch.from_numpy(meta["TransactionDT"].to_numpy(np.int64).copy()).to(dev)
    xcur = Xg[torch.from_numpy(rows).to(dev)]
    bg = np.array(cfg["attribution"]["bg_train_rows"]); bg_t = Xg[torch.from_numpy(bg).to(dev)]

    # continuous columns only for the robustness perturbation (rule logged)
    nun = np.array([len(np.unique(parts["train"][:50000, j])) for j in range(X.shape[1])])
    cont = np.array([(f not in prep["categories"]) and (f not in ID_LIKE) and nun[j] > 50 for j, f in enumerate(feat)])
    tr_sd = X[:n_tr].std(0) + 1e-6
    noise = (np.random.default_rng(7).standard_normal((n, X.shape[1])) * 0.01 * tr_sd * cont).astype(np.float32)
    log = {"explained_rows": n, "n_fraud": int(yex.sum()), "test_fraud_rate": prev, "n_card_keys": int(len(set(keys))),
           "eg_samples": a.eg_samples, "R_faith": R_FAITH, "n_explainer_runs": N_RUNS, "quick": a.quick,
           "perturbation_rule": "Gaussian noise (1% train s.d.) only on columns that are not categorical codes, "
                                "not identifier-like (card1-3,5, addr1-2) and have >50 distinct training values",
           "n_continuous_perturbed": int(cont.sum())}
    print(f"Explaining {n} test rows ({int(yex.sum())} fraud, {len(set(keys))} cards) | continuous cols {cont.sum()} | {dev}", flush=True)

    # ── graph (shared by all GNN seeds) ──────────────────────────────────────
    raw = pd.read_csv(os.path.join(a.raw_dir, "train_transaction.csv"), usecols=["TransactionID", "addr1", "ProductCD"])
    meta_g = meta.merge(raw, on="TransactionID", how="left", validate="one_to_one")
    e, gchk = s3.build_graph(meta_g)
    assert gchk["all_edges_point_forward_in_time"] and not (e[0] == e[1]).any(), "self-loop or non-forward edge"
    A, _ = s3.adjacency(e, len(meta_g), dev)
    log["graph"] = {"n_edges": int(e.shape[1]), "self_loops": 0, "all_forward_in_time": True}

    saved_te = np.load(os.path.join(a.step05_dir, "test_predictions.npz"))
    import xgboost as xgb

    def load_models(sd):
        bst = xgb.Booster(); bst.load_model(os.path.join(a.step02_dir, "models", f"xgb_seed{sd}.json"))
        if dev == "cuda": bst.set_param({"device": "cuda"})
        it = (0, bst.num_boosted_rounds())
        margin = lambda Xm: bst.predict(xgb.DMatrix(Xm), output_margin=True, iteration_range=it)
        d = float(np.abs(s4.sigm(margin(X[rows]).astype(float)) - saved_te[f"xgb_seed{sd}"][loc]).max())
        assert d < 1e-5, f"xgb seed {sd}: explained model != model evaluated in step 5"
        fns = {"xgb_margin": margin, "xgb_check": d}
        for kind, LL in [("lstm", L), ("transformer", L_tf)]:
            m = (s2.LSTMModel(X.shape[1] + 1) if kind == "lstm" else s2.SeqTransformer(X.shape[1] + 1, LL + 1)).to(dev)
            m.load_state_dict(torch.load(os.path.join(a.step02_dir, "models", f"{kind}_L{LL}_seed{sd}.pt"), map_location=dev)); m.eval()
            fns[kind] = make_seq_fn(m, Xg, tg, np.load(os.path.join(a.step02_dir, f"seq_idx_L{LL}.npy")),
                                    np.load(os.path.join(a.step02_dir, f"seq_len_L{LL}.npy")), rows, dev)
            with torch.no_grad():
                p = torch.sigmoid(fns[kind](torch.arange(n, device=dev), xcur)).cpu().numpy()
            dd = float(np.abs(p - saved_te[f"{kind}_selL_seed{sd}"][loc]).max())
            assert dd < 1e-4, f"{kind} seed {sd}: explained model != model evaluated in step 5"
        gm = s3.GNN(X.shape[1]).to(dev)
        gm.load_state_dict(torch.load(os.path.join(a.step03_dir, "models", f"gnn_seed{sd}.pt"), map_location=dev)); gm.eval()
        fns["gnn"], ref = make_gnn_fn(gm, Xg, A, rows, dev)
        with torch.no_grad():
            d0 = float((fns["gnn"](torch.arange(n, device=dev), xcur) - ref).abs().max())
            dt = float(np.abs(torch.sigmoid(ref).cpu().numpy() - saved_te[f"gnn_seed{sd}"][loc]).max())
            # perturbed single-row check: closed form must equal the full graph forward pass
            dp = 0.0
            for i in np.random.default_rng(sd).choice(n, 5 if a.quick else 20, replace=False):
                Xm = Xg.clone(); xp = xcur[i] + torch.from_numpy(noise[i] * 50).to(dev)
                Xm[rows[i]] = xp
                full_i = gm(Xm, A)[rows[i]]
                dp = max(dp, float((fns["gnn"](torch.tensor([i], device=dev), xp[None]) - full_i).abs().max()))
        assert d0 < 1e-3 and dp < 1e-3 and dt < 1e-4, f"GNN closed-form checks failed: {d0}, {dp}, {dt}"
        fns["gnn_checks"] = {"unmodified": d0, "perturbed_single_rows": dp, "vs_step5_test_preds": dt}
        return fns, types.SimpleNamespace(step02_dir=a.step02_dir, seed=sd)

    def attributions(fns, pa, seed_base, runs_needed, extra_perturbed):
        out = {}
        phi_x, info = s4.xgb_contribs(pa, X, rows, saved_prob=s4.sigm(fns["xgb_margin"](X[rows]).astype(float)))
        out["xgb"] = {"runs": [phi_x], "treeshap_check": info}
        if extra_perturbed:
            Xp = X.copy(); Xp[rows] += noise
            out["xgb"]["perturbed"] = s4.xgb_contribs(pa, Xp, rows)[0]
        for m in ["lstm", "transformer", "gnn"]:
            off = m == "lstm"
            rr, gp = [], None
            for r in range(runs_needed):
                bgr = bg_t if r == 0 else Xg[torch.from_numpy(np.random.default_rng(100 + r).choice(n_tr, len(bg), replace=False)).to(dev)]
                phi, g_ = eg(fns[m], xcur, bgr, a.eg_samples, seed_base + r, a.batch, dev, off)
                rr.append(phi)
                if r == 0: gp = g_
            out[m] = {"runs": rr, "eg_gap_median": float(np.median(gp)), "eg_gap_p95": float(np.percentile(gp, 95))}
            if extra_perturbed:
                out[m]["perturbed"] = eg(fns[m], xcur + torch.from_numpy(noise).to(dev), bg_t, a.eg_samples, seed_base, a.batch, dev, off)[0]
        return out

    # ── primary seed: faithfulness, explainer randomness, instance resampling, robustness ──
    fns42, pa42 = load_models(42)
    att = attributions(fns42, pa42, 6000, N_RUNS, True)
    log["model_identity_checks_seed42"] = {"xgb_vs_step5": fns42["xgb_check"], "gnn": fns42["gnn_checks"]}
    log["attribution_checks"] = {m: {k: v for k, v in att[m].items() if k not in ("runs", "perturbed")} for m in MODELS}
    print(f"  seed-42 attributions done ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(a.out_dir, "attributions_seed42_run0.npz"), rows=rows, y=yex,
                        **{m: att[m]["runs"][0] for m in MODELS})

    np_fns = {"xgb": fns42["xgb_margin"], **{m: torch_to_np_fn(fns42[m], n, dev) for m in ["lstm", "transformer", "gnn"]}}
    res = {"faithfulness": {}, "stability": {}, "robustness": {}, "cross_model": {}, "global_top10_features": {}}
    for m in MODELS:
        res["faithfulness"][m] = faithfulness(np_fns[m], X[rows], att[m]["runs"][0], X[bg], 99, yex, draws, prev)
        print(f"  faithfulness {m} done ({time.time()-t0:.0f}s)", flush=True)

    # ── retraining stability: seeds 43, 44 (primary attribution run only) ────
    seed_att = {42: {m: att[m]["runs"][0] for m in MODELS}}
    for sd in [43, 44]:
        f_, p_ = load_models(sd)
        a_ = attributions(f_, p_, 6000, 1, False)
        seed_att[sd] = {m: a_[m]["runs"][0] for m in MODELS}
        print(f"  seed-{sd} attributions done ({time.time()-t0:.0f}s)", flush=True)

    for m in MODELS:
        runs = att[m]["runs"]
        pooled = np.mean([np.abs(p).mean(0) for p in runs], 0)
        top, top20 = np.argsort(-pooled)[:TOP_M], np.argsort(-pooled)[:20]
        imps = np.array([np.abs(seed_att[sd][m]).mean(0)[top] for sd in SEEDS])
        st = {"retraining_W_across_3_seeds": kendall_w(np.array([np.argsort(np.argsort(-v)) + 1 for v in imps])),
              "retraining_rowwise_top20_spearman_vs_seed42": {
                  str(sd): cluster_mean_ci(s4.rowwise_spearman(seed_att[42][m][:, top20], seed_att[sd][m][:, top20]), draws)
                  for sd in [43, 44]}}
        if m == "xgb":
            st["explainer_repeatability"] = "not applicable: TreeSHAP is deterministic (computational property)"
        else:
            st["explainer_repeatability_top20_spearman"] = cluster_mean_ci(s4.rowwise_spearman(runs[0][:, top20], runs[1][:, top20]), draws)
        for ns in sorted({100, 500, n}):
            st[f"instance_resampling_W_n{ns}"] = w_resampling(runs, top, ns, True, 5 + ns)
            if m != "xgb":
                st[f"instance_plus_explainer_W_n{ns}"] = w_resampling(runs, top, ns, False, 7 + ns)
        res["stability"][m] = st
        res["robustness"][m] = {"top10_jaccard_continuous_noise": cluster_mean_ci(jaccard_topk(runs[0], att[m]["perturbed"]), draws)}
        res["global_top10_features"][m] = [feat[i] for i in np.argsort(-pooled)[:10]]
    imp = {m: np.abs(att[m]["runs"][0]).mean(0) for m in MODELS}
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            res["cross_model"][f"{m1}_vs_{m2}_global_importance_spearman"] = float(spearmanr(imp[m1], imp[m2])[0])

    json.dump({**log, **res}, open(os.path.join(a.out_dir, "step06_results.json"), "w"), indent=2, default=float)
    print("\nEXPLANATION QUALITY — conditional current-feature scores (per class; CIs in step06_results.json)")
    print(f"  {'model':12s} {'NC10 fraud':>10s} {'NC10 legit':>10s} {'NS10 fraud':>10s} {'gain':>6s} "
          f"{'W seeds':>8s} {'seed rho':>8s} {'repeat':>7s} {'robust':>7s}")
    for m in MODELS:
        fr, lg = res["faithfulness"][m]["fraud"]["k10"], res["faithfulness"][m]["legit"]["k10"]
        st = res["stability"][m]
        rep = "n/a" if m == "xgb" else f"{st['explainer_repeatability_top20_spearman']['mean']:.3f}"
        print(f"  {m:12s} {fr['norm_comprehensiveness']['value']:10.3f} {lg['norm_comprehensiveness']['value']:10.3f} "
              f"{fr['norm_sufficiency']['value']:10.3f} {fr['comp_gain_over_random']['value']:6.3f} "
              f"{st['retraining_W_across_3_seeds']:8.3f} {st['retraining_rowwise_top20_spearman_vs_seed42']['43']['mean']:8.3f} "
              f"{rep:>7s} {res['robustness'][m]['top10_jaccard_continuous_noise']['mean']:7.3f}")
    print(f"  GNN checks: {log['model_identity_checks_seed42']['gnn']} | continuous columns perturbed: {int(cont.sum())}")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
