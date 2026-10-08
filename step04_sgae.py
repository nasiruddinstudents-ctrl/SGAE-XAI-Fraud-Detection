#!/usr/bin/env python3
"""
SGAE v2 — STEP 4: attributions, SGAE calibration and baselines (VALIDATION ONLY).

Everything here uses train (background, global feature ranking) and validation
(calibration, selection). Test rows are not loaded. The output step04_config.json freezes
every parameter; the final locked step applies it unchanged to the test set.

Base models (primary seed 42, declared in step 2): XGBoost and the LSTM at the history
length selected on validation in step 2.

Attributions (same 445 features, same log-odds output scale for both models)
  XGBoost  exact TreeSHAP via xgboost pred_contribs (margin / log-odds).
  LSTM     Expected Gradients (the algorithm of shap.GradientExplainer) on the LSTM logit,
           for the CURRENT transaction's 445 features, holding the card's history fixed.
           This is our own implementation of the Expected Gradients algorithm (the one
           shap.GradientExplainer uses); the shap library itself is not called.
           The LSTM attribution is CONDITIONAL on the card's fixed history, whereas TreeSHAP
           attributes XGBoost's whole tabular prediction: same feature columns and output
           scale, but not identical scientific quantities (stated in the manuscript).
           Background: 1,000 random TRAIN transactions (fixed seed). 64 samples per row.
           Run twice with independent background/sample seeds (A, B): run A drives SGAE;
           the A-vs-B agreement per row is the LSTM's local explanation reliability.
  Completeness gaps (sum of attributions vs. output difference) are reported.

SGAE rule (v1 Algorithm 1, now fully specified)
  1. Global top-K features: rank features by mean|phi| on a 20,000-row TRAIN sample,
     each model's mean|phi| normalised to sum 1, then summed. Fixed set, learned on train.
  2. A_i = Spearman correlation of phi_LSTM(x_i) and phi_XGB(x_i) over those K features
     (average ranks for ties; A_i = 0 if either vector is constant).
  3. sigma_A = std of A over validation.
  4. w_LSTM,i = clip(w0 + c * tanh(A_i / sigma_A), 0, 1) if A_i >= 0 (or always, for the
     "symmetric" rule); otherwise w_neg (the v1 "negative-agreement" rule).
  5. p_i = w_LSTM,i * p_LSTM,i + (1 - w_LSTM,i) * p_XGB,i
  Base probabilities are Platt-calibrated on validation first (monotone; single-model
  rankings unchanged) because the LSTM was trained with pos_weight and XGBoost was not.
  "sgae_v1_weighting_rule" applies the v1 settings (w0 0.5, c 0.2, w_neg 0.6, K 10,
  clip [0.3, 0.7]) to the v2 models and attributions. It is the v1 RULE re-implemented,
  not a reproduction of the originally reported numbers.

Selection (declared grid, criterion = validation PR-AUC)
  SGAE: w0 in {0, .1, ..., 1}, c in {.1, .2, .3}, K in {5, 10, 20},
        negative rule in {w_neg = 0, .2, .4, .6, symmetric}.
  Tuned constant: w in {0, .05, ..., 1}.
  Because tuning and scoring on the same validation rows is optimistic, every method is
  ALSO scored FORWARD IN TIME: validation is cut into 5 contiguous time blocks; for block
  k = 2..5, the Platt calibrators, grid selections and fitted gates/stackers are fitted on
  blocks 1..k-1 only and applied to block k. The forward-time PR-AUC over blocks 2-5 is the
  fair validation comparison. Final calibrators and parameters are then fitted on all of
  validation and frozen for the test set, which remains the prospective assessment.

Checks: XGBoost TreeSHAP must reproduce the SAVED step-2 validation probabilities
  (sigmoid of summed contributions); the reloaded LSTM must reproduce its saved
  predictions; feature counts of both models must equal the feature file.

Baselines: XGBoost alone, LSTM alone, equal weight, v1 static 0.6/0.4, tuned constant,
  confidence gate (weight by distance of each logit from the train base-rate logit),
  stacking (logistic regression on both logits), learned gate (mixture of experts, gate
  on both logits and their gap), learned gate + agreement A, SGAE with permuted A
  (20 permutations), SGAE without tanh (linear), SGAE with local (per-row) top-K,
  SGAE driven by the independent LSTM attribution run B.

Usage
  python step04_sgae.py --data_dir /workspace/sgae_v2/data --step02_dir /workspace/sgae_v2/step02 \
                        --out_dir /workspace/sgae_v2/step04
"""
import argparse, json, os, sys, time, hashlib, itertools
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step02_train_models as s2   # reuse LSTMModel and make_batch exactly

EPS = 1e-6
GRID = {"w0": [float(round(x, 1)) for x in np.arange(0, 1.01, 0.1)], "c": [0.1, 0.2, 0.3],
        "K": [5, 10, 20], "neg": [0.0, 0.2, 0.4, 0.6, "symmetric"]}
CONST_GRID = [float(round(x, 2)) for x in np.arange(0, 1.001, 0.05)]
N_FOLDS, N_PERM = 5, 20


def pr(y, p): return float(average_precision_score(y, p))
def roc(y, p): return float(roc_auc_score(y, p))
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sigm(z): return 1 / (1 + np.exp(-z))


# ─────────────────────────────────────────────────────────────────────────────
# Data and models
# ─────────────────────────────────────────────────────────────────────────────
def load(a):
    meta = pd.read_csv(os.path.join(a.data_dir, "meta.csv.gz"))
    meta = meta[meta["split"] != "gap"].reset_index(drop=True)
    Xtr, Xva = (np.load(os.path.join(a.data_dir, f"X_{s}.npy")) for s in ["train", "val"])
    ytr, yva = (np.load(os.path.join(a.data_dir, f"y_{s}.npy")) for s in ["train", "val"])
    n_tr, n_va = len(Xtr), len(Xva)
    meta = meta.iloc[:n_tr + n_va].copy()                # test rows not loaded
    assert (meta["split"].to_numpy()[:n_tr] == "train").all() and (meta["split"].to_numpy()[n_tr:] == "val").all()
    X = np.concatenate([Xtr, Xva]).astype(np.float32)
    log2 = json.load(open(os.path.join(a.step02_dir, "step02_log.json")))
    L = log2["lstm"]["selected_L"] if a.lstm_L is None else a.lstm_L
    seq = np.load(os.path.join(a.step02_dir, f"seq_idx_L{L}.npy"))[:n_tr + n_va]
    lens = np.load(os.path.join(a.step02_dir, f"seq_len_L{L}.npy"))[:n_tr + n_va]
    assert seq.max() < n_tr + n_va, "a train/val history points at a test row"
    rows_val = pd.read_csv(os.path.join(a.step02_dir, "preds", "rows_val.csv.gz"))
    assert (rows_val["TransactionID"].to_numpy() == meta["TransactionID"].to_numpy()[n_tr:]).all()
    p = {"xgb": np.load(os.path.join(a.step02_dir, "preds", f"xgb_seed{a.seed}_val.npy")).astype(float),
         "lstm": np.load(os.path.join(a.step02_dir, "preds", f"lstm_L{L}_seed{a.seed}_val.npy")).astype(float)}
    return meta, X, np.asarray(ytr, float), np.asarray(yva, float), n_tr, n_va, seq, lens, L, p


def xgb_contribs(a, X, rows, saved_prob=None):
    """Exact TreeSHAP over ALL trees of the saved booster. Step 2 saved the booster already
    truncated to its early-stopped trees and produced its validation predictions from that
    same truncated booster, so an explicit iteration_range covering every saved tree is used
    here. If saved_prob is given, sigmoid(sum of contributions) must reproduce it."""
    import xgboost as xgb
    bst = xgb.Booster(); bst.load_model(os.path.join(a.step02_dir, "models", f"xgb_seed{a.seed}.json"))
    if torch.cuda.is_available(): bst.set_param({"device": "cuda"})
    assert bst.num_features() == X.shape[1], "XGBoost feature count != feature file"
    it = (0, bst.num_boosted_rounds())
    out = []
    for i in range(0, len(rows), 20000):
        c = bst.predict(xgb.DMatrix(X[rows[i:i + 20000]]), pred_contribs=True, iteration_range=it)
        out.append(c.astype(np.float64))
    c = np.concatenate(out)
    info = {"n_trees": int(bst.num_boosted_rounds())}
    if saved_prob is not None:
        info["max_abs_diff_sigmoid_sum_vs_saved_prob"] = float(np.abs(sigm(c.sum(1)) - saved_prob).max())
        assert info["max_abs_diff_sigmoid_sum_vs_saved_prob"] < 1e-4, \
            "TreeSHAP does not reproduce the saved XGBoost probabilities"
    return c[:, :-1].astype(np.float32), info


def lstm_eg(model, Xg, tg, seq, lens, rows, bg, M, seed, dev, batch):
    """Expected Gradients for the current transaction's features (history held fixed).
    Returns attributions (n, F) and per-row completeness gap."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    F_ = Xg.shape[1]
    seq_t = torch.from_numpy(seq).to(dev); len_t = torch.from_numpy(lens).to(dev)
    bg_t = torch.from_numpy(bg).to(dev)
    attrs, gaps = [], []
    t_start = time.time()
    model.eval()
    with torch.backends.cudnn.flags(enabled=False):     # RNN backward in eval mode
        for i in range(0, len(rows), batch):
            r = torch.from_numpy(rows[i:i + batch]).to(dev)
            B = len(r)
            xs, l = s2.make_batch(Xg, tg, seq_t, len_t, r)              # (B,S,F+1)
            cur = (l - 1)
            ar = torch.arange(B, device=dev)
            x_cur = xs[ar, cur, :F_]                                     # (B,F)
            bi = bg_t[torch.randint(len(bg), (B, M), generator=g).to(dev)]
            b = Xg[bi]                                                   # (B,M,F)
            alpha = torch.rand((B, M, 1), generator=g).to(dev)
            pts = b + alpha * (x_cur[:, None] - b)
            xr = xs.repeat_interleave(M, 0).clone()
            lr = l.repeat_interleave(M, 0)
            ar2 = torch.arange(B * M, device=dev)
            xr[ar2, lr - 1, :F_] = pts.reshape(B * M, F_)
            xr.requires_grad_(True)
            out = model(xr, lr)
            grad, = torch.autograd.grad(out.sum(), xr)
            gcur = grad[ar2, lr - 1, :F_].reshape(B, M, F_)
            att = ((x_cur[:, None] - b) * gcur).mean(1)
            with torch.no_grad():                                        # completeness check
                fx = model(xs, l)
                xb = xs.repeat_interleave(M, 0).clone()
                xb[ar2, lr - 1, :F_] = b.reshape(B * M, F_)
                fb = model(xb, lr).reshape(B, M).mean(1)
            gaps.append((att.sum(1) - (fx - fb)).abs().detach().cpu().numpy())
            attrs.append(att.detach().cpu().numpy().astype(np.float32))
            if i == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
                eta = (time.time() - t_start) * len(rows) / max(B, 1) / 60
                print(f"    EG first batch: peak GPU {mem:.2f} GB | est. {eta:.1f} min for {len(rows):,} rows", flush=True)
    return np.concatenate(attrs), np.concatenate(gaps)


# ─────────────────────────────────────────────────────────────────────────────
# Agreement and ensembles
# ─────────────────────────────────────────────────────────────────────────────
def rowwise_spearman(a, b):
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), 0.0)


def global_ranking(phi_l_tr, phi_x_tr):
    il = np.abs(phi_l_tr).mean(0); ix = np.abs(phi_x_tr).mean(0)
    return np.argsort(-(il / il.sum() + ix / ix.sum()))


def agreement(phi_l, phi_x, rank, K):
    top = rank[:K]
    return rowwise_spearman(phi_l[:, top], phi_x[:, top])


def agreement_local(phi_l, phi_x, K):
    imp = np.abs(phi_l) / (np.abs(phi_l).sum(1, keepdims=True) + EPS) + \
          np.abs(phi_x) / (np.abs(phi_x).sum(1, keepdims=True) + EPS)
    top = np.argpartition(-imp, K, axis=1)[:, :K]
    return rowwise_spearman(np.take_along_axis(phi_l, top, 1), np.take_along_axis(phi_x, top, 1))


def sgae_w(A, sigma, w0, c, neg, lo=0.0, hi=1.0, shape="tanh"):
    s = np.tanh(A / (sigma + 1e-12)) if shape == "tanh" else np.clip(A / (sigma + 1e-12), -1, 1)
    w = np.clip(w0 + c * s, lo, hi)
    if neg != "symmetric":
        w = np.where(A >= 0, w, neg)
    return w


def mix(w, pl, px): return w * pl + (1 - w) * px


class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y); return self
    def __call__(self, p): return self.m.predict_proba(logit(p)[:, None])[:, 1]
    def params(self): return {"coef": float(self.m.coef_[0, 0]), "intercept": float(self.m.intercept_[0])}


def fit_gate(zl, zx, pl, px, y, extra=None, iters=300):
    """Mixture of experts: p = g*pl + (1-g)*px, g = sigmoid(theta . [1, zl, zx, |zl-zx|, extra])."""
    feats = [np.ones_like(zl), zl, zx, np.abs(zl - zx)] + ([extra] if extra is not None else [])
    Fm = torch.tensor(np.stack(feats, 1), dtype=torch.float64)
    mu, sd = Fm[:, 1:].mean(0), Fm[:, 1:].std(0) + 1e-9
    Fm[:, 1:] = (Fm[:, 1:] - mu) / sd
    th = torch.zeros(Fm.shape[1], dtype=torch.float64, requires_grad=True)
    PL, PX, Y = (torch.tensor(v, dtype=torch.float64) for v in (pl, px, y))
    opt = torch.optim.LBFGS([th], max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        g = torch.sigmoid(Fm @ th); p = (g * PL + (1 - g) * PX).clamp(1e-7, 1 - 1e-7)
        loss = -(Y * p.log() + (1 - Y) * (1 - p).log()).mean() + 1e-4 * (th[1:] ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    params = {"theta": th.detach().numpy().tolist(), "mu": mu.numpy().tolist(), "sd": sd.numpy().tolist()}
    return params


def apply_gate(params, zl, zx, pl, px, extra=None):
    feats = [zl, zx, np.abs(zl - zx)] + ([extra] if extra is not None else [])
    Fm = (np.stack(feats, 1) - np.array(params["mu"])) / np.array(params["sd"])
    th = np.array(params["theta"])
    g = sigm(th[0] + Fm @ th[1:])
    return mix(g, pl, px)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--step02_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lstm_L", type=int, default=None)
    ap.add_argument("--eg_samples", type=int, default=64)
    ap.add_argument("--bg_size", type=int, default=1000)
    ap.add_argument("--train_sample", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.eg_samples, a.bg_size, a.train_sample, a.batch = 8, 200, 2000, 256
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    meta, X, ytr, y, n_tr, n_va, seq, lens, L, p_raw = load(a)
    print(f"Device {dev} | train {n_tr:,} val {n_va:,} (test not loaded) | LSTM L={L} seed={a.seed}")
    log = {"protocol": {"seed": a.seed, "lstm_L": L, "eg_samples": a.eg_samples, "bg_size": a.bg_size,
                        "train_sample_for_ranking": a.train_sample, "grid": GRID, "const_grid": CONST_GRID,
                        "n_folds": N_FOLDS, "n_permutations": N_PERM, "test_rows_loaded": False,
                        "selection_metric": "validation PR-AUC", "quick": a.quick}}

    # ── models ────────────────────────────────────────────────────────────────
    feat = json.load(open(os.path.join(a.data_dir, "feature_names.json")))
    assert len(feat) == X.shape[1], "feature file does not match X"
    lstm = s2.LSTMModel(X.shape[1] + 1).to(dev)
    lstm.load_state_dict(torch.load(os.path.join(a.step02_dir, "models", f"lstm_L{L}_seed{a.seed}.pt"), map_location=dev))
    lstm.eval()
    Xg = torch.from_numpy(X).to(dev)
    tg = torch.from_numpy(meta["TransactionDT"].to_numpy(np.int64).copy()).to(dev)
    # reproduce stored LSTM validation predictions exactly (guards against mismatched files)
    with torch.no_grad():
        chk = s2.predict_nn(lstm, Xg, tg, torch.from_numpy(seq).to(dev), torch.from_numpy(lens).to(dev),
                            torch.arange(n_tr, n_tr + min(n_va, 5000), device=dev), 2048)
    log["lstm_reload_max_abs_diff"] = float(np.abs(chk - p_raw["lstm"][:len(chk)]).max())
    assert log["lstm_reload_max_abs_diff"] < 1e-4, "reloaded LSTM does not reproduce step 2 predictions"

    # ── attributions ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(a.seed)
    bg = np.sort(rng.choice(n_tr, a.bg_size, replace=False))
    tr_sample = np.sort(rng.choice(n_tr, a.train_sample, replace=False))
    val_rows = np.arange(n_tr, n_tr + n_va)
    phi_x_tr, _ = xgb_contribs(a, X, tr_sample)
    phi_x, xinfo = xgb_contribs(a, X, val_rows, saved_prob=p_raw["xgb"])
    phi_l_tr, _ = lstm_eg(lstm, Xg, tg, seq, lens, tr_sample, bg, a.eg_samples, 1000 + a.seed, dev, a.batch)
    print(f"Train-sample attributions done ({time.time()-t0:.0f}s)", flush=True)
    phi_l, gapA = lstm_eg(lstm, Xg, tg, seq, lens, val_rows, bg, a.eg_samples, 2000 + a.seed, dev, a.batch)
    print(f"Validation LSTM attributions run A done ({time.time()-t0:.0f}s)", flush=True)
    bg_b = np.sort(np.random.default_rng(a.seed + 7).choice(n_tr, a.bg_size, replace=False))
    phi_l_b, gapB = lstm_eg(lstm, Xg, tg, seq, lens, val_rows, bg_b, a.eg_samples, 3000 + a.seed, dev, a.batch)
    print(f"Validation LSTM attributions run B done ({time.time()-t0:.0f}s)", flush=True)
    log["attribution_checks"] = {
        "xgb_treeshap": xinfo,
        "lstm_eg_completeness_gap_median": float(np.median(gapA)),
        "lstm_eg_completeness_gap_p95": float(np.percentile(gapA, 95)),
        "lstm_input_dim": int(lstm.lstm.input_size), "n_features": len(feat),
        "note": "EG gap is Monte-Carlo error of the sampled estimate (logit units)"}
    for nm, arr in [("shap_val_xgb", phi_x), ("shap_val_lstm_A", phi_l), ("shap_val_lstm_B", phi_l_b)]:
        np.save(os.path.join(a.out_dir, f"{nm}.npy"), arr)

    rank = global_ranking(phi_l_tr, phi_x_tr)
    log["global_top20_features"] = [feat[i] for i in rank[:20]]
    rel = rowwise_spearman(phi_l[:, rank[:20]], phi_l_b[:, rank[:20]])     # LSTM attribution repeatability
    np.save(os.path.join(a.out_dir, "lstm_attribution_repeatability_val.npy"), rel.astype(np.float32))
    log["lstm_attribution_repeatability"] = {"mean": float(rel.mean()), "median": float(np.median(rel)),
                                     "p10": float(np.percentile(rel, 10))}

    # ── agreement (label-free) ───────────────────────────────────────────────
    base_z = logit(np.array([ytr.mean()]))[0]
    A = {K: agreement(phi_l, phi_x, rank, K) for K in GRID["K"]}
    A_B = {K: agreement(phi_l_b, phi_x, rank, K) for K in GRID["K"]}
    Kv1 = 10

    def make_methods(pl, px):
        """All methods, given calibrated base probabilities. Each entry is
        (select_or_fit(idx) -> cfg, predict(cfg, idx) -> probs)."""
        zl, zx = logit(pl), logit(px)

        def sgae_select(idx, A_=A):
            best = None
            for K in GRID["K"]:
                sig = float(A_[K][idx].std())
                for w0, c, neg in itertools.product(GRID["w0"], GRID["c"], GRID["neg"]):
                    s_ = pr(y[idx], mix(sgae_w(A_[K][idx], sig, w0, c, neg), pl[idx], px[idx]))
                    if best is None or s_ > best[0] + 1e-9:
                        best = (s_, {"K": K, "w0": w0, "c": c, "neg": neg, "sigma_A": sig})
            return best[1]

        def sgae_predict(cfg, idx, A_=A, shape="tanh", neg=None, c=None):
            return mix(sgae_w(A_[cfg["K"]][idx], cfg["sigma_A"], cfg["w0"],
                              cfg["c"] if c is None else c, cfg["neg"] if neg is None else neg, shape=shape),
                       pl[idx], px[idx])

        def const_select(idx):
            return {"w": max(CONST_GRID, key=lambda w: pr(y[idx], mix(w, pl[idx], px[idx])))}

        def stack_fit(idx):
            m = LogisticRegression(C=1.0, max_iter=1000).fit(np.stack([zl[idx], zx[idx]], 1), y[idx])
            return {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}

        def conf_w(i):
            cl, cx = np.abs(zl[i] - base_z), np.abs(zx[i] - base_z)
            return cl / (cl + cx + EPS)

        M = {
            "xgb_only":        (None, lambda c, i: px[i]),
            "lstm_only":       (None, lambda c, i: pl[i]),
            "equal_weight":    (None, lambda c, i: mix(0.5, pl[i], px[i])),
            "static_v1_0.6":   (None, lambda c, i: mix(0.6, pl[i], px[i])),
            "tuned_constant":  (const_select, lambda c, i: mix(c["w"], pl[i], px[i])),
            "confidence_gate": (None, lambda c, i: mix(conf_w(i), pl[i], px[i])),
            "stacking":        (stack_fit, lambda c, i: sigm(c["intercept"] + np.stack([zl[i], zx[i]], 1) @ np.array(c["coef"]))),
            "learned_gate":    (lambda i: fit_gate(zl[i], zx[i], pl[i], px[i], y[i]),
                                lambda c, i: apply_gate(c, zl[i], zx[i], pl[i], px[i])),
            "learned_gate_plus_A": (lambda i: fit_gate(zl[i], zx[i], pl[i], px[i], y[i], A[Kv1][i]),
                                    lambda c, i: apply_gate(c, zl[i], zx[i], pl[i], px[i], A[Kv1][i])),
            "sgae_v1_weighting_rule": (lambda i: {"sigma_A": float(A[Kv1][i].std())},
                                       lambda c, i: mix(sgae_w(A[Kv1][i], c["sigma_A"], 0.5, 0.2, 0.6, 0.3, 0.7), pl[i], px[i])),
            "sgae_tuned":      (sgae_select, sgae_predict),
        }
        return M, sgae_predict

    folds = np.array_split(np.arange(n_va), N_FOLDS)          # contiguous time blocks
    eval_idx = np.concatenate(folds[1:])                      # blocks 2..5 (forward-time)
    all_idx = np.arange(n_va)

    # in-sample: calibrate and select on ALL validation (these are the frozen parameters)
    cal = {m: Platt().fit(p_raw[m], y) for m in ["lstm", "xgb"]}
    pl, px = cal["lstm"](p_raw["lstm"]), cal["xgb"](p_raw["xgb"])
    methods, sgae_predict = make_methods(pl, px)
    final_cfg, preds_in = {}, {}
    for name, (sel, pred) in methods.items():
        final_cfg[name] = sel(all_idx) if sel else None
        preds_in[name] = pred(final_cfg[name], all_idx)

    # forward-time: for block k, calibrate + select/fit on blocks < k only
    preds_fwd = {name: np.full(n_va, np.nan) for name in methods}
    fold_cfgs = []
    for k in range(1, N_FOLDS):
        past = np.concatenate(folds[:k])
        cal_k = {m: Platt().fit(p_raw[m][past], y[past]) for m in ["lstm", "xgb"]}
        M_k, _ = make_methods(cal_k["lstm"](p_raw["lstm"]), cal_k["xgb"](p_raw["xgb"]))
        cfgs = {}
        for name, (sel, pred) in M_k.items():
            cfg_k = sel(past) if sel else None
            preds_fwd[name][folds[k]] = pred(cfg_k, folds[k])
            if name in ("sgae_tuned", "tuned_constant"): cfgs[name] = cfg_k
        fold_cfgs.append({"block": k + 1, "n_past": int(len(past)), **cfgs})
        print(f"  forward block {k+1}/{N_FOLDS} done", flush=True)
    log["forward_time_selected_configs"] = fold_cfgs

    results = {}
    for name in methods:
        pf = preds_fwd[name][eval_idx]
        results[name] = {"val_pr_auc_in_sample": pr(y, preds_in[name]),
                         "val_pr_auc_forward_blocks2to5": pr(y[eval_idx], pf),
                         "val_roc_auc_forward_blocks2to5": roc(y[eval_idx], pf)}
        print(f"  {name:24s} PR-AUC in-sample {results[name]['val_pr_auc_in_sample']:.4f} | "
              f"forward-time {results[name]['val_pr_auc_forward_blocks2to5']:.4f}", flush=True)

    # ── SGAE ablations at the selected configuration ─────────────────────────
    cfg = final_cfg["sgae_tuned"]
    abl = {}
    perm = []
    prng = np.random.default_rng(99)
    for _ in range(N_PERM):
        Ap = {K: prng.permutation(A[K]) for K in GRID["K"]}
        perm.append(pr(y, sgae_predict(cfg, all_idx, Ap)))
    abl["permuted_A"] = {"mean": float(np.mean(perm)), "sd": float(np.std(perm, ddof=1)),
                         "min": float(np.min(perm)), "max": float(np.max(perm))}
    abl["constant_weight_at_selected_w0"] = pr(y, mix(cfg["w0"], pl, px))
    abl["negative_rule_only_(c=0,_keeps_w_neg)"] = pr(y, sgae_predict(cfg, all_idx, c=0.0))
    abl["linear_instead_of_tanh"] = pr(y, sgae_predict(cfg, all_idx, shape="linear"))
    abl["symmetric_no_negative_rule"] = pr(y, sgae_predict(cfg, all_idx, neg="symmetric"))
    Aloc = {K: agreement_local(phi_l, phi_x, K) for K in GRID["K"]}
    abl["local_topK"] = pr(y, sgae_predict({**cfg, "sigma_A": float(Aloc[cfg["K"]].std())}, all_idx, Aloc))
    abl["agreement_from_LSTM_run_B"] = pr(y, sgae_predict(cfg, all_idx, A_B))
    abl["K_sensitivity"] = {K: pr(y, sgae_predict({**cfg, "K": K, "sigma_A": float(A[K].std())}, all_idx))
                            for K in GRID["K"]}
    abl["note"] = ("EXPLORATORY: in-sample validation, same labels used to select the configuration; "
                   "not an unbiased estimate and not evidence of mechanism")
    log["sgae_ablations"] = abl
    wv = sgae_w(A[cfg["K"]], cfg["sigma_A"], cfg["w0"], cfg["c"], cfg["neg"])
    log["sgae_weight_distribution"] = {"mean": float(wv.mean()), "sd": float(wv.std()),
                                       "frac_negative_agreement": float((A[cfg["K"]] < 0).mean()),
                                       "agreement_mean": float(A[cfg["K"]].mean())}
    agreeB = np.corrcoef(A[cfg["K"]], A_B[cfg["K"]])[0, 1]
    log["agreement_stability_A_vs_B_pearson"] = float(agreeB)

    # ── attribution-repeatability probe (exploratory) + preregistered test ────
    # rel = agreement between two independent Expected-Gradients runs: REPEATABILITY of the
    # LSTM attribution computation, not faithfulness, and not by itself a causal test.
    cuts = np.quantile(rel, [1 / 3, 2 / 3]).tolist()             # fixed from validation
    groups = np.digitize(rel, cuts)
    probe = {}
    for gi, nm in enumerate(["low", "mid", "high"]):
        m = (groups == gi) & np.isin(np.arange(n_va), eval_idx)
        if y[m].sum() > 0:
            probe[nm] = {"n": int(m.sum()), "n_fraud": int(y[m].sum()),
                         "sgae_tuned_forward": pr(y[m], preds_fwd["sgae_tuned"][m]),
                         "tuned_constant_forward": pr(y[m], preds_fwd["tuned_constant"][m])}
    log["repeatability_probe_exploratory"] = probe
    prereg = {
        "name": "LSTM attribution repeatability and SGAE benefit (confirmatory, test set)",
        "tertile_cutoffs_from_validation": cuts,
        "statistic": "D = [PR-AUC(sgae_tuned) - PR-AUC(tuned_constant)] in HIGH tertile minus the same difference in LOW tertile",
        "inference": "paired bootstrap over card keys, 2,000 resamples, one-sided 95% interval",
        "H1": "D > 0 (SGAE helps more where LSTM attributions are more repeatable)",
        "interpretation_limit": "repeatability is not faithfulness; a positive D is consistent with, not proof of, the mechanism",
        "fixed_before_test": True}
    log["preregistered_mechanism_test"] = prereg

    # ── freeze ───────────────────────────────────────────────────────────────
    config = {"preregistered_mechanism_test": prereg,
              "base": {"seed": a.seed, "lstm_L": L, "platt": {m: cal[m].params() for m in cal},
                       "train_base_rate_logit": float(base_z)},
              "attribution": {"bg_train_rows": bg.tolist(), "eg_samples": a.eg_samples, "eg_seed_run_A": 2000 + a.seed,
                              "global_rank": rank.tolist()},
              "methods": final_cfg}
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    config["sha256"] = hashlib.sha256(blob).hexdigest()
    json.dump(config, open(os.path.join(a.out_dir, "step04_config.json"), "w"), default=str)
    np.savez_compressed(os.path.join(a.out_dir, "val_ensemble_preds.npz"),
                        **{f"in_{k}": v.astype(np.float32) for k, v in preds_in.items()},
                        **{f"fwd_{k}": v.astype(np.float32) for k, v in preds_fwd.items()},
                        agreement=A[cfg["K"]].astype(np.float32), y=y.astype(np.int8))
    log["results"] = results
    log["selected"] = {k: v for k, v in final_cfg.items() if k in ("sgae_tuned", "tuned_constant")}
    log["config_sha256"] = config["sha256"]
    log["secs"] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(a.out_dir, "step04_log.json"), "w"), indent=2, default=str)

    print("\nVALIDATION, forward-time PR-AUC on blocks 2-5 (fair comparison; test untouched):")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["val_pr_auc_forward_blocks2to5"]):
        print(f"  {k:24s} {v['val_pr_auc_forward_blocks2to5']:.4f}")
    print(f"Selected SGAE: {cfg}")
    print(f"Permuted-A control: {abl['permuted_A']['mean']:.4f} ± {abl['permuted_A']['sd']:.4f}"
          f" | SGAE in-sample {results['sgae_tuned']['val_pr_auc_in_sample']:.4f}")
    print(f"LSTM attribution repeatability (run A vs B, top-20 Spearman): mean {rel.mean():.3f}")
    print(f"Frozen config sha256: {config['sha256'][:16]}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
