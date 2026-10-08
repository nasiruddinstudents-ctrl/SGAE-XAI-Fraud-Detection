"""
GraphSAGE Pipeline for IEEE-CIS Fraud Detection — v4 (FINAL, SUBMISSION-READY)
EAAI Paper: SHAP-Guided Adaptive Ensemble Learning
Author: Mohammad Nasir Uddin

v4 fixes:
  #1  evaluate_chunked: removed dead bmask variable (was misleading)
  #2  Full-graph embedding: try/except OOM fallback to chunked embedding
  #3  CRITICAL: final model early stopping uses a held-out validation split
      (not the test set) — eliminates data leakage in held-out results

Outputs:
  gnn_results.csv        — CV + held-out metrics (6 metrics)
  gnn_shap_results.csv   — Kendall's W + faithfulness
  gnn_fold_results.csv   — per-fold breakdown
  gnn_test_probs.csv     — held-out probs for DeLong/McNemar
  gnn_feature_bridge.csv — embedding dim → original feature mapping
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch.nn import BatchNorm1d, Dropout, Linear
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (roc_auc_score, f1_score,
                             average_precision_score,
                             precision_score, recall_score,
                             matthews_corrcoef)
from sklearn.utils import resample
from collections import defaultdict
import itertools
import shap
import warnings
import time

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

SEED    = 42
EPOCHS  = 100
PATIENCE = 10
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 1. LOAD & MERGE ──────────────────────────────────────────────────────────
print("\n[1/8] Loading data...")
t0 = time.time()

df_trans = pd.read_csv('/workspace/train_transaction.csv')
df_id    = pd.read_csv('/workspace/train_identity.csv')
df       = df_trans.merge(df_id, on='TransactionID', how='left')

print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}  |  Fraud: {df['isFraud'].mean()*100:.2f}%")
print(f"  Load time: {time.time()-t0:.1f}s")

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n[2/8] Feature engineering...")

labels    = df['isFraud'].values.astype(int)
trans_ids = df['TransactionID'].values

drop_cols = ['TransactionID', 'TransactionDT', 'isFraud']
df_feat   = df.drop(columns=[c for c in drop_cols if c in df.columns])

for col in df_feat.select_dtypes(include='object').columns:
    df_feat[col] = df_feat[col].fillna('MISSING')
    le = LabelEncoder()
    df_feat[col] = le.fit_transform(df_feat[col].astype(str))

for col in df_feat.select_dtypes(include=[np.number]).columns:
    df_feat[col] = df_feat[col].fillna(df_feat[col].median())

feature_names = df_feat.columns.tolist()
print(f"  Features: {len(feature_names)}")

scaler = MinMaxScaler()
X      = scaler.fit_transform(df_feat.values).astype(np.float32)
print(f"  Node feature matrix: {X.shape}")

# ── 3. GRAPH CONSTRUCTION ────────────────────────────────────────────────────
print("\n[3/8] Building transaction-transaction graph (vectorized)...")
t1 = time.time()

src_list, dst_list = [], []

def add_edges_from_groups(group_dict, max_group=50, max_neighbors=10):
    for gid, trans in group_dict.items():
        n = len(trans)
        if 2 <= n <= max_group:
            for a, b in itertools.combinations(trans[:max_neighbors], 2):
                src_list.extend([a, b])
                dst_list.extend([b, a])

# Card-based edges
card_col = 'card1' if 'card1' in df.columns else None
card_id  = df[card_col].fillna(-1).astype(int).values if card_col else np.arange(len(df))
card_to_trans = defaultdict(list)
for i, cid in enumerate(card_id):
    card_to_trans[cid].append(i)
add_edges_from_groups(card_to_trans, max_group=50, max_neighbors=10)
print(f"  After card edges: {len(src_list):,}")

# Merchant-based edges (addr1 + ProductCD composite key)
addr_col = 'addr1'     if 'addr1'     in df.columns else None
prod_col = 'ProductCD' if 'ProductCD' in df.columns else None
if addr_col and prod_col:
    merch_key = (df[addr_col].fillna(-1).astype(str) + '_' +
                 df[prod_col].fillna('UNK').astype(str)).values
elif 'P_emaildomain' in df.columns:
    merch_key = df['P_emaildomain'].fillna('UNK').values
else:
    merch_key = None

if merch_key is not None:
    merch_to_trans = defaultdict(list)
    for i, mid in enumerate(merch_key):
        merch_to_trans[mid].append(i)
    add_edges_from_groups(merch_to_trans, max_group=30, max_neighbors=5)
    print(f"  After merchant edges: {len(src_list):,}")

edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
print(f"  Total edges: {edge_index.shape[1]:,}")
print(f"  Graph build time: {time.time()-t1:.1f}s")

# ── 4. STRATIFIED SPLITS ─────────────────────────────────────────────────────
print("\n[4/8] Stratified splits...")

# 80% train+val / 20% test
train_val_idx, test_idx_arr = train_test_split(
    np.arange(len(X)), test_size=0.2, stratify=labels, random_state=SEED)

# FIX #3: For the final model, split train_val into 90/10 train/val
# This validation set is used for early stopping — test set is NEVER touched
# until final evaluation
final_train_idx, final_val_idx = train_test_split(
    train_val_idx, test_size=0.1,
    stratify=labels[train_val_idx], random_state=SEED)

# For 5-fold CV, use all of train_val
train_idx_arr = train_val_idx

# Masks
def make_mask(idx, n):
    m = torch.zeros(n, dtype=torch.bool)
    m[idx] = True
    return m

train_mask      = make_mask(train_idx_arr,  len(X))
test_mask       = make_mask(test_idx_arr,   len(X))
final_train_mask = make_mask(final_train_idx, len(X))
final_val_mask   = make_mask(final_val_idx,   len(X))

x_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(labels, dtype=torch.long)

data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

print(f"  CV train+val : {len(train_idx_arr):,}  ({labels[train_idx_arr].mean()*100:.2f}% fraud)")
print(f"  Final train  : {len(final_train_idx):,}  ({labels[final_train_idx].mean()*100:.2f}% fraud)")
print(f"  Final val    : {len(final_val_idx):,}  ({labels[final_val_idx].mean()*100:.2f}% fraud)")
print(f"  Held-out test: {len(test_idx_arr):,}  ({labels[test_idx_arr].mean()*100:.2f}% fraud)")

fraud_rate = labels[final_train_idx].mean()
pos_weight = float((1 - fraud_rate) / (fraud_rate + 1e-8))
print(f"  Pos weight: {pos_weight:.1f}")

# ── 5. MODEL ─────────────────────────────────────────────────────────────────
class GraphSAGEFraud(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, emb_dim=64, dropout=0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.bn1   = BatchNorm1d(hidden)
        self.conv2 = SAGEConv(hidden, emb_dim)
        self.bn2   = BatchNorm1d(emb_dim)
        self.drop  = Dropout(dropout)
        self.fc1   = Linear(emb_dim, 256)
        self.fc2   = Linear(256, 128)
        self.fc3   = Linear(128, 64)
        self.out   = Linear(64, 2)

    def get_embedding(self, x, edge_index):
        x = self.drop(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.drop(F.relu(self.bn2(self.conv2(x, edge_index))))
        return x

    def mlp_head(self, emb):
        x = self.drop(F.relu(self.fc1(emb)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(F.relu(self.fc3(x)))
        return self.out(x)

    def forward(self, x, edge_index):
        return self.mlp_head(self.get_embedding(x, edge_index))

# ── 6. TRAIN / EVAL FUNCTIONS ────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, pw):
    model.train()
    total_loss = 0
    weight = torch.tensor([1.0, pw], device=DEVICE)
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index)
        mask = batch.train_mask[:batch.batch_size]
        if mask.sum() == 0:
            continue
        loss = F.cross_entropy(
            out[:batch.batch_size][mask],
            batch.y[:batch.batch_size][mask],
            weight=weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

# FIX #1: evaluate_chunked — removed dead bmask variable, clean logic
@torch.inference_mode()
def evaluate_chunked(model, data_obj, mask, chunk_size=50000):
    """Memory-safe evaluation using NeighborLoader inference."""
    model.eval()
    d = data_obj.to(DEVICE)
    inf_loader = NeighborLoader(
        d, num_neighbors=[10, 5],
        batch_size=chunk_size,
        input_nodes=mask,
        shuffle=False
    )
    all_probs, all_labels = [], []
    for batch in inf_loader:
        out    = model(batch.x, batch.edge_index)
        probs  = F.softmax(out[:batch.batch_size], dim=1)[:, 1].cpu().numpy()
        ylbls  = batch.y[:batch.batch_size].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(ylbls)
    probs      = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    if len(np.unique(labels_arr)) < 2:
        return None
    preds = (probs >= 0.5).astype(int)
    return {
        'auc_roc':   roc_auc_score(labels_arr, probs),
        'f1':        f1_score(labels_arr, preds, zero_division=0),
        'pr_auc':    average_precision_score(labels_arr, probs),
        'precision': precision_score(labels_arr, preds, zero_division=0),
        'recall':    recall_score(labels_arr, preds, zero_division=0),
        'mcc':       matthews_corrcoef(labels_arr, preds),
        'probs':     probs,
        'labels':    labels_arr
    }

@torch.inference_mode()
def evaluate_full(model, data_obj, mask):
    """Full-graph eval with automatic OOM fallback to chunked."""
    model.eval()
    try:
        d      = data_obj.to(DEVICE)
        out    = model(d.x, d.edge_index)
        probs  = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        y_true = d.y[mask].cpu().numpy()
        if len(np.unique(y_true)) < 2:
            return None
        return {
            'auc_roc':   roc_auc_score(y_true, probs),
            'f1':        f1_score(y_true, preds, zero_division=0),
            'pr_auc':    average_precision_score(y_true, probs),
            'precision': precision_score(y_true, preds, zero_division=0),
            'recall':    recall_score(y_true, preds, zero_division=0),
            'mcc':       matthews_corrcoef(y_true, preds),
            'probs':     probs,
            'labels':    y_true
        }
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("    OOM — falling back to chunked eval")
            torch.cuda.empty_cache()
            return evaluate_chunked(model, data_obj, mask)
        raise

# ── 7. 5-FOLD CV ─────────────────────────────────────────────────────────────
print("\n[5/8] 5-Fold Stratified CV...")

skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_results = []
metrics_list = ['auc_roc', 'f1', 'pr_auc', 'precision', 'recall', 'mcc']

for fold, (tr, val) in enumerate(skf.split(train_idx_arr, labels[train_idx_arr])):
    print(f"\n  ── Fold {fold+1}/5 ──")
    t_fold = time.time()

    fold_train_mask = make_mask(train_idx_arr[tr],  len(X))
    fold_val_mask   = make_mask(train_idx_arr[val], len(X))

    fold_data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor,
                     train_mask=fold_train_mask)

    loader = NeighborLoader(
        fold_data, num_neighbors=[10, 5],
        batch_size=2048, input_nodes=fold_train_mask, shuffle=True)

    model     = GraphSAGEFraud(in_channels=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=False)

    best_pr_auc  = 0
    best_state   = None
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer, pos_weight)
        res  = evaluate_full(model, fold_data, fold_val_mask)
        if res:
            scheduler.step(res['pr_auc'])
            if epoch % 10 == 0 or epoch == 1:
                print(f"    Ep {epoch:3d} | loss={loss:.3f} | "
                      f"AUC={res['auc_roc']:.4f} | F1={res['f1']:.4f} | "
                      f"PR-AUC={res['pr_auc']:.4f}")
            if res['pr_auc'] > best_pr_auc:
                best_pr_auc  = res['pr_auc']
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    res = evaluate_full(model, fold_data, fold_val_mask)
    fold_results.append(res)
    print(f"  Fold {fold+1} → AUC={res['auc_roc']:.4f} | F1={res['f1']:.4f} | "
          f"PR-AUC={res['pr_auc']:.4f} | MCC={res['mcc']:.4f} | "
          f"{time.time()-t_fold:.0f}s")

    del model
    torch.cuda.empty_cache()

cv_means = {m: np.mean([r[m] for r in fold_results]) for m in metrics_list}
cv_stds  = {m: np.std( [r[m] for r in fold_results]) for m in metrics_list}

print("\n  ══ 5-Fold CV SUMMARY ══")
for m in metrics_list:
    print(f"  {m:12s}: {cv_means[m]:.4f} ± {cv_stds[m]:.4f}")

# ── 8. FINAL HELD-OUT MODEL (FIX #3 — no test set leakage) ──────────────────
print("\n[6/8] Final model (train=72%, val=8%, test=20%)...")
print("  Early stopping on VALIDATION set — test set untouched until end")

# FIX #3: use final_train_mask for training, final_val_mask for early stopping
final_data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor,
                  train_mask=final_train_mask)

final_loader = NeighborLoader(
    final_data, num_neighbors=[10, 5],
    batch_size=2048, input_nodes=final_train_mask, shuffle=True)

final_model  = GraphSAGEFraud(in_channels=X.shape[1]).to(DEVICE)
optimizer    = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5, verbose=False)

best_pr_auc  = 0
best_state   = None
patience_cnt = 0

for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch(final_model, final_loader, optimizer, pos_weight)
    # Early stopping on VALIDATION split (not test)
    val_res = evaluate_full(final_model, final_data, final_val_mask)
    if val_res:
        scheduler.step(val_res['pr_auc'])
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | loss={loss:.3f} | "
                  f"val AUC={val_res['auc_roc']:.4f} | "
                  f"val PR-AUC={val_res['pr_auc']:.4f}")
        if val_res['pr_auc'] > best_pr_auc:
            best_pr_auc  = val_res['pr_auc']
            best_state   = {k: v.clone() for k, v in final_model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

final_model.load_state_dict(best_state)

# Evaluate on test set ONCE — only here
print("\n  Evaluating on held-out test set (single evaluation)...")
held_out = evaluate_full(final_model, data, test_mask)

print(f"  Held-out → AUC={held_out['auc_roc']:.4f} | F1={held_out['f1']:.4f} | "
      f"PR-AUC={held_out['pr_auc']:.4f} | MCC={held_out['mcc']:.4f}")

pred_df = pd.DataFrame({
    'TransactionID': trans_ids[test_idx_arr],
    'true_label':    held_out['labels'],
    'gnn_prob':      held_out['probs']
})
pred_df.to_csv('/workspace/gnn_test_probs.csv', index=False)
print("  Saved: gnn_test_probs.csv")

# ── 9. SHAP ON MLP HEAD ──────────────────────────────────────────────────────
print("\n[7/8] SHAP — MLP head on frozen GNN embeddings...")
t_shap = time.time()

final_model.eval()
data_gpu = data.to(DEVICE)

# FIX #2: OOM-safe full-graph embedding computation
print("  Computing GNN embeddings for all nodes...")
try:
    with torch.inference_mode():
        all_emb = final_model.get_embedding(
            data_gpu.x, data_gpu.edge_index).cpu().numpy()
    print(f"  Full-graph embedding OK: {all_emb.shape}")
except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        print("  OOM on full-graph embedding — computing in chunks via NeighborLoader...")
        torch.cuda.empty_cache()
        # Compute embeddings chunk by chunk using NeighborLoader
        all_nodes_mask = torch.ones(len(X), dtype=torch.bool)
        emb_loader = NeighborLoader(
            data_gpu, num_neighbors=[10, 5],
            batch_size=50000, input_nodes=all_nodes_mask, shuffle=False)
        emb_list   = []
        idx_list   = []
        with torch.inference_mode():
            for batch in emb_loader:
                emb = final_model.get_embedding(
                    batch.x, batch.edge_index)[:batch.batch_size]
                emb_list.append(emb.cpu().numpy())
                idx_list.append(batch.n_id[:batch.batch_size].cpu().numpy())
        # Re-order to original node order
        all_emb = np.zeros((len(X), emb_list[0].shape[1]), dtype=np.float32)
        for emb_chunk, idx_chunk in zip(emb_list, idx_list):
            all_emb[idx_chunk] = emb_chunk
        print(f"  Chunked embedding OK: {all_emb.shape}")
    else:
        raise

# Sample 200 test nodes + 100 background training nodes
test_node_idx = np.where(test_mask.numpy())[0]
np.random.seed(SEED)
shap_sample = np.random.choice(test_node_idx, size=200, replace=False)
bg_idx      = np.random.choice(np.where(final_train_mask.numpy())[0],
                                size=100, replace=False)
x_shap = all_emb[shap_sample]
x_bg   = all_emb[bg_idx]

def mlp_predict(emb_np):
    emb_t = torch.tensor(emb_np, dtype=torch.float, device=DEVICE)
    with torch.inference_mode():
        out   = final_model.mlp_head(emb_t)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    return probs

print("  Running KernelExplainer (nsamples=1000, ~30-60 min)...")
explainer   = shap.KernelExplainer(mlp_predict, x_bg)
shap_values = explainer.shap_values(x_shap, nsamples=1000, silent=False)
print(f"  SHAP done in {(time.time()-t_shap)/60:.1f} min")

# Top-10 embedding dims
mean_shap = np.abs(shap_values).mean(axis=0)
top10_idx = np.argsort(mean_shap)[::-1][:10]
print("\n  Top 10 embedding dims by mean |SHAP|:")
for i in top10_idx:
    print(f"    emb_dim_{i}: {mean_shap[i]:.5f}")

# Bridge: embedding dim → original features via correlation
print("\n  Building embedding → feature bridge...")
bridge_idx  = np.random.choice(len(X), size=5000, replace=False)
X_bridge    = X[bridge_idx]
emb_bridge  = all_emb[bridge_idx]

bridge_rows = []
for dim in top10_idx:
    emb_col = emb_bridge[:, dim]
    corrs   = np.array([np.corrcoef(X_bridge[:, f], emb_col)[0, 1]
                        for f in range(X_bridge.shape[1])])
    top5_idx   = np.argsort(np.abs(corrs))[::-1][:5]
    top5_names = [feature_names[f] for f in top5_idx]
    top5_corrs = [float(corrs[f]) for f in top5_idx]
    bridge_rows.append({
        'emb_dim':         int(dim),
        'shap_importance': float(mean_shap[dim]),
        'top_feat_1': top5_names[0], 'corr_1': top5_corrs[0],
        'top_feat_2': top5_names[1], 'corr_2': top5_corrs[1],
        'top_feat_3': top5_names[2], 'corr_3': top5_corrs[2],
        'top_feat_4': top5_names[3], 'corr_4': top5_corrs[3],
        'top_feat_5': top5_names[4], 'corr_5': top5_corrs[4],
    })
    print(f"    emb_dim_{dim} (SHAP={mean_shap[dim]:.4f}) → "
          f"{top5_names[0]} ({top5_corrs[0]:+.3f}), "
          f"{top5_names[1]} ({top5_corrs[1]:+.3f}), "
          f"{top5_names[2]} ({top5_corrs[2]:+.3f})")

pd.DataFrame(bridge_rows).to_csv('/workspace/gnn_feature_bridge.csv', index=False)

# ── 10. FAITHFULNESS ─────────────────────────────────────────────────────────
print("\n  Faithfulness (sufficiency + comprehensiveness, k=5,10,15)...")

def faithfulness(shap_vals, embeddings, node_indices, ks=[5, 10, 15]):
    results    = {}
    base_probs = mlp_predict(embeddings[node_indices[:50]])
    for k in ks:
        suff_list, comp_list = [], []
        for i in range(50):
            sv    = shap_vals[i]
            top_k = np.argsort(np.abs(sv))[::-1][:k]
            e_suff = embeddings[node_indices[i]].copy()
            keep   = np.zeros(len(e_suff), dtype=bool)
            keep[top_k] = True
            e_suff[~keep] = 0.0
            p_suff = mlp_predict(e_suff[np.newaxis])[0]
            e_comp        = embeddings[node_indices[i]].copy()
            e_comp[top_k] = 0.0
            p_comp = mlp_predict(e_comp[np.newaxis])[0]
            orig = base_probs[i]
            suff_list.append(abs(p_suff - orig))
            comp_list.append(abs(orig - p_comp))
        results[f'suff_k{k}'] = float(np.mean(suff_list))
        results[f'comp_k{k}'] = float(np.mean(comp_list))
    return results

faith = faithfulness(shap_values, all_emb, shap_sample)
for k in [5, 10, 15]:
    print(f"  k={k:2d} | Sufficiency={faith[f'suff_k{k}']:.4f}  "
          f"Comprehensiveness={faith[f'comp_k{k}']:.4f}")

# ── 11. KENDALL'S W ───────────────────────────────────────────────────────────
print("\n  Kendall's W (30 bootstraps, n=200, top-30 dims)...")

def kendalls_w(rankings):
    n, k = rankings.shape
    S    = np.sum((rankings.sum(axis=0) - rankings.sum() / k) ** 2)
    return float(12 * S / (n ** 2 * (k ** 3 - k)))

bootstrap_rankings = []
np.random.seed(SEED)
for b in range(30):
    boot  = resample(np.arange(len(shap_values)), n_samples=200, random_state=b)
    sv_b  = shap_values[boot]
    ma    = np.abs(sv_b).mean(axis=0)
    top30 = np.argsort(ma)[::-1][:30]
    rank  = np.argsort(np.argsort(-ma[top30])) + 1
    bootstrap_rankings.append(rank)

kw = kendalls_w(np.array(bootstrap_rankings))
if   kw >= 0.9: interp = "Near-perfect stability (suitable for SR 11-7)"
elif kw >= 0.7: interp = "Strong stability"
elif kw >= 0.5: interp = "Moderate stability"
else:           interp = "Below reliability threshold"
print(f"  Kendall's W: {kw:.4f} — {interp}")

# ── 12. SAVE ALL RESULTS ─────────────────────────────────────────────────────
print("\n[8/8] Saving results...")

pd.DataFrame({
    'Model':             ['GNN-GraphSAGE'],
    'CV_AUC_mean':       [cv_means['auc_roc']],
    'CV_AUC_std':        [cv_stds['auc_roc']],
    'CV_F1_mean':        [cv_means['f1']],
    'CV_F1_std':         [cv_stds['f1']],
    'CV_PRAUC_mean':     [cv_means['pr_auc']],
    'CV_PRAUC_std':      [cv_stds['pr_auc']],
    'CV_Prec_mean':      [cv_means['precision']],
    'CV_Prec_std':       [cv_stds['precision']],
    'CV_Rec_mean':       [cv_means['recall']],
    'CV_Rec_std':        [cv_stds['recall']],
    'CV_MCC_mean':       [cv_means['mcc']],
    'CV_MCC_std':        [cv_stds['mcc']],
    'HO_AUC':            [held_out['auc_roc']],
    'HO_F1':             [held_out['f1']],
    'HO_PRAUC':          [held_out['pr_auc']],
    'HO_Precision':      [held_out['precision']],
    'HO_Recall':         [held_out['recall']],
    'HO_MCC':            [held_out['mcc']],
    'Dataset':           ['IEEE-CIS'],
    'Notes':             ['Class-weighted loss; no SMOTE; SHAP on MLP head; '
                          'early stopping on val split (not test); patience=10 per epoch'],
}).to_csv('/workspace/gnn_results.csv', index=False)

pd.DataFrame({
    'Model':      ['GNN-GraphSAGE'],
    'Kendalls_W': [kw],
    'Stability':  [interp],
    'Suff_k5':    [faith['suff_k5']],  'Comp_k5':  [faith['comp_k5']],
    'Suff_k10':   [faith['suff_k10']], 'Comp_k10': [faith['comp_k10']],
    'Suff_k15':   [faith['suff_k15']], 'Comp_k15': [faith['comp_k15']],
}).to_csv('/workspace/gnn_shap_results.csv', index=False)

pd.DataFrame([
    {**{m: r[m] for m in metrics_list}, 'fold': i+1}
    for i, r in enumerate(fold_results)
]).to_csv('/workspace/gnn_fold_results.csv', index=False)

print("\n" + "="*65)
print("COMPLETE. Output files in /workspace/:")
print("  gnn_results.csv        — main metrics")
print("  gnn_shap_results.csv   — Kendall's W + faithfulness")
print("  gnn_fold_results.csv   — per-fold")
print("  gnn_test_probs.csv     — probs for DeLong/McNemar")
print("  gnn_feature_bridge.csv — emb dim → feature mapping")
print("="*65)
print(f"\nFINAL SUMMARY:")
print(f"  CV  AUC-ROC : {cv_means['auc_roc']:.4f} ± {cv_stds['auc_roc']:.4f}")
print(f"  CV  F1      : {cv_means['f1']:.4f} ± {cv_stds['f1']:.4f}")
print(f"  CV  PR-AUC  : {cv_means['pr_auc']:.4f} ± {cv_stds['pr_auc']:.4f}")
print(f"  CV  MCC     : {cv_means['mcc']:.4f} ± {cv_stds['mcc']:.4f}")
print(f"  HO  AUC-ROC : {held_out['auc_roc']:.4f}")
print(f"  HO  F1      : {held_out['f1']:.4f}")
print(f"  HO  PR-AUC  : {held_out['pr_auc']:.4f}")
print(f"  Kendall's W : {kw:.4f}  [{interp}]")
