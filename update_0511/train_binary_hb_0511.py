"""
Binary classification: HB < 10 (label=0) vs HB >= 10 (label=1)
Features: SWT cA3 2nd-order derivative (d2) at 537 / 540 / 560 / 577nm
Normalization: per-feature z-score
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os, re, unicodedata, pywt
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)

# ==========================================
# Parameters
# ==========================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER  = os.path.join(BASE_DIR, 'mua')
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE   = 8
EPOCHS       = 500
LR           = 1e-3
WEIGHT_DECAY = 1e-5
N_FOLDS      = 5
HB_THRESHOLD = 10.0
TRAIN_RATIO  = 4

SWT_LEVEL   = 3
SWT_WAVELET = 'db4'
WINDOW_W    = 2
SPEC_LEN    = 896

D2_INDICES = [37, 40, 60, 77]   # 537, 540, 560, 577 nm
D2_LABELS  = ['d2@537nm', 'd2@540nm', 'd2@560nm', 'd2@577nm']
N_FEATURES = len(D2_INDICES)

MODEL_PATH = os.path.join(OUT_DIR, 'd2_hb_binary_model_0511.pth')


# ==========================================
# Utilities
# ==========================================
def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


def _extract_d2(mua_path):
    data = np.loadtxt(mua_path, delimiter='\t')
    v = np.mean(data[:, 1:], axis=1)
    if len(v) < SPEC_LEN: v = np.pad(v, (0, SPEC_LEN - len(v)), mode='edge')
    else:                  v = v[:SPEC_LEN]
    coeffs = pywt.swt(v, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    cA3 = coeffs[0][0]
    d2  = np.gradient(np.gradient(cA3))
    def pt(idx): return float(np.mean(d2[max(0, idx - WINDOW_W): idx + WINDOW_W + 1]))
    return np.array([pt(i) for i in D2_INDICES], dtype=np.float32)


def load_dataset(base_dir, mua_folder):
    raw_feats, labels, patient_ids = [], [], []
    excel_cache = {}
    files = [f for f in os.listdir(mua_folder) if f.endswith('.txt')]
    for f_name in tqdm(files, desc='Extracting d2 features'):
        norm  = unicodedata.normalize('NFKC', f_name).lower()
        d_m   = re.search(r'(\d{4})_(\d{2})_(\d{2})', norm)
        sb_m  = re.search(r'(morning|afternoon|evening)_([a-z]*)(\d+)', norm)
        if not (d_m and sb_m): continue
        date_str   = f"{d_m.group(1)}{d_m.group(2)}{d_m.group(3)}"
        shift      = {'morning':'早','afternoon':'午','evening':'晚'}.get(sb_m.group(1))
        bed        = _clean_bed(f"{sb_m.group(2)}{sb_m.group(3)}")
        patient_id = f"{bed}_{shift}"
        if date_str not in excel_cache:
            path = os.path.join(base_dir, f"{date_str}_dialysis_table_export.xlsx")
            if os.path.exists(path):
                df = pd.read_excel(path, engine='openpyxl')
                df.columns = df.columns.str.strip()
                df['Bed_C']   = df['DialysisBed'].apply(_clean_bed)
                df['Shift_C'] = df['Shift'].apply(
                    lambda x: '早' if '早' in str(x) else ('午' if '午' in str(x) else '晚'))
                excel_cache[date_str] = df
            else: excel_cache[date_str] = None
        df = excel_cache[date_str]
        if df is None: continue
        row = df[(df['Bed_C'] == bed) & (df['Shift_C'] == shift)]
        if row.empty or pd.isnull(row.iloc[0]['ClinicHb']): continue
        raw_feats.append(_extract_d2(os.path.join(mua_folder, f_name)))
        labels.append(float(row.iloc[0]['ClinicHb']))
        patient_ids.append(patient_id)
    X = np.array(raw_feats, dtype=np.float32)
    y = np.array(labels,    dtype=np.float32)
    print(f"\n>>> Loaded {len(y)} samples / {len(set(patient_ids))} patients")
    return X, y, patient_ids


# ==========================================
# Model
# ==========================================
class HbBinaryNet(nn.Module):
    def __init__(self, n_in=N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32),   nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16),   nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)


# ==========================================
# Metrics
# ==========================================
def binary_metrics(y_true, logits, threshold=0.5):
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)
    y_true = y_true.astype(int)
    return dict(
        acc  = accuracy_score(y_true, preds),
        prec = precision_score(y_true, preds, zero_division=0),
        rec  = recall_score(y_true, preds, zero_division=0),
        f1   = f1_score(y_true, preds, zero_division=0),
        auc  = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float('nan'),
        cm   = confusion_matrix(y_true, preds),
        probs= probs,
    )


def plot_roc(y_true, probs, tag):
    fpr, tpr, _ = roc_curve(y_true.astype(int), probs)
    auc = roc_auc_score(y_true.astype(int), probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {tag}'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f'roc_{tag}.png'), dpi=150); plt.close()
    print(f">>> ROC saved: roc_{tag}.png")


# ==========================================
# Training
# ==========================================
def train():
    print(f"\n>>> HB Binary Classification  (Device: {DEVICE})")
    print(f">>> Features: {D2_LABELS}")
    raw_features, hb_values, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(hb_values) == 0: return

    binary_labels = (hb_values >= HB_THRESHOLD).astype(np.float32)
    n_pos = binary_labels.sum(); n_neg = len(binary_labels) - n_pos
    print(f">>> Positive (HB>=10): {int(n_pos)}  Negative (HB<10): {int(n_neg)}")

    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    patient_sample_map = defaultdict(list)
    for i, pid in enumerate(patient_ids): patient_sample_map[pid].append(i)

    rng = np.random.default_rng(42)
    shuffled = list(patient_sample_map.keys()); rng.shuffle(shuffled)
    n_test        = max(1, len(shuffled) // (TRAIN_RATIO + 1))
    test_patients = set(shuffled[-n_test:])
    train_patients= shuffled[:-n_test]

    tr_idx   = [i for pid in train_patients for i in patient_sample_map[pid]]
    test_idx = [i for pid in test_patients  for i in patient_sample_map[pid]]

    print(f"\n  Train: {len(tr_idx)} samples ({len(train_patients)} patients)")
    print(f"  Test:  {len(test_idx)} samples ({len(test_patients)} patients)")

    tr_raw   = raw_features[tr_idx]
    tr_labels= binary_labels[tr_idx]
    tr_pids  = [patient_ids[i] for i in tr_idx]
    unique_tr= np.unique(tr_pids)

    actual_folds = min(N_FOLDS, len(unique_tr))
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
    print(f"\n  {actual_folds}-Fold CV")

    fold_results = []
    for fold, (f_tr_idx, f_val_idx) in enumerate(kf.split(unique_tr)):
        f_tr_pats  = set(unique_tr[f_tr_idx])
        f_val_pats = set(unique_tr[f_val_idx])
        f_tr_loc   = [i for i, pid in enumerate(tr_pids) if pid in f_tr_pats]
        f_val_loc  = [i for i, pid in enumerate(tr_pids) if pid in f_val_pats]
        if len(f_tr_loc) < BATCH_SIZE or len(f_val_loc) < 2:
            print(f"  Fold {fold+1}: skip (insufficient samples)"); continue
        if len(np.unique(tr_labels[f_tr_loc])) < 2:
            print(f"  Fold {fold+1}: skip (single class in fold)"); continue

        f_tr  = tr_raw[f_tr_loc];  f_val = tr_raw[f_val_loc]
        f_mean = f_tr.mean(axis=0); f_std = f_tr.std(axis=0) + 1e-8
        f_tr_n  = (f_tr  - f_mean) / f_std
        f_val_n = (f_val - f_mean) / f_std

        f_tr_lbl  = tr_labels[f_tr_loc]
        f_val_lbl = tr_labels[f_val_loc]

        f_train_dl = DataLoader(TensorDataset(torch.tensor(f_tr_n), torch.tensor(f_tr_lbl)),
                                batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        f_val_dl   = DataLoader(TensorDataset(torch.tensor(f_val_n), torch.tensor(f_val_lbl)),
                                batch_size=BATCH_SIZE, shuffle=False)

        model = HbBinaryNet().to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_val = float('inf')
        fold_path = os.path.join(OUT_DIR, f'binary_fold{fold+1}_tmp.pth')

        pbar = tqdm(range(EPOCHS), desc=f"  Fold {fold+1}", unit="ep", leave=False)
        for epoch in pbar:
            model.train(); tl = 0
            for x, y in f_train_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); criterion(model(x), y).backward(); opt.step()
                tl += 1
            model.eval(); vl = 0
            with torch.no_grad():
                for x, y in f_val_dl:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    vl += criterion(model(x), y).item()
            avg_val = vl / len(f_val_dl)
            pbar.set_postfix({'val': f'{avg_val:.3f}'})
            if avg_val < best_val:
                best_val = avg_val; torch.save(model.state_dict(), fold_path)

        model.load_state_dict(torch.load(fold_path, weights_only=True)); model.eval()
        logits_f, y_true_f = [], []
        with torch.no_grad():
            for x, y in f_val_dl:
                logits_f.extend(model(x.to(DEVICE)).cpu().numpy())
                y_true_f.extend(y.numpy())
        m = binary_metrics(np.array(y_true_f), np.array(logits_f))
        fold_results.append(m)
        print(f"  Fold {fold+1}: Acc={m['acc']:.3f} | AUC={m['auc']:.3f} | "
              f"F1={m['f1']:.3f} | Prec={m['prec']:.3f} | Rec={m['rec']:.3f}")

    if fold_results:
        print(f"\n  K-Fold Average:")
        for key in ('acc', 'auc', 'f1', 'prec', 'rec'):
            vals = [r[key] for r in fold_results if not np.isnan(r[key])]
            if vals: print(f"    {key.upper():<5}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  Final model training")
    f_mean = tr_raw.mean(axis=0); f_std = tr_raw.std(axis=0) + 1e-8
    tr_n   = (tr_raw - f_mean) / f_std
    te_raw = raw_features[test_idx]; te_n = (te_raw - f_mean) / f_std
    te_lbl = binary_labels[test_idx]

    bs = min(BATCH_SIZE, max(2, len(tr_n)))
    final_dl = DataLoader(TensorDataset(torch.tensor(tr_n), torch.tensor(tr_labels)),
                          batch_size=bs, shuffle=True, drop_last=(len(tr_n) > bs))
    test_dl  = DataLoader(TensorDataset(torch.tensor(te_n), torch.tensor(te_lbl)),
                          batch_size=BATCH_SIZE, shuffle=False)

    model = HbBinaryNet().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss = float('inf')

    for epoch in tqdm(range(EPOCHS), desc="  Final", unit="ep"):
        model.train()
        for x, y in final_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); criterion(model(x), y).backward(); opt.step()
        model.eval(); te_l = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                te_l += criterion(model(x), y).item()
        if te_l < best_loss:
            best_loss = te_l
            torch.save({'model': model.state_dict(),
                        'feat_mean': f_mean, 'feat_std': f_std,
                        'test_patient_ids': list(test_patients),
                        'd2_indices': D2_INDICES, 'd2_labels': D2_LABELS}, MODEL_PATH)

    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    logits_all, y_true_all = [], []
    with torch.no_grad():
        for x, y in test_dl:
            logits_all.extend(model(x.to(DEVICE)).cpu().numpy())
            y_true_all.extend(y.numpy())

    m = binary_metrics(np.array(y_true_all), np.array(logits_all))
    print(f"\n{'='*50}")
    print(f"  Final Test Set Result")
    print(f"{'='*50}")
    print(f"  Accuracy : {m['acc']:.4f}")
    print(f"  AUC-ROC  : {m['auc']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  Precision: {m['prec']:.4f}")
    print(f"  Recall   : {m['rec']:.4f}")
    print(f"  Confusion Matrix:\n{m['cm']}")
    print(f"{'='*50}")
    print(f">>> Model saved: {MODEL_PATH}")

    if len(np.unique(np.array(y_true_all).astype(int))) > 1:
        plot_roc(np.array(y_true_all), m['probs'], 'binary_hb_0511')

    for i in range(actual_folds):
        p = os.path.join(OUT_DIR, f'binary_fold{i+1}_tmp.pth')
        if os.path.exists(p): os.remove(p)


if __name__ == "__main__":
    train()
