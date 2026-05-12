"""
CNN Binary Classification: treat raw 2D spectrum (wavelength × time) as a grayscale image.
Predicts HB < 10 (label=0) vs HB >= 10 (label=1).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os, re, unicodedata
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix,
                              roc_curve, precision_recall_curve, average_precision_score)
from scipy.ndimage import zoom

# ── Parameters ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER = os.path.join(BASE_DIR, 'mua')
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WAV_LEN      = 300
TIME_LEN     = 150
HB_THRESHOLD = 10.0
BATCH_SIZE   = 8
EPOCHS       = 300
LR           = 1e-4
WEIGHT_DECAY = 1e-4
N_FOLDS      = 5
TRAIN_RATIO  = 4
MODEL_PATH   = os.path.join(OUT_DIR, 'cnn_binary_model_0511.pth')


# ── Utilities ─────────────────────────────────────────────────
def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


def load_image(mua_path):
    data = np.loadtxt(mua_path, delimiter='\t')
    spec = data[:, 1:]
    nw = spec.shape[0]
    if nw < WAV_LEN:
        spec = np.pad(spec, ((0, WAV_LEN - nw), (0, 0)), mode='edge')
    spec = spec[:WAV_LEN, :]
    nt = spec.shape[1]
    if nt != TIME_LEN:
        spec = zoom(spec, (1.0, TIME_LEN / nt), order=1)
    mu, sigma = spec.mean(), spec.std() + 1e-8
    return ((spec - mu) / sigma).astype(np.float32)


def load_dataset(base_dir, mua_folder):
    images, labels, patient_ids = [], [], []
    excel_cache = {}
    files = sorted([f for f in os.listdir(mua_folder) if f.endswith('.txt')])
    for f_name in tqdm(files, desc='Loading images'):
        norm = unicodedata.normalize('NFKC', f_name).lower()
        d_m  = re.search(r'(\d{4})_(\d{2})_(\d{2})', norm)
        sb_m = re.search(r'(morning|afternoon|evening)_([a-z]*)(\d+)', norm)
        if not (d_m and sb_m): continue
        date_str = f"{d_m.group(1)}{d_m.group(2)}{d_m.group(3)}"
        shift    = {'morning': '早', 'afternoon': '午', 'evening': '晚'}[sb_m.group(1)]
        bed      = _clean_bed(f"{sb_m.group(2)}{sb_m.group(3)}")
        if date_str not in excel_cache:
            p = os.path.join(base_dir, f"{date_str}_dialysis_table_export.xlsx")
            if os.path.exists(p):
                df = pd.read_excel(p, engine='openpyxl')
                df.columns = df.columns.str.strip()
                df['Bed_C']   = df['DialysisBed'].apply(_clean_bed)
                df['Shift_C'] = df['Shift'].apply(
                    lambda x: '早' if '早' in str(x) else ('午' if '午' in str(x) else '晚'))
                excel_cache[date_str] = df
            else:
                excel_cache[date_str] = None
        df = excel_cache[date_str]
        if df is None: continue
        row = df[(df['Bed_C'] == bed) & (df['Shift_C'] == shift)]
        if row.empty or pd.isnull(row.iloc[0]['ClinicHb']): continue
        images.append(load_image(os.path.join(mua_folder, f_name)))
        labels.append(float(row.iloc[0]['ClinicHb']))
        patient_ids.append(f"{bed}_{shift}")
    X = np.array(images, dtype=np.float32)[:, np.newaxis, :, :]
    y = np.array(labels, dtype=np.float32)
    print(f"\n>>> Loaded {len(y)} samples / {len(set(patient_ids))} patients")
    return X, y, patient_ids


# ── Model ─────────────────────────────────────────────────────
class HbCNNBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.head(self.features(x)).squeeze(-1)


# ── 4-panel eval figure ───────────────────────────────────────
def draw_4panel(y_true, probs, preds, fig_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    ax.plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'r--', lw=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color='steelblue')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve'); ax.legend(); ax.grid(True, linestyle=':', alpha=0.5)

    ax = axes[0, 1]
    cm = confusion_matrix(y_true, preds)
    ax.imshow(cm, cmap='Blues')
    classes = ['HB<10\n(Neg)', 'HB≥10\n(Pos)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion Matrix')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=14, fontweight='bold')

    ax = axes[1, 0]
    prec_c, rec_c, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    ax.plot(rec_c, prec_c, lw=2, color='darkorange', label=f'AP = {ap:.3f}')
    ax.axhline(y_true.mean(), color='r', linestyle='--', lw=1,
               label=f'Baseline = {y_true.mean():.3f}')
    ax.fill_between(rec_c, prec_c, alpha=0.08, color='darkorange')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('PR Curve'); ax.legend(); ax.grid(True, linestyle=':', alpha=0.5)

    ax = axes[1, 1]
    ax.hist(probs[y_true == 0], bins=20, alpha=0.6, color='tomato',
            label='True Neg (HB<10)', density=True)
    ax.hist(probs[y_true == 1], bins=20, alpha=0.6, color='steelblue',
            label='True Pos (HB≥10)', density=True)
    ax.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold=0.5')
    ax.set_xlabel('Predicted P(HB≥10)'); ax.set_ylabel('Density')
    ax.set_title('Score Distribution'); ax.legend(); ax.grid(True, linestyle=':', alpha=0.5)

    acc  = accuracy_score(y_true, preds)
    f1   = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    plt.suptitle(
        f'CNN Binary Classification  Acc={acc:.3f}  AUC={auc:.3f}  '
        f'F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}',
        fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f">>> Figure saved: {fig_path}")


# ── Training ──────────────────────────────────────────────────
def train():
    print(f"\n>>> CNN Binary Classification  (Device: {DEVICE})")

    X, y_hb, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(y_hb) == 0: return

    y = (y_hb >= HB_THRESHOLD).astype(np.float32)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    print(f"    Positive (HB≥10): {int(n_pos)}  Negative (HB<10): {int(n_neg)}")

    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    pid_map = defaultdict(list)
    for i, pid in enumerate(patient_ids): pid_map[pid].append(i)
    rng = np.random.default_rng(42)
    pids = list(pid_map.keys()); rng.shuffle(pids)
    n_te      = max(1, len(pids) // (TRAIN_RATIO + 1))
    test_pats = set(pids[-n_te:])
    tr_pats   = pids[:-n_te]

    tr_idx   = [i for pid in tr_pats  for i in pid_map[pid]]
    te_idx   = [i for pid in test_pats for i in pid_map[pid]]
    print(f"    Train: {len(tr_idx)}  Test: {len(te_idx)}")

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    tr_pids    = [patient_ids[i] for i in tr_idx]
    unique_tr  = np.unique(tr_pids)

    actual_folds = min(N_FOLDS, len(unique_tr))
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n  {actual_folds}-Fold CV")
    for fold, (f_tr_idx, f_val_idx) in enumerate(kf.split(unique_tr)):
        f_tr_pats  = set(unique_tr[f_tr_idx])
        f_val_pats = set(unique_tr[f_val_idx])
        f_tr_loc   = [i for i, p in enumerate(tr_pids) if p in f_tr_pats]
        f_val_loc  = [i for i, p in enumerate(tr_pids) if p in f_val_pats]
        if len(f_tr_loc) < BATCH_SIZE or len(f_val_loc) < 2: continue
        if len(np.unique(y_tr[f_tr_loc])) < 2:
            print(f"  Fold {fold+1}: skip (single class)"); continue

        f_train_dl = DataLoader(
            TensorDataset(torch.tensor(X_tr[f_tr_loc]), torch.tensor(y_tr[f_tr_loc])),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        f_val_dl   = DataLoader(
            TensorDataset(torch.tensor(X_tr[f_val_loc]), torch.tensor(y_tr[f_val_loc])),
            batch_size=BATCH_SIZE, shuffle=False)

        model = HbCNNBinary().to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_val = float('inf')
        fold_path = os.path.join(OUT_DIR, f'cnn_bin_fold{fold+1}_tmp.pth')

        for epoch in tqdm(range(EPOCHS), desc=f"  Fold {fold+1}", leave=False):
            model.train()
            for x, yb in f_train_dl:
                x, yb = x.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); criterion(model(x), yb).backward(); opt.step()
            model.eval(); vl = 0
            with torch.no_grad():
                for x, yb in f_val_dl:
                    vl += criterion(model(x.to(DEVICE)), yb.to(DEVICE)).item()
            avg_val = vl / len(f_val_dl)
            if avg_val < best_val:
                best_val = avg_val; torch.save(model.state_dict(), fold_path)

        model.load_state_dict(torch.load(fold_path, weights_only=True)); model.eval()
        logits, y_true_f = [], []
        with torch.no_grad():
            for x, yb in f_val_dl:
                logits.extend(model(x.to(DEVICE)).cpu().numpy())
                y_true_f.extend(yb.numpy())
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_arr = np.array(y_true_f).astype(int)
        acc = accuracy_score(y_arr, preds)
        auc = roc_auc_score(y_arr, probs) if len(np.unique(y_arr)) > 1 else float('nan')
        f1  = f1_score(y_arr, preds, zero_division=0)
        fold_results.append({'acc': acc, 'auc': auc, 'f1': f1})
        print(f"  Fold {fold+1}: Acc={acc:.3f} | AUC={auc:.3f} | F1={f1:.3f}")

    if fold_results:
        print(f"\n  CV avg:")
        for key in ('acc', 'auc', 'f1'):
            vals = [r[key] for r in fold_results if not np.isnan(r[key])]
            if vals: print(f"    {key.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  Final model training")
    bs = min(BATCH_SIZE, len(X_tr))
    final_dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                          batch_size=bs, shuffle=True, drop_last=(len(X_tr) > bs))
    test_dl  = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)),
                          batch_size=BATCH_SIZE, shuffle=False)

    model = HbCNNBinary().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss = float('inf')

    for epoch in tqdm(range(EPOCHS), desc="  Final", unit="ep"):
        model.train()
        for x, yb in final_dl:
            x, yb = x.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); criterion(model(x), yb).backward(); opt.step()
        model.eval(); tel = 0
        with torch.no_grad():
            for x, yb in test_dl:
                tel += criterion(model(x.to(DEVICE)), yb.to(DEVICE)).item()
        if tel < best_loss:
            best_loss = tel
            torch.save({'model': model.state_dict(),
                        'test_patient_ids': list(test_pats),
                        'wav_len': WAV_LEN, 'time_len': TIME_LEN}, MODEL_PATH)

    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    logits, y_true_all = [], []
    with torch.no_grad():
        for x, yb in test_dl:
            logits.extend(model(x.to(DEVICE)).cpu().numpy())
            y_true_all.extend(yb.numpy())

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    y_arr = np.array(y_true_all).astype(int)
    acc   = accuracy_score(y_arr, preds)
    auc   = roc_auc_score(y_arr, probs) if len(np.unique(y_arr)) > 1 else float('nan')
    f1    = f1_score(y_arr, preds, zero_division=0)
    prec  = precision_score(y_arr, preds, zero_division=0)
    rec   = recall_score(y_arr, preds, zero_division=0)
    cm    = confusion_matrix(y_arr, preds)

    print(f"\n{'='*50}")
    print(f"  CNN Binary — Test Set Result  (N={len(y_arr)})")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  Confusion:\n    TN={cm[0,0]}  FP={cm[0,1]}\n    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"{'='*50}")
    print(f">>> Model saved: {MODEL_PATH}")

    if len(np.unique(y_arr)) > 1:
        draw_4panel(y_arr, probs, preds,
                    os.path.join(OUT_DIR, 'eval_cnn_binary_0511.png'))

    for i in range(actual_folds):
        p = os.path.join(OUT_DIR, f'cnn_bin_fold{i+1}_tmp.pth')
        if os.path.exists(p): os.remove(p)


if __name__ == '__main__':
    train()
