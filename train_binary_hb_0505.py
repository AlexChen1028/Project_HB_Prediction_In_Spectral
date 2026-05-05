"""
二元分類：預測 HB < 10 (label=0) 或 HB >= 10 (label=1)
特徵：SWT 第 SWT_LEVEL 階近似係數，取 v540 / v560 / v577
標準化：三個 channel 共用同一 mean/std（保留相對關係）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import re
import unicodedata
import pywt
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)

# ==========================================
# 1. 參數設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER  = os.path.join(BASE_DIR, 'mua')
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
IDX_540 = 40; IDX_560 = 60; IDX_577 = 77; WINDOW_W = 2

MODEL_PATH = 'swt_hb_binary_model.pth'


# ==========================================
# 2. 工具函數
# ==========================================
def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


def _extract_spectral(mua_path):
    data = np.loadtxt(mua_path, delimiter='\t')
    v = np.mean(data[:, 1:], axis=1)
    if len(v) < 896: v = np.pad(v, (0, 896 - len(v)), mode='edge')
    else:            v = v[:896]
    coeffs = pywt.swt(v, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    cA = coeffs[0][0]
    def pt(idx): return float(np.mean(cA[max(0, idx - WINDOW_W): idx + WINDOW_W + 1]))
    return np.array([pt(IDX_540), pt(IDX_560), pt(IDX_577)], dtype=np.float32)


def load_dataset(base_dir, mua_folder):
    raw_feats, labels, patient_ids = [], [], []
    excel_cache = {}
    files = [f for f in os.listdir(mua_folder) if f.endswith('.txt')]
    for f_name in tqdm(files, desc="特徵提取中"):
        norm = unicodedata.normalize('NFKC', f_name).lower()
        d_m  = re.search(r'(\d{4})_(\d{2})_(\d{2})', norm)
        sb_m = re.search(r'(morning|afternoon|evening)_([a-z]+)0*(\d+)', norm)
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
                    lambda x: "早" if "早" in str(x) else ("午" if "午" in str(x) else "晚"))
                excel_cache[date_str] = df
            else: excel_cache[date_str] = None
        df = excel_cache[date_str]
        if df is None: continue
        row = df[(df['Bed_C'] == bed) & (df['Shift_C'] == shift)]
        if row.empty or pd.isnull(row.iloc[0]['ClinicHb']): continue
        raw_feats.append(_extract_spectral(os.path.join(mua_folder, f_name)))
        labels.append(float(row.iloc[0]['ClinicHb']))
        patient_ids.append(patient_id)
    raw_feats = np.array(raw_feats, dtype=np.float32)
    labels    = np.array(labels,    dtype=np.float32)
    print(f"\n>>> 配對成功: {len(labels)} 筆 / {len(set(patient_ids))} 位病人")
    return raw_feats, labels, patient_ids


# ==========================================
# 3. 模型
# ==========================================
class HbBinaryNet(nn.Module):
    def __init__(self, n_in=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32),   nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16),   nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1)     # BCEWithLogitsLoss 不需手動加 sigmoid
        )
    def forward(self, x): return self.net(x).squeeze(-1)


# ==========================================
# 4. 評估指標
# ==========================================
def binary_metrics(y_true, logits, threshold=0.5):
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()
    preds  = (probs >= threshold).astype(int)
    y_true = y_true.astype(int)
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    auc  = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float('nan')
    cm   = confusion_matrix(y_true, preds)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc, cm=cm, probs=probs)


def plot_roc(y_true, probs, tag):
    fpr, tpr, _ = roc_curve(y_true.astype(int), probs)
    auc = roc_auc_score(y_true.astype(int), probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {tag}'); plt.legend()
    plt.tight_layout(); plt.savefig(f'roc_{tag}.png', dpi=150); plt.close()
    print(f">>> ROC 圖儲存至 roc_{tag}.png")


# ==========================================
# 5. 主訓練函數
# ==========================================
def train():
    print(f"\n>>> HB 二元分類 (Device: {DEVICE})")
    print(f">>> SWT {SWT_LEVEL} 階，label: 0=HB<{HB_THRESHOLD}  1=HB>={HB_THRESHOLD}")
    raw_features, hb_values, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(hb_values) == 0: return

    binary_labels = (hb_values >= HB_THRESHOLD).astype(np.float32)
    n_pos = binary_labels.sum(); n_neg = len(binary_labels) - n_pos
    print(f">>> 正例 (HB>=10): {int(n_pos)}  負例 (HB<10): {int(n_neg)}")

    # 正負樣本不平衡時調整權重
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    patient_sample_map = defaultdict(list)
    for i, pid in enumerate(patient_ids): patient_sample_map[pid].append(i)

    rng = np.random.default_rng(42)
    shuffled_patients = list(patient_sample_map.keys())
    rng.shuffle(shuffled_patients)

    n_test_patients     = max(1, len(shuffled_patients) // (TRAIN_RATIO + 1))
    test_patients       = set(shuffled_patients[-n_test_patients:])
    train_patients_list = shuffled_patients[:-n_test_patients]

    tr_idx   = [i for pid in train_patients_list for i in patient_sample_map[pid]]
    test_idx = [i for pid in test_patients        for i in patient_sample_map[pid]]

    print(f"\n  Train: {len(tr_idx)} 筆 ({len(train_patients_list)} 人)")
    print(f"  Test:  {len(test_idx)} 筆 ({len(test_patients)} 人)")

    tr_raw   = raw_features[tr_idx]
    tr_labels= binary_labels[tr_idx]
    tr_pids  = [patient_ids[i] for i in tr_idx]
    unique_tr_patients = np.unique(tr_pids)

    actual_folds = min(N_FOLDS, len(unique_tr_patients))
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
    print(f"\n  {actual_folds}-Fold CV")

    fold_results = []
    for fold, (f_tr_pat_idx, f_val_pat_idx) in enumerate(kf.split(unique_tr_patients)):
        f_tr_pats  = set(unique_tr_patients[f_tr_pat_idx])
        f_val_pats = set(unique_tr_patients[f_val_pat_idx])
        f_tr_local  = [i for i, pid in enumerate(tr_pids) if pid in f_tr_pats]
        f_val_local = [i for i, pid in enumerate(tr_pids) if pid in f_val_pats]
        if len(f_tr_local) < BATCH_SIZE or len(f_val_local) < 2:
            print(f"  Fold {fold+1}: 跳過（樣本不足）"); continue

        f_tr_raw  = tr_raw[f_tr_local];  f_val_raw = tr_raw[f_val_local]
        feat_mean = f_tr_raw.mean();     feat_std  = f_tr_raw.std() + 1e-8
        f_tr_norm  = (f_tr_raw  - feat_mean) / feat_std
        f_val_norm = (f_val_raw - feat_mean) / feat_std

        f_tr_lbl  = tr_labels[f_tr_local]
        f_val_lbl = tr_labels[f_val_local]

        f_train_loader = DataLoader(
            TensorDataset(torch.tensor(f_tr_norm), torch.tensor(f_tr_lbl)),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        f_val_loader = DataLoader(
            TensorDataset(torch.tensor(f_val_norm), torch.tensor(f_val_lbl)),
            batch_size=BATCH_SIZE, shuffle=False)

        model     = HbBinaryNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_val_loss = float('inf')
        fold_model_path = f'binary_fold{fold+1}_tmp.pth'

        pbar = tqdm(range(EPOCHS), desc=f"  Fold {fold+1}", unit="ep", leave=False)
        for epoch in pbar:
            model.train(); tr_l = 0
            for x, y in f_train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y); loss.backward(); optimizer.step()
                tr_l += loss.item()
            model.eval(); val_l = 0
            with torch.no_grad():
                for x, y in f_val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    val_l += criterion(model(x), y).item()
            avg_tr = tr_l / len(f_train_loader)
            avg_val= val_l / len(f_val_loader)
            pbar.set_postfix({'tr': f'{avg_tr:.3f}', 'val': f'{avg_val:.3f}'})
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), fold_model_path)

        model.load_state_dict(torch.load(fold_model_path, weights_only=True))
        model.eval()
        logits_f, y_true_f = [], []
        with torch.no_grad():
            for x, y in f_val_loader:
                logits_f.extend(model(x.to(DEVICE)).cpu().numpy())
                y_true_f.extend(y.numpy())
        m = binary_metrics(np.array(y_true_f), np.array(logits_f))
        fold_results.append(m)
        print(f"  Fold {fold+1}: Acc={m['acc']:.3f} | AUC={m['auc']:.3f} | "
              f"F1={m['f1']:.3f} | Prec={m['prec']:.3f} | Rec={m['rec']:.3f}")

    if fold_results:
        print(f"\n  K-Fold 平均:")
        for key in ('acc', 'auc', 'f1', 'prec', 'rec'):
            vals = [r[key] for r in fold_results if not np.isnan(r[key])]
            if vals: print(f"    {key.upper():<5}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ── 最終模型 ──
    print(f"\n  最終模型訓練")
    feat_mean = tr_raw.mean(); feat_std = tr_raw.std() + 1e-8
    tr_norm   = (tr_raw - feat_mean) / feat_std
    test_raw  = raw_features[test_idx]
    test_norm = (test_raw - feat_mean) / feat_std
    test_lbl  = binary_labels[test_idx]

    bs = min(BATCH_SIZE, max(2, len(tr_norm)))
    final_loader = DataLoader(
        TensorDataset(torch.tensor(tr_norm), torch.tensor(tr_labels)),
        batch_size=bs, shuffle=True, drop_last=(len(tr_norm) > bs))
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_norm), torch.tensor(test_lbl)),
        batch_size=BATCH_SIZE, shuffle=False)

    model     = HbBinaryNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss = float('inf')

    for epoch in tqdm(range(EPOCHS), desc="  Final", unit="ep"):
        model.train()
        for x, y in final_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward(); optimizer.step()
        model.eval(); te_l = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                te_l += criterion(model(x), y).item()
        if te_l < best_loss:
            best_loss = te_l
            torch.save({'model': model.state_dict(),
                        'feat_mean': feat_mean, 'feat_std': feat_std,
                        'test_patient_ids': list(test_patients)}, MODEL_PATH)

    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logits_all, y_true_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits_all.extend(model(x.to(DEVICE)).cpu().numpy())
            y_true_all.extend(y.numpy())

    m = binary_metrics(np.array(y_true_all), np.array(logits_all))
    print(f"\n{'='*50}")
    print(f"  最終 Test Set 結果（二元分類）")
    print(f"{'='*50}")
    print(f"  Accuracy : {m['acc']:.4f}")
    print(f"  AUC-ROC  : {m['auc']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  Precision: {m['prec']:.4f}")
    print(f"  Recall   : {m['rec']:.4f}")
    print(f"  Confusion Matrix:\n{m['cm']}")
    print(f"{'='*50}")
    print(f">>> 模型已存為 {MODEL_PATH}")

    if len(np.unique(np.array(y_true_all).astype(int))) > 1:
        plot_roc(np.array(y_true_all), m['probs'], 'binary_hb')

    for i in range(actual_folds):
        p = f'binary_fold{i+1}_tmp.pth'
        if os.path.exists(p): os.remove(p)


if __name__ == "__main__":
    train()
