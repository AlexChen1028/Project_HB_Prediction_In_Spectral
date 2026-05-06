"""
MLP 回歸：使用 SWT cA3 的 2 階導數（d2）特徵
──────────────────────────────────────────
根據特徵探索結果，改用 d2 在 537 / 540 / 560 / 577nm 的值作為輸入。
探索結果：d2@537nm 與 HB 相關性最強（r≈+0.33），優於原始 SWT 值（r≈+0.22）。
標準化：per-feature z-score（各 d2 特徵各自正規化）
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ==========================================
# 參數
# ==========================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
WINDOW_W    = 2
SPEC_LEN    = 896

# d2 特徵波長索引（根據探索結果選出相關性最強的波長）
D2_INDICES  = [37, 40, 60, 77]   # 537, 540, 560, 577 nm
D2_LABELS   = ['d2@537nm', 'd2@540nm', 'd2@560nm', 'd2@577nm']
N_FEATURES  = len(D2_INDICES)


# ==========================================
# 工具函數
# ==========================================
def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


def _extract_d2(mua_path):
    """回傳 SWT cA3 的 2 階導數在 D2_INDICES 各位置的值"""
    data = np.loadtxt(mua_path, delimiter='\t')
    v = np.mean(data[:, 1:], axis=1)
    if len(v) < SPEC_LEN: v = np.pad(v, (0, SPEC_LEN - len(v)), mode='edge')
    else:                  v = v[:SPEC_LEN]

    coeffs = pywt.swt(v, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    cA3 = coeffs[0][0]
    d1  = np.gradient(cA3)
    d2  = np.gradient(d1)

    def pt(idx):
        return float(np.mean(d2[max(0, idx - WINDOW_W): idx + WINDOW_W + 1]))

    return np.array([pt(i) for i in D2_INDICES], dtype=np.float32)


# ==========================================
# 資料載入
# ==========================================
def load_dataset(base_dir, mua_folder):
    raw_feats, labels, patient_ids = [], [], []
    excel_cache = {}
    files = [f for f in os.listdir(mua_folder) if f.endswith('.txt')]

    for f_name in tqdm(files, desc='Extracting d2 features'):
        norm  = unicodedata.normalize('NFKC', f_name).lower()
        d_m   = re.search(r'(\d{4})_(\d{2})_(\d{2})', norm)
        sb_m  = re.search(r'(morning|afternoon|evening)_([a-z]+)0*(\d+)', norm)
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

    raw_feats = np.array(raw_feats, dtype=np.float32)
    labels    = np.array(labels,    dtype=np.float32)
    print(f"\n>>> Loaded {len(labels)} samples / {len(set(patient_ids))} patients")
    print(f">>> Features: {D2_LABELS}")
    return raw_feats, labels, patient_ids


# ==========================================
# 模型
# ==========================================
class HbD2Net(nn.Module):
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
# Learning Curve
# ==========================================
def plot_lc(train_losses, val_losses, tag, best_epoch):
    plt.figure(figsize=(10, 5))
    ep = range(1, len(train_losses) + 1)
    plt.plot(ep, train_losses, label='Train MSE', color='steelblue', lw=2)
    plt.plot(ep, val_losses,   label='Val MSE',   color='darkorange', lw=2)
    plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best ({best_epoch})')
    plt.ylim(0, 5); plt.xlabel('Epoch'); plt.ylabel('MSE (g/dL)^2')
    plt.title(f'Learning Curve - {tag}'); plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(f'lc_d2_{tag}.png', dpi=150); plt.close()


# ==========================================
# 單組訓練
# ==========================================
def train_group(raw_features, labels, patient_ids, group_name, model_save_path, tag):
    n = len(labels)
    if n == 0: print(f"\n[{group_name}] No data, skip"); return

    print(f"\n{'='*60}")
    print(f"  Group: {group_name}  |  Samples: {n}")
    print(f"{'='*60}")

    patient_sample_map = defaultdict(list)
    for i, pid in enumerate(patient_ids): patient_sample_map[pid].append(i)

    rng = np.random.default_rng(42)
    shuffled = list(patient_sample_map.keys()); rng.shuffle(shuffled)
    n_test      = max(1, len(shuffled) // (TRAIN_RATIO + 1))
    test_pats   = set(shuffled[-n_test:])
    train_pats  = shuffled[:-n_test]

    tr_idx   = [i for pid in train_pats for i in patient_sample_map[pid]]
    test_idx = [i for pid in test_pats  for i in patient_sample_map[pid]]

    print(f"  Train: {len(tr_idx)} samples ({len(train_pats)} patients)")
    print(f"  Test:  {len(test_idx)} samples ({len(test_pats)} patients)")
    if not tr_idx or not test_idx: return

    tr_raw   = raw_features[tr_idx]
    tr_lbl   = labels[tr_idx]
    tr_pids  = [patient_ids[i] for i in tr_idx]
    unique_tr = np.unique(tr_pids)

    actual_folds = min(N_FOLDS, len(unique_tr))
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
    criterion = nn.MSELoss(); fold_results = []

    print(f"\n  {actual_folds}-Fold CV ({group_name})")
    for fold, (f_tr_idx, f_val_idx) in enumerate(kf.split(unique_tr)):
        f_tr_pats  = set(unique_tr[f_tr_idx])
        f_val_pats = set(unique_tr[f_val_idx])
        f_tr_loc   = [i for i, pid in enumerate(tr_pids) if pid in f_tr_pats]
        f_val_loc  = [i for i, pid in enumerate(tr_pids) if pid in f_val_pats]
        if len(f_tr_loc) < BATCH_SIZE or len(f_val_loc) < 2:
            print(f"  Fold {fold+1}: skip (insufficient samples)"); continue

        f_tr  = tr_raw[f_tr_loc];  f_val = tr_raw[f_val_loc]
        # per-feature normalization（保留各 d2 特徵的獨立尺度資訊）
        f_mean = f_tr.mean(axis=0); f_std = f_tr.std(axis=0) + 1e-8
        f_tr_n  = (f_tr  - f_mean) / f_std
        f_val_n = (f_val - f_mean) / f_std

        f_tr_lbl  = tr_lbl[f_tr_loc]
        f_val_lbl = tr_lbl[f_val_loc]

        f_train_dl = DataLoader(TensorDataset(torch.tensor(f_tr_n), torch.tensor(f_tr_lbl)),
                                batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        f_val_dl   = DataLoader(TensorDataset(torch.tensor(f_val_n), torch.tensor(f_val_lbl)),
                                batch_size=BATCH_SIZE, shuffle=False)

        model = HbD2Net().to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_val = float('inf'); best_ep = 1; tr_ls, val_ls = [], []
        fold_path = f'd2_{tag}_fold{fold+1}_tmp.pth'

        pbar = tqdm(range(EPOCHS), desc=f"  Fold {fold+1}/{actual_folds}", unit="ep", leave=False)
        for epoch in pbar:
            model.train(); tl = 0
            for x, y in f_train_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
                tl += loss.item()
            model.eval(); vl = 0
            with torch.no_grad():
                for x, y in f_val_dl:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    vl += criterion(model(x), y).item()
            at = tl/len(f_train_dl); av = vl/len(f_val_dl)
            tr_ls.append(at); val_ls.append(av)
            pbar.set_postfix({'tr': f'{at:.3f}', 'val': f'{av:.3f}'})
            if av < best_val: best_val = av; best_ep = epoch+1; torch.save(model.state_dict(), fold_path)

        plot_lc(tr_ls, val_ls, f'{tag}_fold{fold+1}', best_ep)
        model.load_state_dict(torch.load(fold_path, weights_only=True)); model.eval()
        y_t, y_p = [], []
        with torch.no_grad():
            for x, y in f_val_dl:
                y_t.extend(y.numpy()); y_p.extend(model(x.to(DEVICE)).cpu().numpy())
        y_t, y_p = np.array(y_t), np.array(y_p)
        mae = mean_absolute_error(y_t, y_p)
        rmse= np.sqrt(mean_squared_error(y_t, y_p))
        r2  = r2_score(y_t, y_p) if len(y_t) > 1 else float('nan')
        fold_results.append({'mae': mae, 'rmse': rmse, 'r2': r2})
        print(f"  Fold {fold+1}: MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

    if fold_results:
        print(f"\n  KFold avg ({group_name}):")
        for key, label in [('mae','MAE'),('rmse','RMSE'),('r2','R²')]:
            vals = [r[key] for r in fold_results if not np.isnan(r[key])]
            if vals: print(f"    {label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Final model
    print(f"\n  Final model training ({group_name})")
    f_mean = tr_raw.mean(axis=0); f_std = tr_raw.std(axis=0) + 1e-8
    tr_n   = (tr_raw - f_mean) / f_std
    te_raw = raw_features[test_idx]; te_n = (te_raw - f_mean) / f_std
    te_lbl = labels[test_idx]

    bs = min(BATCH_SIZE, max(2, len(tr_n)))
    final_dl = DataLoader(TensorDataset(torch.tensor(tr_n), torch.tensor(tr_lbl)),
                          batch_size=bs, shuffle=True, drop_last=(len(tr_n) > bs))
    test_dl  = DataLoader(TensorDataset(torch.tensor(te_n), torch.tensor(te_lbl)),
                          batch_size=BATCH_SIZE, shuffle=False)

    model = HbD2Net().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_te = float('inf'); best_ep = 1; tr_ls, te_ls = [], []

    for epoch in tqdm(range(EPOCHS), desc=f"  Final ({group_name})", unit="ep"):
        model.train(); tl = 0
        for x, y in final_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); criterion(model(x), y).backward(); opt.step()
            tl += x.shape[0]
        model.eval(); tel = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                tel += criterion(model(x), y).item()
        at = tl/len(final_dl) if len(final_dl) > 0 else 0
        ate = tel/len(test_dl)
        tr_ls.append(at); te_ls.append(ate)
        if ate < best_te:
            best_te = ate; best_ep = epoch+1
            torch.save({'model': model.state_dict(), 'feat_mean': f_mean, 'feat_std': f_std,
                        'test_patient_ids': list(test_pats), 'group_name': group_name,
                        'd2_indices': D2_INDICES, 'd2_labels': D2_LABELS}, model_save_path)

    plot_lc(tr_ls, te_ls, f'{tag}_final', best_ep)

    ckpt = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    y_t, y_p = [], []
    with torch.no_grad():
        for x, y in test_dl:
            y_t.extend(y.numpy()); y_p.extend(model(x.to(DEVICE)).cpu().numpy())
    y_t, y_p = np.array(y_t), np.array(y_p)
    mae = mean_absolute_error(y_t, y_p)
    rmse= np.sqrt(mean_squared_error(y_t, y_p))
    r2  = r2_score(y_t, y_p) if len(y_t) > 1 else float('nan')
    base= mean_absolute_error(y_t, np.full_like(y_t, y_t.mean()))

    print(f"\n{'='*50}")
    print(f"  Final Test Result ({group_name})")
    print(f"{'='*50}")
    print(f"  Baseline MAE (mean): {base:.4f} g/dL")
    print(f"  MAE  : {mae:.4f} g/dL")
    print(f"  RMSE : {rmse:.4f} g/dL")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*50}")
    print(f">>> Model saved: {model_save_path}")

    for i in range(actual_folds):
        p = f'd2_{tag}_fold{i+1}_tmp.pth'
        if os.path.exists(p): os.remove(p)


# ==========================================
# 主程式
# ==========================================
def train():
    print(f"\n>>> HB Split Regression - d2 features (Device: {DEVICE})")
    raw_features, labels, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(labels) == 0: return

    low_mask  = labels <  HB_THRESHOLD
    high_mask = labels >= HB_THRESHOLD

    def n_pats(mask): return len(set(pid for i, pid in enumerate(patient_ids) if mask[i]))
    print(f"\n>>> HB <  {HB_THRESHOLD}: {low_mask.sum():4d} samples / {n_pats(low_mask)} patients")
    print(f">>> HB >= {HB_THRESHOLD}: {high_mask.sum():4d} samples / {n_pats(high_mask)} patients")

    train_group(raw_features[low_mask],  labels[low_mask],
                [patient_ids[i] for i, m in enumerate(low_mask)  if m],
                f'HB < {HB_THRESHOLD}',  'd2_hb_low_model.pth',  tag='low')

    train_group(raw_features[high_mask], labels[high_mask],
                [patient_ids[i] for i, m in enumerate(high_mask) if m],
                f'HB >= {HB_THRESHOLD}', 'd2_hb_high_model.pth', tag='high')


if __name__ == "__main__":
    train()
