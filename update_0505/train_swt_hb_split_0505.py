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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

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
TRAIN_RATIO  = 4      # train : test = 4 : 1  → test = n_patients // 5

SWT_LEVEL    = 3      # SWT 階數（目前使用第 3 階，cA3）
SWT_WAVELET  = 'db4'

# 波長索引 (起點 500nm = Index 0，步距 1nm)
IDX_540 = 40   # 540nm
IDX_560 = 60   # 560nm
IDX_577 = 77   # 577nm
WINDOW_W = 2   # ±2 = 5 點視窗

# 可選人口統計學特徵 (若 Excel 有對應欄位才會啟用)
# 例如: DEMO_COLS = ['Sex', 'Age']  → 模型輸入變為 3 + 2 = 5 維
# Sex 欄位需可轉為數值 (0/1 or M/F)，Age 為數值
DEMO_COLS = []   # 留空 = 只用光譜特徵


# ==========================================
# 2. 工具函數
# ==========================================
def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


def _encode_sex(val):
    """將性別欄位轉成 0/1；無法辨識時回傳 np.nan"""
    s = str(val).strip().upper()
    if s in ('M', 'MALE', '男', '0'): return 0.0
    if s in ('F', 'FEMALE', '女', '1'): return 1.0
    return np.nan


def _extract_spectral(mua_path):
    """回傳 3 維光譜特徵 [v540, v560, v577]（SWT cA 最高階近似值）"""
    data = np.loadtxt(mua_path, delimiter='\t')
    v = np.mean(data[:, 1:], axis=1)
    if len(v) < 896:
        v = np.pad(v, (0, 896 - len(v)), mode='edge')
    else:
        v = v[:896]

    coeffs  = pywt.swt(v, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    cA_top  = coeffs[0][0]   # 最高階近似係數 (cA_SWT_LEVEL)

    def pt(idx):
        return float(np.mean(cA_top[max(0, idx - WINDOW_W): idx + WINDOW_W + 1]))

    return np.array([pt(IDX_540), pt(IDX_560), pt(IDX_577)], dtype=np.float32)


# ==========================================
# 3. 資料讀取
# ==========================================
def load_dataset(base_dir, mua_folder):
    raw_feats, labels, patient_ids = [], [], []
    excel_cache = {}

    if not os.path.exists(mua_folder):
        print(f"找不到資料夾: {mua_folder}"); return np.array([]), np.array([]), []

    files = [f for f in os.listdir(mua_folder) if f.endswith('.txt')]
    for f_name in tqdm(files, desc="SWT 特徵提取中"):
        norm  = unicodedata.normalize('NFKC', f_name).lower()
        d_m   = re.search(r'(\d{4})_(\d{2})_(\d{2})', norm)
        sb_m  = re.search(r'(morning|afternoon|evening)_([a-z]+)0*(\d+)', norm)
        if not (d_m and sb_m): continue

        date_str   = f"{d_m.group(1)}{d_m.group(2)}{d_m.group(3)}"
        shift      = {'morning': '早', 'afternoon': '午', 'evening': '晚'}.get(sb_m.group(1))
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
            else:
                excel_cache[date_str] = None

        df = excel_cache[date_str]
        if df is None: continue

        row = df[(df['Bed_C'] == bed) & (df['Shift_C'] == shift)]
        if row.empty or pd.isnull(row.iloc[0]['ClinicHb']): continue

        spec_feat = _extract_spectral(os.path.join(mua_folder, f_name))

        # 選用人口統計學特徵
        demo_feat = []
        for col in DEMO_COLS:
            if col not in df.columns: continue
            val = row.iloc[0][col]
            if col.lower() in ('sex', 'gender', '性別'):
                val = _encode_sex(val)
            else:
                try: val = float(val)
                except: val = np.nan
            demo_feat.append(val)

        combined = np.concatenate([spec_feat, demo_feat]).astype(np.float32) if demo_feat else spec_feat

        raw_feats.append(combined)
        labels.append(float(row.iloc[0]['ClinicHb']))
        patient_ids.append(patient_id)

    raw_feats = np.array(raw_feats, dtype=np.float32)
    labels    = np.array(labels,    dtype=np.float32)
    print(f"\n>>> 配對成功: {len(labels)} 筆樣本 / {len(set(patient_ids))} 位病人")
    print(f">>> 輸入維度: {raw_feats.shape[1]}  (光譜 3 + 人口 {len(DEMO_COLS)})")
    return raw_feats, labels, patient_ids


# ==========================================
# 4. 模型
# ==========================================
class HbRawNet(nn.Module):
    def __init__(self, n_in=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32),   nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16),   nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ==========================================
# 5. Learning Curve
# ==========================================
def plot_learning_curve(train_losses, val_losses, tag, best_epoch):
    plt.figure(figsize=(10, 5))
    ep = range(1, len(train_losses) + 1)
    plt.plot(ep, train_losses, label='Train MSE', color='steelblue', lw=2)
    plt.plot(ep, val_losses,   label='Val MSE',   color='darkorange', lw=2)
    plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best ({best_epoch})')
    plt.ylim(0, 5)
    plt.title(f'Learning Curve — {tag}', fontsize=14)
    plt.xlabel('Epoch'); plt.ylabel('MSE (g/dL)²')
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig(f'learning_curve_{tag}.png', dpi=150)
    plt.close()


# ==========================================
# 6. 單組訓練函數
# ==========================================
def train_group(raw_features, labels, patient_ids, group_name, model_save_path, tag):
    n_samples = len(labels)
    if n_samples == 0:
        print(f"\n[{group_name}] 無資料，跳過"); return

    n_in = raw_features.shape[1]
    print(f"\n{'='*60}")
    print(f"  群組: {group_name}  |  樣本數: {n_samples}  |  輸入維度: {n_in}")
    print(f"{'='*60}")

    patient_sample_map = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        patient_sample_map[pid].append(i)

    rng = np.random.default_rng(42)
    shuffled_patients = list(patient_sample_map.keys())
    rng.shuffle(shuffled_patients)

    n_test_patients  = max(1, len(shuffled_patients) // (TRAIN_RATIO + 1))
    test_patients    = set(shuffled_patients[-n_test_patients:])
    train_patients_list = shuffled_patients[:-n_test_patients]

    tr_idx   = [i for pid in train_patients_list for i in patient_sample_map[pid]]
    test_idx = [i for pid in test_patients        for i in patient_sample_map[pid]]

    print(f"  Train pool: {len(tr_idx)} 筆 ({len(train_patients_list)} 人)")
    print(f"  Test set:   {len(test_idx)} 筆 ({len(test_patients)} 人)  ← 全程不動")

    if len(tr_idx) == 0 or len(test_idx) == 0:
        print(f"  [{group_name}] 資料不足，跳過"); return

    tr_raw   = raw_features[tr_idx]
    tr_labels= labels[tr_idx]
    tr_pids  = [patient_ids[i] for i in tr_idx]
    unique_tr_patients = np.unique(tr_pids)

    actual_folds = min(N_FOLDS, len(unique_tr_patients))
    kf           = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
    criterion    = nn.MSELoss()
    fold_results = []

    print(f"\n  {actual_folds}-Fold Cross Validation ({group_name})")

    for fold, (f_tr_pat_idx, f_val_pat_idx) in enumerate(kf.split(unique_tr_patients)):
        f_tr_pats  = set(unique_tr_patients[f_tr_pat_idx])
        f_val_pats = set(unique_tr_patients[f_val_pat_idx])

        f_tr_local  = [i for i, pid in enumerate(tr_pids) if pid in f_tr_pats]
        f_val_local = [i for i, pid in enumerate(tr_pids) if pid in f_val_pats]

        if len(f_tr_local) < BATCH_SIZE or len(f_val_local) < 2:
            print(f"  Fold {fold+1}: 樣本不足，跳過"); continue

        f_tr_raw  = tr_raw[f_tr_local]
        f_val_raw = tr_raw[f_val_local]
        # 三個光譜 channel 共用同一 mean/std，保留彼此相對關係
        feat_mean = f_tr_raw.mean()
        feat_std  = f_tr_raw.std() + 1e-8
        f_tr_norm  = (f_tr_raw  - feat_mean) / feat_std
        f_val_norm = (f_val_raw - feat_mean) / feat_std

        f_tr_labels  = tr_labels[f_tr_local]
        f_val_labels = tr_labels[f_val_local]

        f_train_loader = DataLoader(
            TensorDataset(torch.tensor(f_tr_norm), torch.tensor(f_tr_labels)),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        f_val_loader = DataLoader(
            TensorDataset(torch.tensor(f_val_norm), torch.tensor(f_val_labels)),
            batch_size=BATCH_SIZE, shuffle=False)

        model     = HbRawNet(n_in).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        best_epoch = 1
        fold_model_path = f'{tag}_fold{fold+1}_tmp.pth'

        pbar = tqdm(range(EPOCHS), desc=f"  Fold {fold+1}/{actual_folds}", unit="ep", leave=False)
        for epoch in pbar:
            model.train()
            tr_l = 0
            for x, y in f_train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward(); optimizer.step()
                tr_l += loss.item()

            model.eval(); val_l = 0
            with torch.no_grad():
                for x, y in f_val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    val_l += criterion(model(x), y).item()

            avg_tr  = tr_l  / len(f_train_loader)
            avg_val = val_l / len(f_val_loader)
            train_losses.append(avg_tr); val_losses.append(avg_val)
            pbar.set_postfix({'tr': f'{avg_tr:.3f}', 'val': f'{avg_val:.3f}'})

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch    = epoch + 1
                torch.save(model.state_dict(), fold_model_path)

        plot_learning_curve(train_losses, val_losses, f'{tag}_fold{fold+1}', best_epoch)
        model.load_state_dict(torch.load(fold_model_path, weights_only=True))
        model.eval()
        y_true_f, y_pred_f = [], []
        with torch.no_grad():
            for x, y in f_val_loader:
                y_true_f.extend(y.numpy())
                y_pred_f.extend(model(x.to(DEVICE)).cpu().numpy())

        y_true_f, y_pred_f = np.array(y_true_f), np.array(y_pred_f)
        mae  = mean_absolute_error(y_true_f, y_pred_f)
        rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
        r2   = r2_score(y_true_f, y_pred_f) if len(y_true_f) > 1 else float('nan')
        fold_results.append({'mae': mae, 'r2': r2, 'rmse': rmse})
        print(f"  Fold {fold+1}: MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

    if fold_results:
        print(f"\n  K-Fold 平均結果 ({group_name}):")
        for key, label in [('mae', 'MAE'), ('rmse', 'RMSE'), ('r2', 'R²')]:
            vals = [r[key] for r in fold_results if not np.isnan(r[key])]
            if vals:
                print(f"    {label:<6}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ── 最終模型：在全 train pool 上訓練 ──
    print(f"\n  最終模型訓練 ({group_name})")

    feat_mean = tr_raw.mean()
    feat_std  = tr_raw.std() + 1e-8
    tr_norm   = (tr_raw - feat_mean) / feat_std

    test_raw    = raw_features[test_idx]
    test_norm   = (test_raw - feat_mean) / feat_std
    test_labels = labels[test_idx]

    bs = min(BATCH_SIZE, max(2, len(tr_norm)))
    final_train_loader = DataLoader(
        TensorDataset(torch.tensor(tr_norm), torch.tensor(tr_labels)),
        batch_size=bs, shuffle=True, drop_last=(len(tr_norm) > bs))
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_norm), torch.tensor(test_labels)),
        batch_size=BATCH_SIZE, shuffle=False)

    model     = HbRawNet(n_in).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_test_loss  = float('inf')
    best_epoch      = 1
    train_losses, test_losses = [], []

    pbar = tqdm(range(EPOCHS), desc=f"  Final ({group_name})", unit="ep")
    for epoch in pbar:
        model.train(); tr_l = 0
        for x, y in final_train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            tr_l += loss.item()

        model.eval(); te_l = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                te_l += criterion(model(x), y).item()

        avg_tr = tr_l / len(final_train_loader)
        avg_te = te_l / len(test_loader)
        train_losses.append(avg_tr); test_losses.append(avg_te)
        pbar.set_postfix({'tr': f'{avg_tr:.3f}', 'test': f'{avg_te:.3f}'})

        if avg_te < best_test_loss:
            best_test_loss = avg_te
            best_epoch     = epoch + 1
            torch.save({
                'model':          model.state_dict(),
                'feat_mean':      feat_mean,
                'feat_std':       feat_std,
                'n_in':           n_in,
                'swt_level':      SWT_LEVEL,
                'demo_cols':      DEMO_COLS,
                'test_patient_ids': list(test_patients),
                'group_name':     group_name,
            }, model_save_path)

    plot_learning_curve(train_losses, test_losses, f'{tag}_final', best_epoch)

    ckpt = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            y_true.extend(y.numpy())
            y_pred.extend(model(x.to(DEVICE)).cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')

    print(f"\n{'='*50}")
    print(f"  最終 Test Set 結果 ({group_name})")
    print(f"{'='*50}")
    print(f"  MAE  : {mae:.4f} g/dL")
    print(f"  RMSE : {rmse:.4f} g/dL")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*50}")
    print(f"\n>>> 模型已存為 {model_save_path}")

    for i in range(actual_folds):
        p = f'{tag}_fold{i+1}_tmp.pth'
        if os.path.exists(p): os.remove(p)


# ==========================================
# 7. 主程式
# ==========================================
def train():
    print(f"\n>>> HB 分組回歸訓練 (Device: {DEVICE})")
    print(f">>> SWT {SWT_LEVEL} 階 ({SWT_WAVELET})，輸入特徵: v540 / v560 / v577 + {DEMO_COLS}")
    raw_features, labels, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(labels) == 0:
        print("資料集為空，請檢查路徑"); return

    low_mask  = labels <  HB_THRESHOLD
    high_mask = labels >= HB_THRESHOLD

    def pids_of(mask):
        return set(pid for i, pid in enumerate(patient_ids) if mask[i])

    print(f"\n>>> train : test = {TRAIN_RATIO} : 1")
    print(f">>> HB <  {HB_THRESHOLD}: {low_mask.sum():4d} 筆 / {len(pids_of(low_mask))} 位病人")
    print(f">>> HB >= {HB_THRESHOLD}: {high_mask.sum():4d} 筆 / {len(pids_of(high_mask))} 位病人")

    train_group(raw_features[low_mask],  labels[low_mask],
                [patient_ids[i] for i, m in enumerate(low_mask)  if m],
                f'HB < {HB_THRESHOLD}',  'swt_hb_low_model.pth',  tag='low')

    train_group(raw_features[high_mask], labels[high_mask],
                [patient_ids[i] for i, m in enumerate(high_mask) if m],
                f'HB >= {HB_THRESHOLD}', 'swt_hb_high_model.pth', tag='high')


if __name__ == "__main__":
    train()
