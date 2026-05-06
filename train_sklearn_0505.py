"""
sklearn 模型比較：回歸 + 二元分類
──────────────────────────────────
使用 d2 特徵（537 / 540 / 560 / 577nm），
同時比較多種 sklearn 模型並輸出對照表。

回歸模型：LinearRegression, Ridge, Lasso, SVR, RandomForest
分類模型：LogisticRegression, SVC, RandomForest
評估方式：Patient-level 4:1 train/test split + 5-Fold CV
"""
import numpy as np
import pandas as pd
import os, re, unicodedata, pywt
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, mean_squared_error,
                              accuracy_score, roc_auc_score, f1_score)

# ==========================================
# 參數
# ==========================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER  = os.path.join(BASE_DIR, 'mua')

HB_THRESHOLD = 10.0
TRAIN_RATIO  = 4
N_FOLDS      = 5

SWT_LEVEL   = 3
SWT_WAVELET = 'db4'
WINDOW_W    = 2
SPEC_LEN    = 896

D2_INDICES = [37, 40, 60, 77]   # 537, 540, 560, 577 nm
D2_LABELS  = ['d2@537', 'd2@540', 'd2@560', 'd2@577']


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
    data = np.loadtxt(mua_path, delimiter='\t')
    v = np.mean(data[:, 1:], axis=1)
    if len(v) < SPEC_LEN: v = np.pad(v, (0, SPEC_LEN - len(v)), mode='edge')
    else:                  v = v[:SPEC_LEN]
    coeffs = pywt.swt(v, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    cA3 = coeffs[0][0]
    d2  = np.gradient(np.gradient(cA3))
    def pt(idx): return float(np.mean(d2[max(0, idx-WINDOW_W): idx+WINDOW_W+1]))
    return np.array([pt(i) for i in D2_INDICES], dtype=np.float32)


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
    X = np.array(raw_feats, dtype=np.float32)
    y = np.array(labels,    dtype=np.float32)
    print(f"\n>>> Loaded {len(y)} samples / {len(set(patient_ids))} patients")
    return X, y, patient_ids


# ==========================================
# Patient-level train/test split
# ==========================================
def patient_split(patient_ids, train_ratio=TRAIN_RATIO, seed=42):
    patient_sample_map = defaultdict(list)
    for i, pid in enumerate(patient_ids): patient_sample_map[pid].append(i)
    rng = np.random.default_rng(seed)
    shuffled = list(patient_sample_map.keys()); rng.shuffle(shuffled)
    n_test   = max(1, len(shuffled) // (train_ratio + 1))
    test_pats  = set(shuffled[-n_test:])
    train_pats = shuffled[:-n_test]
    tr_idx   = [i for pid in train_pats for i in patient_sample_map[pid]]
    te_idx   = [i for pid in test_pats  for i in patient_sample_map[pid]]
    return tr_idx, te_idx, train_pats, test_pats


# ==========================================
# 回歸評估
# ==========================================
def eval_reg(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
    return mae, rmse, r2


# ==========================================
# 分類評估
# ==========================================
def eval_clf(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if (y_prob is not None and
          len(np.unique(y_true)) > 1) else float('nan')
    return acc, f1, auc


# ==========================================
# 回歸模型比較
# ==========================================
def compare_regression(X, y, patient_ids):
    print(f"\n{'='*65}")
    print(f"  Regression Model Comparison  (N={len(y)})")
    print(f"{'='*65}")

    tr_idx, te_idx, _, _ = patient_split(patient_ids)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge(α=1)':       Ridge(alpha=1.0),
        'Ridge(α=10)':      Ridge(alpha=10.0),
        'Lasso(α=0.1)':     Lasso(alpha=0.1, max_iter=5000),
        'SVR(rbf)':         SVR(kernel='rbf', C=1.0, epsilon=0.5),
        'SVR(linear)':      SVR(kernel='linear', C=1.0, epsilon=0.5),
        'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = []
    tr_pids = [patient_ids[i] for i in tr_idx]
    unique_tr = np.unique(tr_pids)
    kf = KFold(n_splits=min(N_FOLDS, len(unique_tr)), shuffle=True, random_state=42)

    for name, model in models.items():
        cv_maes, cv_r2s = [], []
        for f_tr_pat_idx, f_val_pat_idx in kf.split(unique_tr):
            f_tr_pats  = set(unique_tr[f_tr_pat_idx])
            f_val_pats = set(unique_tr[f_val_pat_idx])
            f_tr_loc   = [i for i, pid in enumerate(tr_pids) if pid in f_tr_pats]
            f_val_loc  = [i for i, pid in enumerate(tr_pids) if pid in f_val_pats]
            if len(f_tr_loc) < 5 or len(f_val_loc) < 2: continue
            sc  = StandardScaler().fit(X_tr[f_tr_loc])
            Xf_tr  = sc.transform(X_tr[f_tr_loc])
            Xf_val = sc.transform(X_tr[f_val_loc])
            model.fit(Xf_tr, y_tr[f_tr_loc])
            pred = model.predict(Xf_val)
            cv_maes.append(mean_absolute_error(y_tr[f_val_loc], pred))
            r2 = r2_score(y_tr[f_val_loc], pred) if len(np.unique(y_tr[f_val_loc])) > 1 else float('nan')
            cv_r2s.append(r2)

        model.fit(X_tr_s, y_tr)
        y_pred_te = model.predict(X_te_s)
        mae, rmse, r2 = eval_reg(y_te, y_pred_te)
        baseline = mean_absolute_error(y_te, np.full_like(y_te, y_te.mean()))

        results.append({
            'Model':      name,
            'CV MAE':     f"{np.nanmean(cv_maes):.4f}±{np.nanstd(cv_maes):.4f}" if cv_maes else 'N/A',
            'CV R²':      f"{np.nanmean(cv_r2s):.4f}" if cv_r2s else 'N/A',
            'Test MAE':   f"{mae:.4f}",
            'Test RMSE':  f"{rmse:.4f}",
            'Test R²':    f"{r2:.4f}",
        })
        print(f"  {name:<22} | Test MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    print(f"\n  Baseline MAE (predict mean): {baseline:.4f}")
    df_res = pd.DataFrame(results)
    df_res.to_csv('sklearn_regression_results.csv', index=False)
    print(f">>> Results saved: sklearn_regression_results.csv")
    return results


# ==========================================
# 分類模型比較
# ==========================================
def compare_classification(X, y_reg, patient_ids):
    y = (y_reg >= HB_THRESHOLD).astype(int)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    print(f"\n{'='*65}")
    print(f"  Binary Classification Comparison  (N={len(y)}, pos={int(n_pos)}, neg={int(n_neg)})")
    print(f"{'='*65}")

    tr_idx, te_idx, _, _ = patient_split(patient_ids)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    cw = {0: n_pos/len(y), 1: n_neg/len(y)}   # class weight for imbalance

    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'SVC(rbf)':           SVC(kernel='rbf', class_weight='balanced', probability=True),
        'SVC(linear)':        SVC(kernel='linear', class_weight='balanced', probability=True),
        'RandomForest':       RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                                     random_state=42),
    }

    tr_pids   = [patient_ids[i] for i in tr_idx]
    unique_tr = np.unique(tr_pids)
    kf = KFold(n_splits=min(N_FOLDS, len(unique_tr)), shuffle=True, random_state=42)

    for name, model in models.items():
        cv_aucs, cv_f1s = [], []
        for f_tr_pat_idx, f_val_pat_idx in kf.split(unique_tr):
            f_tr_pats  = set(unique_tr[f_tr_pat_idx])
            f_val_pats = set(unique_tr[f_val_pat_idx])
            f_tr_loc   = [i for i, pid in enumerate(tr_pids) if pid in f_tr_pats]
            f_val_loc  = [i for i, pid in enumerate(tr_pids) if pid in f_val_pats]
            if len(f_tr_loc) < 5 or len(f_val_loc) < 2: continue
            if len(np.unique(y_tr[f_tr_loc])) < 2: continue
            sc  = StandardScaler().fit(X_tr[f_tr_loc])
            Xf_tr  = sc.transform(X_tr[f_tr_loc])
            Xf_val = sc.transform(X_tr[f_val_loc])
            model.fit(Xf_tr, y_tr[f_tr_loc])
            pred = model.predict(Xf_val)
            prob = model.predict_proba(Xf_val)[:, 1] if hasattr(model, 'predict_proba') else None
            cv_f1s.append(f1_score(y_tr[f_val_loc], pred, zero_division=0))
            if prob is not None and len(np.unique(y_tr[f_val_loc])) > 1:
                cv_aucs.append(roc_auc_score(y_tr[f_val_loc], prob))

        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        y_prob = model.predict_proba(X_te_s)[:, 1] if hasattr(model, 'predict_proba') else None
        acc, f1, auc = eval_clf(y_te, y_pred, y_prob)

        cv_auc_str = f"{np.nanmean(cv_aucs):.4f}" if cv_aucs else 'N/A'
        cv_f1_str  = f"{np.nanmean(cv_f1s):.4f}"  if cv_f1s  else 'N/A'
        print(f"  {name:<22} | Test Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
              f"(CV AUC={cv_auc_str}  CV F1={cv_f1_str})")

    print(f">>> Classification results printed above")


# ==========================================
# 主程式
# ==========================================
def main():
    if not os.path.isdir(MUA_FOLDER):
        print(f'Spectrum folder not found: {MUA_FOLDER}'); return

    X, y, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(y) == 0: return

    compare_regression(X, y, patient_ids)
    compare_classification(X, y, patient_ids)


if __name__ == '__main__':
    main()
