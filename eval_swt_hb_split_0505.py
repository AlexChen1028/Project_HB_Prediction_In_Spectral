import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import re
import unicodedata
import pywt
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm

# ==========================================
# 1. 參數設定 (需與 train_swt_hb_split.py 一致)
# ==========================================
BASE_DIR        = '/home/iir/alex'
MUA_FOLDER      = os.path.join(BASE_DIR, 'mua')
LOW_MODEL_PATH  = os.path.join(BASE_DIR, 'swt_hb_low_model.pth')
HIGH_MODEL_PATH = os.path.join(BASE_DIR, 'swt_hb_high_model.pth')
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HB_THRESHOLD = 10.0
SWT_LEVEL    = 3
SWT_WAVELET  = 'db4'
IDX_540 = 40; IDX_560 = 60; IDX_577 = 77; WINDOW_W = 2


# ==========================================
# 2. 工具函數
# ==========================================
def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


def _extract_features(mua_path):
    data = np.loadtxt(mua_path, delimiter='\t')
    v = np.mean(data[:, 1:], axis=1)
    if len(v) < 896: v = np.pad(v, (0, 896 - len(v)), mode='edge')
    else:            v = v[:896]
    coeffs  = pywt.swt(v, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    cA      = coeffs[0][0]
    def pt(idx): return float(np.mean(cA[max(0, idx - WINDOW_W): idx + WINDOW_W + 1]))
    return np.array([pt(IDX_540), pt(IDX_560), pt(IDX_577)], dtype=np.float32)


def load_dataset(base_dir, mua_folder):
    raw_features, labels, patient_ids = [], [], []
    excel_cache = {}
    files = [f for f in os.listdir(mua_folder) if f.endswith('.txt')]

    for f_name in tqdm(files, desc="讀取資料中"):
        norm  = unicodedata.normalize('NFKC', f_name).lower()
        d_m   = re.search(r'(\d{4})_(\d{2})_(\d{2})', norm)
        s_b_m = re.search(r'(morning|afternoon|evening)_([a-z]+)0*(\d+)', norm)
        if not (d_m and s_b_m): continue

        date_str   = f"{d_m.group(1)}{d_m.group(2)}{d_m.group(3)}"
        shift      = {'morning': '早', 'afternoon': '午', 'evening': '晚'}.get(s_b_m.group(1))
        bed        = _clean_bed(f"{s_b_m.group(2)}{s_b_m.group(3)}")
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

        raw_features.append(_extract_features(os.path.join(mua_folder, f_name)))
        labels.append(float(row.iloc[0]['ClinicHb']))
        patient_ids.append(patient_id)

    return (np.array(raw_features, dtype=np.float32),
            np.array(labels, dtype=np.float32), patient_ids)


# ==========================================
# 3. 模型
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
    def forward(self, x): return self.net(x).squeeze(-1)


# ==========================================
# 4. 4 格圖
# ==========================================
def draw_4panel(y_true, y_pred, group_name, fig_path):
    residuals = y_true - y_pred
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    lim = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax1.scatter(y_true, y_pred, alpha=0.6, color='dodgerblue', edgecolors='none')
    ax1.plot(lim, lim, 'r--', lw=2); ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel('Actual ClinicHb (g/dL)'); ax1.set_ylabel('Predicted ClinicHb (g/dL)')
    ax1.set_title(f'Regression  R²={r2:.3f}  MAE={mae:.3f}')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = axes[0, 1]
    s = np.argsort(y_true)
    ax2.plot(y_true[s], label='Actual', color='black', marker='o', markersize=3, lw=1)
    ax2.plot(y_pred[s], label='Predicted', color='red', marker='x', markersize=3, lw=1)
    ax2.axhline(HB_THRESHOLD, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Sample (sorted by actual HB)'); ax2.set_ylabel('ClinicHb')
    ax2.set_title('Sorted Prediction'); ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax3 = axes[1, 0]
    ax3.scatter(y_pred, residuals, alpha=0.6, color='purple', edgecolors='none')
    ax3.axhline(0, color='red', linestyle='--', lw=2)
    ax3.axhline(1, color='gray', linestyle=':', lw=1); ax3.axhline(-1, color='gray', linestyle=':', lw=1)
    ax3.set_xlabel('Predicted'); ax3.set_ylabel('Residual (Actual − Pred)')
    ax3.set_title('Residual Plot'); ax3.grid(True, linestyle=':', alpha=0.6)

    ax4 = axes[1, 1]
    (osm, osr), (slope, intercept, r_val) = stats.probplot(residuals, dist='norm')
    ax4.scatter(osm, osr, alpha=0.6, color='teal', edgecolors='none')
    ax4.plot(osm, slope * np.array(osm) + intercept, 'r--', lw=2)
    ax4.set_xlabel('Theoretical Quantiles'); ax4.set_ylabel('Sample Quantiles')
    ax4.set_title(f'Normal Probability Plot  (R={r_val:.3f})')
    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f'{group_name} — MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}',
                 fontsize=13, y=1.01)
    plt.tight_layout(); plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f">>> 圖表已儲存至 {fig_path}")


# ==========================================
# 5. 單組評估
# ==========================================
def eval_group(model_path, raw_features_all, labels_all, patient_ids_all,
               group_name, is_low, fig_tag):
    print(f"\n{'='*55}\n  載入模型: {model_path}")
    if not os.path.exists(model_path):
        print("  找不到模型，請先執行 train_swt_hb_split.py"); return

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    feat_mean        = ckpt['feat_mean']
    feat_std         = ckpt['feat_std']
    n_in             = ckpt.get('n_in', 3)
    test_patient_ids = set(ckpt['test_patient_ids'])

    model = HbRawNet(n_in).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()

    group_mask  = labels_all < HB_THRESHOLD if is_low else labels_all >= HB_THRESHOLD
    patient_mask = np.array([pid in test_patient_ids for pid in patient_ids_all])
    mask = group_mask & patient_mask

    raw_features = raw_features_all[mask]
    labels       = labels_all[mask]
    n_pats       = len(set(pid for pid, m in zip(patient_ids_all, mask) if m))
    print(f"  [{group_name}] Test set: {len(labels)} 筆 / {n_pats} 位病人")

    if len(labels) == 0:
        print(f"  [{group_name}] 無測試樣本，跳過"); return

    X = (raw_features - feat_mean) / feat_std
    y_pred = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            y_pred.extend(model(torch.tensor(X[i:i+32]).to(DEVICE)).cpu().numpy())

    y_true = labels; y_pred = np.array(y_pred)
    mae      = mean_absolute_error(y_true, y_pred)
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    r2       = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    base_mae = mean_absolute_error(y_true, np.full_like(y_true, np.mean(y_true)))

    print(f"\n{'='*52}")
    print(f"  評估結果: {group_name}")
    print(f"{'='*52}")
    print(f"  樣本數:              {len(y_true)}")
    print(f"  Baseline MAE (mean): {base_mae:.4f} g/dL")
    print(f"  MAE:                 {mae:.4f} g/dL")
    print(f"  RMSE:                {rmse:.4f} g/dL")
    print(f"  R²:                  {r2:.4f}")
    print(f"{'='*52}")
    draw_4panel(y_true, y_pred, group_name, f'swt_hb_{fig_tag}_eval.png')


# ==========================================
# 6. 主程式
# ==========================================
def evaluate():
    print(f"\n>>> HB 分組評估 (Device: {DEVICE})")
    raw_features, labels, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(labels) == 0: print("資料集為空"); return

    eval_group(LOW_MODEL_PATH,  raw_features, labels, patient_ids,
               f'HB < {HB_THRESHOLD}',  is_low=True,  fig_tag='low')
    eval_group(HIGH_MODEL_PATH, raw_features, labels, patient_ids,
               f'HB >= {HB_THRESHOLD}', is_low=False, fig_tag='high')


if __name__ == "__main__":
    evaluate()
