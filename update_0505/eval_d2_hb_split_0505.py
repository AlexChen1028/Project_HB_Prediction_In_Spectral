"""
Evaluate trained d2 regression models: d2_hb_low_model.pth / d2_hb_high_model.pth
Outputs per-group metrics and a combined 4-panel figure.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os, re, unicodedata, pywt
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ==========================================
# Parameters (must match train_d2_hb_split_0505.py)
# ==========================================
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER      = os.path.join(BASE_DIR, 'mua')
LOW_MODEL_PATH  = 'd2_hb_low_model.pth'
HIGH_MODEL_PATH = 'd2_hb_high_model.pth'
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HB_THRESHOLD = 10.0
SWT_LEVEL    = 3
SWT_WAVELET  = 'db4'
WINDOW_W     = 2
SPEC_LEN     = 896
D2_INDICES   = [37, 40, 60, 77]   # 537, 540, 560, 577 nm


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
    return (np.array(raw_feats, dtype=np.float32),
            np.array(labels, dtype=np.float32), patient_ids)


# ==========================================
# Model
# ==========================================
class HbD2Net(nn.Module):
    def __init__(self, n_in=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32),   nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16),   nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)


# ==========================================
# Evaluate one group, return (y_true, y_pred, color_tag)
# ==========================================
def eval_group(model_path, raw_features, labels, patient_ids, is_low):
    label = f'HB < {HB_THRESHOLD}' if is_low else f'HB >= {HB_THRESHOLD}'
    color = 'dodgerblue' if is_low else 'tomato'

    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}"); return None, None, None

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    feat_mean        = ckpt['feat_mean']          # shape (4,)
    feat_std         = ckpt['feat_std']            # shape (4,)
    test_patient_ids = set(ckpt['test_patient_ids'])
    n_in             = len(ckpt.get('d2_indices', D2_INDICES))

    model = HbD2Net(n_in).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()

    group_mask   = (labels < HB_THRESHOLD) if is_low else (labels >= HB_THRESHOLD)
    patient_mask = np.array([pid in test_patient_ids for pid in patient_ids])
    mask = group_mask & patient_mask

    X    = raw_features[mask]
    y    = labels[mask]
    n_pts= len(y)
    n_pat= len(set(pid for pid, m in zip(patient_ids, mask) if m))

    print(f"\n  [{label}]  test: {n_pts} samples / {n_pat} patients")
    if n_pts == 0:
        print(f"  No test samples — check test_patient_ids in checkpoint"); return None, None, None

    X_norm = (X - feat_mean) / feat_std
    y_pred = []
    with torch.no_grad():
        for i in range(0, len(X_norm), 32):
            batch = torch.tensor(X_norm[i:i+32]).to(DEVICE)
            y_pred.extend(model(batch).cpu().numpy())
    y_pred = np.array(y_pred)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred) if len(np.unique(y)) > 1 else float('nan')
    base = mean_absolute_error(y, np.full_like(y, y.mean()))

    print(f"  Baseline MAE (mean): {base:.4f} g/dL")
    print(f"  MAE  : {mae:.4f} g/dL")
    print(f"  RMSE : {rmse:.4f} g/dL")
    print(f"  R²   : {r2:.4f}")

    return y, y_pred, color


# ==========================================
# 4-panel combined figure
# ==========================================
def draw_4panel(y_true, y_pred, colors, fig_path):
    residuals = y_true - y_pred
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Scatter — actual vs predicted
    ax = axes[0, 0]
    for tag, col in [('dodgerblue', 'Low (<10)'), ('tomato', 'High (>=10)')]:
        mask = colors == tag
        if mask.any():
            ax.scatter(y_true[mask], y_pred[mask], alpha=0.6, color=tag,
                       edgecolors='none', label=col)
    lim = [min(y_true.min(), y_pred.min()) - 0.5,
           max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lim, lim, 'k--', lw=2); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual ClinicHb (g/dL)'); ax.set_ylabel('Predicted ClinicHb (g/dL)')
    ax.set_title(f'Regression  R²={r2:.3f}  MAE={mae:.3f}')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Sorted prediction trend
    ax = axes[0, 1]
    s = np.argsort(y_true)
    ax.plot(y_true[s], color='black', marker='o', markersize=3, lw=1, label='Actual')
    ax.scatter(range(len(y_pred[s])), y_pred[s],
               c=colors[s], marker='x', s=40, zorder=3)
    ax.plot([], [], 'x', color='dodgerblue', label='Pred Low')
    ax.plot([], [], 'x', color='tomato',     label='Pred High')
    ax.axhline(HB_THRESHOLD, color='green', linestyle='--', lw=1, alpha=0.7, label=f'Threshold {HB_THRESHOLD}')
    ax.set_xlabel('Sample (sorted by actual HB)'); ax.set_ylabel('ClinicHb (g/dL)')
    ax.set_title('Sorted Prediction'); ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Residual plot
    ax = axes[1, 0]
    ax.scatter(y_pred, residuals, alpha=0.6, color='purple', edgecolors='none')
    ax.axhline(0,  color='red',  linestyle='--', lw=2)
    ax.axhline(1,  color='gray', linestyle=':', lw=1)
    ax.axhline(-1, color='gray', linestyle=':', lw=1)
    ax.set_xlabel('Predicted ClinicHb'); ax.set_ylabel('Residual (Actual − Pred)')
    ax.set_title('Residual Plot'); ax.grid(True, linestyle=':', alpha=0.6)

    # Panel 4: Q-Q plot
    ax = axes[1, 1]
    (osm, osr), (slope, intercept, r_val) = stats.probplot(residuals, dist='norm')
    ax.scatter(osm, osr, alpha=0.6, color='teal', edgecolors='none')
    ax.plot(osm, slope * np.array(osm) + intercept, 'r--', lw=2)
    ax.set_xlabel('Theoretical Quantiles'); ax.set_ylabel('Sample Quantiles')
    ax.set_title(f'Normal Probability Plot  (R={r_val:.3f})')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(
        f'd2 Regression Evaluation  '
        f'N={len(y_true)}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}',
        fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n>>> Figure saved: {fig_path}")


# ==========================================
# Main
# ==========================================
def evaluate():
    print(f"\n>>> d2 Regression Evaluation  (Device: {DEVICE})")

    if not os.path.isdir(MUA_FOLDER):
        print(f"Spectrum folder not found: {MUA_FOLDER}"); return

    raw_features, labels, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(labels) == 0: print("No data loaded."); return

    print(f"\n{'='*55}")
    results = []
    for model_path, is_low in [(LOW_MODEL_PATH, True), (HIGH_MODEL_PATH, False)]:
        y_t, y_p, col = eval_group(model_path, raw_features, labels, patient_ids, is_low)
        if y_t is not None:
            results.append((y_t, y_p, col))

    if not results:
        print("No results to plot."); return

    # Combined figure
    y_true  = np.concatenate([r[0] for r in results])
    y_pred  = np.concatenate([r[1] for r in results])
    colors  = np.concatenate([np.full(len(r[0]), r[2], dtype=object) for r in results])

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    base = mean_absolute_error(y_true, np.full_like(y_true, y_true.mean()))

    print(f"\n{'='*55}")
    print(f"  Combined Test Set  (N={len(y_true)})")
    print(f"{'='*55}")
    print(f"  Baseline MAE (mean): {base:.4f} g/dL")
    print(f"  MAE  : {mae:.4f} g/dL")
    print(f"  RMSE : {rmse:.4f} g/dL")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*55}")

    draw_4panel(y_true, y_pred, colors, 'eval_d2_hb_split.png')


if __name__ == "__main__":
    evaluate()
