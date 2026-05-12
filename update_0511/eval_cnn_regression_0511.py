"""
Evaluate trained CNN regression model: cnn_regression_model_0511.pth
Outputs 4-panel figure: scatter, sorted prediction, residuals, Q-Q plot
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os, re, unicodedata
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Parameters (must match train_cnn_regression_0511.py) ──────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER  = os.path.join(BASE_DIR, 'mua')
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = os.path.join(OUT_DIR, 'cnn_regression_model_0511.pth')

HB_THRESHOLD = 10.0
WAV_LEN      = 300
TIME_LEN     = 150


# ── Utilities ─────────────────────────────────────────────────
def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


def load_image(mua_path, wav_len, time_len):
    data = np.loadtxt(mua_path, delimiter='\t')
    spec = data[:, 1:]
    nw = spec.shape[0]
    if nw < wav_len:
        spec = np.pad(spec, ((0, wav_len - nw), (0, 0)), mode='edge')
    spec = spec[:wav_len, :]
    nt = spec.shape[1]
    if nt != time_len:
        spec = ndimage.zoom(spec, (1.0, time_len / nt), order=1)
    mu, sigma = spec.mean(), spec.std() + 1e-8
    return ((spec - mu) / sigma).astype(np.float32)


def load_dataset(base_dir, mua_folder, wav_len, time_len):
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
        images.append(load_image(os.path.join(mua_folder, f_name), wav_len, time_len))
        labels.append(float(row.iloc[0]['ClinicHb']))
        patient_ids.append(f"{bed}_{shift}")
    X = np.array(images, dtype=np.float32)[:, np.newaxis, :, :]
    y = np.array(labels, dtype=np.float32)
    return X, y, patient_ids


# ── Model ─────────────────────────────────────────────────────
class HbCNNReg(nn.Module):
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


# ── 4-panel figure ─────────────────────────────────────────────
def draw_4panel(y_true, y_pred, fig_path):
    residuals = y_true - y_pred
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
    low  = y_true < HB_THRESHOLD

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Scatter
    ax = axes[0, 0]
    ax.scatter(y_true[low],  y_pred[low],  c='dodgerblue', alpha=0.7, s=35,
               edgecolors='none', label='HB<10')
    ax.scatter(y_true[~low], y_pred[~low], c='tomato',     alpha=0.7, s=35,
               edgecolors='none', label='HB≥10')
    lim = [min(y_true.min(), y_pred.min()) - 0.3,
           max(y_true.max(), y_pred.max()) + 0.3]
    ax.plot(lim, lim, 'k--', lw=1.5); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual ClinicHb (g/dL)'); ax.set_ylabel('Predicted ClinicHb (g/dL)')
    ax.set_title(f'CNN Regression  R²={r2:.3f}  MAE={mae:.3f}')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    # Sorted prediction
    ax = axes[0, 1]
    s = np.argsort(y_true)
    colors = np.where(low[s], 'dodgerblue', 'tomato')
    ax.plot(y_true[s], color='black', marker='o', markersize=3, lw=1, label='Actual')
    ax.scatter(range(len(y_pred[s])), y_pred[s], c=colors, marker='x', s=40, zorder=3)
    ax.plot([], [], 'x', color='dodgerblue', label='Pred Low')
    ax.plot([], [], 'x', color='tomato',     label='Pred High')
    ax.axhline(HB_THRESHOLD, color='green', linestyle='--', lw=1, alpha=0.7,
               label=f'Threshold {HB_THRESHOLD}')
    ax.set_xlabel('Sample (sorted by actual HB)'); ax.set_ylabel('ClinicHb (g/dL)')
    ax.set_title('Sorted Prediction')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    # Residuals
    ax = axes[1, 0]
    ax.scatter(y_pred, residuals, alpha=0.6, color='purple', edgecolors='none', s=30)
    ax.axhline(0,  color='red',  linestyle='--', lw=2)
    ax.axhline(1,  color='gray', linestyle=':', lw=1)
    ax.axhline(-1, color='gray', linestyle=':', lw=1)
    ax.set_xlabel('Predicted ClinicHb'); ax.set_ylabel('Residual (Actual − Pred)')
    ax.set_title('Residual Plot'); ax.grid(True, linestyle=':', alpha=0.5)

    # Q-Q plot
    ax = axes[1, 1]
    (osm, osr), (slope, intercept, r_val) = stats.probplot(residuals, dist='norm')
    ax.scatter(osm, osr, alpha=0.6, color='teal', edgecolors='none', s=30)
    ax.plot(osm, slope * np.array(osm) + intercept, 'r--', lw=2)
    ax.set_xlabel('Theoretical Quantiles'); ax.set_ylabel('Sample Quantiles')
    ax.set_title(f'Normal Probability Plot  (R={r_val:.3f})')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle(
        f'CNN Regression Evaluation  N={len(y_true)}  '
        f'MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}',
        fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'>>> Figure saved: {fig_path}')


# ── Main ──────────────────────────────────────────────────────
def evaluate():
    print(f'\n>>> CNN Regression Evaluation  (Device: {DEVICE})')

    if not os.path.exists(MODEL_PATH):
        print(f'Model not found: {MODEL_PATH}\nRun train_cnn_regression_0511.py first.')
        return

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    test_patient_ids = set(ckpt['test_patient_ids'])
    wav_len  = ckpt.get('wav_len',  WAV_LEN)
    time_len = ckpt.get('time_len', TIME_LEN)

    model = HbCNNReg().to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()

    X, y, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER, wav_len, time_len)
    if len(y) == 0: print('No data loaded.'); return

    test_mask = np.array([pid in test_patient_ids for pid in patient_ids])
    X_te = X[test_mask]
    y_te = y[test_mask]
    n_pats = len(set(pid for pid, m in zip(patient_ids, test_mask) if m))

    print(f'>>> Test set: {len(y_te)} samples / {n_pats} patients')
    if len(y_te) == 0: print('Test set empty.'); return

    y_pred = []
    with torch.no_grad():
        for i in range(0, len(X_te), 16):
            batch = torch.tensor(X_te[i:i+16]).to(DEVICE)
            y_pred.extend(model(batch).cpu().numpy())
    y_pred = np.array(y_pred)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred) if len(np.unique(y_te)) > 1 else float('nan')
    base = mean_absolute_error(y_te, np.full_like(y_te, y_te.mean()))

    print(f'\n{"="*50}')
    print(f'  CNN Regression — Test Set Result')
    print(f'{"="*50}')
    print(f'  Baseline MAE (mean): {base:.4f} g/dL')
    print(f'  MAE  : {mae:.4f} g/dL')
    print(f'  RMSE : {rmse:.4f} g/dL')
    print(f'  R²   : {r2:.4f}')
    print(f'{"="*50}')

    draw_4panel(y_te, y_pred, os.path.join(OUT_DIR, 'eval_cnn_regression_0511.png'))


if __name__ == '__main__':
    evaluate()
