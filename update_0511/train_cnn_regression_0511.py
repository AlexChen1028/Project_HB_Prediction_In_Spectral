"""
CNN Regression: treat raw 2D spectrum (wavelength × time) as a grayscale image.

Instead of computing d2 features, each MUA file is loaded as a 2D array:
  rows = wavelength (500–800 nm, 300 points)
  cols = time measurements (resized to TIME_LEN)
Per-sample z-score normalization is applied before feeding into CNN.
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.ndimage import zoom

# ── Parameters ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER = os.path.join(BASE_DIR, 'mua')
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WAV_LEN      = 300    # wavelength points (500–800 nm)
TIME_LEN     = 150    # fixed time dimension after resize
HB_THRESHOLD = 10.0
BATCH_SIZE   = 8
EPOCHS       = 300
LR           = 1e-4
WEIGHT_DECAY = 1e-4
N_FOLDS      = 5
TRAIN_RATIO  = 4
MODEL_PATH   = os.path.join(OUT_DIR, 'cnn_regression_model_0511.pth')


# ── Utilities ─────────────────────────────────────────────────
def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


def load_image(mua_path):
    """Load MUA file → (WAV_LEN, TIME_LEN) float32, per-sample z-scored."""
    data = np.loadtxt(mua_path, delimiter='\t')
    spec = data[:, 1:]                         # (n_wav, n_time)
    # Crop / pad wavelength axis
    nw = spec.shape[0]
    if nw < WAV_LEN:
        spec = np.pad(spec, ((0, WAV_LEN - nw), (0, 0)), mode='edge')
    spec = spec[:WAV_LEN, :]                   # (WAV_LEN, n_time)
    # Resize time axis
    nt = spec.shape[1]
    if nt != TIME_LEN:
        spec = zoom(spec, (1.0, TIME_LEN / nt), order=1)
    # Per-sample z-score
    mu, sigma = spec.mean(), spec.std() + 1e-8
    return ((spec - mu) / sigma).astype(np.float32)   # (WAV_LEN, TIME_LEN)


# ── Data Loading ──────────────────────────────────────────────
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

        img = load_image(os.path.join(mua_folder, f_name))
        images.append(img)
        labels.append(float(row.iloc[0]['ClinicHb']))
        patient_ids.append(f"{bed}_{shift}")

    X = np.array(images, dtype=np.float32)[:, np.newaxis, :, :]  # (N,1,WAV,TIME)
    y = np.array(labels, dtype=np.float32)
    print(f"\n>>> Loaded {len(y)} samples / {len(set(patient_ids))} patients")
    print(f">>> Image shape: {X.shape[1:]}  (C x WAV x TIME)")
    return X, y, patient_ids


# ── Model ─────────────────────────────────────────────────────
class HbCNNReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),      # →(16,150,75)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),      # →(32,75,37)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                         # →(64,4,4)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.head(self.features(x)).squeeze(-1)


# ── Learning curve plot ───────────────────────────────────────
def plot_lc(tr_ls, val_ls, tag, best_ep):
    plt.figure(figsize=(10, 4))
    plt.plot(tr_ls, label='Train MSE', color='steelblue', lw=1.5)
    plt.plot(val_ls, label='Val MSE',   color='darkorange', lw=1.5)
    plt.axvline(best_ep - 1, color='red', linestyle='--', label=f'Best epoch {best_ep}')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title(f'CNN Learning Curve — {tag}')
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'lc_cnn_{tag}.png'), dpi=150); plt.close()


# ── Training ──────────────────────────────────────────────────
def train():
    print(f"\n>>> CNN Regression  (Device: {DEVICE})")
    print(f"    Image: 1 × {WAV_LEN} × {TIME_LEN}  (channel × wavelength × time)")

    X, y, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(y) == 0: return

    print(f"    HB range: {y.min():.1f} – {y.max():.1f}  mean={y.mean():.2f}")

    # Patient-level 4:1 split
    pid_map = defaultdict(list)
    for i, pid in enumerate(patient_ids): pid_map[pid].append(i)
    rng = np.random.default_rng(42)
    pids = list(pid_map.keys()); rng.shuffle(pids)
    n_te      = max(1, len(pids) // (TRAIN_RATIO + 1))
    test_pats = set(pids[-n_te:])
    tr_pats   = pids[:-n_te]

    tr_idx   = [i for pid in tr_pats  for i in pid_map[pid]]
    te_idx   = [i for pid in test_pats for i in pid_map[pid]]
    print(f"    Train: {len(tr_idx)} samples ({len(tr_pats)} patients)")
    print(f"    Test:  {len(te_idx)} samples ({len(test_pats)} patients)")

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    tr_pids    = [patient_ids[i] for i in tr_idx]
    unique_tr  = np.unique(tr_pids)

    criterion = nn.MSELoss()
    actual_folds = min(N_FOLDS, len(unique_tr))
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n  {actual_folds}-Fold CV")
    for fold, (f_tr_idx, f_val_idx) in enumerate(kf.split(unique_tr)):
        f_tr_pats  = set(unique_tr[f_tr_idx])
        f_val_pats = set(unique_tr[f_val_idx])
        f_tr_loc   = [i for i, p in enumerate(tr_pids) if p in f_tr_pats]
        f_val_loc  = [i for i, p in enumerate(tr_pids) if p in f_val_pats]
        if len(f_tr_loc) < BATCH_SIZE or len(f_val_loc) < 2:
            print(f"  Fold {fold+1}: skip"); continue

        f_train_dl = DataLoader(
            TensorDataset(torch.tensor(X_tr[f_tr_loc]), torch.tensor(y_tr[f_tr_loc])),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        f_val_dl   = DataLoader(
            TensorDataset(torch.tensor(X_tr[f_val_loc]), torch.tensor(y_tr[f_val_loc])),
            batch_size=BATCH_SIZE, shuffle=False)

        model = HbCNNReg().to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_val, best_ep = float('inf'), 1
        fold_path = os.path.join(OUT_DIR, f'cnn_fold{fold+1}_tmp.pth')
        tr_ls, val_ls = [], []

        for epoch in tqdm(range(EPOCHS), desc=f"  Fold {fold+1}", leave=False):
            model.train(); tl = 0
            for x, yb in f_train_dl:
                x, yb = x.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); loss = criterion(model(x), yb); loss.backward(); opt.step()
                tl += loss.item()
            model.eval(); vl = 0
            with torch.no_grad():
                for x, yb in f_val_dl:
                    vl += criterion(model(x.to(DEVICE)), yb.to(DEVICE)).item()
            tr_ls.append(tl / len(f_train_dl)); val_ls.append(vl / len(f_val_dl))
            if val_ls[-1] < best_val:
                best_val = val_ls[-1]; best_ep = epoch + 1
                torch.save(model.state_dict(), fold_path)

        model.load_state_dict(torch.load(fold_path, weights_only=True)); model.eval()
        y_p = []
        with torch.no_grad():
            for x, _ in f_val_dl:
                y_p.extend(model(x.to(DEVICE)).cpu().numpy())
        y_t = y_tr[f_val_loc]
        mae  = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        r2   = r2_score(y_t, y_p) if len(np.unique(y_t)) > 1 else float('nan')
        fold_results.append({'mae': mae, 'rmse': rmse, 'r2': r2})
        print(f"  Fold {fold+1}: MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

    if fold_results:
        print(f"\n  CV avg:")
        for key, lab in [('mae','MAE'),('rmse','RMSE'),('r2','R²')]:
            vals = [r[key] for r in fold_results if not np.isnan(r[key])]
            if vals: print(f"    {lab}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Final model on full train set
    print(f"\n  Final model training")
    bs = min(BATCH_SIZE, len(X_tr))
    final_dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                          batch_size=bs, shuffle=True, drop_last=(len(X_tr) > bs))
    test_dl  = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)),
                          batch_size=BATCH_SIZE, shuffle=False)

    model  = HbCNNReg().to(DEVICE)
    opt    = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_te, best_ep = float('inf'), 1
    tr_ls, te_ls = [], []

    for epoch in tqdm(range(EPOCHS), desc="  Final", unit="ep"):
        model.train(); tl = 0
        for x, yb in final_dl:
            x, yb = x.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); criterion(model(x), yb).backward(); opt.step()
            tl += criterion(model(x), yb).item()
        model.eval(); tel = 0
        with torch.no_grad():
            for x, yb in test_dl:
                tel += criterion(model(x.to(DEVICE)), yb.to(DEVICE)).item()
        tr_ls.append(tl / len(final_dl)); te_ls.append(tel / len(test_dl))
        if te_ls[-1] < best_te:
            best_te = te_ls[-1]; best_ep = epoch + 1
            torch.save({'model': model.state_dict(),
                        'test_patient_ids': list(test_pats),
                        'wav_len': WAV_LEN, 'time_len': TIME_LEN}, MODEL_PATH)

    plot_lc(tr_ls, te_ls, 'final', best_ep)

    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    y_p = []
    with torch.no_grad():
        for x, _ in test_dl:
            y_p.extend(model(x.to(DEVICE)).cpu().numpy())
    y_p = np.array(y_p)
    mae  = mean_absolute_error(y_te, y_p)
    rmse = np.sqrt(mean_squared_error(y_te, y_p))
    r2   = r2_score(y_te, y_p) if len(np.unique(y_te)) > 1 else float('nan')
    base = mean_absolute_error(y_te, np.full_like(y_te, y_te.mean()))

    print(f"\n{'='*50}")
    print(f"  CNN Regression — Test Set Result  (N={len(y_te)})")
    print(f"{'='*50}")
    print(f"  Baseline MAE (mean): {base:.4f} g/dL")
    print(f"  MAE  : {mae:.4f} g/dL")
    print(f"  RMSE : {rmse:.4f} g/dL")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*50}")
    print(f">>> Model saved: {MODEL_PATH}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    low  = y_te < HB_THRESHOLD
    ax.scatter(y_te[low],  y_p[low],  c='dodgerblue', alpha=0.7, s=40, label='HB<10')
    ax.scatter(y_te[~low], y_p[~low], c='tomato',     alpha=0.7, s=40, label='HB≥10')
    lim = [min(y_te.min(), y_p.min()) - 0.3, max(y_te.max(), y_p.max()) + 0.3]
    ax.plot(lim, lim, 'k--', lw=1.5); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual ClinicHb (g/dL)'); ax.set_ylabel('Predicted ClinicHb (g/dL)')
    ax.set_title(f'CNN Regression  R²={r2:.3f}  MAE={mae:.3f}  N={len(y_te)}')
    ax.legend(); ax.grid(True, linestyle=':', alpha=0.5); plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'eval_cnn_regression_0511.png')
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f">>> Figure saved: {fig_path}")

    for i in range(actual_folds):
        p = os.path.join(OUT_DIR, f'cnn_fold{i+1}_tmp.pth')
        if os.path.exists(p): os.remove(p)


if __name__ == '__main__':
    train()
