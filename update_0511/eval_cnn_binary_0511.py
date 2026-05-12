"""
Evaluate trained CNN binary classifier: cnn_binary_model_0511.pth
Outputs 4-panel figure: ROC curve, Confusion Matrix, PR curve, Score Distribution
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os, re, unicodedata
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import zoom
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix,
                              roc_curve, precision_recall_curve, average_precision_score)

# ── Parameters (must match train_cnn_binary_0511.py) ──────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER  = os.path.join(BASE_DIR, 'mua')
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = os.path.join(OUT_DIR, 'cnn_binary_model_0511.pth')

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
        spec = zoom(spec, (1.0, time_len / nt), order=1)
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


# ── 4-panel figure ─────────────────────────────────────────────
def draw_4panel(y_true, probs, preds, fig_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    ax.plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'r--', lw=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.08, color='steelblue')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve'); ax.legend(); ax.grid(True, linestyle=':', alpha=0.5)

    ax = axes[0, 1]
    cm = confusion_matrix(y_true, preds)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    classes = ['HB<10\n(Neg)', 'HB≥10\n(Pos)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14, fontweight='bold')

    ax = axes[1, 0]
    prec_c, rec_c, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    baseline = y_true.mean()
    ax.plot(rec_c, prec_c, lw=2, color='darkorange', label=f'AP = {ap:.3f}')
    ax.axhline(baseline, color='r', linestyle='--', lw=1,
               label=f'Baseline = {baseline:.3f}')
    ax.fill_between(rec_c, prec_c, alpha=0.08, color='darkorange')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve'); ax.legend(); ax.grid(True, linestyle=':', alpha=0.5)

    ax = axes[1, 1]
    ax.hist(probs[y_true == 0], bins=20, alpha=0.6, color='tomato',
            label='True Neg (HB<10)', density=True)
    ax.hist(probs[y_true == 1], bins=20, alpha=0.6, color='steelblue',
            label='True Pos (HB≥10)', density=True)
    ax.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold=0.5')
    ax.set_xlabel('Predicted P(HB≥10)'); ax.set_ylabel('Density')
    ax.set_title('Score Distribution by True Class')
    ax.legend(); ax.grid(True, linestyle=':', alpha=0.5)

    acc  = accuracy_score(y_true, preds)
    f1   = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    plt.suptitle(
        f'CNN Binary Classification Evaluation  '
        f'Acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}',
        fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'>>> Figure saved: {fig_path}')


# ── Main ──────────────────────────────────────────────────────
def evaluate():
    print(f'\n>>> CNN Binary Classification Evaluation  (Device: {DEVICE})')

    if not os.path.exists(MODEL_PATH):
        print(f'Model not found: {MODEL_PATH}\nRun train_cnn_binary_0511.py first.')
        return

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    test_patient_ids = set(ckpt['test_patient_ids'])
    wav_len  = ckpt.get('wav_len',  WAV_LEN)
    time_len = ckpt.get('time_len', TIME_LEN)

    model = HbCNNBinary().to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()

    X, y_hb, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER, wav_len, time_len)
    if len(y_hb) == 0: print('No data loaded.'); return

    test_mask = np.array([pid in test_patient_ids for pid in patient_ids])
    X_te  = X[test_mask]
    y_te  = (y_hb[test_mask] >= HB_THRESHOLD).astype(int)
    n_pats = len(set(pid for pid, m in zip(patient_ids, test_mask) if m))

    print(f'>>> Test set: {len(y_te)} samples / {n_pats} patients')
    print(f'    Positive (HB≥10): {y_te.sum()}  Negative (HB<10): {(1-y_te).sum()}')
    if len(y_te) == 0: print('Test set empty.'); return

    logits = []
    with torch.no_grad():
        for i in range(0, len(X_te), 16):
            batch = torch.tensor(X_te[i:i+16]).to(DEVICE)
            logits.extend(model(batch).cpu().numpy())

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    acc  = accuracy_score(y_te, preds)
    auc  = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float('nan')
    f1   = f1_score(y_te, preds, zero_division=0)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    cm   = confusion_matrix(y_te, preds)

    print(f'\n{"="*50}')
    print(f'  CNN Binary — Test Set Result')
    print(f'{"="*50}')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  AUC-ROC  : {auc:.4f}')
    print(f'  F1       : {f1:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall   : {rec:.4f}')
    print(f'  Confusion Matrix:')
    print(f'    TN={cm[0,0]}  FP={cm[0,1]}')
    print(f'    FN={cm[1,0]}  TP={cm[1,1]}')
    print(f'{"="*50}')

    if len(np.unique(y_te)) > 1:
        draw_4panel(y_te, probs, preds,
                    os.path.join(OUT_DIR, 'eval_cnn_binary_0511.png'))


if __name__ == '__main__':
    evaluate()
