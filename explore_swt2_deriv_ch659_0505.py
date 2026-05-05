"""
SWT 第 2 階 + 微分 → channel 659nm 斜率
────────────────────────────────────────
對平均光譜做 SWT level=2 (db4)，
取 cA2（近似係數）並用 np.gradient 計算 1 階導數，
在 channel 659nm (index 159) 取斜率值，分析與 ClinicHb 的相關性。
同時繪製 600-700nm 整段導數曲線的相關性分佈。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import unicodedata
import pywt
from tqdm import tqdm
from scipy import stats

# ==========================================
# 參數
# ==========================================
BASE_DIR   = '/home/iir/alex'
MUA_FOLDER = os.path.join(BASE_DIR, 'mua')

SWT_LEVEL   = 2
SWT_WAVELET = 'db4'

CH_659_IDX  = 159   # 659nm = 500nm + 159
WINDOW_W    = 2
SPEC_LEN    = 896


# ==========================================
# 工具函數
# ==========================================
def _clean_bed(val):
    if pd.isnull(val): return ''
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f'{m.group(1)}{m.group(2)}' if m else val


def _pt(arr, idx, w=WINDOW_W):
    return float(np.mean(arr[max(0, idx - w): idx + w + 1]))


def _load_spectrum(mua_path):
    data = np.loadtxt(mua_path, delimiter='\t')
    v = np.mean(data[:, 1:], axis=1)
    if len(v) < SPEC_LEN: v = np.pad(v, (0, SPEC_LEN - len(v)), mode='edge')
    else:                  v = v[:SPEC_LEN]
    return v


def extract_features(v):
    """回傳 cA2、1階導數的完整向量以及 659nm 處的點值"""
    coeffs = pywt.swt(v, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    cA2    = coeffs[0][0]
    d1     = np.gradient(cA2)
    return cA2, d1


# ==========================================
# 資料載入
# ==========================================
def load_all(base_dir, mua_folder):
    """
    回傳 list of dict:
      hb, raw_659, cA2_659, d1_659,
      cA2_full (全長向量), d1_full (全長向量)
    """
    records = []
    excel_cache = {}
    files = [f for f in os.listdir(mua_folder) if f.endswith('.txt')]

    for f_name in tqdm(files, desc='SWT2 + 微分特徵提取中'):
        norm = unicodedata.normalize('NFKC', f_name).lower()
        d_m  = re.search(r'(\d{4})_(\d{2})_(\d{2})', norm)
        sb_m = re.search(r'(morning|afternoon|evening)_([a-z]+)0*(\d+)', norm)
        if not (d_m and sb_m): continue

        date_str = f"{d_m.group(1)}{d_m.group(2)}{d_m.group(3)}"
        shift    = {'morning':'早','afternoon':'午','evening':'晚'}[sb_m.group(1)]
        bed      = _clean_bed(f"{sb_m.group(2)}{sb_m.group(3)}")

        if date_str not in excel_cache:
            path = os.path.join(base_dir, f"{date_str}_dialysis_table_export.xlsx")
            if os.path.exists(path):
                df = pd.read_excel(path, engine='openpyxl')
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

        v      = _load_spectrum(os.path.join(mua_folder, f_name))
        cA2, d1 = extract_features(v)
        records.append({
            'hb':       float(row.iloc[0]['ClinicHb']),
            'raw_659':  v[CH_659_IDX],
            'cA2_659':  _pt(cA2, CH_659_IDX),
            'd1_659':   _pt(d1,  CH_659_IDX),
            'cA2_full': cA2,
            'd1_full':  d1,
        })

    return records


# ==========================================
# 相關性分析 & 繪圖
# ==========================================
def analyse(records):
    hb      = np.array([r['hb']      for r in records])
    raw_659 = np.array([r['raw_659'] for r in records])
    cA2_659 = np.array([r['cA2_659'] for r in records])
    d1_659  = np.array([r['d1_659']  for r in records])

    # ── 三個點值的相關性 ──
    feats = {
        'raw_659 (原始光譜)':       raw_659,
        f'cA2 @ 659nm (SWT{SWT_LEVEL})': cA2_659,
        'd1 (斜率) @ 659nm':        d1_659,
    }

    print(f"\n{'='*55}")
    print(f"  相關性分析  (N={len(records)})")
    print(f"{'='*55}")
    for name, vals in feats.items():
        r, p = stats.pearsonr(vals, hb)
        print(f"  {name:<35}: r={r:+.4f}  p={p:.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, vals) in zip(axes, feats.items()):
        r, p = stats.pearsonr(vals, hb)
        ax.scatter(vals, hb, alpha=0.5, color='steelblue', edgecolors='none', s=20)
        m, b = np.polyfit(vals, hb, 1)
        xs = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, m * xs + b, 'r--', lw=1.5)
        ax.set_xlabel(name, fontsize=9)
        ax.set_ylabel('ClinicHb (g/dL)')
        ax.set_title(f'r={r:+.3f}  p={p:.2e}')
        ax.axhline(10, color='green', linestyle=':', lw=1, alpha=0.7)
        ax.grid(True, alpha=0.4)

    plt.suptitle(f'SWT{SWT_LEVEL} 階 + 微分 @ 659nm × ClinicHb', fontsize=13)
    plt.tight_layout()
    plt.savefig('swt2_deriv_ch659_scatter.png', dpi=150)
    plt.close()

    # ── 600-720nm 整段：d1 各波長與 HB 的相關性分佈 ──
    idx_start, idx_end = 100, 220   # 600nm ~ 720nm
    d1_mat = np.array([r['d1_full'][idx_start:idx_end] for r in records])
    wavelengths = np.arange(500 + idx_start, 500 + idx_end)
    corr_curve = [stats.pearsonr(d1_mat[:, k], hb)[0]
                  for k in range(d1_mat.shape[1])]

    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, corr_curve, lw=1.5, color='darkorange')
    plt.axhline(0, color='black', lw=0.8)
    plt.axvline(659, color='red', linestyle='--', lw=1.2, label='659nm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Pearson r  (d1 × ClinicHb)')
    plt.title(f'SWT{SWT_LEVEL} 1階導數 × ClinicHb 相關性（600–720nm）')
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('swt2_deriv_corr_600to720.png', dpi=150)
    plt.close()

    best_wl = wavelengths[np.argmax(np.abs(corr_curve))]
    best_r  = corr_curve[np.argmax(np.abs(corr_curve))]
    print(f"\n>>> 600-720nm 範圍內 d1 最高相關波長: {best_wl}nm  r={best_r:+.4f}")
    print(f">>> 圖表已儲存: swt2_deriv_ch659_scatter.png, swt2_deriv_corr_600to720.png")


# ==========================================
# 主程式
# ==========================================
if __name__ == '__main__':
    if not os.path.isdir(MUA_FOLDER):
        print(f'找不到光譜資料夾: {MUA_FOLDER}'); exit()
    records = load_all(BASE_DIR, MUA_FOLDER)
    print(f"\n>>> 共載入 {len(records)} 筆")
    analyse(records)
