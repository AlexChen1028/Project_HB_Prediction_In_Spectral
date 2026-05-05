"""
1 階 / 2 階導數特徵探索
──────────────────────
對 SWT 近似係數（cA3）計算 1 階與 2 階導數，
找出在各波長位置與 ClinicHb 相關性最高的導數特徵。

輸出：
  - 相關性熱圖（整條 500-900nm）
  - Top-N 特徵散佈圖
  - 特徵相關性排行表
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

SWT_LEVEL   = 3
SWT_WAVELET = 'db4'
WINDOW_W    = 2
SPEC_LEN    = 896

# 感興趣的波長範圍（索引）
RANGE_START = 0    # 500nm
RANGE_END   = 300  # 800nm

# 重點波長索引（用於詳細散佈圖）
KEY_IDX = {
    '540nm': 40, '542nm': 42, '560nm': 60,
    '577nm': 77, '659nm': 159, '700nm': 200,
}


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


def extract(mua_path):
    """回傳 (raw_v, cA, d1, d2) 四條完整向量"""
    data = np.loadtxt(mua_path, delimiter='\t')
    v = np.mean(data[:, 1:], axis=1)
    if len(v) < SPEC_LEN: v = np.pad(v, (0, SPEC_LEN - len(v)), mode='edge')
    else:                  v = v[:SPEC_LEN]

    coeffs = pywt.swt(v, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    cA     = coeffs[0][0]          # cA3（最高階近似）
    d1     = np.gradient(cA)       # 1 階導數
    d2     = np.gradient(d1)       # 2 階導數
    return v, cA, d1, d2


# ==========================================
# 資料載入
# ==========================================
def load_all(base_dir, mua_folder):
    records = []
    excel_cache = {}
    files = [f for f in os.listdir(mua_folder) if f.endswith('.txt')]

    for f_name in tqdm(files, desc='導數特徵提取中'):
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

        raw_v, cA, d1, d2 = extract(os.path.join(mua_folder, f_name))
        records.append({
            'hb':  float(row.iloc[0]['ClinicHb']),
            'raw': raw_v[RANGE_START:RANGE_END],
            'cA':  cA[RANGE_START:RANGE_END],
            'd1':  d1[RANGE_START:RANGE_END],
            'd2':  d2[RANGE_START:RANGE_END],
        })

    return records


# ==========================================
# 相關性計算
# ==========================================
def corr_curve(matrix, hb):
    """每個波長位置計算 Pearson r"""
    return np.array([stats.pearsonr(matrix[:, k], hb)[0]
                     for k in range(matrix.shape[1])])


# ==========================================
# 繪圖
# ==========================================
def plot_corr_overview(wavelengths, corr_dict, n_samples):
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    colors = ['gray', 'steelblue', 'darkorange', 'purple']
    labels = ['raw spectrum (cA 原始)', f'SWT{SWT_LEVEL} cA', '1 階導數 (d1)', '2 階導數 (d2)']

    for ax, (key, corr), color, label in zip(axes, corr_dict.items(), colors, labels):
        ax.plot(wavelengths, corr, lw=1.2, color=color, label=label)
        ax.axhline(0, color='black', lw=0.6)
        ax.axhline( 0.3, color='red',  lw=0.6, linestyle=':', alpha=0.6)
        ax.axhline(-0.3, color='red',  lw=0.6, linestyle=':', alpha=0.6)
        for wl in [540, 560, 577, 659]:
            ax.axvline(wl, color='green', lw=0.6, linestyle='--', alpha=0.5)
        ax.set_ylabel('Pearson r')
        ax.set_ylim(-0.7, 0.7)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Wavelength (nm)')
    plt.suptitle(f'各波長 × ClinicHb 相關性  (N={n_samples})', fontsize=13)
    plt.tight_layout()
    plt.savefig('derivatives_corr_overview.png', dpi=150)
    plt.close()
    print('>>> 相關性總覽圖 → derivatives_corr_overview.png')


def plot_top_features(records, hb, corr_d1, corr_d2, wavelengths):
    """d1 和 d2 最高相關波長的散佈圖"""
    top_d1_idx = np.argsort(np.abs(corr_d1))[-3:][::-1]
    top_d2_idx = np.argsort(np.abs(corr_d2))[-3:][::-1]

    d1_mat = np.array([r['d1'] for r in records])
    d2_mat = np.array([r['d2'] for r in records])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, idx in enumerate(top_d1_idx):
        ax = axes[0, col]
        wl = wavelengths[idx]
        vals = d1_mat[:, idx]
        r, p = stats.pearsonr(vals, hb)
        ax.scatter(vals, hb, alpha=0.5, color='darkorange', edgecolors='none', s=20)
        m, b = np.polyfit(vals, hb, 1)
        xs = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, m * xs + b, 'r--', lw=1.5)
        ax.set_title(f'd1 @ {wl}nm  r={r:+.3f}')
        ax.set_xlabel('d1 value'); ax.set_ylabel('ClinicHb')
        ax.axhline(10, color='green', lw=0.8, linestyle=':')
        ax.grid(True, alpha=0.4)

    for col, idx in enumerate(top_d2_idx):
        ax = axes[1, col]
        wl = wavelengths[idx]
        vals = d2_mat[:, idx]
        r, p = stats.pearsonr(vals, hb)
        ax.scatter(vals, hb, alpha=0.5, color='purple', edgecolors='none', s=20)
        m, b = np.polyfit(vals, hb, 1)
        xs = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, m * xs + b, 'r--', lw=1.5)
        ax.set_title(f'd2 @ {wl}nm  r={r:+.3f}')
        ax.set_xlabel('d2 value'); ax.set_ylabel('ClinicHb')
        ax.axhline(10, color='green', lw=0.8, linestyle=':')
        ax.grid(True, alpha=0.4)

    plt.suptitle('Top-3 相關波長散佈圖  (上: d1  下: d2)', fontsize=13)
    plt.tight_layout()
    plt.savefig('derivatives_top_scatter.png', dpi=150)
    plt.close()
    print('>>> Top 特徵散佈圖 → derivatives_top_scatter.png')


# ==========================================
# 主程式
# ==========================================
def main():
    if not os.path.isdir(MUA_FOLDER):
        print(f'找不到光譜資料夾: {MUA_FOLDER}'); return

    records = load_all(BASE_DIR, MUA_FOLDER)
    print(f"\n>>> 共載入 {len(records)} 筆")
    if len(records) == 0: return

    hb          = np.array([r['hb']  for r in records])
    raw_mat     = np.array([r['raw'] for r in records])
    cA_mat      = np.array([r['cA']  for r in records])
    d1_mat      = np.array([r['d1']  for r in records])
    d2_mat      = np.array([r['d2']  for r in records])
    wavelengths = np.arange(500 + RANGE_START, 500 + RANGE_END)

    corr_raw = corr_curve(raw_mat, hb)
    corr_cA  = corr_curve(cA_mat,  hb)
    corr_d1  = corr_curve(d1_mat,  hb)
    corr_d2  = corr_curve(d2_mat,  hb)

    # ── 印出重點波長相關性 ──
    print(f"\n{'='*60}")
    print(f"  重點波長相關性  (N={len(records)})")
    print(f"{'='*60}")
    print(f"  {'波長':<10} {'raw':>8} {'cA':>8} {'d1':>8} {'d2':>8}")
    for name, idx in KEY_IDX.items():
        rel_idx = idx - RANGE_START
        if 0 <= rel_idx < len(corr_raw):
            print(f"  {name:<10} {corr_raw[rel_idx]:>+8.4f} {corr_cA[rel_idx]:>+8.4f} "
                  f"{corr_d1[rel_idx]:>+8.4f} {corr_d2[rel_idx]:>+8.4f}")

    # ── Top-5 最高相關特徵 ──
    print(f"\n  Top-5 d1 相關波長:")
    for i in np.argsort(np.abs(corr_d1))[-5:][::-1]:
        print(f"    {500 + RANGE_START + i}nm  r={corr_d1[i]:+.4f}")

    print(f"\n  Top-5 d2 相關波長:")
    for i in np.argsort(np.abs(corr_d2))[-5:][::-1]:
        print(f"    {500 + RANGE_START + i}nm  r={corr_d2[i]:+.4f}")

    plot_corr_overview(wavelengths,
                       {'raw': corr_raw, 'cA': corr_cA, 'd1': corr_d1, 'd2': corr_d2},
                       len(records))
    plot_top_features(records, hb, corr_d1, corr_d2, wavelengths)


if __name__ == '__main__':
    main()
