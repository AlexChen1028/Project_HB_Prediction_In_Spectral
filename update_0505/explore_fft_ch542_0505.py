"""
FFT 分析：channel 542nm (index 42)
─────────────────────────────────
對每個光譜檔，取 542nm 處所有重複量測值做 FFT，
觀察各頻率成分的振幅是否與 ClinicHb 相關。

同時也對整條平均光譜做 FFT，提取頻域特徵後計算相關性。
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER = os.path.join(BASE_DIR, 'mua')

CH_542_IDX = 42    # 542nm = 500nm + 42
SPEC_LEN   = 896


# ==========================================
# 工具函數
# ==========================================
def _clean_bed(val):
    if pd.isnull(val): return ''
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f'{m.group(1)}{m.group(2)}' if m else val


def load_all(base_dir, mua_folder):
    """
    回傳 list of dict:
      hb        : float
      fft_temporal : FFT 振幅 (重複量測的時間序列 @ 542nm)
      fft_spectrum : FFT 振幅 (整條平均光譜)
      n_meas    : 重複量測次數
    """
    records = []
    excel_cache = {}
    files = [f for f in os.listdir(mua_folder) if f.endswith('.txt')]

    for f_name in tqdm(files, desc='Extracting FFT features'):
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
        hb = float(row.iloc[0]['ClinicHb'])

        data = np.loadtxt(os.path.join(mua_folder, f_name), delimiter='\t')
        # data shape: (n_wavelengths, 1+n_measurements)  col0=wavelength labels
        measurements = data[:, 1:]   # (n_wavelengths, n_meas)
        v = np.mean(measurements, axis=1)
        if len(v) < SPEC_LEN: v = np.pad(v, (0, SPEC_LEN - len(v)), mode='edge')
        else:                  v = v[:SPEC_LEN]

        # ── FFT 方案 A：對 542nm 處的重複量測做時間域 FFT ──
        if data.shape[0] > CH_542_IDX and measurements.shape[1] >= 4:
            ts = measurements[CH_542_IDX, :]          # 時間序列
            fft_t = np.abs(np.fft.rfft(ts - ts.mean())) / len(ts)
        else:
            fft_t = np.array([np.nan])

        # ── FFT 方案 B：對整條平均光譜做 FFT（頻域形狀特徵）──
        fft_s = np.abs(np.fft.rfft(v)) / len(v)      # 振幅譜

        records.append({
            'hb':           hb,
            'fft_temporal': fft_t,
            'fft_spectrum': fft_s,
            'n_meas':       measurements.shape[1],
            'v542':         v[CH_542_IDX],            # 542nm 平均值（對照用）
        })

    return records


# ==========================================
# 相關性分析 & 繪圖
# ==========================================
def analyse(records):
    hb    = np.array([r['hb']    for r in records])
    v542  = np.array([r['v542']  for r in records])

    # ── 方案 A：時間域 FFT（若資料足夠）──
    valid_t = [r for r in records if not np.isnan(r['fft_temporal'][0])]
    if valid_t:
        n_meas = min(r['fft_temporal'].shape[0] for r in valid_t)
        fft_t  = np.array([r['fft_temporal'][:n_meas] for r in valid_t])
        hb_t   = np.array([r['hb'] for r in valid_t])
        corr_t = [stats.pearsonr(fft_t[:, k], hb_t)[0] for k in range(n_meas)]
        freqs_t = np.fft.rfftfreq(valid_t[0]['fft_temporal'].shape[0] * 2 - 2)[:n_meas]

        plt.figure(figsize=(10, 4))
        plt.bar(range(n_meas), corr_t, color='steelblue', alpha=0.8)
        plt.axhline(0, color='red', lw=1)
        plt.xlabel('FFT frequency index (temporal @ 542nm)')
        plt.ylabel("Pearson r with ClinicHb")
        plt.title(f"Temporal FFT @ 542nm vs ClinicHb  (N={len(valid_t)})")
        plt.tight_layout(); plt.savefig('fft_temporal_ch542_corr.png', dpi=150); plt.close()
        print(f">>> [Method A] Temporal FFT corr plot -> fft_temporal_ch542_corr.png")
        print(f"    Max |r|: {max(abs(c) for c in corr_t):.4f} (freq idx {np.argmax(np.abs(corr_t))})")
    else:
        print(">>> [Method A] <4 repeated measurements, skipping temporal FFT")

    # ── 方案 B：光譜 FFT 相關性 ──
    n_fft = min(r['fft_spectrum'].shape[0] for r in records)
    fft_s = np.array([r['fft_spectrum'][:n_fft] for r in records])
    corr_s = [stats.pearsonr(fft_s[:, k], hb)[0] for k in range(n_fft)]

    top_k   = np.argsort(np.abs(corr_s))[-10:][::-1]

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(corr_s, lw=1.2, color='darkorange')
    plt.axhline(0, color='red', lw=0.8)
    plt.xlabel('FFT coefficient index (spectral domain)')
    plt.ylabel('Pearson r with ClinicHb')
    plt.title(f'Spectral FFT coef vs ClinicHb  (N={len(records)})')
    plt.grid(True, alpha=0.4)

    plt.subplot(1, 2, 2)
    best_idx = top_k[0]
    plt.scatter(fft_s[:, best_idx], hb, alpha=0.5, color='teal', edgecolors='none')
    r_val, p_val = stats.pearsonr(fft_s[:, best_idx], hb)
    plt.xlabel(f'FFT coef [{best_idx}] amplitude')
    plt.ylabel('ClinicHb (g/dL)')
    plt.title(f'Top correlated FFT coef [{best_idx}]  r={r_val:.3f}  p={p_val:.3e}')
    plt.grid(True, alpha=0.4)

    plt.tight_layout(); plt.savefig('fft_spectrum_corr.png', dpi=150); plt.close()
    print(f">>> [Method B] Spectral FFT corr plot -> fft_spectrum_corr.png")
    print(f"    Top-5 FFT coef index: {top_k[:5].tolist()}")
    print(f"    Corresponding |r|: {[f'{abs(corr_s[i]):.4f}' for i in top_k[:5]]}")

    r_raw, p_raw = stats.pearsonr(v542, hb)
    print(f"\n>>> v542 raw value vs HB: r={r_raw:.4f}  p={p_raw:.3e}")


if __name__ == '__main__':
    if not os.path.isdir(MUA_FOLDER):
        print(f'Spectrum folder not found: {MUA_FOLDER}'); exit()
    records = load_all(BASE_DIR, MUA_FOLDER)
    print(f"\n>>> Loaded {len(records)} samples")
    analyse(records)
