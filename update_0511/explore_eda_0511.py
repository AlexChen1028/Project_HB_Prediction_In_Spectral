"""
EDA: PCA separability, temporal noise analysis, denoising evaluation
─────────────────────────────────────────────────────────────────────
Outputs:
  eda_pca_0511.png     – PCA 2D projection to check HB group separability
  eda_noise_0511.png   – 3-segment temporal SNR analysis
  eda_denoise_0511.png – SavGol denoising vs raw SWT comparison
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, re, unicodedata, pywt
from tqdm import tqdm
from scipy import stats, signal as sp_signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Parameters ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER   = os.path.join(BASE_DIR, 'mua')
HB_THRESHOLD = 10.0
SWT_LEVEL    = 3
SWT_WAVELET  = 'db4'
WINDOW_W     = 2
SPEC_LEN     = 896
D2_INDICES   = [37, 40, 60, 77]   # 537, 540, 560, 577 nm
RANGE_END    = 300                 # use 500–799 nm for PCA / correlation
SAVGOL_WIN   = 11
SAVGOL_ORDER = 3
WAVELENGTHS  = np.arange(500, 500 + RANGE_END)
FEAT_LABELS  = ['d2@537', 'd2@540', 'd2@560', 'd2@577']


# ── Core spectral feature extraction ─────────────────────────
def _swt_d2(v):
    """1-D mean spectrum → (cA3[:RANGE_END], d2[:RANGE_END], d2_feat[4])"""
    if len(v) < SPEC_LEN:
        v = np.pad(v, (0, SPEC_LEN - len(v)), mode='edge')
    else:
        v = v[:SPEC_LEN]
    coeffs = pywt.swt(v, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    cA3 = coeffs[0][0]
    d2  = np.gradient(np.gradient(cA3))
    feat = np.array(
        [np.mean(d2[max(0, i - WINDOW_W): i + WINDOW_W + 1]) for i in D2_INDICES],
        dtype=np.float32)
    return cA3[:RANGE_END], d2[:RANGE_END], feat


def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


# ── Data Loading ──────────────────────────────────────────────
def load_all(base_dir, mua_folder):
    """
    Load all matched samples. Returns list of dicts containing spectral
    features, temporal segment features, and HB label.
    """
    records, cache = [], {}
    files = sorted([f for f in os.listdir(mua_folder) if f.endswith('.txt')])

    for fname in tqdm(files, desc='Loading'):
        norm = unicodedata.normalize('NFKC', fname).lower()
        dm   = re.search(r'(\d{4})_(\d{2})_(\d{2})', norm)
        sbm  = re.search(r'(morning|afternoon|evening)_([a-z]*)(\d+)', norm)
        if not (dm and sbm): continue

        date_str = f"{dm.group(1)}{dm.group(2)}{dm.group(3)}"
        shift    = {'morning': '早', 'afternoon': '午', 'evening': '晚'}[sbm.group(1)]
        bed      = _clean_bed(f"{sbm.group(2)}{sbm.group(3)}")

        if date_str not in cache:
            p = os.path.join(base_dir, f"{date_str}_dialysis_table_export.xlsx")
            if os.path.exists(p):
                df = pd.read_excel(p, engine='openpyxl')
                df.columns = df.columns.str.strip()
                df['Bed_C']   = df['DialysisBed'].apply(_clean_bed)
                df['Shift_C'] = df['Shift'].apply(
                    lambda x: '早' if '早' in str(x) else ('午' if '午' in str(x) else '晚'))
                cache[date_str] = df
            else:
                cache[date_str] = None

        df = cache[date_str]
        if df is None: continue
        row = df[(df['Bed_C'] == bed) & (df['Shift_C'] == shift)]
        if row.empty or pd.isnull(row.iloc[0]['ClinicHb']): continue

        raw = np.loadtxt(os.path.join(mua_folder, fname), delimiter='\t')
        # rows = wavelength, cols = time measurements (col 0 skipped)
        time_cols = raw[:, 1:]
        n_wl      = min(time_cols.shape[0], SPEC_LEN)
        time_cols = time_cols[:n_wl, :]
        if time_cols.shape[1] == 0: continue

        N_time = time_cols.shape[1]

        # Mean spectrum (pad to SPEC_LEN if needed)
        mean_v = time_cols.mean(axis=1)
        if len(mean_v) < SPEC_LEN:
            mean_v = np.pad(mean_v, (0, SPEC_LEN - len(mean_v)), mode='edge')

        # SWT features on mean spectrum
        cA3, d2_spec, d2_feat = _swt_d2(mean_v)

        # SavGol-smoothed spectrum → SWT features
        v_sg = sp_signal.savgol_filter(mean_v, SAVGOL_WIN, SAVGOL_ORDER)
        cA3_sg, d2_sg_spec, d2_sg_feat = _swt_d2(v_sg)

        # 3-segment temporal split
        if N_time >= 3:
            t = N_time // 3
            seg_cols = [time_cols[:, :t],
                        time_cols[:, t:2 * t],
                        time_cols[:, 2 * t:3 * t]]
        else:
            seg_cols = [time_cols, time_cols, time_cols]

        def seg_feats(use_savgol):
            rows = []
            for sc in seg_cols:
                v = sc.mean(axis=1)
                if len(v) < SPEC_LEN:
                    v = np.pad(v, (0, SPEC_LEN - len(v)), mode='edge')
                if use_savgol:
                    v = sp_signal.savgol_filter(v, SAVGOL_WIN, SAVGOL_ORDER)
                rows.append(_swt_d2(v)[2])
            return np.array(rows)   # (3, 4)

        records.append({
            'hb':          float(row.iloc[0]['ClinicHb']),
            'pid':         f"{bed}_{shift}",
            'raw_v':       mean_v[:RANGE_END].astype(np.float32),
            'cA3':         cA3.astype(np.float32),
            'd2_spec':     d2_spec.astype(np.float32),
            'd2_sg_spec':  d2_sg_spec.astype(np.float32),
            'd2_feat':     d2_feat,
            'd2_sg_feat':  d2_sg_feat,
            'seg_d2':      seg_feats(False),
            'seg_d2_sg':   seg_feats(True),
            'n_time':      N_time,
        })

    return records


# ── Figure 1: PCA separability ────────────────────────────────
def plot_pca(records, hb, fig_path):
    low  = hb < HB_THRESHOLD
    high = ~low

    cA3_mat = np.array([r['cA3']      for r in records])   # (N, RANGE_END)
    d2_mat  = np.array([r['d2_spec']  for r in records])   # (N, RANGE_END)
    d2_4    = np.array([r['d2_feat']  for r in records])   # (N, 4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── [0,0]: PCA of SWT cA3, colored by HB group ──
    ax = axes[0, 0]
    pca = PCA(n_components=2)
    z   = pca.fit_transform(StandardScaler().fit_transform(cA3_mat))
    evr = pca.explained_variance_ratio_
    for mask, col, lab in [(low, 'tomato', 'HB<10'), (high, 'steelblue', 'HB≥10')]:
        ax.scatter(z[mask, 0], z[mask, 1], c=col, alpha=0.7, edgecolors='none',
                   s=30, label=f'{lab} (n={mask.sum()})')
    ax.set_title(f'PCA of SWT cA3  [{evr[0]:.1%} + {evr[1]:.1%} = {evr[:2].sum():.1%}]')
    ax.set_xlabel(f'PC1 ({evr[0]:.1%})'); ax.set_ylabel(f'PC2 ({evr[1]:.1%})')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    # ── [0,1]: PCA of SWT cA3, continuous HB colormap ──
    ax = axes[0, 1]
    sc = ax.scatter(z[:, 0], z[:, 1], c=hb, cmap='RdYlGn',
                    alpha=0.8, edgecolors='none', s=30,
                    vmin=hb.min(), vmax=hb.max())
    plt.colorbar(sc, ax=ax, label='ClinicHb (g/dL)')
    ax.set_title('PCA of SWT cA3  (continuous HB)')
    ax.set_xlabel(f'PC1 ({evr[0]:.1%})'); ax.set_ylabel(f'PC2 ({evr[1]:.1%})')
    ax.grid(True, linestyle=':', alpha=0.5)

    # ── [1,0]: PCA of full d2 spectrum, colored by HB group ──
    ax = axes[1, 0]
    pca2 = PCA(n_components=2)
    z2   = pca2.fit_transform(StandardScaler().fit_transform(d2_mat))
    evr2 = pca2.explained_variance_ratio_
    for mask, col, lab in [(low, 'tomato', 'HB<10'), (high, 'steelblue', 'HB≥10')]:
        ax.scatter(z2[mask, 0], z2[mask, 1], c=col, alpha=0.7, edgecolors='none',
                   s=30, label=f'{lab}')
    ax.set_title(f'PCA of d2 spectrum  [{evr2[0]:.1%} + {evr2[1]:.1%} = {evr2[:2].sum():.1%}]')
    ax.set_xlabel(f'PC1 ({evr2[0]:.1%})'); ax.set_ylabel(f'PC2 ({evr2[1]:.1%})')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    # ── [1,1]: d2@537 vs d2@540 scatter (actual model features) ──
    ax = axes[1, 1]
    for mask, col, lab in [(low, 'tomato', 'HB<10'), (high, 'steelblue', 'HB≥10')]:
        ax.scatter(d2_4[mask, 0], d2_4[mask, 1], c=col, alpha=0.7, edgecolors='none',
                   s=30, label=lab)
    ax.set_xlabel('d2 @ 537 nm'); ax.set_ylabel('d2 @ 540 nm')
    ax.set_title('Feature Scatter: d2@537 vs d2@540  (actual model inputs)')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle(f'PCA Separability Analysis  (N={len(records)})', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'>>> Saved: {fig_path}')


# ── Figure 2: Temporal noise / SNR ────────────────────────────
def plot_noise(records, hb, fig_path):
    seg_d2  = np.array([r['seg_d2']  for r in records])   # (N, 3, 4)
    mean_d2 = seg_d2.mean(axis=1)                          # (N, 4)
    noise   = seg_d2.std(axis=1)                           # (N, 4) within-session std

    signal_std = mean_d2.std(axis=0)          # (4,) between-patient spread
    avg_noise  = noise.mean(axis=0)           # (4,) mean within-session noise
    snr        = signal_std / (avg_noise + 1e-12)

    low = hb < HB_THRESHOLD
    N   = len(records)

    print(f"\n  SNR summary  (signal_std / noise_std):")
    for i, lab in enumerate(FEAT_LABELS):
        print(f"    {lab}: signal={signal_std[i]:.5f}  "
              f"noise={avg_noise[i]:.5f}  SNR={snr[i]:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── [0,0]: 3-segment d2@537 per sample, sorted by mean ──
    ax = axes[0, 0]
    fi    = 0
    order = np.argsort(mean_d2[:, fi])
    x     = np.arange(N)
    for si, (col, lab) in enumerate(zip(
            ['steelblue', 'darkorange', 'green'], ['Seg 1', 'Seg 2', 'Seg 3'])):
        ax.scatter(x, seg_d2[order, si, fi], alpha=0.45, s=12, color=col, label=lab, zorder=3)
    ax.plot(x, mean_d2[order, fi], 'k-', lw=1.2, alpha=0.8, label='Mean', zorder=4)
    ax.set_xlabel('Sample  (sorted by mean d2@537)'); ax.set_ylabel('d2@537')
    ax.set_title('3-segment spread: d2@537 per sample')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    # ── [0,1]: SNR bar chart ──
    ax = axes[0, 1]
    bars = ax.bar(range(4), snr,
                  color=['steelblue', 'darkorange', 'green', 'purple'], alpha=0.8)
    ax.axhline(1.0, color='red',    lw=1.5, linestyle='--', label='SNR=1  (signal = noise)')
    ax.axhline(2.0, color='orange', lw=1.0, linestyle=':',  label='SNR=2')
    ax.set_xticks(range(4)); ax.set_xticklabels(FEAT_LABELS, fontsize=10)
    ax.set_ylabel('SNR  =  between-patient σ  /  within-session σ')
    ax.set_title('Signal-to-Noise Ratio per d2 Feature')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    for bar, val in zip(bars, snr):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(snr) * 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ── [1,0]: Box plot of within-session noise per feature ──
    ax = axes[1, 0]
    ax.boxplot([noise[:, i] for i in range(4)],
               labels=FEAT_LABELS,
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', lw=2),
               whiskerprops=dict(lw=1.2),
               flierprops=dict(marker='o', markersize=3, alpha=0.4))
    ax.set_ylabel('Within-session std  (across 3 segments)')
    ax.set_title('Noise Distribution per Feature')
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    # ── [1,1]: mean d2@537 vs within-session noise, colored by HB group ──
    ax = axes[1, 1]
    for mask, col, lab in [(low, 'tomato', 'HB<10'), (~low, 'steelblue', 'HB≥10')]:
        ax.scatter(mean_d2[mask, 0], noise[mask, 0],
                   c=col, alpha=0.7, edgecolors='none', s=30, label=lab)
    ax.set_xlabel('Mean d2@537  (session average)'); ax.set_ylabel('Noise (σ across 3 segments)')
    ax.set_title('Feature Mean vs Noise Level  (d2@537)')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle(f'Temporal Noise Analysis  (N={N}, 3-segment split)', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'>>> Saved: {fig_path}')


# ── Figure 3: Denoising evaluation ────────────────────────────
def plot_denoise(records, hb, fig_path):
    d2_mat    = np.array([r['d2_spec']    for r in records])   # (N, RANGE_END)
    d2_sg_mat = np.array([r['d2_sg_spec'] for r in records])   # (N, RANGE_END)
    d2_4      = np.array([r['d2_feat']    for r in records])   # (N, 4)
    d2_sg_4   = np.array([r['d2_sg_feat'] for r in records])   # (N, 4)
    seg_d2    = np.array([r['seg_d2']     for r in records])   # (N, 3, 4)
    seg_d2_sg = np.array([r['seg_d2_sg']  for r in records])   # (N, 3, 4)

    noise_raw = seg_d2.std(axis=1).mean(axis=0)      # (4,)
    noise_sg  = seg_d2_sg.std(axis=1).mean(axis=0)   # (4,)

    def corr_curve(mat):
        return np.array([
            stats.pearsonr(mat[:, i], hb)[0]
            if mat[:, i].std() > 0 else 0.0
            for i in range(RANGE_END)
        ])

    corr_d2    = corr_curve(d2_mat)
    corr_d2_sg = corr_curve(d2_sg_mat)

    low = hb < HB_THRESHOLD
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── [0,0]: Example spectra — raw / SavGol / SWT cA3 ──
    ax = axes[0, 0]
    for idx, lc in zip([0, min(4, len(records) - 1)], ['steelblue', 'darkorange']):
        r  = records[idx]
        sg = sp_signal.savgol_filter(r['raw_v'], SAVGOL_WIN, SAVGOL_ORDER)
        ax.plot(WAVELENGTHS, r['raw_v'], alpha=0.35, color=lc, lw=1)
        ax.plot(WAVELENGTHS, sg,         alpha=0.80, color=lc, lw=1.4, linestyle='--')
        ax.plot(WAVELENGTHS, r['cA3'],   alpha=0.90, color=lc, lw=2.0, linestyle=':',
                label=f'HB={r["hb"]:.1f}')
    ax.legend(handles=[
        Line2D([0], [0], color='gray', lw=1,   alpha=0.4, label='Raw mean spectrum'),
        Line2D([0], [0], color='gray', lw=1.4, linestyle='--', label=f'SavGol (win={SAVGOL_WIN})'),
        Line2D([0], [0], color='gray', lw=2,   linestyle=':',  label='SWT cA3'),
    ], fontsize=8)
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Absorption (a.u.)')
    ax.set_title('Spectrum: Raw vs SavGol vs SWT cA3')
    ax.grid(True, linestyle=':', alpha=0.4)

    # ── [0,1]: d2 correlation with HB: SWT vs SavGol+SWT ──
    ax = axes[0, 1]
    ax.plot(WAVELENGTHS, corr_d2,    lw=1.5, color='steelblue',  label='d2 (SWT only)')
    ax.plot(WAVELENGTHS, corr_d2_sg, lw=1.5, color='darkorange',
            linestyle='--', label=f'd2 (SavGol+SWT)')
    ax.axhline(0,    color='black', lw=0.6)
    ax.axhline( 0.3, color='red',   lw=0.6, linestyle=':', alpha=0.6)
    ax.axhline(-0.3, color='red',   lw=0.6, linestyle=':', alpha=0.6)
    for wl in [537, 540, 560, 577]:
        ax.axvline(wl, color='green', lw=0.6, linestyle='--', alpha=0.5)
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Pearson r  with ClinicHb')
    ax.set_title('d2 Correlation with HB: SWT vs SavGol+SWT')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_ylim(-0.65, 0.65)

    # ── [1,0]: Noise comparison per feature (paired bars) ──
    ax = axes[1, 0]
    x = np.arange(4)
    w = 0.38
    b1 = ax.bar(x - w / 2, noise_raw, w, label='SWT only',   color='steelblue',  alpha=0.85)
    b2 = ax.bar(x + w / 2, noise_sg,  w, label='SavGol+SWT', color='darkorange', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(FEAT_LABELS, fontsize=10)
    ax.set_ylabel('Mean within-session σ  (3 segments)')
    ax.set_title('Intra-session Noise: SWT vs SavGol+SWT')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.4, axis='y')
    top = max(noise_raw.max(), noise_sg.max())
    for bar, val in zip(list(b1) + list(b2), list(noise_raw) + list(noise_sg)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7.5)

    # ── [1,1]: d2@537 SWT vs SavGol+SWT scatter ──
    ax = axes[1, 1]
    for mask, col, lab in [(low, 'tomato', 'HB<10'), (~low, 'steelblue', 'HB≥10')]:
        ax.scatter(d2_4[mask, 0], d2_sg_4[mask, 0],
                   c=col, alpha=0.7, edgecolors='none', s=30, label=lab)
    mn = min(d2_4[:, 0].min(), d2_sg_4[:, 0].min())
    mx = max(d2_4[:, 0].max(), d2_sg_4[:, 0].max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1, alpha=0.5, label='y=x')
    ax.set_xlabel('d2@537  (SWT only)'); ax.set_ylabel('d2@537  (SavGol+SWT)')
    ax.set_title('Feature Comparison @ 537 nm: SWT vs SavGol+SWT')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.4)

    print(f"\n  Noise reduction (SWT → SavGol+SWT):")
    for i, lab in enumerate(FEAT_LABELS):
        red = (1 - noise_sg[i] / (noise_raw[i] + 1e-12)) * 100
        print(f"    {lab}: {noise_raw[i]:.5f} → {noise_sg[i]:.5f}  ({red:+.1f}%)")

    plt.suptitle(f'Denoising Evaluation  (N={len(records)})', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'>>> Saved: {fig_path}')


# ── Main ──────────────────────────────────────────────────────
def main():
    if not os.path.isdir(MUA_FOLDER):
        print(f'MUA folder not found: {MUA_FOLDER}'); return

    records = load_all(BASE_DIR, MUA_FOLDER)
    print(f'\n>>> Loaded {len(records)} samples')
    if not records: return

    hb = np.array([r['hb'] for r in records])
    print(f'  HB  min={hb.min():.1f}  max={hb.max():.1f}  '
          f'mean={hb.mean():.1f}  std={hb.std():.2f}')
    print(f'  Low  (HB<10) : {(hb < HB_THRESHOLD).sum()}')
    print(f'  High (HB>=10): {(hb >= HB_THRESHOLD).sum()}')
    n_times = [r['n_time'] for r in records]
    print(f'  N_time per file: min={min(n_times)}  max={max(n_times)}  '
          f'median={int(np.median(n_times))}')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_pca(    records, hb, os.path.join(out_dir, 'eda_pca_0511.png'))
    plot_noise(  records, hb, os.path.join(out_dir, 'eda_noise_0511.png'))
    plot_denoise(records, hb, os.path.join(out_dir, 'eda_denoise_0511.png'))

    print('\n>>> Done. Outputs saved to update_0511/')


if __name__ == '__main__':
    main()
