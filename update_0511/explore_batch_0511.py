"""
Batch effect analysis: check whether d2 features differ systematically across
collection dates, independent of ClinicHb.

Outputs:
  batch_pca_0511.png   – PCA colored by date vs HB group
  batch_feat_0511.png  – d2 feature distributions & HB distributions per date
  batch_stat_0511.png  – partial correlation and per-date scatter
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, re, unicodedata, pywt
from tqdm import tqdm
from scipy import stats
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
D2_INDICES   = [37, 40, 60, 77]
RANGE_END    = 300
FEAT_LABELS  = ['d2@537', 'd2@540', 'd2@560', 'd2@577']
DATE_COLORS  = ['steelblue', 'darkorange', 'green', 'purple']


# ── Utilities ─────────────────────────────────────────────────
def _clean_bed(val):
    if pd.isnull(val): return ""
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f"{m.group(1)}{m.group(2)}" if m else val


def _swt_d2_feat(v):
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
    return cA3[:RANGE_END], feat


# ── Data Loading ──────────────────────────────────────────────
def load_all(base_dir, mua_folder):
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
        mean_v = raw[:, 1:].mean(axis=1)
        if len(mean_v) < SPEC_LEN:
            mean_v = np.pad(mean_v, (0, SPEC_LEN - len(mean_v)), mode='edge')

        cA3, d2_feat = _swt_d2_feat(mean_v)
        records.append({
            'hb':      float(row.iloc[0]['ClinicHb']),
            'date':    date_str,
            'cA3':     cA3.astype(np.float32),
            'd2_feat': d2_feat,
        })

    return records


# ── Figure 1: PCA colored by date vs HB group ─────────────────
def plot_pca_batch(records, dates_sorted, fig_path):
    hb      = np.array([r['hb']   for r in records])
    cA3_mat = np.array([r['cA3']  for r in records])
    date_arr = np.array([r['date'] for r in records])
    low_mask = hb < HB_THRESHOLD

    pca = PCA(n_components=2)
    z   = pca.fit_transform(StandardScaler().fit_transform(cA3_mat))
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by date
    ax = axes[0]
    for date, col in zip(dates_sorted, DATE_COLORS):
        mask = date_arr == date
        ax.scatter(z[mask, 0], z[mask, 1], c=col, alpha=0.7, edgecolors='none',
                   s=35, label=f'{date}  (n={mask.sum()})')
    ax.set_title(f'PCA of SWT cA3 — colored by DATE\n'
                 f'PC1 {evr[0]:.1%} + PC2 {evr[1]:.1%}')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    # Right: colored by HB group
    ax = axes[1]
    for mask, col, lab in [(low_mask, 'tomato', 'HB<10'), (~low_mask, 'steelblue', 'HB≥10')]:
        ax.scatter(z[mask, 0], z[mask, 1], c=col, alpha=0.7, edgecolors='none',
                   s=35, label=f'{lab} (n={mask.sum()})')
    ax.set_title(f'PCA of SWT cA3 — colored by HB group\n'
                 f'PC1 {evr[0]:.1%} + PC2 {evr[1]:.1%}')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle('Batch Effect Check: Does date drive PCA clustering?', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'>>> Saved: {fig_path}')


# ── Figure 2: Feature & HB distributions per date ─────────────
def plot_feat_dist(records, dates_sorted, fig_path):
    hb       = np.array([r['hb']   for r in records])
    d2_4     = np.array([r['d2_feat'] for r in records])
    date_arr = np.array([r['date'] for r in records])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 0: d2 feature distributions per date (4 features → 4 box groups + 1 HB)
    for fi, (lab, ax) in enumerate(zip(FEAT_LABELS, axes[0, :3])):
        data_per_date = [d2_4[date_arr == d, fi] for d in dates_sorted]
        bp = ax.boxplot(data_per_date, labels=dates_sorted,
                        patch_artist=True,
                        medianprops=dict(color='red', lw=2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4))
        for patch, col in zip(bp['boxes'], DATE_COLORS):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        ax.set_title(f'{lab} distribution per date')
        ax.set_ylabel('d2 value')
        ax.tick_params(axis='x', labelsize=8, rotation=15)
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    # Row 1, col 0: HB distribution per date
    ax = axes[1, 0]
    data_hb = [hb[date_arr == d] for d in dates_sorted]
    bp = ax.boxplot(data_hb, labels=dates_sorted,
                    patch_artist=True,
                    medianprops=dict(color='red', lw=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))
    for patch, col in zip(bp['boxes'], DATE_COLORS):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(HB_THRESHOLD, color='black', lw=1, linestyle='--', alpha=0.6)
    ax.set_title('ClinicHb distribution per date')
    ax.set_ylabel('ClinicHb (g/dL)')
    ax.tick_params(axis='x', labelsize=8, rotation=15)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    # Row 1, col 1: HB<10 ratio per date (stacked bar)
    ax = axes[1, 1]
    ratios_low  = [(hb[date_arr == d] < HB_THRESHOLD).mean() for d in dates_sorted]
    ratios_high = [1 - r for r in ratios_low]
    x = np.arange(len(dates_sorted))
    ax.bar(x, ratios_high, color='steelblue', alpha=0.8, label='HB≥10')
    ax.bar(x, ratios_low,  bottom=ratios_high, color='tomato', alpha=0.8, label='HB<10')
    ax.set_xticks(x); ax.set_xticklabels(dates_sorted, fontsize=8, rotation=15)
    ax.set_ylabel('Fraction of samples')
    ax.set_title('HB class ratio per date')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    for xi, (rl, rh) in enumerate(zip(ratios_low, ratios_high)):
        n = (date_arr == dates_sorted[xi]).sum()
        ax.text(xi, 0.5, f'n={n}\n{rl:.0%} low', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')

    # Row 1, col 2: sample count per date
    ax = axes[1, 2]
    counts = [(date_arr == d).sum() for d in dates_sorted]
    bars = ax.bar(range(len(dates_sorted)), counts,
                  color=DATE_COLORS[:len(dates_sorted)], alpha=0.85)
    ax.set_xticks(range(len(dates_sorted)))
    ax.set_xticklabels(dates_sorted, fontsize=8, rotation=15)
    ax.set_ylabel('Number of samples'); ax.set_title('Sample count per date')
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Feature & HB Distributions per Collection Date', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'>>> Saved: {fig_path}')


# ── Figure 3: Per-date scatter + partial correlation ──────────
def plot_stat(records, dates_sorted, fig_path):
    hb       = np.array([r['hb']      for r in records])
    d2_4     = np.array([r['d2_feat'] for r in records])
    date_arr = np.array([r['date']    for r in records])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── [0,0]: d2@537 vs HB, one color per date ──
    ax = axes[0, 0]
    for date, col in zip(dates_sorted, DATE_COLORS):
        mask = date_arr == date
        ax.scatter(d2_4[mask, 0], hb[mask], c=col, alpha=0.6, s=25,
                   edgecolors='none', label=date)
    # Overall regression line
    m, b, r, p, _ = stats.linregress(d2_4[:, 0], hb)
    xs = np.linspace(d2_4[:, 0].min(), d2_4[:, 0].max(), 200)
    ax.plot(xs, m * xs + b, 'k--', lw=1.5, label=f'Overall r={r:+.3f}')
    ax.axhline(HB_THRESHOLD, color='gray', lw=0.8, linestyle=':')
    ax.set_xlabel('d2@537'); ax.set_ylabel('ClinicHb (g/dL)')
    ax.set_title('d2@537 vs HB  (colored by date)')
    ax.legend(fontsize=8); ax.grid(True, linestyle=':', alpha=0.5)

    # ── [0,1]: per-date Pearson r for each feature ──
    ax = axes[0, 1]
    x = np.arange(4)
    w = 0.8 / len(dates_sorted)
    for di, (date, col) in enumerate(zip(dates_sorted, DATE_COLORS)):
        mask = date_arr == date
        rs   = [stats.pearsonr(d2_4[mask, fi], hb[mask])[0]
                for fi in range(4)]
        offset = (di - len(dates_sorted) / 2 + 0.5) * w
        ax.bar(x + offset, rs, w * 0.85, label=date, color=col, alpha=0.8)
    # Overall r
    overall_r = [stats.pearsonr(d2_4[:, fi], hb)[0] for fi in range(4)]
    ax.plot(x, overall_r, 'ko-', lw=1.5, ms=6, label='Overall', zorder=5)
    ax.axhline(0, color='black', lw=0.6)
    ax.axhline( 0.3, color='red', lw=0.8, linestyle=':', alpha=0.6)
    ax.axhline(-0.3, color='red', lw=0.8, linestyle=':', alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(FEAT_LABELS, fontsize=10)
    ax.set_ylabel('Pearson r  with ClinicHb')
    ax.set_title('Per-date vs Overall Correlation')
    ax.legend(fontsize=8); ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    # ── [1,0]: Kruskal-Wallis test — are features different across dates? ──
    ax = axes[1, 0]
    kw_stats, kw_pvals = [], []
    for fi in range(4):
        groups = [d2_4[date_arr == d, fi] for d in dates_sorted]
        H, p = stats.kruskal(*groups)
        kw_stats.append(H); kw_pvals.append(p)

    bar_cols = ['green' if p > 0.05 else 'red' for p in kw_pvals]
    bars = ax.bar(range(4), kw_pvals, color=bar_cols, alpha=0.8)
    ax.axhline(0.05, color='red', lw=1.5, linestyle='--', label='p = 0.05')
    ax.set_xticks(range(4)); ax.set_xticklabels(FEAT_LABELS, fontsize=10)
    ax.set_ylabel('Kruskal-Wallis p-value')
    ax.set_title('Batch Effect Test: p < 0.05 = significant date difference\n'
                 '(green = OK, red = batch effect detected)')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    for bar, p, H in zip(bars, kw_pvals, kw_stats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'p={p:.3f}\nH={H:.1f}', ha='center', va='bottom', fontsize=8)

    # ── [1,1]: Partial correlation — remove date mean before correlating ──
    ax = axes[1, 1]
    # Remove date-level mean from each feature and HB
    d2_demean = d2_4.copy()
    hb_demean = hb.copy()
    for date in dates_sorted:
        mask = date_arr == date
        d2_demean[mask] -= d2_4[mask].mean(axis=0)
        hb_demean[mask] -= hb[mask].mean()

    raw_r     = [stats.pearsonr(d2_4[:, fi],      hb)[0]      for fi in range(4)]
    partial_r = [stats.pearsonr(d2_demean[:, fi], hb_demean)[0] for fi in range(4)]

    x = np.arange(4); w = 0.38
    ax.bar(x - w / 2, raw_r,     w, label='Raw correlation',     color='steelblue',  alpha=0.85)
    ax.bar(x + w / 2, partial_r, w, label='Partial (date removed)', color='darkorange', alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline( 0.3, color='red', lw=0.8, linestyle=':', alpha=0.6)
    ax.axhline(-0.3, color='red', lw=0.8, linestyle=':', alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(FEAT_LABELS, fontsize=10)
    ax.set_ylabel('Pearson r  with ClinicHb')
    ax.set_title('Partial Correlation\n(after removing date mean — true signal vs batch)')
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    plt.suptitle('Batch Effect Statistical Analysis', fontsize=13)
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

    hb       = np.array([r['hb']   for r in records])
    date_arr = np.array([r['date'] for r in records])
    d2_4     = np.array([r['d2_feat'] for r in records])
    dates_sorted = sorted(set(date_arr))

    print(f'\n  Collection dates: {dates_sorted}')
    print(f'\n  {"Date":<12} {"N":>5} {"HB mean":>9} {"HB std":>8} {"Low%":>7}')
    print(f'  {"-"*46}')
    for date in dates_sorted:
        mask = date_arr == date
        h    = hb[mask]
        print(f'  {date:<12} {mask.sum():>5} {h.mean():>9.2f} {h.std():>8.2f} '
              f'{(h < HB_THRESHOLD).mean():>7.1%}')

    print(f'\n  Kruskal-Wallis test (batch effect on d2 features):')
    for fi, lab in enumerate(FEAT_LABELS):
        groups  = [d2_4[date_arr == d, fi] for d in dates_sorted]
        H, p    = stats.kruskal(*groups)
        verdict = 'SIGNIFICANT (batch effect!)' if p < 0.05 else 'not significant'
        print(f'    {lab}: H={H:.2f}  p={p:.4f}  → {verdict}')

    # Partial correlation
    d2_dm = d2_4.copy(); hb_dm = hb.copy()
    for date in dates_sorted:
        mask = date_arr == date
        d2_dm[mask] -= d2_4[mask].mean(axis=0)
        hb_dm[mask] -= hb[mask].mean()
    print(f'\n  Partial correlation (date mean removed):')
    for fi, lab in enumerate(FEAT_LABELS):
        r_raw  = stats.pearsonr(d2_4[:, fi], hb)[0]
        r_part = stats.pearsonr(d2_dm[:, fi], hb_dm)[0]
        print(f'    {lab}: raw r={r_raw:+.3f}  partial r={r_part:+.3f}')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_pca_batch(records, dates_sorted, os.path.join(out_dir, 'batch_pca_0511.png'))
    plot_feat_dist(records, dates_sorted, os.path.join(out_dir, 'batch_feat_0511.png'))
    plot_stat(     records, dates_sorted, os.path.join(out_dir, 'batch_stat_0511.png'))

    print('\n>>> Done.')


if __name__ == '__main__':
    main()
