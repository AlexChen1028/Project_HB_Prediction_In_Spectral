"""
光譜配對後 HB 分布統計
─────────────────────
以光譜檔 (.txt) 為基準配對 Excel，統計：
  - 配對成功筆數 / 病人數
  - HB < 10 / HB >= 10 各組樣本與病人數
  - 4:1 train/test 預計分配
"""
import pandas as pd
import numpy as np
import glob
import re
import unicodedata
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER  = os.path.join(BASE_DIR, 'mua')

HB_THRESHOLD = 10.0
TRAIN_RATIO  = 4    # train : test = 4 : 1

# ── 工具函數 ──────────────────────────────────────────
def clean_bed(val):
    if pd.isnull(val): return ''
    val = unicodedata.normalize('NFKC', str(val)).upper()
    val = re.sub(r'[^A-Z0-9]', '', val)
    m = re.search(r'([A-Z])0*(\d+)', val)
    return f'{m.group(1)}{m.group(2)}' if m else val

# ── 讀入所有 Excel ────────────────────────────────────
excel_cache = {}
for path in glob.glob(os.path.join(BASE_DIR, '2026*_dialysis_table_export.xlsx')):
    df = pd.read_excel(path, engine='openpyxl')
    df.columns = df.columns.str.strip()
    if not all(c in df.columns for c in ['DialysisBed', 'Shift', 'ClinicHb']):
        continue
    df['Bed_C']   = df['DialysisBed'].apply(clean_bed)
    df['Shift_C'] = df['Shift'].apply(
        lambda x: '早' if '早' in str(x) else ('午' if '午' in str(x) else '晚'))
    key = re.search(r'(\d{8})', os.path.basename(path))
    if key: excel_cache[key.group(1)] = df

if not excel_cache:
    print(f'找不到 Excel，請確認路徑: {BASE_DIR}'); exit()

# ── 掃描光譜檔，配對 Excel ─────────────────────────────
if not os.path.isdir(MUA_FOLDER):
    print(f'找不到光譜資料夾: {MUA_FOLDER}'); exit()

records = []
txt_files = [f for f in os.listdir(MUA_FOLDER) if f.endswith('.txt')]

for fname in txt_files:
    norm = unicodedata.normalize('NFKC', fname).lower()
    d_m  = re.search(r'(\d{4})_(\d{2})_(\d{2})', norm)
    sb_m = re.search(r'(morning|afternoon|evening)_([a-z]+)0*(\d+)', norm)
    if not (d_m and sb_m): continue

    date_str = f"{d_m.group(1)}{d_m.group(2)}{d_m.group(3)}"
    shift    = {'morning': '早', 'afternoon': '午', 'evening': '晚'}[sb_m.group(1)]
    bed      = clean_bed(f"{sb_m.group(2)}{sb_m.group(3)}")
    pid      = f"{bed}_{shift}"

    df = excel_cache.get(date_str)
    if df is None: continue

    row = df[(df['Bed_C'] == bed) & (df['Shift_C'] == shift)]
    if row.empty or pd.isnull(row.iloc[0]['ClinicHb']): continue

    records.append({'fname': fname, 'date': date_str, 'pid': pid,
                    'hb': float(row.iloc[0]['ClinicHb'])})

if not records:
    print('沒有配對到任何記錄，請確認檔名格式'); exit()

df_all = pd.DataFrame(records)

# ── 分兩組 ────────────────────────────────────────────
grp_low  = df_all[df_all['hb'] <  HB_THRESHOLD]
grp_high = df_all[df_all['hb'] >= HB_THRESHOLD]

def summary(grp, name):
    n_s = len(grp); n_p = grp['pid'].nunique()
    n_t = max(1, n_p // (TRAIN_RATIO + 1)) if n_p > 0 else 0
    n_r = n_p - n_t
    print(f'  {name}')
    print(f'    樣本數:  {n_s:4d} 筆')
    print(f'    病人數:  {n_p:4d} 人')
    if n_p > 0:
        print(f'    → Train: {n_r} 人  ({n_r/n_p*100:.0f}%)')
        print(f'    → Test:  {n_t} 人  ({n_t/n_p*100:.0f}%)')
        print(f'    HB 範圍: {grp["hb"].min():.1f} ~ {grp["hb"].max():.1f}  '
              f'(mean={grp["hb"].mean():.2f})')
    print()

print('=' * 58)
print('  光譜配對後 HB 分布統計')
print('=' * 58)
print(f'  光譜檔總數:  {len(txt_files):4d} 個')
print(f'  配對成功:    {len(df_all):4d} 筆')
print(f'  HB min/max/mean: {df_all["hb"].min():.1f} / '
      f'{df_all["hb"].max():.1f} / {df_all["hb"].mean():.2f}')
print(f'  train : test = {TRAIN_RATIO} : 1')
print()

summary(grp_low,  f'HB <  {HB_THRESHOLD}')
summary(grp_high, f'HB >= {HB_THRESHOLD}  (含 ==10)')

print('=' * 58)
print('  各日期明細（配對後）')
print('=' * 58)
for date, g in df_all.groupby('date'):
    lo = (g['hb'] <  HB_THRESHOLD).sum()
    hi = (g['hb'] >= HB_THRESHOLD).sum()
    print(f'  {date}:  <10={lo}筆  >=10={hi}筆  合計={len(g)}筆')

print()
print(f'  未配對光譜（無對應 Excel 或無 ClinicHb）: '
      f'{len(txt_files) - len(df_all)} 個')
