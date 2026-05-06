import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import re
import unicodedata
import pywt
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm

# ==========================================
# 1. 參數設定 (嚴格對齊訓練配置)
# ==========================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUA_FOLDER  = os.path.join(BASE_DIR, 'mua')
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOW_MODEL_PATH  = 'd2_hb_low_model.pth'
HIGH_MODEL_PATH = 'd2_hb_high_model.pth'

HB_THRESHOLD= 10.0
SWT_LEVEL   = 3
SWT_WAVELET = 'db4'
WINDOW_W    = 2
SPEC_LEN    = 896
D2_INDICES  = [37, 40, 60, 77]
N_FEATURES  = len(D2_INDICES)

# ==========================================
# 2. 工具函數
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
    d1  = np.gradient(cA3)
    d2  = np.gradient(d1)

    def pt(idx):
        return float(np.mean(d2[max(0, idx - WINDOW_W): idx + WINDOW_W + 1]))

    return np.array([pt(i) for i in D2_INDICES], dtype=np.float32)

# ==========================================
# 3. 資料載入
# ==========================================
def load_dataset(base_dir, mua_folder):
    raw_feats, labels, patient_ids = [], [], []
    excel_cache = {}
    
    if not os.path.exists(mua_folder): return np.array([]), np.array([]), []
    files = [f for f in os.listdir(mua_folder) if f.endswith('.txt')]

    for f_name in tqdm(files, desc='Extracting d2 features for Eval'):
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
                df['Bed_C'] = df['DialysisBed'].apply(_clean_bed)
                df['Shift_C'] = df['Shift'].apply(lambda x: "早" if "早" in str(x) else ("午" if "午" in str(x) else "晚"))
                excel_cache[date_str] = df
            else: excel_cache[date_str] = None

        df = excel_cache[date_str]
        if df is None: continue

        row = df[(df['Bed_C'] == bed) & (df['Shift_C'] == shift)]
        if row.empty or pd.isnull(row.iloc[0]['ClinicHb']): continue

        feats = _extract_d2(os.path.join(mua_folder, f_name))
        raw_feats.append(feats)
        labels.append(float(row.iloc[0]['ClinicHb']))
        patient_ids.append(patient_id)

    return np.array(raw_feats, dtype=np.float32), np.array(labels, dtype=np.float32), patient_ids

# ==========================================
# 4. 模型架構
# ==========================================
class HbD2Net(nn.Module):
    def __init__(self, n_in=N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32),   nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16),   nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ==========================================
# 5. 評估與繪圖
# ==========================================
def evaluate():
    raw_feats, labels, patient_ids = load_dataset(BASE_DIR, MUA_FOLDER)
    if len(labels) == 0:
        print("資料量不足。")
        return

    y_true_all = []
    y_pred_all = []
    colors_all = [] # 區分 low / high 模型點的顏色
    
    # 處理 LOW 模型 (< 10.0)
    if os.path.exists(LOW_MODEL_PATH):
        print(f"載入 {LOW_MODEL_PATH} ...")
        ckpt_low = torch.load(LOW_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model_low = HbD2Net().to(DEVICE)
        model_low.load_state_dict(ckpt_low['model'])
        model_low.eval()
        
        f_mean_low = ckpt_low['feat_mean']
        f_std_low = ckpt_low['feat_std']
        test_pids_low = set(ckpt_low['test_patient_ids'])
        
        for i, pid in enumerate(patient_ids):
            # 必須同時符合 PID 在獨立測試集內，且真實標籤低於閾值
            if pid in test_pids_low and labels[i] < HB_THRESHOLD:
                x_norm = (raw_feats[i] - f_mean_low) / f_std_low
                x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred = model_low(x_t).item()
                y_true_all.append(labels[i])
                y_pred_all.append(pred)
                colors_all.append('blue') 
    else:
        print(f"找不到 {LOW_MODEL_PATH}")

    # 處理 HIGH 模型 (>= 10.0)
    if os.path.exists(HIGH_MODEL_PATH):
        print(f"載入 {HIGH_MODEL_PATH} ...")
        ckpt_high = torch.load(HIGH_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model_high = HbD2Net().to(DEVICE)
        model_high.load_state_dict(ckpt_high['model'])
        model_high.eval()
        
        f_mean_high = ckpt_high['feat_mean']
        f_std_high = ckpt_high['feat_std']
        test_pids_high = set(ckpt_high['test_patient_ids'])
        
        for i, pid in enumerate(patient_ids):
            if pid in test_pids_high and labels[i] >= HB_THRESHOLD:
                x_norm = (raw_feats[i] - f_mean_high) / f_std_high
                x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred = model_high(x_t).item()
                y_true_all.append(labels[i])
                y_pred_all.append(pred)
                colors_all.append('red') 
    else:
        print(f"找不到 {HIGH_MODEL_PATH}")

    if len(y_true_all) == 0:
        print("Test Set 為空，無法評估。請確認 test_patient_ids 是否有對應樣本。")
        return

    # 計算整體指標
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    colors_all = np.array(colors_all)
    residuals = y_true_all - y_pred_all

    r2 = r2_score(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse)
    
    y_mean = np.mean(y_true_all)
    y_dummy = np.full_like(y_true_all, y_mean)
    base_mse = mean_squared_error(y_true_all, y_dummy)
    base_mae = mean_absolute_error(y_true_all, y_dummy)

    print("\n" + "="*50)
    print(" 📊 分軌模型 (d2 二階導數特徵) 聯合評估報告")
    print("="*50)
    print(f"聯合測試樣本數:     {len(y_true_all)}")
    print(f"群體真實平均血紅素: {y_mean:.4f} g/dL")
    print("-" * 50)
    print("【MAE 絕對誤差對決】")
    print(f"瞎猜基準線: {base_mae:.4f} g/dL")
    print(f"MLP 模型:   {mae:.4f} g/dL")
    diff_mae = base_mae - mae
    print(f"表現結論:   " + ("🚀 進步了" if diff_mae > 0 else "❌ 退步了") + f" {abs(diff_mae):.4f} g/dL")
    print("-" * 50)
    print("【MSE 均方誤差對決】")
    print(f"瞎猜基準線: {base_mse:.4f}")
    print(f"MLP 模型:   {mse:.4f}")
    diff_mse = base_mse - mse
    print(f"表現結論:   " + ("🚀 進步了" if diff_mse > 0 else "❌ 退步了") + f" {abs(diff_mse):.4f}")
    print("-" * 50)
    print(f"決定係數 (R²):      {r2:.4f}")
    print(f"均方根誤差 (RMSE):  {rmse:.4f}")
    print("="*50)

    # ==========================================
    # 6. 進階統計繪圖 (2x2 Matrix)
    # ==========================================
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 圖 1: 預測對散佈圖 (Scatter Plot)
    axs[0, 0].scatter(y_true_all[colors_all=='blue'], y_pred_all[colors_all=='blue'], alpha=0.6, color='dodgerblue', label='Low Model (<10)')
    axs[0, 0].scatter(y_true_all[colors_all=='red'], y_pred_all[colors_all=='red'], alpha=0.6, color='tomato', label='High Model (>=10)')
    axs[0, 0].plot([y_true_all.min(), y_true_all.max()], [y_true_all.min(), y_true_all.max()], 'k--', lw=2)
    axs[0, 0].set_xlabel('Actual ClinicHb')
    axs[0, 0].set_ylabel('Predicted ClinicHb')
    axs[0, 0].set_title(f'Test Set Regression (N={len(y_true_all)}, R²={r2:.3f})')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle=':', alpha=0.7)

    # 圖 2: 預測趨勢 (升冪排序)
    sorted_idx = np.argsort(y_true_all)
    y_true_s = y_true_all[sorted_idx]
    y_pred_s = y_pred_all[sorted_idx]
    colors_s = colors_all[sorted_idx]
    
    axs[0, 1].plot(y_true_s, label='Actual', marker='o', alpha=0.7, color='black')
    axs[0, 1].scatter(range(len(y_pred_s)), y_pred_s, c=np.where(colors_s=='blue', 'dodgerblue', 'tomato'), marker='x', alpha=0.9, zorder=3)
    axs[0, 1].plot([], [], 'x', color='dodgerblue', label='Pred Low')
    axs[0, 1].plot([], [], 'x', color='tomato', label='Pred High')
    axs[0, 1].axhline(y=y_mean, color='green', linestyle='--', alpha=0.5, label='Baseline (Mean)')
    axs[0, 1].legend()
    axs[0, 1].set_title('Prediction Trend (Sorted by Actual Hb)')
    axs[0, 1].set_xlabel('Samples (Low Hb to High Hb)')
    axs[0, 1].grid(True, linestyle=':', alpha=0.7)

    # 圖 3: Residual Plot (殘差圖)
    # 觀察殘差是否隨預測值變大而發散 (Homoscedasticity 檢查)
    axs[1, 0].scatter(y_pred_all, residuals, alpha=0.6, color='purple')
    axs[1, 0].axhline(0, color='red', linestyle='--', lw=2)
    axs[1, 0].set_xlabel('Predicted ClinicHb')
    axs[1, 0].set_ylabel('Residuals (Actual - Predicted)')
    axs[1, 0].set_title('Residual Plot (Homoscedasticity Check)')
    axs[1, 0].grid(True, linestyle=':', alpha=0.7)

    # 圖 4: Q-Q Plot (殘差常態檢驗)
    # 觀察點是否緊貼紅色對角線，若偏差嚴重代表模型存在系統性偏差
    stats.probplot(residuals, dist="norm", plot=axs[1, 1])
    axs[1, 1].get_lines()[0].set_markerfacecolor('mediumseagreen')
    axs[1, 1].get_lines()[0].set_markeredgecolor('mediumseagreen')
    axs[1, 1].get_lines()[0].set_alpha(0.6)
    axs[1, 1].get_lines()[1].set_color('red')
    axs[1, 1].get_lines()[1].set_linewidth(2)
    axs[1, 1].set_title('Q-Q Plot of Residuals (Normality Check)')
    axs[1, 1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('eval_plots_d2.png', dpi=150)
    print("\n>>> 評估完成！所有圖表已存入 eval_plots_d2.png")

if __name__ == "__main__":
    evaluate()