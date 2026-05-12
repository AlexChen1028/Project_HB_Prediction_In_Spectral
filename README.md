# 血液透析患者血紅素預測（光譜分析）

利用近紅外線（NIR）吸收光譜資料，結合機器學習，預測血液透析患者的血紅素（HB）濃度。

---

## 專案背景

- **資料來源**：血液透析過程中收集的近紅外線光譜檔（`.txt`），與 Excel 臨床記錄中的 ClinicHb 值進行配對
- **資料規模**：112 位病人 / 342 筆配對樣本，跨 6 個收案日期（成大醫院 4 天 + 風典診所 1 天 + 成大醫院 0417 1 天）
- **預測目標**：
  - 回歸：預測 ClinicHb 數值（g/dL）
  - 分類：判斷 HB < 10 或 HB ≥ 10

---

## 開發紀錄

| 日期 | 內容 |
|------|------|
| 2026-04-25 | 建立基礎 SWT 回歸模型，使用 5 個差值特徵（v540-v500 等），加入 patient-level KFold |
| 2026-04-28 | 依 HB 分成兩組（< 10 / ≥ 10）各自訓練模型；改為 4:1 train/test 切分 |
| 2026-05-05 | 改為 3-channel 聯合標準化；移除 early stopping；新增二元分類、FFT 與導數探索腳本；程式碼推上 GitHub |
| 2026-05-06 | 特徵全面改為 SWT cA3 二階導數（d2）；新增 d2 回歸訓練/評估、sklearn 模型對照、二元分類評估腳本 |
| 2026-05-11 | 新增 EDA 腳本：PCA 可分性、時間段 SNR 分析、SavGol 降噪評估、批次效應檢定；加入風典診所（20260319）與成大醫院 0417 資料，總樣本數 292→342，低 HB 樣本 60→79 |
| 2026-05-12 | 新增 CNN 模型：以 2D 光譜圖（波長×時間）作為圖片輸入，分別訓練回歸與二元分類；更新 regex 相容純數字床號；所有輸出統一存至 update_0511/ |

---

## 目前進度（2026-05-12）

| 項目 | 狀態 |
|------|------|
| 資料配對（光譜 ↔ Excel）| ✅ 完成（6 個日期） |
| 特徵探索（導數、FFT、批次效應）| ✅ 完成 |
| 回歸模型（SWT + 3-channel，舊特徵）| ✅ 基準比較用 |
| 回歸模型（d2 特徵，MLP）| ✅ 完成｜Test R²=0.589 MAE=0.449 |
| 回歸模型（d2 特徵，sklearn 多模型）| ✅ 完成｜LinearReg Test R²=0.20 |
| 二元分類模型（d2 特徵，MLP）| ✅ 完成｜AUC=0.699 |
| EDA / 批次效應分析 | ✅ 完成｜SNR=6–10，批次效應存在但訊號真實 |
| **CNN 回歸（2D 光譜圖）** | ✅ 腳本完成，待在 server 執行 |
| **CNN 二元分類（2D 光譜圖）** | ✅ 腳本完成，待在 server 執行 |

**最佳模型**：d2 MLP 回歸（HB 分組）— Test R²=0.589，MAE=0.449 g/dL

---

## 資料說明

| 日期 | 機構 | N | HB 平均 | Low HB (< 10) 比例 |
|------|------|---|---------|------------------|
| 20260115 | 成大醫院 | 47 | 10.54 | 38.3% |
| 20260122 | 成大醫院 | 99 | 10.73 | 9.1% |
| 20260128 | 成大醫院 | 76 | 10.52 | 21.1% |
| 20260205 | 成大醫院 | 70 | 10.64 | 24.3% |
| 20260319 | 風典診所 | 32 | 9.90 | 53.1% |
| 20260417 | 成大醫院 | 18 | 11.01 | 11.1% |

**批次效應**：Kruskal-Wallis 顯著（p < 0.05），但去除日期均值後相關係數維持相近，訊號為真實 HB 訊號而非批次偽像。

---

## 檔案說明

### update_0505／ — d2 特徵 MLP（基準版）

| 檔案 | 功能 |
|------|------|
| `check_hb_split_0505.py` | 資料配對與 HB 組別統計 |
| `explore_derivatives_0505.py` | SWT cA3 各波長 d1/d2 相關性掃描 |
| `explore_fft_ch542_0505.py` | FFT 特徵探索 |
| `explore_swt2_deriv_ch659_0505.py` | 659nm 導數特徵探索 |
| `train_d2_hb_split_0505.py` | MLP 回歸，HB 分兩組（d2 特徵）|
| `eval_d2_hb_split_0505.py` | 回歸評估，輸出 4 格圖 |
| `train_sklearn_0505.py` | sklearn 7 種回歸 + 4 種分類模型比較 |
| `train_binary_hb_0505.py` | MLP 二元分類（d2 特徵）|
| `eval_binary_hb_0505.py` | 分類評估，輸出 ROC/CM/PR/分布圖 |

---

### update_0511／ — 新資料 + EDA + CNN

| 檔案 | 功能 |
|------|------|
| `explore_eda_0511.py` | EDA：PCA 可分性、3-segment SNR 分析、SavGol 降噪評估 |
| `explore_batch_0511.py` | 批次效應：PCA by date、Kruskal-Wallis、Partial correlation |
| `train_d2_hb_split_0511.py` | MLP 回歸（加入新資料，相容純數字床號）|
| `eval_d2_hb_split_0511.py` | 回歸評估 |
| `train_binary_hb_0511.py` | MLP 二元分類（加入新資料）|
| `eval_binary_hb_0511.py` | 分類評估 |
| `train_sklearn_0511.py` | sklearn 多模型比較（新資料）|
| `train_cnn_regression_0511.py` | **CNN 回歸**：2D 光譜圖（300×150）→ Conv→Conv→GAP→FC |
| `eval_cnn_regression_0511.py` | CNN 回歸評估，輸出 4 格圖（散佈、排序預測、殘差、Q-Q） |
| `train_cnn_binary_0511.py` | **CNN 二元分類**：同架構，BCEWithLogitsLoss + pos_weight |
| `eval_cnn_binary_0511.py` | CNN 分類評估，輸出 4 格圖（ROC、Confusion Matrix、PR、分布圖） |

**模型輸出**（存至 `update_0511/`）：
- `d2_hb_low_model_0511.pth`、`d2_hb_high_model_0511.pth`
- `d2_hb_binary_model_0511.pth`
- `cnn_regression_model_0511.pth`
- `cnn_binary_model_0511.pth`

---

## 特徵探索結果彙整

| 特徵 | 最佳波長 | Pearson r | 備註 |
|------|---------|-----------|------|
| 原始光譜 / SWT cA3 | 540–580nm | ~+0.24 | 舊版訓練使用 |
| 1 階導數（d1） | 570nm | ~−0.28 | 略優於原始值 |
| **2 階導數（d2）** | **537nm** | **~+0.33** | **MLP 使用，最強單點訊號** |
| 659nm 任何特徵 | 659nm | ~0.04 | 無效，已排除 |
| FFT 光譜特徵 | coef #26 | ~+0.29 | 與原始值相當，無額外增益 |
| **2D 光譜圖（全時間軸）** | — | — | **CNN 輸入，保留時間維度資訊** |

---

## EDA 關鍵結論

- **SNR**：4 個 d2 特徵的 SNR = 6–10（訊號比雜訊大 6–10 倍），資料品質可訓練
- **SavGol 降噪**：無效，SWT db4 level-3 已足夠平滑，不需要額外濾波
- **批次效應**：各日期間 d2 特徵分布顯著不同（p < 0.05），但屬加噪型而非偽訊號型
- **PCA**：兩 HB 組大量重疊，無法線性分離，分類天花板有限

---

## 執行順序

```bash
# 第一步：EDA（可選，已執行）
python3 update_0511/explore_eda_0511.py
python3 update_0511/explore_batch_0511.py

# 第二步：MLP 訓練（新資料）
python3 update_0511/train_d2_hb_split_0511.py
python3 update_0511/train_binary_hb_0511.py
python3 update_0511/train_sklearn_0511.py

# 第三步：MLP 評估
python3 update_0511/eval_d2_hb_split_0511.py
python3 update_0511/eval_binary_hb_0511.py

# 第四步：CNN 訓練（新方法）
python3 update_0511/train_cnn_regression_0511.py
python3 update_0511/train_cnn_binary_0511.py

# 第五步：CNN 評估
python3 update_0511/eval_cnn_regression_0511.py   # → eval_cnn_regression_0511.png
python3 update_0511/eval_cnn_binary_0511.py       # → eval_cnn_binary_0511.png
```

---

## 注意事項

- **不可公開**：`.txt` 光譜原始檔、`.xlsx` Excel 臨床資料、`.pth` 模型檔（內含 test 病人 session ID）
- **可公開**：所有 `.py` 腳本
- `BASE_DIR` 自動偵測為腳本所在資料夾的上一層，無需手動修改路徑
- 風典診所資料（20260319）的床號為純數字格式（`201`、`202`…），update_0511 腳本已相容；update_0505 腳本不支援此格式
