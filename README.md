# 血液透析患者血紅素預測（光譜分析）

利用近紅外線（NIR）吸收光譜資料，結合機器學習，預測血液透析患者的血紅素（HB）濃度。

---

## 專案背景

- **資料來源**：血液透析過程中收集的近紅外線光譜檔（`.txt`），與 Excel 臨床記錄中的 ClinicHb 值進行配對
- **資料規模**：約 82 位病人 / 292 筆配對樣本，跨 4 個收案日期
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

---

## 目前進度（2026-05-06）

| 項目 | 狀態 |
|------|------|
| 資料配對（光譜 ↔ Excel）| ✅ 完成 |
| 特徵探索（導數、FFT）| ✅ 完成 |
| 回歸模型（SWT + 3-channel，舊特徵）| ✅ 已完成（基準比較用） |
| 回歸模型（d2 特徵，MLP）| ✅ 訓練 + 評估腳本已完成，待在 server 執行 |
| 回歸模型（d2 特徵，sklearn 多模型對照）| ✅ 腳本已完成，待在 server 執行 |
| 二元分類模型（d2 特徵）| ✅ 訓練 + 評估腳本已完成，待在 server 執行 |

**特徵決策**：探索結果確認 SWT cA3 的二階導數（d2）在 537–540nm 處與 HB 相關性最強（r ≈ 0.33），所有新腳本均改用 d2@537/540/560/577nm 作為輸入特徵，採 per-feature z-score 標準化。

---

## 檔案說明

### 資料統計

| 檔案 | 功能 |
|------|------|
| `check_hb_split.py` | 以光譜檔為基準配對 Excel，統計 HB < 10 / HB ≥ 10 各組的樣本數、病人數，以及預計的 4:1 train/test 分配 |
| `check_hb_split_0505.py` | 同上，更新版 |

---

### 回歸模型訓練與評估（舊特徵 — 基準）

| 檔案 | 功能 |
|------|------|
| `train_swt_hb_split.py` | 回歸訓練（HB 兩組）。特徵：SWT 第 3 階 cA3 在 540/560/577nm，聯合標準化。Patient-level 4:1 + 5-Fold CV |
| `train_swt_hb_split_0505.py` | 更新版，新增 `SWT_LEVEL` 參數、可選人口統計特徵、自動偵測路徑 |
| `eval_swt_hb_split.py` | 載入兩組模型評估，輸出 4 格圖 |
| `eval_swt_hb_split_0505.py` | 更新版 eval |

**模型輸出檔**：`swt_hb_low_model.pth`、`swt_hb_high_model.pth`

---

### 回歸模型訓練與評估（d2 特徵 — 新版）

| 檔案 | 功能 |
|------|------|
| `train_d2_hb_split_0505.py` | 回歸訓練（HB 兩組）。特徵：SWT cA3 二階導數在 537/540/560/577nm（4 維），per-feature z-score。Patient-level 4:1 + 5-Fold CV。輸出 learning curve 圖 |
| `eval_d2_hb_split_0505.py` | 載入 `d2_hb_low_model.pth` 與 `d2_hb_high_model.pth`，輸出 4 格圖（散佈圖、排序預測圖、殘差圖、Q-Q 圖）。低/高組用不同顏色區分 |

**模型輸出檔**：`d2_hb_low_model.pth`、`d2_hb_high_model.pth`

---

### sklearn 多模型對照

| 檔案 | 功能 |
|------|------|
| `train_sklearn_0505.py` | 同一份 d2 特徵，同時比較 7 種回歸模型（LinearRegression、Ridge、Lasso、SVR、RandomForest 等）與 4 種分類模型（LogisticRegression、SVC、RandomForest 等）。輸出 CSV 對照表 |

不需要額外的 eval 腳本，結果直接在訓練時輸出。

---

### 二元分類模型（d2 特徵）

| 檔案 | 功能 |
|------|------|
| `train_binary_hb_0505.py` | 訓練二元分類 MLP，預測 HB < 10（label=0）或 HB ≥ 10（label=1）。特徵：d2@537/540/560/577nm，per-feature z-score。BCEWithLogitsLoss 加正負樣本自動權重。Patient-level 4:1 + 5-Fold CV |
| `eval_binary_hb_0505.py` | 載入 `d2_hb_binary_model.pth`，輸出 4 格圖（ROC 曲線、Confusion Matrix、PR 曲線、Score Distribution）並列印所有指標 |

**模型輸出檔**：`d2_hb_binary_model.pth`

---

### 特徵探索腳本（訓練前先跑，了解資料特性）

| 檔案 | 功能 |
|------|------|
| `explore_derivatives_0505.py` | 計算 SWT cA3 在 500–800nm 全段的 1 階與 2 階導數，對每個波長位置計算與 ClinicHb 的 Pearson 相關係數。輸出：全段相關性總覽圖、d1 與 d2 最相關波長的散佈圖 |
| `explore_fft_ch542_0505.py` | 對 542nm channel 進行兩種 FFT 分析：(A) 對同一檔案內重複量測的時間序列做 FFT；(B) 對整條平均光譜做 FFT，提取頻域特徵後計算與 HB 的相關性 |
| `explore_swt2_deriv_ch659_0505.py` | 對光譜套用 SWT 第 2 階（db4），計算 cA2 的 1 階導數，在 659nm 處取斜率值，並掃描 600–720nm 全段找出 d1 與 HB 相關性最高的波長 |

---

## 特徵探索結果彙整

| 特徵 | 最佳波長 | Pearson r | 備註 |
|------|---------|-----------|------|
| 原始光譜 / SWT cA3 | 540–580nm | ~+0.24 | 舊版訓練使用 |
| 1 階導數（d1） | 570nm | ~−0.28 | 略優於原始值 |
| **2 階導數（d2）** | **537nm** | **~+0.33** | **目前使用，最強訊號** |
| 659nm 任何特徵 | 659nm | ~0.04 | 無效，已排除 |
| FFT 光譜特徵 | coef #26 | ~+0.29 | 與原始值相當，無額外增益 |

---

## 執行順序

```bash
# 第一步：確認資料配對正確
python3 update_0505/check_hb_split_0505.py

# 第二步：特徵探索（三個腳本可同時跑）
python3 update_0505/explore_derivatives_0505.py
python3 update_0505/explore_fft_ch542_0505.py
python3 update_0505/explore_swt2_deriv_ch659_0505.py

# 第三步：訓練（三個腳本可同時跑）
python3 update_0505/train_d2_hb_split_0505.py   # MLP 回歸（d2 特徵）
python3 update_0505/train_sklearn_0505.py        # sklearn 多模型對照
python3 update_0505/train_binary_hb_0505.py      # 二元分類（d2 特徵）

# 第四步：評估
python3 update_0505/eval_d2_hb_split_0505.py    # → eval_plots_d2.png
python3 update_0505/eval_binary_hb_0505.py      # → eval_binary_hb.png
```

---

## 注意事項

- **不可公開**：`.txt` 光譜原始檔、`.xlsx` Excel 臨床資料、`.pth` 模型檔（內含 test 病人 session ID）
- **可公開**：本資料夾內所有 `.py` 檔案
- 程式碼中的 `BASE_DIR` 會自動偵測為腳本所在資料夾的上一層，無需手動修改路徑
