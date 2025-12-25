# Colab 訓練版本比較

## 快速選擇

| 需求 | 選擇 |
|------|------|
| **Colab 免費版，首次嘗試** | **V4 優化版** ✓ |
| **Colab Pro，追求最高準確度** | **V5 完整版** ✓ |
| **時間有限，快速完成** | **V4 優化版** ✓ |
| **需要 40 個模型** | **V5 完整版** ✓ |

---

## V4 優化版 (推薦用於免費 Colab)

### 特點

✓ **完全適配 Colab 免費版限制**
- 2.5 小時完成訓練
- Colab 免費版限制：3 小時 20 分鐘
- 安全裕度：50 分鐘

✓ **簡化但有效的特徵**
- 使用 OHLCV (Open, High, Low, Close, Volume)
- 無複雜的技術指標計算
- 更快的資料預處理

✓ **優化的模型架構**
- 1 層 LSTM (vs V5 的 2 層)
- Hidden size 128 (vs V5 的 256)
- 總參數: ~90k (vs V5 的 ~300k)

✓ **激進的早期停止**
- 每個模型僅 10 個 epoch
- Early stopping patience: 5
- 平均每個模型 6-8 分鐘

### 規格

```
模型數量: 20 (10 coins × 2 timeframes)

幣種 (Top 10):
  BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, LTCUSDT,
  ADAUSDT, SOLUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT

時間框架:
  15分鐘, 1小時

每個模型:
  - Epochs: 10 (減少 50%)
  - Batch size: 32
  - 訓練時間: ~6-8 分鐘
  - GPU 記憶體: 0.1-0.2GB

總耗時: ~2.5 小時
```

### 性能預期

```
成功率: 95-100%
Average MAPE: ~0.05
Best MAPE: ~0.01-0.02
Worst MAPE: ~0.08-0.10
```

### 使用方式

#### 方式 1: 自動選擇 (推薦)

```python
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/COLAB_QUICK_START.py'
).text)
```

自動檢測 GPU 並選擇合適版本

#### 方式 2: 直接執行

```python
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_train_colab_optimized.py'
).text)
```

#### 方式 3: 分步驟執行

**Cell 1: 安裝依賴**
```python
!pip install -q torch pandas numpy scikit-learn requests
```

**Cell 2: 執行訓練**
```python
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_train_colab_optimized.py'
).text)
```

---

## V5 完整版 (用於 Colab Pro)

### 特點

✓ **完整的特徵工程**
- 40+ 技術指標 (RSI, MACD, Bollinger Bands 等)
- 多時間框架分析
- 更好的模型準確性

✓ **更大的模型**
- 2 層 LSTM
- Hidden size 256
- 總參數: ~300k
- 更強的表達能力

✓ **更多的訓練數據**
- 20 種幣種
- 2 個時間框架
- 總共 40 個模型
- 更全面的市場覆蓋

✓ **改進的訓練策略**
- 100 個 epoch
- Early stopping patience: 15
- Learning rate scheduling
- Gradient clipping

### 規格

```
模型數量: 40 (20 coins × 2 timeframes)

幣種 (Full List):
  Top tier: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, LTCUSDT
  Large cap: ADAUSDT, SOLUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT
  Mid cap: UNIUSDT, ATOMUSDT, NEARUSDT, ARBUSDT, OPUSDT
  Alt coins: PEPEUSDT, INJUSDT, SHIBUSDT, ETCUSDT, LUNAUSDT

時間框架:
  15分鐘, 1小時

每個模型:
  - Epochs: 100
  - Batch size: 64
  - 訓練時間: ~3-5 分鐘
  - GPU 記憶體: 0.2-0.3GB

總耗時: ~2-2.5 小時
```

### 性能預期

```
成功率: 95-100%
Average MAPE: ~0.015-0.02
Best MAPE: ~0.005-0.01
Worst MAPE: ~0.04-0.06
```

比 V4 準確度提升 25-30%

### 使用方式

#### 方式 1: 自動選擇 (推薦)

```python
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/COLAB_QUICK_START.py'
).text)
```

檢測到 A100 時自動選擇 V5

#### 方式 2: 直接執行

```python
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete_fixed.py'
).text)
```

#### 方式 3: 使用 Loader

```python
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_colab_loader.py'
).text)
```

---

## 詳細對比

### 訓練時間

| 版本 | Colab 免費版 | Colab Pro T4 | Colab Pro A100 |
|------|------------|------------|---------------|
| **V4 優化版** | 2.5h ✓ | 2h | 1h |
| **V5 完整版** | 超時 ✗ | 2-2.5h ✓ | 1-1.5h ✓ |

### 特徵數量

| 版本 | 原始特徵 | 工程特徵 | 總計 |
|------|--------|--------|------|
| **V4 優化版** | 5 (OHLCV) | 0 | 5 |
| **V5 完整版** | 5 (OHLCV) | 35+ | 40+ |

### 模型規模

| 項目 | V4 優化版 | V5 完整版 |
|------|---------|----------|
| LSTM 層數 | 1 | 2 |
| Hidden size | 128 | 256 |
| 總參數數 | ~90k | ~300k |
| GPU 記憶體/模型 | 0.1GB | 0.2GB |

### 準確度

| 指標 | V4 優化版 | V5 完整版 |
|------|---------|----------|
| Avg MAPE | ~0.05 | ~0.015-0.02 |
| Best MAPE | ~0.01-0.02 | ~0.005-0.01 |
| Worst MAPE | ~0.08-0.10 | ~0.04-0.06 |
| 改進幅度 | 基準 | +25-30% |

### 幣種覆蓋

| 項目 | V4 優化版 | V5 完整版 |
|------|---------|----------|
| 總幣種 | 10 | 20 |
| 模型數 | 20 | 40 |
| 市值覆蓋 | Top 50 | Top 100+ |

---

## 選擇指南

### 選 V4 優化版如果:

✓ 使用 Colab 免費版
✓ 時間有限 (2.5h 內)
✓ 首次嘗試/實驗用途
✓ 只關心主要幣種 (Top 10)
✓ 想要可靠的快速結果
✓ GPU 是 T4 或 L4

### 選 V5 完整版如果:

✓ 有 Colab Pro 訂閱
✓ 需要 40 個模型
✓ 追求最高準確度
✓ 覆蓋更多幣種
✓ 有充足的時間
✓ GPU 是 A100 或 H100

---

## 運行檢查清單

### 前置需求

- [ ] 已選擇 GPU Runtime (不能使用 CPU)
- [ ] 網路連接穩定
- [ ] 瀏覽器標籤保持開啟
- [ ] 至少 3.5 小時可用時間

### V4 (Colab 免費版)

- [ ] Runtime type: GPU (T4 或 L4)
- [ ] 預期時間: 2.5 小時
- [ ] 安全裕度: 50 分鐘
- [ ] 幣種: 10 個
- [ ] 模型: 20 個

### V5 (Colab Pro)

- [ ] Runtime type: GPU (建議 A100)
- [ ] 預期時間: 2-2.5 小時
- [ ] 幣種: 20 個
- [ ] 模型: 40 個

---

## 常見問題

### Q1: 運行中斷了，怎麼辦?

**對於 V4**:
- 中斷點通常不會遺失（每個模型自動保存）
- 重新運行，會跳過已訓練的模型
- 無需擔心

**對於 V5**:
- 如果超過 3h 20m，免費版會中斷
- 必須使用 Colab Pro
- 或者改用 V4 優化版

### Q2: 如何選擇正確的版本?

```
if GPU == 'T4' or GPU == 'L4':
    use_v4()  # 肯定沒問題
elif GPU == 'A100':
    use_v5()  # 快速完成
else:
    use_v4()  # 保險選擇
```

### Q3: 能否同時運行多個?

不建議。建議分別在不同時間運行。

### Q4: 訓練速度太慢了

**V4 優化版**:
- 每個模型 6-8 分鐘是正常的
- 總耗時 2.5 小時不可避免

**V5 完整版**:
- 如果超過 3 小時，改用 Colab Pro
- 或改用 V4 優化版

### Q5: 如何確保不會超時?

**最安全的方式**:
1. 使用 V4 優化版 (2.5h)
2. 確認有 3.5 小時可用
3. 使用 Colab Pro (無限時間)

---

## 訓練結果保存

### V4 優化版

```
/content/v4_models_optimized/
  ├── BTCUSDT_15m.pt
  ├── BTCUSDT_1h.pt
  ├── ETHUSDT_15m.pt
  ├── ...
  └── results.json
```

### V5 完整版

```
/content/all_models/model_v5/
  ├── BTCUSDT_15m.pt
  ├── BTCUSDT_1h.pt
  ├── ...
  └── training_results.json
```

### 下載結果

```python
# 如果要下載模型
from google.colab import files
files.download('/content/v4_models_optimized/results.json')
```

---

## 上傳到 Hugging Face

### 安裝

```bash
!pip install -q huggingface-hub
!huggingface-cli login
```

### 上傳 V4

```bash
!huggingface-cli upload your-username/cpb-models \
  /content/v4_models_optimized model_v4 --repo-type model
```

### 上傳 V5

```bash
!huggingface-cli upload your-username/cpb-models \
  /content/all_models/model_v5 model_v5 --repo-type model
```

---

## 版本歷史

| 版本 | 日期 | 說明 |
|------|------|------|
| V4 優化版 | 2025-12-25 | 首次發佈，針對 Colab 免費版 |
| V5 完整版 | 2025-12-25 | 原始完整版本，修復 RSI bug |
| COLAB_QUICK_START | 2025-12-25 | 自動選擇版本 |

---

**推薦使用**: COLAB_QUICK_START.py (自動為您選擇最佳版本)

**文件最後更新**: 2025-12-25
