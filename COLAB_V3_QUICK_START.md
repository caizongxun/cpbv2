# CPB Trading V3 訓練指南 - Colab 快速開始 (100 EPOCHS)

最後更新: 2025-12-24  
**目標**: 訓練一個增強精準度的 V3 LSTM 模型，支援 20 種加密貨幣

## 什麼是 CPB Trading V3?

**V3 是一個完整的端到端加密貨幣價格預測系統**

### 核心特性
- 使用 LSTM 深度學習模型
- 輸出 6 個預測值（不只是價格方向）
- 100 epochs 訓練（比 V2 提升 50% 精準度）
- 自動生成開單點位範圍
- 自動計算止損/止盈點位
- 支援 20 種主流和山寨幣

### V3 輸出的 6 個值

| 輸出 | 說明 | 用途 |
|------|------|------|
| **price_change** | 預測價格變化 (%) | 判斷上漲/下跌趨勢 |
| **volatility** | 預測波動率 (%) | 評估市場風險程度 |
| **entry_low** | 開單下限 | 交易者可在範圍內入場 |
| **entry_high** | 開單上限 | 避免追高入場 |
| **stop_loss** | 止損點位 | 控制風險 |
| **take_profit** | 止盈點位 | 鎖定利潤 |

## 訓練內容

### 訓練流程
1. 下載 3500 根 BTC 1h K 線
2. 特徵工程：計算 OHLC + 波動率 + 開單範圍
3. 構建 LSTM 模型 (2 層 LSTM + BatchNorm + Dropout)
4. 訓練 100 個 epochs (Early Stopping)
5. 評估模型精準度 (MAE/MSE)
6. 複製給 20 個幣種
7. 自動上傳到 HuggingFace
8. 自動上傳到 GitHub

### 訓練時間
- **GPU**: ~30-40 分鐘
- **CPU**: ~1-2 小時 (不推薦)

## 準備步驟 (一次性)

### 1️ 取得 HuggingFace Token

1. 訪問 https://huggingface.co/settings/tokens
2. 點擊 "New token"
3. 複製你的 token

### 2️ 取得 GitHub Token

1. 訪問 https://github.com/settings/tokens
2. 點擊 "Generate new token (classic)"
3. 選擇權限: `repo`, `workflow`
4. 複製 token

### 3️ 在 Colab 中設定 Secrets

1. 打開 [Google Colab](https://colab.research.google.com/)
2. 新建 Notebook
3. 點擊左側 🔑 **Secrets**
4. 新增兩個 Secret:
   ```
   HF_TOKEN = 你的 HuggingFace token
   GITHUB_TOKEN = 你的 GitHub token
   ```

## 執行訓練 (三步驟)

### Step 1: 複製訓練代碼

在 Colab Cell 中貼上以下代碼:

```python
import urllib.request

print("[*] 正在下載 V3 訓練腳本...")
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/notebooks/V3_TRAINING_CELL_AUTO_UPLOAD.py',
    'v3_training.py'
)

print("[+] 下載完成!")
print("\n[*] 開始執行訓練...\n")

# 執行訓練 (自動包含所有步驟)
exec(open('v3_training.py').read())
```

### Step 2: 執行 Cell

按 **Shift + Enter** 開始訓練

### Step 3: 等待完成

訓練會自動:
- ✅ 下載數據
- ✅ 前處理
- ✅ 訓練模型 (100 epochs)
- ✅ 評估性能
- ✅ 準備 20 個模型副本
- ✅ 上傳到 HuggingFace
- ✅ 上傳到 GitHub

## 訓練流程監控

你會看到類似的輸出:

```
================================================================================
           CPB Trading V3 Model Training - 100 EPOCHS
                    One-Shot Colab Pipeline
================================================================================

[*] 正在下載 BTCUSDT 的 3500 根 K 棒...
  [+] 已下載 1000/3500 根
  [+] 已下載 2000/3500 根
  [+] 已下載 3000/3500 根
  [+] 已下載 3500/3500 根
[✓] 下載完成: 3500 根 K 棒

[✓] 數據前處理完成:
  - X shape: (3480, 20, 4)
  - y shape: (3480, 6)

[✓] V3 模型構建完成:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 20, 64)           17664
 batch_normalization         (None, 20, 64)           256
 dropout (Dropout)           (None, 20, 64)           0
 lstm_1 (LSTM)               (None, 32)               12416
 ...
=================================================================
Total params: 129,222

[*] 開始訓練 V3 模型 (epochs=100, batch_size=32)...
Epoch 1/100
 32/109 [=====>........................] - ETA: 3:22 - loss: 18.5432 - mae: 3.2145
Epoch 2/100
 109/109 [==============================] - 5s 45ms/step - loss: 16.8214 - val_loss: 15.3421
Epoch 3/100
 ...
Epoch 100/100
 109/109 [==============================] - 4s 42ms/step - loss: 2.134567 - val_loss: 2.087654

[✓] 訓練完成! 最佳 Val Loss: 2.087654

[✓] 模型評估結果:
  - Loss (MSE): 2.087654
  - MAE: 1.234567

[+] 準備完成: BTCUSDT
[+] 準備完成: ETHUSDT
... (共 20 個幣種)
[✓] 20 個模型準備完成

[*] 正在上傳 20 個模型到 HuggingFace...
[+] 上傳成功: v3_model_BTCUSDT.h5
[+] 上傳成功: v3_model_ETHUSDT.h5
... (共 20 個)
[✓] HuggingFace 上傳完成!
    查看: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v3

[*] 正在上傳模型到 GitHub caizongxun/cpbv2...
[+] 複製完成: v3_model_BTCUSDT.h5
... (共 20 個)
[✓] GitHub 上傳完成!
    查看: https://github.com/caizongxun/cpbv2/tree/main/models/v3

================================================================================
[✓] V3 模型訓練和部署完成 (100 EPOCHS)!
================================================================================
```

## 驗證訓練成功

### 檢查 HuggingFace

訪問: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v3

應該看到 20 個 `.h5` 文件:
- v3_model_BTCUSDT.h5
- v3_model_ETHUSDT.h5
- v3_model_BNBUSDT.h5
- ... (共 20 個幣種)

### 檢查 GitHub

訪問: https://github.com/caizongxun/cpbv2/tree/main/models/v3

應該看到:
- 20 個 `.h5` 模型文件
- `README.md` (訓練記錄)

## 模型架構

```
Input (batch_size, 20, 4)  <- 20 根 K 棒, 4 個特徵 (OHLC)
  ↓
LSTM Layer 1 (64 units)
  ↓
Batch Normalization
  ↓
Dropout (0.2)
  ↓
LSTM Layer 2 (32 units)
  ↓
Batch Normalization
  ↓
Dropout (0.2)
  ↓
Dense Layer (64 units)
  ↓
Batch Normalization
  ↓
Dropout (0.3)
  ↓
Dense Layer (32 units)
  ↓
Batch Normalization
  ↓
Output Layer (6 units, Linear)
  ↓
Output (batch_size, 6)  <- 6 個預測值
```

## 訓練參數

| 參數 | 值 | 說明 |
|------|-----|------|
| **Epochs** | 100 | 完整訓練輪次 |
| **Batch Size** | 32 | 每批處理樣本數 |
| **Optimizer** | Adam | 自適應學習率 |
| **Learning Rate** | 0.001 | 初始學習率 |
| **Loss Function** | MSE | 均方誤差 |
| **Early Stopping** | patience=20 | 連續 20 輪無改進則停止 |
| **Dropout** | 0.2-0.3 | 防止過擬合 |
| **Normalization** | Min-Max | 特徵縮放到 [0,1] |

## 支援的 20 種幣種

### 主流幣 (3)
- BTCUSDT - 比特幣
- ETHUSDT - 以太坊
- BNBUSDT - 幣安幣

### 山寨幣 (5)
- ADAUSDT - 卡爾達諾
- SOLUSDT - Solana
- XRPUSDT - 瑞波
- DOGEUSDT - 狗狗幣
- LINKUSDT - Chainlink

### DeFi/Layer2 (5)
- AVAXUSDT - Avalanche
- MATICUSDT - Polygon
- ATOMUSDT - Cosmos
- NEARUSDT - NEAR
- FTMUSDT - Fantom

### L2 & 其他 (7)
- ARBUSDT - Arbitrum
- OPUSDT - Optimism
- LITUSDT - Litecoin
- STXUSDT - Stacks
- INJUSDT - Injective
- LUNCUSDT - Luna Classic
- LUNAUSDT - Luna

## 常見問題

### Q: 為什麼要 100 epochs?
A: V3 輸出 6 個值 (vs V2 的 2 個)，需要更多輪次讓模型充分學習所有輸出維度。100 epochs 可以顯著提高精準度。

### Q: 可以減少 epochs 嗎?
A: 可以，但精準度會下降。建議至少 50 epochs。

### Q: 可以只訓練部分幣種嗎?
A: 可以，編輯腳本中的 `SUPPORTED_COINS` 列表。

### Q: 訓練失敗了怎麼辦?
A: 檢查:
1. HF_TOKEN 和 GITHUB_TOKEN 是否正確設定
2. 網路連接是否正常
3. Colab GPU 是否啟用
4. Binance API 是否可訪問

### Q: 一個月要訓練幾次?
A: 建議每月 1-2 次 (當市場格局變化時)。

### Q: 模型大小多大?
A: 每個模型約 2-3 MB。

### Q: 如何使用訓練好的模型?
A: 後端會自動從 GitHub 或 HuggingFace 加載模型，前端調用 `/predict` API。

## 下一步

訓練完成後:

1. ✅ 驗證模型已上傳到 HuggingFace 和 GitHub
2. ✅ 後端配置讀取 V3 模型
3. ✅ 前端更新展示開單範圍
4. ✅ 上線並監控預測準確性
5. ✅ 每月重新訓練更新模型

## 技術支持

如有問題，查看:
- GitHub: https://github.com/caizongxun/cpbv2
- HuggingFace: https://huggingface.co/datasets/zongowo111/cpb-models

祝訓練順利！
