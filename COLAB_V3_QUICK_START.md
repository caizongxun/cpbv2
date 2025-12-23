# CPB Trading V3 模型訓練 - Colab 快速開始指南

最後更新: 2025-12-24

## 概述

這個指南會幫助你在 Google Colab 上一鍵訓練和部署 V3 模型，支援 20 種加密貨幣。

所有過程都是**自動化的一個 Cell**，包含：
- 數據下載 (Binance API)
- 特徵工程
- 模型訓練 (LSTM, 80 epochs)
- 模型評估
- 自動上傳到 HuggingFace
- 自動上傳到 GitHub cpbv2 倉庫

## 訓練時間

- **GPU (推薦)**: ~20-30 分鐘
- **CPU**: ~1-2 小時

## 步驟 1: 準備 Colab Notebook

1. 打開 [Google Colab](https://colab.research.google.com/)
2. 新建 Notebook
3. 新增一個 Cell

## 步驟 2: 設置 Secrets (可選但強烈推薦)

### 設置 HuggingFace Token

1. 從 [huggingface.co](https://huggingface.co/settings/tokens) 獲取你的 token
2. 在 Colab 左側點擊「Secrets」🔑
3. 新增 Secret:
   - Key: `HF_TOKEN`
   - Value: 你的 HuggingFace token

### 設置 GitHub Token

1. 從 [GitHub Settings](https://github.com/settings/tokens) 生成 Personal Access Token
   - 勾選: `repo`, `workflow` 權限
2. 在 Colab Secrets 新增:
   - Key: `GITHUB_TOKEN`
   - Value: 你的 GitHub token

## 步驟 3: 複製訓練 Cell

在 Colab Cell 中貼上下面的代碼，然後執行：

```python
import urllib.request
import subprocess

# 下載最新的訓練腳本
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

## 步驟 4: 執行!

按 `Shift + Enter` 執行 Cell，然後坐著喝杯咖啡等待...

### 執行流程如下:

```
[✓] 環境初始化完成
[✓] HuggingFace 認證完成
[✓] GitHub 認證準備完成

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
 batch_normalization (Batch  (None, 20, 64)           256
 dropout (Dropout)           (None, 20, 64)           0
 lstm_1 (LSTM)               (None, 32)               12416
 ...
=================================================================
Total params: 129,222
Trainable params: 128,966
Non-trainable params: 256
_________________________________________________________________

[*] 開始訓練 V3 模型 (epochs=80, batch_size=32)...
Epoch 1/80
 32/109 [=====>........................] - ETA: 3:22 - loss: 18.5432 - mae: 3.2145
 64/109 [=================>..........] - ETA: 1:45 - loss: 16.8214 - mae: 2.9876
 87/109 [=======================>...] - ETA: 0:26 - loss: 15.3421 - mae: 2.7654
Epoch 2/80
 ...

[✓] 訓練完成! 最佳 Val Loss: 2.134567

[✓] 模型評估結果:
  - Loss (MSE): 2.134567
  - MAE: 1.234567

[+] 準備完成: BTCUSDT
[+] 準備完成: ETHUSDT
[+] 準備完成: BNBUSDT
... (共 20 個幣種)
[✓] 20 個模型準備完成

[*] 正在上傳 20 個模型到 HuggingFace...
    目標: zongowo111/cpb-models/v3/
[+] 上傳成功: v3_model_BTCUSDT.h5
[+] 上傳成功: v3_model_ETHUSDT.h5
... (共 20 個)
[✓] HuggingFace 上傳完成!
    查看: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v3

[*] 正在上傳模型到 GitHub caizongxun/cpbv2...
[+] 複製完成: v3_model_BTCUSDT.h5
[+] 複製完成: v3_model_ETHUSDT.h5
... (共 20 個)
[✓] GitHub 上傳完成!
    查看: https://github.com/caizongxun/cpbv2/tree/main/models/v3

================================================================================
[✓] V3 模型訓練和部署完成!
================================================================================
```

## 驗證結果

### 檢查 HuggingFace 上傳

訪問: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v3

應該能看到 20 個模型文件：
- v3_model_BTCUSDT.h5
- v3_model_ETHUSDT.h5
- v3_model_BNBUSDT.h5
- ... 等等 (共 20 個)

### 檢查 GitHub 上傳

訪問: https://github.com/caizongxun/cpbv2/tree/main/models/v3

應該能看到：
- models/v3/v3_model_BTCUSDT.h5
- models/v3/v3_model_ETHUSDT.h5
- models/v3/README.md (訓練記錄)

## 常見問題

### Q: 為什麼需要 Token？
A: 為了自動上傳模型到 HuggingFace 和 GitHub，不需要手動操作。

### Q: 可以在 CPU 上訓練嗎？
A: 可以，但會很慢。建議使用 GPU (Colab 免費版有 GPU)。

### Q: 訓練中斷了怎麼辦？
A: 可以重新執行 Cell，腳本會自動恢復。

### Q: 如何修改訓練參數？
A: 編輯下載的 `v3_training.py` 文件，修改這些變數：
```python
TRAINING_COIN = 'BTCUSDT'  # 訓練幣種
DATA_LIMIT = 3500           # 下載數據量 (3000-5000)
EPOCHS = 80                 # 訓練輪次 (50-100)
```

### Q: 20 個幣種都用同一個模型嗎？
A: 是的。訓練一個基礎模型 (用 BTC)，然後複製給 20 個幣種。如果想要更高精準度，可以為每個幣種單獨訓練。

## V3 模型特性

### 輸入
- 20 根 1 小時 K 棒
- 特徵: OHLC (4 列)
- 形狀: (batch, 20, 4)

### 輸出
- 6 個預測值:
  1. **price_change** - 價格變化百分比 (%)
  2. **volatility** - 波動率預測 (%)
  3. **entry_range_low** - 開單下限點位
  4. **entry_range_high** - 開單上限點位
  5. **stop_loss** - 止損點位
  6. **take_profit** - 止盈點位

### 架構
```
Input (batch, 20, 4)
  ↓
LSTM(64) + BatchNorm + Dropout(0.2)
  ↓
LSTM(32) + BatchNorm + Dropout(0.2)
  ↓
Dense(64) + BatchNorm + Dropout(0.3)
  ↓
Dense(32) + BatchNorm
  ↓
Dense(6) - Linear Output
```

## 支援幣種

```
主流幣 (3): BTC, ETH, BNB
山寨幣 (5): ADA, SOL, XRP, DOGE, LINK
DeFi/Layer2 (5): AVAX, MATIC, ATOM, NEAR, FTM
L2 & 其他 (7): ARB, OP, LIT, STX, INJ, LUNC, LUNA
```

## 下一步

1. 訓練完成後，模型會自動上傳到 HuggingFace 和 GitHub
2. 後端可以從 GitHub 或 HuggingFace 加載模型
3. 前端可以調用後端 API 獲取預測結果
4. 顯示開單點位範圍和止損止盈

## 技術支持

如遇問題，請檢查：
1. HF_TOKEN 和 GITHUB_TOKEN 是否正確設置
2. Colab GPU 是否啟用 (運行時類型 > GPU)
3. 網絡連接是否正常
4. Binance API 是否可訪問

祝訓練順利！
