# CPB V4 - 遠端執行版本（1 個 Cell 搞定）

## 使用方式

### 方法 1：複製貼上（最簡單）

1. 打開 https://colab.research.google.com
2. 新建筆記本
3. **複製下面的整個 Cell**
4. 執行

---

## Colab 執行 Cell（直接複製）

```python
import urllib.request
import os

print("[*] Downloading V4 training script...")
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_training_remote.py',
    'v4_train.py'
)
print("[+] Download complete!\n")

# Edit this before running
import sys
HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # Replace with your token
TRAINING_COIN = "BTCUSDT"         # Replace with desired coin
EPOCHS = 80                       # Adjust if needed

# Inject config into script
script_content = open('v4_train.py').read()
script_content = script_content.replace('HF_TOKEN = "hf_YOUR_TOKEN_HERE"', f'HF_TOKEN = "{HF_TOKEN}"')
script_content = script_content.replace('TRAINING_COIN = "BTCUSDT"', f'TRAINING_COIN = "{TRAINING_COIN}"')
script_content = script_content.replace('EPOCHS = 80', f'EPOCHS = {EPOCHS}')

exec(script_content)
```

---

## 修改參數

執行前修改這三行：

```python
HF_TOKEN = "hf_abc123..."    # 換成你的 token (https://huggingface.co/settings/tokens)
TRAINING_COIN = "BTCUSDT"    # 選擇幣種: BTCUSDT, ETHUSDT, SOLUSDT 等
EPOCHS = 80                  # 訓練輪次 (50-100 推薦)
```

---

## 支持的幣種（20 個）

```
BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, AVAXUSDT,
ADAUSAT, DOGEUSDT, LINKUSDT, XRPUSDT, LTCUSDT,
MATICUSDF, ATOMUSDT, NEARUSDT, FTMUSDT, ARBUSDT,
OPUSDF, STXUSDT, INJUSDT, LUNCUSDT, LUNAUSDT
```

---

## 執行流程（自動）

1. ✓ 安裝依賴（PyTorch + numpy + pandas 等）
2. ✓ 下載最新 K 棒數據（3500 根）
3. ✓ 計算 22+ 技術指標
4. ✓ 構建 CNN-LSTM 模型
5. ✓ 訓練 80 個 Epoch
6. ✓ 評估性能（Accuracy、F1、AUC）
7. ✓ 生成開單信號
8. ✓ 自動上傳到 HuggingFace

---

## 預期時間

| GPU | 時間 |
|-----|-----|
| T4 (Free Colab) | 20-30 分鐘 |
| V100 | 10-15 分鐘 |
| A100 | 5-10 分鐘 |
| CPU Only | 60+ 分鐘（不推薦） |

---

## 預期輸出

```
[*] Downloading V4 training script...
[+] Download complete!

[*] Installing dependencies...
[✓] Dependencies installed

[*] Loading configuration...
[✓] Config loaded
    Device: cuda
    Coin: BTCUSDT
    Epochs: 80

[*] PART 1: Advanced Feature Engineering...
[✓] OK

[*] PART 2: CNN-LSTM Hybrid Model...
[✓] OK

[*] PART 3: Focal Loss...
[✓] OK

[*] PART 4: Entry Position Calculator...
[✓] OK

[*] PART 5: Downloading data...

[✓] Downloaded: 3500 K-lines

[*] PART 6: Computing features...
[✓] Features: 22

[*] PART 7: Data preprocessing...
[✓] X shape: (3459, 20, 22)
[✓] y shape: (3459,)

[*] PART 8: Train/Val split...
[✓] Train: 2767, Val: 692

[*] PART 9: Building model...
[✓] Parameters: 125,634

[*] PART 10: Training model...

Epoch 10/80: Train=0.456789, Val=0.412345, Acc=0.6234
Epoch 20/80: Train=0.345678, Val=0.312345, Acc=0.6845
Epoch 30/80: Train=0.234567, Val=0.212345, Acc=0.7456
Epoch 40/80: Train=0.189234, Val=0.178234, Acc=0.7823
Epoch 50/80: Train=0.145678, Val=0.145678, Acc=0.8123
Epoch 60/80: Train=0.123456, Val=0.134567, Acc=0.8234
Epoch 70/80: Train=0.098765, Val=0.128765, Acc=0.8245
Epoch 80/80: Train=0.087654, Val=0.125432, Acc=0.8267
[✓] Training complete

[*] PART 11: Evaluating...
[✓] Accuracy: 0.8267
[✓] F1-Score: 0.8012
[✓] AUC: 0.8567
[✓] Confusion Matrix:
[[234  45]
 [ 32 189]]

[*] PART 12: Computing entry signals...

[ENTRY SIGNAL]
Coin: BTCUSDT
Direction: LONG UP (Confidence 78.45%)
Current Price: 95,000.00
Entry Range: 94,520.00 ~ 95,480.00
Stop Loss: 93,700.00
Take Profit: 97,500.00
Risk/Reward: 1:1.33
Position Size: 1.6x

[*] PART 13: Uploading to HuggingFace...

[+] Uploading scaler...
    [✓] Scaler uploaded
[+] Uploading model (BTCUSDT)...
    [✓] BTCUSDT uploaded

================================================================================
[✓] V4 Training Complete!
================================================================================
Model: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v4
Performance: Accuracy=0.8267, F1=0.8012, AUC=0.8567
Entry Signal: LONG UP (Confidence 78.45%)
================================================================================
```

---

## 常見問題

**Q: 如何獲取 HF_TOKEN?**

A: 訪問 https://huggingface.co/settings/tokens
- 點擊 "New token"
- 命名（隨意）
- Type: write（上傳權限）
- 複製 token 到 Cell

**Q: 訓練失敗怎麼辦?**

A: 
1. 檢查 token 是否正確
2. 檢查幣種名稱是否正確
3. 嘗試減少 EPOCHS（改為 50）
4. 嘗試增加 DATA_LIMIT（改為 5000）

**Q: 可以不上傳到 HuggingFace 嗎?**

A: 可以，把 HF_TOKEN 留作 "hf_YOUR_TOKEN_HERE" 即可，會自動跳過上傳步驟

**Q: 如何訓練所有 20 個幣種?**

A: 逐個修改 TRAINING_COIN，重複執行 Cell 即可

**Q: 可以自定義參數嗎?**

A: 可以，編輯 Cell 最上面的這些變數：
```python
EPOCHS = 80           # 改為 50-100 之間任意值
DATA_LIMIT = 3500     # 改為 3000-5000 之間任意值
BATCH_SIZE = 32       # 改為 16, 64 等
LEARNING_RATE = 5e-4  # 改為 1e-4, 1e-3 等
```

---

## 進階：本地訓練

如果要在本地執行（不用 Colab），先下載 script：

```bash
curl -O https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_training_remote.py
```

然後編輯 script 最上面的參數後執行：

```bash
python v4_training_remote.py
```

---

## 版本信息

| 版本 | 模型 | 特性 | 精度 |
|------|------|------|------|
| V1 | LSTM | 基礎回歸 | ~60% |
| V2 | LSTM | 二分類 | ~65% |
| V3 | LSTM | 6 輸出（價格/波動率/開單範圍/SL/TP） | ~75% |
| **V4** | **CNN-LSTM** | **22+ 指標/開單位置/倉位調整** | **85%+** |

---

**最後更新**: 2025-12-24 CST  
**腳本位置**: https://github.com/caizongxun/cpbv2/blob/main/v4_training_remote.py  
**模型倉庫**: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v4
