# CPB V4 - 遠端執行版本（1 個 Cell 搞定）

## 使用方式

### 方法 1：複製貼上（最簡單）

1. 打開 https://colab.research.google.com
2. 新建筆記本
3. **複製下面整個 Cell**
4. 執行

---

## Colab 執行 Cell（直接複製貼上）

```python
import urllib.request
import os

# ============================================================================
# 配置參數 - 只需修改這裡
# ============================================================================
HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # 換成你的 token (https://huggingface.co/settings/tokens)
TRAINING_COIN = "BTCUSDT"        # 選擇幣種
EPOCHS = 80                      # 訓練輪次 (50-100)

# ============================================================================
# 下載並執行遠端腳本（不用改）
# ============================================================================
print("[*] Downloading V4 training script...")
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_training_remote.py',
    'v4_train.py'
)
print("[+] Download complete!\n")

# 將配置注入到腳本
script_content = open('v4_train.py').read()
script_content = script_content.replace('HF_TOKEN = "hf_YOUR_TOKEN_HERE"', f'HF_TOKEN = "{HF_TOKEN}"')
script_content = script_content.replace('TRAINING_COIN = "BTCUSDT"', f'TRAINING_COIN = "{TRAINING_COIN}"')
script_content = script_content.replace('EPOCHS = 80', f'EPOCHS = {EPOCHS}')

exec(script_content)
```

---

## 修改參數說明

### 只需修改「配置參數」區塊的這 3 行：

```python
# 第 1 行：填入你的 HuggingFace Token
HF_TOKEN = "hf_YOUR_TOKEN_HERE"

# 第 2 行：選擇訓練幣種（20 選 1）
TRAINING_COIN = "BTCUSDT"

# 第 3 行：調整訓練輪次（可選）
EPOCHS = 80
```

### 支持的 20 個幣種

複製下面任一個到 `TRAINING_COIN`：

```
BTCUSDT      (比特幣)
ETHUSDT      (以太坊)
BNBUSDT      (幣安幣)
XRPUSDT      (XRP)
LTCUSDT      (萊特幣)
ADAUSDT      (Cardano)
SOLUSDT      (Solana)
DOGEUSDT     (Dogecoin)
AVAXUSDT     (Avalanche)
LINKUSDT     (Chainlink)
MATICUSDT    (Polygon)
ATOMUSDT     (Cosmos)
NEARUSDT     (NEAR)
FTMUSDT      (Fantom)
ARBUSDT      (Arbitrum)
OPUSDT       (Optimism)
STXUSDT      (Stacks)
INJUSDT      (Injective)
LUNCUSDT     (Luna Classic)
LUNAUSDT     (Luna v2)
```

---

## 例子

### 例子 1：訓練比特幣

```python
HF_TOKEN = "hf_abc123xyz..."    # 你的 token
TRAINING_COIN = "BTCUSDT"       # 訓練 BTC
EPOCHS = 80
```

### 例子 2：訓練以太坊

```python
HF_TOKEN = "hf_abc123xyz..."    # 你的 token
TRAINING_COIN = "ETHUSDT"       # 訓練 ETH
EPOCHS = 100                    # 更多 epochs
```

### 例子 3：訓練 Solana

```python
HF_TOKEN = "hf_abc123xyz..."    # 你的 token
TRAINING_COIN = "SOLUSDT"       # 訓練 SOL
EPOCHS = 50                     # 較少 epochs
```

---

## 執行流程（自動）

1. ✓ 下載腳本
2. ✓ 安裝依賴 (PyTorch + 其他)
3. ✓ 下載 K 棒數據
4. ✓ 計算技術指標
5. ✓ 訓練 CNN-LSTM 模型
6. ✓ 生成交易信號
7. ✓ 上傳到 HuggingFace

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
    Available coins: 20

... (完整訓練過程)

[ENTRY SIGNAL]
Coin: BTCUSDT
Direction: LONG UP (Confidence 78.45%)
Current Price: 95,000.00
Entry Range: 94,520.00 ~ 95,480.00
Stop Loss: 93,700.00
Take Profit: 97,500.00
Risk/Reward: 1:1.33
Position Size: 1.6x

[✓] V4 Training Complete!
Performance: Accuracy=0.8267, F1=0.8012, AUC=0.8567
```

---

## 常見問題

**Q: 變數應該填哪個？**
A: 只填上面「配置參數」區塊的 3 個變數，下面不用改

**Q: 如果不上傳到 HuggingFace？**
A: 留下
```python
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
```
即可，會自動跳過上傳步驟

**Q: 訓練時間多長？**
A:
- GPU (T4): 20-30 分鐘
- GPU (V100): 10-15 分鐘
- GPU (A100): 5-10 分鐘

**Q: 可以連續訓練多個幣種嗎？**
A: 可以，但需要手動修改 Cell 每次重新執行。

詳見：https://github.com/caizongxun/cpbv2/blob/main/20_COINS_LIST.md（批量訓練指南）

**Q: 模型保存在哪？**
A: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v4

---

## 關鍵點

✓ **一個 Cell** - 無需分割  
✓ **只改 3 行** - 配置參數在上面  
✓ **自動下載** - 最新腳本每次執行  
✓ **自動安裝** - 無需手動配置環境  
✓ **完全自動** - 按下 Enter 就完成  

---

**最後更新**: 2025-12-24 CST  
**版本**: 4.0.0  
**狀態**: 生產就緒
