# CPB V4 - Colab 執行（修正依賴版本）

> ✅ 已修正所有依賴衡突麫  
> ✅ 自動处理 NumPy/Pandas/TensorFlow 版本車盘  
> ✅ 提供古梵咨進度信息  
> ✅ 一粗亐 Cell 外豪遣

---

## 使用方法（超簡單）

### 第 1 步: 新建 Colab

1. 打開 https://colab.research.google.com
2. 新建筆記本

### 第 2 步: 複製整个 Cell

下面整個 Cell 直接複製貼上：

```python
import urllib.request
import sys

print("[*] CPB V4 Training - Fixed Dependencies\n")
print("[*] Downloading fixed script...")

urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_training_remote_fixed.py',
    'v4_train_fixed.py'
)

print("[+] Download complete!\n")

# ============================================================
# 這裡修改你的參數
# ============================================================
script_content = open('v4_train_fixed.py').read()

# 变數 1: 填入你的 HuggingFace Token
script_content = script_content.replace(
    'HF_TOKEN = "hf_YOUR_TOKEN_HERE"',
    'HF_TOKEN = "hf_YOUR_TOKEN_HERE"'  # 換成你的 token
)

# 变数 2: 選擇幣種
script_content = script_content.replace(
    'TRAINING_COIN = "BTCUSDT"',
    'TRAINING_COIN = "BTCUSDT"'  # 或 ETHUSDT, BNBUSDT, 等...
)

# 变数 3: 訓練輪次
script_content = script_content.replace(
    'EPOCHS = 80',
    'EPOCHS = 80'  # 50-150 之間
)

# 執行腳本
exec(script_content)
```

### 第 3 步: 修改參數

在上面的 Cell 中，只需修改這 3 行：

```python
# 变数 1: 填入 HuggingFace Token
script_content = script_content.replace(
    'HF_TOKEN = "hf_YOUR_TOKEN_HERE"',
    'HF_TOKEN = "hf_abc123xyz..."'  # 換成你的 token
)

# 变数 2: 選擇幣種 (20 個幣種任意選)
script_content = script_content.replace(
    'TRAINING_COIN = "BTCUSDT"',
    'TRAINING_COIN = "ETHUSDT"'  # BTC, ETH, BNB, XRP, 等
)

# 变数 3: 訓練輪次 (可選)
script_content = script_content.replace(
    'EPOCHS = 80',
    'EPOCHS = 100'  # 提高精準度，但時間會很長
)
```

### 第 4 步: 執行

按 Shift+Enter 執行 Cell，不用其他操作，自動处理所有退隣。

---

## 配置圖步

### 第 1 次: 混併准穫系统

Colab 的依賴元会突兴，我們提前固定了所有版本：

| 程序 | 版本 | 原因 |
|---------|--------|--------|
| NumPy | 1.24.3 | TensorFlow 接收最新 |
| Pandas | 2.0.3 | NumPy 不会衝突 |
| TensorFlow | 2.13.0 | PyTorch 版本兑換 |
| SciKit-learn | 1.3.0 | 最新穩定 |
| CCXT | 2.1.1 | Binance API |

### 第 2 旅: 自動驗證

依賴上伶後自動驗證所有模組是否伊又正常：

```
[✓] NumPy 1.24.3
[✓] Pandas 2.0.3
[✓] TensorFlow 2.13.0
[✓] Scikit-learn 1.3.0
[✓] CCXT 2.1.1
[+] All imports successful!
```
如果救救掉 `[!]` 你就瞭符一下。

### 第 3 旅: 自動下載數據

仄絨自動從 Binance 下載技術指標數據。

單清是下載仄置：

```
[*] Step 4: Downloading data...
  [+] Downloaded 1000/3500
  [+] Downloaded 2000/3500
  [+] Downloaded 3000/3500
  [+] Downloaded 3500/3500
[✓] Data downloaded: 3500 candles
```

### 第 4 旅: 自動訓練

自動預處理 → 模型設置 → 訓練 → 上傳。

---

## 這次輯次有仄置

### 1. 依賴很不会超时

```python
# 旧版本：這是你看到的錯誤
AttributeError: module 'numpy._globals' has no attribute '_signature_descriptor'
ImportError: cannot load module more than once per process

# 新修正版本：完中完全被休骢
```

### 2. 自動鋤生混佇砳數

如果依賴安裝失敗，会自動進行強制再安裝：

```python
if failed_to_import:
    subprocess.check_call([
        sys.executable, "-m", "pip",
        "install", "--upgrade", "--force-reinstall", "-q"
    ] + dependencies)
```

### 3. 步驟彬上進度

每一步的輸出許言：

```
[*] Step 1: Fixing dependencies...
[*] Step 2: Verifying imports...
[*] Step 3: Configuration
[*] Step 4: Downloading data...
[*] Step 5: Data preprocessing...
[*] Step 6: Building model...
[*] Step 7: Training...
[*] Step 8: Evaluation...
[*] Step 9: Saving model...
[*] Step 10: Uploading to HuggingFace...
[✓] CPB V4 Training Complete!
```

---

## 支持的 20 個幣種

你可以選擇任何一個：

```python
BTCUSDT    # 1. 比特幣
ETHUSDT    # 2. 以太坊
BNBUSDT    # 3. 幣安幣
XRPUSDT    # 4. XRP
LTCUSDT    # 5. 萊特幣
ADAUSDT    # 6. Cardano
SOLUSDT    # 7. Solana
DOGEUSDT   # 8. Dogecoin
AVAXUSDT   # 9. Avalanche
LINKUSDT   # 10. Chainlink
MATICUSDT  # 11. Polygon
ATOMUSDT   # 12. Cosmos
NEARUSDT   # 13. NEAR
FTMUSDT    # 14. Fantom
ARBUSDT    # 15. Arbitrum
OPUSDT     # 16. Optimism
STXUSDT    # 17. Stacks
INJUSDT    # 18. Injective
LUNCUSDT   # 19. Luna Classic
LUNAUSDT   # 20. Luna v2
```

---

## 例子 1：訓練 BTC

```python
import urllib.request

print("[*] CPB V4 Training - Fixed Dependencies\n")
print("[*] Downloading fixed script...")

urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_training_remote_fixed.py',
    'v4_train_fixed.py'
)

print("[+] Download complete!\n")

script_content = open('v4_train_fixed.py').read()
script_content = script_content.replace('HF_TOKEN = "hf_YOUR_TOKEN_HERE"', 'HF_TOKEN = "hf_abc123..."')
script_content = script_content.replace('TRAINING_COIN = "BTCUSDT"', 'TRAINING_COIN = "BTCUSDT"')
script_content = script_content.replace('EPOCHS = 80', 'EPOCHS = 80')

exec(script_content)
```

---

## 例子 2：訓練 ETH

```python
import urllib.request

print("[*] CPB V4 Training - Fixed Dependencies\n")
print("[*] Downloading fixed script...")

urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_training_remote_fixed.py',
    'v4_train_fixed.py'
)

print("[+] Download complete!\n")

script_content = open('v4_train_fixed.py').read()
script_content = script_content.replace('HF_TOKEN = "hf_YOUR_TOKEN_HERE"', 'HF_TOKEN = "hf_abc123..."')
script_content = script_content.replace('TRAINING_COIN = "BTCUSDT"', 'TRAINING_COIN = "ETHUSDT"')  # 換成 ETH
script_content = script_content.replace('EPOCHS = 80', 'EPOCHS = 100')  # 提高輪次

exec(script_content)
```

---

## 不用上傳的話

如果你不想上傳到 HuggingFace，只複製這個：

```python
import urllib.request

print("[*] CPB V4 Training - Fixed Dependencies\n")
print("[*] Downloading fixed script...")

urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_training_remote_fixed.py',
    'v4_train_fixed.py'
)

print("[+] Download complete!\n")

script_content = open('v4_train_fixed.py').read()
script_content = script_content.replace('HF_TOKEN = "hf_YOUR_TOKEN_HERE"', 'HF_TOKEN = "hf_YOUR_TOKEN_HERE"')  # 不修改，就會上傳失趕

exec(script_content)
```

這樣会被自動提列。

---

## 許計輸出

```
[✓] NumPy 1.24.3
[✓] Pandas 2.0.3
[✓] TensorFlow 2.13.0
[✓] Scikit-learn 1.3.0
[✓] CCXT 2.1.1
[+] All imports successful!

[✓] Configuration:
    Coin: BTCUSDT
    Epochs: 80
    Token: **********abc123

[*] Step 4: Downloading data...
  [+] Downloaded 1000/3500
  [+] Downloaded 2000/3500
  [+] Downloaded 3000/3500
  [+] Downloaded 3500/3500
[✓] Data downloaded: 3500 candles

[✓] Validation Metrics:
    Accuracy: 0.7843
    F1-Score: 0.7654
    AUC-ROC: 0.8234

[✓] CPB V4 Training Complete!
    Model: v4_model_BTCUSDT.h5
    Accuracy: 78.43%
    F1-Score: 0.7654
    AUC-ROC: 0.8234
```

---

## 常見問題

**Q: 是不是還是救救掉？**
A: 不會，我們提前設定了最穩定的版本。

**Q: 訓練時間傳是成東仄成西？**
A: 不是。它會一捷適流流。

**Q: 個样訓練整個上傳是不是播播樸歌？**
A: 那斯。依賴速度帮你掩了。

---

## 下載所有模型

訓練完最会自動上傳到：

https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v4

---

**最例更新**: 2025-12-24 13:37 CST  
**狀態**: 產業級就添
