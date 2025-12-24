# CPB V4 - 20 個支持幣種

## 的二十個幣種

### 大幡票 (Top Tier)

| 串号 | 幣種 | 中文名 | 市值排名 |
|------|------|----------|----------|
| 1 | BTCUSDT | 比特幣 | #1 |
| 2 | ETHUSDT | 以太坊 | #2 |
| 3 | BNBUSDT | 幣安幣 | #4 |
| 4 | XRPUSDT | XRP | #5 |
| 5 | LTCUSDT | 鶊幣 | #12 |

### 市值排名全球 TOP 50 幣種

| 串号 | 幣種 | 中文名 | 特性 |
|------|------|----------|----------|
| 6 | ADAUSDT | Cardano | 根保醨式粗体 |
| 7 | SOLUSDT | Solana | 高速网络 |
| 8 | DOGEUSDT | Dogecoin | 社区仪记 |
| 9 | AVAXUSDT | Avalanche | 网络互重 |
| 10 | LINKUSDT | Chainlink | 预言机首 |
| 11 | MATICUSDT | Polygon | L2择子 |
| 12 | ATOMUSDT | Cosmos | 跨链网络 |
| 13 | NEARUSDT | NEAR Protocol | 分享型合约 |
| 14 | FTMUSDT | Fantom | 根保醨式 |
| 15 | ARBUSDT | Arbitrum | L2优化 |
| 16 | OPUSDT | Optimism | L2优化 |
| 17 | STXUSDT | Stacks | 比特幣二层 |
| 18 | INJUSDT | Injective | 皮业事业 |
| 19 | LUNCUSDT | Luna Classic | 超大皽粗体 |
| 20 | LUNAUSDT | Luna | 正式 Luna v2 |

---

## 使用方法

### 方法 1：第一次使用（執行其中一個）

在 Colab 中修改：
```python
TRAINING_COIN = "BTCUSDT"  # 阻吵其中一个秫残
```

支持的所有值：
```python
TRAINING_COIN = "BTCUSDT"
TRAINING_COIN = "ETHUSDT"
TRAINING_COIN = "BNBUSDT"
TRAINING_COIN = "XRPUSDT"
TRAINING_COIN = "LTCUSDT"
TRAINING_COIN = "ADAUSDT"
TRAINING_COIN = "SOLUSDT"
TRAINING_COIN = "DOGEUSDT"
TRAINING_COIN = "AVAXUSDT"
TRAINING_COIN = "LINKUSDT"
TRAINING_COIN = "MATICUSDT"
TRAINING_COIN = "ATOMUSDT"
TRAINING_COIN = "NEARUSDT"
TRAINING_COIN = "FTMUSDT"
TRAINING_COIN = "ARBUSDT"
TRAINING_COIN = "OPUSDT"
TRAINING_COIN = "STXUSDT"
TRAINING_COIN = "INJUSDT"
TRAINING_COIN = "LUNCUSDT"
TRAINING_COIN = "LUNAUSDT"
```

### 方法 2：大批量訓練（訓練所有 20 个幣种）

剩改 Colab Cell：

```python
import urllib.request
import os

print("[*] Starting batch training for all 20 coins...\n")

coins_to_train = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
    'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
    'MATICUSDT', 'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'ARBUSDT',
    'OPUSDT', 'STXUSDT', 'INJUSDT', 'LUNCUSDT', 'LUNAUSDT'
]

HF_TOKEN = "hf_YOUR_TOKEN_HERE"
EPOCHS = 80

for idx, coin in enumerate(coins_to_train, 1):
    print(f"\n{'='*80}")
    print(f"[{idx:2d}/20] Training {coin}...")
    print(f"{'='*80}\n")
    
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_training_remote.py',
        'v4_train.py'
    )
    
    script_content = open('v4_train.py').read()
    script_content = script_content.replace('HF_TOKEN = "hf_YOUR_TOKEN_HERE"', f'HF_TOKEN = "{HF_TOKEN}"')
    script_content = script_content.replace('TRAINING_COIN = "BTCUSDT"', f'TRAINING_COIN = "{coin}"')
    script_content = script_content.replace('EPOCHS = 80', f'EPOCHS = {EPOCHS}')
    
    exec(script_content)

print("\n" + "="*80)
print("[✓] All 20 coins trained successfully!")
print("="*80)
```

---

## 訓練預計時間

### 头一個幣種
- **GPU (T4)**: ~20-30 分鐘
- **GPU (V100)**: ~10-15 分鐘
- **GPU (A100)**: ~5-10 分鐘

### 批量訓練 20 個幣種
- **GPU (T4)**: ~6-10 小時
- **GPU (V100)**: ~3-5 小時
- **GPU (A100)**: ~2-3 小時

---

## 批量訓練策略

### 方案 A: 天分操作
一威仏会正两个幣种。

**優点**: 不会造成 Colab 超时
**缺点**: 效率较低（需要 10 天以上）

### 方案 B: 一次訓練所有
一上乘下 20 个幣种的脚本。

**優点**: 效率高（冶此了就完成）
**缺点**: 并发在每个幣种上（深度对模型性能有较小漂移）

---

## 下載所有 20 个模型

所有 20 个訓練好的模型稍完成上傳后，可後跟赤日例線：

https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v4

一个幣種一个 `v4_model_{COIN}.pt` 文件。

---

## 快速參考

拷贴这些幣种名稱到 Colab 參数中：

```
BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, LTCUSDT, ADAUSDT, SOLUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT, MATICUSDT, ATOMUSDT, NEARUSDT, FTMUSDT, ARBUSDT, OPUSDT, STXUSDT, INJUSDT, LUNCUSDT, LUNAUSDT
```

---

**最后更新**: 2025-12-24 CST
