# 訓練文件完整總結

## 所有訓練文件清單

### 1. **v4_train_transformer.py** 最新推薦
- **架構**: Transformer + Encoder-Decoder
- **GPU使用**: 最高（Multi-Head Attention 全 GPU 計算）
- **速度**: 最快（比 LSTM 快 10-20 倍）
- **特點**:
  - Positional Encoding
  - Multi-Head Attention
  - Batch size 128（大 batch 發揮 Transformer 優勢）
  - 預計每模型 3-5 分鐘
  - GPU 利用率 80-90%

**何時用**: 想要最快速度 + 最高 GPU 利用率

---

### 2. **v4_train_fast_gru.py** 次佳
- **架構**: GRU + Simple Attention
- **GPU使用**: 中等（GRU 比 LSTM 快，但還是有循環）
- **速度**: 比 LSTM 快 30-40%
- **特點**:
  - GRU（少一個門控，比 LSTM 快）
  - Simple Attention（輕量級）
  - Batch size 64
  - Teacher Forcing
  - 預計每模型 5-8 分鐘
  - CPU 用量會較高 (40-50%)

**何時用**: 想要快速但不要 GPU 複雜度太高

---

### 3. **v4_train_unified.py** 通用版本
- **架構**: Seq2Seq LSTM + Attention
- **GPU使用**: 中等
- **速度**: 標準（每模型 10-15 分鐘）
- **特點**:
  - Multi-head Attention
  - Encoder + Decoder
  - Teacher Forcing
  - Hidden size 256（比較深）
  - Batch size 32
  - 詳細的 GPU 監控輸出

**何時用**: 想要平衡性能和準確度

---

## 其他版本（供參考）

| 文件 | 說明 | 用途 |
|------|------|------|
| `v4_colab_training.py` | 原始版本，LSTM+注意力 | 實驗參考 |
| `v4_colab_training_clean.py` | 清理版本，移除冗餘 | 實驗參考 |
| `v4_gpu_optimized.py` | 嘗試優化 GPU 使用 | 實驗參考 |
| `v4_model_cuda_forced.py` | 強制 CUDA（可能有問題） | 實驗參考 |
| `v4_model_cuda_forced_v2.py` | CUDA 優化 v2 | 實驗參考 |
| `v4_train_cuda.py` | 純 CUDA 版本（不穩定） | 實驗參考 |
| `v4_train_cuda_fixed.py` | CUDA 修復版本 | 實驗參考 |
| `v4_train_cuda_v2.py` | CUDA v2 | 實驗參考 |
| `v3_colab_training.py` | V3 版本 LSTM | 舊版參考 |
| `v3_colab_training_yfinance.py` | V3 + YFinance | 舊版參考 |

---

## 推薦使用順序

### 方案 A: 速度最優（推薦）
```
1. 試 v4_train_transformer.py (3-5 分鐘/模型)
   → 預期: GPU 80-90%, CPU < 30%
   → 40 個模型: 2-3 小時
```

### 方案 B: 穩定優先
```
1. 先試 v4_train_fast_gru.py (5-8 分鐘/模型)
   → 預期: GPU 50-70%, CPU 40-50%
   → 40 個模型: 3-5 小時
```

### 方案 C: 平衡
```
1. 用 v4_train_unified.py (10-15 分鐘/模型)
   → 預期: GPU 60-75%, CPU 30-40%
   → 40 個模型: 6-10 小時
```

---

## 每個版本的核心代碼差異

### Transformer 版本（最新）
```python
# 核心：Multi-Head Attention 做矩陣運算（全 GPU）
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=8,  # 8 個注意力頭
    dim_feedforward=512,
    batch_first=True,
    activation='gelu'
)
# 優勢：並行計算，GPU 最愛
```

### GRU 版本（次選）
```python
# 核心：GRU 替代 LSTM
self.encoder_gru = nn.GRU(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=2,
    batch_first=True,
    dropout=0.3
)
# 優勢：參數少，訓練快
```

### LSTM 版本（通用）
```python
# 核心：傳統 LSTM + Attention
self.lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True
)
# 優勢：穩定，準確度通常最好
```

---

## GPU 使用情況對比

| 版本 | CPU 用量 | GPU 用量 | GPU Memory | 速度 |
|------|----------|----------|-----------|------|
| Transformer | < 30% | 80-90% | 3-5GB | 最快 |
| GRU | 40-50% | 50-70% | 2-3GB | 中等 |
| LSTM | 30-40% | 60-75% | 3-4GB | 標準 |

---

## 數據下載

所有版本都相同的數據下載邏輯：

```python
# 20 種幣別，2 個時間框 = 40 個模型
COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
    'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
    'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'DYDXUSDT', 'ARBUSDT',
    'OPUSDT', 'PEPEUSDT', 'INJUSDT', 'SHIBUSDT', 'LUNAUSDT'
]

# 每個幣別下載
# - 15 分鐘 (15m) K 線
# - 1 小時 (1h) K 線

# 數據來源：Binance Vision (免費)
# 下載時間：通常 5-15 分鐘
```

---

## 訓練配置對比

### Transformer (v4_train_transformer.py)
```python
model = TransformerModel(
    input_size=4,      # Open, High, Low, Close
    d_model=128,       # 模型維度
    nhead=8,           # 8 個注意力頭
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=512,
    steps_ahead=10     # 預測 10 步（40 分鐘/1 小時）
)

# 訓練參數
optimizer = optim.AdamW(model.parameters(), lr=0.001)
epochs = 50
batch_size = 128
```

### GRU (v4_train_fast_gru.py)
```python
model = FastSeq2Seq(
    input_size=4,
    hidden_size=128,
    output_size=4,
    steps_ahead=10
)

# 訓練參數
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
batch_size = 64
```

### LSTM (v4_train_unified.py)
```python
model = Seq2SeqLSTM(
    input_size=4,
    hidden_size=256,
    num_layers=2,
    dropout=0.3,
    steps_ahead=10,
    output_size=4
)

# 訓練參數
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
epochs = 200
batch_size = 32
```

---

## 如何選擇

1. **想要最快速度**
   → 用 `v4_train_transformer.py`
   → GPU 利用率最高
   → 完成時間: 2-3 小時

2. **GPU 內存有限 (< 8GB)**
   → 用 `v4_train_fast_gru.py`
   → GPU Memory 用量最少
   → 完成時間: 3-5 小時

3. **追求準確度**
   → 用 `v4_train_unified.py`
   → LSTM 通常最穩定
   → 完成時間: 6-10 小時

4. **不確定選哪個**
   → 先試 Transformer（目前最好的選擇）

---

## Colab 執行方法

### 執行 Transformer（推薦）
```python
import os
os.system('curl -sS -o /content/v4_train_transformer.py https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_train_transformer.py')

print("Ready - Transformer (Real GPU Usage)\n")
exec(open('/content/v4_train_transformer.py').read())
```

### 執行 GRU
```python
import os
os.system('curl -sS -o /content/v4_train_fast_gru.py https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_train_fast_gru.py')

print("Ready - Fast GRU\n")
exec(open('/content/v4_train_fast_gru.py').read())
```

### 執行 LSTM
```python
import os
os.system('curl -sS -o /content/v4_train_unified.py https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_train_unified.py')

print("Ready - Unified (LSTM)\n")
exec(open('/content/v4_train_unified.py').read())
```

---

## 常見問題

### Q: 為什麼 CPU 用量那麼高？
**A**: 
- 數據預處理在 CPU 上
- LSTM/GRU 有循環，需要 CPU 同步
- 解決: 用 Transformer（並行度高）

### Q: GPU Memory 不足？
**A**:
- 用 GRU 版本（參數少）
- 或降低 batch_size
- 或用 Kaggle (P100 GPU)

### Q: 訓練太慢？
**A**:
- 優先用 Transformer
- 或用 Kaggle (更快的 GPU)
- 或降低 epochs 數量

### Q: 模型準確度?
**A**:
- Transformer ≈ LSTM > GRU
- 差異通常 < 5%
- 推薦 Transformer (快 + 準)

---

## 下一步

1. 選擇一個版本執行
2. 監控 GPU 使用情況
3. 訓練完成後檢查 `v4_results/` 的 JSON 結果
4. 上傳到 HuggingFace
5. 部署 Web App
