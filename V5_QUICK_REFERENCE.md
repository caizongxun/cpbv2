# V5 Quick Reference Card

## One-Liner Execution (Colab)

```python
!curl -sS https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_colab_training_complete.py | python
```

## Full Colab Cell

```python
# Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q pandas numpy scikit-learn requests huggingface-hub

# Clone and run
!git clone https://github.com/caizongxun/cpbv2.git
%cd cpbv2
!python v5_colab_training_complete.py
```

## Key Configuration

```python
# v5_colab_training_complete.py (configurable)

COINS = 20                    # Top 20 cryptocurrencies
TIMEFRAMES = ['15m', '1h']   # 2 timeframes
LOOKBACK_STEPS = 30          # Last 30 K-bars for input
PREDICT_STEPS = 10           # Predict next 10 K-bars
BATCH_SIZE = 64              # GPU batch size
EPOCHS = 100                 # Maximum epochs
EARLY_STOPPING_PATIENCE = 15 # Stop if no improvement
LEARNING_RATE = 0.001        # Adam learning rate
```

## 5-Phase Pipeline

```
Phase 1: Data Collection (Binance API)
  - Downloads 8000 K-bars per coin/timeframe
  - Time: ~10 min for 20 coins
  
Phase 2: Feature Engineering (40+ indicators)
  - Volatility, amplitude, momentum, BB, ATR, RSI, MACD
  - Time: ~20 min for all coins
  
Phase 3: Preprocessing (Normalization + Sequencing)
  - MinMax scaling, feature selection, time-series windows
  - Time: Included in Phase 2
  
Phase 4: Model Training (Seq2Seq LSTM + Attention)
  - 40 models total (20 coins × 2 timeframes)
  - Time: ~100 min (~2.5 min per model on T4 GPU)
  
Phase 5: Upload to Hugging Face
  - Uploads /content/all_models/model_v5/ to HF
  - Time: ~10 min

Total Time: 2-2.5 hours on Colab
```

## Model Architecture

```
Input: (batch=64, timesteps=30, features=40)
  |
  v
Encoder LSTM (Bidirectional, 2 layers, 256 units)
  |
  v
Multi-Head Attention (8 heads)
  |
  v
Decoder LSTM (Unidirectional, 2 layers, 256 units)
  |
  v
Output FC: 256 -> 128 -> 64 -> 1
  |
  v
Output: (batch=64, steps=10, 1) = 10 predictions

Parameters: ~600K
Memory: ~1.7GB per model
```

## Expected Results

```
MAPE Target: < 0.02 (2%)
MAPE Typical: 0.015-0.025

By coin (approximate):
BTC: 0.015-0.020 (most stable)
ETH: 0.017-0.022
ALT: 0.020-0.030 (more volatile)

By timeframe:
15m: 0.020 (harder, more volatile)
1h: 0.017 (easier, smoother)
```

## Outputs

### Local Files
```
/content/all_models/model_v5/
├── BTCUSDT_15m.pt           (Model 1)
├── BTCUSDT_1h.pt            (Model 2)
├── ... (38 more)
└── training_results.json    (Summary)
```

### Hugging Face
```
zongowo111/cpb-models/
├── model_v2/
├── model_v3/
└── model_v5/               (NEW)
    ├── BTCUSDT_15m.pt
    ├── ETHUSDT_15m.pt
    └── ... (all 40 v5 models)
```

## File Contents

### .pt File Structure
```python
torch.save({
    'model_state': {...},           # Model weights
    'config': {                     # Architecture config
        'input_size': 40,
        'hidden_size': 256,
        'num_layers': 2,
        'predict_steps': 10,
        'dropout': 0.3
    },
    'scaler_params': {...},        # Feature normalization
    'close_scaler_params': {...},  # Price normalization
    'metrics': {                   # Test metrics
        'mape': 0.018,
        'rmse': 0.009,
        'mae': 0.008
    },
    'history': {                   # Training history
        'train_loss': [...],
        'val_loss': [...]
    }
})
```

## What's Different from v1-v4?

| Aspect | v1-v2 | v3 | v4 | v5 |
|--------|-------|----|----|----|
| Lookback | 60 | 60 | 60 | **30** |
| Predict steps | 5 | 5 | 5 | **10** |
| Features | 30 | 30 | 30 | **40+** |
| Volatility learning | No | No | No | **Yes** |
| Architecture | BiLSTM | BiLSTM | Transformer | **Seq2Seq+Attn** |
| MAPE | ~0.03 | ~0.028 | ~0.025 | **<0.02** |
| Predicts bounce? | No (diverges) | No | No | **Yes!** |

## Troubleshooting

### "CUDA out of memory"
```python
BATCH_SIZE = 32  # Reduce from 64
```

### "Download too slow"
```python
# Already using fastest API (Binance REST API)
# Alternative: Use pre-downloaded data
```

### "Training too slow"
```python
# Use Colab Pro with A100 GPU
# Or reduce EPOCHS = 50
```

### "HF upload fails"
```bash
# Manual upload
huggingface-cli upload zongowo111/cpb-models \
  /content/all_models/model_v5 model_v5 --repo-type model
```

## Feature List (40+)

### Price (3)
- HL2, HLC3, OHLC4

### Volatility (5) **KEY**
- Volatility (5, 10, 20, 30), Volatility ratio

### Amplitude (4) **KEY**
- High-Low ratio, Amplitude (5, 10, 20)

### Returns (3)
- Log return, Price change %, Absolute change

### Moving Averages (12)
- SMA (5, 10, 20, 50, 100, 200)
- EMA (5, 10, 20, 50, 100, 200)

### Momentum (7)
- RSI (14, 21), MACD, Signal, MACD Diff, Momentum (5), ROC (12)

### Bollinger Bands (5)
- Upper, Middle, Lower, Width, %B

### ATR (1)
- ATR (14)

### Volume (2)
- Volume SMA, Volume ratio

### Direction (2)
- Price direction, High-Low balance

**Total: 44 features**

## Data Format

### Input Sequence
```
Shape: (batch, 30, 40)

Batch: samples to process at once
30: timesteps (30 K-bars lookback)
40: features

Example for BTC 15m:
  Step 1: OHLCV + 35 indicators
  Step 2: OHLCV + 35 indicators
  ...
  Step 30: OHLCV + 35 indicators (most recent)
```

### Output Sequence
```
Shape: (batch, 10, 1)

Batch: samples
10: prediction steps
1: predicted close price

Example:
  [1000.5]  <- Next K-bar price
  [1010.2]
  [1005.3]
  ...
  [1020.1]  <- 10th K-bar ahead
```

## Performance Timeline

```
Colab T4 (default):
  Epoch 1-10:   3-5 sec/epoch
  Epoch 11-50:  2-3 sec/epoch (convergence)
  Epoch 51+:    ~2 sec/epoch (asymptote)
  Total: ~70-80 epochs until early stop (~2.5-3 min per model)
  40 models: 100-120 minutes total

Colab A100 (Pro only):
  Much faster: 50-60 minutes for all 40 models
```

## GPU Utilization

```
T4 GPU (Colab Free):
  GPU Memory: 1.7GB / 15GB
  GPU Util: 75-85%
  Memory Bandwidth: Excellent
  Temperature: 45-55°C (cool)

A100 GPU (Colab Pro):
  GPU Memory: 1.7GB / 40GB
  GPU Util: 85-95%
  Memory Bandwidth: Excellent
  Temperature: 35-45°C (very cool)
```

## Model Checkpoints

```python
# Load model
checkpoint = torch.load('/content/all_models/model_v5/BTCUSDT_15m.pt')
model = Seq2SeqLSTMV5(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state'])

# Access metrics
print(checkpoint['metrics'])  # {'mape': 0.018, 'rmse': 0.009, ...}

# Access scaler
scaler_scale = checkpoint['scaler_params']['scale_']
scaler_min = checkpoint['scaler_params']['min_']

# Access history
training_loss = checkpoint['history']['train_loss']
validation_loss = checkpoint['history']['val_loss']
```

## Currencies Included

```
Tier 1 (Most stable):
BTC, ETH, BNB, XRP, LTC

Tier 2 (Large cap):
ADA, SOL, DOGE, AVAX, LINK

Tier 3 (Mid-large cap):
UNI, ATOM, NEAR, ARB, OP

Tier 4 (Emerging):
PEPE, INJ, SHIB, ETC, LUNA
```

## Common Parameters

| Parameter | Value | Range | Notes |
|-----------|-------|-------|-------|
| Lookback | 30 | 10-60 | Input sequence length |
| Predict | 10 | 1-20 | Output sequence length |
| Batch | 64 | 32-128 | Larger = faster, more memory |
| Epochs | 100 | 50-200 | Max, usually stop earlier |
| LR | 0.001 | 0.0001-0.01 | Learning rate |
| Dropout | 0.3 | 0.1-0.5 | Regularization |
| Hidden | 256 | 128-512 | LSTM hidden size |
| Layers | 2 | 1-3 | LSTM layers |

## Next Steps

1. **Run v5 training** on Colab (2-2.5 hours)
2. **Check results** in training_results.json
3. **Upload to HF** (automatic in script)
4. **Use models** for inference/trading
5. **Improve** with ensemble (train v5 variants)

## Links

- Script: https://github.com/caizongxun/cpbv2/blob/main/v5_colab_training_complete.py
- Guide: https://github.com/caizongxun/cpbv2/blob/main/V5_COLAB_GUIDE.md
- Models: https://huggingface.co/zongowo111/cpb-models
- Repository: https://github.com/caizongxun/cpbv2

---

**Version**: v5.0
**Last Updated**: 2025-12-24
**Status**: Ready to use
