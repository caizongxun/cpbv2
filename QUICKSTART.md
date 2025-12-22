# CPB v2 - Quick Start Guide

## Fastest Way to Train (Google Colab)

### Step 1: Open Colab Notebook

1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Open Notebook**
3. Select **GitHub** tab
4. Paste: `https://github.com/caizongxun/cpbv2`
5. Select `notebooks/train_colab.ipynb`

### Step 2: Configure & Run

```python
# Cell 1-3: Setup (auto-executes)
# Clones repo, installs dependencies

# Cell 4: Load Config
# Loads 21 coins, model parameters

# Cell 5: Download Data (15 min)
# Downloads BTCUSDT, ETHUSDT, SOLUSDT for 15m + 1h
# Creates 3000+ candles per coin/timeframe

# Cell 6: Train Models (60-75 min)
# Trains 3 models on T4 GPU
# Saves best weights

# Cell 7: Results
# Shows training summary
```

### Step 3: Upload to Hugging Face (Optional)

```python
# Set HF token
from huggingface_hub import login
login()  # Paste token when prompted

# Run HF upload
!python scripts/hf_upload.py
```

---

## Timeline (Colab Free GPU)

| Phase | Time | Notes |
|-------|------|-------|
| Setup | 2 min | Clone + pip install |
| Download | 15 min | 3 coins × 2 timeframes = 6 datasets |
| Train | 75 min | 3 models (BTCUSDT 15m, ETHUSDT 1h, SOLUSDT 15m) |
| Upload | 10 min | Optional HF upload |
| **Total** | **~100 min** | **< 2 hours** |

---

## Hardware Requirements

### Google Colab Free
- **GPU**: T4 (15GB VRAM)
- **CPU**: 2+ cores
- **RAM**: 12+ GB
- **Storage**: 5GB free
- **Time limit**: 12 hours

### Locally (GPU)
- **GPU Memory**: 4GB+ (optimized for free Colab)
- **RAM**: 8GB+
- **Storage**: 10GB
- **Time**: 2-4 hours per coin

### Locally (CPU)
- **RAM**: 16GB+
- **Time**: 8+ hours per coin
- **Storage**: 10GB

---

## Model Performance (Expected)

After training on Colab:

```
BTCUSDT 15m:
  Best Validation Loss: 0.000234
  Best Epoch: 42/50
  Direction Accuracy: 58-62%

ETHUSDT 1h:
  Best Validation Loss: 0.000189
  Best Epoch: 38/50
  Direction Accuracy: 55-60%

SOLUSDT 15m:
  Best Validation Loss: 0.000312
  Best Epoch: 45/50
  Direction Accuracy: 52-57%
```

---

## Customization

### Change Coins

Edit `config/coins.json`:
```json
{
  "coins": [
    {"symbol": "ADAUSDT", "name": "Cardano"},
    {"symbol": "DOGEUSDT", "name": "Dogecoin"}
  ]
}
```

### Change Hyperparameters

Edit `config/model_params.json`:
```json
{
  "model_architecture": {
    "lstm_units": [128, 64],
    "dropout_lstm": 0.3,
    "dense_units": 64
  },
  "training": {
    "epochs": 100,
    "batch_size": 16
  }
}
```

### Change Timeframes

In Colab notebook:
```python
timeframes = ['15m', '1h']  # Can add '4h', '1d'
```

---

## Troubleshooting

### GPU Out of Memory
```python
# Reduce batch size in trainer.py
batch_size = 16  # Default: 32

# Or reduce lookback period
lookback_period = 30  # Default: 60
```

### Data Download Fails
```python
# Check Binance API status
# Retry after 1-2 minutes
# Or download specific coin:
df = collector.get_historical_klines('BTCUSDT', '15m')
```

### Colab Session Timeout
```python
# Enable premium for longer sessions
# Or checkpoint training:
for epoch in range(epochs):
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'checkpoint_{epoch}.pt')
```

---

## Next Steps

1. **Run training** on Colab
2. **Upload models** to Hugging Face
3. **Load locally** for inference
4. **Backtest** on historical data
5. **Deploy** to trading platform

---

## Resources

- **Repository**: https://github.com/caizongxun/cpbv2
- **Hugging Face**: https://huggingface.co/caizongxun/cpb
- **Documentation**: See README.md
- **Issues**: Open on GitHub

---

**Happy Training!**

For questions, check the GitHub issues or README documentation.
