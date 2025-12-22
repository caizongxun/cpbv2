# CPB v2 Project Summary

## Overview

**CPB v2** (Cryptocurrency Price Prediction v2) is a complete deep learning pipeline for training LSTM models to predict cryptocurrency price movements.

- **Framework**: PyTorch
- **Model**: Bidirectional LSTM (2 layers)
- **Data**: Binance REST API (21 coins, 2 timeframes)
- **Features**: 35+ technical indicators
- **Training**: Google Colab free GPU (< 2 hours)
- **Deployment**: Hugging Face Hub

---

## Project Structure

```
cpbv2/
├── config/
│   ├── coins.json              # 21 cryptocurrency configurations
│   ├── model_params.json       # LSTM hyperparameters (Colab-optimized)
│   └── indicators.json         # 35+ technical indicator specs
│
├── src/
│   ├── data_collector.py       # Binance API data collection (retry logic)
│   ├── feature_engineer.py     # 35+ technical indicator calculations
│   ├── data_preprocessor.py    # Normalization, feature selection, sequences
│   ├── model.py                # PyTorch LSTM architecture
│   ├── trainer.py              # Training pipeline with early stopping
│   └── evaluator.py            # [Future] Model evaluation metrics
│
├── scripts/
│   ├── train_models.py         # Complete training pipeline
│   ├── hf_upload.py            # Upload models to Hugging Face
│   └── inference.py            # [Future] Local inference
│
├── notebooks/
│   └── train_colab.ipynb       # Google Colab training notebook
│
├── data/
│   └── raw/                    # CSV files from Binance API
│
├── models/
│   └── *.pt                    # Trained model weights
│
├── results/
│   └── training_results.json   # Training metrics and logs
│
├── requirements.txt            # Python dependencies
├── README.md                   # Full documentation
├── QUICKSTART.md               # Colab quick start guide
├── .gitignore                  # Git ignore patterns
└── PROJECT_SUMMARY.md          # This file
```

---

## Key Components

### 1. Data Collection (`src/data_collector.py`)

**Purpose**: Download historical K-line data from Binance API

**Features**:
- Retry logic with exponential backoff
- Handles API rate limiting (0.1s delay)
- Data validation (3000+ candles, no NaNs, ordering)
- Batch downloading (1000 candles per request)
- Error recovery for failed coins

**Output**: CSV files with OHLCV data

```python
collector = BinanceDataCollector()
df = collector.get_historical_klines('BTCUSDT', '15m', limit=3000)
# Columns: timestamp, symbol, interval, open, high, low, close, volume
```

### 2. Feature Engineering (`src/feature_engineer.py`)

**Purpose**: Calculate 35+ technical indicators

**Indicators** (organized by category):

| Category | Count | Examples |
|----------|-------|----------|
| Price & Volume | 7 | open, high, low, close, volume, hl2, hlc3 |
| Moving Averages | 10 | SMA(10,20,50,100,200), EMA(10,20,50,100,200) |
| Momentum | 9 | RSI(14,21), MACD, Momentum(5), ROC(12), Stochastic(K,D) |
| Volatility | 6 | BB(upper,middle,lower,width,%B), ATR(14) |
| Trend | 7 | ADX(14), DI+/-(14), Keltner Channels, NATR |
| Volume | 4 | OBV, CMF(20), MFI(14), VPT |
| Changes | 3 | Price Change%, Volume Change%, Close Change |
| **Total** | **46** | **35+ after feature selection** |

**Output**: DataFrame with all indicators calculated

### 3. Data Preprocessing (`src/data_preprocessor.py`)

**Purpose**: Prepare data for LSTM training

**Pipeline**:
1. **Remove NaNs**: Drop rows with missing values
2. **Feature Selection**: 
   - Correlation analysis (drop >0.95 correlated pairs)
   - PCA dimensionality reduction (35 → 30 features)
3. **Normalization**: MinMaxScaler (0-1 range)
4. **Sequence Creation**: Time-series windows (lookback=60)
5. **Train/Val/Test Split**: 70/15/15 (time-series aware)

**Output**: Numpy arrays (X, y) ready for training

```python
X: (samples, 60, 30)   # 60 timesteps, 30 features
y: (samples, 1)        # Target values
```

### 4. Model Architecture (`src/model.py`)

**PyTorch LSTM**:

```
Input Layer: (batch_size, 60, 30)
    ↓
LSTM Layer 1: 96 units, bidirectional, dropout=0.2
    ↓
LSTM Layer 2: 64 units, bidirectional, dropout=0.2
    ↓
Dense Layer 1: 32 units, ReLU, dropout=0.1
    ↓
Output Layer: 1 unit, Linear
    ↓
Output: (batch_size, 1)
```

**Optimizations for Colab Free GPU**:
- LSTM units: [96, 64] (not [128, 64])
- Lookback: 60 (not 90)
- Batch size: 32
- Features: 30 (reduced via PCA)
- Parameters: ~180K (fits in 15GB VRAM)

### 5. Training Pipeline (`src/trainer.py`)

**Features**:
- Adam optimizer (lr=0.001)
- MSE loss function
- Early stopping (patience=15, min_delta=0.0001)
- Gradient clipping (max_norm=1.0)
- Checkpoint saving (best model)

**Training Configuration**:
- Epochs: 50 (with early stopping)
- Batch size: 32
- Time: ~5-7 min per model on Colab
- Total for 3 models: ~20 min

---

## Coins & Timeframes

### 21 Cryptocurrencies

| Layer 1 | Payment | DeFi | Scaling | Others |
|---------|---------|------|---------|--------|
| BTC | XRP | UNI | MATIC | LINK |
| ETH | LTC | - | OP | FTM |
| SOL | BCH | - | ARB | ATOM |
| BNB | - | - | - | APT |
| ADA | - | - | - | SUI |
| AVAX | - | - | - | - |
| DOGE | - | - | - | - |
| NEO | - | - | - | - |
| ETC | - | - | - | - |

### Timeframes
- **15m**: Intraday trading signals
- **1h**: Medium-term trends

**Total Models**: 21 coins × 2 timeframes = **42 models**

---

## Workflow

### Phase 1: Data Preparation (15-20 min)

```
Binance API
    ↓
Download 3000+ candles/coin/timeframe
    ↓
Validate data (no NaNs, proper ordering, min length)
    ↓
Save to CSV (data/raw/)
```

### Phase 2: Feature Engineering (10-15 min)

```
Raw OHLCV
    ↓
Calculate 35+ technical indicators
    ↓
Select top 30 features (correlation + PCA)
    ↓
Normalize to [0, 1]
    ↓
Create time-series sequences
```

### Phase 3: Model Training (60-75 min)

```
Preprocessed sequences
    ↓
Split train/val/test (70/15/15)
    ↓
Build LSTM model
    ↓
Train with early stopping
    ↓
Save best weights
```

### Phase 4: Upload & Deployment (10-15 min)

```
Trained models
    ↓
Create model cards (README)
    ↓
Upload to Hugging Face
    ↓
Ready for inference
```

---

## Performance Metrics

### Training Metrics
- **Loss**: MSE (Mean Squared Error)
- **Validation Loss**: Monitored for early stopping
- **Best Epoch**: Epoch with lowest validation loss

### Expected Results (Colab Free GPU)

```
BTC 15m:   Val Loss ≈ 0.0002-0.0005, 40-50 epochs
ETH 1h:    Val Loss ≈ 0.0002-0.0004, 38-48 epochs
SOL 15m:   Val Loss ≈ 0.0003-0.0006, 35-45 epochs
```

### Evaluation (Future)
- Direction Accuracy (UP/DOWN prediction)
- Confusion Matrix (TP, TN, FP, FN)
- ROC-AUC Score
- MAPE (Mean Absolute Percentage Error)

---

## Hardware Requirements

### Google Colab Free Tier
- GPU: T4 (15GB VRAM) ✓
- RAM: 12GB ✓
- Storage: 5GB free ✓
- Time limit: 12 hours ✓
- Estimated time: 1.5-2 hours ✓

### Local GPU (RTX 3060+)
- VRAM: 12GB+
- RAM: 16GB
- Time per model: 5-10 min
- Time for all: 3-7 hours

### Local CPU
- RAM: 32GB+
- Time per model: 30-60 min
- Time for all: 20-40 hours
- Not recommended

---

## Hugging Face Integration

### Model Repository: `caizongxun/cpb`

**Structure**:
```
models/
├── BTCUSDT_15m.pt
├── BTCUSDT_1h.pt
├── ETHUSDT_15m.pt
├── ...

cards/
├── BTCUSDT_15m.md
├── BTCUSDT_1h.md
├── ...

training_results.json
```

**Features**:
- Model weights in PyTorch format
- Detailed model cards (markdown)
- Training results summary
- Automatic versioning

**Usage**:
```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="caizongxun/cpb",
    filename="models/BTCUSDT_15m.pt"
)
```

---

## Technology Stack

### Core ML
- **PyTorch**: Neural network framework
- **Scikit-learn**: Preprocessing & metrics
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

### Data
- **Binance API**: Cryptocurrency data source
- **ta (Technical Analysis)**: Indicator calculations
- **requests**: HTTP client

### Deployment
- **Hugging Face Hub**: Model repository
- **GitHub**: Source code & documentation
- **Google Colab**: Training environment

---

## Training Time Breakdown (Colab Free GPU)

| Task | Time | Notes |
|------|------|-------|
| Setup (clone + install) | 2 min | One-time |
| Download data (6 datasets) | 15 min | Binance API rate limit |
| Feature engineering (3 coins) | 3 min | Per coin: ~1 min |
| Model training (3 models) | 75 min | Per model: ~25 min (50 epochs) |
| Evaluation & saving | 5 min | Model checkpoints |
| Upload to HF | 10 min | Optional |
| **Total** | **~110 min** | **< 2 hours** |

---

## Future Enhancements

### Phase 2
- [ ] Inference pipeline (`scripts/inference.py`)
- [ ] Model evaluation metrics (`src/evaluator.py`)
- [ ] Backtesting framework
- [ ] Web dashboard for results

### Phase 3
- [ ] Ensemble models (multiple LSTM variants)
- [ ] Transformer architecture (attention mechanism)
- [ ] Real-time prediction API
- [ ] Model serving (TorchServe, FastAPI)

### Phase 4
- [ ] Reinforcement learning for trading
- [ ] Multi-objective optimization
- [ ] Federated learning
- [ ] Crypto native integration (on-chain)

---

## Usage Examples

### Quick Start (Colab)
1. Open [Colab notebook](https://github.com/caizongxun/cpbv2/blob/main/notebooks/train_colab.ipynb)
2. Run all cells (auto-setup)
3. Models train automatically
4. Results saved locally + HF Hub

### Local Training
```bash
git clone https://github.com/caizongxun/cpbv2.git
cd cpbv2
pip install -r requirements.txt

# Download data, train models, upload results
python scripts/train_models.py

# Upload to Hugging Face
export HF_TOKEN="your_token"
python scripts/hf_upload.py
```

### Inference (Future)
```python
import torch
from huggingface_hub import hf_hub_download

# Load model
model_path = hf_hub_download(repo_id="caizongxun/cpb", filename="models/BTCUSDT_15m.pt")
model = torch.load(model_path)

# Make prediction
with torch.no_grad():
    output = model(input_data)
```

---

## References

### Papers
- Hochreiter & Schmidhuber (1997): LSTM original paper
- Graves (2012): LSTM with bidirectional
- Wilder (1978): Technical analysis foundations
- Guyon & Elisseeff (2003): Feature selection

### Resources
- [PyTorch Docs](https://pytorch.org/docs/)
- [Binance API](https://binance-docs.github.io/apidocs/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/)
- [Technical Analysis Library](https://github.com/bukosabino/ta)

---

## License

MIT License - See LICENSE file

---

## Contact

**Author**: Cai Zongxun  
**GitHub**: https://github.com/caizongxun  
**Email**: 69517696+caizongxun@users.noreply.github.com

---

**Last Updated**: 2025-12-22  
**Status**: Development Phase 1 (Training Pipeline Complete)
