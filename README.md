# CPB v2: Cryptocurrency Price Prediction with LSTM

Multi-timeframe LSTM model for predicting cryptocurrency price movements using 20+ major coins.

## Project Overview

- **Model Type**: PyTorch LSTM (Bidirectional, Multi-layer)
- **Coins**: 20+ (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, MATIC, OP, ARB, etc.)
- **Timeframes**: 15m, 1h
- **Data Source**: Binance API
- **Training**: Google Colab (Free tier, <2 hours)
- **Model Upload**: Hugging Face Hub (repo: `cpb`)
- **Features**: 35+ technical indicators
- **Target**: 7-day price change prediction

## Project Structure

```
cpbv2/
├── config/
│   ├── coins.json              # 20+ coins configuration
│   ├── indicators.json         # 35+ technical indicators
│   └── model_params.json       # LSTM hyperparameters
├── src/
│   ├── data_collector.py       # Binance API data collection
│   ├── feature_engineer.py     # Technical indicator calculation
│   ├── data_preprocessor.py    # Data cleaning & normalization
│   ├── model.py                # PyTorch LSTM architecture
│   ├── trainer.py              # Training pipeline
│   └── evaluator.py            # Model evaluation & metrics
├── notebooks/
│   └── train_colab.ipynb       # Google Colab training notebook
├── scripts/
│   ├── download_data.py        # Download data from Binance
│   ├── train_models.py         # Complete training pipeline
│   └── hf_upload.py            # Upload to Hugging Face
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md

## Setup & Usage

### Prerequisites
```bash
python >= 3.8
torch >= 2.0
pandas >= 1.3
numpy >= 1.21
ta-lib (or talib-python)
```

### Installation
```bash
git clone https://github.com/caizongxun/cpbv2.git
cd cpbv2
pip install -r requirements.txt
```

### Configure API Keys
```bash
cp .env.example .env
# Edit .env with your Binance API keys
export BINANCE_API_KEY="your_key"
export BINANCE_SECRET_KEY="your_secret"
```

### Training on Colab
1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Mount Google Drive
3. Configure coins and timeframes
4. Run training (should complete in <2 hours)
5. Models automatically saved to HF Hub

### Local Inference
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("caizongxun/cpb")
# Make predictions
```

## Model Architecture

```
Input: (batch_size, lookback=60, features=30)
    ↓
LSTM Layer 1: 96 units, bidirectional, dropout=0.2
    ↓
LSTM Layer 2: 64 units, bidirectional, dropout=0.2
    ↓
Dense Layer: 32 units, activation=relu, dropout=0.1
    ↓
Output: (batch_size, 1) - 7-day price change %
```

## Key Hyperparameters (Colab-optimized)

- **lookback_period**: 60 (steps)
- **LSTM units**: [96, 64] (bidirectional)
- **num_features**: 30 (after feature selection)
- **dropout**: 0.2
- **batch_size**: 32
- **epochs**: 50 (with early stopping)
- **learning_rate**: 0.001
- **train/val/test split**: 70/15/15 (time-series)

## Technical Indicators (35+)

### Price & Volume (5)
- open, high, low, close, volume

### Moving Averages (10)
- SMA(10,20,50,100,200), EMA(10,20,50,100,200)

### Momentum (9)
- RSI(14,21), MACD, Momentum(5), ROC(12), Stochastic

### Volatility (6)
- Bollinger Bands, ATR(14)

### Trend (7)
- ADX, DI+/-, Keltner Channels, NATR

### Volume (4)
- OBV, CMF(20), MFI(14), VPT

### Others (5)
- Price Change, Volume Change, HL2, Close Change, VWAP

## Performance Metrics

- **Accuracy**: Direction accuracy (UP/DOWN prediction)
- **RMSE**: Root mean squared error on test set
- **MAE**: Mean absolute error
- **MAPE**: Mean absolute percentage error
- **AUC**: ROC-AUC for binary classification

## Data Pipeline

```
Binance API
    ↓
Raw OHLCV data (3000+ candles per coin)
    ↓
Data validation & cleaning
    ↓
Technical indicator calculation
    ↓
Feature selection (30 from 35+)
    ↓
Normalization (MinMaxScaler)
    ↓
Time-series sequence creation (lookback=60)
    ↓
Train/Val/Test split (70/15/15)
    ↓
LSTM training
```

## Coins Included (20+)

1. BTCUSDT (Bitcoin)
2. ETHUSDT (Ethereum)
3. SOLUSDT (Solana)
4. BNBUSDT (Binance Coin)
5. XRPUSDT (XRP)
6. ADAUSDT (Cardano)
7. DOGEUSDT (Dogecoin)
8. AVAXUSDT (Avalanche)
9. MATICUSDT (Polygon)
10. OPUSDT (Optimism)
11. ARBUSDT (Arbitrum)
12. FTMUSDT (Fantom)
13. JPYUSDT (? - correcting to actual coin)
14. LINKUSDT (Chainlink)
15. UNIUSDT (Uniswap)
16. LTCUSDT (Litecoin)
17. BCUSDT (Bitcoin Cash)
18. ETCUSDT (Ethereum Classic)
19. NEOUSDT (NEO)
20. ATOMUSDT (Cosmos)
21. APTUSDT (Aptos)
22. SUIUSDT (Sui)

## Timeline

- **Data Collection**: ~15 min (Binance API)
- **Feature Engineering**: ~15 min
- **Model Training** (Colab free): ~90 min
- **Evaluation & Upload**: ~20 min
- **Total**: ~2 hours

## GPU/Memory Optimization (Colab Free)

- **Batch size**: 32 (fits in 15GB VRAM)
- **Lookback period**: 60 (not 90)
- **LSTM units**: [96, 64] (not [128, 64])
- **Num features**: 30 (reduced via PCA)
- **Mixed precision**: FP16 for speed
- **Gradient accumulation**: No (single forward pass)

## Model Upload

Models are automatically uploaded to:
- **Hugging Face**: `caizongxun/cpb`
- **Format**: PyTorch (.pt), configs, model card

## References

- LSTM: Hochreiter & Schmidhuber (1997)
- Technical Analysis: Wilder (1978), Bollinger (1984)
- Feature Selection: Guyon & Elisseeff (2003)
- Time-series validation: Hyndman & Athanasopoulos (2021)

## License

MIT License - See LICENSE file

## Author

CPB Team (Zongxun Cai)

---

**Last Updated**: 2025-12-22
**Status**: Development Phase 1
