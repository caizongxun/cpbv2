# CPB V2 Web Application Guide

## Overview

Local web interface to load and invoke V2 models for real-time cryptocurrency price and volatility predictions.

**Features:**
- Support for 20 cryptocurrency pairs
- Real-time inference using V2 models
- Sample K-line data generation
- Beautiful, responsive UI
- RESTful API endpoints

---

## Installation

### 1. Install Dependencies

```bash
pip install flask tensorflow numpy
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Prepare Models

Ensure V2 models are available in the correct location:

```
ALL_MODELS/
└── MODEL_V2/
    ├── v2_model_BTC_USDT.h5
    ├── v2_model_ETH_USDT.h5
    ├── ... (20 models total)
    ├── metadata.json
    └── README.md
```

If models don't exist, run the training pipeline first:

```python
import urllib.request

urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/notebooks/complete_v2_pipeline.py',
    'complete_v2_pipeline.py'
)

exec(open('complete_v2_pipeline.py').read())
```

---

## Running the Application

### Start the Web Server

```bash
python web_app.py
```

You should see:

```
================================================================================
               CPB V2 Model Web Application
================================================================================

Model Version: V2
Supported Pairs: 20
Pairs: BTC_USDT, ETH_USDT, SOL_USDT, ...

Starting server...
Open your browser: http://localhost:5000

API Endpoints:
  POST /api/generate-sample  - Generate sample K-line data
  POST /api/predict          - Run model inference
  GET  /api/status           - Check system status

================================================================================
```

### Access the Web Interface

Open your browser and go to:

```
http://localhost:5000
```

---

## Supported Cryptocurrency Pairs (20)

```
BTC_USDT    ETH_USDT    SOL_USDT    XRP_USDT    ADA_USDT
BNB_USDT    DOGE_USDT   LINK_USDT   AVAX_USDT   MATIC_USDT
ATOM_USDT   NEAR_USDT   FTM_USDT    ARB_USDT    OP_USDT
LIT_USDT    STX_USDT    INJ_USDT    LUNC_USDT   LUNA_USDT
```

---

## User Interface

### Main Page

1. **Model Version Display** - Shows V2 model information
2. **Pair Selection** - Dropdown to select from 20 pairs
3. **Candle Count Input** - Configure number of candles (20-100)
4. **Generate Sample Data** - Create synthetic K-line data for testing
5. **Run Inference** - Execute model prediction
6. **Results Display** - Show predicted price and volatility
7. **K-line Viewer** - Display input OHLCV data

### Workflow

```
1. Select a cryptocurrency pair
   ↓
2. Generate sample K-line data (or provide real data)
   ↓
3. Run inference on V2 model
   ↓
4. View predicted price and volatility
```

---

## API Endpoints

### 1. Generate Sample Data

**Endpoint:** `POST /api/generate-sample`

**Request:**

```json
{
  "pair": "BTC_USDT",
  "num_candles": 20
}
```

**Response:**

```json
{
  "success": true,
  "pair": "BTC_USDT",
  "klines": [
    {
      "open": 50000.0,
      "high": 50500.5,
      "low": 49500.2,
      "close": 50250.3,
      "volume": 500.5
    },
    ...
  ],
  "count": 20
}
```

### 2. Run Prediction

**Endpoint:** `POST /api/predict`

**Request:**

```json
{
  "pair": "BTC_USDT",
  "klines": [
    {"open": 50000, "high": 50500, "low": 49500, "close": 50250, "volume": 500},
    ...
  ]
}
```

**Response:**

```json
{
  "success": true,
  "pair": "BTC_USDT",
  "prediction": {
    "price": 0.5234,
    "volatility": 1.2456,
    "timestamp": "2025-12-23T15:30:00.123456"
  },
  "input_shape": [1, 20, 4]
}
```

### 3. System Status

**Endpoint:** `GET /api/status`

**Response:**

```json
{
  "model_version": "V2",
  "total_pairs": 20,
  "available_models": 20,
  "models": [
    {
      "pair": "BTC_USDT",
      "available": true
    },
    ...
  ],
  "timestamp": "2025-12-23T15:30:00.123456"
}
```

---

## Example Usage (cURL)

### Check System Status

```bash
curl http://localhost:5000/api/status
```

### Generate Sample Data

```bash
curl -X POST http://localhost:5000/api/generate-sample \
  -H "Content-Type: application/json" \
  -d '{
    "pair": "BTC_USDT",
    "num_candles": 20
  }'
```

### Run Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pair": "BTC_USDT",
    "klines": [...]
  }'
```

---

## Model Output

### Prediction Output: [price, volatility]

**Price (Index 0):**
- Predicted percentage change from first candle
- Range: typically -10% to +10%
- Example: 0.5234 = +0.5234% change

**Volatility (Index 1):**
- Predicted volatility (standard deviation of returns)
- Range: typically 0% to 3%
- Example: 1.2456 = 1.2456% volatility

---

## Troubleshooting

### Models Not Found

**Error:** `Failed to load model for BTC_USDT`

**Solution:**
1. Check that `ALL_MODELS/MODEL_V2/` exists
2. Ensure all model files (`v2_model_*.h5`) are present
3. Run training pipeline to generate models

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
1. Change port in `web_app.py`: `app.run(port=5001)`
2. Or kill the process using port 5000:
   ```bash
   lsof -i :5000
   kill -9 <PID>
   ```

### TensorFlow Import Error

**Error:** `ImportError: No module named tensorflow`

**Solution:**
```bash
pip install tensorflow
```

### No Module Named Flask

**Error:** `ImportError: No module named flask`

**Solution:**
```bash
pip install flask
```

---

## Configuration

Edit `web_app.py` to customize:

```python
CONFIG = {
    'model_version': 'V2',        # Model version
    'pairs': [...],               # 20 currency pairs
    'model_dir': 'ALL_MODELS/MODEL_V2',  # Model directory
    'input_shape': (20, 4)        # Input shape (seq_len, features)
}
```

---

## Switching Model Versions

To switch from V2 to V3 (when available):

1. Update `CONFIG['model_version']` in `web_app.py`:
   ```python
   CONFIG['model_version'] = 'V3'
   CONFIG['model_dir'] = 'ALL_MODELS/MODEL_V3'
   ```

2. Restart the server:
   ```bash
   python web_app.py
   ```

---

## Performance Tips

1. **Model Caching** - Models are cached in memory after first load
2. **Batch Predictions** - API accepts single predictions
3. **Parallel Requests** - Flask handles multiple concurrent requests
4. **GPU Support** - TensorFlow automatically uses GPU if available

---

## File Structure

```
.
├── web_app.py                    # Main application (this file)
├── ALL_MODELS/
│   └── MODEL_V2/                 # V2 models directory
│       ├── v2_model_BTC_USDT.h5
│       ├── v2_model_ETH_USDT.h5
│       ├── ... (20 models)
│       ├── metadata.json
│       └── README.md
├── requirements.txt              # Dependencies
└── WEB_APP_GUIDE.md             # This file
```

---

## Next Steps

1. Start the web application: `python web_app.py`
2. Open browser: `http://localhost:5000`
3. Select a trading pair
4. Generate sample data
5. Run inference
6. View results

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review console logs in terminal
3. Check model availability: `http://localhost:5000/api/status`
4. Verify models exist in `ALL_MODELS/MODEL_V2/`

---

**Created:** 2025-12-23  
**Model Version:** V2  
**Supported Pairs:** 20  
**Output:** [price, volatility]
