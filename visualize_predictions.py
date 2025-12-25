#!/usr/bin/env python3
"""
Interactive visualization of CPB v4 model predictions vs actual prices
Handles multiple model storage formats
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    os.system('pip install plotly -q')
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    os.system('pip install huggingface-hub -q')
    from huggingface_hub import hf_hub_download

try:
    import yfinance as yf
except ImportError:
    os.system('pip install yfinance -q')
    import yfinance as yf

# ========== MODEL ARCHITECTURE ==========

class TransformerModel(nn.Module):
    def __init__(self, input_size=4, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, seq_len=30, forecast_len=10):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.forecast_len = forecast_len
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.forecast_start = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = x + self.pos_encoder
        memory = self.encoder(x)
        tgt = self.forecast_start.expand(batch_size, -1, -1)
        output = self.decoder(tgt, memory)
        forecast = [output]
        for _ in range(self.forecast_len - 1):
            output = self.decoder(output, memory)
            forecast.append(output)
        output = torch.cat(forecast, dim=1)
        return self.output_proj(output)


# ========== CONFIG ==========

REPO_ID = "zongowo111/cpb-models"
MODEL_FOLDER = "model_v4"

COINS = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'ADA', 'SOL', 'DOGE', 'AVAX', 'LINK',
         'UNI', 'ATOM', 'NEAR', 'DYDX', 'ARB', 'OP', 'PEPE', 'INJ', 'SHIB', 'LUNA']

TIMEFRAMES = ['15m', '1h']

print("="*60)
print("CPB v4 Model Prediction Visualizer")
print("="*60 + "\n")

# ========== HELPER FUNCTIONS ==========

def build_model():
    return TransformerModel(input_size=4, d_model=64, nhead=4, num_layers=2,
                          dim_feedforward=256, seq_len=30, forecast_len=10)


def download_model(coin, timeframe):
    try:
        model_name = f"{coin}USDT_{timeframe}.pt"
        print(f"  {model_name:20s}", end=' ', flush=True)
        model_path = hf_hub_download(
            repo_id=REPO_ID, filename=f"{MODEL_FOLDER}/{model_name}",
            repo_type="dataset", cache_dir="./hf_models"
        )
        print("✓")
        return model_path
    except Exception as e:
        print(f"✗ {str(e)[:30]}")
        return None


def load_model_state(model_path):
    """Try multiple loading strategies"""
    try:
        model = build_model()
        data = torch.load(model_path, map_location='cpu')
        
        # Strategy 1: Direct state dict
        if isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()):
            try:
                model.load_state_dict(data)
                return model
            except:
                pass
        
        # Strategy 2: Wrapped in 'state_dict' key
        if isinstance(data, dict) and 'state_dict' in data:
            try:
                model.load_state_dict(data['state_dict'])
                return model
            except:
                pass
        
        # Strategy 3: Wrapped in 'model_state_dict' key
        if isinstance(data, dict) and 'model_state_dict' in data:
            try:
                model.load_state_dict(data['model_state_dict'])
                return model
            except:
                pass
        
        # Strategy 4: Already a model
        if isinstance(data, nn.Module):
            return data
        
        return None
    except Exception as e:
        return None


def get_crypto_data(symbol, timeframe, days=7):
    try:
        interval_map = {'15m': '15m', '1h': '1h'}
        ticker = f"{symbol}-USD"
        
        df = yf.download(ticker, period=f"{days}d", interval=interval_map[timeframe],
                        progress=False, threads=False)
        
        if len(df) == 0:
            return None
        
        return df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
    except:
        return None


def normalize_data(data):
    data_min = data.min(axis=0, keepdims=True)
    data_max = data.max(axis=0, keepdims=True)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1
    return (data - data_min) / data_range, data_min, data_max


def denormalize_data(data, data_min, data_max):
    return data * (data_max - data_min) + data_min


def predict_prices(model, data, lookback=30, forecast_len=10):
    try:
        normalized, data_min, data_max = normalize_data(data)
        if len(normalized) < lookback:
            return None
        
        input_seq = torch.FloatTensor(normalized[-lookback:]).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            forecast = model(input_seq)
        
        forecast_np = forecast[0, :, 3].cpu().numpy()
        forecast_denorm = denormalize_data(forecast_np.reshape(-1, 1),
                                          data_min[3:4], data_max[3:4]).flatten()
        actual_close = data[-forecast_len:, 3]
        
        return actual_close, forecast_denorm[:forecast_len]
    except:
        return None


def create_visualization(predictions_dict):
    valid_preds = {k: v for k, v in predictions_dict.items() if v is not None}
    
    if not valid_preds:
        return None
    
    rows, cols = 4, 5
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[k.replace('_', ' ') for k in sorted(valid_preds.keys())],
        specs=[[{} for _ in range(cols)] for _ in range(rows)]
    )
    
    for plot_idx, key in enumerate(sorted(valid_preds.keys())):
        actual, predicted = valid_preds[key]
        row = plot_idx // cols + 1
        col = plot_idx % cols + 1
        x_vals = list(range(1, len(actual) + 1))
        
        fig.add_trace(go.Scatter(x=x_vals, y=actual, mode='lines+markers',
                                name='Actual', line=dict(color='#2180a0', width=2),
                                marker=dict(size=4), showlegend=(plot_idx==0)),
                     row=row, col=col)
        fig.add_trace(go.Scatter(x=x_vals, y=predicted, mode='lines+markers',
                                name='Predicted', line=dict(color='#ff6b9d', width=2, dash='dash'),
                                marker=dict(size=4), showlegend=(plot_idx==0)),
                     row=row, col=col)
    
    fig.update_layout(title="CPB v4 Transformer: Actual vs Predicted Prices",
                     height=1400, showlegend=True, hovermode='closest')
    return fig


def create_summary_table(predictions_dict):
    metrics = []
    for key, pred_data in predictions_dict.items():
        if pred_data is None:
            continue
        actual, predicted = pred_data
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))
        metrics.append({'Model': key, 'MAE': f"${mae:.2f}", 'RMSE': f"{np.sqrt(mse):.2f}"})
    return pd.DataFrame(metrics)


# ========== MAIN ==========

if __name__ == "__main__":
    print("Step 1: Download & Load Models")
    print("-" * 60)
    
    models = {}
    for coin in COINS:
        for timeframe in TIMEFRAMES:
            model_path = download_model(coin, timeframe)
            if model_path:
                model = load_model_state(model_path)
                if model:
                    models[f"{coin}_{timeframe}"] = model
    
    print(f"\nLoaded {len(models)}/40 models\n")
    
    if not models:
        print("Failed to load any models")
        sys.exit(1)
    
    print("Step 2: Generate Predictions")
    print("-" * 60)
    
    predictions = {}
    count = 0
    for key in sorted(models.keys()):
        coin, timeframe = key.split('_')
        data = get_crypto_data(coin, timeframe, days=7)
        
        if data is None:
            print(f"{key:20s} No price data")
            continue
        
        model = models[key]
        result = predict_prices(model, data)
        
        if result is not None:
            predictions[key] = result
            actual, pred = result
            mae = np.mean(np.abs(actual - pred))
            print(f"{key:20s} {len(data):4d} candles MAE: ${mae:.2f}")
            count += 1
        else:
            print(f"{key:20s} Prediction failed")
    
    print(f"\nSuccessfully predicted: {count}/40\n")
    
    if predictions:
        print("\nPerformance Metrics")
        print("=" * 80)
        print(create_summary_table(predictions).to_string(index=False))
        
        print("\nCreating visualization...")
        fig = create_visualization(predictions)
        if fig:
            fig.write_html("cpb_predictions_visualization.html")
            print("✓ Saved: cpb_predictions_visualization.html")
    else:
        print("No predictions generated")
    
    print("\nDone!")
