#!/usr/bin/env python3
"""
Interactive visualization of CPB v4 model predictions vs actual prices
Loads models from HF and shows real vs predicted price trends
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("Installing required packages...")
    os.system('pip install plotly -q')
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface-hub...")
    os.system('pip install huggingface-hub -q')
    from huggingface_hub import hf_hub_download

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system('pip install yfinance -q')
    import yfinance as yf

# ========== CONFIG ==========

REPO_ID = "zongowo111/cpb-models"
MODEL_FOLDER = "model_v4"

# 支持的幣種
COINS = [
    'BTC', 'ETH', 'BNB', 'XRP', 'LTC',
    'ADA', 'SOL', 'DOGE', 'AVAX', 'LINK',
    'UNI', 'ATOM', 'NEAR', 'DYDX', 'ARB',
    'OP', 'PEPE', 'INJ', 'SHIB', 'LUNA'
]

TIMEFRAMES = ['15m', '1h']

print("="*60)
print("CPB v4 Model Prediction Visualizer")
print("="*60 + "\n")

# ========== HELPER FUNCTIONS ==========

def get_model_path(coin, timeframe):
    """Get local model path"""
    return f"{coin}USDT_{timeframe}.pt"


def download_model(coin, timeframe):
    """Download model from HF"""
    try:
        model_name = get_model_path(coin, timeframe)
        print(f"Downloading {model_name}...", end=' ', flush=True)
        
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{MODEL_FOLDER}/{model_name}",
            repo_type="dataset",
            cache_dir="./hf_models"
        )
        print("✓")
        return model_path
    except Exception as e:
        print(f"✗ {e}")
        return None


def get_crypto_data(symbol, timeframe, days=7):
    """Get cryptocurrency price data from yfinance"""
    try:
        # Map timeframe to yfinance interval
        interval_map = {'15m': '15m', '1h': '1h'}
        interval = interval_map.get(timeframe, '1h')
        
        ticker = f"{symbol}-USD"
        print(f"  Fetching {ticker} ({timeframe})...", end=' ', flush=True)
        
        df = yf.download(
            ticker,
            period=f"{days}d",
            interval=interval,
            progress=False,
            threads=False
        )
        
        if len(df) == 0:
            print("✗ No data")
            return None
        
        # Normalize data
        data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
        print(f"✓ ({len(data)} candles)")
        return data
    except Exception as e:
        print(f"✗ {e}")
        return None


def normalize_data(data):
    """Normalize using min-max scaling"""
    data_min = data.min(axis=0, keepdims=True)
    data_max = data.max(axis=0, keepdims=True)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1  # Avoid division by zero
    return (data - data_min) / data_range, data_min, data_max


def denormalize_data(data, data_min, data_max):
    """Denormalize from min-max scaling"""
    data_range = data_max - data_min
    return data * data_range + data_min


def predict_prices(model, data, lookback=30, forecast_len=10):
    """Generate predictions using the model"""
    try:
        # Normalize
        normalized, data_min, data_max = normalize_data(data)
        
        # Prepare input
        if len(normalized) < lookback:
            return None
        
        # Use last lookback window
        input_seq = torch.FloatTensor(normalized[-lookback:]).unsqueeze(0)
        
        # Predict
        model.eval()
        with torch.no_grad():
            forecast = model(input_seq)
        
        # Denormalize (use Close prices for visualization)
        forecast_np = forecast[0, :, 3].cpu().numpy()  # Column 3 = Close
        forecast_denorm = denormalize_data(
            forecast_np.reshape(-1, 1),
            data_min[3:4],
            data_max[3:4]
        ).flatten()
        
        # Actual close prices
        actual_close = data[-forecast_len:, 3]
        
        return actual_close, forecast_denorm[:forecast_len]
    except Exception as e:
        print(f"  Prediction error: {e}")
        return None


def create_visualization(predictions_dict):
    """Create interactive Plotly visualization"""
    print("\nCreating visualization...")
    
    # Count valid predictions
    valid_preds = {k: v for k, v in predictions_dict.items() if v is not None}
    
    if not valid_preds:
        print("No valid predictions to visualize")
        return None
    
    # Create subplots (4 rows x 5 cols for 20 coins)
    rows, cols = 4, 5
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{k.replace('_', ' ')}" for k in sorted(valid_preds.keys())],
        specs=[[{} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Add traces
    plot_idx = 0
    for key in sorted(valid_preds.keys()):
        actual, predicted = valid_preds[key]
        
        row = plot_idx // cols + 1
        col = plot_idx % cols + 1
        
        x_vals = list(range(1, len(actual) + 1))
        
        # Actual prices
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=actual,
                mode='lines+markers',
                name='Actual',
                line=dict(color='#2180a0', width=2),
                marker=dict(size=4),
                hovertemplate='Step %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Predicted prices
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=predicted,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#ff6b9d', width=2, dash='dash'),
                marker=dict(size=4),
                hovertemplate='Step %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Calculate metrics
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))
        
        plot_idx += 1
    
    # Update layout
    fig.update_xaxes(title_text="Step", row=rows, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    
    fig.update_layout(
        title="CPB v4 Transformer: Actual vs Predicted Prices",
        height=1200,
        showlegend=True,
        hovermode='closest',
        font=dict(size=10)
    )
    
    return fig


def create_summary_table(predictions_dict):
    """Create summary metrics table"""
    metrics = []
    
    for key, pred_data in predictions_dict.items():
        if pred_data is None:
            continue
        
        actual, predicted = pred_data
        
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100
        
        metrics.append({
            'Model': key,
            'MSE': f"{mse:.6f}",
            'RMSE': f"{rmse:.2f}",
            'MAE': f"{mae:.2f}",
            'MAPE': f"{mape:.2f}%"
        })
    
    return pd.DataFrame(metrics)


# ========== MAIN ==========

if __name__ == "__main__":
    print("\nStep 1: Download Models")
    print("-" * 60)
    
    models = {}
    for coin in COINS:
        for timeframe in TIMEFRAMES:
            model_path = download_model(coin, timeframe)
            if model_path:
                try:
                    model = torch.load(model_path, map_location='cpu')
                    models[f"{coin}_{timeframe}"] = model
                except Exception as e:
                    print(f"  Error loading {coin}_{timeframe}: {e}")
    
    print(f"\nLoaded {len(models)} models\n")
    
    if not models:
        print("No models loaded")
        sys.exit(1)
    
    print("\nStep 2: Fetch Price Data")
    print("-" * 60)
    
    predictions = {}
    for key in sorted(models.keys()):
        coin, timeframe = key.split('_')
        print(f"\n{coin} {timeframe}:")
        
        # Get data
        data = get_crypto_data(coin, timeframe, days=7)
        if data is None:
            continue
        
        # Predict
        model = models[key]
        result = predict_prices(model, data)
        if result is not None:
            predictions[key] = result
    
    print(f"\n\nGenerated {len(predictions)} predictions\n")
    
    # Summary table
    print("\nPerformance Metrics")
    print("=" * 60)
    summary_df = create_summary_table(predictions)
    print(summary_df.to_string(index=False))
    
    # Create visualization
    fig = create_visualization(predictions)
    if fig:
        output_file = "cpb_predictions_visualization.html"
        fig.write_html(output_file)
        print(f"\n✓ Visualization saved to: {output_file}")
        print(f"Open in browser: file://{os.path.abspath(output_file)}")
    
    print("\nDone!")
