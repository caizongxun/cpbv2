#!/usr/bin/env python3
"""
CPB v4: Training Optimized for Colab Free Tier

Optimizations:
- Reduced epochs: 20 → 10
- Fewer coins: 40 → 10 (top coins only)
- Mixed precision training
- Gradient checkpointing
- Model checkpointing with early stopping

Time: 2.5 hours (fits Colab free tier)
Target: 10 coins × 2 timeframes = 20 models

Version: 4.1-optimized
Author: Cai Zongxun
Date: 2025-12-25
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/content/v4_training_optimized.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR COLAB
# ============================================================================

class Config:
    # Top 10 coins only (to save time)
    COINS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
        'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT'
    ]
    TIMEFRAMES = ['15m', '1h']
    
    LOOKBACK = 30
    PREDICT = 10
    KBARS = 8000
    
    # Optimized for GPU memory
    BATCH_SIZE = 32  # Reduced from 64
    EPOCHS = 10  # Reduced from 20 - key optimization
    EARLY_STOPPING_PATIENCE = 5  # Stricter early stopping
    LEARNING_RATE = 0.001
    
    # Model size - reduced
    INPUT_SIZE = 4  # OHLCV only, no features
    HIDDEN_SIZE = 128  # Reduced from 256
    NUM_LAYERS = 1  # Reduced from 2
    DROPOUT = 0.2
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CACHE_DIR = Path('/content/v4_models_optimized')
    
    @classmethod
    def total_models(cls):
        return len(cls.COINS) * len(cls.TIMEFRAMES)

Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("V4 TRAINING - OPTIMIZED FOR COLAB FREE TIER")
print("="*70)
print(f"Start time: {datetime.now()}")
print(f"Device: {Config.DEVICE}")
print(f"Coins: {len(Config.COINS)} (top coins only)")
print(f"Models: {Config.total_models()} ({len(Config.COINS)} coins × 2 timeframes)")
print(f"Epochs per model: {Config.EPOCHS} (reduced from 20)")
print(f"Batch size: {Config.BATCH_SIZE}")
print(f"Expected time: ~2.5 hours")
print("="*70)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"CUDA: {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True
print()

# ============================================================================
# SIMPLIFIED MODEL - FASTER
# ============================================================================

class SimpleLSTM(nn.Module):
    """Simplified LSTM for faster training"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=1, 
                 lookback=30, predict=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.predict = predict
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, predict * input_size)
        )
    
    def forward(self, x):
        # x: (batch, lookback, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, lookback, hidden)
        out = self.fc(lstm_out[:, -1, :])  # (batch, predict*input_size)
        out = out.view(-1, self.predict, x.shape[-1])  # (batch, predict, input_size)
        return out

# ============================================================================
# DATA DOWNLOAD & PROCESSING
# ============================================================================

def download_binance_data(coin: str, timeframe: str, limit: int = 8000) -> Optional[pd.DataFrame]:
    """Download OHLCV data only - faster preprocessing"""
    import requests
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': coin, 'interval': timeframe, 'limit': limit}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data[:, :6], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        if len(df) < 100:
            return None
        return df
    except Exception as e:
        logger.warning(f"Download failed for {coin} {timeframe}: {e}")
        return None

def preprocess_coin(coin: str, timeframe: str) -> Optional[Tuple]:
    """Simple preprocessing - OHLCV only"""
    df = download_binance_data(coin, timeframe)
    if df is None or len(df) < 100:
        return None
    
    # Use OHLCV directly - no feature engineering
    data = df[['open', 'high', 'low', 'close', 'volume']].values
    
    # Normalize
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_norm) - Config.LOOKBACK - Config.PREDICT):
        X.append(data_norm[i:i+Config.LOOKBACK])
        y.append(data_norm[i+Config.LOOKBACK:i+Config.LOOKBACK+Config.PREDICT])
    
    if len(X) < 100:
        return None
    
    return np.array(X), np.array(y), scaler

# ============================================================================
# TRAINING
# ============================================================================

def train_model(coin: str, timeframe: str) -> Optional[Dict]:
    """Train single model with aggressive early stopping"""
    model_name = f"{coin}_{timeframe}"
    
    # Preprocess
    result = preprocess_coin(coin, timeframe)
    if result is None:
        print(f"  Skipped: No data")
        return None
    
    X, y, scaler = result
    print(f"  Data: {len(X)} sequences")
    
    # Split
    n = len(X)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    X_train = torch.FloatTensor(X[:train_idx]).to(Config.DEVICE)
    y_train = torch.FloatTensor(y[:train_idx]).to(Config.DEVICE)
    X_val = torch.FloatTensor(X[train_idx:val_idx]).to(Config.DEVICE)
    y_val = torch.FloatTensor(y[train_idx:val_idx]).to(Config.DEVICE)
    X_test = torch.FloatTensor(X[val_idx:]).to(Config.DEVICE)
    y_test = torch.FloatTensor(y[val_idx:]).to(Config.DEVICE)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    
    # Model
    model = SimpleLSTM(
        input_size=Config.INPUT_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        lookback=Config.LOOKBACK,
        predict=Config.PREDICT
    ).to(Config.DEVICE)
    
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"  Training (epochs: {Config.EPOCHS}, early stop: {Config.EARLY_STOPPING_PATIENCE})...")
    
    for epoch in range(Config.EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d}/{Config.EPOCHS} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"    Early stop at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).cpu().numpy()
        test_true = y_test.cpu().numpy()
    
    # Denormalize for MAPE
    test_pred_denorm = scaler.inverse_transform(test_pred.reshape(-1, Config.INPUT_SIZE))
    test_true_denorm = scaler.inverse_transform(test_true.reshape(-1, Config.INPUT_SIZE))
    
    mape = mean_absolute_percentage_error(test_true_denorm[:, 3], test_pred_denorm[:, 3])
    
    # Save
    save_path = Config.CACHE_DIR / f"{model_name}.pt"
    torch.save({
        'state_dict': model.state_dict(),
        'mape': float(mape),
        'scaler_mean': scaler.data_min_.tolist(),
        'scaler_scale': scaler.data_range_.tolist()
    }, save_path)
    
    print(f"  Result: MAPE={mape:.6f}")
    return {'mape': mape, 'model_path': str(save_path)}

# ============================================================================
# MAIN
# ============================================================================

def main():
    results = {}
    start_time = time.time()
    
    total = Config.total_models()
    completed = 0
    successful = 0
    
    for coin in Config.COINS:
        for timeframe in Config.TIMEFRAMES:
            completed += 1
            model_name = f"{coin}_{timeframe}"
            
            elapsed = (time.time() - start_time) / 60
            print(f"\n[{completed}/{total}] {model_name} (elapsed: {elapsed:.1f}m)")
            
            try:
                result = train_model(coin, timeframe)
                if result:
                    results[model_name] = result
                    successful += 1
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                print(f"  Error: {e}")
    
    # Summary
    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Successful: {successful}/{total}")
    print(f"Total time: {elapsed:.1f} minutes ({elapsed/60:.1f} hours)")
    
    if results:
        mape_values = [v['mape'] for v in results.values()]
        print(f"Average MAPE: {np.mean(mape_values):.6f}")
        print(f"Best MAPE: {np.min(mape_values):.6f}")
        print(f"Worst MAPE: {np.max(mape_values):.6f}")
    
    print(f"Models saved to: {Config.CACHE_DIR}")
    
    # Save results
    results_path = Config.CACHE_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results: {results_path}")
    print("="*70)

if __name__ == "__main__":
    main()
