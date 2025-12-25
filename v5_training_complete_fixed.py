#!/usr/bin/env python3
"""
CPB v5: Complete Training Pipeline (All-in-One) - FIXED

Fix for: 'numpy.ndarray' object has no attribute 'index'

Version: 5.0.1
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
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/content/training.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    COINS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
        'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
        'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT',
        'PEPEUSDT', 'INJUSDT', 'SHIBUSDT', 'ETCUSDT', 'LUNAUSDT'
    ]
    TIMEFRAMES = ['15m', '1h']
    LOOKBACK_STEPS = 30
    PREDICT_STEPS = 10
    KBARS_TO_DOWNLOAD = 8000
    
    INPUT_SIZE = 40
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    NUM_HEADS = 8
    
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 15
    GRADIENT_CLIP = 1.0
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CACHE_DIR = Path('/content/all_models')
    MODEL_V5_DIR = CACHE_DIR / 'model_v5'
    
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

Config.MODEL_V5_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FEATURE ENGINEERING - FIXED
# ============================================================================

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """計算 RSI，正確處理 pandas Series"""
        deltas = prices.diff()
        seed = deltas.iloc[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 1
        rsi = pd.Series(100. - 100. / (1. + rs), index=prices.index[:period])
        
        for i in range(period, len(prices)):
            delta = deltas.iloc[i]
            upval = delta if delta > 0 else 0
            downval = -delta if delta < 0 else 0
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 1
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """計算所有技術指標"""
        df = df.copy()
        
        # Price features
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        df['volatility_5'] = df['log_return'].rolling(5).std()
        df['volatility_10'] = df['log_return'].rolling(10).std()
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['volatility_30'] = df['log_return'].rolling(30).std()
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']
        df['volatility_ratio'] = df['volatility_ratio'].fillna(1)
        
        # Amplitude features
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['amplitude_5'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close']
        df['amplitude_10'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close']
        df['amplitude_20'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
        
        # Returns
        df['returns'] = df['log_return']
        df['returns_pct'] = df['close'].pct_change()
        df['abs_returns'] = np.abs(df['returns'])
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Momentum - FIXED: Use the corrected RSI function
        df['rsi_14'] = TechnicalIndicators.calculate_rsi(df['close'], 14)
        df['rsi_21'] = TechnicalIndicators.calculate_rsi(df['close'], 21)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['roc_12'] = df['close'].pct_change(12)
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_middle'] = sma20
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma20 + 1e-8)
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # ATR
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = tr.rolling(14).mean()
        
        # Volume
        if 'volume' in df.columns:
            volume_sma = df['volume'].rolling(20).mean()
            df['volume_sma'] = volume_sma
            df['volume_ratio'] = df['volume'] / (volume_sma + 1e-8)
        else:
            df['volume_sma'] = 1
            df['volume_ratio'] = 1
        
        # Direction
        df['price_direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['hl_balance'] = (df['high'] - df['close']) / (df['close'] - df['low'] + 1e-8)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df.dropna()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class Seq2SeqLSTMV5(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_layers=2, 
                 predict_steps=10, dropout=0.3, num_heads=8):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.predict_steps = predict_steps
        
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x):
        encoder_output, (h_n, c_n) = self.encoder(x)
        context, _ = self.attention(encoder_output, encoder_output, encoder_output)
        h_decoder = h_n[-1:].unsqueeze(0)
        c_decoder = c_n[-1:].unsqueeze(0)
        decoder_input = context[:, -1:, :].repeat(1, self.predict_steps, 1)
        decoder_output, _ = self.decoder(decoder_input, (h_decoder, c_decoder))
        output = self.relu(self.fc1(decoder_output))
        output = self.dropout_layer(output)
        output = self.relu(self.fc2(output))
        output = self.dropout_layer(output)
        output = self.fc_out(output)
        return output

# ============================================================================
# DATA PROCESSING
# ============================================================================

def download_binance_data(coin: str, timeframe: str, limit: int = 8000) -> pd.DataFrame:
    import requests
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': coin, 'interval': timeframe, 'limit': limit}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        return df
    except Exception as e:
        logger.error(f"Error downloading {coin} {timeframe}: {e}")
        return None

def create_sequences(features, prices, lookback=30, predict_steps=10):
    X, y = [], []
    for i in range(len(features) - lookback - predict_steps + 1):
        X.append(features[i:i+lookback])
        y.append(prices[i+lookback:i+lookback+predict_steps])
    return np.array(X), np.array(y)

def preprocess_coin_data(coin: str, timeframe: str):
    logger.info(f"Processing {coin} {timeframe}...")
    
    df = download_binance_data(coin, timeframe)
    if df is None or len(df) < 1000:
        return None
    
    df = TechnicalIndicators.calculate_all_features(df)
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    if len(feature_cols) == 0:
        logger.warning(f"No features calculated for {coin} {timeframe}")
        return None
    
    feature_scaler = MinMaxScaler()
    features_normalized = feature_scaler.fit_transform(df[feature_cols])
    
    prices = df['close'].values.reshape(-1, 1)
    price_scaler = MinMaxScaler()
    prices_normalized = price_scaler.fit_transform(prices).ravel()
    
    X, y = create_sequences(
        features_normalized,
        prices_normalized,
        lookback=Config.LOOKBACK_STEPS,
        predict_steps=Config.PREDICT_STEPS
    )
    
    if len(X) == 0:
        logger.warning(f"Not enough sequences for {coin} {timeframe}")
        return None
    
    return X, y, feature_scaler, price_scaler

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, device, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, 
                          weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(-1), y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions.squeeze(-1), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(train_loader)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return model

def evaluate_model(model, test_loader, price_scaler, device):
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).squeeze(-1).cpu().numpy()
            predictions.append(pred)
            targets.append(y_batch.numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    predictions_denorm = price_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    targets_denorm = price_scaler.inverse_transform(targets.reshape(-1, 1)).ravel()
    
    mape = mean_absolute_percentage_error(targets_denorm, predictions_denorm)
    rmse = np.sqrt(mean_squared_error(targets_denorm, predictions_denorm))
    mae = mean_absolute_error(targets_denorm, predictions_denorm)
    
    return {'mape': mape, 'rmse': rmse, 'mae': mae}

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("CPB v5: Complete Training Pipeline (FIXED)")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Total models: {len(Config.COINS) * len(Config.TIMEFRAMES)}")
    print("="*60)
    
    results_by_coin = {}
    total_models = len(Config.COINS) * len(Config.TIMEFRAMES)
    current = 0
    successful = 0
    
    for coin in Config.COINS:
        for timeframe in Config.TIMEFRAMES:
            current += 1
            model_name = f"{coin}_{timeframe}"
            
            print(f"\n[{current}/{total_models}] {model_name}")
            
            try:
                result = preprocess_coin_data(coin, timeframe)
                if result is None:
                    print(f"  Skipped: Data preprocessing failed")
                    continue
                
                X, y, feature_scaler, price_scaler = result
                
                n = len(X)
                train_idx = int(n * Config.TRAIN_RATIO)
                val_idx = train_idx + int(n * Config.VAL_RATIO)
                
                train_loader = DataLoader(
                    TensorDataset(torch.FloatTensor(X[:train_idx]), 
                                torch.FloatTensor(y[:train_idx])),
                    batch_size=Config.BATCH_SIZE, shuffle=True
                )
                val_loader = DataLoader(
                    TensorDataset(torch.FloatTensor(X[train_idx:val_idx]), 
                                torch.FloatTensor(y[train_idx:val_idx])),
                    batch_size=Config.BATCH_SIZE
                )
                test_loader = DataLoader(
                    TensorDataset(torch.FloatTensor(X[val_idx:]), 
                                torch.FloatTensor(y[val_idx:])),
                    batch_size=Config.BATCH_SIZE
                )
                
                print(f"  Training...")
                model = Seq2SeqLSTMV5(
                    input_size=Config.INPUT_SIZE,
                    hidden_size=Config.HIDDEN_SIZE,
                    num_layers=Config.NUM_LAYERS,
                    predict_steps=Config.PREDICT_STEPS,
                    dropout=Config.DROPOUT,
                    num_heads=Config.NUM_HEADS
                )
                model = train_model(model, train_loader, val_loader, Config.DEVICE, Config.EPOCHS)
                
                metrics = evaluate_model(model, test_loader, price_scaler, Config.DEVICE)
                
                model_path = Config.MODEL_V5_DIR / f"{model_name}.pt"
                torch.save({
                    'model_state': model.state_dict(),
                    'config': {
                        'input_size': Config.INPUT_SIZE,
                        'hidden_size': Config.HIDDEN_SIZE,
                        'num_layers': Config.NUM_LAYERS,
                        'predict_steps': Config.PREDICT_STEPS,
                        'dropout': Config.DROPOUT,
                        'num_heads': Config.NUM_HEADS
                    },
                    'metrics': metrics
                }, model_path)
                
                results_by_coin[model_name] = metrics
                successful += 1
                print(f"  Success: MAPE={metrics['mape']:.6f}")
                
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    summary_path = Config.MODEL_V5_DIR / 'training_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results_by_coin, f, indent=2)
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETED")
    print(f"Successful: {successful}/{total_models}")
    if successful > 0:
        mape_values = [v['mape'] for v in results_by_coin.values()]
        print(f"Average MAPE: {sum(mape_values)/len(mape_values):.6f}")
        print(f"Best MAPE: {min(mape_values):.6f}")
    print(f"Models saved to: {Config.MODEL_V5_DIR}")
    print(f"Results: {summary_path}")
    print("="*60)

if __name__ == "__main__":
    main()
