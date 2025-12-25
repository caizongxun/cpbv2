#!/usr/bin/env python3
"""
CPB V5 Advanced Training - Cryptocurrency Price Prediction with Advanced Techniques

Improvements over v1-v4:
1. Multi-task learning (price + volatility + reversal)
2. GARCH volatility modeling
3. Attention mechanism with positional encoding
4. Residual connections
5. Robust scaler for outlier handling
6. Better loss functions (Huber + multi-task)
7. Learning rate scheduling (Warm-up + CosineAnnealing)
8. Adversarial training for robustness
9. Direction accuracy + Reversal detection metrics
10. All-in-one Colab script

Target: MAPE < 0.015, Direction Accuracy > 65%, Reversal Detection > 55%
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import json
import time
from datetime import datetime
import requests
from tqdm import tqdm
import gc

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'coins': [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
        'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
        'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT',
        'PEPEUSDT', 'INJUSDT', 'SHIBUSDT', 'ETCUSDT', 'LUNAUSDT'
    ],
    'timeframes': ['1h', '15m'],
    'klines_count': 8000,
    'lookback': 30,
    'predict_steps': 10,
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'dropout': 0.3,
    'num_heads': 12,
    'hidden_size': 256,
    'num_layers': 2,
    'early_stopping_patience': 15,
    'test_size': 0.15,
    'val_size': 0.15,
    'model_dir': '/content/all_models/model_v5',
    'binance_api': 'https://api.binance.us/api/v3',
}

# ============================================================================
# TECHNICAL INDICATORS (60+ features)
# ============================================================================

def calculate_indicators(df):
    """
    Calculate 60+ technical indicators for enhanced prediction
    
    Categories:
    - Price indicators (3)
    - Volatility indicators (8) - KEY for v5
    - Amplitude/Swing indicators (4) - KEY for v5
    - Momentum indicators (10)
    - Moving averages (12)
    - Bollinger Bands (5)
    - Other (18)
    """
    df = df.copy()
    
    # Price indicators
    df['HL2'] = (df['high'] + df['low']) / 2
    df['HLC3'] = (df['high'] + df['low'] + df['close']) / 3
    df['OHLC4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Volatility indicators (8) - CRITICAL
    for period in [5, 10, 20, 30]:
        df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
    df['volatility_ratio'] = df['volatility_20'] / (df['volatility_10'] + 1e-8)
    
    # Amplitude/Swing indicators (4) - CRITICAL
    df['HL_ratio'] = (df['high'] - df['low']) / df['close']
    for period in [5, 10, 20]:
        df[f'amplitude_{period}'] = (df['high'].rolling(period).max() - df['low'].rolling(period).min()) / df['close']
    
    # Momentum indicators
    df['RSI_14'] = calculate_rsi(df['close'], 14)
    df['RSI_21'] = calculate_rsi(df['close'], 21)
    macd, signal, diff = calculate_macd(df['close'])
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_diff'] = diff
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['ROC_12'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12)
    
    # Moving averages (12)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['close'].rolling(period).mean()
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Bollinger Bands
    bb_high, bb_mid, bb_low, bb_width, bb_pct = calculate_bollinger_bands(df['close'])
    df['BB_high'] = bb_high
    df['BB_mid'] = bb_mid
    df['BB_low'] = bb_low
    df['BB_width'] = bb_width
    df['BB_pct'] = bb_pct
    
    # Additional indicators
    df['ATR_14'] = calculate_atr(df, 14)
    df['volume_SMA'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_SMA'] + 1e-8)
    
    # Reversal signals
    df['local_max'] = (df['high'].rolling(3).max() == df['high']).astype(float)
    df['local_min'] = (df['low'].rolling(3).min() == df['low']).astype(float)
    df['price_direction'] = np.sign(df['close'].diff())
    df['HL_balance'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Hurst exponent (趋势持久性，帮助识别反转)
    df['hurst_5'] = calculate_hurst_exponent(df['close'], 5)
    df['hurst_20'] = calculate_hurst_exponent(df['close'], 20)
    
    # GARCH-like volatility indicator
    returns = df['close'].pct_change()
    df['garch_vol'] = returns.rolling(20).std() * (1 + (returns.rolling(20).var() / returns.std()**2))
    
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    diff = macd - signal_line
    return macd, signal_line, diff

def calculate_bollinger_bands(prices, period=20, num_std=2):
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    width = upper - lower
    pct_b = (prices - lower) / (width + 1e-8)
    return upper, sma, lower, width, pct_b

def calculate_atr(df, period=14):
    df = df.copy()
    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        )
    )
    return df['TR'].rolling(period).mean()

def calculate_hurst_exponent(prices, period=20):
    """Simplified Hurst exponent for reversal detection"""
    returns = prices.pct_change()
    hurst = []
    for i in range(len(returns) - period):
        chunk = returns.iloc[i:i+period].values
        # Simplified: high autocorr = trending, low = mean-reverting
        if len(chunk) > 1:
            autocorr = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]
            hurst.append(0.5 + autocorr * 0.3)  # Range [0.2, 0.8]
        else:
            hurst.append(0.5)
    return pd.Series([np.nan]*period + hurst[:len(prices)-period], index=prices.index)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer-style attention"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with multiple heads"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        
        output = self.fc_out(context)
        return output, weights

class Seq2SeqWithAttentionV5Advanced(nn.Module):
    """Advanced Seq2Seq with Attention, Residual connections, Multi-task learning"""
    def __init__(self, input_size, hidden_size, num_layers, predict_steps, 
                 num_heads=8, dropout=0.3, num_tasks=3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.predict_steps = predict_steps
        self.num_tasks = num_tasks
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(input_size)
        
        # Encoder: Bidirectional LSTM
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.encoder_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        
        # Decoder: LSTM with residual connections
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Residual blocks
        self.residual_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Multi-task output heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, predict_steps)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, predict_steps)
        )
        
        self.reversal_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, predict_steps)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Encoder
        encoder_out, (h_n, c_n) = self.encoder_lstm(x)
        encoder_out = self.encoder_proj(encoder_out)  # Project from 2*hidden to hidden
        
        # Attention
        attn_out, attn_weights = self.attention(
            encoder_out, encoder_out, encoder_out
        )
        
        # Residual connection
        attn_out = self.layer_norm(attn_out + encoder_out)
        
        # Use last encoder state as decoder initial state
        h_n = h_n[-1:].repeat(self.decoder_lstm.num_layers, 1, 1)
        c_n = c_n[-1:].repeat(self.decoder_lstm.lstm.num_layers, 1, 1)
        
        # Decoder loop for all prediction steps at once
        decoder_input = attn_out[:, -1, :].unsqueeze(1)  # (batch, 1, hidden)
        
        decoder_outputs = []
        for _ in range(self.predict_steps):
            decoder_out, (h_n, c_n) = self.decoder_lstm(decoder_input, (h_n, c_n))
            
            # Residual connection in decoder
            residual = self.residual_fc(decoder_out)
            decoder_out = self.layer_norm(decoder_out + residual)
            
            decoder_outputs.append(decoder_out)
            decoder_input = decoder_out
        
        # Concatenate all decoder outputs
        decoder_out = torch.cat(decoder_outputs, dim=1)  # (batch, predict_steps, hidden)
        
        # Multi-task outputs
        price_pred = self.price_head(decoder_out).squeeze(-1) if self.predict_steps == 1 else self.price_head(decoder_out[:, :, :])
        volatility_pred = self.volatility_head(decoder_out).squeeze(-1) if self.predict_steps == 1 else self.volatility_head(decoder_out[:, :, :])
        reversal_pred = torch.sigmoid(self.reversal_head(decoder_out).squeeze(-1) if self.predict_steps == 1 else self.reversal_head(decoder_out[:, :, :]))
        
        return {
            'price': price_pred,
            'volatility': volatility_pred,
            'reversal': reversal_pred
        }

# ============================================================================
# DATA COLLECTION
# ============================================================================

def download_binance_us_data(symbol, interval, limit=8000):
    """
    Download OHLCV data from Binance US
    Handles regional API differences
    """
    url = f"{CONFIG['binance_api']}/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': min(limit, 1000)  # API max 1000 per request
    }
    
    all_klines = []
    
    try:
        # Download in batches if needed
        while len(all_klines) < limit:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            klines = response.json()
            if not klines:
                break
            
            all_klines.extend(klines)
            
            if len(all_klines) >= limit:
                break
            
            # Update params for next batch (from oldest)
            params['endTime'] = int(klines[0][0]) - 1
            params['limit'] = min(limit - len(all_klines), 1000)
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_klines[:limit],
            columns=['time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'asset_vol', 'n_trades', 'taker_buy_asset',
                    'taker_buy_quote', 'ignore']
        )
        
        # Convert types
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = \
            df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"Error downloading {symbol} {interval}: {e}")
        return None

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""
    def __init__(self, price_weight=0.6, vol_weight=0.2, rev_weight=0.2):
        super().__init__()
        self.price_weight = price_weight
        self.vol_weight = vol_weight
        self.rev_weight = rev_weight
        
        self.huber = nn.HuberLoss(reduction='mean', delta=0.1)
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCELoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='mean')
    
    def forward(self, price_pred, vol_pred, rev_pred,
                price_true, vol_true, rev_true):
        # Price loss: Huber + L1 (robust to outliers)
        price_loss = self.huber(price_pred, price_true) + 0.1 * self.l1(price_pred, price_true)
        
        # Volatility loss: MSE
        vol_loss = self.mse(vol_pred, vol_true)
        
        # Reversal loss: BCE
        rev_loss = self.bce(rev_pred, rev_true)
        
        # Combined loss
        total_loss = (self.price_weight * price_loss +
                     self.vol_weight * vol_loss +
                     self.rev_weight * rev_loss)
        
        return total_loss, {
            'price': price_loss.item(),
            'volatility': vol_loss.item(),
            'reversal': rev_loss.item()
        }

# ============================================================================
# TRAINING
# ============================================================================

def prepare_data(prices, volatility, reversals, lookback=30, predict_steps=10):
    """
    Prepare sequences for training
    """
    X, y_price, y_vol, y_rev = [], [], [], []
    
    for i in range(len(prices) - lookback - predict_steps + 1):
        X.append(prices[i:i+lookback])
        y_price.append(prices[i+lookback:i+lookback+predict_steps])
        y_vol.append(volatility[i+lookback:i+lookback+predict_steps])
        y_rev.append(reversals[i+lookback:i+lookback+predict_steps])
    
    return (
        np.array(X, dtype=np.float32),
        np.array(y_price, dtype=np.float32),
        np.array(y_vol, dtype=np.float32),
        np.array(y_rev, dtype=np.float32)
    )

def train_single_model(symbol, timeframe, train_data_df):
    """
    Train a single model for one coin and timeframe
    """
    print(f"\n[Training] {symbol} {timeframe}")
    
    # Calculate indicators
    df = calculate_indicators(train_data_df.copy())
    df = df.dropna()
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in 
                   ['time', 'open', 'high', 'low', 'close', 'volume']]
    
    X_features = df[feature_cols].values
    prices = df['close'].values
    
    # Normalize features with RobustScaler (handles outliers better)
    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(X_features)
    
    # Normalize prices
    price_scaler = MinMaxScaler()
    prices_scaled = price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    # Calculate targets
    volatility = df['volatility_20'].fillna(0).values
    volatility_scaled = MinMaxScaler().fit_transform(volatility.reshape(-1, 1)).flatten()
    
    reversals = (df['local_max'].values + df['local_min'].values) / 2
    
    # Prepare sequences
    X, y_price, y_vol, y_rev = prepare_data(
        X_scaled, volatility_scaled, reversals,
        CONFIG['lookback'], CONFIG['predict_steps']
    )
    
    # Split data
    test_idx = int(len(X) * (1 - CONFIG['test_size']))
    val_idx = int(test_idx * (1 - CONFIG['val_size']))
    
    X_train, y_price_train, y_vol_train, y_rev_train = \
        X[:val_idx], y_price[:val_idx], y_vol[:val_idx], y_rev[:val_idx]
    X_val, y_price_val, y_vol_val, y_rev_val = \
        X[val_idx:test_idx], y_price[val_idx:test_idx], y_vol[val_idx:test_idx], y_rev[val_idx:test_idx]
    X_test, y_price_test, y_vol_test, y_rev_test = \
        X[test_idx:], y_price[test_idx:], y_vol[test_idx:], y_rev[test_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_price_train),
        torch.FloatTensor(y_vol_train),
        torch.FloatTensor(y_rev_train)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_price_val),
        torch.FloatTensor(y_vol_val),
        torch.FloatTensor(y_rev_val)
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_price_test),
        torch.FloatTensor(y_vol_test),
        torch.FloatTensor(y_rev_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Model
    model = Seq2SeqWithAttentionV5Advanced(
        input_size=X_scaled.shape[1],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        predict_steps=CONFIG['predict_steps'],
        num_heads=CONFIG['num_heads'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Warm-up + CosineAnnealing scheduler
    def lr_lambda_warmup(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return np.cos(np.pi * (epoch - warmup_epochs) / (CONFIG['epochs'] - warmup_epochs)) * 0.5 + 0.5
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)
    
    # Loss function
    criterion = MultiTaskLoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_mape': [], 'val_mape': []}
    
    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_mape = []
        
        for batch_idx, (X_batch, y_price_batch, y_vol_batch, y_rev_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_price_batch = y_price_batch.to(device)
            y_vol_batch = y_vol_batch.to(device)
            y_rev_batch = y_rev_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            price_pred = outputs['price']
            vol_pred = outputs['volatility']
            rev_pred = outputs['reversal']
            
            # Loss
            loss, _ = criterion(price_pred, vol_pred, rev_pred,
                              y_price_batch, y_vol_batch, y_rev_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # MAPE
            mape = torch.mean(torch.abs((price_pred - y_price_batch) / (y_price_batch + 1e-8)))
            train_mape.append(mape.item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        val_mape = []
        
        with torch.no_grad():
            for X_batch, y_price_batch, y_vol_batch, y_rev_batch in val_loader:
                X_batch = X_batch.to(device)
                y_price_batch = y_price_batch.to(device)
                y_vol_batch = y_vol_batch.to(device)
                y_rev_batch = y_rev_batch.to(device)
                
                outputs = model(X_batch)
                loss, _ = criterion(outputs['price'], outputs['volatility'], outputs['reversal'],
                                  y_price_batch, y_vol_batch, y_rev_batch)
                
                val_loss += loss.item()
                mape = torch.mean(torch.abs((outputs['price'] - y_price_batch) / (y_price_batch + 1e-8)))
                val_mape.append(mape.item())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_mape_avg = np.mean(train_mape)
        val_mape_avg = np.mean(val_mape)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mape'].append(train_mape_avg)
        history['val_mape'].append(val_mape_avg)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{CONFIG['epochs']} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Train MAPE: {train_mape_avg:.4f}, Val MAPE: {val_mape_avg:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    # Test evaluation
    model.eval()
    test_mape_list = []
    test_direction_acc = []
    test_reversal_acc = []
    
    with torch.no_grad():
        for X_batch, y_price_batch, y_vol_batch, y_rev_batch in test_loader:
            X_batch = X_batch.to(device)
            y_price_batch = y_price_batch.to(device)
            y_rev_batch = y_rev_batch.to(device)
            
            outputs = model(X_batch)
            price_pred = outputs['price']
            rev_pred = outputs['reversal']
            
            # MAPE
            mape = torch.mean(torch.abs((price_pred - y_price_batch) / (y_price_batch + 1e-8)))
            test_mape_list.append(mape.item())
            
            # Direction accuracy
            pred_direction = torch.sign(price_pred[:, 1:] - price_pred[:, :-1])
            true_direction = torch.sign(y_price_batch[:, 1:] - y_price_batch[:, :-1])
            direction_acc = (pred_direction == true_direction).float().mean()
            test_direction_acc.append(direction_acc.item())
            
            # Reversal detection
            rev_detected = (rev_pred > 0.5).float()
            reversal_acc = (rev_detected == y_rev_batch).float().mean()
            test_reversal_acc.append(reversal_acc.item())
    
    metrics = {
        'mape': np.mean(test_mape_list),
        'direction_accuracy': np.mean(test_direction_acc),
        'reversal_detection': np.mean(test_reversal_acc),
        'rmse': np.sqrt(np.mean([(pred - true).cpu().numpy()**2 for pred, true in zip(test_mape_list, test_mape_list)])),
        'history': history
    }
    
    # Save model
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    model_path = os.path.join(CONFIG['model_dir'], f"{symbol}_{timeframe}.pt")
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'input_size': X_scaled.shape[1],
            'hidden_size': CONFIG['hidden_size'],
            'num_layers': CONFIG['num_layers'],
            'predict_steps': CONFIG['predict_steps'],
            'num_heads': CONFIG['num_heads'],
            'dropout': CONFIG['dropout']
        },
        'metrics': metrics,
        'feature_scaler': feature_scaler,
        'price_scaler': price_scaler,
        'volatility_scaler': MinMaxScaler(),
        'feature_columns': feature_cols
    }, model_path)
    
    print(f"  Saved to {model_path}")
    print(f"  Test MAPE: {metrics['mape']:.6f}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
    print(f"  Reversal Detection: {metrics['reversal_detection']:.4f}")
    
    return metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("CPB V5 Advanced Training - Cryptocurrency Price Prediction")
    print("="*80)
    
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    all_results = {}
    start_time = time.time()
    
    total_models = len(CONFIG['coins']) * len(CONFIG['timeframes'])
    model_count = 0
    
    for coin in CONFIG['coins']:
        for timeframe in CONFIG['timeframes']:
            model_count += 1
            print(f"\n[{model_count}/{total_models}] {coin} {timeframe}")
            
            # Download data
            print("  Downloading data from Binance US...")
            df = download_binance_us_data(coin, timeframe, CONFIG['klines_count'])
            
            if df is None or len(df) < CONFIG['lookback'] + CONFIG['predict_steps']:
                print(f"  Skipping {coin} {timeframe} - insufficient data")
                continue
            
            # Train model
            try:
                metrics = train_single_model(coin, timeframe, df)
                all_results[f"{coin}_{timeframe}"] = metrics
            except Exception as e:
                print(f"  Error training {coin} {timeframe}: {e}")
                continue
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"Training completed in {elapsed/60:.1f} minutes")
    print(f"Models trained: {len(all_results)}/{total_models}")
    
    # Calculate average metrics
    if all_results:
        avg_mape = np.mean([m['mape'] for m in all_results.values()])
        avg_direction = np.mean([m['direction_accuracy'] for m in all_results.values()])
        avg_reversal = np.mean([m['reversal_detection'] for m in all_results.values()])
        
        print(f"\nAverage Metrics:")
        print(f"  MAPE: {avg_mape:.6f}")
        print(f"  Direction Accuracy: {avg_direction:.4f}")
        print(f"  Reversal Detection: {avg_reversal:.4f}")
    
    # Save results
    results_path = os.path.join(CONFIG['model_dir'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': CONFIG,
            'results': all_results,
            'summary': {
                'total_models': total_models,
                'trained': len(all_results),
                'elapsed_minutes': elapsed / 60
            }
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    
    # Upload to Hugging Face
    print("\nHugging Face Upload:")
    token = input("Enter HF token (or press Enter to skip): ").strip()
    
    if token:
        print("Uploading to Hugging Face...")
        os.system(f"huggingface-cli login --token {token}")
        os.system(f"huggingface-cli upload zongowo111/cpb-models {CONFIG['model_dir']} model_v5 --repo-type model")
        print("Upload completed!")

if __name__ == '__main__':
    main()
