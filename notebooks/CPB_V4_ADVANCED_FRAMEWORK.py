#!/usr/bin/env python3
# ============================================================================
# CPB V4: Advanced CNN-LSTM Framework
# ============================================================================
# Version: 4.0.0 (2025-12-24)
# Features:
#   - CNN-LSTM Hybrid Architecture
#   - 20+ Technical Indicators
#   - Entry Position Calculation
#   - Position Sizing Algorithm
#   - Stop Loss & Take Profit Targets
# ============================================================================

print('='*80)
print('CPB V4: Advanced CNN-LSTM Trading Model')
print('='*80)

import sys
import os
import time
import warnings
from datetime import datetime, timedelta
import math

warnings.filterwarnings('ignore')

print('\n[STEP 0] Loading libraries...')

import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print('OK - All libraries loaded')

# ============================================================================
# CONFIGURATION
# ============================================================================

print('\n[STEP 1] Configuring parameters...')

CONFIG = {
    'coins': [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT',
        'ADAUSDT', 'DOGEUSDT', 'LINKUSDT', 'XRPUSDT', 'LTCUSDT',
        'MATICUSDT', 'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'ARBUSDT',
        'OPUSDT', 'STXUSDT', 'INJUSDT', 'LUNCUSDT', 'LUNAUSDT'
    ],
    'timeframes': ['1h'],
    'epochs': 80,
    'batch_size': 32,
    'learning_rate': 5e-4,
    'lookback': 20,  # 20 hours of history
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'warmup_epochs': 5,
    'early_stop_patience': 20,
    'focal_alpha': 0.5,
    'focal_gamma': 3.0,
    'atr_period': 14,
    'cnn_kernel': 3,
    'cnn_filters': 32,
}

if torch.cuda.is_available():
    device = 'cuda'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'
    print('WARNING: Using CPU, training will be slow')

print(f'\nConfiguration:')
print(f'  Coins: {len(CONFIG["coins"])}')
print(f'  Lookback: {CONFIG["lookback"]} hours')
print(f'  Model: CNN-LSTM Hybrid')
print(f'  Features: 20+ Technical Indicators')

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureEngineer:
    """Enhanced feature engineering with 20+ technical indicators"""
    
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum indicators
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        df['momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        df['acceleration'] = df['momentum_5'] - df['momentum_5'].shift(1)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # Trend indicators
        df['sma_ratio'] = df['sma_5'] / (df['sma_20'] + 1e-10)
        df['trend_strength'] = (df['sma_20'] - df['sma_50']) / (df['sma_50'] + 1e-10)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50).clip(0, 100) / 100  # Normalize to 0-1
        
        # ATR (Average True Range) - key for position sizing
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_normalized'] = df['atr_14'] / (df['close'] + 1e-10) * 100
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # Price position in daily range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['price_position'] = df['price_position'].clip(0, 1)
        
        # Volatility measures
        df['volatility'] = df['returns'].rolling(20).std() * 100
        df['volatility'] = df['volatility'].clip(0, 10)
        df['volatility_ratio'] = df['volatility'] / (df['volatility'].rolling(50).mean() + 1e-10)
        df['volatility_ratio'] = df['volatility_ratio'].clip(0, 5)
        
        # Volume indicators
        df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        df['volume_ma_ratio'] = df['volume_ma_ratio'].clip(0.1, 10)
        df['volume_trend'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).mean() + 1e-10)
        
        # Trend classification
        df['trend'] = (df['close'] > df['sma_20']).astype(int)
        
        # Fill NaN values
        self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Clip extreme values
        for col in self.df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].clip(-100, 100)
        
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'tr']
        return [col for col in self.df.columns if col not in exclude]

# ============================================================================
# ENTRY POSITION CALCULATOR
# ============================================================================

class EntryPositionCalculator:
    """Calculate optimal entry, stop loss, and take profit levels"""
    
    @staticmethod
    def calculate(
        current_price,
        atr,
        volatility,
        prediction_confidence=0.5,
        risk_reward_ratio=1.33
    ):
        """Calculate trading parameters"""
        
        # Entry range based on ATR and volatility
        entry_range = atr * 0.5  # 50% of ATR for entry flexibility
        entry_low = max(0, current_price - entry_range)
        entry_high = current_price + entry_range
        
        # Stop loss (conservative: 1.5x ATR below entry)
        stop_loss = max(0, current_price - atr * 1.5)
        
        # Take profit (risk/reward ratio)
        stop_distance = current_price - stop_loss
        take_profit = current_price + (stop_distance * risk_reward_ratio)
        
        # Position size adjustment based on volatility
        # High volatility → smaller position
        # Low volatility → larger position
        vol_multiplier = max(0.5, min(2.0, 1.0 / (volatility / 100.0 + 0.1)))
        
        # Confidence-based position sizing
        confidence_multiplier = max(0.5, min(2.0, prediction_confidence * 2))
        
        position_size_multiplier = vol_multiplier * confidence_multiplier
        
        return {
            'entry_low': entry_low,
            'entry_high': entry_high,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_multiplier': position_size_multiplier,
            'risk_per_trade': stop_distance
        }

# ============================================================================
# CNN-LSTM HYBRID MODEL
# ============================================================================

class CNNLSTMHybridModel(nn.Module):
    """CNN-LSTM hybrid architecture for time series prediction"""
    
    def __init__(
        self,
        input_size=20,
        cnn_filters=32,
        cnn_kernel=3,
        lstm_hidden=64,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        # 1D CNN for local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=cnn_filters,
                kernel_size=cnn_kernel,
                padding=1
            ),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal dependency
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention-like layer
        self.fc_attention = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 4),
            nn.ReLU(),
            nn.Linear(lstm_hidden // 4, 1),
            nn.Sigmoid()
        )
        
        # Dense layers for classification
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.bn_hidden = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        
        # Output layer for binary classification
        self.fc_class = nn.Linear(64, 2)
        
        # Regression outputs (optional)
        self.fc_regression = nn.Linear(64, 4)  # price_change, volatility, entry_range, sl_distance
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # CNN expects (batch, features, seq_len)
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn(x_cnn)
        
        # Back to (batch, seq_len, filters) for LSTM
        x_cnn = x_cnn.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_cnn)
        
        # Simple attention mechanism
        attn_weights = self.fc_attention(lstm_out)  # (batch, seq_len, 1)
        attn_out = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, lstm_hidden)
        
        # Dense layers
        out = self.fc1(attn_out)
        out = self.bn_hidden(out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        
        # Classification
        class_logits = self.fc_class(out)
        
        # Regression (optional)
        regression_out = self.fc_regression(out)
        
        return class_logits, regression_out
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# DATA COLLECTION (from V2)
# ============================================================================

class BinanceDataCollector:
    BASE_URL = "https://api.binance.us/api/v3"
    MAX_CANDLES = 1000
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_klines(self, symbol, interval="1h", limit=3000):
        all_klines = []
        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(days=150)).timestamp() * 1000)
        current_start = start_time
        retry_count = 0
        
        while current_start < end_time and len(all_klines) < limit:
            try:
                params = {
                    "symbol": symbol, "interval": interval,
                    "startTime": current_start,
                    "limit": min(self.MAX_CANDLES, limit - len(all_klines))
                }
                response = self.session.get(f"{self.BASE_URL}/klines", params=params, timeout=15)
                
                if response.status_code == 400:
                    return pd.DataFrame()
                
                response.raise_for_status()
                klines = response.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                current_start = int(klines[-1][0]) + 1
                retry_count = 0
                time.sleep(0.3)
            except Exception as e:
                retry_count += 1
                if retry_count >= 3:
                    break
                time.sleep(2 ** retry_count)
        
        if not all_klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].drop_duplicates().sort_values('timestamp').reset_index(drop=True)
        return df
    
    @staticmethod
    def validate(df):
        if len(df) < 200:
            return False
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            return False
        if (df[['open', 'high', 'low', 'close', 'volume']] <= 0).any().any():
            return False
        return True

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class AdvancedPreprocessor:
    def __init__(self, df, lookback=20):
        self.df = df.copy()
        self.lookback = lookback
        self.scaler = MinMaxScaler((0, 1))
        self.feature_scaler = StandardScaler()
    
    def prepare(self, feature_cols):
        self.df = self.df.dropna()
        feature_data = self.df[feature_cols].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        self.df = self.df.loc[feature_data.index]
        
        # Scale features
        feature_data = self.scaler.fit_transform(feature_data)
        self.features = feature_data
        self.feature_cols = feature_cols
        
        # Store ATR, close, and other needed columns for position calculation
        self.atr = self.df['atr_normalized'].values
        self.volatility = self.df['volatility'].values
        self.close = self.df['close'].values
        
        return feature_data, feature_cols
    
    def create_sequences(self):
        X, y, metadata = [], [], []
        data = self.features
        close_prices = self.close
        atr = self.atr
        vol = self.volatility
        
        for i in range(self.lookback, len(data) - 1):
            X.append(data[i - self.lookback:i])
            
            # Binary label
            price_change = (close_prices[i + 1] - close_prices[i]) / close_prices[i]
            label = 1 if price_change > 0 else 0
            y.append(label)
            
            # Metadata for position calculation
            metadata.append({
                'price': close_prices[i],
                'atr': atr[i],
                'volatility': vol[i],
                'price_change_pct': price_change * 100
            })
        
        return np.array(X), np.array(y), metadata
    
    def split_data(self, X, y, train_ratio=0.7):
        n = len(X)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + 0.15))
        
        return {
            'X_train': X[:train_idx], 'y_train': y[:train_idx],
            'X_val': X[train_idx:val_idx], 'y_val': y[train_idx:val_idx],
            'X_test': X[val_idx:], 'y_test': y[val_idx:]
        }

# ============================================================================
# IMPROVED FOCAL LOSS
# ============================================================================

class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce)
        focal_loss = self.alpha * ((1 - p) ** self.gamma) * ce
        return focal_loss.mean()

# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

# ============================================================================
# TRAINER
# ============================================================================

class AdvancedTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, X_train, y_train, X_val, y_val, config):
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        # Class weighting
        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor([1.0 / count if count > 0 else 1.0 for count in class_counts])
        class_weights = class_weights / class_weights.sum() * 2
        class_weights = class_weights.to(self.device)
        
        print(f'    Class weights: {class_weights.cpu().numpy()}')
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        lr_scheduler = WarmupCosineScheduler(
            optimizer,
            config['warmup_epochs'],
            config['epochs'],
            config['learning_rate']
        )
        
        criterion = ImprovedFocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(config['epochs']):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits, _ = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config['gradient_clip']
                )
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            self.model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    logits, _ = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
                    
                    preds = logits.argmax(dim=1)
                    val_acc += (preds == y_batch).sum().item()
            
            val_loss /= len(val_loader)
            val_acc /= len(y_val)
            
            current_lr = lr_scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1:3d}/{config["epochs"]}: Train={train_loss:.6f}, Val={val_loss:.6f}, Acc={val_acc:.4f}')
            
            if val_loss < best_val_loss - 0.0001:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= config['early_stop_patience']:
                    print(f'    Early Stop at epoch {epoch+1}')
                    if best_weights:
                        self.model.load_state_dict(best_weights)
                    break
        
        return {'best_val_loss': float(best_val_loss), 'epochs': epoch+1}

print('OK - All classes defined')

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print('\n[STEP 2] Downloading data...')

all_data = {}
collector = BinanceDataCollector()

for coin in CONFIG['coins']:
    try:
        print(f'  {coin}...', end=' ', flush=True)
        df = collector.get_klines(coin, '1h', limit=3000)
        if BinanceDataCollector.validate(df):
            all_data[coin] = df
            print(f'OK {len(df)} candles')
        else:
            print(f'X validation failed')
    except Exception as e:
        print(f'X {str(e)[:40]}')

print(f'\nOK - Downloaded {len(all_data)}/{len(CONFIG["coins"])} coins')

if len(all_data) == 0:
    print('\nERROR: No data downloaded')
    exit(1)

print('\n[STEP 3] Training models...')

trained_models = {}
results = []

for coin in all_data:
    print(f'\n  {coin}')
    print('  ' + '-'*60)
    
    try:
        df = all_data[coin]
        
        # Feature engineering
        fe = AdvancedFeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        print(f'    Features: {len(feature_cols)}')
        
        # Preprocessing
        prep = AdvancedPreprocessor(df_features, lookback=CONFIG['lookback'])
        features, _ = prep.prepare(feature_cols)
        X, y, metadata = prep.create_sequences()
        data = prep.split_data(X, y)
        
        if len(X) < 200:
            print(f'    X Sequence too short ({len(X)} < 200)')
            continue
        
        # Model
        model = CNNLSTMHybridModel(
            input_size=features.shape[-1],
            cnn_filters=CONFIG['cnn_filters'],
            cnn_kernel=CONFIG['cnn_kernel'],
            lstm_hidden=64,
            num_layers=2,
            dropout=0.3
        )
        print(f'    Params: {model.count_params():,}')
        
        # Training
        trainer = AdvancedTrainer(model, device=device)
        history = trainer.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            CONFIG
        )
        
        # Evaluation
        model.eval()
        X_test_t = torch.FloatTensor(data['X_test']).to(device)
        y_test_t = torch.LongTensor(data['y_test']).to(device)
        
        with torch.no_grad():
            logits, _ = model(X_test_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = logits.argmax(dim=1).cpu().numpy()
        
        y_test = data['y_test']
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, probs[:, 1])
        
        trained_models[coin] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': probs,
            'y_test': y_test,
            'metadata': metadata[-len(y_test):]
        }
        
        results.append({
            'coin': coin,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'epochs': history['epochs']
        })
        
        print(f'    Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}')
        print(f'    OK - Model trained')
        
    except Exception as e:
        print(f'    X Error: {str(e)[:60]}')

print(f'\nOK - Training complete: {len(trained_models)} models')

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print('\n' + '='*80)
print('[STEP 4] Results Summary')
print('='*80)

if results:
    print('\n{:12s} {:>10s} {:>10s} {:>8s}'.format('Coin', 'Accuracy', 'F1-Score', 'AUC'))
    print('-'*50)
    
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print('{:12s} {:>10.4f} {:>10.4f} {:>8.4f}'.format(
            r['coin'], r['accuracy'], r['f1'], r['auc']))
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_auc = np.mean([r['auc'] for r in results])
    
    print('-'*50)
    print(f'Average Accuracy: {avg_acc:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f}')
    print(f'Average AUC: {avg_auc:.4f}')
    print(f'\nSuccessfully trained: {len(trained_models)}/{len(all_data)} coins')

print('\n' + '='*80)
print('OK - V4 Training Pipeline Complete!')
print('Next: Feature extraction and position sizing analysis')
print('='*80)
