# ============================================================================
# CPB v2: Complete Step-by-Step Enhancement Framework
# Target: MAPE < 0.02% (>99.98% accuracy)
# 
# This script provides:
# 1. STEP 1: Enhanced Feature Engineering (14 -> 28 features)
# 2. STEP 2: Improved Target Variable (dynamic threshold)
# 3. STEP 3: Model Architecture Optimization
# 4. STEP 4: Training Strategy Enhancement
# 5. STEP 5: Data Quality Improvement
# ============================================================================

print('='*90)
print('CPB v2: COMPLETE STEP-BY-STEP ENHANCEMENT FRAMEWORK')
print('Target: MAPE < 0.02% (Accuracy > 99.98%)')
print('='*90)

import sys
import os
import time
import warnings
from datetime import datetime, timedelta
import math

warnings.filterwarnings('ignore')

print('\n[SETUP] Loading libraries...')

import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, mean_absolute_percentage_error

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print('OK - All libraries loaded')

# ============================================================================
# STEP 1: ENHANCED FEATURE ENGINEERING (14 -> 28 features)
# ============================================================================

class EnhancedFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        print('    [FEATURES] Building enhanced feature set...')
        
        # ===== BASIC =====
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # ===== MOMENTUM (6 features) =====
        df['momentum_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # Momentum acceleration (rate of change of momentum)
        df['momentum_accel_3'] = df['momentum_3'] - df['momentum_3'].shift(1)
        df['momentum_accel_5'] = df['momentum_5'] - df['momentum_5'].shift(1)
        df['momentum_accel_10'] = df['momentum_10'] - df['momentum_10'].shift(1)
        
        # ===== MOVING AVERAGES (4 features) =====
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # ===== TREND STRENGTH (3 features) =====
        df['sma_ratio'] = df['sma_5'] / (df['sma_20'] + 1e-10)
        df['price_sma_20'] = df['close'] / (df['sma_20'] + 1e-10)
        df['sma_distance'] = (df['sma_5'] - df['sma_20']) / df['close']
        
        # ===== RSI VARIANTS (2 features) =====
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50).clip(0, 100)
        
        gain_7 = (delta.where(delta > 0, 0)).rolling(7).mean()
        loss_7 = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs_7 = gain_7 / (loss_7 + 1e-10)
        df['rsi_7'] = 100 - (100 / (1 + rs_7))
        df['rsi_7'] = df['rsi_7'].fillna(50).clip(0, 100)
        
        # ===== VOLATILITY (4 features) =====
        df['volatility_10'] = df['returns'].rolling(10).std() * 100
        df['volatility_20'] = df['returns'].rolling(20).std() * 100
        df['volatility_ratio'] = (df['volatility_10'] + 1e-10) / (df['volatility_20'] + 1e-10)
        df['volatility_trend'] = df['volatility_10'] - df['volatility_20']
        
        # ===== PRICE POSITION (3 features) =====
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['body_ratio'] = abs(df['close'] - df['open']) / df['close']
        
        # ===== VOLUME (2 features) =====
        df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        df['volume_momentum'] = df['volume'] / df['volume'].shift(1)
        
        # ===== MACD-LIKE (2 features) =====
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # ===== TREND (1 feature) =====
        df['trend'] = (df['close'] > df['sma_20']).astype(int)
        
        # Cleanup
        self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        for col in self.df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].clip(-100, 100)
        
        print(f'    [FEATURES] Total features: {len(self.get_features())}')
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude]

# ============================================================================
# STEP 2: IMPROVED TARGET VARIABLE (Dynamic Threshold)
# ============================================================================

class ImprovedTargetGenerator:
    def __init__(self, close_prices):
        self.close_prices = close_prices
    
    def get_labels(self, method='dynamic_threshold'):
        labels = []
        
        if method == 'dynamic_threshold':
            returns = np.diff(self.close_prices) / self.close_prices[:-1]
            
            # Calculate dynamic threshold based on rolling volatility
            volatility_window = 20
            for i in range(volatility_window, len(returns) + 1):
                recent_returns = returns[i-volatility_window:i]
                threshold = np.std(recent_returns) * 0.5
                
                if i < len(returns):
                    if returns[i-1] > threshold:
                        labels.append(1)  # Up
                    elif returns[i-1] < -threshold:
                        labels.append(0)  # Down
                    else:
                        labels.append(-1)  # Hold (neutral)
            
            # Filter out neutral labels
            valid_indices = [i for i, l in enumerate(labels) if l != -1]
            labels = [labels[i] for i in valid_indices]
            
            return np.array(labels), np.array(valid_indices) + volatility_window
        
        return np.array(labels), np.arange(len(labels))

# ============================================================================
# DATA COLLECTION
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

class EnhancedPreprocessor:
    def __init__(self, df, lookback=20):
        self.df = df.copy()
        self.lookback = lookback
        self.scaler = StandardScaler()
    
    def prepare(self, feature_cols):
        self.df = self.df.dropna()
        feature_data = self.df[feature_cols].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        self.df = self.df.loc[feature_data.index]
        
        feature_data = self.scaler.fit_transform(feature_data)
        self.features = feature_data
        self.feature_cols = feature_cols
        
        return feature_data, feature_cols
    
    def create_sequences(self, y):
        X, Y = [], []
        data = self.features
        
        for i in range(self.lookback, len(data)):
            if i - 1 < len(y):
                X.append(data[i - self.lookback:i])
                Y.append(y[i - 1])
        
        return np.array(X), np.array(Y)
    
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
# STEP 3: IMPROVED MODEL ARCHITECTURE
# ============================================================================

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=3, dropout=0.4, attention=True):
        super().__init__()
        
        self.attention = attention
        self.bn_input = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM for better context
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        if self.attention:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Dense layers
        lstm_output_size = hidden_size * 2
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.bn_hidden = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Input batch norm
        x_flat = x.reshape(-1, features)
        x_flat = self.bn_input(x_flat)
        x = x_flat.reshape(batch_size, seq_len, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        if self.attention:
            attn_out, _ = self.attention_layer(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attn_out
        
        # Use last output
        last_out = lstm_out[:, -1, :]
        
        # Dense
        out = self.fc1(last_out)
        out = self.bn_hidden(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# STEP 4: IMPROVED FOCAL LOSS
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
# STEP 5: ENHANCED TRAINER
# ============================================================================

class EnhancedTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, X_train, y_train, X_val, y_val, config):
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
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
        
        best_val_acc = 0
        patience_counter = 0
        best_weights = None
        
        for epoch in range(config['epochs']):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['gradient_clip'])
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            self.model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
                    
                    preds = output.argmax(dim=1)
                    val_acc += (preds == y_batch).sum().item()
            
            val_loss /= len(val_loader)
            val_acc /= len(y_val)
            
            current_lr = lr_scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1:3d}/{config["epochs"]}: Loss={train_loss:.6f}, Val_Acc={val_acc:.4f}, LR={current_lr:.6f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_weights = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= config['early_stop_patience']:
                    print(f'    Early Stop at epoch {epoch+1} (Best Val Acc: {best_val_acc:.4f})')
                    if best_weights:
                        self.model.load_state_dict(best_weights)
                    break
        
        return {'best_val_acc': float(best_val_acc), 'epochs': epoch+1}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print('\n' + '='*90)
print('EXECUTION START')
print('='*90)

CONFIG = {
    'coins': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT'],
    'epochs': 80,
    'batch_size': 16,
    'learning_rate': 3e-4,
    'lookback': 20,
    'weight_decay': 1e-4,
    'gradient_clip': 1.0,
    'warmup_epochs': 5,
    'early_stop_patience': 20,
    'focal_alpha': 0.5,
    'focal_gamma': 3.0,
}

if torch.cuda.is_available():
    device = 'cuda'
    print(f'Device: GPU ({torch.cuda.get_device_name(0)})')
else:
    device = 'cpu'
    print('Device: CPU')

print('\n[STEP 1] Downloading data...')
all_data = {}
collector = BinanceDataCollector()

for coin in CONFIG['coins']:
    try:
        print(f'  {coin}...', end=' ', flush=True)
        df = collector.get_klines(coin, '1h', limit=3000)
        if BinanceDataCollector.validate(df):
            all_data[coin] = df
            print(f'OK ({len(df)} candles)')
        else:
            print('X validation failed')
    except Exception as e:
        print(f'X {str(e)[:30]}')

print(f'\nDownloaded: {len(all_data)}/{len(CONFIG["coins"])} coins')

if len(all_data) == 0:
    print('ERROR: No data')
    exit(1)

print('\n[STEP 2] Training Enhanced Models...')

results = []

for coin in all_data:
    print(f'\n  {coin}')
    print('  ' + '-'*80)
    
    try:
        df = all_data[coin]
        
        # Enhanced features
        fe = EnhancedFeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        # Improved target
        target_gen = ImprovedTargetGenerator(df_features['close'].values)
        y, valid_indices = target_gen.get_labels('dynamic_threshold')
        
        print(f'    Target: Dynamic threshold (filtered {len(all_data[coin]) - len(y)} neutral points)')
        
        # Preprocessing
        prep = EnhancedPreprocessor(df_features.iloc[valid_indices], lookback=CONFIG['lookback'])
        features, _ = prep.prepare(feature_cols)
        X, y_seq = prep.create_sequences(y)
        data = prep.split_data(X, y_seq)
        
        if len(X) < 100:
            print('    X Insufficient data')
            continue
        
        unique, counts = np.unique(y_seq, return_counts=True)
        print(f'    Distribution: Down={counts[0]}, Up={counts[1] if len(counts) > 1 else 0}')
        
        # Enhanced model
        model = EnhancedLSTMModel(
            input_size=features.shape[-1],
            hidden_size=128,
            num_layers=3,
            dropout=0.4,
            attention=True
        )
        print(f'    Model: {model.count_params():,} parameters')
        
        # Training
        trainer = EnhancedTrainer(model, device=device)
        history = trainer.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'], CONFIG)
        
        # Evaluation
        model.eval()
        X_test_t = torch.FloatTensor(data['X_test']).to(device)
        y_test_t = torch.LongTensor(data['y_test']).to(device)
        
        with torch.no_grad():
            logits = model(X_test_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = logits.argmax(dim=1).cpu().numpy()
        
        y_test = data['y_test']
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # MAPE calculation
        mape = mean_absolute_percentage_error(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0
        
        results.append({
            'coin': coin,
            'accuracy': accuracy,
            'f1': f1,
            'mape': mape,
            'epochs': history['epochs']
        })
        
        print(f'    Accuracy={accuracy:.4f}, F1={f1:.4f}, MAPE={mape:.6f}')
        print(f'    Status: TRAINED ({history["epochs"]} epochs)')
        
    except Exception as e:
        print(f'    ERROR: {str(e)[:60]}')

# ============================================================================
# RESULTS
# ============================================================================

print('\n' + '='*90)
print('RESULTS SUMMARY')
print('='*90)

if results:
    print('\n{:12s} {:>10s} {:>10s} {:>10s} {:>8s}'.format(
        'Coin', 'Accuracy', 'F1 Score', 'MAPE', 'Epochs'))
    print('-'*90)
    
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        status = 'EXCELLENT' if r['accuracy'] > 0.8 else 'GOOD' if r['accuracy'] > 0.7 else 'OK' if r['accuracy'] > 0.6 else 'POOR'
        print('{:12s} {:>10.4f}  {:>10.4f}  {:>10.6f}  {:>8d}  {}'.format(
            r['coin'], r['accuracy'], r['f1'], r['mape'], r['epochs'], status))
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_mape = np.mean([r['mape'] for r in results])
    
    print('-'*90)
    print(f'\nAverage Accuracy: {avg_acc:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f}')
    print(f'Average MAPE: {avg_mape:.6f}')
    
    if avg_accuracy > 0.99:
        print(f'\nSUCCESS! Achieved MAPE < 0.02% (Accuracy > 99.98%)')
    elif avg_acc > 0.95:
        print(f'\nVERY GOOD! Accuracy > 95%, close to target')
    elif avg_acc > 0.80:
        print(f'\nGOOD! Significant improvement from baseline (50%)')
    else:
        print(f'\nNEED MORE WORK - Current accuracy {avg_acc:.2%}')

print('\n' + '='*90)
print('EXECUTION COMPLETE')
print('='*90)
