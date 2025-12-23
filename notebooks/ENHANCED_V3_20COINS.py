# ============================================================================
# CPB v2: ENHANCED VERSION 3 - 20 COINS
# Target: Accuracy > 70%, MAPE < 0.3%
# Expanded from 5 to 20 trading pairs
# ============================================================================

print('='*90)
print('CPB v2: ENHANCED VERSION 3 - 20 COINS TRAINING')
print('Target: Accuracy > 70%, MAPE < 0.3%')
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

print('OK - All libraries loaded')

# ============================================================================
# CONFIGURATION - 20 COINS
# ============================================================================

COINS_TO_TRAIN = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT',
    'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'LINKUSDT', 'LTCUSDT',
    'MATICUSDT', 'UNIUSDT', 'BCHUSDT', 'XLMUSDT', 'VETUSDT',
    'ATOMUSDT', 'AXSUSDT', 'GRTUSDT', 'SANDUSDT', 'MANAUSDT'
]

CONFIG = {
    'coins': COINS_TO_TRAIN,
    'epochs': 60,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'lookback': 20,
    'weight_decay': 1e-4,
    'gradient_clip': 1.0,
    'warmup_epochs': 3,
    'early_stop_patience': 15,
    'focal_alpha': 0.5,
    'focal_gamma': 3.0,
}

print(f'\nTraining Configuration:')
print(f'  Total Coins: {len(CONFIG["coins"])}')
print(f'  Coins: {COINS_TO_TRAIN}')
print(f'  Epochs: {CONFIG["epochs"]}')
print(f'  Batch Size: {CONFIG["batch_size"]}')
print(f'  Learning Rate: {CONFIG["learning_rate"]}')

if torch.cuda.is_available():
    device = 'cuda'
    print(f'\nDevice: GPU ({torch.cuda.get_device_name(0)})')
else:
    device = 'cpu'
    print(f'\nDevice: CPU')

# ============================================================================
# DATA COLLECTION - 20 COINS
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
# FEATURE ENGINEERING
# ============================================================================

class EnhancedFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        # Basic
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum
        df['momentum_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        df['momentum_accel_3'] = df['momentum_3'] - df['momentum_3'].shift(1)
        df['momentum_accel_5'] = df['momentum_5'] - df['momentum_5'].shift(1)
        df['momentum_accel_10'] = df['momentum_10'] - df['momentum_10'].shift(1)
        
        # Moving Averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # Trend Strength
        df['sma_ratio'] = df['sma_5'] / (df['sma_20'] + 1e-10)
        df['price_sma_20'] = df['close'] / (df['sma_20'] + 1e-10)
        df['sma_distance'] = (df['sma_5'] - df['sma_20']) / df['close']
        
        # RSI
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
        
        # Volatility
        df['volatility_10'] = df['returns'].rolling(10).std() * 100
        df['volatility_20'] = df['returns'].rolling(20).std() * 100
        df['volatility_ratio'] = (df['volatility_10'] + 1e-10) / (df['volatility_20'] + 1e-10)
        df['volatility_trend'] = df['volatility_10'] - df['volatility_20']
        
        # Price Position
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['body_ratio'] = abs(df['close'] - df['open']) / df['close']
        
        # Volume
        df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        df['volume_momentum'] = df['volume'] / df['volume'].shift(1)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Trend
        df['trend'] = (df['close'] > df['sma_20']).astype(int)
        
        self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        for col in self.df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].clip(-100, 100)
        
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude]

# ============================================================================
# TARGET GENERATION
# ============================================================================

class ImprovedTargetGenerator:
    def __init__(self, close_prices):
        self.close_prices = close_prices
    
    def get_labels(self):
        returns = np.diff(self.close_prices) / self.close_prices[:-1]
        labels = []
        valid_indices = []
        
        volatility_window = 20
        for i in range(volatility_window, len(returns)):
            recent_returns = returns[i-volatility_window:i]
            threshold = np.std(recent_returns) * 0.5
            
            if returns[i-1] > threshold:
                labels.append(1)
                valid_indices.append(i)
            elif returns[i-1] < -threshold:
                labels.append(0)
                valid_indices.append(i)
        
        return np.array(labels), np.array(valid_indices) + 1

# ============================================================================
# PREPROCESSING
# ============================================================================

class EnhancedPreprocessor:
    def __init__(self, df, lookback=20):
        self.df = df.copy()
        self.lookback = lookback
        self.scaler = MinMaxScaler((0, 1))
    
    def prepare(self, feature_cols):
        self.df = self.df.dropna()
        feature_data = self.df[feature_cols].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        valid_idx = feature_data.index
        self.df = self.df.loc[valid_idx]
        
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
# MODEL
# ============================================================================

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=3, dropout=0.4):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# LOSS FUNCTION
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
                print(f'    Epoch {epoch+1:3d}/{config["epochs"]}: Loss={train_loss:.6f}, Val_Acc={val_acc:.4f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_weights = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= config['early_stop_patience']:
                    if best_weights:
                        self.model.load_state_dict(best_weights)
                    break
        
        return {'best_val_acc': float(best_val_acc), 'epochs': epoch+1}

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print('\n' + '='*90)
print('EXECUTION START')
print('='*90)

print('\n[STEP 1] Downloading data for 20 coins...')
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
    print('ERROR: No data downloaded')
    exit(1)

print('\n[STEP 2] Training Enhanced Models for all coins...')

results = []

for coin in sorted(all_data.keys()):
    print(f'\n  {coin}')
    print('  ' + '-'*80)
    
    try:
        df = all_data[coin]
        
        fe = EnhancedFeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        target_gen = ImprovedTargetGenerator(df_features['close'].values)
        y, valid_indices = target_gen.get_labels()
        
        print(f'    Target: Dynamic threshold (removed {len(df) - len(y)} neutral points)')
        
        prep = EnhancedPreprocessor(df_features.iloc[valid_indices], lookback=CONFIG['lookback'])
        features, _ = prep.prepare(feature_cols)
        X, y_seq = prep.create_sequences(y)
        data = prep.split_data(X, y_seq)
        
        if len(X) < 50:
            print(f'    X Insufficient data')
            continue
        
        unique, counts = np.unique(y_seq, return_counts=True)
        print(f'    Data: {len(X)} sequences, Down={counts[0]}, Up={counts[1] if len(counts) > 1 else 0}')
        
        model = ImprovedLSTMModel(
            input_size=features.shape[-1],
            hidden_size=128,
            num_layers=3,
            dropout=0.4
        )
        print(f'    Model: {model.count_params():,} parameters')
        
        trainer = EnhancedTrainer(model, device=device)
        history = trainer.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'], CONFIG)
        
        model.eval()
        X_test_t = torch.FloatTensor(data['X_test']).to(device)
        y_test_t = torch.LongTensor(data['y_test']).to(device)
        
        with torch.no_grad():
            logits = model(X_test_t)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        
        y_test = data['y_test']
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'coin': coin,
            'accuracy': accuracy,
            'f1': f1,
            'epochs': history['epochs']
        })
        
        print(f'    Accuracy={accuracy:.4f}, F1={f1:.4f}, Status=OK')
        
    except Exception as e:
        print(f'    ERROR: {str(e)[:80]}')

# ============================================================================
# RESULTS
# ============================================================================

print('\n' + '='*90)
print('RESULTS SUMMARY - 20 COINS')
print('='*90)

if results:
    print('\n{:12s} {:>12s} {:>12s} {:>8s} {:>12s}'.format(
        'Coin', 'Accuracy', 'F1 Score', 'Epochs', 'Status'))
    print('-'*90)
    
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        status = 'EXCELLENT' if r['accuracy'] > 0.8 else 'GOOD' if r['accuracy'] > 0.7 else 'OK' if r['accuracy'] > 0.6 else 'NEED WORK'
        print('{:12s} {:>12.4f}  {:>12.4f}  {:>8d}  {:>12s}'.format(
            r['coin'], r['accuracy'], r['f1'], r['epochs'], status))
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    excellent_count = len([r for r in results if r['accuracy'] > 0.8])
    good_count = len([r for r in results if 0.7 <= r['accuracy'] <= 0.8])
    
    print('-'*90)
    print(f'\nAverage Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)')
    print(f'Average F1 Score: {avg_f1:.4f}')
    print(f'\nPerformance Distribution:')
    print(f'  EXCELLENT (>80%): {excellent_count} coins')
    print(f'  GOOD (70-80%): {good_count} coins')
    print(f'  OK (60-70%): {len([r for r in results if 0.6 <= r["accuracy"] < 0.7])} coins')
    print(f'  NEED WORK (<60%): {len([r for r in results if r["accuracy"] < 0.6])} coins')
    
    if avg_acc > 0.80:
        print(f'\nStatus: EXCELLENT! Outstanding results across 20 coins')
    elif avg_acc > 0.70:
        print(f'\nStatus: GOOD! Solid performance on multiple trading pairs')
    elif avg_acc > 0.60:
        print(f'\nStatus: OK - Reasonable foundation for further optimization')
    else:
        print(f'\nStatus: Continue improving features and architecture')
else:
    print('\nNo results - check errors above')

print('\n' + '='*90)
print('COMPLETE')
print('='*90)
