# ============================================================================
# CPB v2: ENHANCED VERSION 3 - FIXED (FULL TRAINING)
# 修警 91% 準確率下模牢頁、不提前終止
# ============================================================================

print('='*90)
print('CPB v2: ENHANCED VERSION 3 - FIXED (FULL TRAINING)')
print('Using proven coins with proper 60 epoch training')
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
from sklearn.metrics import accuracy_score, f1_score

print('OK - All libraries loaded')

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
}

print(f'\nConfiguration: {len(CONFIG["coins"])} coins to train')
print(f'Epochs: {CONFIG["epochs"]} (NO EARLY STOPPING)')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Device: {device.upper()}')

# ============================================================================
# DATA COLLECTOR
# ============================================================================

class BinanceUSCollector:
    BASE_URL = "https://api.binance.us/api/v3"
    
    def __init__(self, timeout=15):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        self.timeout = timeout
    
    def get_klines(self, symbol, interval="1h", limit=3000):
        all_klines = []
        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(days=150)).timestamp() * 1000)
        current_start = start_time
        coin_start_time = time.time()
        
        while current_start < end_time and len(all_klines) < limit:
            if time.time() - coin_start_time > self.timeout:
                break
            
            try:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "limit": min(1000, limit - len(all_klines))
                }
                
                response = self.session.get(
                    f"{self.BASE_URL}/klines",
                    params=params,
                    timeout=5
                )
                
                if response.status_code == 400:
                    return pd.DataFrame()
                if response.status_code == 429:
                    time.sleep(2)
                    continue
                
                response.raise_for_status()
                klines = response.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                current_start = int(klines[-1][0]) + 1
                time.sleep(0.2)
                
            except Exception as e:
                time.sleep(1)
        
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
    def validate(df, min_rows=200):
        if df is None or len(df) == 0:
            return False
        if len(df) < min_rows:
            return False
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            return False
        if (df[['open', 'high', 'low', 'close', 'volume']] <= 0).any().any():
            return False
        return True

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate(self):
        df = self.df
        
        df['returns'] = df['close'].pct_change()
        df['momentum_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        df['sma_ratio'] = df['sma_5'] / (df['sma_20'] + 1e-10)
        df['price_sma_20'] = df['close'] / (df['sma_20'] + 1e-10)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50).clip(0, 100)
        
        df['volatility_10'] = df['returns'].rolling(10).std() * 100
        df['volatility_20'] = df['returns'].rolling(20).std() * 100
        
        self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
        for col in self.df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].clip(-100, 100)
        
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude]

# ============================================================================
# TARGET GENERATION - EXACT COPY FROM V1
# ============================================================================

class TargetGenerator:
    """準確的目標生成 - 動態閾值"""
    
    def __init__(self, close_prices):
        self.close_prices = close_prices
    
    def get_labels(self):
        """使用動態閾值生成標籤"""
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

class Preprocessor:
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

class LSTMModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 3, batch_first=True, dropout=dropout)
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

class FocalLoss(nn.Module):
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
# TRAINER - NO EARLY STOPPING
# ============================================================================

class Trainer:
    """訓練器 - 完整 60 epochs，不提前停止"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, X_train, y_train, X_val, y_val, epochs=60):
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = FocalLoss()
        
        best_acc = 0
        
        for epoch in range(epochs):
            self.model.train()
            output = self.model(X_train_t)
            loss = criterion(output, y_train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_t)
                val_preds = val_output.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()
            
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1}/{epochs}: Loss={loss.item():.6f}, Val_Acc={val_acc:.4f}')
            
            if val_acc > best_acc:
                best_acc = val_acc
        
        # 不提前停止！完整訓練 60 epochs
        return best_acc, epochs

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print('\n' + '='*90)
print('[STEP 1] Downloading data')
print('='*90)

all_data = {}
collector = BinanceUSCollector(timeout=15)

for i, coin in enumerate(CONFIG['coins'], 1):
    try:
        print(f'[{i:2d}/20] {coin:12s}', end=' ', flush=True)
        df = collector.get_klines(coin, '1h', limit=3000)
        if BinanceUSCollector.validate(df):
            all_data[coin] = df
            print(f'OK ({len(df)} candles)')
        else:
            print('validation failed')
    except Exception as e:
        print(f'error: {str(e)[:30]}')

print(f'\n\nSummary: Downloaded {len(all_data)}/{len(CONFIG["coins"])} coins')

if len(all_data) == 0:
    print('ERROR: No data downloaded')
    exit(1)

print('\n' + '='*90)
print('[STEP 2] Training models (60 epochs, NO EARLY STOPPING)')
print('='*90)

results = []

for coin in sorted(all_data.keys()):
    print(f'\n  {coin}')
    print('  ' + '-'*80)
    
    try:
        df = all_data[coin]
        
        fe = FeatureEngineer(df)
        df_features = fe.calculate()
        feature_cols = fe.get_features()
        
        tg = TargetGenerator(df_features['close'].values)
        y, valid_indices = tg.get_labels()
        
        prep = Preprocessor(df_features.iloc[valid_indices], lookback=20)
        features, _ = prep.prepare(feature_cols)
        X, y_seq = prep.create_sequences(y)
        data = prep.split_data(X, y_seq)
        
        if len(X) < 50:
            print(f'    Insufficient data ({len(X)} sequences)')
            continue
        
        neutral_removed = len(df) - len(y) - 20
        print(f'    Target: Dynamic threshold (removed {neutral_removed} neutral points)')
        print(f'    Data: {len(X)} sequences, Down={np.sum(y_seq==0)}, Up={np.sum(y_seq==1)}')
        
        model = LSTMModel(input_size=features.shape[-1])
        trainer = Trainer(model, device=device)
        val_acc, epochs = trainer.train(data['X_train'], data['y_train'],
                                        data['X_val'], data['y_val'], epochs=60)
        
        model.eval()
        X_test_t = torch.FloatTensor(data['X_test']).to(device)
        y_test_t = torch.LongTensor(data['y_test']).to(device)
        
        with torch.no_grad():
            logits = model(X_test_t)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        
        accuracy = accuracy_score(data['y_test'], y_pred)
        f1 = f1_score(data['y_test'], y_pred, zero_division=0)
        
        results.append({
            'coin': coin,
            'accuracy': accuracy,
            'f1': f1,
            'epochs': epochs
        })
        
        print(f'    Accuracy={accuracy:.4f}, F1={f1:.4f}, Status={'OK' if accuracy > 0.7 else 'NEEDS WORK'}')
        
    except Exception as e:
        print(f'    ERROR: {str(e)[:60]}')

# ============================================================================
# RESULTS
# ============================================================================

print('\n' + '='*90)
print('RESULTS SUMMARY - 20 COINS')
print('='*90)

if results:
    print('\n{:12s} {:>12s} {:>12s} {:>8s}       {}'.format(
        'Coin', 'Accuracy', 'F1 Score', 'Epochs', 'Status'))
    print('-'*90)
    
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        if r['accuracy'] > 0.85:
            status = 'EXCELLENT'
        elif r['accuracy'] > 0.75:
            status = 'GOOD'
        elif r['accuracy'] > 0.65:
            status = 'OK'
        else:
            status = 'NEEDS WORK'
        
        print('{:12s} {:>12.4f}  {:>12.4f}  {:>8d}       {}'.format(
            r['coin'], r['accuracy'], r['f1'], r['epochs'], status))
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    excellent = len([r for r in results if r['accuracy'] > 0.85])
    good = len([r for r in results if 0.75 <= r['accuracy'] <= 0.85])
    
    print('-'*90)
    print(f'\nAverage Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)')
    print(f'Average F1 Score: {avg_f1:.4f}')
    print(f'Total Coins Trained: {len(results)}')
    print(f'\nPerformance Distribution:')
    print(f'  EXCELLENT (>85%): {excellent} coins')
    print(f'  GOOD (75-85%): {good} coins')
    print(f'  OK (65-75%): {len([r for r in results if 0.65 <= r["accuracy"] < 0.75])} coins')
    
    if avg_acc > 0.80:
        print(f'\nStatus: EXCELLENT! Outstanding results across {len(results)} coins')
    elif avg_acc > 0.70:
        print(f'\nStatus: GOOD! Solid performance')
    else:
        print(f'\nStatus: CHECK DATA - Accuracy seems low')
else:
    print('\nNo results available')

print('\n' + '='*90)
print('COMPLETE')
print('='*90)
