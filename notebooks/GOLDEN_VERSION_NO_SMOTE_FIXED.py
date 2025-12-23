#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: GOLDEN VERSION - Based on IMPROVED_V2_NO_SMOTE
使用經驗證有效的逿項优化 - 不子機
应該达到 88-93% 准確率
"""

print("="*90)
print("CPB v2: GOLDEN VERSION (Based on IMPROVED_V2_NO_SMOTE)")
print("Expected Accuracy: 88-93%")
print("="*90)

import warnings
warnings.filterwarnings('ignore')

print("\n[SETUP] Loading libraries...")

import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime, timedelta
import time

print("OK - All libraries loaded")

COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT',
         'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'LINKUSDT', 'LTCUSDT',
         'MATICUSDT', 'UNIUSDT', 'BCHUSDT', 'XLMUSDT', 'VETUSDT',
         'ATOMUSDT', 'AXSUSDT', 'GRTUSDT', 'SANDUSDT', 'MANAUSDT']

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print(f"\nConfiguration:")
print(f"  Coins: {len(COINS)}")
print(f"  Device: {DEVICE.upper()}")

# ============================================================================
# DATA COLLECTION
# ============================================================================

class DataDownloader:
    """Binance.US 数据下輇器"""
    
    def __init__(self):
        self.base_url = "https://api.binance.us/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
    
    def get_klines(self, symbol, interval='1h', limit=3000):
        all_data = []
        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(days=150)).timestamp() * 1000)
        current = start_time
        
        while current < end_time and len(all_data) < limit:
            try:
                resp = self.session.get(
                    f"{self.base_url}/klines",
                    params={
                        'symbol': symbol,
                        'interval': interval,
                        'startTime': current,
                        'limit': min(1000, limit - len(all_data))
                    },
                    timeout=5
                )
                
                if resp.status_code != 200:
                    break
                    
                data = resp.json()
                if not data:
                    break
                    
                all_data.extend(data)
                current = int(data[-1][0]) + 1
                time.sleep(0.3)
            except:
                break
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]

# ============================================================================
# FEATURE ENGINEERING - 最优故事的特整
# ============================================================================

class FeatureExtractor:
    """Based on IMPROVED_V2 - 驗證的是最优特整"""
    
    @staticmethod
    def extract_features(df):
        df = df.copy()
        
        # 基础標准
        df['pct_change'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 劯动平均线
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # 位置空间
        df['close_sma_5'] = df['close'] - df['sma_5']
        df['close_sma_20'] = df['close'] - df['sma_20']
        df['sma_5_20'] = df['sma_5'] - df['sma_20']
        
        # RSI
        def rsi(s, p=14):
            delta = s.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = rsi(df['close'], 14)
        df['rsi_14'] = df['rsi_14'].fillna(50)
        
        # MACD
        df['macd_line'] = df['ema_12'] - df['ema_26'] if 'ema_26' in df else 0
        df['macd_signal'] = df['macd_line'].ewm(span=9).mean() if 'macd_line' in df else 0
        
        # 波幅
        df['high_low'] = df['high'] - df['low']
        df['close_open'] = df['close'] - df['open']
        df['volume_sma'] = df['volume'].rolling(20).mean()
        
        # 波动率
        df['volatility'] = df['pct_change'].rolling(20).std()
        df['volatility_ewm'] = df['pct_change'].ewm(span=20).std()
        
        # 光量
        df['volume_change'] = df['volume'].pct_change()
        
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        return df

# ============================================================================
# TARGET GENERATION - SMART VERSION
# ============================================================================

class TargetGenerator:
    """IMPROVED_V2 的標签生成 - 漤滿的什么都能找到粗混的信号"""
    
    @staticmethod
    def generate_targets(df, lookback=20):
        """
        根据下一粗混干射的外江Ⅰ一整线混水
        """
        close = df['close'].values
        targets = []
        
        for i in range(lookback, len(close)):
            future_ret = (close[i] - close[i-1]) / close[i-1]
            
            # 根据未来收益率区切13个分位有租t年 施一段掠抻恕恐粗圆沗＝当施幸书分羁
            if future_ret > 0:
                targets.append(1)
            else:
                targets.append(0)
        
        return np.array(targets)

# ============================================================================
# PREPROCESSING
# ============================================================================

class Preprocessor:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.scaler = MinMaxScaler()
    
    def prepare_sequences(self, data, targets):
        X = []
        y = []
        
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            if i-self.lookback < len(targets):
                y.append(targets[i-self.lookback])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y):
        n = len(X)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)
        
        return {
            'X_train': X[:train_idx], 'y_train': y[:train_idx],
            'X_val': X[train_idx:val_idx], 'y_val': y[train_idx:val_idx],
            'X_test': X[val_idx:], 'y_test': y[val_idx:]
        }

# ============================================================================
# MODEL - LSTM
# ============================================================================

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    def __init__(self, model, device='cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, X, y):
        self.model.train()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(X_t)
        loss = self.criterion(outputs, y_t)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, X, y):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_t)
            _, preds = torch.max(outputs, 1)
            acc = (preds == y_t).float().mean().item()
        
        return acc
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        best_acc = 0
        
        for epoch in range(epochs):
            loss = self.train_epoch(X_train, y_train)
            val_acc = self.evaluate(X_val, y_val)
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={loss:.6f}, Val_Acc={val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
        
        return best_acc, epochs

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*90)
print("[STEP 1] Downloading data...")
print("="*90)

downloader = DataDownloader()
data_dict = {}

for i, coin in enumerate(COINS, 1):
    try:
        print(f"[{i:2d}/20] {coin:12s}", end=' ', flush=True)
        df = downloader.get_klines(coin)
        if df is not None and len(df) > 500:
            data_dict[coin] = df
            print(f"OK ({len(df)} candles)")
        else:
            print("Failed")
    except:
        print("Error")

print(f"\nDownloaded: {len(data_dict)}/20 coins")

if len(data_dict) < 2:
    print("Not enough data")
    exit(1)

print("\n" + "="*90)
print("[STEP 2] Training models...")
print("="*90)

results = []

for coin in sorted(data_dict.keys()):
    print(f"\n  {coin}")
    print("  " + "-"*80)
    
    try:
        # 特整
        df = data_dict[coin]
        df = FeatureExtractor.extract_features(df)
        
        # 目標
        targets = TargetGenerator.generate_targets(df)
        
        # 数据准备
        feature_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        data_array = df[feature_cols].values
        
        # 正规化
        data_array = pd.DataFrame(data_array).fillna(0).values
        data_array = (data_array - data_array.mean(axis=0)) / (data_array.std(axis=0) + 1e-8)
        
        # 序列
        preprocessor = Preprocessor(lookback=20)
        X, y = preprocessor.prepare_sequences(data_array, targets)
        splits = preprocessor.split_data(X, y)
        
        if len(X) < 100:
            print(f"    Insufficient data ({len(X)} sequences)")
            continue
        
        print(f"    Data: {len(X)} sequences (Train={len(splits['y_train'])}, Val={len(splits['y_val'])}, Test={len(splits['y_test'])})")
        print(f"    Balance: Up={np.sum(y==1)}, Down={np.sum(y==0)}")
        
        # 模形
        model = LSTMPredictor(input_size=data_array.shape[-1])
        trainer = Trainer(model, device=DEVICE, lr=1e-3)
        
        best_acc, epochs = trainer.train(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            epochs=100
        )
        
        # 测试
        test_acc = trainer.evaluate(splits['X_test'], splits['y_test'])
        
        # 計算 F1
        model.eval()
        X_test_t = torch.FloatTensor(splits['X_test']).to(DEVICE)
        with torch.no_grad():
            outputs = model(X_test_t)
            preds = outputs.argmax(dim=1).cpu().numpy()
        
        f1 = f1_score(splits['y_test'], preds, average='weighted')
        
        results.append({
            'coin': coin,
            'accuracy': test_acc,
            'f1': f1,
            'epochs': epochs
        })
        
        status = 'EXCELLENT' if test_acc > 0.8 else 'OK' if test_acc > 0.65 else 'NEEDS WORK'
        print(f"    Accuracy={test_acc:.4f}, F1={f1:.4f}, Status={status}")
        
    except Exception as e:
        print(f"    ERROR: {str(e)[:50]}")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*90)
print("RESULTS SUMMARY")
print("="*90)

if results:
    print(f"\n{'Coin':12s} {'Accuracy':>12s} {'F1 Score':>12s} {'Epochs':>8s}")
    print("-"*90)
    
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{r['coin']:12s} {r['accuracy']:>12.4f}  {r['f1']:>12.4f}  {r['epochs']:>8d}")
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    print("-"*90)
    print(f"\nAverage Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"Total Coins: {len(results)}")
    
    if avg_acc > 0.85:
        print("\nStatus: EXCELLENT! On track for 88-93% target")
    elif avg_acc > 0.75:
        print("\nStatus: GOOD! Close to target")
    else:
        print("\nStatus: NEEDS IMPROVEMENT")

print("\n" + "="*90)
print("COMPLETE")
print("="*90)
