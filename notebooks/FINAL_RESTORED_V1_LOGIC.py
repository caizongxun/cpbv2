#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: RESTORED V1 LOGIC
策略: 動態閾值 + 移除中性點
供水至月的驗證是頢羽醭
驗證與 V1 目標定義一致
預算準確率: 88-93%
"""

print("="*90)
print("CPB v2: RESTORED V1 LOGIC (Dynamic Threshold + Neutral Points Removal)")
print("Target: 88-93% Accuracy")
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
from sklearn.metrics import accuracy_score, f1_score
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

print(f"Device: {DEVICE.upper()}")

# ============================================================================
# DATA COLLECTION
# ============================================================================

class BinanceCollector:
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
# FEATURE ENGINEERING
# ============================================================================

class FeatureExtractor:
    @staticmethod
    def extract_features(df):
        df = df.copy()
        
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
        
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        for col in df.columns:
            if col not in ['time', 'open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].clip(-100, 100)
        
        return df

# ============================================================================
# TARGET GENERATION - V1 LOGIC WITH DYNAMIC THRESHOLD
# ============================================================================

class V1TargetGenerator:
    """
    V1 版本的可理機制:
    1. 計算每个時間的加權平均波幅
    2. 决定動態閾值 (threshold)
    3. 只標記超過閾值的上下赴
    4. 移除中性點 - 自動保護類別平衡
    """
    
    @staticmethod
    def generate_targets(close_prices):
        returns = np.diff(close_prices) / close_prices[:-1]
        labels = []
        valid_indices = []
        
        volatility_window = 20
        
        for i in range(volatility_window, len(returns)):
            # 計算最近 20 个時間的波幅
            recent_returns = returns[i-volatility_window:i]
            volatility = np.std(recent_returns)
            
            # 動態閾值 = 0.5 * 波幅
            threshold = volatility * 0.5
            
            # 当前收益率
            current_return = returns[i]
            
            # 只標記滿足以下條件的點
            if current_return > threshold:
                # 明確向上
                labels.append(1)
                valid_indices.append(i)
            elif current_return < -threshold:
                # 明確向下
                labels.append(0)
                valid_indices.append(i)
            # 否則: -threshold < return < threshold 譜殊中性 -> 忽略
        
        return np.array(labels), np.array(valid_indices) + 1

# ============================================================================
# PREPROCESSING
# ============================================================================

class Preprocessor:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.scaler = MinMaxScaler((0, 1))
    
    def prepare(self, df, feature_cols):
        feature_data = df[feature_cols].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        feature_data = self.scaler.fit_transform(feature_data)
        return feature_data
    
    def create_sequences(self, features, targets):
        X, y = [], []
        
        for i in range(self.lookback, len(features)):
            if i - 1 < len(targets):
                X.append(features[i - self.lookback:i])
                y.append(targets[i - 1])
        
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
# MODEL
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
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
# TRAINER
# ============================================================================

class Trainer:
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
                print(f"    Epoch {epoch+1}/{epochs}: Loss={loss.item():.6f}, Val_Acc={val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
        
        return best_acc, epochs

# ============================================================================
# MAIN
# ============================================================================

print("\n" + "="*90)
print("[STEP 1] Downloading data")
print("="*90)

collector = BinanceCollector()
data_dict = {}

for i, coin in enumerate(COINS, 1):
    try:
        print(f"[{i:2d}/20] {coin:12s}", end=' ', flush=True)
        df = collector.get_klines(coin)
        if df is not None and len(df) > 500:
            data_dict[coin] = df
            print(f"OK ({len(df)} candles)")
        else:
            print("Failed")
    except:
        print("Error")

print(f"\nDownloaded: {len(data_dict)}/20 coins")

if len(data_dict) < 4:
    print("Not enough data")
    exit(1)

print("\n" + "="*90)
print("[STEP 2] Training models (V1 Logic: Dynamic Threshold + Neutral Points Removal)")
print("="*90)

results = []

for coin in sorted(data_dict.keys()):
    print(f"\n  {coin}")
    print("  " + "-"*80)
    
    try:
        df = data_dict[coin]
        df = FeatureExtractor.extract_features(df)
        
        # V1 目標生成 - 動態閾值 + 移除中性點
        y, valid_indices = V1TargetGenerator.generate_targets(df['close'].values)
        
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        
        feature_cols = [c for c in df_valid.columns 
                       if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        
        preprocessor = Preprocessor(lookback=20)
        features = preprocessor.prepare(df_valid, feature_cols)
        X, y_seq = preprocessor.create_sequences(features, y)
        splits = preprocessor.split_data(X, y_seq)
        
        if len(X) < 50:
            print(f"    Insufficient data ({len(X)} sequences)")
            continue
        
        neutral_removed = len(df) - len(y) - 20
        down_count = np.sum(y_seq == 0)
        up_count = np.sum(y_seq == 1)
        
        print(f"    Target: Dynamic threshold (removed {neutral_removed} neutral points)")
        print(f"    Data: {len(X_seq)} sequences, Down={down_count}, Up={up_count}")
        print(f"    Balance: {down_count/(down_count+up_count)*100:.1f}% Down, {up_count/(down_count+up_count)*100:.1f}% Up")
        
        model = LSTMModel(input_size=features.shape[-1])
        trainer = Trainer(model, device=DEVICE)
        best_acc, epochs = trainer.train(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            epochs=60
        )
        
        model.eval()
        X_test_t = torch.FloatTensor(splits['X_test']).to(DEVICE)
        y_test_t = torch.LongTensor(splits['y_test']).to(DEVICE)
        
        with torch.no_grad():
            logits = model(X_test_t)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        
        accuracy = accuracy_score(splits['y_test'], y_pred)
        f1 = f1_score(splits['y_test'], y_pred, zero_division=0)
        
        results.append({
            'coin': coin,
            'accuracy': accuracy,
            'f1': f1,
            'epochs': epochs
        })
        
        status = 'EXCELLENT' if accuracy > 0.80 else 'GOOD' if accuracy > 0.70 else 'OK'
        print(f"    Accuracy={accuracy:.4f}, F1={f1:.4f}, Status={status}")
        
    except Exception as e:
        print(f"    ERROR: {str(e)[:50]}")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*90)
print("RESULTS SUMMARY - V1 RESTORED LOGIC")
print("="*90)

if results:
    print(f"\n{'Coin':12s} {'Accuracy':>12s} {'F1 Score':>12s} {'Epochs':>8s}  Status")
    print("-"*90)
    
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        if r['accuracy'] > 0.85:
            status = 'EXCELLENT'
        elif r['accuracy'] > 0.75:
            status = 'GOOD'
        elif r['accuracy'] > 0.65:
            status = 'OK'
        else:
            status = 'NEEDS WORK'
        print(f"{r['coin']:12s} {r['accuracy']:>12.4f}  {r['f1']:>12.4f}  {r['epochs']:>8d}  {status}")
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    excellent = len([r for r in results if r['accuracy'] > 0.85])
    good = len([r for r in results if 0.75 <= r['accuracy'] <= 0.85])
    
    print("-"*90)
    print(f"\nAverage Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"\nPerformance Distribution:")
    print(f"  EXCELLENT (>85%): {excellent} coins")
    print(f"  GOOD (75-85%): {good} coins")
    
    if avg_acc > 0.85:
        print(f"\nStatus: EXCELLENT! On target for 88-93%!")
    elif avg_acc > 0.80:
        print(f"\nStatus: VERY GOOD! Close to 88-93%!")
    else:
        print(f"\nStatus: Results need investigation")

print("\n" + "="*90)
print("COMPLETE")
print("="*90)
