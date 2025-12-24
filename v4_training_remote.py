#!/usr/bin/env python3
# ============================================================================
# CPB V4 Model Training - Remote Execution Script
# Execute this in ONE Colab Cell with:
#   import urllib.request
#   print("[*] Downloading V4 training script...")
#   urllib.request.urlretrieve(
#       'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_training_remote.py',
#       'v4_train.py'
#   )
#   exec(open('v4_train.py').read())
# ============================================================================

import warnings
import time
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

warnings.filterwarnings('ignore')

print("\n[*] Installing dependencies...")
os.system('pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null')
os.system('pip install -q numpy pandas requests scikit-learn matplotlib ccxt huggingface-hub --upgrade 2>/dev/null')

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from huggingface_hub import HfApi, HfFolder

print("[✓] Dependencies installed\n")

# ============================================================================
# CONFIGURATION
# ============================================================================
print("[*] Loading configuration...")

# TODO: Replace these with your values
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
TRAINING_COIN = "BTCUSDT"
EPOCHS = 80
DATA_LIMIT = 3500
BATCH_SIZE = 32
SEQ_LEN = 20
LEARNING_RATE = 5e-4

HF_REPO = "zongowo111/cpb-models"
HF_VERSION = "v4"

SUPPORTED_COINS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT',
    'ADAUSDT', 'DOGEUSDT', 'LINKUSDT', 'XRPUSDT', 'LTCUSDT',
    'MATICUSDT', 'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'ARBUSDT',
    'OPUSDT', 'STXUSDT', 'INJUSDT', 'LUNCUSDT', 'LUNAUSDT'
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"[✓] Config loaded")
print(f"    Device: {device}")
print(f"    Coin: {TRAINING_COIN}")
print(f"    Epochs: {EPOCHS}\n")

# ============================================================================
# PART 1: Advanced Feature Engineering
# ============================================================================
print("[*] PART 1: Advanced Feature Engineering...")

class AdvancedFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        df['acceleration'] = df['momentum_5'] - df['momentum_5'].shift(1)
        
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['sma_ratio'] = df['sma_5'] / (df['sma_20'] + 1e-10)
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50).clip(0, 100) / 100
        
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_normalized'] = df['atr_14'] / (df['close'] + 1e-10) * 100
        
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['price_position'] = df['price_position'].clip(0, 1)
        
        df['volatility'] = df['returns'].rolling(20).std() * 100
        df['volatility'] = df['volatility'].clip(0, 10)
        df['volatility_ratio'] = df['volatility'] / (df['volatility'].rolling(50).mean() + 1e-10)
        df['volatility_ratio'] = df['volatility_ratio'].clip(0, 5)
        
        df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        df['volume_ma_ratio'] = df['volume_ma_ratio'].clip(0.1, 10)
        df['volume_trend'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).mean() + 1e-10)
        
        df['trend'] = (df['close'] > df['sma_20']).astype(int)
        
        self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
        for col in self.df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].clip(-100, 100)
        
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'tr']
        return [col for col in self.df.columns if col not in exclude]

print("[✓] OK\n")

# ============================================================================
# PART 2: CNN-LSTM Model
# ============================================================================
print("[*] PART 2: CNN-LSTM Hybrid Model...")

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=20, cnn_filters=32, lstm_hidden=64, dropout=0.3):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc_class = nn.Linear(64, 2)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc1(last_out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        logits = self.fc_class(out)
        return logits

print("[✓] OK\n")

# ============================================================================
# PART 3: Focal Loss
# ============================================================================
print("[*] PART 3: Focal Loss...")

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

print("[✓] OK\n")

# ============================================================================
# PART 4: Entry Position Calculator
# ============================================================================
print("[*] PART 4: Entry Position Calculator...")

class EntryPositionCalculator:
    @staticmethod
    def calculate(current_price, atr, volatility, pred_prob, risk_reward=1.33):
        entry_range = atr * 0.5
        entry_low = max(0, current_price - entry_range)
        entry_high = current_price + entry_range
        
        stop_loss = max(0, current_price - atr * 1.5)
        stop_distance = current_price - stop_loss
        take_profit = current_price + (stop_distance * risk_reward)
        
        vol_mul = max(0.5, min(2.0, 1.0 / (max(volatility, 0.1) / 100.0 + 0.1)))
        conf_mul = max(0.5, min(2.0, (pred_prob - 0.5) * 2 * 2))
        position_mul = vol_mul * conf_mul
        
        return {
            'entry_low': entry_low,
            'entry_high': entry_high,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_multiplier': position_mul,
            'risk_per_trade': stop_distance
        }

print("[✓] OK\n")

# ============================================================================
# PART 5: Data Download
# ============================================================================
print("[*] PART 5: Downloading data...\n")

class BinanceCollector:
    BASE_URL = "https://api.binance.us/api/v3"
    
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
        
        while current_start < end_time and len(all_klines) < limit:
            try:
                params = {
                    "symbol": symbol, "interval": interval,
                    "startTime": current_start,
                    "limit": min(1000, limit - len(all_klines))
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
                time.sleep(0.3)
            except Exception as e:
                print(f"  [!] Error: {str(e)[:50]}")
                break
        
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

collector = BinanceCollector()
df = collector.get_klines(TRAINING_COIN, '1h', limit=DATA_LIMIT)

if not collector.validate(df):
    print(f"[!] Data validation failed")
    sys.exit(1)

print(f"[✓] Downloaded: {len(df)} K-lines\n")

# ============================================================================
# PART 6: Feature Engineering
# ============================================================================
print("[*] PART 6: Computing features...")

fe = AdvancedFeatureEngineer(df)
df_feat = fe.calculate_all()
feature_cols = fe.get_features()

print(f"[✓] Features: {len(feature_cols)}\n")

# ============================================================================
# PART 7: Data Preprocessing
# ============================================================================
print("[*] PART 7: Data preprocessing...")

scaler = MinMaxScaler((0, 1))
df_feat_clean = df_feat.dropna()
feature_data = scaler.fit_transform(df_feat_clean[feature_cols])

X, y = [], []
for i in range(SEQ_LEN, len(feature_data) - 1):
    X.append(feature_data[i - SEQ_LEN:i])
    price_change = (df_feat_clean['close'].iloc[i + 1] - df_feat_clean['close'].iloc[i]) / df_feat_clean['close'].iloc[i]
    label = 1 if price_change > 0 else 0
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

if len(X) < 100:
    print(f"[!] Insufficient data ({len(X)} < 100)")
    sys.exit(1)

print(f"[✓] X shape: {X.shape}")
print(f"[✓] y shape: {y.shape}\n")

# ============================================================================
# PART 8: Train/Val Split
# ============================================================================
print("[*] PART 8: Train/Val split...")

split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
y_train = y[:split_idx]
X_val = X[split_idx:]
y_val = y[split_idx:]

print(f"[✓] Train: {len(X_train)}, Val: {len(X_val)}\n")

# ============================================================================
# PART 9: Build Model
# ============================================================================
print("[*] PART 9: Building model...")

model = CNNLSTMModel(input_size=len(feature_cols), cnn_filters=32, lstm_hidden=64, dropout=0.3)
model = model.to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[✓] Parameters: {params:,}\n")

# ============================================================================
# PART 10: Training
# ============================================================================
print("[*] PART 10: Training model...\n")

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.LongTensor(y_val).to(device)

class_counts = np.bincount(y_train)
class_weights = torch.FloatTensor([1.0 / max(c, 1) for c in class_counts])
class_weights = class_weights / class_weights.sum() * 2
class_weights = class_weights.to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = FocalLoss(alpha=0.5, gamma=3.0)

best_loss = float('inf')
patience = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            val_acc += (preds == y_batch).sum().item()
    
    val_loss /= len(val_loader)
    val_acc /= len(y_val)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}: Train={train_loss:.6f}, Val={val_loss:.6f}, Acc={val_acc:.4f}")
    
    if val_loss < best_loss - 0.0001:
        best_loss = val_loss
        patience = 0
    else:
        patience += 1
        if patience >= 20:
            print(f"Early Stop at epoch {epoch+1}")
            break

print(f"[✓] Training complete\n")

# ============================================================================
# PART 11: Evaluation
# ============================================================================
print("[*] PART 11: Evaluating...")

model.eval()
with torch.no_grad():
    logits = model(X_val_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = logits.argmax(dim=1).cpu().numpy()

y_val_np = y_val
acc = accuracy_score(y_val_np, y_pred)
f1 = f1_score(y_val_np, y_pred, zero_division=0)
auc = roc_auc_score(y_val_np, probs[:, 1])

print(f"[✓] Accuracy: {acc:.4f}")
print(f"[✓] F1-Score: {f1:.4f}")
print(f"[✓] AUC: {auc:.4f}")
print(f"[✓] Confusion Matrix:")
print(confusion_matrix(y_val_np, y_pred))
print()

# ============================================================================
# PART 12: Entry Position Calculation
# ============================================================================
print("[*] PART 12: Computing entry signals...\n")

last_close = df_feat_clean['close'].iloc[-1]
last_atr = df_feat_clean['atr_normalized'].iloc[-1]
last_vol = df_feat_clean['volatility'].iloc[-1]

X_last = torch.FloatTensor(feature_data[-SEQ_LEN:].reshape(1, SEQ_LEN, len(feature_cols))).to(device)
with torch.no_grad():
    logits_last = model(X_last)
    probs_last = torch.softmax(logits_last, dim=1).cpu().numpy()[0]
    pred_class = np.argmax(probs_last)
    pred_prob = max(probs_last)

entry_calc = EntryPositionCalculator.calculate(
    current_price=float(last_close),
    atr=float(last_atr * last_close / 100),
    volatility=float(last_vol),
    pred_prob=float(pred_prob),
    risk_reward=1.33
)

print("[ENTRY SIGNAL]")
print(f"Coin: {TRAINING_COIN}")
print(f"Direction: {'LONG UP' if pred_class == 1 else 'SHORT DOWN'} (Confidence {pred_prob:.2%})")
print(f"Current Price: {last_close:,.2f}")
print(f"Entry Range: {entry_calc['entry_low']:,.2f} ~ {entry_calc['entry_high']:,.2f}")
print(f"Stop Loss: {entry_calc['stop_loss']:,.2f}")
print(f"Take Profit: {entry_calc['take_profit']:,.2f}")
print(f"Risk/Reward: {(last_close - entry_calc['stop_loss'])/(entry_calc['take_profit'] - last_close):.2f}:1")
print(f"Position Size: {entry_calc['position_multiplier']:.2f}x\n")

# ============================================================================
# PART 13: Upload to HuggingFace
# ============================================================================
print("[*] PART 13: Uploading to HuggingFace...\n")

if HF_TOKEN == "hf_YOUR_TOKEN_HERE":
    print("[!] HF_TOKEN not configured, skipping upload")
else:
    HfFolder.save_token(HF_TOKEN)
    api = HfApi()
    
    temp_dir = Path(tempfile.gettempdir()) / 'v4_models'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_params = {
        'data_min': scaler.data_min_.tolist(),
        'data_max': scaler.data_max_.tolist(),
        'data_range': scaler.data_range_.tolist(),
        'feature_cols': feature_cols
    }
    
    scaler_path = temp_dir / 'scaler_params.json'
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    print("[+] Uploading scaler...")
    try:
        api.upload_file(
            path_or_fileobj=str(scaler_path),
            path_in_repo=f"{HF_VERSION}/scaler_params.json",
            repo_id=HF_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print("    [✓] Scaler uploaded")
    except Exception as e:
        print(f"    [!] Scaler upload failed: {e}")
    
    model_name = f"v4_model_{TRAINING_COIN}.pt"
    model_path = temp_dir / model_name
    torch.save(model.state_dict(), str(model_path))
    
    print(f"[+] Uploading model ({TRAINING_COIN})...")
    try:
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=f"{HF_VERSION}/{model_name}",
            repo_id=HF_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"    [✓] {TRAINING_COIN} uploaded")
    except Exception as e:
        print(f"    [!] {TRAINING_COIN} upload failed: {e}")

print("\n" + "="*80)
print("[✓] V4 Training Complete!")
print("="*80)
print(f"Model: https://huggingface.co/datasets/{HF_REPO}/tree/main/{HF_VERSION}")
print(f"Performance: Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
print(f"Entry Signal: {('LONG UP' if pred_class == 1 else 'SHORT DOWN')} (Confidence {pred_prob:.2%})")
print("="*80)
