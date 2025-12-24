#!/usr/bin/env python3
# ============================================================================
# Colab Training - Cell 2 (Execute this after Cell 1 and Runtime restart)
# ============================================================================
# Version: 1.0
# Purpose: Train CPB V3 model
# ============================================================================

print("\n" + "="*80)
print("CPB V3 Model Training - Cell 2")
print("After NumPy 1.26.4 setup and Runtime restart")
print("="*80 + "\n")

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ccxt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from huggingface_hub import HfApi, HfFolder
import json
from pathlib import Path
import tempfile

print(f"[+] NumPy: {np.__version__}")
print(f"[+] TensorFlow: {tf.__version__}\n")

# ============================================================================
# Configuration
# ============================================================================
print("[*] Configuring parameters...\n")

TRAINING_COIN = "BTCUSDT"
EPOCHS = 80
DATA_LIMIT = 3500
BATCH_SIZE = 32
SEQ_LEN = 20

HF_REPO = "zongowo111/cpb-models"
HF_VERSION = "v3"

SUPPORTED_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
    'XRPUSDT', 'DOGEUSDT', 'LINKUSDT', 'AVAXUSDT', 'MATICUSDT',
    'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'ARBUSDT', 'OPUSDT',
    'LITUSDT', 'STXUSDT', 'INJUSDT', 'LUNCUSDT', 'LUNAUSDT'
]

print(f"[✓] Configuration complete:")
print(f"    Training coin: {TRAINING_COIN}")
print(f"    Epochs: {EPOCHS}")
print(f"    K-line limit: {DATA_LIMIT}\n")

# ============================================================================
# PART 1: Download Data
# ============================================================================
print("[*] PART 1: Downloading data from Binance...\n")

def fetch_klines(coin, limit=3000, timeframe='1h'):
    exchange = ccxt.binance()
    all_klines = []
    batch_size = 1000
    
    for i in range(0, limit, batch_size):
        try:
            klines = exchange.fetch_ohlcv(coin, timeframe, limit=min(batch_size, limit - i))
            all_klines.extend(klines)
            print(f"  [+] Downloaded {len(all_klines)}/{limit}")
        except Exception as e:
            print(f"  [!] Error: {e}")
            break
    
    data = np.array([[k[1], k[2], k[3], k[4], k[5]] for k in all_klines[-limit:]])
    print(f"[✓] Download complete: {len(data)} candles\n")
    return data

data = fetch_klines(TRAINING_COIN, limit=DATA_LIMIT)

# ============================================================================
# PART 2: Preprocess Data
# ============================================================================
print("[*] PART 2: Preprocessing data...\n")

def preprocess_data(data, seq_len=20):
    opens = data[:, 0]
    highs = data[:, 1]
    lows = data[:, 2]
    closes = data[:, 3]
    
    price_changes = []
    volatilities = []
    entry_ranges_low = []
    entry_ranges_high = []
    stop_losses = []
    take_profits = []
    
    for i in range(len(closes) - seq_len):
        current_price = closes[i]
        future_price = closes[i + seq_len]
        pct_change = (future_price - current_price) / current_price * 100
        price_changes.append(pct_change)
        
        vol = np.mean([(highs[i+j] - lows[i+j]) / closes[i+j] * 100 for j in range(seq_len)])
        volatilities.append(vol)
        
        atr = np.mean([max(highs[i+j] - lows[i+j],
                           abs(highs[i+j] - closes[i+j-1] if i+j > 0 else 0),
                           abs(lows[i+j] - closes[i+j-1] if i+j > 0 else 0))
                       for j in range(seq_len)])
        
        entry_range = (vol / 100) * atr
        entry_ranges_low.append(max(0, current_price - entry_range))
        entry_ranges_high.append(current_price + entry_range)
        stop_losses.append(max(0, current_price - atr * 1.5))
        take_profits.append(current_price + atr * 2)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    
    X = []
    y = []
    
    for i in range(len(data_normalized) - seq_len):
        X.append(data_normalized[i:i+seq_len])
        y.append([
            price_changes[i],
            volatilities[i],
            entry_ranges_low[i],
            entry_ranges_high[i],
            stop_losses[i],
            take_profits[i]
        ])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"[✓] X shape: {X.shape}")
    print(f"[✓] y shape: {y.shape}\n")
    
    return X, y, scaler

X, y, scaler = preprocess_data(data, seq_len=SEQ_LEN)

# ============================================================================
# PART 3: Build Model
# ============================================================================
print("[*] PART 3: Building model...\n")

def build_model(seq_len=20, input_features=4, output_size=6):
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_len, input_features)),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(32, activation='relu', return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        
        Dense(output_size, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = build_model(seq_len=SEQ_LEN)
print("[✓] Model built\n")

# ============================================================================
# PART 4: Training
# ============================================================================
print("[*] PART 4: Training model...\n")

split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

loss, mae = model.evaluate(X, y, verbose=0)
print(f"\n[✓] Training complete!")
print(f"    Final Loss: {loss:.6f}")
print(f"    Final MAE: {mae:.6f}\n")

# ============================================================================
# PART 5: Upload to HuggingFace
# ============================================================================
print("[*] PART 5: Uploading to HuggingFace...\n")

hf_token = input("[!] Enter your HuggingFace token (https://huggingface.co/settings/tokens):\n> ")

if not hf_token or hf_token.strip() == "":
    print("[!] No token provided, skipping upload")
else:
    print("\n[*] Starting upload...\n")
    
    HfFolder.save_token(hf_token)
    api = HfApi()
    
    temp_dir = Path(tempfile.gettempdir()) / 'v3_models'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_params = {
        'data_min': scaler.data_min_.tolist(),
        'data_max': scaler.data_max_.tolist(),
        'data_range': scaler.data_range_.tolist()
    }
    
    scaler_path = temp_dir / 'scaler_params.json'
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f)
    
    print("[+] Uploading scaler...")
    try:
        api.upload_file(
            path_or_fileobj=str(scaler_path),
            path_in_repo=f"{HF_VERSION}/scaler_params.json",
            repo_id=HF_REPO,
            repo_type="dataset",
            token=hf_token
        )
        print("    [✓] Scaler uploaded\n")
    except Exception as e:
        print(f"    [!] Scaler upload failed: {e}\n")
    
    for i, coin in enumerate(SUPPORTED_COINS, 1):
        model_name = f"v3_model_{coin}.h5"
        model_path = temp_dir / model_name
        
        model.save(str(model_path))
        
        try:
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=f"{HF_VERSION}/{model_name}",
                repo_id=HF_REPO,
                repo_type="dataset",
                token=hf_token
            )
            print(f"[{i:2d}/20] [✓] {coin}")
        except Exception as e:
            print(f"[{i:2d}/20] [!] {coin}: {e}")

print("\n" + "="*80)
print("[✓] V3 model training and upload complete!")
print("="*80)
print(f"View models: https://huggingface.co/datasets/{HF_REPO}/tree/main/{HF_VERSION}\n")
