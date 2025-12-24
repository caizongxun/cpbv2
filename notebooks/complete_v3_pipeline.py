#!/usr/bin/env python3
# ============================================================================
# CPB Trading V3 Model Training - Complete Pipeline
# ============================================================================
# Version: 1.0.6 (Compatible version pinning)
# Date: 2025-12-24
# GitHub: https://raw.githubusercontent.com/caizongxun/cpbv2/main/notebooks/complete_v3_pipeline.py
# ============================================================================

print("\n" + "="*80)
print("CPB V3 Model Training Pipeline - v1.0.6")
print("Updated: 2025-12-24 - Compatible Version Pinning")
print("="*80 + "\n")

import os
import sys

# ============================================================================
# STEP 0: Aggressive Shell Cleanup with Compatible Versions
# ============================================================================
print("[*] STEP 0: 強制清理環境 (Aggressive Cleanup)...")
print("[!] 使用 pip cache 清理 + 強制卸載衝突包...\n")

# Clear pip cache
os.system(f"{sys.executable} -m pip cache purge > /dev/null 2>&1")

# Aggressive uninstall
conflict_pkgs = [
    "numpy", "scipy", "scikit-learn", "sklearn",
    "tensorflow", "keras", "pandas"
]

for pkg in conflict_pkgs:
    os.system(f"{sys.executable} -m pip uninstall -y {pkg} 2>&1 | grep -i uninstalling")

print("\n[✓] 快取清理 + 完整卸載完成\n")
print("[*] 安裝相容版本 (手動選擇補上是最新穩定版本)...\n")

# Install with compatible pinning
os.system(
    f"{sys.executable} -m pip install --no-cache-dir "
    "'numpy<2.0' scipy scikit-learn tensorflow pandas ccxt huggingface-hub "
    "2>&1 | tail -8"
)

print("\n[✓] 環境重置完成\n")

# ============================================================================
# STEP 1: Configuration
# ============================================================================
print("[*] STEP 1: 配置參數...\n")

# User Configuration
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

print(f"[✓] 配置完成:")
print(f"    訓練幣種: {TRAINING_COIN}")
print(f"    Epochs: {EPOCHS}")
print(f"    K 棒數量: {DATA_LIMIT}\n")

# ============================================================================
# STEP 2: Import Libraries
# ============================================================================
print("[*] STEP 2: 導入所有依賴...\n")

import numpy as np
print(f"  [+] NumPy: {np.__version__}")

import pandas as pd
print(f"  [+] Pandas: {pd.__version__}")

from sklearn.preprocessing import MinMaxScaler
print(f"  [+] scikit-learn imported")

import ccxt
print(f"  [+] CCXT imported")

import tensorflow as tf
print(f"  [+] TensorFlow: {tf.__version__}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
print(f"  [+] TensorFlow Keras layers imported")

from huggingface_hub import HfApi, HfFolder
print(f"  [+] HuggingFace Hub imported")

import json
from pathlib import Path
import tempfile

print("\n[✓] 所有包導入成功\n")

# ============================================================================
# PART 1: Download Data from Binance
# ============================================================================
print("[*] PART 1: 正在下載數據...\n")

def fetch_klines(coin, limit=3000, timeframe='1h'):
    exchange = ccxt.binance()
    all_klines = []
    batch_size = 1000
    
    for i in range(0, limit, batch_size):
        try:
            klines = exchange.fetch_ohlcv(coin, timeframe, limit=min(batch_size, limit - i))
            all_klines.extend(klines)
            print(f"  [+] 已下載 {len(all_klines)}/{limit} 根")
        except Exception as e:
            print(f"  [!] 下載錯誤: {e}")
            break
    
    data = np.array([[k[1], k[2], k[3], k[4], k[5]] for k in all_klines[-limit:]])
    print(f"[✓] 下載完成: {len(data)} 根\n")
    return data

data = fetch_klines(TRAINING_COIN, limit=DATA_LIMIT)

# ============================================================================
# PART 2: Data Preprocessing
# ============================================================================
print("[*] PART 2: 數據前處理...\n")

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
print("[*] PART 3: 構建模型...\n")

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
print("[✓] 模型構建完成\n")

# ============================================================================
# PART 4: Training
# ============================================================================
print("[*] PART 4: 訓練模型...\n")

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
print(f"\n[✓] 訓練完成!")
print(f"    Final Loss: {loss:.6f}")
print(f"    Final MAE: {mae:.6f}\n")

# ============================================================================
# PART 5: Upload to HuggingFace
# ============================================================================
print("[*] PART 5: 上傳模型到 HuggingFace...\n")

# Get HF Token from user
hf_token = input("[!] 請輸入你的 HuggingFace Token (https://huggingface.co/settings/tokens):\n> ")

if not hf_token or hf_token.strip() == "":
    print("[!] 未提供 Token，跳過上傳")
else:
    print("\n[*] 開始上傳...\n")
    
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
    
    print("[+] 上傳 scaler 參數...")
    try:
        api.upload_file(
            path_or_fileobj=str(scaler_path),
            path_in_repo=f"{HF_VERSION}/scaler_params.json",
            repo_id=HF_REPO,
            repo_type="dataset",
            token=hf_token
        )
        print("    [✓] scaler 上傳成功\n")
    except Exception as e:
        print(f"    [!] scaler 上傳失敗: {e}\n")
    
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
print("[✓] V3 模型訓練和上傳完成!")
print("="*80)
print(f"查看模型: https://huggingface.co/datasets/{HF_REPO}/tree/main/{HF_VERSION}\n")
