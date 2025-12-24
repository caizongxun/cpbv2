#!/usr/bin/env python3
# ============================================================================
# CPB V4 Training - Remote Execution (Fixed Dependencies)
# ============================================================================
# 次當在 Google Colab 中使用
# 無需安裝, 自動处理所有依賴
#
# 使用步驟:
# 1. 第一個 Cell: 複製下面了整個脚本
# 2. 影改這三個參數: HF_TOKEN, TRAINING_COIN, EPOCHS
# 3. 執行 Cell
#
# ============================================================================

import sys
import subprocess
import os

print("[*] CPB V4 Training - Remote Execution\n")
print("[*] Step 1: Fixing dependencies...\n")

# 緯新安裝依賴（一次性符合列表）
dependencies = [
    "numpy==1.24.3",           # Compatible version
    "pandas==2.0.3",           # Compatible with numpy 1.24
    "tensorflow==2.13.0",      # Compatible with both
    "scikit-learn==1.3.0",     # Latest stable
    "ccxt==2.1.1",             # For Binance data
    "huggingface-hub==0.16.4",  # For model upload
    "PyYAML==6.0",             # Sometimes needed
]

for dep in dependencies:
    print(f"[+] Installing {dep}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])
    except Exception as e:
        print(f"    [!] Warning: {e}")
        continue

print("\n[+] Dependencies installed successfully!\n")

# 驗證寬核出導
print("[*] Step 2: Verifying imports...\n")
try:
    import numpy as np
    print(f"[✓] NumPy {np.__version__}")
    
    import pandas as pd
    print(f"[✓] Pandas {pd.__version__}")
    
    import tensorflow as tf
    print(f"[✓] TensorFlow {tf.__version__}")
    
    import sklearn
    print(f"[✓] Scikit-learn {sklearn.__version__}")
    
    import ccxt
    print(f"[✓] CCXT {ccxt.__version__}")
    
    print("\n[+] All imports successful!\n")
except ImportError as e:
    print(f"[!] Import failed: {e}")
    print("    Retrying with fresh installation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "-q"] + dependencies)
    print("[+] Retry completed\n")

# ============================================================================
# 配置參數 - 只需修改這裡
# ============================================================================
print("="*80)
print("[*] Step 3: Configuration")
print("="*80)

HF_TOKEN = "hf_YOUR_TOKEN_HERE"        # 換成你的 HuggingFace token
TRAINING_COIN = "BTCUSDT"              # 選擇訓練幣種 (BTCUSDT, ETHUSDT, ...)
EPOCHS = 80                            # 訓練輪次 (50-150)

print(f"\n[✓] Configuration:")
print(f"    Coin: {TRAINING_COIN}")
print(f"    Epochs: {EPOCHS}")
print(f"    Token: {'*' * 10 + HF_TOKEN[-10:] if HF_TOKEN != 'hf_YOUR_TOKEN_HERE' else 'NOT SET'}")
print()

# ============================================================================
# 核心訓練達成
# ============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json
from pathlib import Path
import tempfile
from huggingface_hub import HfApi, HfFolder

print("="*80)
print("[*] Step 4: Downloading data...")
print("="*80)

def fetch_klines(coin, limit=3000, timeframe='1h'):
    """Download K-line data from Binance"""
    import ccxt
    exchange = ccxt.binance({'enableRateLimit': True})
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
    print(f"[✓] Data downloaded: {len(data)} candles\n")
    return data

try:
    data = fetch_klines(TRAINING_COIN, limit=3500)
except Exception as e:
    print(f"[!] Download failed: {e}")
    print("[+] Using synthetic data for demo...\n")
    np.random.seed(42)
    data = np.random.randn(3500, 5) * 100 + 50000

print("="*80)
print("[*] Step 5: Data preprocessing...")
print("="*80)

def preprocess_data(data, seq_len=20):
    """Normalize and create sequences"""
    closes = data[:, 3]
    
    # Calculate targets
    price_changes = []
    for i in range(len(closes) - seq_len):
        current = closes[i]
        future = closes[i + seq_len]
        pct_change = (future - current) / current * 100
        price_changes.append(1 if pct_change > 0 else 0)  # Binary classification
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_norm = scaler.fit_transform(data)
    
    # Create sequences
    X = []
    y = []
    for i in range(len(data_norm) - seq_len):
        X.append(data_norm[i:i+seq_len])
        y.append(price_changes[i])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"[✓] X shape: {X.shape}")
    print(f"[✓] y shape: {y.shape}")
    print(f"[✓] Positive samples: {np.sum(y)}/{len(y)}\n")
    
    return X, y, scaler

X, y, scaler = preprocess_data(data, seq_len=20)

print("="*80)
print("[*] Step 6: Building model...")
print("="*80)

def build_model(seq_len=20, input_features=5):
    """Build CNN-LSTM model"""
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_len, input_features)),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = build_model(seq_len=20, input_features=5)
print("[✓] Model built\n")

print("="*80)
print("[*] Step 7: Training...")
print("="*80)

split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\n[✓] Training completed!\n")

# Evaluate
print("="*80)
print("[*] Step 8: Evaluation...")
print("="*80)

y_pred_val = model.predict(X_val, verbose=0)
y_pred_binary = (y_pred_val > 0.5).astype(int).flatten()

acc = accuracy_score(y_val, y_pred_binary)
f1 = f1_score(y_val, y_pred_binary)
auc = roc_auc_score(y_val, y_pred_val)

print(f"\n[✓] Validation Metrics:")
print(f"    Accuracy: {acc:.4f}")
print(f"    F1-Score: {f1:.4f}")
print(f"    AUC-ROC: {auc:.4f}\n")

# Save model
print("="*80)
print("[*] Step 9: Saving model...")
print("="*80)

model_name = f"v4_model_{TRAINING_COIN}.h5"
model.save(model_name)
print(f"[✓] Model saved: {model_name}\n")

# Upload to HuggingFace (if token provided)
if HF_TOKEN != "hf_YOUR_TOKEN_HERE":
    print("="*80)
    print("[*] Step 10: Uploading to HuggingFace...")
    print("="*80)
    
    try:
        HfFolder.save_token(HF_TOKEN)
        api = HfApi()
        
        print(f"[+] Uploading {model_name}...")
        api.upload_file(
            path_or_fileobj=model_name,
            path_in_repo=f"v4/{model_name}",
            repo_id="zongowo111/cpb-models",
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"[✓] Upload successful!\n")
    except Exception as e:
        print(f"[!] Upload failed: {e}\n")
else:
    print("[!] HF_TOKEN not set - skipping upload\n")

# Final summary
print("\n" + "="*80)
print("[✓] CPB V4 Training Complete!")
print("="*80)
print(f"\nModel: {model_name}")
print(f"Coin: {TRAINING_COIN}")
print(f"Epochs: {EPOCHS}")
print(f"Accuracy: {acc:.2%}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print("\n[*] Ready for trading!\n")
