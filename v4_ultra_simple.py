#!/usr/bin/env python3
# ============================================================================
# CPB V4 - Ultra Simple Version
# ============================================================================
# 简化來的版本。不需要編輔 NumPy, 不需要陪轉圓、粗高運作
# 只是純綿的訓練 + 驗證
#
# 使用步驟:
# 1. 影改這三个參數
# 2. 執行。完成。
#
# ============================================================================

print("\n" + "="*80)
print("CPB V4 Training - Ultra Simple Version")
print("="*80 + "\n")

# ============================================================================
# 1. 需要修改的配置
# ============================================================================

HF_TOKEN = "hf_YOUR_TOKEN_HERE"        # 誓時填你的 HuggingFace Token
TRAINING_COIN = "BTCUSDT"              # 選擇幣種: BTCUSDT, ETHUSDT, BNBUSDT, ...
EPOCHS = 50                            # 訓練輪次 (30-100, 准完成機会提前停止)

print(f"[✓] Configuration:")
print(f"    Coin: {TRAINING_COIN}")
print(f"    Epochs: {EPOCHS}")
print(f"    Token: {'SET' if HF_TOKEN != 'hf_YOUR_TOKEN_HERE' else 'NOT SET'}")
print()

# ============================================================================
# 2. 對储依賴整旧情冶
# ============================================================================

print("[*] Importing libraries...\n")

try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import warnings
    warnings.filterwarnings('ignore')
    print("[✓] Core libraries loaded")
except ImportError as e:
    print(f"[!] Error: {e}")
    print("[*] This version requires: numpy, pandas, scikit-learn")
    print("[*] But Colab should have them pre-installed")
    exit(1)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    print("[✓] TensorFlow loaded")
except ImportError as e:
    print(f"[!] TensorFlow not available: {e}")
    exit(1)

try:
    import ccxt
    print("[✓] CCXT loaded")
except ImportError:
    print("[!] CCXT not available, using synthetic data")
    ccxt = None

print()

# ============================================================================
# 3. 下載數據
# ============================================================================

print("[*] Downloading data...\n")

def fetch_klines(coin, limit=3000):
    """Download K-line data from Binance"""
    if ccxt is None:
        print("[!] CCXT not available, generating synthetic data")
        np.random.seed(42)
        data = np.random.randn(limit, 5) * 100 + 50000
        print(f"[✓] Synthetic data: {len(data)} candles\n")
        return data
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        all_klines = []
        batch_size = 1000
        
        for i in range(0, limit, batch_size):
            try:
                klines = exchange.fetch_ohlcv(coin, '1h', limit=min(batch_size, limit - i))
                all_klines.extend(klines)
                print(f"  [+] Downloaded {len(all_klines)}/{limit}")
            except Exception as e:
                print(f"  [!] Error: {e}")
                break
        
        if len(all_klines) == 0:
            raise Exception("No data fetched")
        
        data = np.array([[k[1], k[2], k[3], k[4], k[5]] for k in all_klines[-limit:]])
        print(f"[✓] Real data: {len(data)} candles\n")
        return data
    except Exception as e:
        print(f"[!] Binance fetch failed: {e}")
        print("[*] Using synthetic data instead\n")
        np.random.seed(42)
        data = np.random.randn(limit, 5) * 100 + 50000
        print(f"[✓] Synthetic data: {len(data)} candles\n")
        return data

data = fetch_klines(TRAINING_COIN, limit=3000)

# ============================================================================
# 4. 數據前處理
# ============================================================================

print("[*] Data preprocessing...\n")

def preprocess(data, seq_len=20):
    """Normalize data and create sequences"""
    closes = data[:, 3]
    
    # Binary target: 1 if price goes up, 0 if down
    y = []
    for i in range(len(closes) - seq_len):
        future_price = closes[i + seq_len]
        current_price = closes[i]
        y.append(1 if future_price > current_price else 0)
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    
    # Create sequences
    X = []
    for i in range(len(data_normalized) - seq_len):
        X.append(data_normalized[i:i+seq_len])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"[✓] X shape: {X.shape}")
    print(f"[✓] y shape: {y.shape}")
    print(f"[✓] Positive samples: {np.sum(y)}/{len(y)}\n")
    
    return X, y, scaler

X, y, scaler = preprocess(data, seq_len=20)

# ============================================================================
# 5. 模型構造
# ============================================================================

print("[*] Building model...\n")

def build_model(seq_len=20, input_features=5):
    """Simple LSTM for binary classification"""
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_len, input_features)),
        Dropout(0.2),
        
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
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

# ============================================================================
# 6. 訓練
# ============================================================================

print("[*] Training...\n")

split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\n[✓] Training completed!\n")

# ============================================================================
# 7. 計估
# ============================================================================

print("[*] Evaluation...\n")

y_pred_val = model.predict(X_val, verbose=0)
y_pred_binary = (y_pred_val > 0.5).astype(int).flatten()

acc = accuracy_score(y_val, y_pred_binary)
f1 = f1_score(y_val, y_pred_binary, zero_division=0)
auc = roc_auc_score(y_val, y_pred_val)

print(f"[✓] Validation Metrics:")
print(f"    Accuracy: {acc:.4f}")
print(f"    F1-Score: {f1:.4f}")
print(f"    AUC-ROC: {auc:.4f}\n")

# ============================================================================
# 8. 上傳
# ============================================================================

if HF_TOKEN != "hf_YOUR_TOKEN_HERE":
    print("[*] Saving and uploading...\n")
    
    model_name = f"v4_model_{TRAINING_COIN}.h5"
    model.save(model_name)
    print(f"[✓] Model saved: {model_name}")
    
    try:
        from huggingface_hub import HfApi, HfFolder
        HfFolder.save_token(HF_TOKEN)
        api = HfApi()
        
        print(f"[+] Uploading to HuggingFace...")
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

# ============================================================================
# 9. 完成
# ============================================================================

print("\n" + "="*80)
print("[✓] CPB V4 Training Complete!")
print("="*80)
print(f"\nCoin: {TRAINING_COIN}")
print(f"Accuracy: {acc:.2%}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print("\n[*] Ready for trading!\n")
