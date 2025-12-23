#!/usr/bin/env python3
"""
CPB Trading V3 Model Training - Colab One-Shot Execution
一个 Cell 就能完成所有訓練、評估和上傳

上推在 Colab 中直接运行（不要用 exec()）
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import json

# ============================================================================
# Step 0: 安裝依賴
# ============================================================================
print("[*] 安裝依賴中...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "tensorflow", "numpy", "pandas", "ccxt", 
                       "huggingface-hub", "scikit-learn"])
print("[✓] 依賴安裝完成")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import ccxt
from huggingface_hub import HfApi, HfFolder

# ============================================================================
# 配置
# ============================================================================
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
    print("[✓] Colab Secrets 讀取成功")
except:
    print("[!] 未偵渫到 Colab 環境，使用不了 GitHub 自動上傳功能")
    HF_TOKEN = input("\n請粘貼 HuggingFace Token: ").strip()
    GITHUB_TOKEN = None

if HF_TOKEN:
    HfFolder.save_token(HF_TOKEN)
    print("[✓] HuggingFace 認證完成")

# ============================================================================
# Step 1: 下載數據
# ============================================================================
def fetch_klines_for_training(coin: str, limit: int = 3000, timeframe: str = '1h') -> np.ndarray:
    """Binance API 下載 K 線"""
    exchange = ccxt.binance()
    all_klines = []
    
    print(f"\n[*] 正在下載 {coin} 的 {limit} 根 K 棒...")
    
    batch_size = 1000
    for i in range(0, limit, batch_size):
        try:
            klines = exchange.fetch_ohlcv(coin, timeframe, limit=min(batch_size, limit - i))
            all_klines.extend(klines)
            print(f"  [+] 已下載 {len(all_klines)}/{limit} 根")
        except Exception as e:
            print(f"  [!] 下載失敗: {e}")
            break
    
    data = np.array([[k[1], k[2], k[3], k[4], k[5]] for k in all_klines[-limit:]])
    print(f"[✓] 下載完成: {len(data)} 根 K 棒")
    return data

# ============================================================================
# Step 2: 數據前處理
# ============================================================================
def preprocess_data(data: np.ndarray, seq_len: int = 20) -> tuple:
    """OHLC 批正規化和目標計算"""
    opens = data[:, 0]
    highs = data[:, 1]
    lows = data[:, 2]
    closes = data[:, 3]
    
    # 撤個價格變化
    price_changes = []
    for i in range(len(closes) - seq_len):
        current_price = closes[i]
        future_price = closes[i + seq_len]
        pct_change = (future_price - current_price) / current_price * 100
        price_changes.append(pct_change)
    
    # 波動率
    volatilities = []
    for i in range(len(closes) - seq_len):
        vol = np.mean([(highs[i+j] - lows[i+j]) / closes[i+j] * 100 
                       for j in range(seq_len)])
        volatilities.append(vol)
    
    # 開單範圍、止損、止盈
    entry_ranges_low = []
    entry_ranges_high = []
    stop_losses = []
    take_profits = []
    
    for i in range(len(closes) - seq_len):
        current_price = closes[i]
        future_vol = volatilities[i]
        
        # ATR 計算
        atr = np.mean([max(highs[i+j] - lows[i+j],
                           abs(highs[i+j] - closes[i+j-1] if i+j > 0 else 0),
                           abs(lows[i+j] - closes[i+j-1] if i+j > 0 else 0))
                       for j in range(seq_len)])
        
        entry_range = (future_vol / 100) * atr
        entry_ranges_low.append(max(0, current_price - entry_range))
        entry_ranges_high.append(current_price + entry_range)
        
        stop_losses.append(max(0, current_price - atr * 1.5))
        take_profits.append(current_price + atr * 2)
    
    # Min-Max 正規化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    
    # 構怠序列
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
    
    print(f"[✓] 數據前處理完成:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    
    return X, y, scaler

# ============================================================================
# Step 3: 構建 V3 模型
# ============================================================================
def build_v3_model(seq_len: int = 20, input_features: int = 4) -> tf.keras.Model:
    """LSTM 模型 - 輸出 6 個預測值"""
    model = Sequential([
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True,
                             input_shape=(seq_len, input_features)),
        BatchNormalization(),
        Dropout(0.2),
        
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        
        Dense(6, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print("[✓] V3 模型構建完成:")
    model.summary()
    
    return model

# ============================================================================
# Step 4: 訓練模型 (100 EPOCHS)
# ============================================================================
def train_v3_model(model: tf.keras.Model, X: np.ndarray, y: np.ndarray,
                   epochs: int = 100, batch_size: int = 32) -> tf.keras.Model:
    """100 epochs 訓練"""
    print(f"\n[*] 開始訓練 V3 模型 (epochs={epochs}, batch_size={batch_size})...")
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    print(f"[✓] 訓練完成! 最佳 Val Loss: {min(history.history['val_loss']):.6f}")
    
    return model

# ============================================================================
# Step 5: 模型評估
# ============================================================================
def evaluate_model(model: tf.keras.Model, X: np.ndarray, y: np.ndarray):
    """MSE 和 MAE 評估"""
    loss, mae = model.evaluate(X, y, verbose=0)
    print(f"\n[✓] 模型評估結果:")
    print(f"  - Loss (MSE): {loss:.6f}")
    print(f"  - MAE: {mae:.6f}")

# ============================================================================
# Step 6: 準備模型 (20 種幣)
# ============================================================================
SUPPORTED_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
    'XRPUSDT', 'DOGEUSDT', 'LINKUSDT', 'AVAXUSDT', 'MATICUSDT',
    'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'ARBUSDT', 'OPUSDT',
    'LITUSDT', 'STXUSDT', 'INJUSDT', 'LUNCUSDT', 'LUNAUSDT'
]

def prepare_models_for_upload(trained_model: tf.keras.Model,
                              scaler,
                              output_dir: str = '/tmp/v3_models') -> list:
    """20 个幣種模型文件"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model_files = []
    
    print(f"\n[*] 正在準備 {len(SUPPORTED_COINS)} 个模型文件...")
    
    for coin in SUPPORTED_COINS:
        model_name = f"v3_model_{coin}.h5"
        model_path = os.path.join(output_dir, model_name)
        trained_model.save(model_path)
        model_files.append(model_path)
        print(f"[+] 準備完成: {coin}")
    
    print(f"[✓] {len(model_files)} 个模型準備完成")
    return model_files

# ============================================================================
# Step 7: 上傳到 HuggingFace
# ============================================================================
def upload_to_huggingface(model_files: list, repo_id: str = "zongowo111/cpb-models",
                          version: str = "v3"):
    """HuggingFace 上傳"""
    if not HF_TOKEN:
        print("\n[!] 沒有 HF_TOKEN，跳過 HuggingFace 上傳")
        return
    
    api = HfApi()
    
    print(f"\n[*] 正在上傳 {len(model_files)} 个模型到 HuggingFace...")
    print(f"    目標: {repo_id}/{version}/")
    
    for model_file in model_files:
        try:
            filename = os.path.basename(model_file)
            
            api.upload_file(
                path_or_fileobj=model_file,
                path_in_repo=f"{version}/{filename}",
                repo_id=repo_id,
                repo_type="dataset",
                token=HF_TOKEN
            )
            print(f"[+] 上傳成功: {filename}")
        except Exception as e:
            print(f"[!] 上傳失敗 {filename}: {e}")
    
    print(f"\n[✓] HuggingFace 上傳完成!")
    print(f"    查看: https://huggingface.co/datasets/{repo_id}/tree/main/{version}")

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*10 + "CPB Trading V3 Model Training")
    print(" "*15 + "One-Shot Colab Execution (100 EPOCHS)")
    print("="*80 + "\n")
    
    # 訓練配置
    TRAINING_COIN = 'BTCUSDT'
    DATA_LIMIT = 3500
    EPOCHS = 100
    
    # Step 1: 下載数据
    data = fetch_klines_for_training(TRAINING_COIN, limit=DATA_LIMIT)
    
    # Step 2: 前處理
    X, y, scaler = preprocess_data(data, seq_len=20)
    
    # Step 3: 構建模型
    model = build_v3_model(seq_len=20, input_features=4)
    
    # Step 4: 訓練模型 (100 epochs)
    model = train_v3_model(model, X, y, epochs=EPOCHS, batch_size=32)
    
    # Step 5: 評估模型
    evaluate_model(model, X, y)
    
    # Step 6: 準備模型文件
    model_files = prepare_models_for_upload(model, scaler)
    
    # Step 7: 上傳到 HuggingFace
    upload_to_huggingface(model_files, repo_id="zongowo111/cpb-models", version="v3")
    
    print("\n" + "="*80)
    print("[✓] V3 模型訓練和部署完成 (100 EPOCHS)!")
    print("="*80 + "\n")
