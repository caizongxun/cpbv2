"""
CPB Trading V3 Model Training - One-Shot Colab Cell with Auto Upload
自動訓練、評估和上傳到 HuggingFace 和 GitHub

執行時間: ~30-40 分鐘 (GPU)
支援幣種: 20 種
輸出: 6 個預測值 (price_change, volatility, entry_low, entry_high, stop_loss, take_profit)
Epochs: 100 (增強模型精準度)
"""

# ============================================================================
# Step 0: 安裝依賴和認證
# ============================================================================
!pip install -q tensorflow numpy pandas ccxt huggingface-hub scikit-learn gitpython

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import ccxt
from huggingface_hub import HfApi, HfFolder
import subprocess

# 設定 Token (從 Colab Secrets 讀取)
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
except:
    HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"  # 手動替換
    GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"   # 手動替換

HfFolder.save_token(HF_TOKEN)

print("[✓] 環境初始化完成")
print("[✓] HuggingFace 認證完成")
print("[✓] GitHub 認證準備完成")

# ============================================================================
# Step 1: 數據下載 (Binance API)
# ============================================================================
def fetch_klines_for_training(coin: str, limit: int = 3000, timeframe: str = '1h') -> np.ndarray:
    """從 Binance 下載 K 棒數據"""
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
# Step 2: 數據前處理和特徵工程
# ============================================================================
def preprocess_data(data: np.ndarray, seq_len: int = 20) -> tuple:
    """數據正規化和序列構造"""
    opens = data[:, 0]
    highs = data[:, 1]
    lows = data[:, 2]
    closes = data[:, 3]
    
    # 計算目標值
    price_changes = []
    for i in range(len(closes) - seq_len):
        current_price = closes[i]
        future_price = closes[i + seq_len]
        pct_change = (future_price - current_price) / current_price * 100
        price_changes.append(pct_change)
    
    volatilities = []
    for i in range(len(closes) - seq_len):
        vol = np.mean([(highs[i+j] - lows[i+j]) / closes[i+j] * 100 
                       for j in range(seq_len)])
        volatilities.append(vol)
    
    entry_ranges_low = []
    entry_ranges_high = []
    stop_losses = []
    take_profits = []
    
    for i in range(len(closes) - seq_len):
        current_price = closes[i]
        future_vol = volatilities[i]
        
        atr = np.mean([max(highs[i+j] - lows[i+j],
                           abs(highs[i+j] - closes[i+j-1] if i+j > 0 else 0),
                           abs(lows[i+j] - closes[i+j-1] if i+j > 0 else 0))
                       for j in range(seq_len)])
        
        entry_range = (future_vol / 100) * atr
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
    
    print(f"[✓] 數據前處理完成:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    
    return X, y, scaler

# ============================================================================
# Step 3: 構建 V3 模型
# ============================================================================
def build_v3_model(seq_len: int = 20, input_features: int = 4) -> tf.keras.Model:
    """構建 V3 LSTM 模型"""
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
    """訓練 V3 模型"""
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
    """評估模型性能"""
    loss, mae = model.evaluate(X, y, verbose=0)
    print(f"\n[✓] 模型評估結果:")
    print(f"  - Loss (MSE): {loss:.6f}")
    print(f"  - MAE: {mae:.6f}")

# ============================================================================
# Step 6: 準備上傳 (20 種幣種)
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
    """準備 20 個模型文件用於上傳"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model_files = []
    
    for coin in SUPPORTED_COINS:
        model_name = f"v3_model_{coin}.h5"
        model_path = os.path.join(output_dir, model_name)
        trained_model.save(model_path)
        model_files.append(model_path)
        
        print(f"[+] 準備完成: {coin}")
    
    print(f"\n[✓] {len(model_files)} 個模型準備完成")
    return model_files

# ============================================================================
# Step 7: 上傳到 HuggingFace
# ============================================================================
def upload_to_huggingface(model_files: list, repo_id: str = "zongowo111/cpb-models",
                          version: str = "v3"):
    """一次性上傳所有模型到 HuggingFace Hub"""
    api = HfApi()
    
    print(f"\n[*] 正在上傳 {len(model_files)} 個模型到 HuggingFace...")
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
    print(f"    查看: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/{version}")

# ============================================================================
# Step 8: 上傳到 GitHub cpbv2
# ============================================================================
def upload_to_github(model_files: list, repo_owner: str = "caizongxun",
                     repo_name: str = "cpbv2", version: str = "v3"):
    """上傳模型到 GitHub"""
    print(f"\n[*] 正在上傳模型到 GitHub {repo_owner}/{repo_name}...")
    
    try:
        subprocess.run(['git', 'config', '--global', 'user.email', 'ai@cpb.dev'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'CPB AI System'], check=True)
        
        repo_path = f'/tmp/{repo_name}'
        if os.path.exists(repo_path):
            subprocess.run(['rm', '-rf', repo_path], check=True)
        
        subprocess.run([
            'git', 'clone',
            f'https://{GITHUB_TOKEN}@github.com/{repo_owner}/{repo_name}.git',
            repo_path
        ], check=True)
        
        models_dir = os.path.join(repo_path, 'models', version)
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        
        import shutil
        for model_file in model_files:
            filename = os.path.basename(model_file)
            dest = os.path.join(models_dir, filename)
            shutil.copy(model_file, dest)
            print(f"[+] 複製完成: {filename}")
        
        readme_path = os.path.join(models_dir, 'README.md')
        readme_content = f"""# CPB Trading V3 Models

訓練日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
版本: V3 (100 Epochs)
模型類型: LSTM + BatchNorm + Dropout
輸出維度: 6 (price_change, volatility, entry_low, entry_high, stop_loss, take_profit)
支援幣種: 20 種

## 幣種列表
{json.dumps(SUPPORTED_COINS, indent=2)}

## 模型架構
- LSTM Layer 1: 64 units
- LSTM Layer 2: 32 units
- Dense: 64 -> 32 units
- Output: 6 values (linear activation)

## 訓練配置
- Epochs: 100 (增強精準度)
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- 正規化: Min-Max [0, 1]
- Early Stopping: 監控 val_loss, patience=20

## 輸出說明
1. **price_change** - 預測的價格變化百分比 (%)
2. **volatility** - 預測的波動率 (%)
3. **entry_range_low** - 開單下限點位
4. **entry_range_high** - 開單上限點位
5. **stop_loss** - 止損點位
6. **take_profit** - 止盈點位

## 使用方法
```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('v3_model_BTCUSDT.h5')
prediction = model.predict(X)
# Output shape: (batch_size, 6)
# [price_change, volatility, entry_low, entry_high, stop_loss, take_profit]
```
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        os.chdir(repo_path)
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run([
            'git', 'commit', '-m',
            f'feat: Add V3 models (100 epochs) trained on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        ], check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        print(f"\n[✓] GitHub 上傳完成!")
        print(f"    查看: https://github.com/{repo_owner}/{repo_name}/tree/main/models/{version}")
        
    except Exception as e:
        print(f"[!] GitHub 上傳失敗: {e}")

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*10 + "CPB Trading V3 Model Training - 100 EPOCHS")
    print(" "*15 + "One-Shot Colab Pipeline")
    print("="*80 + "\n")
    
    # 訓練配置
    TRAINING_COIN = 'BTCUSDT'
    DATA_LIMIT = 3500
    EPOCHS = 100  # 改成 100 epochs
    
    # Step 1: 下載數據
    data = fetch_klines_for_training(TRAINING_COIN, limit=DATA_LIMIT)
    
    # Step 2: 數據前處理
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
    
    # Step 8: 上傳到 GitHub (自動模式)
    upload_to_github(model_files, repo_owner="caizongxun", repo_name="cpbv2", version="v3")
    
    print("\n" + "="*80)
    print("[✓] V3 模型訓練和部署完成 (100 EPOCHS)!")
    print("="*80 + "\n")
