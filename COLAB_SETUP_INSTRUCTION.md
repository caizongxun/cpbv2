# Colab 快速開始 - CPB V4 三個 Cell 直接複製

## 步驟

1. 打開: https://colab.research.google.com
2. 新建筆記本
3. 下面三個 Cell **複製貼上** 按順序執行

---

## Cell 1: 安裝依賴

把下面代碼複製到 Colab 第一個 Cell，執行：

```
import os
import sys

print("[*] 安裝 PyTorch (GPU support)...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

print("[*] 安裝其他依賴...")
!pip install -q numpy pandas requests scikit-learn matplotlib ccxt huggingface-hub --upgrade

print("[✓] 所有依賴安裝完成\n")
print("[✓] 可用 GPU:")
import torch
if torch.cuda.is_available():
    print(f"    {torch.cuda.get_device_name(0)}")
else:
    print("    未檢測到 GPU，將使用 CPU（較慢）")
```

等待完成（約 2-3 分鐘）

---

## Cell 2: 配置參數

把下面代碼複製到第二個 Cell，**修改你的 token 和選擇幣種**，執行：

```
# === 必須配置 ===
HF_TOKEN = "hf_YOUR_TOKEN_HERE"        # 從 https://huggingface.co/settings/tokens 獲取
TRAINING_COIN = "BTCUSDT"              # 訓練幣種: BTCUSDT, ETHUSDT, SOLUSDT 等
EPOCHS = 80                            # 訓練輪次

# === 可選配置 ===
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

print("[✓] 配置完成:")
print(f"    訓練幣種: {TRAINING_COIN}")
print(f"    Epochs: {EPOCHS}")
print(f"    K 棒數量: {DATA_LIMIT}")
print(f"    模型: CNN-LSTM Hybrid")
```

---

## Cell 3: 完整訓練流程

這是核心 Cell，複製下面全部代碼到第三個 Cell，執行：

```python
import warnings
import time
import math
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

warnings.filterwarnings('ignore')

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[✓] 使用設備: {device}\n")

# ==================== PART 1: 進階特徵工程 ====================
print("[*] PART 1: 構建進階特徵工程類...")

class AdvancedFeatureEngineer:
    """20+ 技術指標特徵工程"""
    
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        # 價格特徵
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 動量指標
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        df['acceleration'] = df['momentum_5'] - df['momentum_5'].shift(1)
        
        # 移動平均線
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['sma_ratio'] = df['sma_5'] / (df['sma_20'] + 1e-10)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50).clip(0, 100) / 100
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_normalized'] = df['atr_14'] / (df['close'] + 1e-10) * 100
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # 價格位置
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['price_position'] = df['price_position'].clip(0, 1)
        
        # 波動率
        df['volatility'] = df['returns'].rolling(20).std() * 100
        df['volatility'] = df['volatility'].clip(0, 10)
        df['volatility_ratio'] = df['volatility'] / (df['volatility'].rolling(50).mean() + 1e-10)
        df['volatility_ratio'] = df['volatility_ratio'].clip(0, 5)
        
        # 成交量
        df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        df['volume_ma_ratio'] = df['volume_ma_ratio'].clip(0.1, 10)
        df['volume_trend'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).mean() + 1e-10)
        
        # 趨勢
        df['trend'] = (df['close'] > df['sma_20']).astype(int)
        
        # 清理
        self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
        for col in self.df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].clip(-100, 100)
        
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'tr']
        return [col for col in self.df.columns if col not in exclude]

print("[✓] OK\n")

# ==================== PART 2: CNN-LSTM 模型 ====================
print("[*] PART 2: 構建 CNN-LSTM 混合模型...")

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

# ==================== PART 3: Focal Loss ====================
print("[*] PART 3: 構建 Focal Loss...")

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

# ==================== PART 4: 開單位置計算器 ====================
print("[*] PART 4: 構建開單位置計算器...")

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

# ==================== PART 5: 下載數據 ====================
print("[*] PART 5: 正在下載數據...\n")

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
                print(f"  [!] 錯誤: {str(e)[:50]}")
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
    print(f"[!] 數據驗證失敗")
    exit(1)

print(f"[✓] 下載完成: {len(df)} 根 K 棒\n")

# ==================== PART 6: 特徵工程 ====================
print("[*] PART 6: 計算特徵...")

fe = AdvancedFeatureEngineer(df)
df_feat = fe.calculate_all()
feature_cols = fe.get_features()

print(f"[✓] 特徵數: {len(feature_cols)}\n")

# ==================== PART 7: 數據預處理 ====================
print("[*] PART 7: 數據預處理...")

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
    print(f"[!] 數據不足 ({len(X)} < 100)")
    exit(1)

print(f"[✓] X shape: {X.shape}")
print(f"[✓] y shape: {y.shape}\n")

# ==================== PART 8: 訓練/驗證分割 ====================
print("[*] PART 8: 分割訓練/驗證集...")

split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
y_train = y[:split_idx]
X_val = X[split_idx:]
y_val = y[split_idx:]

print(f"[✓] Train: {len(X_train)}, Val: {len(X_val)}\n")

# ==================== PART 9: 構建模型 ====================
print("[*] PART 9: 構建模型...")

model = CNNLSTMModel(input_size=len(feature_cols), cnn_filters=32, lstm_hidden=64, dropout=0.3)
model = model.to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[✓] 參數數: {params:,}\n")

# ==================== PART 10: 訓練 ====================
print("[*] PART 10: 訓練模型...\n")

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
    # Train
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
    
    # Val
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

print(f"[✓] 訓練完成\n")

# ==================== PART 11: 評估 ====================
print("[*] PART 11: 評估模型...")

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

# ==================== PART 12: 開單位置計算 ====================
print("[*] PART 12: 計算開單位置建議...\n")

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
print(f"幣種: {TRAINING_COIN}")
print(f"方向: {'LONG ↑' if pred_class == 1 else 'SHORT ↓'} (機率 {pred_prob:.2%})")
print(f"現價: {last_close:,.2f}")
print(f"進場範圍: {entry_calc['entry_low']:,.2f} ~ {entry_calc['entry_high']:,.2f}")
print(f"止損位置: {entry_calc['stop_loss']:,.2f}")
print(f"獲利位置: {entry_calc['take_profit']:,.2f}")
print(f"風險/根本: {(last_close - entry_calc['stop_loss'])/(entry_calc['take_profit'] - last_close):.2f}:1")
print(f"建議倉位倍率: {entry_calc['position_multiplier']:.2f}x\n")

# ==================== PART 13: 上傳到 HuggingFace ====================
print("[*] PART 13: 上傳模型到 HuggingFace...\n")

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

print("[+] 上傳 scaler...")
try:
    api.upload_file(
        path_or_fileobj=str(scaler_path),
        path_in_repo=f"{HF_VERSION}/scaler_params.json",
        repo_id=HF_REPO,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("    [✓] scaler 上傳成功")
except Exception as e:
    print(f"    [!] scaler 上傳失敗: {e}")

model_name = f"v4_model_{TRAINING_COIN}.pt"
model_path = temp_dir / model_name
torch.save(model.state_dict(), str(model_path))

print(f"[+] 上傳模型 ({TRAINING_COIN})...")
try:
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=f"{HF_VERSION}/{model_name}",
        repo_id=HF_REPO,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f"    [✓] {TRAINING_COIN} 上傳成功")
except Exception as e:
    print(f"    [!] {TRAINING_COIN} 上傳失敗: {e}")

print("\n" + "="*80)
print("[✓] V4 訓練完成!")
print("="*80)
print(f"查看模型: https://huggingface.co/datasets/{HF_REPO}/tree/main/{HF_VERSION}")
print(f"性能: Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
print(f"開單信號已生成 (信心: {pred_prob:.2%})")
print("="*80)
```

---

## ✓ 完成！

執行完三個 Cell 後會自動：
1. ✓ 下載最新 K 棒數據
2. ✓ 計算 22+ 項技術指標
3. ✓ 訓練 CNN-LSTM 模型
4. ✓ 生成開單信號（進場位置、止損、獲利）
5. ✓ 自動上傳到 HuggingFace

模型會保存在: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/v4

---

**時間**: GPU (T4) 約 20-30 分鐘，GPU (V100) 約 10-15 分鐘
