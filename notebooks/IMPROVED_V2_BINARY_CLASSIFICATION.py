# ============================================================================
# CPB v2: 改進版 - 二元分類 + 更好的特徵 + 强化Focal Loss
# 
# 關鍵改進次章：
# 1. 二元分類 (上升 vs 下降) - 變簡化問題
# 2. 新特徵：動量、加速度、波動率比、价格位置
# 3. 强化 Focal Loss (alpha=0.5, gamma=3.0)
# 4. 縮短 Lookback (36 -> 12 小時)
# 5. 改進的教师強制 (溋換了 SMOTE 簡化策略)
# ============================================================================

print('='*80)
print('CPB v2: Improved Binary Classification with Better Features')
print('='*80)

import subprocess
import sys
import os
import time
import warnings
from datetime import datetime, timedelta
import math

warnings.filterwarnings('ignore')

print('\n[STEP 0] 安裝依賴...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'pandas', 'numpy', 'scikit-learn', 'requests', 'matplotlib', 'imbalanced-learn'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sample import SMOTE

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print('✓ 所有庫導入完成')

# ============================================================================
# 配置
# ============================================================================

print('\n[STEP 1] 配置參數...')

CONFIG = {
    'coins': [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT',
        'ADAUSDT', 'DOGEUSDT', 'LINKUSDT', 'XRPUSDT', 'LTCUSDT'
    ],
    'timeframes': ['1h'],
    'epochs': 50,
    'batch_size': 32,  # 減少到 32
    'learning_rate': 5e-4,  # 減低學習率
    'lookback': 12,  # 減少到 12 小時
    'n_features': 12,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'warmup_epochs': 3,  # 減少
    'early_stop_patience': 15,  # 減少 patience
    'focal_alpha': 0.5,  # 强化 Focal Loss
    'focal_gamma': 3.0,
}

if torch.cuda.is_available():
    device = 'cuda'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'
    print('警告：使用 CPU 訓練會很慢')

print(f'\n配置:')
print(f'  幣種: {len(CONFIG["coins"])} 個')
print(f'  任務: 二元分類 (上升 vs 下降)')
print(f'  Lookback: {CONFIG["lookback"]} 小時')
print(f'  Focal Loss: alpha={CONFIG["focal_alpha"]}, gamma={CONFIG["focal_gamma"]}')

# ============================================================================
# 改進的焦點損失
# ============================================================================

class ImprovedFocalLoss(nn.Module):
    "改進的焦點損失—伸縮鯨重（齐次方磊）"
    def __init__(self, alpha=0.5, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # inputs: (batch_size, 2) - logits for [Down, Up]
        # targets: (batch_size,) - 0 or 1
        
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce)
        
        # 焦點損失：强化效惨齐轃艶
        focal_loss = self.alpha * ((1 - p) ** self.gamma) * ce
        
        return focal_loss.mean()

# ============================================================================
# 數據收集與預處理
# ============================================================================

class BinanceDataCollector:
    BASE_URL = "https://api.binance.us/api/v3"
    MAX_CANDLES = 1000
    
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
        retry_count = 0
        
        while current_start < end_time and len(all_klines) < limit:
            try:
                params = {
                    "symbol": symbol, "interval": interval,
                    "startTime": current_start,
                    "limit": min(self.MAX_CANDLES, limit - len(all_klines))
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
                retry_count = 0
                time.sleep(0.3)
            except Exception as e:
                retry_count += 1
                if retry_count >= 3:
                    break
                time.sleep(2 ** retry_count)
        
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

class ImprovedFeatureEngineer:
    "改進的特徵工程—添加動量、加速度、波動率比"—
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        # === 基本特徵 ===
        df['returns'] = df['close'].pct_change()
        
        # === 動量特徵 (新) ===
        # 動量: 5時間之前的价格變化
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # 加速度: 動量的動量
        df['acceleration'] = df['momentum_5'] - df['momentum_5'].shift(1)
        
        # === SMA 等探訊指標 ===
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # SMA 比率
        df['sma_ratio'] = df['sma_5'] / (df['sma_20'] + 1e-10)
        
        # === RSI (相對強弱指數) ===
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50).clip(0, 100)
        
        # === 波動率特徵 (新) ===
        # 現在的波動率
        df['volatility'] = df['returns'].rolling(20).std() * 100
        df['volatility'] = df['volatility'].clip(0, 10)
        
        # 波動率比（輄迷的波動率）
        df['volatility_ratio'] = df['volatility'] / (df['volatility'].rolling(50).mean() + 1e-10)
        df['volatility_ratio'] = df['volatility_ratio'].clip(0, 5)
        
        # === 价格位置特徵 (新) ===
        # 價格位置：今天的價格位於今天高低之間
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['price_position'] = df['price_position'].clip(0, 1)
        
        # === 成交量特徵 ===
        df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        df['volume_ma_ratio'] = df['volume_ma_ratio'].clip(0.1, 10)
        
        # === 趨勢指標 ===
        df['trend'] = (df['close'] > df['sma_20']).astype(int)
        
        # 清理
        self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        for col in self.df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].clip(-100, 100)
        
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude]

class BinaryClassificationPreprocessor:
    "二元分類數據預處理"—
    def __init__(self, df, lookback=12):
        self.df = df.copy()
        self.lookback = lookback
        self.scaler = MinMaxScaler((0, 1))
    
    def prepare(self, feature_cols):
        self.df = self.df.dropna()
        feature_data = self.df[feature_cols].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        self.df = self.df.loc[feature_data.index]
        
        feature_data = self.scaler.fit_transform(feature_data)
        self.features = feature_data
        self.feature_cols = feature_cols
        
        return feature_data, feature_cols
    
    def create_sequences(self):
        X, y = [], []
        data = self.features
        close_prices = self.df['close'].values
        
        for i in range(self.lookback, len(data) - 1):
            X.append(data[i - self.lookback:i])
            
            # 二元分類: Up (1) vs Down (0)
            price_change = (close_prices[i + 1] - close_prices[i]) / close_prices[i]
            label = 1 if price_change > 0 else 0
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, train_ratio=0.7):
        n = len(X)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + 0.15))
        
        return {
            'X_train': X[:train_idx], 'y_train': y[:train_idx],
            'X_val': X[train_idx:val_idx], 'y_val': y[train_idx:val_idx],
            'X_test': X[val_idx:], 'y_test': y[val_idx:]
        }

# ============================================================================
# 改進的 LSTM 模型
# ============================================================================

class ImprovedLSTMModel(nn.Module):
    "擅新的 LSTM 模型"
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.bn_input = nn.BatchNorm1d(input_size)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.bn_hidden = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)  # 二元分類
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        x_flat = x.reshape(-1, features)
        x_flat = self.bn_input(x_flat)
        x = x_flat.reshape(batch_size, seq_len, features)
        
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        out = self.fc1(last_out)
        out = self.bn_hidden(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# 學習率調度器
# ============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

# ============================================================================
# 訓練器
# ============================================================================

class BinaryClassificationTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, X_train, y_train, X_val, y_val, config):
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        # SMOTE 過ኋ換解決類不平衡 (有時候)
        try:
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
            smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train == 0) - 1, np.sum(y_train == 1) - 1))
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
            X_train_resampled = X_train_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
            X_train_t = torch.FloatTensor(X_train_resampled).to(self.device)
            y_train_t = torch.LongTensor(y_train_resampled).to(self.device)
            print(f'    SMOTE: {len(y_train)} -> {len(y_train_resampled)} samples')
        except:
            print(f'    SMOTE skipped (not enough samples)')
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        lr_scheduler = WarmupCosineScheduler(
            optimizer,
            config['warmup_epochs'],
            config['epochs'],
            config['learning_rate']
        )
        
        criterion = ImprovedFocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(config['epochs']):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config['gradient_clip']
                )
                
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            self.model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
                    
                    preds = output.argmax(dim=1)
                    val_acc += (preds == y_batch).sum().item()
            
            val_loss /= len(val_loader)
            val_acc /= len(y_val)
            
            current_lr = lr_scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1:3d}/{config["epochs"]}: Train={train_loss:.6f}, Val={val_loss:.6f}, Acc={val_acc:.4f}, LR={current_lr:.6f}')
            
            if val_loss < best_val_loss - 0.0001:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= config['early_stop_patience']:
                    print(f'    Early Stop at epoch {epoch+1}')
                    if best_weights:
                        self.model.load_state_dict(best_weights)
                    break
        
        return {'best_val_loss': float(best_val_loss), 'epochs': epoch+1}

# ============================================================================
# 可視化函數
# ============================================================================

def plot_binary_results(coin, y_true, y_pred, y_prob):
    "策畫二元分類結果""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{coin} - Binary Classification Results (Up vs Down)', fontsize=14, fontweight='bold')
    
    # 1. 真實 vs 預測
    ax = axes[0, 0]
    x_pos = np.arange(len(y_true[-100:]))
    ax.scatter(x_pos[y_true[-100:] == 0], np.zeros(np.sum(y_true[-100:] == 0)), c='red', s=100, alpha=0.6, label='True Down')
    ax.scatter(x_pos[y_true[-100:] == 1], np.ones(np.sum(y_true[-100:] == 1)), c='green', s=100, alpha=0.6, label='True Up')
    ax.scatter(x_pos[y_pred[-100:] == 0], np.zeros(np.sum(y_pred[-100:] == 0)) + 0.15, c='red', s=50, marker='x', linewidth=3, label='Pred Down')
    ax.scatter(x_pos[y_pred[-100:] == 1], np.ones(np.sum(y_pred[-100:] == 1)) + 0.15, c='green', s=50, marker='x', linewidth=3, label='Pred Up')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Down', 'Up'])
    ax.set_xlabel('Time Steps (Last 100)')
    ax.set_title('True vs Predicted Classes')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # 2. 混淆矩陣
    ax = axes[0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(2), yticks=np.arange(2),
           xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                   color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    
    # 3. 沼篇性能指標
    ax = axes[1, 0]
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    bars = ax.bar(metrics, values, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Metrics')
    ax.grid(alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. ROC 曲線
    ax = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc = roc_auc_score(y_true, y_prob[:, 1])
    ax.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})', linewidth=2, color='blue')
    ax.plot([0, 1], [0, 1], 'r--', label='Random', linewidth=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{coin}_binary_classification_results.png', dpi=100, bbox_inches='tight')
    plt.show()

# ============================================================================
# 主程式
# ============================================================================

print('\n[STEP 2] 定義類...')
print('✓ 所有類定義完成')

print('\n[STEP 3] 下載數據...')

all_data = {}
collector = BinanceDataCollector()

for coin in CONFIG['coins']:
    try:
        print(f'  下載 {coin}...', end=' ', flush=True)
        df = collector.get_klines(coin, '1h', limit=3000)
        if BinanceDataCollector.validate(df):
            all_data[coin] = df
            print(f'✓ {len(df)} candles')
        else:
            print(f'✗ validation failed')
    except Exception as e:
        print(f'✗ {str(e)[:40]}')

print(f'\n✓ 下載完成: {len(all_data)}/{len(CONFIG["coins"])} 幣種')

if len(all_data) == 0:
    print('\n錯誤：沒有下載任何數據')
    exit(1)

print('\n[STEP 4] 訓練模型...')

trained_models = {}
results = []

for coin in all_data:
    print(f'\n  {coin}')
    print('  ' + '-'*50)
    
    try:
        df = all_data[coin]
        
        fe = ImprovedFeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        print(f'    Features: {len(feature_cols)} (新特徵已添加)')
        
        prep = BinaryClassificationPreprocessor(
            df_features,
            lookback=CONFIG['lookback']
        )
        features, feature_cols = prep.prepare(feature_cols)
        X, y = prep.create_sequences()
        data = prep.split_data(X, y)
        
        if len(X) < 200:
            print(f'    ✗ 序列太短')
            continue
        
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {f'Class {u}': c for u, c in zip(unique, counts)}
        print(f'    Distribution: {class_dist}')
        
        model = ImprovedLSTMModel(
            input_size=features.shape[-1],
            hidden_size=64,
            num_layers=2,
            dropout=0.3
        )
        print(f'    Params: {model.count_params():,}')
        
        trainer = BinaryClassificationTrainer(model, device=device)
        history = trainer.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            CONFIG
        )
        
        model.eval()
        X_test_t = torch.FloatTensor(data['X_test']).to(device)
        y_test_t = torch.LongTensor(data['y_test']).to(device)
        
        with torch.no_grad():
            logits = model(X_test_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = logits.argmax(dim=1).cpu().numpy()
        
        y_test = data['y_test']
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, probs[:, 1])
        
        trained_models[coin] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': probs,
            'y_test': y_test
        }
        
        results.append({
            'coin': coin,
            'loss': history['best_val_loss'],
            'epochs': history['epochs'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        })
        
        print(f'    Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}')
        print(f'    ✓ 訓練完成')
        
    except Exception as e:
        print(f'    ✗ 錯誤: {str(e)[:60]}')

print(f'\n✓ 訓練完成: {len(trained_models)} 個模型')

# ============================================================================
# 可視化結果
# ============================================================================

print('\n[STEP 5] 繪製結果...')

for coin in trained_models:
    try:
        y_true = trained_models[coin]['y_test']
        y_pred = trained_models[coin]['y_pred']
        y_prob = trained_models[coin]['y_prob']
        plot_binary_results(coin, y_true, y_pred, y_prob)
        print(f'  ✓ {coin}: 已保存圖表')
    except Exception as e:
        print(f'  ✗ {coin}: {str(e)[:40]}')

# ============================================================================
# 結果彙總
# ============================================================================

print('\n' + '='*80)
print('[STEP 6] 結果彙總')
print('='*80)

if results:
    print('\n{:12s} {:>10s} {:>8s} {:>10s} {:>10s} {:>8s} {:>8s}'.format(
        'Coin', 'Loss', 'Epochs', 'Accuracy', 'F1', 'Recall', 'AUC'))
    print('-'*90)
    
    for r in sorted(results, key=lambda x: x['f1'], reverse=True):
        print('{:12s} {:>10.6f} {:>8d} {:>10.4f} {:>10.4f} {:>8.4f} {:>8.4f}'.format(
            r['coin'], r['loss'], r['epochs'], r['accuracy'], r['f1'], r['recall'], r['auc']))
    
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_auc = np.mean([r['auc'] for r in results])
        print('-'*90)
        print(f'\n平均 Accuracy: {avg_acc:.4f}')
        print(f'平均 F1 Score: {avg_f1:.4f}')
        print(f'平均 AUC: {avg_auc:.4f}')
        print(f'\n成功訓練: {len(trained_models)}/{len(all_data)} 幣種')
        print(f'\n推誋：若 Accuracy > 0.55 穆住誋显著改進')

print('\n' + '='*80)
print(f'✓ 所有流程完成！已產生{len(trained_models)}張結果圖表')
print('='*80)
