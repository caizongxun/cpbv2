# ============================================================================
# CPB v2: 完全重新設計 - 分類問題 + 焦點損失 + 注意力機制
# 
# 關鍵改進：
# 1. 從回歸改為分類（預測價格方向）
# 2. 使用焦點損失處理類不平衡
# 3. 添加自注意力機制捕捉關鍵時步
# 4. 改進數據採樣和目標定義
# 5. 多層防過擬合
# ============================================================================

print('='*80)
print('CPB v2: Classification with Attention & Focal Loss')
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
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'pandas', 'numpy', 'scikit-learn', 'requests', 'matplotlib'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    'batch_size': 64,
    'learning_rate': 1e-3,
    'lookback': 36,  # 36小時（更短，捕捉近期趨勢）
    'n_features': 12,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'warmup_epochs': 5,
    'early_stop_patience': 20,
    'price_change_threshold': 0.003,  # 0.3% 閾值用於分類
    'num_classes': 3,  # 上升、下降、保持
}

if torch.cuda.is_available():
    device = 'cuda'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'
    print('警告：使用 CPU 訓練會很慢')

print(f'\n配置:')
print(f'  幣種: {len(CONFIG["coins"])} 個')
print(f'  任務: 3分類（上升/下降/保持）')
print(f'  Lookback: {CONFIG["lookback"]} 小時')
print(f'  閾值: {CONFIG["price_change_threshold"]*100:.1f}%')
print(f'  設備: {device}')

# ============================================================================
# 焦點損失函數
# ============================================================================

class FocalLoss(nn.Module):
    """焦點損失：處理類不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce
        return focal_loss.mean()

# ============================================================================
# 自注意力層
# ============================================================================

class SelfAttention(nn.Module):
    """自注意力機制"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        Q = self.query(x)  # (batch, seq_len, hidden_size)
        K = self.key(x)
        V = self.value(x)
        
        # 注意力權重
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 加權輸出
        context = torch.bmm(attention_weights, V)  # (batch, seq_len, hidden_size)
        
        return context, attention_weights

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
        start_time = int((datetime.utcnow() - timedelta(days=120)).timestamp() * 1000)
        current_start = start_time
        retry_count = 0
        
        while current_start < end_time and len(all_klines) < limit:
            try:
                params = {
                    "symbol": symbol, "interval": interval,
                    "startTime": current_start,
                    "limit": min(self.MAX_CANDLES, limit - len(all_klines))
                }
                response = self.session.get(f"{self.BASE_URL}/klines", params=params, timeout=10)
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

class FeatureEngineer:
    """特徵工程"""
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        # 價格變化
        df['price_change_1h'] = df['close'].pct_change()
        df['price_change_24h'] = df['close'].pct_change(24)
        
        # SMA
        df['sma_12'] = df['close'].rolling(12).mean()
        df['sma_26'] = df['close'].rolling(26).mean()
        df['sma_ratio'] = df['sma_12'] / (df['sma_26'] + 1e-10)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50).clip(0, 100)
        
        # 成交量比
        df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        df['volume_ma_ratio'] = df['volume_ma_ratio'].clip(0.1, 10)
        
        # 波動率
        df['volatility'] = df['close'].pct_change().rolling(20).std() * 100
        df['volatility'] = df['volatility'].clip(0, 10)
        
        # 趨勢指標
        df['trend_1h'] = (df['close'] > df['sma_12']).astype(int)
        df['trend_24h'] = (df['close'] > df['sma_26']).astype(int)
        
        # 清理數據
        self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        for col in self.df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].clip(-100, 100)
        
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude]

class ClassificationDataPreprocessor:
    """分類任務的數據預處理"""
    def __init__(self, df, lookback=36, threshold=0.003):
        self.df = df.copy()
        self.lookback = lookback
        self.threshold = threshold
        self.scaler = MinMaxScaler((0, 1))
    
    def prepare(self, feature_cols):
        self.df = self.df.dropna()
        feature_data = self.df[feature_cols].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.dropna()
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
            
            # 定義目標：下一時步的價格變化方向
            price_change = (close_prices[i + 1] - close_prices[i]) / close_prices[i]
            
            if price_change > self.threshold:
                label = 1  # 上升
            elif price_change < -self.threshold:
                label = 0  # 下降
            else:
                label = 2  # 保持
            
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
# 改進的 LSTM 模型 + 注意力
# ============================================================================

class LSTMWithAttention(nn.Module):
    """LSTM + 自注意力機制用於分類"""
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2):
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
        
        # 自注意力
        self.attention = SelfAttention(hidden_size)
        
        # 分類頭
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # hidden_size + attention context
        self.bn_hidden = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Batch Norm 在輸入
        x_flat = x.reshape(-1, features)
        x_flat = self.bn_input(x_flat)
        x = x_flat.reshape(batch_size, seq_len, features)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 注意力
        attention_context, _ = self.attention(lstm_out)  # (batch, seq_len, hidden_size)
        
        # 取最後時步的 LSTM 輸出
        last_lstm = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 取注意力的平均
        attention_avg = attention_context.mean(dim=1)  # (batch, hidden_size)
        
        # 連接
        combined = torch.cat([last_lstm, attention_avg], dim=1)  # (batch, hidden_size*2)
        
        # 分類頭
        out = self.fc1(combined)
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

class ClassificationTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, X_train, y_train, X_val, y_val, config):
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        # 計算類權重（處理類不平衡）
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum()
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=config['batch_size'],
            sampler=sampler
        )
        
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # 優化器和損失
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
        
        # 焦點損失
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(config['epochs']):
            # 訓練
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
            
            # 驗證
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
            
            # 學習率
            current_lr = lr_scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1:3d}/{config["epochs"]}: Train={train_loss:.6f}, Val={val_loss:.6f}, Acc={val_acc:.4f}, LR={current_lr:.6f}')
            
            # 早停
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
# 主程式
# ============================================================================

print('\n[STEP 2] 定義類...')
print('✓ 所有類定義完成')

print('\n[STEP 3] 下載數據...')

all_data = {}
collector = BinanceDataCollector()

for coin in CONFIG['coins']:
    try:
        df = collector.get_klines(coin, '1h', limit=3000)
        if BinanceDataCollector.validate(df):
            all_data[coin] = df
            print(f'  ✓ {coin:10s}: {len(df):4d} candles')
        else:
            print(f'  ✗ {coin:10s}: validation failed')
    except Exception as e:
        print(f'  ✗ {coin:10s}: {str(e)[:40]}')

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
        
        # 特徵工程
        fe = FeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        print(f'    Features: {len(feature_cols)}')
        
        # 數據預處理
        prep = ClassificationDataPreprocessor(
            df_features,
            lookback=CONFIG['lookback'],
            threshold=CONFIG['price_change_threshold']
        )
        features, feature_cols = prep.prepare(feature_cols)
        X, y = prep.create_sequences()
        data = prep.split_data(X, y)
        
        if len(X) < 300:
            print(f'    ✗ 序列太短')
            continue
        
        # 類分佈
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {f'Class {u}': c for u, c in zip(unique, counts)}
        print(f'    Distribution: {class_dist}')
        
        # 構建模型
        model = LSTMWithAttention(
            input_size=features.shape[-1],
            hidden_size=64,
            num_layers=2,
            num_classes=CONFIG['num_classes'],
            dropout=0.2
        )
        print(f'    Params: {model.count_params():,}')
        
        # 訓練
        trainer = ClassificationTrainer(model, device=device)
        history = trainer.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            CONFIG
        )
        
        # 評估
        model.eval()
        X_test_t = torch.FloatTensor(data['X_test']).to(device)
        y_test_t = torch.LongTensor(data['y_test']).to(device)
        
        with torch.no_grad():
            logits = model(X_test_t)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        
        y_test = data['y_test']
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        trained_models[coin] = {
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        results.append({
            'coin': coin,
            'loss': history['best_val_loss'],
            'epochs': history['epochs'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f'    Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}')
        print(f'    ✓ 訓練完成')
        
    except Exception as e:
        print(f'    ✗ 錯誤: {str(e)[:60]}')

print(f'\n✓ 訓練完成: {len(trained_models)} 個模型')

# ============================================================================
# 結果
# ============================================================================

print('\n' + '='*80)
print('[STEP 5] 結果彙總')
print('='*80)

if results:
    print('\n{:12s} {:>10s} {:>8s} {:>10s} {:>10s} {:>8s}'.format(
        'Coin', 'Loss', 'Epochs', 'Accuracy', 'F1', 'Recall'))
    print('-'*80)
    
    for r in sorted(results, key=lambda x: x['f1'], reverse=True):
        print('{:12s} {:>10.6f} {:>8d} {:>10.4f} {:>10.4f} {:>8.4f}'.format(
            r['coin'], r['loss'], r['epochs'], r['accuracy'], r['f1'], r['recall']))
    
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        print('-'*80)
        print(f'\n平均 Accuracy: {avg_acc:.4f}')
        print(f'平均 F1 Score: {avg_f1:.4f}')
        print(f'\n成功訓練: {len(trained_models)}/{len(all_data)} 幣種')

print('\n' + '='*80)
print('✓ 所有流程完成！')
print('='*80)
