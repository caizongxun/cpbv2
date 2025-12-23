# ============================================================================
# CPB v2: 基於大規模研究的完整訓練邏輯改進版
# 採用最新深度學習最佳實踐
# - Warmup + Cosine Annealing學習率調度
# - 更好的特徵工程與數據清理
# - Batch Normalization應用
# - Gradient Clipping
# - 多層防過擬合
# ============================================================================

print('='*80)
print('CPB v2: Research-Based Training Optimization')
print('='*80)

import subprocess
import sys
import os
import json
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print('✓ 所有库导入完成')

# ============================================================================
# 配置
# ============================================================================

print('\n[STEP 1] 配置参数...')

CONFIG = {
    'coins': [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT',
        'ADAUSDT', 'DOGEUSDT', 'LINKUSDT', 'XRPUSDT', 'LTCUSDT'
    ],
    'timeframes': ['1h'],
    'epochs': 50,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'lookback': 72,  # 72小時
    'n_features': 12,  # 簡化特徵
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'warmup_epochs': 5,  # Warmup持續5個epoch
    'early_stop_patience': 20
}

if torch.cuda.is_available():
    device = 'cuda'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'
    print('警告：使用 CPU 訓練會很慢')

print(f'\n配置:')
print(f'  幣種: {len(CONFIG["coins"])} 個')
print(f'  Epochs: {CONFIG["epochs"]}')
print(f'  Batch Size: {CONFIG["batch_size"]}')
print(f'  Learning Rate: {CONFIG["learning_rate"]}')
print(f'  Lookback: {CONFIG["lookback"]} 小時')
print(f'  設備: {device}')

# ============================================================================
# 工具函數
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

class OptimizedFeatureEngineer:
    """改進的特徵工程：簡化且穩定"""
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        try:
            # 價格變化率（平滑且安全）
            df['price_change_1h'] = df['close'].pct_change()
            df['price_change_24h'] = df['close'].pct_change(24)
            df['price_change_7d'] = df['close'].pct_change(168)
            
            # SMA (簡單移動平均)
            df['sma_12'] = df['close'].rolling(12).mean()
            df['sma_26'] = df['close'].rolling(26).mean()
            df['sma_ratio'] = df['sma_12'] / (df['sma_26'] + 1e-10)
            
            # RSI (相對強弱指數) - 安全計算
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi_14'] = 100 - (100 / (1 + rs))
            df['rsi_14'] = df['rsi_14'].fillna(50).clip(0, 100)  # 限制在 0-100
            
            # 成交量趨勢
            df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
            df['volume_ma_ratio'] = df['volume_ma_ratio'].clip(0.1, 10)  # 限制異常值
            
            # 波動率 (簡化版)
            df['volatility'] = df['close'].pct_change().rolling(20).std() * 100
            df['volatility'] = df['volatility'].clip(0, 10)  # 限制極端波動
            
            # 清理數據
            self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
            
            # 確保所有值都在合理範圍內
            for col in self.df.columns:
                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    self.df[col] = self.df[col].clip(-100, 100)
            
            return self.df
        except Exception as e:
            print(f'特徵錯誤: {e}')
            raise
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude]

class RobustDataPreprocessor:
    """強化的數據預處理"""
    def __init__(self, df, lookback=72):
        self.df = df.copy()
        self.lookback = lookback
        self.scaler = MinMaxScaler((0, 1))
    
    def prepare(self, feature_cols):
        # 移除缺失值
        self.df = self.df.dropna()
        
        # 提取特徵
        feature_data = self.df[feature_cols].copy()
        
        # 雙重檢查：移除含 inf 或 nan 的行
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.dropna()
        self.df = self.df.loc[feature_data.index]
        
        # 歸一化（MinMax更適合時間序列）
        feature_data = self.scaler.fit_transform(feature_data)
        self.features = feature_data
        self.feature_cols = feature_cols
        
        return feature_data, feature_cols
    
    def create_sequences(self):
        X, y = [], []
        data = self.features
        
        # 使用 close 價格變化作為目標
        close_prices = self.df['close'].values
        price_changes = np.diff(close_prices) / (close_prices[:-1] + 1e-10)
        
        for i in range(self.lookback, len(data)):
            if i - 1 < len(price_changes):
                X.append(data[i - self.lookback:i])
                y.append(price_changes[i - 1])  # 下一時段的價格變化
        
        return np.array(X), np.array(y).reshape(-1, 1)
    
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
    """應用最佳實踐的 LSTM 模型"""
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Batch Norm 在輸入層
        self.bn_input = nn.BatchNorm1d(input_size)
        
        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 全連接層
        self.fc1 = nn.Linear(hidden_size, 32)
        self.bn_hidden = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Batch Norm 在輸入（reshape 為 (batch*seq, features)）
        x_flat = x.reshape(-1, features)
        x_flat = self.bn_input(x_flat)
        x = x_flat.reshape(batch_size, seq_len, features)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # 取最後時步
        
        # 全連接
        out = self.fc1(last_out)
        out = self.bn_hidden(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# 學習率調度器：Warmup + Cosine Annealing
# ============================================================================

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 學習率調度"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup: 線性增長
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine Annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

# ============================================================================
# 訓練器
# ============================================================================

class AdvancedTrainer:
    """應用所有最佳實踐的訓練器"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, X_train, y_train, X_val, y_val, config):
        # 準備數據
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=config['batch_size'],
            shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # 優化器：AdamW（包含權重衰減）
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 學習率調度器
        lr_scheduler = WarmupCosineScheduler(
            optimizer,
            config['warmup_epochs'],
            config['epochs'],
            config['learning_rate']
        )
        
        criterion = nn.MSELoss()
        
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
                
                # 梯度裁剪：防止梯度爆炸
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
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            # 學習率調度
            current_lr = lr_scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1:3d}/{config["epochs"]}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.6f}')
            
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
        fe = OptimizedFeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        print(f'    Features: {len(feature_cols)}')
        
        # 數據預處理
        prep = RobustDataPreprocessor(df_features, lookback=CONFIG['lookback'])
        features, feature_cols = prep.prepare(feature_cols)
        X, y = prep.create_sequences()
        data = prep.split_data(X, y)
        
        if len(X) < 300:
            print(f'    ✗ 序列太短')
            continue
        
        # 構建模型
        model = ImprovedLSTMModel(
            input_size=features.shape[-1],
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        print(f'    Params: {model.count_params():,}')
        
        # 訓練
        trainer = AdvancedTrainer(model, device=device)
        history = trainer.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            CONFIG
        )
        
        # 評估
        model.eval()
        X_test_t = torch.FloatTensor(data['X_test']).to(device)
        with torch.no_grad():
            y_pred = model(X_test_t).cpu().numpy()
        
        y_test = data['y_test']
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        trained_models[coin] = {
            'model': model,
            'y_pred': y_pred.flatten(),
            'y_test': y_test.flatten()
        }
        
        results.append({
            'coin': coin,
            'loss': history['best_val_loss'],
            'epochs': history['epochs'],
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
        
        print(f'    MSE={mse:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}')
        print(f'    ✓ 訓練完成')
        
    except Exception as e:
        print(f'    ✗ 錯誤: {str(e)[:60]}')

print(f'\n✓ 訓練完成: {len(trained_models)} 個模型')

print('\n[STEP 5] 繪製預測圖...')

for coin in trained_models:
    try:
        y_true = trained_models[coin]['y_test']
        y_pred = trained_models[coin]['y_pred']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'{coin} - Price Change Prediction', fontsize=12, fontweight='bold')
        
        ax = axes[0, 0]
        ax.plot(y_true[-100:], label='True', linewidth=2, color='blue', alpha=0.8)
        ax.plot(y_pred[-100:], label='Predicted', linewidth=2, color='red', alpha=0.8)
        ax.set_title('Last 100 Timesteps')
        ax.set_ylabel('Price Change')
        ax.legend()
        ax.grid(alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(y_true, label='True', linewidth=1, color='blue', alpha=0.6)
        ax.plot(y_pred, label='Predicted', linewidth=1, color='red', alpha=0.6)
        ax.set_title('Full Sequence')
        ax.set_ylabel('Price Change')
        ax.legend()
        ax.grid(alpha=0.3)
        
        residuals = y_true - y_pred
        ax = axes[1, 0]
        ax.hist(residuals, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.set_title('Residuals Distribution')
        ax.set_xlabel('Residuals')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.grid(alpha=0.3)
        
        ax = axes[1, 1]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='purple')
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('True vs Predicted')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{coin}_prediction.png', dpi=100, bbox_inches='tight')
        print(f'  ✓ {coin}')
        plt.show()
    except Exception as e:
        print(f'  ✗ {coin}: {str(e)[:40]}')

print('\n' + '='*80)
print('[STEP 6] 結果彙總')
print('='*80)

if results:
    print('\n{:12s} {:>10s} {:>8s} {:>10s} {:>8s} {:>10s}'.format(
        'Coin', 'Loss', 'Epochs', 'RMSE', 'MAE', 'R²'))
    print('-'*80)
    
    for r in sorted(results, key=lambda x: x['r2'], reverse=True):
        print('{:12s} {:>10.6f} {:>8d} {:>10.6f} {:>8.6f} {:>10.4f}'.format(
            r['coin'], r['loss'], r['epochs'], r['rmse'], r['mae'], r['r2']))
    
    if results:
        avg_r2 = np.mean([r['r2'] for r in results])
        max_r2 = max([r['r2'] for r in results])
        min_r2 = min([r['r2'] for r in results])
        print('-'*80)
        print(f'\n平均 R²: {avg_r2:.4f}')
        print(f'最高 R²: {max_r2:.4f}')
        print(f'最低 R²: {min_r2:.4f}')
        print(f'\n成功訓練: {len(trained_models)}/{len(all_data)} 幣種')

print('\n' + '='*80)
print('✓ 所有流程完成！')
print('='*80)
