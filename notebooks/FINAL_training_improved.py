# ============================================================================
# CPB v2: 改进版 - 更好的数据清理与特征工程
# 解决 Infinity 错误、负数 R2、不使用 CPU
# ============================================================================

print('='*80)
print('CPB v2: 改进版 - 更好的数据清理')
print('='*80)

import subprocess
import sys
import os
import json
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

print('\n[STEP 0] 安装依赖...')
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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print('✓ 所有库导入完成')

print('\n[STEP 1] 配置参数...')

# 简化配置 - 仅用最稳定的 5 个币种
CONFIG = {
    'coins': [
        'BTCUSDT',      # Bitcoin
        'ETHUSDT',      # Ethereum
        'SOLUSDT',      # Solana
        'BNBUSDT',      # BNB
        'AVAXUSDT',     # Avalanche
    ],
    'timeframes': ['1h'],
    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 0.001,
    'lookback': 60,
    'n_features': 20,  # 改小
    'use_dummy_data': False
}

# 检查 GPU
if torch.cuda.is_available():
    device = 'cuda'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠ 没有 GPU，会使用 CPU（比较慢）')
    device = 'cpu'

print(f'  币种: {CONFIG["coins"]}')
print(f'  轮数: {CONFIG["epochs"]}')
print(f'  使用: {device}')

print('\n[STEP 2] 定义类...')

class BinanceDataCollector:
    """Binance API"""
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
        start_time = int((datetime.utcnow() - timedelta(days=90)).timestamp() * 1000)
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
        if len(df) < 100:
            return False
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            return False
        if (df[['open', 'high', 'low', 'close', 'volume']] == 0).any().any():
            return False
        return True

class FeatureEngineer:
    """Technical indicators - 改进版，更干准"""
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        try:
            # 简化指标 - 只保留最重要的
            
            # 1. SMA
            for period in [10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
            
            # 2. RSI (14)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / (loss + 1e-10)  # 拷会至少不是无穷
            df['rsi_14'] = 100 - (100 / (1 + rs))
            df['rsi_14'] = df['rsi_14'].clip(0, 100)  # 限制在 0-100
            
            # 3. MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = (ema12 - ema26)
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # 4. 价格动震
            df['price_change'] = df['close'].pct_change()
            df['price_change'] = df['price_change'].clip(-1, 1)  # 限制不超过 ±100%
            
            # 5. 成交量动震
            df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
            df['volume_ma_ratio'] = df['volume_ma_ratio'].clip(0, 5)  # 限制不超过 5倍
            
            # 6. 可作为目标的下一时段价格变化
            df['target'] = df['close'].shift(-1).pct_change()
            df['target'] = df['target'].clip(-0.1, 0.1)  # 限制目标为 ±10%
            
            self.df = df.fillna(0).replace([np.inf, -np.inf], 0)
            
            return self.df
        except Exception as e:
            print(f'    Feature error: {e}')
            raise
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
        features = [col for col in self.df.columns if col not in exclude]
        return features

class DataPreprocessor:
    """Data preprocessing"""
    def __init__(self, df, lookback=60):
        self.df = df.copy()
        self.lookback = lookback
        self.scaler = MinMaxScaler((0, 1))
        self.pca = None
    
    def prepare(self, feature_cols, n_components=20):
        self.df = self.df.dropna()
        self.df = self.df[(self.df != 0).any(axis=1)]  # 移除全零行
        
        feature_data = self.df[feature_cols].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], 0)
        feature_data = feature_data.fillna(0)
        
        if len(feature_cols) > n_components:
            self.pca = PCA(n_components=n_components)
            feature_data = self.pca.fit_transform(feature_data)
            feature_cols = [f'pc_{i}' for i in range(n_components)]
            feature_data = pd.DataFrame(feature_data, columns=feature_cols, index=self.df.index)
        
        feature_data = self.scaler.fit_transform(feature_data)
        self.features = feature_data
        self.feature_cols = feature_cols
        
        return feature_data, feature_cols
    
    def create_sequences(self, target_col=None):
        X, y = [], []
        data = self.features
        if target_col is not None:
            target = self.df[target_col].values
        else:
            target = data[:, 0]
        
        for i in range(self.lookback, len(data)):
            if i < len(target):
                X.append(data[i - self.lookback:i])
                y.append(target[i])
        
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

class LSTMModel(nn.Module):
    """LSTM Model"""
    def __init__(self, input_size=20):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.dense1 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(16, 1)
    
    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)
        last_out = lstm2_out[:, -1, :]
        dense_out = self.dense1(last_out)
        dense_out = self.relu(dense_out)
        dense_out = self.dropout(dense_out)
        output = self.dense2(dense_out)
        return output
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Trainer:
    """Model trainer"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, lr=0.001):
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1:2d}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}')
            
            if val_loss < best_val_loss - 0.0001:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    if best_weights:
                        self.model.load_state_dict(best_weights)
                    break
        
        return {'best_val_loss': float(best_val_loss), 'epochs': epoch+1}

print('✓ 所有类定义完成')

print('\n[STEP 3] 下载数据...')

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

print(f'\n✓ 下载完成: {len(all_data)}/{len(CONFIG["coins"])} 币种')

print('\n[STEP 4] 训练模型...')

trained_models = {}
results = []

for coin in all_data:
    print(f'\n  {coin}')
    print('  ' + '-'*50)
    
    try:
        df = all_data[coin]
        fe = FeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        prep = DataPreprocessor(df_features, lookback=CONFIG['lookback'])
        features, feature_cols = prep.prepare(feature_cols, CONFIG['n_features'])
        X, y = prep.create_sequences(target_col='target' if 'target' in df_features.columns else None)
        data = prep.split_data(X, y)
        
        if len(X) < 200:
            print(f'    ✗ 数据序列太短')
            continue
        
        model = LSTMModel(input_size=features.shape[-1])
        print(f'    Params: {model.count_params():,}')
        
        trainer = Trainer(model, device=device)
        history = trainer.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            lr=CONFIG['learning_rate']
        )
        
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
        
        print(f'    MSE={mse:.6f}, RMSE={rmse:.6f}, R2={r2:.4f}')
        print(f'    ✓ Training complete')
        
    except Exception as e:
        print(f'    ✗ Error: {str(e)[:60]}')

print(f'\n✓ 训练完成: {len(trained_models)} 个模型')

print('\n[STEP 5] 绘制预测图...')

for coin in trained_models:
    try:
        y_true = trained_models[coin]['y_test']
        y_pred = trained_models[coin]['y_pred']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'{coin} - Prediction', fontsize=12, fontweight='bold')
        
        ax = axes[0, 0]
        ax.plot(y_true[-100:], label='True', linewidth=2, color='blue')
        ax.plot(y_pred[-100:], label='Predicted', linewidth=2, color='red')
        ax.set_title('Last 100')
        ax.legend()
        ax.grid(alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(y_true, label='True', linewidth=1, color='blue', alpha=0.7)
        ax.plot(y_pred, label='Predicted', linewidth=1, color='red', alpha=0.7)
        ax.set_title('Full')
        ax.legend()
        ax.grid(alpha=0.3)
        
        residuals = y_true - y_pred
        ax = axes[1, 0]
        ax.hist(residuals, bins=50, color='green', alpha=0.7)
        ax.set_title('Residuals')
        ax.grid(alpha=0.3)
        
        ax = axes[1, 1]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
        ax.set_title('True vs Pred')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{coin}_pred.png', dpi=100, bbox_inches='tight')
        print(f'  ✓ {coin}')
        plt.show()
    except Exception as e:
        print(f'  ✗ {coin}: {str(e)[:40]}')

print('\n' + '='*80)
print('[STEP 6] 总结')
print('='*80)

if results:
    print('\n{:12s} {:>10s} {:>8s} {:>10s} {:>10s}'.format('Coin', 'Loss', 'Epochs', 'RMSE', 'R2'))
    print('-'*80)
    for r in sorted(results, key=lambda x: x['r2'], reverse=True):
        print('{:12s} {:>10.6f} {:>8d} {:>10.6f} {:>10.4f}'.format(
            r['coin'], r['loss'], r['epochs'], r['rmse'], r['r2']))
    
    avg_r2 = np.mean([r['r2'] for r in results])
    print('-'*80)
    print(f'\n平均 R2: {avg_r2:.4f}')
    print(f'最高 R2: {max([r["r2"] for r in results]):.4f}')
    print(f'最低 R2: {min([r["r2"] for r in results]):.4f}')

print('\n' + '='*80)
print('✓ 全部完成！')
print('='*80)
