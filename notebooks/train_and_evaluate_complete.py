# ============================================================================
# CPB v2: 完整训练 + 保存 + 评估可视化 (一次执行版)
# 直接复制粘贴到 Colab 执行！
# ============================================================================

print('='*70)
print('CPB v2: 完整训练 + 评估 (单 Cell 版本)')
print('='*70)

# ============================================================================
# PART 0: 安装 & 导入
# ============================================================================

print('\n[PART 0] 安装依赖 & 导入库...')

import subprocess
import sys
import os
import json
import time
import pickle
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'pandas', 'numpy', 'scikit-learn', 'requests', 'matplotlib'], check=False)

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
import matplotlib.dates as mdates
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print('✓ 所有库导入完成')

# ============================================================================
# PART 1: 配置 & 类定义
# ============================================================================

print('\n[PART 1] 配置 & 类定义...')

CONFIG = {
    'coins': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    'timeframes': ['15m', '1h'],
    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 0.001,
    'lookback': 60,
    'n_features': 30,
    'use_dummy_data': False
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BinanceDataCollector:
    BASE_URL = "https://api.binance.us/api/v3"
    MAX_CANDLES = 1000
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_klines(self, symbol, interval="15m", limit=3000):
        all_klines = []
        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(days=90)).timestamp() * 1000)
        current_start = start_time
        retry_count = 0
        
        while current_start < end_time and len(all_klines) < limit:
            try:
                params = {
                    "symbol": symbol,
                    "interval": interval,
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
                time.sleep(0.5)
                
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
        return True

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        df['price_change'] = df['close'].pct_change() * 100
        df['volume_change'] = df['volume'].pct_change() * 100
        
        self.df = df.fillna(0)
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude]

class DataPreprocessor:
    def __init__(self, df, lookback=60):
        self.df = df.copy()
        self.lookback = lookback
        self.scaler = MinMaxScaler((0, 1))
        self.pca = None
    
    def prepare(self, feature_cols, n_components=30):
        self.df = self.df.dropna()
        feature_data = self.df[feature_cols].copy()
        
        if len(feature_cols) > n_components:
            self.pca = PCA(n_components=n_components)
            feature_data = self.pca.fit_transform(feature_data)
            feature_cols = [f'pc_{i}' for i in range(n_components)]
            feature_data = pd.DataFrame(feature_data, columns=feature_cols, index=self.df.index)
        
        feature_data = self.scaler.fit_transform(feature_data)
        self.features = feature_data
        self.feature_cols = feature_cols
        
        return feature_data, feature_cols
    
    def create_sequences(self):
        X, y = [], []
        data = self.features
        
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i, 0])
        
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
    def __init__(self, input_size=30, lstm_units=[96, 64], dense_units=32, dropout=0.2):
        super().__init__()
        
        self.lstm1 = nn.LSTM(input_size, lstm_units[0], batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_units[0] * 2, lstm_units[1], batch_first=True, dropout=dropout, bidirectional=True)
        
        lstm_output = lstm_units[1] * 2
        self.dense1 = nn.Linear(lstm_output, dense_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(dense_units, 1)
    
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
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
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
                print(f'    Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}')
            
            if val_loss < best_val_loss - 0.0001:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'    Early Stop: Epoch {epoch+1}')
                    if best_weights:
                        self.model.load_state_dict(best_weights)
                    break
        
        return {'best_val_loss': float(best_val_loss), 'epochs': epoch+1}

print('✓ 所有类定义完成')

# ============================================================================
# PART 2: 下载数据
# ============================================================================

print('\n[PART 2] 下载数据...')

all_data = {}
collector = BinanceDataCollector()

for coin in CONFIG['coins']:
    coin_data = {}
    for timeframe in CONFIG['timeframes']:
        try:
            df = collector.get_klines(coin, timeframe, limit=3000)
            if BinanceDataCollector.validate(df):
                coin_data[timeframe] = df
                print(f'  ✓ {coin} {timeframe}: {len(df)} candles')
            else:
                print(f'  ✗ {coin} {timeframe}: validation failed')
        except Exception as e:
            print(f'  ✗ {coin} {timeframe}: {str(e)[:60]}')
    
    if coin_data:
        all_data[coin] = coin_data

print(f'\n  Total: {sum([len(v) for v in all_data.values()])} datasets')

# ============================================================================
# PART 3: 训练 & 保存模型
# ============================================================================

print('\n[PART 3] 训练 & 保存模型...')

trained_models = {}  # 存储训练母子

for coin in all_data:
    for timeframe in all_data[coin]:
        key = f'{coin}_{timeframe}'
        print(f'\n  {key}')
        print('  ' + '-'*50)
        
        try:
            df = all_data[coin][timeframe]
            
            fe = FeatureEngineer(df)
            df_features = fe.calculate_all()
            feature_cols = fe.get_features()
            
            prep = DataPreprocessor(df_features, lookback=CONFIG['lookback'])
            features, feature_cols = prep.prepare(feature_cols, CONFIG['n_features'])
            X, y = prep.create_sequences()
            data = prep.split_data(X, y)
            
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
            
            # 保存训练好的模型
            trained_models[key] = {
                'model': model,
                'preprocessor': prep,
                'data': data,
                'X': X,
                'y': y,
                'history': history
            }
            
            print(f'    ✓ Best Val Loss: {history["best_val_loss"]:.6f}')
            
        except Exception as e:
            print(f'    ✗ Error: {str(e)[:100]}')

print(f'\n  ✓ Trained: {len(trained_models)} models')

# ============================================================================
# PART 4: 评估与可视化
# ============================================================================

print('\n[PART 4] 评估与可视化...')

evaluation_results = {}

for key in trained_models:
    print(f'\n  {key}')
    print('  ' + '-'*50)
    
    try:
        stored = trained_models[key]
        model = stored['model']
        data = stored['data']
        
        # 预测测试集
        model.eval()
        X_test_t = torch.FloatTensor(data['X_test']).to(device)
        
        with torch.no_grad():
            y_pred = model(X_test_t).cpu().numpy()
        
        y_test = data['y_test']
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        evaluation_results[key] = {
            'y_pred': y_pred.flatten(),
            'y_test': y_test.flatten(),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f'    MSE:  {mse:.6f}')
        print(f'    RMSE: {rmse:.6f}')
        print(f'    MAE:  {mae:.6f}')
        print(f'    R2:   {r2:.6f}')
        
    except Exception as e:
        print(f'    ✗ Error: {str(e)[:100]}')

# ============================================================================
# PART 5: 绘制对比图
# ============================================================================

print('\n[PART 5] 绘制对比图...')

fig_count = 0
for key in evaluation_results:
    print(f'\n  {key}: 正在绘制...')
    
    try:
        result = evaluation_results[key]
        y_true = result['y_test']
        y_pred = result['y_pred']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'{key} - Prediction Comparison\nR2={result["r2"]:.4f}, RMSE={result["rmse"]:.6f}', 
                     fontsize=12, fontweight='bold')
        
        # 上左: 预测对比 (最后 200 步)
        ax = axes[0, 0]
        ax.plot(y_true[-200:], label='True', linewidth=2, color='blue', alpha=0.8)
        ax.plot(y_pred[-200:], label='Predicted', linewidth=2, color='red', alpha=0.8)
        ax.set_title('Predictions vs True (Last 200)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 上右: 完整序列
        ax = axes[0, 1]
        ax.plot(y_true, label='True', linewidth=1, color='blue', alpha=0.6)
        ax.plot(y_pred, label='Predicted', linewidth=1, color='red', alpha=0.6)
        ax.set_title('Full Sequence')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 下左: 残差
        residuals = y_true - y_pred
        ax = axes[1, 0]
        ax.hist(residuals, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.set_title('Residuals Distribution')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.grid(True, alpha=0.3)
        
        # 下右: 散点图
        ax = axes[1, 1]
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='purple')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax.set_title('True vs Predicted')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        filename = f'{key}_prediction.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f'    ✓ Saved: {filename}')
        
        plt.show()
        fig_count += 1
        
    except Exception as e:
        print(f'    ✗ Error: {str(e)[:100]}')

print(f'\n  ✓ Generated: {fig_count} charts')

# ============================================================================
# PART 6: 总结
# ============================================================================

print('\n' + '='*70)
print('[PART 6] 评估总结')
print('='*70)

if evaluation_results:
    print('\n  Performance Summary:')
    print('  ' + '-'*50)
    
    for key in sorted(evaluation_results.keys()):
        result = evaluation_results[key]
        print(f'\n  {key}:')
        print(f'    MSE:  {result["mse"]:.8f}')
        print(f'    RMSE: {result["rmse"]:.8f}')
        print(f'    MAE:  {result["mae"]:.8f}')
        print(f'    R2:   {result["r2"]:.6f}')
    
    best_r2 = max([v['r2'] for v in evaluation_results.values()])
    worst_r2 = min([v['r2'] for v in evaluation_results.values()])
    
    print('\n  ' + '-'*50)
    print(f'  Best R2:  {best_r2:.6f}')
    print(f'  Worst R2: {worst_r2:.6f}')
else:
    print('  No successful models')

print('\n' + '='*70)
print('✓ Training & Evaluation Complete!')
print('='*70)
