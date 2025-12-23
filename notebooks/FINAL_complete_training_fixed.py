# ============================================================================
# CPB v2: 完整训练流程 (前20主流币种) - 最终版本
# 直接复制整个 Cell 粘贴到 Colab 执行
# ============================================================================

print('='*80)
print('CPB v2: 完整训练流程 (前20主流币种)')
print('='*80)

# ============================================================================
# STEP 0: 安装依赖 & 导入
# ============================================================================

print('\n[STEP 0] 安装依赖 & 导入库...')

import subprocess
import sys
import os
import json
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 安装 (使用 subprocess 代替 !pip)
print('  Installing packages...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'pandas', 'numpy', 'scikit-learn', 'requests', 'matplotlib'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

print('  Importing libraries...')
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

# ============================================================================
# STEP 1: 配置
# ============================================================================

print('\n[STEP 1] 配置参数...')

# 前 20 主流币种
CONFIG = {
    'coins': [
        'BTCUSDT',      # 1. Bitcoin
        'ETHUSDT',      # 2. Ethereum
        'XRPUSDT',      # 3. XRP
        'BNBUSDT',      # 4. Binance Coin
        'SOLUSDT',      # 5. Solana
        'ADAUSDT',      # 6. Cardano
        'TRXUSDT',      # 7. TRON
        'DOGEUSDT',     # 8. Dogecoin
        'AVAXUSDT',     # 9. Avalanche
        'LINKUSDT',     # 10. Chainlink
        'POLUSDT',      # 11. Polygon (prev. MATIC)
        'LTCUSDT',      # 12. Litecoin
        'BCHUSDT',      # 13. Bitcoin Cash
        'ATOMUSDT',     # 14. Cosmos
        'APTUSDT',      # 15. Aptos
        'FILUSDT',      # 16. Filecoin
        'SUIUSDT',      # 17. Sui
        'ARBUSDT',      # 18. Arbitrum
        'NEARUSDT',     # 19. NEAR
        'INJUSDT',      # 20. Injective
    ],
    'timeframes': ['1h'],  # 只用 1h (不用 15m 以不出现 NaN/Inf)
    'epochs': 20,
    'batch_size': 16,  # GPU 友好
    'learning_rate': 0.001,
    'lookback': 60,
    'n_features': 25,
    'use_dummy_data': False
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'\n配置:')
print(f'  币种数: {len(CONFIG["coins"])}')
print(f'  时间框: {CONFIG["timeframes"]}')
print(f'  训练轮数: {CONFIG["epochs"]}')
print(f'  批大小: {CONFIG["batch_size"]}')
print(f'  使用设备: {device}')

# ============================================================================
# STEP 2: 类定义
# ============================================================================

print('\n[STEP 2] 定义类...')

class BinanceDataCollector:
    """Binance API 数据采集器"""
    BASE_URL = "https://api.binance.us/api/v3"
    MAX_CANDLES = 1000
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_klines(self, symbol, interval="1h", limit=3000):
        """Download klines"""
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
        return True

class FeatureEngineer:
    """Technical indicators"""
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_all(self):
        df = self.df
        
        # SMA/EMA
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        for period in [14]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_width'] = (sma20 + std20 * 2) - (sma20 - std20 * 2)
        
        # Price momentum
        df['price_change'] = df['close'].pct_change() * 100
        df['volume_change'] = df['volume'].pct_change() * 100
        
        self.df = df.fillna(0)
        return self.df
    
    def get_features(self):
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude]

class DataPreprocessor:
    """Data preprocessing"""
    def __init__(self, df, lookback=60):
        self.df = df.copy()
        self.lookback = lookback
        self.scaler = MinMaxScaler((0, 1))
        self.pca = None
    
    def prepare(self, feature_cols, n_components=25):
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
    """LSTM Model"""
    def __init__(self, input_size=25, lstm_units=[96, 64], dense_units=32, dropout=0.2):
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
    """Model trainer"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=16, lr=0.001):
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
            
            if (epoch + 1) % 5 == 0:
                print(f'    Epoch {epoch+1:2d}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}')
            
            if val_loss < best_val_loss - 0.0001:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f'    Early Stop at epoch {epoch+1}')
                    if best_weights:
                        self.model.load_state_dict(best_weights)
                    break
        
        return {'best_val_loss': float(best_val_loss), 'epochs': epoch+1}

print('✓ 所有类定义完成')

# ============================================================================
# STEP 3: 下载数据
# ============================================================================

print('\n[STEP 3] 下载数据...')

all_data = {}
collector = BinanceDataCollector()
success_count = 0

for coin in CONFIG['coins']:
    try:
        df = collector.get_klines(coin, '1h', limit=3000)
        if BinanceDataCollector.validate(df):
            all_data[coin] = df
            print(f'  ✓ {coin:10s}: {len(df):4d} candles')
            success_count += 1
        else:
            print(f'  ✗ {coin:10s}: validation failed')
    except Exception as e:
        print(f'  ✗ {coin:10s}: {str(e)[:50]}')

print(f'\n✓ 下载完成: {success_count}/{len(CONFIG["coins"])} 币种')

if success_count == 0:
    print('\n➡ 没有数据，将使用测试数据...')
    CONFIG['use_dummy_data'] = True

# ============================================================================
# STEP 4: 训练模型
# ============================================================================

print('\n[STEP 4] 训练模型...')

trained_models = {}
results_summary = []

for coin in all_data:
    print(f'\n  {coin}')
    print('  ' + '-'*60)
    
    try:
        df = all_data[coin]
        
        # Feature engineering
        fe = FeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        # Preprocessing
        prep = DataPreprocessor(df_features, lookback=CONFIG['lookback'])
        features, feature_cols = prep.prepare(feature_cols, CONFIG['n_features'])
        X, y = prep.create_sequences()
        data = prep.split_data(X, y)
        
        # Build model
        model = LSTMModel(input_size=features.shape[-1])
        print(f'    Params: {model.count_params():,}')
        print(f'    Data: Train={len(data["X_train"])}, Val={len(data["X_val"])}, Test={len(data["X_test"])}')
        
        # Train
        trainer = Trainer(model, device=device)
        history = trainer.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            lr=CONFIG['learning_rate']
        )
        
        # Evaluate
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
        
        results_summary.append({
            'coin': coin,
            'params': model.count_params(),
            'loss': history['best_val_loss'],
            'epochs': history['epochs'],
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
        
        print(f'    MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.4f}')
        print(f'    ✓ Training complete')
        
    except Exception as e:
        print(f'    ✗ Error: {str(e)[:80]}')

print(f'\n✓ 训练完成: {len(trained_models)} 个模型')

# ============================================================================
# STEP 5: 绘制预测图
# ============================================================================

print('\n[STEP 5] 绘制预测图...')

fig_count = 0
for coin in trained_models:
    try:
        y_true = trained_models[coin]['y_test']
        y_pred = trained_models[coin]['y_pred']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'{coin} - Prediction Comparison', fontsize=12, fontweight='bold')
        
        # Plot 1: Last 200 steps
        ax = axes[0, 0]
        ax.plot(y_true[-200:], label='True', linewidth=2, color='blue', alpha=0.8)
        ax.plot(y_pred[-200:], label='Predicted', linewidth=2, color='red', alpha=0.8)
        ax.set_title('Last 200 Steps')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Full sequence
        ax = axes[0, 1]
        ax.plot(y_true, label='True', linewidth=1, color='blue', alpha=0.6)
        ax.plot(y_pred, label='Predicted', linewidth=1, color='red', alpha=0.6)
        ax.set_title('Full Sequence')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        residuals = y_true - y_pred
        ax = axes[1, 0]
        ax.hist(residuals, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.set_title('Residuals Distribution')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Scatter
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
        plt.savefig(f'{coin}_prediction.png', dpi=100, bbox_inches='tight')
        print(f'  ✓ {coin}: saved')
        plt.show()
        fig_count += 1
        
    except Exception as e:
        print(f'  ✗ {coin}: {str(e)[:50]}')

print(f'\n✓ 完成: {fig_count} 个图表')

# ============================================================================
# STEP 6: 汇总
# ============================================================================

print('\n' + '='*80)
print('[STEP 6] 汇总结果')
print('='*80)

if results_summary:
    print('\n{:12s} {:>10s} {:>10s} {:>8s} {:>10s} {:>10s} {:>10s}'.format(
        'Coin', 'Loss', 'Epochs', 'MSE', 'RMSE', 'MAE', 'R2'))
    print('-'*80)
    
    for r in sorted(results_summary, key=lambda x: x['r2'], reverse=True):
        print('{:12s} {:>10.6f} {:>10d} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.4f}'.format(
            r['coin'], r['loss'], r['epochs'], r['mse'], r['rmse'], r['mae'], r['r2']))
    
    avg_r2 = np.mean([r['r2'] for r in results_summary])
    print('-'*80)
    print(f'\n平均 R2: {avg_r2:.4f}')
    print(f'最高 R2: {max([r["r2"] for r in results_summary]):.4f}')
    print(f'最低 R2: {min([r["r2"] for r in results_summary]):.4f}')

print('\n' + '='*80)
print('✓ 所有流程完成！')
print('='*80)
