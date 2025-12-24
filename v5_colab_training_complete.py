#!/usr/bin/env python3
"""
CPB v5: Complete Cryptocurrency Price Prediction Pipeline
Target: Predict next 10 K-bars with MAPE < 0.02
Features: 40+ including volatility and price swing metrics
Model: Enhanced Seq2Seq LSTM with Attention
Hardware: Google Colab GPU (< 2 hours)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import warnings
import time
from datetime import datetime
from collections import defaultdict
import requests
from typing import Tuple, Dict, List

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================

COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
    'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
    'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT',
    'PEPEUSDT', 'INJUSDT', 'SHIBUSDT', 'ETCUSDT', 'LUNAUSDT'
]

TIMEFRAMES = ['15m', '1h']
PREDICT_STEPS = 10  # Predict next 10 K-bars
LOOKBACK_STEPS = 30  # Use last 30 K-bars for prediction
BAR_SIZES = {'15m': 15, '1h': 60}  # Minutes

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
CHECKPOINT_DIR = '/content/all_models/model_v5'

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Coins: {len(COINS)}, Timeframes: {len(TIMEFRAMES)}, Total models: {len(COINS) * len(TIMEFRAMES)}")

# ==============================================================================
# PHASE 1: DATA COLLECTION FROM BINANCE
# ==============================================================================

class BinanceDataCollector:
    """Download cryptocurrency data from Binance API"""
    
    def __init__(self):
        self.base_url = 'https://api.binance.com/api/v3/klines'
        self.limit_per_request = 1000
        
    def get_historical_klines(self, symbol: str, interval: str, limit: int = 8000) -> pd.DataFrame:
        """Download historical K-line data from Binance"""
        all_data = []
        remaining = limit
        
        while remaining > 0:
            current_limit = min(self.limit_per_request, remaining)
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': current_limit
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                    
                # If we have previous data, start from after the last one
                if all_data:
                    data = [x for x in data if x[0] > all_data[-1][0]]
                
                all_data.extend(data)
                remaining -= len(data)
                
                if len(data) < current_limit:
                    break
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"[WARN] Error downloading {symbol} {interval}: {e}")
                break
        
        if not all_data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        return df

# ==============================================================================
# PHASE 2: FEATURE ENGINEERING (40+ Features with Volatility)
# ==============================================================================

class FeatureEngineer:
    """Calculate 40+ technical indicators including volatility metrics"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # Basic price features
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Returns and changes
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = np.abs(df['price_change'])
        
        # Volatility Features (KEY FOR V5)
        df['volatility_5'] = df['log_return'].rolling(5).std()
        df['volatility_10'] = df['log_return'].rolling(10).std()
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['volatility_30'] = df['log_return'].rolling(30).std()
        df['volatility_ratio'] = df['volatility_10'] / (df['volatility_20'] + 1e-6)
        
        # Price swing/amplitude features
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['amplitude_5'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close']
        df['amplitude_10'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close']
        df['amplitude_20'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Momentum indicators
        df['rsi_14'] = FeatureEngineer._calculate_rsi(df['close'], 14)
        df['rsi_21'] = FeatureEngineer._calculate_rsi(df['close'], 21)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle_20'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper_20'] = df['bb_middle_20'] + (bb_std * 2)
        df['bb_lower_20'] = df['bb_middle_20'] - (bb_std * 2)
        df['bb_width_20'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['bb_middle_20']
        df['bb_pct_20'] = (df['close'] - df['bb_lower_20']) / (df['bb_upper_20'] - df['bb_lower_20'])
        
        # ATR (volatility indicator)
        df['atr_14'] = FeatureEngineer._calculate_atr(df, 14)
        
        # Volume indicators
        df['volume_sma_5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_5'] + 1e-6)
        
        # Direction indicators (for learning pattern recognition)
        df['price_direction'] = (df['close'] > df['open']).astype(int)
        df['high_low_direction'] = ((df['high'] - df['close']) - (df['close'] - df['low'])).astype(float)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_atr(df, period=14):
        df = df.copy()
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = np.abs(df['high'] - df['close'].shift())
        df['tr3'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr = df['tr'].rolling(period).mean()
        return atr

# ==============================================================================
# PHASE 3: DATA PREPROCESSING
# ==============================================================================

class DataPreprocessor:
    """Prepare data for LSTM training"""
    
    def __init__(self, lookback=30, predict_steps=10):
        self.lookback = lookback
        self.predict_steps = predict_steps
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Complete preprocessing pipeline"""
        df = df.copy()
        
        # Drop NaNs
        df = df.dropna().reset_index(drop=True)
        
        # Select features (exclude timestamp and OHLCV basics)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # For output: use close price as target
        # For input: use all features
        X_data = df[feature_cols].values
        
        # Normalize
        X_normalized = self.scaler.fit_transform(X_data)
        close_prices = df['close'].values.reshape(-1, 1)
        close_scaler = MinMaxScaler()
        close_normalized = close_scaler.fit_transform(close_prices)
        
        self.feature_names = feature_cols
        self.close_scaler = close_scaler
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_normalized) - self.lookback - self.predict_steps + 1):
            X_sequences.append(X_normalized[i:i+self.lookback])
            # Target: next predict_steps close prices
            y_sequences.append(close_normalized[i+self.lookback:i+self.lookback+self.predict_steps].flatten())
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        return X_sequences, y_sequences, close_normalized

# ==============================================================================
# PHASE 4: PYTORCH DATASET AND MODEL
# ==============================================================================

class CryptoDataset(Dataset):
    """PyTorch Dataset for cryptocurrency data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Seq2SeqLSTMV5(nn.Module):
    """Enhanced Seq2Seq LSTM for 10-step prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_layers: int = 2, predict_steps: int = 10, dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.predict_steps = predict_steps
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            batch_first=True,
            dropout=dropout
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """x shape: (batch_size, lookback, input_size)"""
        # Encoder
        encoder_output, (hidden, cell) = self.encoder_lstm(x)
        
        # Attention
        attn_output, _ = self.attention(encoder_output, encoder_output, encoder_output)
        
        # Decoder initialization with encoder state
        # Generate predict_steps outputs
        outputs = []
        decoder_input = attn_output[:, -1:, :]  # Last encoder output
        decoder_hidden = hidden
        decoder_cell = cell
        
        for _ in range(self.predict_steps):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            step_output = self.fc(decoder_output[:, -1, :hidden_size])
            outputs.append(step_output)
            decoder_input = decoder_output
        
        # Stack outputs: (batch_size, predict_steps, 1)
        return torch.cat(outputs, dim=1)

# ==============================================================================
# PHASE 5: TRAINING LOOP
# ==============================================================================

class Trainer:
    """Training pipeline for cryptocurrency models"""
    
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=15):
        """Complete training with early stopping"""
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")
            
            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history

# ==============================================================================
# PHASE 6: EVALUATION AND METRICS
# ==============================================================================

class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def evaluate(model, test_loader, close_scaler, device):
        """Calculate evaluation metrics"""
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform to original scale
        pred_original = close_scaler.inverse_transform(predictions)
        actual_original = close_scaler.inverse_transform(actuals)
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(actual_original, pred_original)
        rmse = np.sqrt(mean_squared_error(actual_original, pred_original))
        mae = mean_absolute_error(actual_original, pred_original)
        
        return {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'predictions': pred_original,
            'actuals': actual_original
        }

# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def train_single_coin(symbol: str, interval: str):
    """Train model for single coin and timeframe"""
    print(f"\n{'='*60}")
    print(f"Training {symbol} {interval}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 1. Download data
    print(f"[1/5] Downloading data from Binance...")
    collector = BinanceDataCollector()
    df = collector.get_historical_klines(symbol, interval, limit=8000)
    
    if df is None or len(df) < 100:
        print(f"[WARN] Insufficient data for {symbol} {interval}")
        return None
    
    print(f"  Downloaded {len(df)} K-bars")
    
    # 2. Feature engineering
    print(f"[2/5] Calculating technical indicators...")
    engineer = FeatureEngineer()
    df = engineer.calculate_indicators(df)
    print(f"  Created {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features")
    
    # 3. Preprocessing
    print(f"[3/5] Preprocessing data...")
    preprocessor = DataPreprocessor(lookback=LOOKBACK_STEPS, predict_steps=PREDICT_STEPS)
    X, y, close_prices = preprocessor.preprocess(df)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    
    # 4. Split data
    total_samples = len(X)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Create data loaders
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    test_dataset = CryptoDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 5. Train model
    print(f"[4/5] Training model (GPU={torch.cuda.is_available()})...")
    model = Seq2SeqLSTMV5(
        input_size=X.shape[2],
        hidden_size=256,
        num_layers=2,
        predict_steps=PREDICT_STEPS,
        dropout=0.3
    )
    
    trainer = Trainer(model, DEVICE, learning_rate=LEARNING_RATE)
    history = trainer.train(
        train_loader, val_loader,
        epochs=EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # 6. Evaluate
    print(f"[5/5] Evaluating model...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, test_loader, preprocessor.close_scaler, DEVICE)
    
    elapsed = time.time() - start_time
    print(f"\nResults:")
    print(f"  MAPE: {metrics['mape']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Time: {elapsed:.1f}s")
    
    # Save model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, f"{symbol}_{interval}.pt")
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'input_size': X.shape[2],
            'hidden_size': 256,
            'num_layers': 2,
            'predict_steps': PREDICT_STEPS,
            'dropout': 0.3
        },
        'scaler_params': {
            'scale_': preprocessor.scaler.scale_.tolist(),
            'min_': preprocessor.scaler.min_.tolist(),
        },
        'close_scaler_params': {
            'scale_': preprocessor.close_scaler.scale_.tolist(),
            'min_': preprocessor.close_scaler.min_.tolist(),
        },
        'metrics': metrics,
        'history': history
    }, model_path)
    
    print(f"  Model saved to {model_path}")
    
    return {
        'symbol': symbol,
        'interval': interval,
        'mape': metrics['mape'],
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'elapsed': elapsed,
        'model_path': model_path,
        'best_epoch': len(history['val_loss'])
    }

# ==============================================================================
# PHASE 7: HUGGINGFACE UPLOAD
# ==============================================================================

def upload_to_huggingface():
    """Upload all models to Hugging Face"""
    print(f"\n{'='*60}")
    print("Uploading to Hugging Face")
    print(f"{'='*60}")
    
    try:
        from huggingface_hub import HfApi, HfFolder
    except ImportError:
        print("[WARN] huggingface_hub not installed, skipping HF upload")
        return
    
    hf_token = input("Enter your Hugging Face token: ").strip()
    if not hf_token:
        print("[WARN] No token provided, skipping upload")
        return
    
    api = HfApi()
    
    try:
        # Create repo structure
        repo_id = "zongowo111/cpb-models"
        print(f"Uploading to {repo_id}...")
        
        # Upload entire v5 folder
        api.upload_folder(
            folder_path=CHECKPOINT_DIR,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="model_v5",
            commit_message="Add v5 models with 10-step prediction",
            token=hf_token,
            create_pr=False,
        )
        
        print(f"[SUCCESS] Uploaded to {repo_id}/model_v5")
        
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("CPB v5: Cryptocurrency Price Prediction Pipeline")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Coins: {len(COINS)}")
    print(f"Timeframes: {len(TIMEFRAMES)}")
    print(f"Total models: {len(COINS) * len(TIMEFRAMES)}")
    print(f"Predict steps: {PREDICT_STEPS}")
    print(f"Lookback steps: {LOOKBACK_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("="*60 + "\n")
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Train models
    results = []
    total_start = time.time()
    
    for i, coin in enumerate(COINS):
        for interval in TIMEFRAMES:
            try:
                result = train_single_coin(coin, interval)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[ERROR] Training failed for {coin} {interval}: {e}")
                import traceback
                traceback.print_exc()
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Models trained: {len(results)}/{len(COINS)*len(TIMEFRAMES)}")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    
    if results:
        avg_mape = np.mean([r['mape'] for r in results])
        best_mape = min(results, key=lambda x: x['mape'])
        worst_mape = max(results, key=lambda x: x['mape'])
        
        print(f"\nMAPE Statistics:")
        print(f"  Average: {avg_mape:.6f}")
        print(f"  Best: {best_mape['symbol']} {best_mape['interval']} ({best_mape['mape']:.6f})")
        print(f"  Worst: {worst_mape['symbol']} {worst_mape['interval']} ({worst_mape['mape']:.6f})")
        print(f"  Target: < 0.02")
        
        # Save results
        results_path = os.path.join(CHECKPOINT_DIR, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")
    
    # Upload to HuggingFace
    upload_to_huggingface()
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == '__main__':
    main()
