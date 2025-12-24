#!/usr/bin/env python3
"""
CPB v5: Cryptocurrency Price Prediction Model - Complete Training Pipeline

Architecture:
  - Seq2Seq LSTM with Multi-Head Attention
  - Bidirectional Encoder (2 layers, 256 units)
  - Unidirectional Decoder (2 layers, 256 units)
  - 8-head attention mechanism
  - 40+ technical indicator features
  - Predicts 10 K-bars ahead with volatility awareness

Target: MAPE < 0.02 (2%)
Training Time: ~2 hours on Colab T4
Coins: 20 (top by market cap)
Timeframes: 2 (15m, 1h)
Total Models: 40

Author: Cai Zongxun
Version: v5.0
Date: 2025-12-24
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    
    # Data configuration
    COINS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
        'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
        'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT',
        'PEPEUSDT', 'INJUSDT', 'SHIBUSDT', 'ETCUSDT', 'LUNAUSDT'
    ]
    TIMEFRAMES = ['15m', '1h']
    LOOKBACK_STEPS = 30  # Input sequence length
    PREDICT_STEPS = 10   # Output sequence length
    KBARS_TO_DOWNLOAD = 8000  # Approximate K-bars from Binance
    
    # Model configuration
    INPUT_SIZE = 40  # Number of features (technical indicators)
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    NUM_HEADS = 8  # Attention heads
    
    # Training configuration
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 15
    GRADIENT_CLIP = 1.0
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    CACHE_DIR = Path('/content/all_models')
    MODEL_V5_DIR = CACHE_DIR / 'model_v5'
    
    # Validation/Test split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Target metric
    TARGET_MAPE = 0.02  # 2%
    MAX_TRAINING_TIME = 7200  # 2 hours in seconds


# ============================================================================
# FEATURE ENGINEERING (40+ indicators)
# ============================================================================

class TechnicalIndicators:
    """Calculate 40+ technical indicators for cryptocurrency price data"""
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 40+ technical indicators
        
        Input: OHLCV data
        Output: OHLCV + 40 indicators
        """
        
        df = df.copy()
        
        # Price features (3)
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Log returns for volatility
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # === VOLATILITY FEATURES (5) === KEY FOR v5
        # These help model learn when to expect big swings
        df['volatility_5'] = df['log_return'].rolling(5).std()
        df['volatility_10'] = df['log_return'].rolling(10).std()
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['volatility_30'] = df['log_return'].rolling(30).std()
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']  # Change in volatility
        
        # === AMPLITUDE FEATURES (4) === KEY FOR v5
        # Different from volatility: actual price range
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['amplitude_5'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close']
        df['amplitude_10'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close']
        df['amplitude_20'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
        
        # Returns features (3)
        df['returns'] = df['log_return']  # Already calculated
        df['returns_pct'] = df['close'].pct_change()
        df['abs_returns'] = np.abs(df['returns'])
        
        # Moving Averages (12)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Momentum Indicators (7)
        # RSI
        df['rsi_14'] = _calculate_rsi(df['close'], 14)
        df['rsi_21'] = _calculate_rsi(df['close'], 21)
        
        # MACD
        macd, signal, diff = _calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_diff'] = diff
        
        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['roc_12'] = df['close'].pct_change(12)  # Rate of Change
        
        # Bollinger Bands (5)
        bb_upper, bb_middle, bb_lower, bb_width, bb_pct = _calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = bb_width
        df['bb_pct'] = bb_pct
        
        # ATR (1)
        df['atr_14'] = _calculate_atr(df, 14)
        
        # Volume features (2)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        else:
            df['volume_sma'] = 0
            df['volume_ratio'] = 1
        
        # Direction indicators (2)
        df['price_direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['hl_balance'] = (df['high'] - df['close']) / (df['close'] - df['low'])
        
        # Drop NaN rows
        df = df.dropna()
        
        return df


def _calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    
    return pd.Series(rsi, index=prices.index)


def _calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    diff = macd - signal_line
    return macd, signal_line, diff


def _calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    width = (upper - lower) / sma
    pct_b = (prices - lower) / (upper - lower)
    return upper, sma, lower, width, pct_b


def _calculate_atr(df, period=14):
    """Calculate Average True Range"""
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = tr.rolling(period).mean()
    return atr


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, values, keys, query, mask=None):
        """
        Args:
            values: (batch_size, seq_len, hidden_size)
            keys: (batch_size, seq_len, hidden_size)
            query: (batch_size, query_len, hidden_size)
            mask: (batch_size, 1, 1, seq_len)
        """
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.query(query)  # (batch, query_len, hidden)
        K = self.key(keys)     # (batch, seq_len, hidden)
        V = self.value(values) # (batch, seq_len, hidden)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        
        # Final linear layer
        output = self.fc_out(context)
        
        return output, attention_weights


class Seq2SeqLSTMV5(nn.Module):
    """Sequence-to-Sequence LSTM with Attention for 10-step price prediction"""
    
    def __init__(
        self,
        input_size: int = 40,
        hidden_size: int = 256,
        num_layers: int = 2,
        predict_steps: int = 10,
        dropout: float = 0.3,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_steps = predict_steps
        self.dropout = dropout
        
        # Encoder: Bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Process both directions
        )
        
        # Attention
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size * 2,  # Because bidirectional
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Decoder: Unidirectional LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,  # Takes encoder output
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, lookback_steps, input_size)
        Returns:
            predictions: (batch_size, predict_steps, 1)
        """
        batch_size = x.shape[0]
        
        # Encoder: Bidirectional LSTM processes all input
        encoder_output, (h_n, c_n) = self.encoder(x)
        # encoder_output: (batch, lookback, hidden*2)
        # h_n, c_n: (2*num_layers, batch, hidden)
        
        # Attention: Learn which parts of encoder output matter
        context, attention_weights = self.attention(
            values=encoder_output,
            keys=encoder_output,
            query=encoder_output  # Attend to own output
        )
        # context: (batch, lookback, hidden*2)
        
        # Decoder: Generate predictions
        # Initialize with last encoder state
        # Need to project bidirectional state to unidirectional
        h_decoder = h_n[-1:].unsqueeze(0)  # (1, batch, hidden)
        c_decoder = c_n[-1:].unsqueeze(0)
        
        # Use context for all decode steps
        decoder_input = context[:, -1:, :]  # Take last context: (batch, 1, hidden*2)
        decoder_input = decoder_input.repeat(1, self.predict_steps, 1)  # Repeat for all steps
        
        decoder_output, _ = self.decoder(decoder_input, (h_decoder, c_decoder))
        # decoder_output: (batch, predict_steps, hidden)
        
        # Output projection: hidden -> 1 price
        output = self.relu(self.fc1(decoder_output))
        output = self.dropout_layer(output)
        output = self.relu(self.fc2(output))
        output = self.dropout_layer(output)
        output = self.fc_out(output)
        # output: (batch, predict_steps, 1)
        
        return output


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class DataProcessor:
    """Handle data downloading, preprocessing, and batching"""
    
    @staticmethod
    def download_binance_data(coin: str, timeframe: str, limit: int = 8000) -> pd.DataFrame:
        """
        Download OHLCV data from Binance
        
        Args:
            coin: e.g., 'BTCUSDT'
            timeframe: e.g., '15m', '1h'
            limit: Number of K-bars to download
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        try:
            import requests
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': coin,
                'interval': timeframe,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {coin} {timeframe}: {e}")
            return None
    
    @staticmethod
    def create_sequences(
        features: np.ndarray,
        prices: np.ndarray,
        lookback: int = 30,
        predict_steps: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences
        
        Args:
            features: (n_samples, n_features) - normalized features
            prices: (n_samples,) - normalized prices
            lookback: Input sequence length
            predict_steps: Output sequence length
        
        Returns:
            X: (n_sequences, lookback, n_features)
            y: (n_sequences, predict_steps)
        """
        X, y = [], []
        
        for i in range(len(features) - lookback - predict_steps + 1):
            X.append(features[i:i+lookback])
            y.append(prices[i+lookback:i+lookback+predict_steps])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def preprocess_coin_data(
        coin: str,
        timeframe: str,
        lookback: int = 30,
        predict_steps: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, dict]:
        """
        Complete preprocessing pipeline
        
        Returns:
            X, y: Training sequences
            feature_scaler, price_scaler: For denormalization
            data_info: Metadata about data
        """
        logger.info(f"Preprocessing {coin} {timeframe}...")
        
        # Download data
        df = DataProcessor.download_binance_data(coin, timeframe)
        if df is None or len(df) < 1000:
            logger.error(f"Insufficient data for {coin} {timeframe}")
            return None
        
        logger.info(f"  Downloaded {len(df)} K-bars")
        
        # Calculate indicators
        df = TechnicalIndicators.calculate_all_features(df)
        
        # Select features (exclude OHLCV columns)
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"  Created {len(feature_cols)} features")
        
        # Normalize features
        feature_scaler = MinMaxScaler()
        features_normalized = feature_scaler.fit_transform(df[feature_cols])
        
        # Normalize prices
        prices = df['close'].values.reshape(-1, 1)
        price_scaler = MinMaxScaler()
        prices_normalized = price_scaler.fit_transform(prices).ravel()
        
        # Create sequences
        X, y = DataProcessor.create_sequences(
            features_normalized,
            prices_normalized,
            lookback=lookback,
            predict_steps=predict_steps
        )
        
        logger.info(f"  Created sequences: X={X.shape}, y={y.shape}")
        
        data_info = {
            'coin': coin,
            'timeframe': timeframe,
            'n_kbars': len(df),
            'n_features': len(feature_cols),
            'n_sequences': len(X),
            'feature_cols': feature_cols
        }
        
        return X, y, feature_scaler, price_scaler, data_info


# ============================================================================
# TRAINING LOOP
# ============================================================================

class ModelTrainer:
    """Train Seq2Seq model with early stopping and gradient clipping"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions.squeeze(-1), y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRADIENT_CLIP
            )
            
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
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions.squeeze(-1), y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader,
        val_loader,
        max_epochs=100,
        early_stopping_patience=15
    ):
        """Complete training loop with early stopping"""
        
        logger.info(f"Starting training for {max_epochs} epochs...")
        
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{max_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                    break
        
        return self.history


# ============================================================================
# EVALUATION
# ============================================================================

class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def evaluate(
        model,
        test_loader,
        price_scaler,
        device
    ) -> Dict:
        """
        Evaluate model on test set
        
        Returns:
            Metrics dictionary with MAPE, RMSE, MAE
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                
                predictions = model(X_batch).squeeze(-1).cpu().numpy()
                targets = y_batch.cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # Denormalize to original price scale
        all_predictions_denorm = price_scaler.inverse_transform(
            all_predictions.reshape(-1, 1)
        ).ravel()
        
        all_targets_denorm = price_scaler.inverse_transform(
            all_targets.reshape(-1, 1)
        ).ravel()
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(all_targets_denorm, all_predictions_denorm)
        rmse = np.sqrt(mean_squared_error(all_targets_denorm, all_predictions_denorm))
        mae = mean_absolute_error(all_targets_denorm, all_predictions_denorm)
        
        return {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'rmse_pct': rmse / np.mean(all_targets_denorm),
            'mae_pct': mae / np.mean(all_targets_denorm)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Complete training pipeline"""
    
    print("=" * 60)
    print("CPB v5: Cryptocurrency Price Prediction Pipeline")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print(f"Device: {Config.DEVICE}")
    print(f"Coins: {len(Config.COINS)}")
    print(f"Timeframes: {len(Config.TIMEFRAMES)}")
    print(f"Total models: {len(Config.COINS) * len(Config.TIMEFRAMES)}")
    print(f"Predict steps: {Config.PREDICT_STEPS}")
    print(f"Lookback steps: {Config.LOOKBACK_STEPS}")
    print("=" * 60)
    
    # Create cache directory
    Config.MODEL_V5_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training results storage
    results_by_coin = {}
    
    # Train models
    total_coins = len(Config.COINS) * len(Config.TIMEFRAMES)
    current_model = 0
    
    for coin in Config.COINS:
        for timeframe in Config.TIMEFRAMES:
            current_model += 1
            print(f"\n[{current_model}/{total_coins}] Training {coin} {timeframe}")
            
            try:
                # Data preprocessing
                print("  [1/5] Downloading data from Binance...")
                X, y, feature_scaler, price_scaler, data_info = DataProcessor.preprocess_coin_data(
                    coin, timeframe,
                    lookback=Config.LOOKBACK_STEPS,
                    predict_steps=Config.PREDICT_STEPS
                )
                
                if X is None:
                    logger.warning(f"Skipping {coin} {timeframe}")
                    continue
                
                print(f"  [2/5] Calculating technical indicators...")
                print(f"    Created {data_info['n_features']} features")
                
                # Train/val/test split
                print(f"  [3/5] Preprocessing data...")
                n = len(X)
                train_idx = int(n * Config.TRAIN_RATIO)
                val_idx = train_idx + int(n * Config.VAL_RATIO)
                
                X_train, y_train = X[:train_idx], y[:train_idx]
                X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
                X_test, y_test = X[val_idx:], y[val_idx:]
                
                print(f"    X shape: {X.shape}, y shape: {y.shape}")
                print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
                
                # Create dataloaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.FloatTensor(y_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.FloatTensor(y_val)
                )
                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test),
                    torch.FloatTensor(y_test)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
                
                # Create model
                print(f"  [4/5] Training model (GPU={torch.cuda.is_available()})...")
                model = Seq2SeqLSTMV5(
                    input_size=Config.INPUT_SIZE,
                    hidden_size=Config.HIDDEN_SIZE,
                    num_layers=Config.NUM_LAYERS,
                    predict_steps=Config.PREDICT_STEPS,
                    dropout=Config.DROPOUT,
                    num_heads=Config.NUM_HEADS
                )
                
                trainer = ModelTrainer(model, Config.DEVICE, Config)
                history = trainer.train(
                    train_loader, val_loader,
                    max_epochs=Config.EPOCHS,
                    early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
                )
                
                # Evaluate
                print(f"  [5/5] Evaluating model...")
                metrics = ModelEvaluator.evaluate(model, test_loader, price_scaler, Config.DEVICE)
                
                # Save model
                model_path = Config.MODEL_V5_DIR / f"{coin}_{timeframe}.pt"
                torch.save({
                    'model_state': model.state_dict(),
                    'config': {
                        'input_size': Config.INPUT_SIZE,
                        'hidden_size': Config.HIDDEN_SIZE,
                        'num_layers': Config.NUM_LAYERS,
                        'predict_steps': Config.PREDICT_STEPS,
                        'dropout': Config.DROPOUT,
                        'num_heads': Config.NUM_HEADS
                    },
                    'scaler_params': {
                        'feature_mean': feature_scaler.data_mean_,
                        'feature_scale': feature_scaler.data_range_,
                    },
                    'price_scaler_params': {
                        'price_mean': price_scaler.data_min_,
                        'price_scale': price_scaler.data_range_,
                    },
                    'metrics': metrics,
                    'history': history,
                    'data_info': data_info
                }, model_path)
                
                # Log results
                print(f"\nResults:")
                print(f"  MAPE: {metrics['mape']:.6f}")
                print(f"  RMSE: {metrics['rmse']:.6f}")
                print(f"  MAE: {metrics['mae']:.6f}")
                print(f"  Model saved to {model_path}")
                
                results_by_coin[f"{coin}_{timeframe}"] = metrics
                
            except Exception as e:
                logger.error(f"Error training {coin} {timeframe}: {e}")
                continue
    
    # Save summary
    summary_path = Config.MODEL_V5_DIR / 'training_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results_by_coin, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training completed at {datetime.now()}")
    print(f"Results saved to {Config.MODEL_V5_DIR}")
    print(f"Summary saved to {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
