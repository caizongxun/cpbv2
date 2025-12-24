#!/usr/bin/env python3
"""
V3 Advanced Data Processor for Cryptocurrency
Features: Enhanced technical indicators, volatility measures, market microstructure
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict
import logging


class V3DataProcessor:
    """Advanced data processor with V3 features"""
    
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.pca = None
        self.feature_names = []
        self.mean = None
        self.std = None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = df.copy()
        
        # Verify required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 1. Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [10, 20]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # 2. Volatility indicators
        # Historical volatility
        for period in [10, 20]:
            df[f'HV_{period}'] = df['Log_Returns'].rolling(window=period).std() * np.sqrt(252)
        
        # Bollinger Bands
        bb_period = 20
        bb_sma = df['Close'].rolling(window=bb_period).mean()
        bb_std = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = bb_sma + (2 * bb_std)
        df['BB_Lower'] = bb_sma - (2 * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_sma
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # 3. Momentum indicators
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['Close'], period=14)
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        # Rate of Change
        df['ROC'] = (df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)
        
        # 4. Volume indicators
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-8)
            
            # On Balance Volume
            obv = np.zeros(len(df))
            obv[0] = 0
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] + df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] - df['Volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
            df['OBV'] = obv
            df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
        
        # 5. Price action indicators
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
        df['True_Range'] = np.maximum(
            np.maximum(df['High'] - df['Low'], 
                      np.abs(df['High'] - df['Close'].shift(1))),
            np.abs(df['Low'] - df['Close'].shift(1))
        )
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # 6. Mean reversion indicators
        df['Z_Score'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).std() + 1e-8)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        seq_length: int = 60,
        prediction_horizon: int = 1,
        target_col: str = 'Close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training - returns 2D (samples, features)"""
        
        # Drop rows with NaN from indicator calculation
        df = df.dropna()
        
        # Select features (exclude price and volume columns for normalization)
        feature_cols = [col for col in df.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'OpenTime', 'CloseTime']]
        
        if len(feature_cols) == 0:
            raise ValueError("No features found after dropping standard columns")
        
        X_data = df[feature_cols].values  # (total_samples, n_features)
        y_data = df[target_col].values
        
        # Normalize features
        X_normalized = self.scaler.fit_transform(X_data)
        
        # Normalize target separately for MAPE calculation
        y_scaler = MinMaxScaler()
        y_normalized = y_scaler.fit_transform(y_data.reshape(-1, 1)).squeeze()
        
        self.mean = X_normalized.mean(axis=0)
        self.std = X_normalized.std(axis=0) + 1e-8
        
        # Apply z-score normalization after min-max
        X_normalized = (X_normalized - self.mean) / self.std
        
        # Create sequences - 2D output: (sequence_samples, features)
        # This is used with TimeSeriesSplit or just passed to LSTM directly
        X_seq = []
        y_seq = []
        
        for i in range(len(X_normalized) - seq_length - prediction_horizon + 1):
            # Take averaged features over sequence length as input
            X_seq.append(X_normalized[i:i+seq_length].mean(axis=0))
            y_seq.append(y_normalized[i+seq_length+prediction_horizon-1])
        
        self.feature_names = feature_cols
        
        return np.array(X_seq), np.array(y_seq)
    
    def apply_pca(
        self,
        X: np.ndarray,
        n_components: int = 30,
        fit: bool = True
    ) -> np.ndarray:
        """Apply PCA for dimensionality reduction
        Input: 2D array (samples, features)
        Output: 2D array (samples, n_components)
        """
        
        # Ensure 2D input
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got {X.ndim}D")
        
        if fit:
            n_comp = min(n_components, X.shape[1])
            self.pca = PCA(n_components=n_comp)
            X_pca = self.pca.fit_transform(X)
            explained_var = self.pca.explained_variance_ratio_.sum()
            logging.info(f"PCA: {X.shape[1]} -> {n_comp} components, explained variance: {explained_var*100:.2f}%")
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit=True first.")
            X_pca = self.pca.transform(X)
        
        return X_pca
    
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Dict[str, np.ndarray]:
        """Split data into train, validation, and test sets"""
        
        total_samples = len(X)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logging.info(f"Data split: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    processor = V3DataProcessor()
    print("V3 Data Processor initialized successfully")
