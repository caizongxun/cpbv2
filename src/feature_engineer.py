import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Technical indicator calculation for crypto trading data.
    Implements 35+ indicators for LSTM feature engineering.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.indicators = {}
        
    def add_price_indicators(self) -> pd.DataFrame:
        """Add basic price indicators (5)."""
        # Already in data: open, high, low, close, volume
        self.df['hl2'] = (self.df['high'] + self.df['low']) / 2
        self.df['hlc3'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        return self.df
    
    def add_moving_averages(self) -> pd.DataFrame:
        """Add SMA and EMA (10 indicators)."""
        for period in [10, 20, 50, 100, 200]:
            self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
        return self.df
    
    def add_momentum_indicators(self) -> pd.DataFrame:
        """Add RSI, MACD, Momentum, ROC, Stochastic (9 indicators)."""
        
        # RSI (Relative Strength Index)
        for period in [14, 21]:
            delta = self.df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd_line'] = ema12 - ema26
        self.df['macd_signal'] = self.df['macd_line'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd_line'] - self.df['macd_signal']
        
        # Momentum
        self.df['momentum_5'] = self.df['close'] - self.df['close'].shift(5)
        
        # ROC (Rate of Change)
        self.df['roc_12'] = ((self.df['close'] - self.df['close'].shift(12)) / self.df['close'].shift(12)) * 100
        
        # Stochastic Oscillator
        low_min = self.df['low'].rolling(window=14).min()
        high_max = self.df['high'].rolling(window=14).max()
        self.df['stoch_k'] = 100 * ((self.df['close'] - low_min) / (high_max - low_min))
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=3).mean()
        
        return self.df
    
    def add_volatility_indicators(self) -> pd.DataFrame:
        """Add Bollinger Bands and ATR (6 indicators)."""
        
        # Bollinger Bands
        sma20 = self.df['close'].rolling(window=20).mean()
        std20 = self.df['close'].rolling(window=20).std()
        self.df['bb_upper'] = sma20 + (std20 * 2)
        self.df['bb_middle'] = sma20
        self.df['bb_lower'] = sma20 - (std20 * 2)
        self.df['bb_width'] = self.df['bb_upper'] - self.df['bb_lower']
        self.df['bb_pct'] = (self.df['close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        # ATR (Average True Range)
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr_14'] = tr.rolling(window=14).mean()
        
        return self.df
    
    def add_trend_indicators(self) -> pd.DataFrame:
        """Add ADX, DI+/-, Keltner Channels, NATR (7 indicators)."""
        
        # ADX and DI+/-
        high_diff = self.df['high'].diff()
        low_diff = -self.df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = pd.concat([
            self.df['high'] - self.df['low'],
            abs(self.df['high'] - self.df['close'].shift()),
            abs(self.df['low'] - self.df['close'].shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=14).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / atr)
        
        self.df['di_plus'] = plus_di
        self.df['di_minus'] = minus_di
        
        # ADX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * (di_diff / di_sum)
        self.df['adx_14'] = dx.rolling(window=14).mean()
        
        # Keltner Channels
        ema20 = self.df['close'].ewm(span=20, adjust=False).mean()
        atr10 = tr.rolling(window=10).mean()
        self.df['kc_upper'] = ema20 + (atr10 * 2)
        self.df['kc_middle'] = ema20
        self.df['kc_lower'] = ema20 - (atr10 * 2)
        
        # NATR (Normalized ATR)
        self.df['natr'] = (atr / self.df['close']) * 100
        
        return self.df
    
    def add_volume_indicators(self) -> pd.DataFrame:
        """Add OBV, CMF, MFI, VPT (4 indicators)."""
        
        # OBV (On Balance Volume)
        self.df['obv'] = (np.sign(self.df['close'].diff()) * self.df['volume']).fillna(0).cumsum()
        
        # CMF (Chaikin Money Flow)
        mfm = ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / (self.df['high'] - self.df['low'])
        self.df['cmf_20'] = (mfm * self.df['volume']).rolling(window=20).sum() / self.df['volume'].rolling(window=20).sum()
        
        # MFI (Money Flow Index)
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        mf = tp * self.df['volume']
        pmf = np.where(tp > tp.shift(1), mf, 0)
        nmf = np.where(tp < tp.shift(1), mf, 0)
        
        pmf_14 = pd.Series(pmf).rolling(window=14).sum()
        nmf_14 = pd.Series(nmf).rolling(window=14).sum()
        mfi = 100 - (100 / (1 + pmf_14 / nmf_14))
        self.df['mfi_14'] = mfi
        
        # VPT (Volume Price Trend)
        self.df['vpt'] = self.df['volume'] * (self.df['close'].pct_change())
        
        return self.df
    
    def add_change_indicators(self) -> pd.DataFrame:
        """Add price and volume change indicators (3)."""
        self.df['price_change'] = self.df['close'].pct_change() * 100
        self.df['volume_change'] = self.df['volume'].pct_change() * 100
        self.df['close_change'] = self.df['close'].diff()
        
        return self.df
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all 35+ indicators."""
        logger.info("Calculating technical indicators...")
        
        self.add_price_indicators()
        self.add_moving_averages()
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_trend_indicators()
        self.add_volume_indicators()
        self.add_change_indicators()
        
        logger.info(f"Calculated {len(self.df.columns)} total columns")
        return self.df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns (excluding price/time)."""
        exclude = ['timestamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude]
    
    def get_df(self) -> pd.DataFrame:
        """Get final DataFrame with all indicators."""
        return self.df
