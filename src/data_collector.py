import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Optional, Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceDataCollector:
    """
    Binance API data collector with retry logic and data validation.
    Optimized for Colab environment.
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    MAX_CANDLES_PER_REQUEST = 1000
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        
    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "15m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 3000,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Download historical K-line data from Binance.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Time interval ("15m", "1h", "4h", "1d")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Number of candles to download
            max_retries: Max retry attempts
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Downloading {symbol} {interval} data...")
        
        # Default date range: last 3 months
        if end_date is None:
            end_date = datetime.utcnow()
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        if start_date is None:
            start_date = end_date - timedelta(days=90)
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        all_klines = []
        current_start = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        retry_count = 0
        
        while current_start < end_ms and len(all_klines) < limit:
            try:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "limit": min(self.MAX_CANDLES_PER_REQUEST, limit - len(all_klines))
                }
                
                response = self.session.get(
                    f"{self.BASE_URL}/klines",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                
                klines = response.json()
                if not klines:
                    logger.warning(f"No more data for {symbol}")
                    break
                    
                all_klines.extend(klines)
                
                # Update start time for next batch
                current_start = int(klines[-1][0]) + 1
                
                retry_count = 0  # Reset retry counter on success
                time.sleep(0.1)  # Rate limiting
                
                logger.info(f"Downloaded {len(all_klines)} candles...")
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to download {symbol} after {max_retries} retries: {e}")
                    break
                    
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Retry {retry_count}/{max_retries} for {symbol} after {wait_time}s")
                time.sleep(wait_time)
        
        # Convert to DataFrame
        if not all_klines:
            logger.error(f"No data collected for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(
            all_klines,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'num_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
        )
        
        # Data cleaning
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df['interval'] = interval
        
        # Convert to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Drop unnecessary columns
        df = df[['timestamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume']]
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Collected {len(df)} candles for {symbol}")
        return df
    
    def download_multiple_coins(
        self,
        coins: List[str],
        intervals: List[str] = ["15m", "1h"],
        output_dir: str = "data/raw"
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download data for multiple coins and timeframes.
        
        Args:
            coins: List of coin symbols
            intervals: List of timeframes
            output_dir: Directory to save CSV files
            
        Returns:
            Dictionary of DataFrames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = {}
        
        for coin in coins:
            all_data[coin] = {}
            
            for interval in intervals:
                try:
                    df = self.get_historical_klines(coin, interval, limit=3000)
                    all_data[coin][interval] = df
                    
                    # Save to CSV
                    filename = f"{output_dir}/{coin}_{interval}.csv"
                    df.to_csv(filename, index=False)
                    logger.info(f"Saved {filename}")
                    
                except Exception as e:
                    logger.error(f"Error downloading {coin} {interval}: {e}")
                    all_data[coin][interval] = pd.DataFrame()
        
        return all_data
    
    @staticmethod
    def validate_data(df: pd.DataFrame, min_candles: int = 3000) -> bool:
        """
        Validate downloaded data.
        
        Args:
            df: DataFrame to validate
            min_candles: Minimum required candles
            
        Returns:
            True if valid, False otherwise
        """
        # Check for minimum candles
        if len(df) < min_candles:
            logger.warning(f"Insufficient candles: {len(df)} < {min_candles}")
            return False
        
        # Check for NaN values
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            logger.warning("Found NaN values in OHLCV data")
            return False
        
        # Check timestamp ordering
        if not df['timestamp'].is_monotonic_increasing:
            logger.warning("Timestamps are not in ascending order")
            return False
        
        logger.info(f"Data validation passed: {len(df)} candles")
        return True
