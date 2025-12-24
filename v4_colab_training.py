#!/usr/bin/env python3
"""
V4 Training Pipeline for Colab
Goal: Train Seq2Seq LSTM with Attention to predict next 10 OHLC candles
Features:
  - Input: 30 historical OHLC candles
  - Output: 10 future OHLC candles (Open, High, Low, Close)
  - Data: 7000-10000 bars per coin
  - Timeframes: 15m, 1h
  - 20 cryptocurrencies
  - Early stopping + dropout to prevent overfitting
"""

import os
import sys
import logging
import json
import warnings
import zipfile
import requests
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import model
from v4_model_architecture import Seq2SeqLSTM


class V4CoLabPipeline:
    """V4 Training Pipeline with Seq2Seq"""
    
    # 20 coins (same as before)
    COINS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
        'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
        'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'DYDXUSDT', 'ARBUSDT',
        'OPUSDT', 'PEPEUSDT', 'INJUSDT', 'SHIBUSDT', 'LUNAUSDT'
    ]
    
    TIMEFRAMES = ['15m', '1h']
    
    def __init__(self, base_dir: str = '/content'):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data_v4'
        self.models_dir = self.base_dir / 'v4_models'
        self.results_dir = self.base_dir / 'v4_results'
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Binance data.binance.vision base URL
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines"
    
    def step_1_setup_environment(self):
        """Step 1: Setup environment"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: 設定環境")
        logger.info("="*80)
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        logger.info(f"目錄已建立")
    
    def step_2_download_binance_data(self):
        """Step 2: Download data from Binance (more data: 7000-10000 bars)"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: 下載 Binance 數據 (需要 7000-10000 根K線)")
        logger.info("="*80)
        logger.info(f"幣種: {len(self.COINS)} 個, 時間: {self.TIMEFRAMES}")
        
        total_pairs = len(self.COINS) * len(self.TIMEFRAMES)
        completed = 0
        successful = []
        failed = []
        
        # Download more months to get 7000+ bars
        now = datetime.now()
        months_to_download = []
        for i in range(12):  # 12 months of data
            month = now.month - i
            year = now.year
            if month <= 0:
                month += 12
                year -= 1
            months_to_download.append((year, month))
        
        logger.info(f"下載月份: {len(months_to_download)} 個月\n")
        
        for coin in self.COINS:
            for timeframe in self.TIMEFRAMES:
                completed += 1
                logger.info(f"[{completed}/{total_pairs}] {coin} {timeframe}...", end='')
                
                all_data = []
                
                for year, month in months_to_download:
                    try:
                        month_str = f"{month:02d}"
                        url = f"{self.base_url}/{coin}/{timeframe}/{coin}-{timeframe}-{year}-{month_str}.zip"
                        response = requests.get(url, timeout=30)
                        
                        if response.status_code == 200:
                            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                                file_list = zip_ref.namelist()
                                if file_list:
                                    with zip_ref.open(file_list[0]) as f:
                                        df_month = pd.read_csv(f, header=None)
                                        if len(df_month) > 0:
                                            all_data.append(df_month)
                    except:
                        pass
                
                if len(all_data) > 0:
                    df = pd.concat(all_data, ignore_index=True)
                    df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                                'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore']
                    
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    
                    if len(df) >= 7000:  # Require at least 7000 bars
                        csv_path = self.data_dir / f"{coin}_{timeframe}.csv"
                        df[['Open', 'High', 'Low', 'Close', 'Volume']].to_csv(csv_path, index=False)
                        logger.info(f" ✓ {len(df)} K線")
                        successful.append(f"{coin}_{timeframe}")
                    else:
                        logger.info(f" ✗ 數據不足 ({len(df)} < 7000)")
                        failed.append(f"{coin}_{timeframe}")
                else:
                    logger.info(f" ✗ 無法下載")
                    failed.append(f"{coin}_{timeframe}")
        
        logger.info(f"\n下載摘要: {len(successful)}/{total_pairs} 成功")
        if failed:
            logger.info(f"失敗: {failed[:5]}..." if len(failed) > 5 else f"失敗: {failed}")
        
        return len(successful) > 0
    
    def step_3_train_models(self, epochs: int = 200, batch_size: int = 16, learning_rate: float = 0.001):
        """Step 3: Train V4 models"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: 訓練 V4 模型 (Seq2Seq + Attention)")
        logger.info("="*80)
        logger.info(f"預測目標: 30根 K線 → 下一个 10根 K線")
        logger.info(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}\n")
        
        data_files = sorted(list(self.data_dir.glob("*.csv")))
        logger.info(f"找到 {len(data_files)} 個數據文件\n")
        
        training_results = {}
        
        for idx, csv_file in enumerate(data_files, 1):
            pair_name = csv_file.stem
            logger.info(f"[{idx}/{len(data_files)}] 訓練 {pair_name}")
            
            try:
                df = pd.read_csv(csv_file)
                
                if len(df) < 7000:
                    logger.warning(f"  數據不足: {len(df)}")
                    continue
                
                # Use last 7000 bars
                df = df.tail(7000)
                
                # Normalize OHLC (per-candle normalization)
                data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
                
                # Normalize each candle individually
                normalized_data = np.zeros_like(data)
                for i in range(len(data)):
                    min_val = data[i].min()
                    max_val = data[i].max()
                    if max_val > min_val:
                        normalized_data[i] = (data[i] - min_val) / (max_val - min_val)
                    else:
                        normalized_data[i] = data[i]
                
                # Create sequences: 30 input, 10 output
                seq_length_in = 30
                seq_length_out = 10
                X, y = [], []
                
                for i in range(len(normalized_data) - seq_length_in - seq_length_out):
                    X.append(normalized_data[i:i+seq_length_in])
                    y.append(normalized_data[i+seq_length_in:i+seq_length_in+seq_length_out])
                
                X = np.array(X, dtype=np.float32)
                y = np.array(y, dtype=np.float32)
                
                if len(X) < 100:
                    logger.warning(f"  序列不足: {len(X)}")
                    continue
                
                # Split data
                train_size = int(len(X) * 0.7)
                val_size = int(len(X) * 0.15)
                
                X_train, y_train = X[:train_size], y[:train_size]
                X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
                
                # Create dataloaders
                train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
                val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Create model
                model = Seq2SeqLSTM(
                    input_size=4,
                    hidden_size=128,
                    num_layers=2,
                    dropout=0.3,
                    steps_ahead=10,
                    output_size=4
                ).to(self.device)
                
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                                 patience=10, verbose=False)
                
                # Training loop with early stopping
                best_val_loss = float('inf')
                patience = 20
                patience_counter = 0
                
                for epoch in range(epochs):
                    # Train
                    model.train()
                    train_loss = 0
                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        optimizer.zero_grad()
                        pred = model(X_batch, y_batch, teacher_forcing_ratio=0.5)
                        loss = criterion(pred, y_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    # Validate
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch = X_batch.to(self.device)
                            y_batch = y_batch.to(self.device)
                            pred = model(X_batch, y_batch, teacher_forcing_ratio=0)
                            loss = criterion(pred, y_batch)
                            val_loss += loss.item()
                    
                    train_loss /= len(train_loader)
                    val_loss /= len(val_loader)
                    scheduler.step(val_loss)
                    
                    if (epoch + 1) % 20 == 0:
                        logger.info(f"  Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        model_path = self.models_dir / f"v4_model_{pair_name}.pt"
                        torch.save(model.state_dict(), model_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"  早停 (epoch {epoch+1})")
                            break
                
                training_results[pair_name] = {
                    'status': 'success',
                    'best_val_loss': float(best_val_loss),
                    'epochs_trained': epoch + 1,
                    'data_points': len(X)
                }
                
                logger.info(f"  ✓ 完成 - Val Loss: {best_val_loss:.6f}")
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"  ✗ {str(e)[:60]}")
                training_results[pair_name] = {'status': 'failed', 'error': str(e)}
        
        # Save results
        results_path = self.results_dir / 'v4_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        successful = sum(1 for r in training_results.values() if r['status'] == 'success')
        logger.info(f"\n訓練摘要: {successful}/{len(data_files)} 成功")
        
        return successful > 0
    
    def run_full_pipeline(self, epochs: int = 200, batch_size: int = 16, learning_rate: float = 0.001):
        """Run complete pipeline"""
        logger.info("\n" + "#"*80)
        logger.info("# V4 Training Pipeline - Seq2Seq LSTM with Attention")
        logger.info("# Goal: Predict next 10 OHLC candles based on previous 30")
        logger.info("#"*80)
        
        self.step_1_setup_environment()
        
        if not self.step_2_download_binance_data():
            logger.error("數據下載失敗")
            return False
        
        if not self.step_3_train_models(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate):
            logger.error("訓練失敗")
            return False
        
        logger.info("\n" + "#"*80)
        logger.info("# V4 訓練完成")
        logger.info("#"*80)
        
        return True


if __name__ == "__main__":
    pipeline = V4CoLabPipeline()
    success = pipeline.run_full_pipeline(epochs=200, batch_size=16, learning_rate=0.001)
    sys.exit(0 if success else 1)
