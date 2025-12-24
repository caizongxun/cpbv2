#!/usr/bin/env python3
"""
V3 Complete Training Pipeline for Colab (Binance data.binance.vision)
Steps: 1. Setup env 2. Download data from Binance data.binance.vision 3. Train models
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
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V3CoLabPipeline:
    """Complete V3 training pipeline using Binance data.binance.vision"""
    
    # 20 coins to train
    COINS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
        'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
        'AAVUSDT', 'ATOMUSDT', 'NEARUSDT', 'COMPUSDT', 'ARBUSDT',
        'OPUSDT', 'STXUSDT', 'INJUSDT', 'SHIBUSDT', 'LUNAUSDT'
    ]
    
    TIMEFRAMES = ['1h']
    
    def __init__(self, base_dir: str = '/content'):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.base_dir / 'all_models'
        self.results_dir = self.base_dir / 'results'
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Binance data.binance.vision base URL
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines"
    
    def step_1_setup_environment(self):
        """Step 1: Setup environment"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: 設定環境")
        logger.info("="*80)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("GPU 不可用。訓練將較緩慢。")
        
        logger.info(f"目錄已建立")
    
    def step_2_download_binance_data(self):
        """Step 2: Download data from Binance data.binance.vision"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: 下載 Binance 數據")
        logger.info("="*80)
        logger.info("數據源: Binance data.binance.vision (官方公開)")
        
        total_pairs = len(self.COINS) * len(self.TIMEFRAMES)
        completed = 0
        failed = []
        
        # Get last 6 months (more data for better training)
        now = datetime.now()
        months_to_download = []
        for i in range(6):
            month = now.month - i
            year = now.year
            if month <= 0:
                month += 12
                year -= 1
            months_to_download.append((year, month))
        
        for coin in self.COINS:
            for timeframe in self.TIMEFRAMES:
                try:
                    completed += 1
                    logger.info(f"[{completed}/{total_pairs}] 下載 {coin} {timeframe}")
                    
                    all_data = []
                    success_count = 0
                    
                    # Download multiple months
                    for year, month in months_to_download:
                        try:
                            month_str = f"{month:02d}"
                            url = f"{self.base_url}/{coin}/{timeframe}/{coin}-{timeframe}-{year}-{month_str}.zip"
                            
                            response = requests.get(url, timeout=30)
                            
                            if response.status_code == 200:
                                try:
                                    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                                        file_list = zip_ref.namelist()
                                        if file_list:
                                            csv_filename = file_list[0]
                                            with zip_ref.open(csv_filename) as f:
                                                df_month = pd.read_csv(f, header=None)
                                                if len(df_month) > 0:
                                                    all_data.append(df_month)
                                                    success_count += 1
                                except Exception as zip_error:
                                    logger.debug(f"    Zip error for {year}-{month_str}: {zip_error}")
                                    pass
                        except Exception as e:
                            logger.debug(f"    Request error for {year}-{month_str}: {e}")
                            pass
                    
                    if len(all_data) > 0:
                        # Combine all months
                        df = pd.concat(all_data, ignore_index=True)
                        
                        # Set column names
                        df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                                    'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore']
                        
                        # Convert to numeric
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Remove NaN rows
                        df = df.dropna()
                        
                        if len(df) > 0:
                            # Save
                            csv_path = self.data_dir / f"{coin}_{timeframe}.csv"
                            df[['Open', 'High', 'Low', 'Close', 'Volume']].to_csv(csv_path, index=False)
                            logger.info(f"  ✓ 已保存 {len(df)} 根K線 ({success_count} 個月)")
                        else:
                            logger.warning(f"  ✗ 數據為空")
                            failed.append(f"{coin}_{timeframe}")
                    else:
                        logger.error(f"  ✗ 無法下載任何數據")
                        failed.append(f"{coin}_{timeframe}")
                    
                except Exception as e:
                    logger.error(f"  ✗ 異常: {e}")
                    failed.append(f"{coin}_{timeframe}")
        
        logger.info(f"\n下載摘要: {total_pairs - len(failed)}/{total_pairs} 成功")
        if failed:
            logger.warning(f"失敗: {failed}")
        
        return len(failed) < (total_pairs * 0.3)  # Success if at least 70% downloaded
    
    def step_3_train_models(self, epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        """Step 3: Train models"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: 訓練模型")
        logger.info("="*80)
        
        # Get available data files
        data_files = sorted(list(self.data_dir.glob("*_1h.csv")))
        logger.info(f"找到 {len(data_files)} 個數據文件")
        
        if len(data_files) == 0:
            logger.error("沒有找到任何數據文件！")
            return False
        
        training_results = {}
        
        for idx, csv_file in enumerate(data_files, 1):
            pair_name = csv_file.stem
            
            logger.info(f"\n[{idx}/{len(data_files)}] 訓練 {pair_name}")
            
            try:
                df = pd.read_csv(csv_file)
                
                if len(df) < 200:
                    logger.warning(f"數據不足: {len(df)} 列")
                    training_results[pair_name] = {'status': 'skipped', 'reason': 'insufficient_data'}
                    continue
                
                # Simple preprocessing
                close_prices = df['Close'].values.astype(np.float32)
                
                # Normalize
                mean_price = close_prices.mean()
                std_price = close_prices.std()
                normalized = (close_prices - mean_price) / (std_price + 1e-8)
                
                # Create sequences
                seq_length = 30
                X, y = [], []
                for i in range(len(normalized) - seq_length):
                    X.append(normalized[i:i+seq_length])
                    y.append(normalized[i+seq_length])
                
                X = np.array(X, dtype=np.float32)
                y = np.array(y, dtype=np.float32)
                
                if len(X) < 50:
                    logger.warning(f"序列不足: {len(X)}")
                    training_results[pair_name] = {'status': 'skipped', 'reason': 'insufficient_sequences'}
                    continue
                
                # Split data
                train_size = int(len(X) * 0.7)
                val_size = int(len(X) * 0.15)
                
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_val = X[train_size:train_size+val_size]
                y_val = y[train_size:train_size+val_size]
                
                # Create dataloaders
                train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
                val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Simple LSTM model
                class SimpleLSTM(nn.Module):
                    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                        self.fc = nn.Linear(hidden_size, 1)
                    
                    def forward(self, x):
                        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
                        lstm_out, _ = self.lstm(x)
                        last_out = lstm_out[:, -1, :]
                        pred = self.fc(last_out)
                        return pred.squeeze()
                
                # Train
                model = SimpleLSTM().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()
                
                best_val_loss = float('inf')
                patience_counter = 0
                patience = 10
                
                for epoch in range(epochs):
                    # Train
                    model.train()
                    train_loss = 0
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        
                        optimizer.zero_grad()
                        pred = model(X_batch)
                        loss = criterion(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    # Validate
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                            pred = model(X_batch)
                            loss = criterion(pred, y_batch)
                            val_loss += loss.item()
                    
                    train_loss /= len(train_loader)
                    val_loss /= len(val_loader)
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save model
                        model_path = self.models_dir / f"v3_model_{pair_name}.pt"
                        torch.save(model.state_dict(), model_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"  早停在 epoch {epoch+1}")
                            break
                
                training_results[pair_name] = {
                    'status': 'success',
                    'best_val_loss': float(best_val_loss),
                    'epochs_trained': epoch + 1
                }
                
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"訓練 {pair_name} 失敗: {e}")
                training_results[pair_name] = {'status': 'failed', 'error': str(e)}
        
        # Save results
        results_path = self.results_dir / 'v3_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"\n訓練結果已保存")
        
        # Summary
        successful = sum(1 for r in training_results.values() if r['status'] == 'success')
        logger.info(f"\n訓練摘要: {successful}/{len(training_results)} 成功")
        
        return successful > 0
    
    def run_full_pipeline(self, epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        """Run complete pipeline"""
        logger.info("\n" + "#"*80)
        logger.info("# V3 Binance data.binance.vision 訓練")
        logger.info("#"*80)
        
        # Step 1
        self.step_1_setup_environment()
        
        # Step 2
        if not self.step_2_download_binance_data():
            logger.warning("警告: 部分數據下載失敗，但繼續使用可用數據")
        
        # Step 3
        if not self.step_3_train_models(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate):
            logger.error("訓練失敗")
            return False
        
        logger.info("\n" + "#"*80)
        logger.info("# 訓練完成")
        logger.info("#"*80)
        
        return True


if __name__ == "__main__":
    pipeline = V3CoLabPipeline()
    success = pipeline.run_full_pipeline(epochs=50, batch_size=32, learning_rate=0.001)
    sys.exit(0 if success else 1)
