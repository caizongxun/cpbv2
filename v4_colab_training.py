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

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
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
        logger.info(f"使用設備: {self.device}")
        
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
        
        logger.debug(f"數據目錄: {self.data_dir}")
        logger.debug(f"模型目錄: {self.models_dir}")
        logger.debug(f"結果目錄: {self.results_dir}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("使用 CPU (無 GPU 可用)")
        
        logger.info("目錄已建立")
    
    def step_2_download_binance_data(self):
        """Step 2: Download data from Binance (more data: 7000-10000 bars)"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: 下載 Binance 數據 (需要 7000-10000 根K線)")
        logger.info("="*80)
        logger.info(f"幣種: {len(self.COINS)} 個, 時間框: {self.TIMEFRAMES}")
        
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
        
        logger.info(f"下載月份: {len(months_to_download)} 個月")
        logger.debug(f"月份範圍: {months_to_download[0]} ~ {months_to_download[-1]}\n")
        
        for coin in self.COINS:
            for timeframe in self.TIMEFRAMES:
                completed += 1
                logger.info(f"[{completed}/{total_pairs}] {coin} {timeframe}...", end='')
                
                all_data = []
                months_found = 0
                
                for year, month in months_to_download:
                    try:
                        month_str = f"{month:02d}"
                        url = f"{self.base_url}/{coin}/{timeframe}/{coin}-{timeframe}-{year}-{month_str}.zip"
                        logger.debug(f"  嘗試下載: {year}-{month_str}")
                        response = requests.get(url, timeout=30)
                        
                        if response.status_code == 200:
                            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                                file_list = zip_ref.namelist()
                                if file_list:
                                    with zip_ref.open(file_list[0]) as f:
                                        df_month = pd.read_csv(f, header=None)
                                        if len(df_month) > 0:
                                            all_data.append(df_month)
                                            months_found += 1
                                            logger.debug(f"    成功: {len(df_month)} 行")
                    except Exception as e:
                        logger.debug(f"  {year}-{month_str} 失敗: {str(e)[:40]}")
                
                if len(all_data) > 0:
                    df = pd.concat(all_data, ignore_index=True)
                    df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                                'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore']
                    
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    logger.debug(f"  合併後: {len(df)} 行, 從 {months_found} 個月")
                    
                    if len(df) >= 7000:
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
            logger.debug(f"失敗清單: {failed[:5]}..." if len(failed) > 5 else f"失敗清單: {failed}")
        
        return len(successful) > 0
    
    def step_3_train_models(self, epochs: int = 200, batch_size: int = 16, learning_rate: float = 0.001):
        """Step 3: Train V4 models"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: 訓練 V4 模型 (Seq2Seq + Attention)")
        logger.info("="*80)
        logger.info(f"預測目標: 30根 K線 → 下一個 10根 K線")
        logger.info(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}\n")
        
        data_files = sorted(list(self.data_dir.glob("*.csv")))
        logger.info(f"找到 {len(data_files)} 個數據文件\n")
        
        if len(data_files) == 0:
            logger.error("沒有找到數據文件!")
            return False
        
        training_results = {}
        
        for idx, csv_file in enumerate(data_files, 1):
            pair_name = csv_file.stem
            logger.info(f"[{idx}/{len(data_files)}] 訓練 {pair_name}")
            
            try:
                # 讀取數據
                logger.debug(f"  讀取: {csv_file}")
                df = pd.read_csv(csv_file)
                logger.debug(f"  原始行數: {len(df)}")
                
                if len(df) < 7000:
                    logger.warning(f"  數據不足: {len(df)}")
                    training_results[pair_name] = {'status': 'insufficient_data', 'rows': len(df)}
                    continue
                
                # Use last 7000 bars
                df = df.tail(7000)
                logger.debug(f"  使用最後 7000 行")
                
                # Normalize OHLC (per-candle normalization)
                data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
                logger.debug(f"  OHLC 數據範圍: [{data.min():.2f}, {data.max():.2f}]")
                
                # Normalize each candle individually
                normalized_data = np.zeros_like(data)
                for i in range(len(data)):
                    min_val = data[i].min()
                    max_val = data[i].max()
                    if max_val > min_val:
                        normalized_data[i] = (data[i] - min_val) / (max_val - min_val)
                    else:
                        normalized_data[i] = data[i]
                
                logger.debug(f"  標準化完成: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
                
                # Create sequences: 30 input, 10 output
                seq_length_in = 30
                seq_length_out = 10
                X, y = [], []
                
                for i in range(len(normalized_data) - seq_length_in - seq_length_out):
                    X.append(normalized_data[i:i+seq_length_in])
                    y.append(normalized_data[i+seq_length_in:i+seq_length_in+seq_length_out])
                
                X = np.array(X, dtype=np.float32)
                y = np.array(y, dtype=np.float32)
                logger.debug(f"  序列: X={X.shape}, y={y.shape}")
                
                if len(X) < 100:
                    logger.warning(f"  序列不足: {len(X)}")
                    training_results[pair_name] = {'status': 'insufficient_sequences', 'sequences': len(X)}
                    continue
                
                # Split data
                train_size = int(len(X) * 0.7)
                val_size = int(len(X) * 0.15)
                
                X_train, y_train = X[:train_size], y[:train_size]
                X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
                logger.debug(f"  訓練: {len(X_train)}, 驗證: {len(X_val)}, 測試: {len(X)-train_size-val_size}")
                
                # Create dataloaders
                train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
                val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                logger.debug(f"  DataLoader 創建完成")
                
                # Create model
                logger.debug(f"  創建模型...")
                model = Seq2SeqLSTM(
                    input_size=4,
                    hidden_size=128,
                    num_layers=2,
                    dropout=0.3,
                    steps_ahead=10,
                    output_size=4
                ).to(self.device)
                
                total_params = sum(p.numel() for p in model.parameters())
                logger.debug(f"  模型參數: {total_params:,}")
                
                # Optimizer
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
                criterion = nn.MSELoss()
                logger.debug(f"  優化器: Adam, LR={learning_rate}, weight_decay=1e-5")
                
                # 使用簡單的學習率調度 (StepLR - 不使用 ReduceLROnPlateau)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
                logger.debug(f"  學習率調度: StepLR (每30個epoch減半)")
                
                # Training loop with early stopping
                best_val_loss = float('inf')
                patience = 20
                patience_counter = 0
                best_epoch = 0
                
                logger.debug(f"  開始訓練 (early stopping patience={patience})...")
                
                for epoch in range(epochs):
                    # Train
                    model.train()
                    train_loss = 0
                    train_batches = 0
                    
                    try:
                        for X_batch, y_batch in train_loader:
                            X_batch = X_batch.to(self.device)
                            y_batch = y_batch.to(self.device)
                            
                            optimizer.zero_grad()
                            pred = model(X_batch, y_batch, teacher_forcing_ratio=0.5)
                            loss = criterion(pred, y_batch)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            
                            train_loss += loss.item()
                            train_batches += 1
                        
                        train_loss /= train_batches if train_batches > 0 else 1
                    except Exception as e:
                        logger.error(f"  訓練批次失敗: {str(e)[:60]}")
                        raise
                    
                    # Validate
                    model.eval()
                    val_loss = 0
                    val_batches = 0
                    
                    try:
                        with torch.no_grad():
                            for X_batch, y_batch in val_loader:
                                X_batch = X_batch.to(self.device)
                                y_batch = y_batch.to(self.device)
                                pred = model(X_batch, y_batch, teacher_forcing_ratio=0)
                                loss = criterion(pred, y_batch)
                                val_loss += loss.item()
                                val_batches += 1
                        
                        val_loss /= val_batches if val_batches > 0 else 1
                    except Exception as e:
                        logger.error(f"  驗證批次失敗: {str(e)[:60]}")
                        raise
                    
                    scheduler.step()
                    
                    if (epoch + 1) % 20 == 0:
                        logger.info(f"  Epoch {epoch+1:3d}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
                    else:
                        logger.debug(f"  Epoch {epoch+1:3d}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1
                        patience_counter = 0
                        
                        model_path = self.models_dir / f"v4_model_{pair_name}.pt"
                        torch.save(model.state_dict(), model_path)
                        logger.debug(f"    保存模型: {model_path}")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.debug(f"  早停 (epoch {epoch+1}, patience 已滿)")
                            break
                
                training_results[pair_name] = {
                    'status': 'success',
                    'best_val_loss': float(best_val_loss),
                    'best_epoch': best_epoch,
                    'epochs_trained': epoch + 1,
                    'data_points': len(X),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val)
                }
                
                logger.info(f"  ✓ 完成 - Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch})")
                torch.cuda.empty_cache()
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"  ✗ 錯誤: {error_msg[:80]}")
                logger.debug(f"    完整錯誤: {error_msg}")
                training_results[pair_name] = {'status': 'failed', 'error': error_msg}
        
        # Save results
        results_path = self.results_dir / 'v4_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        logger.debug(f"結果已保存: {results_path}")
        
        successful = sum(1 for r in training_results.values() if r['status'] == 'success')
        logger.info(f"\n訓練摘要: {successful}/{len(data_files)} 成功")
        
        return successful > 0
    
    def run_full_pipeline(self, epochs: int = 200, batch_size: int = 16, learning_rate: float = 0.001):
        """Run complete pipeline"""
        logger.info("\n" + "#"*80)
        logger.info("# V4 Training Pipeline - Seq2Seq LSTM with Attention")
        logger.info("# Goal: Predict next 10 OHLC candles based on previous 30")
        logger.info("#"*80)
        
        try:
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
        
        except Exception as e:
            logger.error(f"管道執行失敗: {str(e)}")
            logger.debug(f"完整錯誤信息: ", exc_info=True)
            return False


if __name__ == "__main__":
    pipeline = V4CoLabPipeline()
    success = pipeline.run_full_pipeline(epochs=200, batch_size=16, learning_rate=0.001)
    sys.exit(0 if success else 1)
