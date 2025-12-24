#!/usr/bin/env python3
"""
V4 Training Pipeline for Colab - GPU Optimized & Clean
Goal: Train Seq2Seq LSTM with Attention to predict next 10 OHLC candles
GPU Memory: Limited to 13GB
"""

import os
import sys
import logging
import json
import warnings
import zipfile
import requests
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from io import BytesIO

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

from v4_model_architecture import Seq2SeqLSTM


class V4Pipeline:
    def __init__(self, base_dir='/content'):
        if not torch.cuda.is_available():
            print("ERROR: GPU not available!")
            sys.exit(1)
        
        self.device = torch.device('cuda')
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data_v4'
        self.models_dir = self.base_dir / 'v4_models'
        self.results_dir = self.base_dir / 'v4_results'
        
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print()
    
    def setup(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print("✓ Directories created")
    
    def download_data(self):
        print("\nDownloading data...")
        
        COINS = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
            'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
            'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'DYDXUSDT', 'ARBUSDT',
            'OPUSDT', 'PEPEUSDT', 'INJUSDT', 'SHIBUSDT', 'LUNAUSDT'
        ]
        TIMEFRAMES = ['15m', '1h']
        
        base_url = "https://data.binance.vision/data/spot/monthly/klines"
        total = len(COINS) * len(TIMEFRAMES)
        completed = 0
        successful = 0
        
        now = datetime.now()
        months = []
        for i in range(12):
            month = now.month - i
            year = now.year
            if month <= 0:
                month += 12
                year -= 1
            months.append((year, month))
        
        for coin in COINS:
            for tf in TIMEFRAMES:
                completed += 1
                print(f"[{completed}/{total}] {coin} {tf}...", end=' ')
                sys.stdout.flush()
                
                all_data = []
                for year, month in months:
                    try:
                        month_str = f"{month:02d}"
                        url = f"{base_url}/{coin}/{tf}/{coin}-{tf}-{year}-{month_str}.zip"
                        r = requests.get(url, timeout=30)
                        if r.status_code == 200:
                            with zipfile.ZipFile(BytesIO(r.content)) as z:
                                for name in z.namelist():
                                    with z.open(name) as f:
                                        df = pd.read_csv(f, header=None)
                                        if len(df) > 0:
                                            all_data.append(df)
                                            break
                    except:
                        pass
                
                if all_data:
                    df = pd.concat(all_data, ignore_index=True)
                    df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                                'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore']
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna()
                    
                    if len(df) >= 7000:
                        path = self.data_dir / f"{coin}_{tf}.csv"
                        df[['Open', 'High', 'Low', 'Close', 'Volume']].to_csv(path, index=False)
                        print(f"✓ {len(df)} bars")
                        successful += 1
                    else:
                        print(f"✗ {len(df)} bars (need 7000)")
                else:
                    print(f"✗ Failed")
        
        print(f"\nDownloaded: {successful}/{total} pairs")
        return successful > 0
    
    def train(self, epochs=200, lr=0.001):
        print("\nStarting training...\n")
        
        files = sorted(list(self.data_dir.glob("*.csv")))
        if not files:
            print("No data files found!")
            return False
        
        print(f"Found {len(files)} files\n")
        
        results = {}
        batch_size = 8
        
        for idx, csv_file in enumerate(files, 1):
            pair_name = csv_file.stem
            print(f"[{idx}/{len(files)}] Training {pair_name}...")
            sys.stdout.flush()
            
            try:
                df = pd.read_csv(csv_file)
                if len(df) < 7000:
                    print(f"  ✗ Insufficient data: {len(df)}")
                    continue
                
                df = df.tail(7000)
                data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
                
                normalized_data = np.zeros_like(data)
                for i in range(len(data)):
                    min_val = data[i].min()
                    max_val = data[i].max()
                    if max_val > min_val:
                        normalized_data[i] = (data[i] - min_val) / (max_val - min_val)
                    else:
                        normalized_data[i] = data[i]
                
                seq_len_in, seq_len_out = 30, 10
                X, y = [], []
                for i in range(len(normalized_data) - seq_len_in - seq_len_out):
                    X.append(normalized_data[i:i+seq_len_in])
                    y.append(normalized_data[i+seq_len_in:i+seq_len_in+seq_len_out])
                
                X = np.array(X, dtype=np.float32)
                y = np.array(y, dtype=np.float32)
                
                if len(X) < 100:
                    print(f"  ✗ Insufficient sequences: {len(X)}")
                    continue
                
                train_size = int(len(X) * 0.7)
                val_size = int(len(X) * 0.15)
                
                X_train, y_train = X[:train_size], y[:train_size]
                X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
                
                train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
                val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
                
                model = Seq2SeqLSTM(input_size=4, hidden_size=96, num_layers=2, dropout=0.3, steps_ahead=10, output_size=4)
                model = model.to(self.device)
                
                print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
                
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
                
                best_val_loss = float('inf')
                patience = 20
                patience_count = 0
                best_epoch = 0
                
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0
                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(self.device, non_blocking=True)
                        y_batch = y_batch.to(self.device, non_blocking=True)
                        optimizer.zero_grad()
                        pred = model(X_batch, y_batch, teacher_forcing_ratio=0.5)
                        loss = criterion(pred, y_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        train_loss += loss.item()
                    train_loss /= len(train_loader)
                    
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch = X_batch.to(self.device, non_blocking=True)
                            y_batch = y_batch.to(self.device, non_blocking=True)
                            pred = model(X_batch, y_batch, teacher_forcing_ratio=0)
                            loss = criterion(pred, y_batch)
                            val_loss += loss.item()
                    val_loss /= len(val_loader)
                    
                    scheduler.step()
                    
                    if (epoch + 1) % 20 == 0:
                        print(f"  Epoch {epoch+1:3d} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
                        sys.stdout.flush()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1
                        patience_count = 0
                        model_path = self.models_dir / f"v4_model_{pair_name}.pt"
                        torch.save(model.state_dict(), model_path)
                    else:
                        patience_count += 1
                        if patience_count >= patience:
                            break
                
                results[pair_name] = {
                    'status': 'success',
                    'best_val_loss': float(best_val_loss),
                    'best_epoch': best_epoch,
                    'epochs_trained': epoch + 1
                }
                
                print(f"  ✓ Done - Best Loss: {best_val_loss:.6f}")
                
                del model, optimizer, criterion, scheduler
                del train_loader, val_loader
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:80]}")
                results[pair_name] = {'status': 'failed', 'error': str(e)}
            
            sys.stdout.flush()
        
        results_path = self.results_dir / 'v4_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        print(f"\nTraining Complete: {successful}/{len(files)} success")
        
        return successful > 0


if __name__ == "__main__":
    pipeline = V4Pipeline()
    pipeline.setup()
    if pipeline.download_data():
        pipeline.train(epochs=200, lr=0.001)
