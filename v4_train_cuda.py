#!/usr/bin/env python3
"""
V4 Training with Forced CUDA Model
"""

import os
import sys
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
from io import BytesIO

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from v4_model_cuda_forced import Seq2SeqLSTMGPU


class TrainingPipeline:
    def __init__(self):
        if not torch.cuda.is_available():
            print("ERROR: GPU not available!")
            sys.exit(1)
        
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.set_float32_matmul_precision('medium')
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print()
        
        self.base_dir = Path('/content')
        self.data_dir = self.base_dir / 'data_v4'
        self.models_dir = self.base_dir / 'v4_models'
        self.results_dir = self.base_dir / 'v4_results'
    
    def setup(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def download_data(self):
        print("Downloading data...\n")
        
        COINS = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
            'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
            'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'DYDXUSDT', 'ARBUSDT',
            'OPUSDT', 'PEPEUSDT', 'INJUSDT', 'SHIBUSDT', 'LUNAUSDT'
        ]
        
        base_url = "https://data.binance.vision/data/spot/monthly/klines"
        total = len(COINS) * 2
        completed = 0
        successful = 0
        
        now = datetime.now()
        months = []
        for i in range(12):
            m = now.month - i
            y = now.year
            if m <= 0:
                m += 12
                y -= 1
            months.append((y, m))
        
        for coin in COINS:
            for tf in ['15m', '1h']:
                completed += 1
                print(f"[{completed}/{total}] {coin} {tf}...", end=' ')
                sys.stdout.flush()
                
                all_data = []
                for year, month in months:
                    try:
                        url = f"{base_url}/{coin}/{tf}/{coin}-{tf}-{year}-{month:02d}.zip"
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
                        print(f"OK {len(df)}")
                        successful += 1
                    else:
                        print(f"SKIP")
                else:
                    print(f"FAIL")
        
        print(f"\nDownloaded: {successful}/{total}\n")
        return successful > 0
    
    def train(self, epochs=200):
        print("Starting GPU Training\n")
        
        files = sorted(list(self.data_dir.glob("*.csv")))
        if not files:
            return False
        
        print(f"Training {len(files)} models\n")
        
        results = {}
        batch_size = 32  # 大 batch size 以充分利用 GPU
        
        for idx, csv_file in enumerate(files, 1):
            name = csv_file.stem
            print(f"[{idx}/{len(files)}] {name}")
            
            try:
                df = pd.read_csv(csv_file)
                if len(df) < 7000:
                    print(f"  Skip")
                    continue
                
                df = df.tail(7000)
                data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
                
                # 標準化
                norm = np.zeros_like(data)
                for i in range(len(data)):
                    mn, mx = data[i].min(), data[i].max()
                    if mx > mn:
                        norm[i] = (data[i] - mn) / (mx - mn)
                    else:
                        norm[i] = data[i]
                
                # 序列
                X, y = [], []
                for i in range(len(norm) - 40):
                    X.append(norm[i:i+30])
                    y.append(norm[i+30:i+40])
                
                X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self.device)
                y = torch.from_numpy(np.array(y, dtype=np.float32)).to(self.device)
                
                if len(X) < 100:
                    print(f"  Skip")
                    continue
                
                # 分割
                train_idx = int(len(X) * 0.7)
                val_idx = int(len(X) * 0.85)
                
                X_train, y_train = X[:train_idx], y[:train_idx]
                X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
                
                # 创建模型
                model = Seq2SeqLSTMGPU(
                    input_size=4, hidden_size=256, num_layers=2,
                    dropout=0.3, steps_ahead=10, output_size=4
                ).to(self.device)
                
                # GPU 設定
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                mem_start = torch.cuda.memory_allocated() / 1e9
                print(f"  GPU Start: {mem_start:.2f}GB")
                
                # 優化器
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
                
                best_loss = float('inf')
                patience = 20
                patience_count = 0
                
                # 訓練
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0
                    
                    for i in range(0, len(X_train), batch_size):
                        end = min(i + batch_size, len(X_train))
                        X_b, y_b = X_train[i:end], y_train[i:end]
                        
                        optimizer.zero_grad()
                        pred = model(X_b, y_b, teacher_forcing_ratio=0.5)
                        loss = criterion(pred, y_b)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    train_loss /= ((len(X_train) + batch_size - 1) // batch_size)
                    
                    # 驗證
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for i in range(0, len(X_val), batch_size):
                            end = min(i + batch_size, len(X_val))
                            X_b, y_b = X_val[i:end], y_val[i:end]
                            pred = model(X_b, y_b, teacher_forcing_ratio=0)
                            loss = criterion(pred, y_b)
                            val_loss += loss.item()
                    
                    val_loss /= ((len(X_val) + batch_size - 1) // batch_size)
                    scheduler.step()
                    
                    # 進度
                    if (epoch + 1) % 20 == 0:
                        torch.cuda.synchronize()
                        mem = torch.cuda.memory_allocated() / 1e9
                        mem_peak = torch.cuda.max_memory_allocated() / 1e9
                        print(f"    Epoch {epoch+1}: Loss={train_loss:.6f}, Val={val_loss:.6f}")
                        print(f"    GPU: {mem:.2f}GB (Peak: {mem_peak:.2f}GB)")
                        sys.stdout.flush()
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_count = 0
                        torch.save(model.state_dict(), self.models_dir / f"model_{name}.pt")
                    else:
                        patience_count += 1
                        if patience_count >= patience:
                            break
                
                torch.cuda.synchronize()
                mem_final = torch.cuda.memory_allocated() / 1e9
                print(f"  Done - Loss: {best_loss:.6f}, Final GPU: {mem_final:.2f}GB\n")
                
                results[name] = {'status': 'success', 'loss': float(best_loss), 'epochs': epoch + 1}
                
                del model, optimizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error: {str(e)[:50]}\n")
                results[name] = {'status': 'failed'}
        
        with open(self.results_dir / 'results.json', 'w') as f:
            json.dump(results, f)
        
        success = sum(1 for r in results.values() if r['status'] == 'success')
        print(f"Complete: {success}/{len(files)}")
        return True


if __name__ == "__main__":
    p = TrainingPipeline()
    p.setup()
    if p.download_data():
        p.train(epochs=200)
