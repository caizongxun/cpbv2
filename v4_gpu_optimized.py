#!/usr/bin/env python3
"""
V4 GPU Optimized Training - All Operations on GPU
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
from torch.utils.data import DataLoader, TensorDataset
from io import BytesIO

warnings.filterwarnings('ignore')

from v4_model_architecture import Seq2SeqLSTM


class GPUOptimizedPipeline:
    def __init__(self):
        # 設置 GPU
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # GPU 同步執行
        
        if not torch.cuda.is_available():
            print("ERROR: GPU not available!")
            sys.exit(1)
        
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        
        # GPU 優化
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat32
        torch.set_float32_matmul_precision('medium')   # 加速 float32 計算
        
        print("="*80)
        print("GPU Configuration")
        print("="*80)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
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
        months = [(now.year, now.month - i if now.month - i > 0 else 12 + now.month - i, 
                  now.year if now.month - i > 0 else now.year - 1) 
                 for i in range(12)]
        months = [(y, m) for y, m in [(m[2], m[1]) for m in months]]
        
        for coin in COINS:
            for tf in ['15m', '1h']:
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
                        print(f"OK {len(df)} bars")
                        successful += 1
                    else:
                        print(f"SKIP {len(df)}")
                else:
                    print(f"FAIL")
        
        print(f"\nDownloaded: {successful}/{total}")
        return successful > 0
    
    def train(self, epochs=200):
        print("\n" + "="*80)
        print("GPU Training Start")
        print("="*80 + "\n")
        
        files = sorted(list(self.data_dir.glob("*.csv")))
        if not files:
            print("No files!")
            return False
        
        print(f"Training {len(files)} models\n")
        
        results = {}
        batch_size = 16  # 增大 batch size 以充分利用 GPU
        
        for idx, csv_file in enumerate(files, 1):
            name = csv_file.stem
            print(f"[{idx}/{len(files)}] {name}")
            sys.stdout.flush()
            
            try:
                # 1. 讀取數據
                df = pd.read_csv(csv_file)
                if len(df) < 7000:
                    print(f"  Skip: {len(df)}")
                    continue
                
                df = df.tail(7000)
                data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
                
                # 2. 標準化（CPU 上進行，只做一次）
                normalized = np.zeros_like(data)
                for i in range(len(data)):
                    min_v = data[i].min()
                    max_v = data[i].max()
                    if max_v > min_v:
                        normalized[i] = (data[i] - min_v) / (max_v - min_v)
                    else:
                        normalized[i] = data[i]
                
                # 3. 創建序列
                X, y = [], []
                for i in range(len(normalized) - 40):
                    X.append(normalized[i:i+30])
                    y.append(normalized[i+30:i+40])
                
                X = torch.from_numpy(np.array(X, dtype=np.float32))
                y = torch.from_numpy(np.array(y, dtype=np.float32))
                
                if len(X) < 100:
                    print(f"  Skip: {len(X)} sequences")
                    continue
                
                # 4. 分割數據
                train_idx = int(len(X) * 0.7)
                val_idx = int(len(X) * 0.85)
                
                X_train, y_train = X[:train_idx], y[:train_idx]
                X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
                
                # 5. 立即轉移到 GPU（不用 DataLoader）
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                
                print(f"  Data on GPU: {X_train.device}")
                
                # 6. 創建模型（GPU 上）
                model = Seq2SeqLSTM(
                    input_size=4, hidden_size=128, num_layers=2,
                    dropout=0.3, steps_ahead=10, output_size=4
                ).to(self.device)
                
                # 驗證模型在 GPU
                model_device = next(model.parameters()).device
                print(f"  Model on GPU: {model_device}")
                
                # 初始 GPU 記憶體
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()  # 同步
                mem_before = torch.cuda.memory_allocated() / 1e9
                print(f"  GPU Memory: {mem_before:.2f}GB")
                
                # 7. 優化器
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
                
                best_loss = float('inf')
                patience = 20
                patience_count = 0
                
                # 8. 訓練循環
                for epoch in range(epochs):
                    # 訓練
                    model.train()
                    train_loss = 0
                    
                    # 批次訓練在 GPU 上
                    for i in range(0, len(X_train), batch_size):
                        batch_end = min(i + batch_size, len(X_train))
                        X_b = X_train[i:batch_end]
                        y_b = y_train[i:batch_end]
                        
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
                            batch_end = min(i + batch_size, len(X_val))
                            X_b = X_val[i:batch_end]
                            y_b = y_val[i:batch_end]
                            pred = model(X_b, y_b, teacher_forcing_ratio=0)
                            loss = criterion(pred, y_b)
                            val_loss += loss.item()
                    
                    val_loss /= ((len(X_val) + batch_size - 1) // batch_size)
                    scheduler.step()
                    
                    # 進度
                    if (epoch + 1) % 20 == 0:
                        torch.cuda.synchronize()  # 同步 GPU
                        mem_current = torch.cuda.memory_allocated() / 1e9
                        mem_peak = torch.cuda.max_memory_allocated() / 1e9
                        print(f"    Epoch {epoch+1} - Loss: {train_loss:.6f}, Val: {val_loss:.6f}")
                        print(f"    GPU Mem: {mem_current:.2f}GB (Peak: {mem_peak:.2f}GB)")
                        sys.stdout.flush()
                    
                    # Early stopping
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_count = 0
                        model_path = self.models_dir / f"model_{name}.pt"
                        torch.save(model.state_dict(), model_path)
                    else:
                        patience_count += 1
                        if patience_count >= patience:
                            break
                
                torch.cuda.synchronize()
                mem_final = torch.cuda.memory_allocated() / 1e9
                print(f"  Done - Best Loss: {best_loss:.6f}, GPU Final: {mem_final:.2f}GB\n")
                
                results[name] = {
                    'status': 'success',
                    'best_loss': float(best_loss),
                    'epochs': epoch + 1
                }
                
                # 清理
                del model, optimizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error: {str(e)[:60]}\n")
                results[name] = {'status': 'failed'}
        
        # 保存結果
        with open(self.results_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        success_count = sum(1 for r in results.values() if r['status'] == 'success')
        print(f"\nComplete: {success_count}/{len(files)} success")
        return True


if __name__ == "__main__":
    pipeline = GPUOptimizedPipeline()
    pipeline.setup()
    if pipeline.download_data():
        pipeline.train(epochs=200)
