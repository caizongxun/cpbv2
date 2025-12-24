#!/usr/bin/env python3
"""
V4 Training with VERIFIED GPU Computation
Fixes: GPU memory not increasing, CPU fallback detection
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


class GPUMonitor:
    """Monitor GPU usage in real-time"""
    def __init__(self, device):
        self.device = device
    
    def check_tensor_device(self, tensor, name="tensor"):
        """Verify tensor is actually on GPU"""
        if tensor.device.type != 'cuda':
            raise RuntimeError(f"ERROR: {name} is on {tensor.device}, not GPU!")
    
    def log_memory(self, label=""):
        """Log current GPU memory usage"""
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[GPU Monitor {label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {peak:.2f}GB")
        return allocated, reserved, peak
    
    def verify_forward_pass(self, model, x, target):
        """Verify forward pass actually runs on GPU"""
        print("\n[GPU Verification] Starting forward pass verification...")
        
        # Check input tensors
        self.check_tensor_device(x, "input X")
        self.check_tensor_device(target, "target")
        
        # Check model parameters
        for name, param in model.named_parameters():
            if param.device.type != 'cuda':
                raise RuntimeError(f"ERROR: Model parameter {name} is on {param.device}")
        
        # Memory before
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1e9
        
        # Forward pass
        with torch.no_grad():
            output = model(x, target, teacher_forcing_ratio=0.5)
        
        # Check output
        self.check_tensor_device(output, "output")
        
        # Memory after
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1e9
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"[GPU Verification] Forward pass executed on GPU")
        print(f"[GPU Verification] Memory used: {(mem_after - mem_before):.3f}GB")
        print(f"[GPU Verification] Peak memory: {peak_mem:.3f}GB")
        print(f"[GPU Verification] Output shape: {output.shape}")
        print()


class TrainingPipeline:
    def __init__(self):
        if not torch.cuda.is_available():
            print("ERROR: GPU not available!")
            sys.exit(1)
        
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        self.gpu_monitor = GPUMonitor(self.device)
        
        # Enhanced CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.set_float32_matmul_precision('medium')
        torch.cuda.empty_cache()
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
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
        batch_size = 32
        
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
                
                # Normalization
                norm = np.zeros_like(data)
                for i in range(len(data)):
                    mn, mx = data[i].min(), data[i].max()
                    if mx > mn:
                        norm[i] = (data[i] - mn) / (mx - mn)
                    else:
                        norm[i] = data[i]
                
                # Sequences
                X, y = [], []
                for i in range(len(norm) - 40):
                    X.append(norm[i:i+30])
                    y.append(norm[i+30:i+40])
                
                # CRITICAL: Move to GPU immediately
                X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self.device, non_blocking=True)
                y = torch.from_numpy(np.array(y, dtype=np.float32)).to(self.device, non_blocking=True)
                
                # Verify tensors are on GPU
                self.gpu_monitor.check_tensor_device(X, f"{name} input")
                self.gpu_monitor.check_tensor_device(y, f"{name} target")
                
                if len(X) < 100:
                    print(f"  Skip")
                    continue
                
                # Split
                train_idx = int(len(X) * 0.7)
                val_idx = int(len(X) * 0.85)
                
                X_train, y_train = X[:train_idx], y[:train_idx]
                X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
                
                # Create model
                model = Seq2SeqLSTMGPU(
                    input_size=4, hidden_size=256, num_layers=2,
                    dropout=0.3, steps_ahead=10, output_size=4
                ).to(self.device)
                
                # VERIFY GPU before training
                print(f"  Verifying GPU execution...")
                self.gpu_monitor.verify_forward_pass(
                    model, 
                    X_train[:batch_size], 
                    y_train[:batch_size]
                )
                
                # GPU setup
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                mem_start = torch.cuda.memory_allocated() / 1e9
                print(f"  GPU Memory Start: {mem_start:.2f}GB")
                
                # Optimizer
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
                
                best_loss = float('inf')
                patience = 20
                patience_count = 0
                
                # Training loop with GPU verification
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0
                    
                    for i in range(0, len(X_train), batch_size):
                        end = min(i + batch_size, len(X_train))
                        X_b, y_b = X_train[i:end], y_train[i:end]
                        
                        # Verify batch is on GPU
                        self.gpu_monitor.check_tensor_device(X_b, f"batch at iter {i}")
                        
                        optimizer.zero_grad()
                        pred = model(X_b, y_b, teacher_forcing_ratio=0.5)
                        
                        # Verify prediction is on GPU
                        self.gpu_monitor.check_tensor_device(pred, f"prediction at iter {i}")
                        
                        loss = criterion(pred, y_b)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    train_loss /= ((len(X_train) + batch_size - 1) // batch_size)
                    
                    # Validation
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
                    
                    # Progress with GPU monitoring
                    if (epoch + 1) % 10 == 0:
                        torch.cuda.synchronize()
                        allocated, reserved, peak = self.gpu_monitor.log_memory(f"Epoch {epoch+1}")
                        print(f"    Epoch {epoch+1}: Loss={train_loss:.6f}, Val={val_loss:.6f}")
                        
                        # Alert if GPU memory is too low
                        if allocated < 0.1:
                            print("    WARNING: GPU memory is very low - possible CPU fallback!")
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
                print(f"  Error: {str(e)[:100]}\n")
                import traceback
                traceback.print_exc()
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
