#!/usr/bin/env python3
"""
V4 Training with Aggressive GPU Forcing (Version 2)
Uses manual LSTM to eliminate PyTorch cuDNN fallback
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

from v4_model_cuda_forced_v2 import Seq2SeqLSTMGPUv2


class GPUValidator:
    """Validate GPU execution and memory usage"""
    def __init__(self, device):
        self.device = device
    
    def validate_tensor_location(self, tensor, name):
        if tensor.device.type != 'cuda':
            raise RuntimeError(f"CRITICAL: {name} is on {tensor.device}, not GPU!")
        return True
    
    def print_gpu_status(self, label=""):
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        
        status = f"GPU Status [{label}] Allocated: {alloc:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {peak:.2f}GB"
        print(f"\n{status}")
        return alloc, reserved, peak


class TrainingPipeline:
    def __init__(self):
        if not torch.cuda.is_available():
            print("ERROR: GPU not available!")
            sys.exit(1)
        
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        self.validator = GPUValidator(self.device)
        
        # AGGRESSIVE GPU SETTINGS
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('highest')  # Maximize GPU usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        props = torch.cuda.get_device_properties(0)
        print(f"\n{'='*60}")
        print(f"GPU Configuration")
        print(f"{'='*60}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f}GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Max Threads Per Block: {props.max_threads_per_block}")
        print(f"{'='*60}\n")
        
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
                print(f"[{completed:2d}/{total}] {coin:12s} {tf}...", end=' ', flush=True)
                
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
        print("\n" + "="*60)
        print("Starting GPU Training with Forced LSTM")
        print("="*60 + "\n")
        
        files = sorted(list(self.data_dir.glob("*.csv")))
        if not files:
            return False
        
        print(f"Training {len(files)} models\n")
        
        results = {}
        batch_size = 32
        
        for idx, csv_file in enumerate(files, 1):
            name = csv_file.stem
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(files)}] Training {name}")
            print(f"{'='*60}")
            
            try:
                # Load data
                df = pd.read_csv(csv_file)
                if len(df) < 7000:
                    print(f"Skipping: insufficient data")
                    continue
                
                df = df.tail(7000)
                data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
                
                # Normalize
                norm = np.zeros_like(data)
                for i in range(len(data)):
                    mn, mx = data[i].min(), data[i].max()
                    if mx > mn:
                        norm[i] = (data[i] - mn) / (mx - mn)
                    else:
                        norm[i] = data[i]
                
                # Create sequences
                X, y = [], []
                for i in range(len(norm) - 40):
                    X.append(norm[i:i+30])
                    y.append(norm[i+30:i+40])
                
                # Move to GPU IMMEDIATELY with non_blocking
                X = torch.from_numpy(np.array(X, dtype=np.float32)).to(
                    self.device, non_blocking=True, memory_format=torch.contiguous_format
                )
                y = torch.from_numpy(np.array(y, dtype=np.float32)).to(
                    self.device, non_blocking=True, memory_format=torch.contiguous_format
                )
                
                # Validate location
                self.validator.validate_tensor_location(X, "Input X")
                self.validator.validate_tensor_location(y, "Target y")
                
                if len(X) < 100:
                    print(f"Skipping: insufficient sequences")
                    continue
                
                # Split data
                train_idx = int(len(X) * 0.7)
                val_idx = int(len(X) * 0.85)
                
                X_train, y_train = X[:train_idx], y[:train_idx]
                X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
                
                print(f"Data: {len(X)} sequences | Train: {len(X_train)} | Val: {len(X_val)}")
                
                # Create model with FORCED GPU
                torch.cuda.reset_peak_memory_stats()
                model = Seq2SeqLSTMGPUv2(
                    input_size=4, hidden_size=256, num_layers=2,
                    dropout=0.3, steps_ahead=10, output_size=4,
                    device=self.device
                )
                
                print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
                
                # Test forward pass
                print(f"\nTesting GPU execution...")
                with torch.no_grad():
                    test_output = model(
                        X_train[:batch_size],
                        y_train[:batch_size],
                        teacher_forcing_ratio=0.5
                    )
                    self.validator.validate_tensor_location(test_output, "Model output")
                
                alloc, reserved, peak = self.validator.print_gpu_status("After test forward")
                
                if alloc < 0.05:
                    print("\nWARNING: GPU memory usage is suspiciously low!")
                    print("Possible causes:")
                    print("  1. Model computation happening on CPU")
                    print("  2. Automatic mixed precision too aggressive")
                    print("  3. cuDNN has disabled GPU LSTM")
                
                # Optimizer and loss
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
                
                best_loss = float('inf')
                patience = 20
                patience_count = 0
                
                print(f"\nStarting training...\n")
                
                # Training loop
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0
                    batch_count = 0
                    
                    for i in range(0, len(X_train), batch_size):
                        end = min(i + batch_size, len(X_train))
                        X_b, y_b = X_train[i:end], y_train[i:end]
                        batch_count += 1
                        
                        optimizer.zero_grad()
                        pred = model(X_b, y_b, teacher_forcing_ratio=0.5)
                        
                        # Verify GPU
                        self.validator.validate_tensor_location(pred, f"Batch {i}")
                        
                        loss = criterion(pred, y_b)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    train_loss /= batch_count
                    
                    # Validation
                    model.eval()
                    val_loss = 0
                    val_count = 0
                    with torch.no_grad():
                        for i in range(0, len(X_val), batch_size):
                            end = min(i + batch_size, len(X_val))
                            X_b, y_b = X_val[i:end], y_val[i:end]
                            pred = model(X_b, y_b, teacher_forcing_ratio=0)
                            loss = criterion(pred, y_b)
                            val_loss += loss.item()
                            val_count += 1
                    
                    val_loss /= val_count
                    scheduler.step()
                    
                    # Print progress
                    if (epoch + 1) % 20 == 0 or epoch == 0:
                        alloc, reserved, peak = self.validator.print_gpu_status(
                            f"Epoch {epoch+1}"
                        )
                        print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
                    
                    # Early stopping
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_count = 0
                        torch.save(model.state_dict(), self.models_dir / f"model_{name}.pt")
                    else:
                        patience_count += 1
                        if patience_count >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
                
                torch.cuda.synchronize()
                final_alloc, final_reserved, final_peak = self.validator.print_gpu_status("Final")
                
                print(f"\nCompleted: {name}")
                print(f"Final Loss: {best_loss:.6f} | Epochs: {epoch+1}")
                print(f"Peak GPU Memory: {final_peak:.2f}GB")
                
                results[name] = {
                    'status': 'success',
                    'loss': float(best_loss),
                    'epochs': epoch + 1,
                    'peak_gpu_memory': float(final_peak)
                }
                
                del model, optimizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error: {str(e)[:200]}")
                import traceback
                traceback.print_exc()
                results[name] = {'status': 'failed', 'error': str(e)[:100]}
        
        # Save results
        with open(self.results_dir / 'results_v2.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        success_count = sum(1 for r in results.values() if r['status'] == 'success')
        print(f"\n{'='*60}")
        print(f"Training Complete: {success_count}/{len(files)} successful")
        print(f"{'='*60}")
        return True


if __name__ == "__main__":
    p = TrainingPipeline()
    p.setup()
    if p.download_data():
        p.train(epochs=200)
