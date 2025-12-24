#!/usr/bin/env python3
"""
V4 Training - Complete Fixed Version
完整的訓練流程合了所有GPU修正
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

print("="*70)
print("V4 TRAINING - COMPLETE FIXED VERSION")
print("="*70)
sys.stdout.flush()

# 棘清關鍵 LSTM 模型
class SimpleSeq2SeqLSTM(nn.Module):
    """簡單湅Seq2Seq LSTM - 全GPU執行"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 steps_ahead, output_size, device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.steps_ahead = steps_ahead
        self.output_size = output_size
        
        # 編碼器 LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            device=device  # 直接在GPU上
        )
        
        # 解碼器 LSTM
        self.decoder = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            device=device
        )
        
        # 輸出層
        self.fc = nn.Linear(hidden_size, output_size, device=device)
        
        # 注冊設備缸
        self.register_buffer('scale', torch.tensor(1.0, device=device))
    
    def forward(self, encoder_input, decoder_input, teacher_forcing_ratio=0.5):
        batch_size = encoder_input.size(0)
        
        # 編碼
        encoder_output, (h_t, c_t) = self.encoder(encoder_input)
        
        # 預先創建輸出張量在GPU上
        decoder_outputs = torch.zeros(
            batch_size, self.steps_ahead, self.output_size,
            device=self.device, dtype=encoder_input.dtype
        )
        
        # 初始解碼輸入
        decoder_input_t = encoder_output[:, -1:, :self.output_size]
        
        # 解碼迴圈
        for t in range(self.steps_ahead):
            decoder_output, (h_t, c_t) = self.decoder(decoder_input_t, (h_t, c_t))
            
            # 輸出層
            pred = self.fc(decoder_output)
            decoder_outputs[:, t:t+1, :] = pred
            
            # Teacher forcing - GPU張量操作
            if torch.rand(1, device=self.device).item() < teacher_forcing_ratio and t < decoder_input.size(1):
                decoder_input_t = decoder_input[:, t:t+1, :]
            else:
                decoder_input_t = pred
        
        return decoder_outputs


class TrainingManager:
    """訓練管理器"""
    
    def __init__(self):
        print("\n[INIT] Setting up device...")
        sys.stdout.flush()
        
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available!")
            sys.exit(1)
        
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        
        # GPU優化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('highest')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f}GB")
        print(f"CUDA: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
        sys.stdout.flush()
        
        self.base_dir = Path('/content')
        self.data_dir = self.base_dir / 'data_v4'
        self.models_dir = self.base_dir / 'v4_models'
        self.results_dir = self.base_dir / 'v4_results'
    
    def setup_dirs(self):
        print("\n[SETUP] Creating directories...")
        sys.stdout.flush()
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Data dir: {self.data_dir}")
        print(f"Models dir: {self.models_dir}")
        sys.stdout.flush()
    
    def download_data(self, num_coins=5):
        print(f"\n[DATA] Downloading {num_coins} coins...")
        sys.stdout.flush()
        
        COINS = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
            'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
        ][:num_coins]
        
        base_url = "https://data.binance.vision/data/spot/monthly/klines"
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
                print(f"  {coin} {tf}...", end=' ', flush=True)
                sys.stdout.flush()
                
                all_data = []
                for year, month in months:
                    try:
                        url = f"{base_url}/{coin}/{tf}/{coin}-{tf}-{year}-{month:02d}.zip"
                        r = requests.get(url, timeout=10)
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
                        print(f"OK ({len(df)} bars)")
                        successful += 1
                    else:
                        print(f"SKIP (only {len(df)} bars)")
                else:
                    print(f"FAIL")
                
                sys.stdout.flush()
        
        print(f"\nDownloaded: {successful}/{len(COINS)*2}")
        sys.stdout.flush()
        return successful > 0
    
    def train(self, epochs=50, batch_size=32):
        print(f"\n[TRAIN] Starting training...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        sys.stdout.flush()
        
        files = sorted(list(self.data_dir.glob("*.csv")))
        if not files:
            print("ERROR: No data files found!")
            return False
        
        print(f"Found {len(files)} training files\n")
        sys.stdout.flush()
        
        results = {}
        
        for idx, csv_file in enumerate(files, 1):
            name = csv_file.stem
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(files)}] {name}")
            print(f"{'='*70}")
            sys.stdout.flush()
            
            try:
                # 載入
                print("Loading data...")
                df = pd.read_csv(csv_file)
                if len(df) < 7000:
                    print("Skipping: insufficient data")
                    continue
                
                df = df.tail(7000)
                data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
                
                # 正規化
                norm = np.zeros_like(data)
                for i in range(len(data)):
                    mn, mx = data[i].min(), data[i].max()
                    if mx > mn:
                        norm[i] = (data[i] - mn) / (mx - mn)
                    else:
                        norm[i] = data[i]
                
                # 建立序列
                X, y = [], []
                for i in range(len(norm) - 40):
                    X.append(norm[i:i+30])
                    y.append(norm[i+30:i+40])
                
                print(f"Data shape: {len(X)} sequences")
                
                # 關鍵：一次性轉移到GPU
                print("Transferring to GPU...")
                X_gpu = torch.from_numpy(np.array(X, dtype=np.float32)).to(
                    self.device, non_blocking=True
                ).contiguous()
                y_gpu = torch.from_numpy(np.array(y, dtype=np.float32)).to(
                    self.device, non_blocking=True
                ).contiguous()
                
                torch.cuda.synchronize()
                print(f"X: {X_gpu.shape} on {X_gpu.device}")
                print(f"y: {y_gpu.shape} on {y_gpu.device}")
                sys.stdout.flush()
                
                if len(X_gpu) < 100:
                    print("Skipping: insufficient sequences")
                    continue
                
                # 分割
                train_idx = int(len(X_gpu) * 0.7)
                val_idx = int(len(X_gpu) * 0.85)
                
                X_train = X_gpu[:train_idx]
                y_train = y_gpu[:train_idx]
                X_val = X_gpu[train_idx:val_idx]
                y_val = y_gpu[train_idx:val_idx]
                
                print(f"Train: {len(X_train)}, Val: {len(X_val)}")
                sys.stdout.flush()
                
                # 模型
                print("\nCreating model...")
                torch.cuda.reset_peak_memory_stats()
                
                model = SimpleSeq2SeqLSTM(
                    input_size=4, hidden_size=256, num_layers=2,
                    dropout=0.3, steps_ahead=10, output_size=4,
                    device=self.device
                )
                
                print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
                sys.stdout.flush()
                
                # 鄭理器
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
                
                best_loss = float('inf')
                patience = 10
                patience_count = 0
                
                print(f"\nStarting training...\n")
                sys.stdout.flush()
                
                # 訓練迴圈
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0
                    batch_count = 0
                    
                    for i in range(0, len(X_train), batch_size):
                        end = min(i + batch_size, len(X_train))
                        X_b = X_train[i:end].contiguous()
                        y_b = y_train[i:end].contiguous()
                        
                        optimizer.zero_grad()
                        pred = model(X_b, y_b, teacher_forcing_ratio=0.5)
                        
                        torch.cuda.synchronize()  # 關鍵！
                        
                        loss = criterion(pred, y_b)
                        loss.backward()
                        
                        torch.cuda.synchronize()  # 關鍵！
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                        batch_count += 1
                    
                    train_loss /= batch_count
                    
                    # 驗證
                    model.eval()
                    val_loss = 0
                    val_count = 0
                    with torch.no_grad():
                        for i in range(0, len(X_val), batch_size):
                            end = min(i + batch_size, len(X_val))
                            X_b = X_val[i:end].contiguous()
                            y_b = y_val[i:end].contiguous()
                            pred = model(X_b, y_b, teacher_forcing_ratio=0)
                            loss = criterion(pred, y_b)
                            val_loss += loss.item()
                            val_count += 1
                    
                    val_loss /= val_count
                    scheduler.step()
                    
                    # 報告
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        torch.cuda.synchronize()
                        alloc = torch.cuda.memory_allocated() / 1e9
                        peak = torch.cuda.max_memory_allocated() / 1e9
                        print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | GPU: {alloc:.2f}GB (peak: {peak:.2f}GB)")
                        sys.stdout.flush()
                    
                    # 早停
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
                final_alloc = torch.cuda.memory_allocated() / 1e9
                final_peak = torch.cuda.max_memory_allocated() / 1e9
                
                print(f"\nCompleted: {name}")
                print(f"Final loss: {best_loss:.6f} | Epochs: {epoch+1}")
                print(f"Peak GPU: {final_peak:.2f}GB")
                sys.stdout.flush()
                
                results[name] = {
                    'status': 'success',
                    'loss': float(best_loss),
                    'epochs': epoch + 1,
                    'peak_gpu_memory': float(final_peak)
                }
                
                del model, optimizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"ERROR: {str(e)[:200]}")
                import traceback
                traceback.print_exc()
                results[name] = {'status': 'failed', 'error': str(e)[:100]}
                sys.stdout.flush()
        
        # 保存結果
        with open(self.results_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        success_count = sum(1 for r in results.values() if r['status'] == 'success')
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE: {success_count}/{len(files)} successful")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        return success_count > 0


if __name__ == "__main__":
    print("\nStarting V4 Training Pipeline...\n")
    sys.stdout.flush()
    
    manager = TrainingManager()
    manager.setup_dirs()
    
    # 可選：第一次詳細下載，了解程式流程，之後冊下載更多
    if manager.download_data(num_coins=2):  # 先下轉⃓2个侶
        manager.train(epochs=20, batch_size=32)  # 少epochs渫試
    
    print("\nDone!")
    sys.stdout.flush()
