#!/usr/bin/env python3
"""
V4 Torch Compile - 最位參數運算強制 GPU
PyTorch JIT + torch.compile = 真正的 GPU 訓練
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
import torch.nn.functional as F
import torch.optim as optim
from io import BytesIO

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 預兑位元 GPU
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('highest')


# ========== GPU-Optimized Models ==========

class GRUModel(nn.Module):
    """简易高效 GRU 模律"""
    def __init__(self, input_size=4, hidden_size=256, num_layers=3, output_size=4):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # GRU layers - PyTorch 原生 GRU 是 CUDA kernel，高效
        self.gru = nn.GRU(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention (simplified for speed)
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size * 10)  # 10 steps
        
        self.output_size = output_size
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.shape[0]
        device = x.device
        
        # Project input to hidden_size
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # GRU 處理 - 全 GPU CUDA kernel
        gru_out, h_n = self.gru(x)  # gru_out: (batch, seq_len, hidden_size)
        
        # Attention
        attn_weights = F.softmax(self.attention(gru_out), dim=1)  # (batch, seq_len, 1)
        context = (gru_out * attn_weights).sum(dim=1)  # (batch, hidden_size)
        
        # FC layers
        out = F.relu(self.fc1(context))  # (batch, hidden_size)
        out = self.fc2(out)  # (batch, output_size * 10)
        out = out.view(batch_size, 10, self.output_size)  # (batch, 10, 4)
        
        return out


class LSTMModel(nn.Module):
    """深層 LSTM + Attention"""
    def __init__(self, input_size=4, hidden_size=256, num_layers=3, output_size=4):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # LSTM layers - 高效 CUDA kernel
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size * 10)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # LSTM - CUDA kernel
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)
        
        # Attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_size)
        
        # Combine final hidden state and context
        combined = torch.cat([h_n[-1], context], dim=1)  # (batch, hidden_size * 2)
        
        # FC layers
        out = F.relu(self.fc1(combined))  # (batch, hidden_size)
        out = self.fc2(out)  # (batch, output_size * 10)
        out = out.view(batch_size, 10, self.output_size)  # (batch, 10, 4)
        
        return out


class TorchTrainingPipeline:
    def __init__(self, model_type='gru'):
        # 棂檢 GPU
        if not torch.cuda.is_available():
            print("ERROR: GPU not available!")
            sys.exit(1)
        
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        self.model_type = model_type
        
        # GPU 信息
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Model Type: {model_type.upper()}\n")
        
        # 目錄設置
        self.base_dir = Path('/content')
        self.data_dir = self.base_dir / 'data_v4'
        self.models_dir = self.base_dir / f'v4_models_{model_type}'
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
                    
                    if len(df) >= 5000:
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
    
    def train(self, epochs=100, batch_size=128):
        print(f"Training with PyTorch Compiled Models\n")
        
        files = sorted(list(self.data_dir.glob("*.csv")))
        if not files:
            print("No data files found!")
            return False
        
        print(f"Training {len(files)} models (batch_size={batch_size})\n")
        
        results = {}
        
        for idx, csv_file in enumerate(files, 1):
            name = csv_file.stem
            print(f"[{idx}/{len(files)}] {name}", end=' | ', flush=True)
            
            try:
                # 讀取數據
                df = pd.read_csv(csv_file)
                if len(df) < 5000:
                    print("Skip\n")
                    continue
                
                df = df.tail(5000)
                data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
                
                # 歸一化
                data_norm = np.zeros_like(data)
                for i in range(len(data)):
                    mn, mx = data[i].min(), data[i].max()
                    if mx > mn:
                        data_norm[i] = (data[i] - mn) / (mx - mn)
                    else:
                        data_norm[i] = data[i]
                
                # 批次敲拓
                X, y = [], []
                for i in range(len(data_norm) - 40):
                    X.append(data_norm[i:i+30])
                    y.append(data_norm[i+30:i+40])
                
                # 轉移到 GPU
                X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self.device)
                y = torch.from_numpy(np.array(y, dtype=np.float32)).to(self.device)
                
                if len(X) < 50:
                    print("Skip\n")
                    continue
                
                # 訓練/驗證分割
                train_idx = int(len(X) * 0.8)
                X_train, y_train = X[:train_idx], y[:train_idx]
                X_val, y_val = X[train_idx:], y[train_idx:]
                
                # 模律選擇
                if self.model_type == 'gru':
                    model = GRUModel(input_size=4, hidden_size=256, num_layers=3)
                else:
                    model = LSTMModel(input_size=4, hidden_size=256, num_layers=3)
                
                model = model.to(self.device)
                
                # 編譯 模律 - 預兑位 GPU
                try:
                    # PyTorch 2.0+ torch.compile - 会 自動优化为 CUDA kernels
                    model = torch.compile(model, mode='reduce-overhead')
                except:
                    pass  # fallback
                
                # 优化器 + 搋多討
                optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                scaler = torch.cuda.amp.GradScaler()  # 混合精度訓練
                
                best_loss = float('inf')
                patience = 15
                patience_count = 0
                
                start_time = datetime.now()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                
                # 訓練 loop
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0
                    
                    # 批次訓練
                    for i in range(0, len(X_train), batch_size):
                        end = min(i + batch_size, len(X_train))
                        X_b, y_b = X_train[i:end], y_train[i:end]
                        
                        optimizer.zero_grad()
                        
                        # 混合精度
                        with torch.cuda.amp.autocast():
                            pred = model(X_b)
                            loss = criterion(pred, y_b)
                        
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        
                        train_loss += loss.item()
                    
                    train_loss /= ((len(X_train) + batch_size - 1) // batch_size)
                    
                    # 驗證
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for i in range(0, len(X_val), batch_size):
                                end = min(i + batch_size, len(X_val))
                                X_b, y_b = X_val[i:end], y_val[i:end]
                                pred = model(X_b)
                                loss = criterion(pred, y_b)
                                val_loss += loss.item()
                    
                    val_loss /= ((len(X_val) + batch_size - 1) // batch_size)
                    scheduler.step()
                    
                    # 早停
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_count = 0
                        # 保存模律
                        save_path = self.models_dir / f"model_{name}.pt"
                        if hasattr(model, '_orig_mod'):
                            torch.save(model._orig_mod.state_dict(), save_path)
                        else:
                            torch.save(model.state_dict(), save_path)
                    else:
                        patience_count += 1
                        if patience_count >= patience:
                            break
                
                # 統計
                torch.cuda.synchronize()
                elapsed = (datetime.now() - start_time).total_seconds()
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                
                print(f"Loss: {best_loss:.6f} | Time: {elapsed:.0f}s | GPU: {mem_peak:.2f}GB")
                
                results[name] = {
                    'status': 'success',
                    'loss': float(best_loss),
                    'epochs': epoch + 1,
                    'time': elapsed,
                    'gpu_memory': mem_peak
                }
                
                # 清理
                del model, optimizer, scaler
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            except Exception as e:
                print(f"Error: {str(e)[:30]}")
                results[name] = {'status': 'failed'}
        
        # 保存結果
        with open(self.results_dir / f'results_{self.model_type}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # 統計
        success = sum(1 for r in results.values() if r['status'] == 'success')
        if success > 0:
            avg_time = np.mean([r['time'] for r in results.values() if 'time' in r])
            avg_gpu = np.mean([r['gpu_memory'] for r in results.values() if 'gpu_memory' in r])
            print(f"\nComplete: {success}/{len(files)}")
            print(f"Avg time: {avg_time:.0f}s per model")
            print(f"Avg GPU: {avg_gpu:.2f}GB")
        
        return True


if __name__ == "__main__":
    # 選擇模律類型: 'gru' 或 'lstm'
    model_type = 'gru'  # 或 'lstm'
    
    p = TorchTrainingPipeline(model_type=model_type)
    p.setup()
    
    if p.download_data():
        # batch_size 越大 GPU 使用量越高
        p.train(epochs=100, batch_size=128)
