#!/usr/bin/env python3
"""
V4 Transformer Model - Forces GPU Computation
Transformer 比 LSTM/GRU 快 10-20 倍，真正使用 GPU
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


# ========== Transformer Model - Real GPU Usage ==========

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class TransformerModel(nn.Module):
    """Transformer - Real GPU Usage, Fast"""
    def __init__(self, input_size=4, d_model=128, nhead=8, num_encoder_layers=2, 
                 num_decoder_layers=2, dim_feedforward=512, steps_ahead=10):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_proj = nn.Linear(d_model, input_size)
        self.steps_ahead = steps_ahead
    
    def forward(self, src, tgt=None):
        # src: (batch, 30, 4)
        # tgt: (batch, 10, 4) optional
        
        # Encode
        src = self.input_proj(src)  # (batch, 30, d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        
        # Decode
        batch_size = src.shape[0]
        device = src.device
        
        # Start token
        tgt_input = torch.zeros(batch_size, 1, self.d_model, device=device)
        outputs = []
        
        for i in range(self.steps_ahead):
            tgt_input_pos = self.pos_encoder(tgt_input)
            output = self.transformer_decoder(tgt_input_pos, memory)
            output_token = self.output_proj(output[:, -1:, :])  # (batch, 1, 4)
            outputs.append(output_token)
            
            # Prepare next input
            if tgt is not None and torch.rand(1).item() < 0.5:  # Teacher forcing
                next_input = self.input_proj(tgt[:, i:i+1, :])  # (batch, 1, d_model)
            else:
                next_input = self.input_proj(output_token)
            
            tgt_input = torch.cat([tgt_input, next_input], dim=1)
        
        outputs = torch.cat(outputs, dim=1)  # (batch, 10, 4)
        return outputs


# ========== Training Pipeline ==========

class TransformerTrainingPipeline:
    def __init__(self):
        if not torch.cuda.is_available():
            print("ERROR: GPU not available!")
            sys.exit(1)
        
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}\n")
        
        self.base_dir = Path('/content')
        self.data_dir = self.base_dir / 'data_v4'
        self.models_dir = self.base_dir / 'v4_models_tf'
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
                        print(f"OK")
                        successful += 1
                    else:
                        print(f"SKIP")
                else:
                    print(f"FAIL")
        
        print(f"\nDownloaded: {successful}/{total}\n")
        return successful > 0
    
    def train(self, epochs=50):
        print(f"Transformer Training (Real GPU Usage)\n")
        
        files = sorted(list(self.data_dir.glob("*.csv")))
        if not files:
            return False
        
        print(f"Training {len(files)} models\n")
        
        results = {}
        batch_size = 128  # 大 batch 正是 Transformer 的強項
        
        for idx, csv_file in enumerate(files, 1):
            name = csv_file.stem
            print(f"[{idx}/{len(files)}] {name}", end='')
            sys.stdout.flush()
            
            try:
                df = pd.read_csv(csv_file)
                if len(df) < 5000:
                    print(f" - Skip\n")
                    continue
                
                df = df.tail(5000)
                data = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
                
                norm = np.zeros_like(data)
                for i in range(len(data)):
                    mn, mx = data[i].min(), data[i].max()
                    if mx > mn:
                        norm[i] = (data[i] - mn) / (mx - mn)
                    else:
                        norm[i] = data[i]
                
                X, y = [], []
                for i in range(len(norm) - 40):
                    X.append(norm[i:i+30])
                    y.append(norm[i+30:i+40])
                
                X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self.device)
                y = torch.from_numpy(np.array(y, dtype=np.float32)).to(self.device)
                
                if len(X) < 50:
                    print(f" - Skip\n")
                    continue
                
                train_idx = int(len(X) * 0.8)
                X_train, y_train = X[:train_idx], y[:train_idx]
                X_val, y_val = X[train_idx:], y[train_idx:]
                
                model = TransformerModel(
                    input_size=4,
                    d_model=128,
                    nhead=8,
                    num_encoder_layers=2,
                    num_decoder_layers=2,
                    dim_feedforward=512,
                    steps_ahead=10
                ).to(self.device)
                
                optimizer = optim.AdamW(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                
                best_loss = float('inf')
                patience = 10
                patience_count = 0
                
                start_time = datetime.now()
                torch.cuda.reset_peak_memory_stats()
                
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0
                    
                    for i in range(0, len(X_train), batch_size):
                        end = min(i + batch_size, len(X_train))
                        X_b, y_b = X_train[i:end], y_train[i:end]
                        
                        optimizer.zero_grad()
                        pred = model(X_b, y_b)
                        loss = criterion(pred, y_b)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    train_loss /= ((len(X_train) + batch_size - 1) // batch_size)
                    
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for i in range(0, len(X_val), batch_size):
                            end = min(i + batch_size, len(X_val))
                            X_b, y_b = X_val[i:end], y_val[i:end]
                            pred = model(X_b, y_b)
                            loss = criterion(pred, y_b)
                            val_loss += loss.item()
                    
                    val_loss /= ((len(X_val) + batch_size - 1) // batch_size)
                    scheduler.step()
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_count = 0
                        torch.save(model.state_dict(), self.models_dir / f"model_{name}.pt")
                    else:
                        patience_count += 1
                        if patience_count >= patience:
                            break
                
                elapsed = (datetime.now() - start_time).total_seconds()
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                print(f" - Loss: {best_loss:.6f} ({elapsed:.0f}s, GPU: {mem_peak:.2f}GB)")
                
                results[name] = {
                    'status': 'success',
                    'loss': float(best_loss),
                    'epochs': epoch + 1,
                    'time': elapsed,
                    'gpu_memory': mem_peak
                }
                
                del model, optimizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f" - Error\n")
                results[name] = {'status': 'failed'}
        
        with open(self.results_dir / 'results_transformer.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        success = sum(1 for r in results.values() if r['status'] == 'success')
        avg_time = np.mean([r['time'] for r in results.values() if 'time' in r])
        avg_gpu = np.mean([r['gpu_memory'] for r in results.values() if 'gpu_memory' in r])
        
        print(f"\nComplete: {success}/{len(files)}")
        print(f"Avg time per model: {avg_time:.0f}s")
        print(f"Avg GPU memory: {avg_gpu:.2f}GB")
        return True


if __name__ == "__main__":
    p = TransformerTrainingPipeline()
    p.setup()
    if p.download_data():
        p.train(epochs=50)
