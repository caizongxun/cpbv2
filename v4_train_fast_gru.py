#!/usr/bin/env python3
"""
V4 Fast Training - GRU Model (30-40% faster than LSTM)
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


# ========== Fast Model with GRU ==========

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, encoder_output):
        # encoder_output: (batch, seq_len, hidden_size)
        scores = self.fc(encoder_output)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = (encoder_output * weights).sum(dim=1)  # (batch, hidden_size)
        return context


class FastSeq2Seq(nn.Module):
    """Fast Seq2Seq with GRU - simpler and faster"""
    def __init__(self, input_size=4, hidden_size=128, output_size=4, steps_ahead=10):
        super(FastSeq2Seq, self).__init__()
        
        # Encoder
        self.encoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Attention
        self.attention = SimpleAttention(hidden_size)
        
        # Decoder
        self.decoder_gru = nn.GRU(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Output
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        self.steps_ahead = steps_ahead
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        # x: (batch, 30, 4)
        batch_size = x.shape[0]
        device = x.device
        
        # Encoder
        enc_out, h_n = self.encoder_gru(x)  # enc_out: (batch, 30, hidden)
        context = self.attention(enc_out)  # (batch, hidden)
        
        # Decoder
        predictions = []
        dec_input = torch.zeros(batch_size, 1, 4, device=device, dtype=x.dtype)
        h = h_n
        
        use_teacher_forcing = target is not None and torch.rand(1, device=device).item() < teacher_forcing_ratio
        
        for t in range(self.steps_ahead):
            dec_out, h = self.decoder_gru(dec_input, h)
            # dec_out: (batch, 1, hidden)
            combined = torch.cat([dec_out, context.unsqueeze(1)], dim=-1)
            output = self.fc_out(combined)  # (batch, 1, 4)
            predictions.append(output)
            
            if use_teacher_forcing and target is not None:
                dec_input = target[:, t:t+1, :]
            else:
                dec_input = output
        
        predictions = torch.cat(predictions, dim=1)  # (batch, 10, 4)
        return predictions


# ========== Training Pipeline ==========

class FastTrainingPipeline:
    def __init__(self):
        if not torch.cuda.is_available():
            print("ERROR: GPU not available!")
            sys.exit(1)
        
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')  # 最快速
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}\n")
        
        self.base_dir = Path('/content')
        self.data_dir = self.base_dir / 'data_v4'
        self.models_dir = self.base_dir / 'v4_models_fast'
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
                        print(f"OK")
                        successful += 1
                    else:
                        print(f"SKIP")
                else:
                    print(f"FAIL")
        
        print(f"\nDownloaded: {successful}/{total}\n")
        return successful > 0
    
    def train(self, epochs=100):
        print(f"Fast Training (GRU - 30-40% faster)\n")
        
        files = sorted(list(self.data_dir.glob("*.csv")))
        if not files:
            return False
        
        print(f"Training {len(files)} models\n")
        
        results = {}
        batch_size = 64  # 更大的 batch
        
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
                
                model = FastSeq2Seq(input_size=4, hidden_size=128, output_size=4, steps_ahead=10)
                model = model.to(self.device)
                
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                best_loss = float('inf')
                patience = 15
                patience_count = 0
                
                start_time = datetime.now()
                
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
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_count = 0
                        torch.save(model.state_dict(), self.models_dir / f"model_{name}.pt")
                    else:
                        patience_count += 1
                        if patience_count >= patience:
                            break
                
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f" - Loss: {best_loss:.6f} ({elapsed:.1f}s)")
                
                results[name] = {'status': 'success', 'loss': float(best_loss), 'epochs': epoch + 1, 'time': elapsed}
                
                del model, optimizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f" - Error\n")
                results[name] = {'status': 'failed'}
        
        with open(self.results_dir / 'results_fast.json', 'w') as f:
            json.dump(results, f)
        
        success = sum(1 for r in results.values() if r['status'] == 'success')
        avg_time = np.mean([r['time'] for r in results.values() if 'time' in r])
        print(f"\nComplete: {success}/{len(files)} | Avg time per model: {avg_time:.1f}s")
        return True


if __name__ == "__main__":
    p = FastTrainingPipeline()
    p.setup()
    if p.download_data():
        p.train(epochs=100)
