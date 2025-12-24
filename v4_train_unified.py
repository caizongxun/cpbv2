#!/usr/bin/env python3
"""
V4 Training - Self-contained with embedded model
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

# ========== Model Architecture ==========

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        output = self.fc_out(context)
        
        return output, attention_weights


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        return lstm_out, h_n, c_n


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_size, num_heads=4)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, encoder_outputs, encoder_hidden, encoder_cell, target_input=None, steps_ahead=10):
        batch_size = encoder_outputs.shape[0]
        device = encoder_outputs.device
        
        h = encoder_hidden
        c = encoder_cell
        predictions = []
        
        if target_input is None:
            current_input = torch.zeros(batch_size, 1, self.output_size, device=device, dtype=encoder_outputs.dtype)
        else:
            current_input = target_input
        
        for step in range(steps_ahead):
            lstm_out, (h, c) = self.lstm(current_input, (h, c))
            attention_out, _ = self.attention(lstm_out, encoder_outputs, encoder_outputs)
            
            combined = torch.cat([lstm_out, attention_out], dim=-1)
            combined = self.dropout(combined)
            combined = F.relu(self.fc(combined))
            output = self.output_layer(combined)
            
            predictions.append(output)
            current_input = output
        
        predictions = torch.cat(predictions, dim=1)
        return predictions


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3, 
                 steps_ahead=10, output_size=4):
        super(Seq2SeqLSTM, self).__init__()
        
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(output_size, hidden_size, num_layers, dropout)
        self.steps_ahead = steps_ahead
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(x)
        
        if target is not None and torch.rand(1, device=x.device).item() < teacher_forcing_ratio:
            target_input = target[:, :1, :]
            predictions = self.decoder(
                encoder_outputs, encoder_hidden, encoder_cell,
                target_input=target_input,
                steps_ahead=self.steps_ahead
            )
        else:
            predictions = self.decoder(
                encoder_outputs, encoder_hidden, encoder_cell,
                target_input=None,
                steps_ahead=self.steps_ahead
            )
        
        return predictions


# ========== Training Pipeline ==========

class TrainingPipeline:
    def __init__(self):
        if not torch.cuda.is_available():
            print("ERROR: GPU not available!")
            sys.exit(1)
        
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('medium')
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}\n")
        
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
                
                if len(X) < 100:
                    print(f"  Skip")
                    continue
                
                train_idx = int(len(X) * 0.7)
                val_idx = int(len(X) * 0.85)
                
                X_train, y_train = X[:train_idx], y[:train_idx]
                X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
                
                model = Seq2SeqLSTM(
                    input_size=4, hidden_size=256, num_layers=2,
                    dropout=0.3, steps_ahead=10, output_size=4
                ).to(self.device)
                
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                mem_start = torch.cuda.memory_allocated() / 1e9
                print(f"  GPU Start: {mem_start:.2f}GB")
                
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
                
                best_loss = float('inf')
                patience = 20
                patience_count = 0
                
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
                    scheduler.step()
                    
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
