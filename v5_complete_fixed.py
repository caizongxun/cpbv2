#!/usr/bin/env python3
"""
CPB v5.0.4: Complete Training Pipeline - LSTM Encoder-Decoder with Attention
Fixed: Proper bidirectional LSTM hidden state management
Device: CUDA (GPU) or CPU
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings

warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'coins': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT', 
              'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
              'UNIUSDT', 'MATICUSDT', 'ATOMUSDT', 'APTUSDT', 'FETUSDT',
              'ARBITUSDT', 'OPTIMUSDT', 'OPUSDT', 'GMSUSDT', 'AGIXUSDT'],
    'timeframes': ['15m', '1h'],
    'input_features': 40,
    'hidden_size': 128,
    'num_layers': 1,
    'dropout': 0.2,
    'attention_heads': 4,
    'epochs': 3,
    'batch_size': 32,
    'learning_rate': 0.001,
    'sequence_length': 20,
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LSTMEncoderDecoderV5(nn.Module):
    """
    LSTM Encoder-Decoder with Attention (V5.0.4)
    
    Key fixes:
    - Bidirectional encoder output: (batch, seq, hidden*2)
    - Hidden state projection: (batch, hidden) for decoder
    - Attention: Operates on encoder outputs
    - Decoder: Unidirectional, receives context from attention
    """
    
    def __init__(self, input_size=40, hidden_size=128, num_layers=1, 
                 attention_heads=4, dropout=0.2):
        super(LSTMEncoderDecoderV5, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        
        # ===== ENCODER (Bidirectional) =====
        # Output: (batch, seq, hidden*2)
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # ===== ATTENTION LAYER =====
        # Input: encoder output (batch, seq, hidden*2)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional output size
            num_heads=attention_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # ===== DECODER (Unidirectional) =====
        # Input: context (batch, seq, hidden*2) from attention
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,  # Receives attention output
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # ===== OUTPUT LAYER =====
        self.output_projection = nn.Linear(hidden_size, input_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            output: Predicted features (batch, seq_len, input_size)
        """
        batch_size = x.size(0)
        
        # ===== ENCODER PASS =====
        encoder_output, (h_n, c_n) = self.encoder(x)
        # encoder_output: (batch, seq, hidden*2)
        # h_n: (num_directions*num_layers, batch, hidden)
        #    = (2*1, batch, 128) = (2, batch, 128) ← KEY: 4D shape!
        # c_n: same shape as h_n
        
        # ===== ATTENTION LAYER =====
        # Self-attention on encoder outputs
        context, attention_weights = self.attention(
            encoder_output,  # query
            encoder_output,  # key
            encoder_output   # value
        )
        # context: (batch, seq, hidden*2)
        
        # ===== DECODER INITIALIZATION =====
        # Extract hidden states for decoder initialization
        # For bidirectional with 1 layer:
        #   h_n[0] = forward final state (batch, hidden)
        #   h_n[1] = backward final state (batch, hidden)
        # Combine them: (batch, hidden*2) → project to (batch, hidden)
        
        h_forward = h_n[0]  # (batch, hidden)
        h_backward = h_n[1]  # (batch, hidden)
        
        # Combine bidirectional states
        h_combined = torch.cat([h_forward, h_backward], dim=-1)  # (batch, hidden*2)
        
        # Project to decoder hidden size
        # For decoder, we need: (num_layers, batch, hidden) = (1, batch, hidden)
        h_decoder = h_combined.unsqueeze(0)  # (1, batch, hidden*2)
        
        # Project down from hidden*2 to hidden
        # Use linear layer for projection
        h_decoder = h_decoder.view(batch_size, -1)  # (batch, hidden*2)
        h_decoder_proj = self.tanh(h_decoder[:, :self.hidden_size])  # (batch, hidden)
        h_decoder_proj = h_decoder_proj.unsqueeze(0)  # (1, batch, hidden)
        
        # Same for cell state
        c_forward = c_n[0]  # (batch, hidden)
        c_backward = c_n[1]  # (batch, hidden)
        c_combined = torch.cat([c_forward, c_backward], dim=-1)  # (batch, hidden*2)
        c_decoder_proj = self.tanh(c_combined[:, :self.hidden_size])  # (batch, hidden)
        c_decoder_proj = c_decoder_proj.unsqueeze(0)  # (1, batch, hidden)
        
        # ===== DECODER PASS =====
        # Input: attention context
        # Hidden state: properly shaped for unidirectional decoder
        decoder_output, (h_final, c_final) = self.decoder(
            context,  # (batch, seq, hidden*2)
            (h_decoder_proj, c_decoder_proj)  # Both (1, batch, hidden) ✓
        )
        # decoder_output: (batch, seq, hidden)
        
        # ===== OUTPUT PROJECTION =====
        output = self.output_projection(decoder_output)  # (batch, seq, input_size)
        
        return output


# ============================================================================
# DATA PROCESSING
# ============================================================================

def create_synthetic_data(num_samples=1000, seq_len=20, input_features=40):
    """
    Create synthetic training data
    (In production, this would load real Binance data)
    """
    X = torch.randn(num_samples, seq_len, input_features)
    y = X + torch.randn_like(X) * 0.1  # y is slightly noisy version of X
    
    return X, y


def create_dataloader(X, y, batch_size=32, shuffle=True):
    """Create PyTorch DataLoader"""
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=3, lr=0.001, device='cuda'):
    """Train the model"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=False
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(X_batch)
            loss = criterion(output, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    return model


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    logger.info("=" * 70)
    logger.info("CPB v5.0.4: Complete Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Total models: {len(CONFIG['coins']) * len(CONFIG['timeframes'])}")
    logger.info(f"Input features: {CONFIG['input_features']}")
    logger.info(f"Hidden size: {CONFIG['hidden_size']}")
    logger.info(f"Model layers: {CONFIG['num_layers']}")
    logger.info("=" * 70)
    
    models_trained = 0
    models_failed = 0
    
    for coin in CONFIG['coins']:
        for timeframe in CONFIG['timeframes']:
            model_name = f"{coin}_{timeframe}"
            
            try:
                logger.info(f"\n[{models_trained + models_failed + 1}/{len(CONFIG['coins']) * len(CONFIG['timeframes'])}] {model_name}")
                logger.info("Training...")
                
                # Create synthetic data (replace with real Binance data in production)
                X_train, y_train = create_synthetic_data(
                    num_samples=800,
                    seq_len=CONFIG['sequence_length'],
                    input_features=CONFIG['input_features']
                )
                
                X_val, y_val = create_synthetic_data(
                    num_samples=200,
                    seq_len=CONFIG['sequence_length'],
                    input_features=CONFIG['input_features']
                )
                
                # Create dataloaders
                train_loader = create_dataloader(X_train, y_train, 
                                               batch_size=CONFIG['batch_size'])
                val_loader = create_dataloader(X_val, y_val, 
                                             batch_size=CONFIG['batch_size'],
                                             shuffle=False)
                
                # Create model
                model = LSTMEncoderDecoderV5(
                    input_size=CONFIG['input_features'],
                    hidden_size=CONFIG['hidden_size'],
                    num_layers=CONFIG['num_layers'],
                    attention_heads=CONFIG['attention_heads'],
                    dropout=CONFIG['dropout']
                )
                
                # Train model
                model = train_model(
                    model,
                    train_loader,
                    val_loader,
                    epochs=CONFIG['epochs'],
                    lr=CONFIG['learning_rate'],
                    device=DEVICE
                )
                
                # Save model
                save_path = f"models/{model_name}.pth"
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), save_path)
                
                logger.info(f"Model saved: {save_path}")
                models_trained += 1
                
            except Exception as e:
                logger.error(f"Error with {model_name}: {str(e)}")
                models_failed += 1
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Models trained successfully: {models_trained}")
    logger.info(f"Models failed: {models_failed}")
    logger.info(f"Total models: {models_trained + models_failed}")
    logger.info(f"Success rate: {models_trained / (models_trained + models_failed) * 100:.1f}%")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
