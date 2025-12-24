#!/usr/bin/env python3
"""
V4 Model Architecture: Seq2Seq LSTM with Attention Mechanism
Goal: Predict next 10 OHLC candles based on previous 30 candles
Features:
  - Encoder-Decoder structure (Seq2Seq)
  - Attention mechanism for weighted focus
  - Multi-output: predicts Open, High, Low, Close for 10 future candles
  - Dropout regularization to prevent overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, hidden_size, num_heads=4):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.query(query)  # (batch_size, seq_len, hidden_size)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        
        output = self.fc_out(context)
        
        return output, attention_weights


class Encoder(nn.Module):
    """Encoder: process historical 30 candles"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_len=30, input_size=4)  [O, H, L, C]
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, 30, hidden_size)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        return lstm_out, h_n, c_n


class Decoder(nn.Module):
    """Decoder: generate next 10 candles with attention"""
    def __init__(self, output_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size  # 4 for OHLC
        
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_size, num_heads=4)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, encoder_outputs, encoder_hidden, encoder_cell, target_input=None, steps_ahead=10):
        # encoder_outputs: (batch_size, 30, hidden_size)
        # encoder_hidden: (num_layers, batch_size, hidden_size)
        # target_input: (batch_size, 1, output_size) - first target (last candle from input)
        
        batch_size = encoder_outputs.shape[0]
        device = encoder_outputs.device
        
        # Initialize decoder hidden state with encoder's final state
        h = encoder_hidden
        c = encoder_cell
        
        predictions = []
        
        # Use teacher forcing during training, autoregressive during inference
        if target_input is None:
            # Inference mode: use previous prediction as next input
            current_input = encoder_outputs[:, -1:, :output_size]  # Last candle from encoder
        else:
            current_input = target_input
        
        for step in range(steps_ahead):
            # LSTM step
            lstm_out, (h, c) = self.lstm(current_input, (h, c))
            # lstm_out: (batch_size, 1, hidden_size)
            
            # Apply attention over encoder outputs
            attention_out, _ = self.attention(
                query=lstm_out,
                key=encoder_outputs,
                value=encoder_outputs
            )
            # attention_out: (batch_size, 1, hidden_size)
            
            # Combine LSTM output with attention
            combined = torch.cat([lstm_out, attention_out], dim=-1)  # (batch_size, 1, hidden_size*2)
            combined = self.dropout(combined)
            
            # Project to hidden_size
            combined = torch.relu(self.fc(combined))
            
            # Generate output
            output = self.output_layer(combined)  # (batch_size, 1, 4)
            predictions.append(output)
            
            # Prepare next input
            current_input = output  # Use predicted OHLC as next input
        
        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=1)  # (batch_size, 10, 4)
        
        return predictions


class Seq2SeqLSTM(nn.Module):
    """Seq2Seq model: Encoder-Decoder with Attention"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3, 
                 steps_ahead=10, output_size=4):
        super(Seq2SeqLSTM, self).__init__()
        
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(output_size, hidden_size, num_layers, dropout)
        self.steps_ahead = steps_ahead
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        """
        Forward pass
        Args:
            x: (batch_size, 30, 4)  - 30 historical OHLC candles
            target: (batch_size, 10, 4) - 10 future OHLC candles (for training)
            teacher_forcing_ratio: probability of using ground truth vs prediction
        
        Returns:
            predictions: (batch_size, 10, 4) - predicted next 10 OHLC candles
        """
        
        # Encode
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(x)
        
        # Decode
        # Use teacher forcing during training
        if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing: use ground truth as input
            target_input = target[:, :1, :]  # First target candle
            predictions = self.decoder(
                encoder_outputs, encoder_hidden, encoder_cell,
                target_input=target_input,
                steps_ahead=self.steps_ahead
            )
        else:
            # Autoregressive: use previous prediction
            predictions = self.decoder(
                encoder_outputs, encoder_hidden, encoder_cell,
                target_input=None,
                steps_ahead=self.steps_ahead
            )
        
        return predictions


if __name__ == "__main__":
    # Test model
    batch_size = 8
    seq_len = 30  # Input sequence: 30 candles
    input_size = 4  # OHLC
    steps_ahead = 10  # Predict next 10 candles
    
    model = Seq2SeqLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        steps_ahead=steps_ahead,
        output_size=4
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, input_size)
    target = torch.randn(batch_size, steps_ahead, input_size)
    
    # Forward pass
    output = model(x, target, teacher_forcing_ratio=0.5)
    
    print(f"Model architecture: Seq2Seq LSTM with Attention")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output: (batch_size={batch_size}, steps_ahead={steps_ahead}, features=4)")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
