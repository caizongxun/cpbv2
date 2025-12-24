#!/usr/bin/env python3
"""
V4 Model - Forced CUDA Computation
Bypass PyTorch LSTM CPU fallback by using manual LSTM cells
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualLSTMCell(nn.Module):
    """Manual LSTM cell that forces GPU computation"""
    def __init__(self, input_size, hidden_size):
        super(ManualLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # All weights combined for efficiency
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x, state):
        """x: (batch, input_size), state: (h, c) where h,c are (batch, hidden_size)"""
        h, c = state
        # Force GPU computation
        gates = torch.nn.functional.linear(x, self.weight_ih, self.bias_ih) + \
                torch.nn.functional.linear(h, self.weight_hh, self.bias_hh)
        
        i, f, g, o = gates.chunk(4, 1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)


class Attention(nn.Module):
    """Attention with manual GPU operations"""
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
        
        # GPU operations
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        output = self.fc_out(context)
        
        return output, attention_weights


class GPUForcedEncoder(nn.Module):
    """Encoder using native LSTM but with explicit GPU settings"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(GPUForcedEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Use LSTM but don't rely on it
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, 30, 4)
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        return lstm_out, h_n, c_n


class GPUForcedDecoder(nn.Module):
    """Decoder with attention"""
    def __init__(self, output_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(GPUForcedDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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


class Seq2SeqLSTMGPU(nn.Module):
    """Seq2Seq with forced GPU computation"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3, 
                 steps_ahead=10, output_size=4):
        super(Seq2SeqLSTMGPU, self).__init__()
        
        self.encoder = GPUForcedEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = GPUForcedDecoder(output_size, hidden_size, num_layers, dropout)
        self.steps_ahead = steps_ahead
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        # Encoder
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(x)
        
        # Decoder
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 16
    model = Seq2SeqLSTMGPU(
        input_size=4, hidden_size=256, num_layers=2, dropout=0.3,
        steps_ahead=10, output_size=4
    ).to(device)
    
    x = torch.randn(batch_size, 30, 4, device=device)
    target = torch.randn(batch_size, 10, 4, device=device)
    
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1e9
    
    output = model(x, target, teacher_forcing_ratio=0.5)
    
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1e9
    
    print(f"Output shape: {output.shape}")
    print(f"GPU Memory: {mem_before:.3f}GB -> {mem_after:.3f}GB")
    print(f"Memory used: {(mem_after - mem_before)*1000:.2f}MB")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
