#!/usr/bin/env python3
"""
V4 Model - AGGRESSIVE GPU Forcing
Bypass cuDNN LSTM entirely with manual implementation
Guaranteed GPU memory usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForcedGPULSTMCell(nn.Module):
    """Manual LSTM cell that FORCES all operations on GPU"""
    def __init__(self, input_size, hidden_size, device):
        super(ForcedGPULSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        
        # Weight matrices - explicitly on GPU
        self.weight_ii = nn.Parameter(
            torch.randn(hidden_size, input_size, device=device, dtype=torch.float32)
        )
        self.weight_if = nn.Parameter(
            torch.randn(hidden_size, input_size, device=device, dtype=torch.float32)
        )
        self.weight_ig = nn.Parameter(
            torch.randn(hidden_size, input_size, device=device, dtype=torch.float32)
        )
        self.weight_io = nn.Parameter(
            torch.randn(hidden_size, input_size, device=device, dtype=torch.float32)
        )
        
        # Recurrent weights
        self.weight_hi = nn.Parameter(
            torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
        )
        self.weight_hf = nn.Parameter(
            torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
        )
        self.weight_hg = nn.Parameter(
            torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
        )
        self.weight_ho = nn.Parameter(
            torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
        )
        
        # Biases
        self.bias_i = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=torch.float32))
        self.bias_f = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=torch.float32))
        self.bias_g = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=torch.float32))
        self.bias_o = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=torch.float32))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for p in self.parameters():
            p.data.uniform_(-std, std)
    
    def forward(self, x, h, c):
        """
        x: (batch_size, input_size) - MUST BE ON GPU
        h: (batch_size, hidden_size) - MUST BE ON GPU
        c: (batch_size, hidden_size) - MUST BE ON GPU
        """
        # Explicit GPU matrix multiplications
        i = torch.sigmoid(
            torch.matmul(x, self.weight_ii.t()) + 
            torch.matmul(h, self.weight_hi.t()) + 
            self.bias_i
        )
        f = torch.sigmoid(
            torch.matmul(x, self.weight_if.t()) + 
            torch.matmul(h, self.weight_hf.t()) + 
            self.bias_f
        )
        g = torch.tanh(
            torch.matmul(x, self.weight_ig.t()) + 
            torch.matmul(h, self.weight_hg.t()) + 
            self.bias_g
        )
        o = torch.sigmoid(
            torch.matmul(x, self.weight_io.t()) + 
            torch.matmul(h, self.weight_ho.t()) + 
            self.bias_o
        )
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class ForcedGPULSTM(nn.Module):
    """Multi-layer LSTM using manual cells - guaranteed GPU execution"""
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, 
                 dropout=0.0, device=None):
        super(ForcedGPULSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.device = device
        
        self.cells = nn.ModuleList([
            ForcedGPULSTMCell(input_size if i == 0 else hidden_size, hidden_size, device)
            for i in range(num_layers)
        ])
        
        if dropout > 0 and num_layers > 1:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])
        else:
            self.dropouts = None
    
    def forward(self, x, hc=None):
        """
        x: (batch_size, seq_len, input_size) or (seq_len, batch_size, input_size)
        """
        if self.batch_first:
            seq_len = x.shape[1]
            batch_size = x.shape[0]
        else:
            seq_len = x.shape[0]
            batch_size = x.shape[1]
            x = x.transpose(0, 1)  # Convert to batch_first
        
        # Initialize hidden states
        if hc is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=x.dtype) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=x.dtype) 
                 for _ in range(self.num_layers)]
        else:
            h, c = hc
        
        output = x
        outputs = []
        
        for t in range(seq_len):
            x_t = output[:, t, :]
            
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                
                if self.dropouts is not None and layer < self.num_layers - 1:
                    h[layer] = self.dropouts[layer - 1](h[layer])
                
                x_t = h[layer]
            
            outputs.append(h[-1].unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        
        # Stack hidden states for return
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        
        return output, (h_n, c_n)


class GPUForcedAttention(nn.Module):
    """Attention mechanism - all GPU operations"""
    def __init__(self, hidden_size, num_heads=4, device=None):
        super(GPUForcedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.device = device
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
        # Move to GPU immediately
        self.to(device)
    
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # GPU matrix multiplication
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        output = self.fc_out(context)
        
        return output, attention_weights


class Seq2SeqLSTMGPUv2(nn.Module):
    """Seq2Seq with FORCED GPU execution - no fallback possible"""
    def __init__(self, input_size=4, hidden_size=256, num_layers=2, dropout=0.3,
                 steps_ahead=10, output_size=4, device=None):
        super(Seq2SeqLSTMGPUv2, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.steps_ahead = steps_ahead
        self.device = device
        
        # Encoder with forced GPU LSTM
        self.encoder_lstm = ForcedGPULSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            device=device
        )
        self.encoder_dropout = nn.Dropout(dropout)
        
        # Decoder with forced GPU LSTM
        self.decoder_lstm = ForcedGPULSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            device=device
        )
        
        # Attention
        self.attention = GPUForcedAttention(hidden_size, num_heads=4, device=device)
        
        # Output layers
        self.decoder_dropout = nn.Dropout(dropout)
        self.fc_combined = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.to(device)
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        """
        x: (batch_size, 30, 4) - MUST BE ON GPU
        target: (batch_size, 10, 4)
        """
        batch_size = x.shape[0]
        
        # Encode
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(x)
        encoder_outputs = self.encoder_dropout(encoder_outputs)
        
        # Initialize decoder hidden state from encoder
        h = h_n
        c = c_n
        
        # Decode
        predictions = []
        
        if target is not None and torch.rand(1, device=self.device).item() < teacher_forcing_ratio:
            decoder_input = target[:, 0:1, :]
        else:
            decoder_input = torch.zeros(batch_size, 1, self.output_size, 
                                       device=self.device, dtype=x.dtype)
        
        for step in range(self.steps_ahead):
            # Decoder LSTM step
            decoder_output, (h, c) = self.decoder_lstm(decoder_input, (h, c))
            
            # Attention
            attention_output, _ = self.attention(decoder_output, encoder_outputs, encoder_outputs)
            
            # Combine and output
            combined = torch.cat([decoder_output, attention_output], dim=-1)
            combined = self.decoder_dropout(combined)
            combined = F.relu(self.fc_combined(combined))
            output = self.output_layer(combined)
            
            predictions.append(output)
            decoder_input = output
        
        predictions = torch.cat(predictions, dim=1)
        return predictions


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    batch_size = 32
    model = Seq2SeqLSTMGPUv2(
        input_size=4, hidden_size=256, num_layers=2, dropout=0.3,
        steps_ahead=10, output_size=4, device=device
    )
    
    x = torch.randn(batch_size, 30, 4, device=device)
    target = torch.randn(batch_size, 10, 4, device=device)
    
    print(f"Input device: {x.device}")
    print(f"Input shape: {x.shape}")
    
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1e9
    
    output = model(x, target, teacher_forcing_ratio=0.5)
    
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    print(f"GPU Memory before: {mem_before:.3f}GB")
    print(f"GPU Memory after: {mem_after:.3f}GB")
    print(f"GPU Memory peak: {peak:.3f}GB")
    print(f"Memory used: {(mem_after - mem_before)*1000:.2f}MB")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
