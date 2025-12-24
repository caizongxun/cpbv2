#!/usr/bin/env python3
"""
V3 Advanced LSTM Model for Cryptocurrency Price Prediction
Features: Attention mechanism, Bi-LSTM, Advanced regularization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class AttentionLayer(nn.Module):
    """Attention mechanism layer for LSTM"""
    def __init__(self, hidden_size: int, attention_size: int = None):
        super(AttentionLayer, self).__init__()
        if attention_size is None:
            attention_size = hidden_size
        
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size)
            attention_weights: (batch_size, seq_len, 1)
        """
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context, attention_weights


class V3LSTMModel(nn.Module):
    """V3 Advanced LSTM with Attention and Bidirectional processing"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        output_size: int = 1,
        use_layer_norm: bool = True
    ):
        super(V3LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.output_size = output_size
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size) if use_layer_norm else nn.Identity()
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(lstm_output_size, attention_size=hidden_size)
            fusion_input_size = lstm_output_size
        else:
            fusion_input_size = lstm_output_size
        
        # Dense layers with layer normalization
        self.dense_layers = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for layer in self.dense_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_size)
            hidden: Optional initial hidden state
        Returns:
            output: (batch_size, output_size)
            hidden: Updated hidden state
        """
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out: (batch_size, seq_len, hidden_size*2) if bidirectional
        
        # Attention mechanism
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
            # context: (batch_size, hidden_size*2)
        else:
            context = lstm_out[:, -1, :]
        
        # Dense layers
        output = self.dense_layers(context)
        
        return output, hidden
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability"""
        if not self.use_attention:
            return None
        
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        _, attention_weights = self.attention(lstm_out)
        return attention_weights.squeeze(-1)  # (batch, seq_len)


class V3TrainingConfig:
    """V3 Model training configuration"""
    
    # Model architecture
    INPUT_SIZE = 30  # Number of features after PCA/selection
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT = 0.3
    BIDIRECTIONAL = True
    USE_ATTENTION = True
    OUTPUT_SIZE = 1
    
    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MIN_LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-5
    
    # Regularization
    EARLY_STOPPING_PATIENCE = 20
    GRADIENT_CLIP_VALUE = 1.0
    
    # Data split
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Sequence parameters
    SEQUENCE_LENGTH = 60  # 60 timesteps
    PREDICTION_HORIZON = 1  # Predict next 1 step


def create_v3_model(
    config: V3TrainingConfig,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> V3LSTMModel:
    """Create V3 LSTM model with config"""
    model = V3LSTMModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
        use_attention=config.USE_ATTENTION,
        output_size=config.OUTPUT_SIZE,
        use_layer_norm=True
    )
    return model.to(device)


if __name__ == "__main__":
    # Test model
    config = V3TrainingConfig()
    model = create_v3_model(config)
    
    # Dummy input
    batch_size = 32
    seq_len = 60
    x = torch.randn(batch_size, seq_len, config.INPUT_SIZE)
    
    output, hidden = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get attention weights
    attention_weights = model.get_attention_weights(x)
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")
