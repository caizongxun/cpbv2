import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM model for cryptocurrency price prediction.
    Optimized for Google Colab free tier GPU memory constraints.
    """
    
    def __init__(
        self,
        input_size: int = 30,
        lstm_units: list = None,
        dropout_lstm: float = 0.2,
        dropout_dense: float = 0.1,
        dense_units: int = 32,
        output_size: int = 1,
        bidirectional: bool = True
    ):
        super(LSTMModel, self).__init__()
        
        if lstm_units is None:
            lstm_units = [96, 64]
        
        self.input_size = input_size
        self.lstm_units = lstm_units
        self.bidirectional = bidirectional
        self.dropout_lstm = dropout_lstm
        self.dropout_dense = dropout_dense
        
        # LSTM Layer 1
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units[0],
            batch_first=True,
            dropout=dropout_lstm if len(lstm_units) > 1 else 0,
            bidirectional=bidirectional
        )
        
        # LSTM Layer 2
        lstm1_output_size = lstm_units[0] * (2 if bidirectional else 1)
        self.lstm2 = nn.LSTM(
            input_size=lstm1_output_size,
            hidden_size=lstm_units[1],
            batch_first=True,
            dropout=dropout_lstm,
            bidirectional=bidirectional
        )
        
        # Calculate dense input size
        lstm2_output_size = lstm_units[1] * (2 if bidirectional else 1)
        
        # Dense Layers
        self.dropout1 = nn.Dropout(dropout_dense)
        self.dense1 = nn.Linear(lstm2_output_size, dense_units)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_dense)
        self.dense2 = nn.Linear(dense_units, output_size)
        
        logger.info(f"LSTM Model initialized:")
        logger.info(f"  - Input size: {input_size}")
        logger.info(f"  - LSTM units: {lstm_units}")
        logger.info(f"  - Bidirectional: {bidirectional}")
        logger.info(f"  - Dense units: {dense_units}")
        logger.info(f"  - Output size: {output_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM layers
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Take last output
        last_output = lstm2_out[:, -1, :]
        
        # Dense layers
        dense_out = self.dense1(last_output)
        dense_out = self.relu(dense_out)
        dense_out = self.dropout1(dense_out)
        
        output = self.dense2(dense_out)
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @staticmethod
    def create_model(config: dict, device: str = 'cuda'):
        """
        Factory method to create model from config dict.
        
        Args:
            config: Configuration dictionary
            device: Device to place model on
            
        Returns:
            Model instance on specified device
        """
        model_config = config.get('model_architecture', {})
        model = LSTMModel(
            input_size=model_config.get('input_size', 30),
            lstm_units=model_config.get('lstm_units', [96, 64]),
            dropout_lstm=model_config.get('dropout_lstm', 0.2),
            dropout_dense=model_config.get('dropout_dense', 0.1),
            dense_units=model_config.get('dense_units', 32),
            output_size=model_config.get('output_size', 1),
            bidirectional=model_config.get('bidirectional', True)
        )
        
        model = model.to(device)
        return model


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0001,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.patience_counter = 0
        self.best_weights = None
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Validation loss
            model: Model instance
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.best_weights = model.state_dict()
            self.best_epoch += 1
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info(f"Restored weights from epoch {self.best_epoch}")
                return True
        
        return False
