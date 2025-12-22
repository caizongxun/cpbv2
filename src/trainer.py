import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Trainer:
    """
    PyTorch LSTM trainer with early stopping and checkpoint management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        config: Dict = None
    ):
        self.model = model
        self.device = device
        self.config = config or {}
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {model.count_parameters():,}")
    
    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        return train_loader, val_loader
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 15,
        min_delta: float = 0.0001,
        save_path: str = None
    ) -> Dict:
        """
        Complete training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            min_delta: Minimum improvement threshold
            save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Learning rate: {learning_rate}")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            
            # Log
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.best_epoch = epoch + 1
                
                # Save best model
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Saved best model at epoch {self.best_epoch}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    if save_path:
                        self.model.load_state_dict(torch.load(save_path))
                    break
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': self.best_epoch,
            'total_epochs': epoch + 1
        }
        
        logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
        return history
    
    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"Model loaded from {filepath}")
