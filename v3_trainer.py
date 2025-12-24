#!/usr/bin/env python3
"""
V3 Advanced Trainer for Cryptocurrency Price Prediction
Features: Dynamic learning rate, warm-up, early stopping, gradient clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path
import json


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 20, verbose: bool = True, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


class GradientAccumulator:
    """Gradient accumulation for larger effective batch size"""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.step = 0
    
    def should_step(self) -> bool:
        self.step += 1
        return self.step % self.accumulation_steps == 0
    
    def reset(self):
        self.step = 0


class V3Trainer:
    """V3 Advanced Trainer"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        gradient_clip_value: float = 1.0
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.gradient_clip_value = gradient_clip_value
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Loss function with smoothing
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        # Learning rate scheduler with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-5,
            last_epoch=-1
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mape': [],
            'val_mape': [],
            'learning_rate': []
        }
    
    def train_epoch(
        self,
        train_loader,
        accumulation_steps: int = 1
    ) -> Tuple[float, float]:
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        total_mape = 0.0
        num_batches = 0
        
        accumulator = GradientAccumulator(accumulation_steps)
        self.optimizer.zero_grad()
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            output, _ = self.model(X)
            loss = self.loss_fn(output, y)
            
            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            # Accumulate metrics
            total_loss += loss.item() * accumulation_steps
            total_mape += self._calculate_mape(output.detach(), y.detach())
            num_batches += 1
            
            # Optimizer step
            if accumulator.should_step():
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Final step if needed
        if accumulator.step % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
            self.optimizer.step()
        
        avg_loss = total_loss / num_batches
        avg_mape = total_mape / num_batches
        
        return avg_loss, avg_mape
    
    @torch.no_grad()
    def validate(
        self,
        val_loader
    ) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        total_mape = 0.0
        num_batches = 0
        
        for X, y in val_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            output, _ = self.model(X)
            loss = self.loss_fn(output, y)
            
            total_loss += loss.item()
            total_mape += self._calculate_mape(output, y)
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mape = total_mape / num_batches
        
        return avg_loss, avg_mape
    
    @staticmethod
    def _calculate_mape(output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mape = torch.mean(torch.abs((target - output) / (torch.abs(target) + 1e-8))) * 100
        return mape.item()
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        patience: int = 20,
        save_path: Optional[str] = None,
        accumulation_steps: int = 1
    ) -> Dict:
        """Full training loop"""
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        best_val_loss = float('inf')
        
        logging.info(f"Starting training for {epochs} epochs...")
        logging.info(f"Device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_mape = self.train_epoch(train_loader, accumulation_steps)
            
            # Validation
            val_loss, val_mape = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mape'].append(train_mape)
            self.history['val_mape'].append(val_mape)
            self.history['learning_rate'].append(current_lr)
            
            # Logging
            if epoch % 5 == 0 or epoch == 1:
                logging.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Train MAPE: {train_mape:.4f}% | "
                    f"Val MAPE: {val_mape:.4f}% | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logging.info(f"Model saved to {save_path}")
            
            # Early stopping
            if early_stopping(val_loss, epoch):
                logging.info(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'best_val_loss': best_val_loss,
            'best_epoch': early_stopping.best_epoch,
            'total_epochs': epoch,
            'history': self.history
        }
    
    def save_history(self, path: str):
        """Save training history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logging.info(f"Checkpoint loaded from {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("V3 Trainer module loaded successfully")
