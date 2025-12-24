#!/usr/bin/env python3
"""
V3 Model Visualization - Visualize predictions and trading signals
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 10


class SimpleLSTM(nn.Module):
    """Simple LSTM model (same as training)"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        pred = self.fc(last_out)
        return pred.squeeze()


class ModelVisualizer:
    """Visualize model predictions and generate trading signals"""
    
    def __init__(self, base_dir: str = '/content'):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.base_dir / 'all_models'
        self.results_dir = self.base_dir / 'results'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self, pair_name: str):
        """Load trained model"""
        model_path = self.models_dir / f"v3_model_{pair_name}.pt"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return None
        
        model = SimpleLSTM().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def generate_predictions(self, pair_name: str, num_predictions: int = 100):
        """Generate predictions for the last num_predictions samples"""
        
        # Load data
        csv_path = self.data_dir / f"{pair_name}.csv"
        if not csv_path.exists():
            print(f"Data not found: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        close_prices = df['Close'].values.astype(np.float32)
        
        # Normalize
        mean_price = close_prices.mean()
        std_price = close_prices.std()
        normalized = (close_prices - mean_price) / (std_price + 1e-8)
        
        # Create sequences
        seq_length = 30
        X, y = [], []
        indices = []
        for i in range(len(normalized) - seq_length):
            X.append(normalized[i:i+seq_length])
            y.append(normalized[i+seq_length])
            indices.append(i+seq_length)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        indices = np.array(indices)
        
        # Load model
        model = self.load_model(pair_name)
        if model is None:
            return None
        
        # Generate predictions on last num_predictions samples
        predictions = []
        with torch.no_grad():
            for i in range(len(X) - num_predictions, len(X)):
                X_seq = torch.FloatTensor(X[i]).to(self.device)
                pred = model(X_seq.unsqueeze(0)).cpu().item()
                predictions.append(pred)
        
        # Denormalize
        predictions = np.array(predictions)
        predictions_price = predictions * std_price + mean_price
        actual_prices = close_prices[-num_predictions:]
        actual_normalized = y[-num_predictions:]
        
        return {
            'actual_prices': actual_prices,
            'predicted_prices': predictions_price,
            'actual_normalized': actual_normalized,
            'predicted_normalized': predictions,
            'mean_price': mean_price,
            'std_price': std_price
        }
    
    def generate_signals(self, predicted_prices: np.ndarray, actual_prices: np.ndarray, threshold: float = 0.002):
        """Generate trading signals based on predictions
        
        Args:
            predicted_prices: Model predicted prices
            actual_prices: Actual prices
            threshold: Min price change threshold to generate signal
        
        Returns:
            List of (index, signal_type, signal_price) tuples
            signal_type: 'BUY' or 'SELL'
        """
        signals = []
        
        for i in range(len(predicted_prices) - 1):
            current_price = actual_prices[i]
            next_price = actual_prices[i + 1]
            predicted_price = predicted_prices[i]
            
            # Price change
            price_change = (predicted_price - current_price) / current_price
            
            # Generate signal
            if price_change > threshold:
                signals.append((i, 'BUY', current_price))
            elif price_change < -threshold:
                signals.append((i, 'SELL', current_price))
        
        return signals
    
    def visualize_pair(self, pair_name: str, num_predictions: int = 100):
        """Visualize predictions and signals for a pair"""
        
        print(f"\nVisualizing {pair_name}...")
        
        # Generate predictions
        result = self.generate_predictions(pair_name, num_predictions)
        if result is None:
            return
        
        actual_prices = result['actual_prices']
        predicted_prices = result['predicted_prices']
        
        # Generate signals
        signals = self.generate_signals(predicted_prices, actual_prices, threshold=0.002)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Plot prices
        x = np.arange(len(actual_prices))
        ax.plot(x, actual_prices, label='Actual Price', color='blue', linewidth=2, alpha=0.7)
        ax.plot(x, predicted_prices, label='Predicted Price', color='orange', linewidth=2, alpha=0.7)
        
        # Plot signals
        buy_signals = [s for s in signals if s[1] == 'BUY']
        sell_signals = [s for s in signals if s[1] == 'SELL']
        
        if buy_signals:
            buy_indices = [s[0] for s in buy_signals]
            buy_prices = [s[2] for s in buy_signals]
            ax.scatter(buy_indices, buy_prices, color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        if sell_signals:
            sell_indices = [s[0] for s in sell_signals]
            sell_prices = [s[2] for s in sell_signals]
            ax.scatter(sell_indices, sell_prices, color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        # Labels and formatting
        ax.set_xlabel('Time Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title(f'{pair_name} - Model Predictions & Trading Signals', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        signal_count = len(signals)
        
        stats_text = f'MAPE: {mape:.2f}% | Signals: {signal_count} | Buys: {len(buy_signals)} | Sells: {len(sell_signals)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        # Print details
        print(f"\n=== {pair_name} Analysis ===")
        print(f"Actual Price Range: {actual_prices.min():.2f} - {actual_prices.max():.2f}")
        print(f"Predicted Price Range: {predicted_prices.min():.2f} - {predicted_prices.max():.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"\nTrading Signals: {signal_count}")
        print(f"  Buy Signals: {len(buy_signals)}")
        print(f"  Sell Signals: {len(sell_signals)}")
        
        if signals:
            print(f"\nFirst 10 Signals:")
            for i, (idx, signal_type, price) in enumerate(signals[:10]):
                print(f"  {i+1}. Index {idx}: {signal_type} @ ${price:.2f}")
        
        return {
            'mape': mape,
            'signals': signals,
            'actual_prices': actual_prices,
            'predicted_prices': predicted_prices
        }
    
    def visualize_all(self, num_predictions: int = 100):
        """Visualize all available models"""
        
        # Get available models
        model_files = sorted(list(self.models_dir.glob("v3_model_*.pt")))
        
        if not model_files:
            print("No models found!")
            return
        
        print(f"\nFound {len(model_files)} models")
        print(f"Visualizing with {num_predictions} predictions per model\n")
        
        all_results = {}
        
        for idx, model_file in enumerate(model_files, 1):
            pair_name = model_file.stem.replace('v3_model_', '')
            print(f"[{idx}/{len(model_files)}] Processing {pair_name}...")
            
            try:
                result = self.visualize_pair(pair_name, num_predictions)
                if result:
                    all_results[pair_name] = result
            except Exception as e:
                print(f"  Error: {e}")
        
        # Summary
        print(f"\n\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        if all_results:
            mapes = [r['mape'] for r in all_results.values()]
            signal_counts = [len(r['signals']) for r in all_results.values()]
            
            print(f"\nAverage MAPE: {np.mean(mapes):.2f}%")
            print(f"MAPE Range: {np.min(mapes):.2f}% - {np.max(mapes):.2f}%")
            print(f"\nTotal Signals: {sum(signal_counts)}")
            print(f"Signals per Model: {np.mean(signal_counts):.1f} (avg)")
            
            # Best performing models
            print(f"\nBest Performing Models (Lowest MAPE):")
            sorted_models = sorted(all_results.items(), key=lambda x: x[1]['mape'])
            for pair, result in sorted_models[:5]:
                print(f"  {pair}: {result['mape']:.2f}%")
        
        return all_results


if __name__ == "__main__":
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Visualize all models
    print("\n" + "#"*80)
    print("# V3 Model Visualization - Predictions & Trading Signals")
    print("#"*80)
    
    results = visualizer.visualize_all(num_predictions=100)
    
    print("\nVisualization complete!")
