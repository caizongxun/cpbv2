#!/usr/bin/env python3
"""
V3 Model Visualization V2 - Improved signal generation with trend analysis
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
plt.rcParams['figure.figsize'] = (18, 8)
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


class ModelVisualizerV2:
    """Visualize model predictions with improved trend-following signals"""
    
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
    
    def calculate_trend(self, prices: np.ndarray, period: int = 5):
        """Calculate trend: 1=up, -1=down, 0=neutral"""
        trends = []
        for i in range(len(prices)):
            if i < period:
                trends.append(0)
            else:
                # Compare with period ago
                current = prices[i]
                prev = prices[i - period]
                if current > prev * 1.001:  # 0.1% threshold
                    trends.append(1)  # Uptrend
                elif current < prev * 0.999:
                    trends.append(-1)  # Downtrend
                else:
                    trends.append(0)  # Neutral
        return np.array(trends)
    
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
    
    def generate_signals_v2(self, predicted_prices: np.ndarray, actual_prices: np.ndarray, 
                           threshold: float = 0.002, trend_period: int = 5):
        """Improved signal generation with trend confirmation
        
        Strategy:
        - BUY: Model predicts UP + Current downtrend (bottom reversal)
        - SELL: Model predicts DOWN + Current uptrend (top reversal)  
        - Ignore signals that go against strong trends
        """
        signals = []
        trends = self.calculate_trend(actual_prices, period=trend_period)
        
        for i in range(len(predicted_prices) - 1):
            current_price = actual_prices[i]
            predicted_price = predicted_prices[i]
            current_trend = trends[i]
            
            # Model's prediction
            price_change = (predicted_price - current_price) / current_price
            
            # Signal logic:
            # - Strong reversal signal: model predicts opposite of trend
            # - Only generate if confidence is high (|price_change| > threshold)
            
            if price_change > threshold:  # Model predicts UP
                # Good signal: DOWN trend turning UP (buy dip)
                # Weak signal: already in UP trend (over-bought)
                if current_trend == -1:
                    signals.append((i, 'BUY', current_price, 'STRONG'))  # Reversal buy
                elif current_trend == 1:
                    signals.append((i, 'BUY', current_price, 'WEAK'))   # Trend continuation
                
            elif price_change < -threshold:  # Model predicts DOWN
                # Good signal: UP trend turning DOWN (sell rally)
                # Weak signal: already in DOWN trend (over-sold)
                if current_trend == 1:
                    signals.append((i, 'SELL', current_price, 'STRONG'))  # Reversal sell
                elif current_trend == -1:
                    signals.append((i, 'SELL', current_price, 'WEAK'))   # Trend continuation
        
        return signals
    
    def visualize_pair_v2(self, pair_name: str, num_predictions: int = 100):
        """Visualize predictions with trend and signals"""
        
        print(f"\nVisualizing {pair_name}...")
        
        # Generate predictions
        result = self.generate_predictions(pair_name, num_predictions)
        if result is None:
            return
        
        actual_prices = result['actual_prices']
        predicted_prices = result['predicted_prices']
        
        # Generate signals
        signals = self.generate_signals_v2(predicted_prices, actual_prices, 
                                          threshold=0.002, trend_period=5)
        
        # Calculate trends
        trends = self.calculate_trend(actual_prices, period=5)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Subplot 1: Price and signals
        x = np.arange(len(actual_prices))
        ax1.plot(x, actual_prices, label='Actual Price', color='blue', linewidth=2.5, alpha=0.7)
        ax1.plot(x, predicted_prices, label='Predicted Price', color='orange', linewidth=2, alpha=0.6, linestyle='--')
        
        # Plot signals
        strong_buys = [s for s in signals if s[1] == 'BUY' and s[3] == 'STRONG']
        weak_buys = [s for s in signals if s[1] == 'BUY' and s[3] == 'WEAK']
        strong_sells = [s for s in signals if s[1] == 'SELL' and s[3] == 'STRONG']
        weak_sells = [s for s in signals if s[1] == 'SELL' and s[3] == 'WEAK']
        
        if strong_buys:
            indices = [s[0] for s in strong_buys]
            prices = [s[2] for s in strong_buys]
            ax1.scatter(indices, prices, color='lime', marker='^', s=150, 
                       label=f'Strong BUY ({len(strong_buys)})', zorder=5, edgecolors='darkgreen', linewidth=1.5)
        
        if weak_buys:
            indices = [s[0] for s in weak_buys]
            prices = [s[2] for s in weak_buys]
            ax1.scatter(indices, prices, color='lightgreen', marker='^', s=80, 
                       label=f'Weak BUY ({len(weak_buys)})', zorder=4, alpha=0.6)
        
        if strong_sells:
            indices = [s[0] for s in strong_sells]
            prices = [s[2] for s in strong_sells]
            ax1.scatter(indices, prices, color='red', marker='v', s=150, 
                       label=f'Strong SELL ({len(strong_sells)})', zorder=5, edgecolors='darkred', linewidth=1.5)
        
        if weak_sells:
            indices = [s[0] for s in weak_sells]
            prices = [s[2] for s in weak_sells]
            ax1.scatter(indices, prices, color='lightcoral', marker='v', s=80, 
                       label=f'Weak SELL ({len(weak_sells)})', zorder=4, alpha=0.6)
        
        ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{pair_name} - LSTM Predictions with Trend-Based Signals', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10, ncol=3)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Trend indicator
        trend_colors = ['red' if t == -1 else 'green' if t == 1 else 'gray' for t in trends[-len(actual_prices):]]
        ax2.bar(x, np.ones(len(x)), color=trend_colors, alpha=0.6, width=1.0)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Trend', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time Index', fontsize=12, fontweight='bold')
        ax2.set_yticks([0.5])
        ax2.set_yticklabels([''])
        ax2.grid(True, alpha=0.2, axis='x')
        
        # Add legend for trend
        green_patch = mpatches.Patch(color='green', alpha=0.6, label='Uptrend')
        red_patch = mpatches.Patch(color='red', alpha=0.6, label='Downtrend')
        gray_patch = mpatches.Patch(color='gray', alpha=0.6, label='Neutral')
        ax2.legend(handles=[green_patch, red_patch, gray_patch], loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Statistics
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        total_signals = len(signals)
        strong_signals = len(strong_buys) + len(strong_sells)
        weak_signals = len(weak_buys) + len(weak_sells)
        
        print(f"\n=== {pair_name} Analysis ===")
        print(f"Price Range: {actual_prices.min():.6f} - {actual_prices.max():.6f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"\nTrading Signals Analysis:")
        print(f"  Total Signals: {total_signals}")
        print(f"  Strong Signals (Reversals): {strong_signals} ({strong_buys.__len__()} buys, {strong_sells.__len__()} sells)")
        print(f"  Weak Signals (Continuations): {weak_signals} ({weak_buys.__len__()} buys, {weak_sells.__len__()} sells)")
        print(f"\n  Signal Quality: {strong_signals/max(total_signals, 1)*100:.1f}% are high-confidence reversals")
        
        if strong_buys:
            print(f"\nFirst 5 Strong BUY Signals (Reversal from downtrend):")
            for i, (idx, _, price, _) in enumerate(strong_buys[:5]):
                print(f"  {i+1}. Index {idx}: BUY @ ${price:.6f}")
        
        if strong_sells:
            print(f"\nFirst 5 Strong SELL Signals (Reversal from uptrend):")
            for i, (idx, _, price, _) in enumerate(strong_sells[:5]):
                print(f"  {i+1}. Index {idx}: SELL @ ${price:.6f}")
        
        return {
            'mape': mape,
            'signals': signals,
            'strong_signals': strong_signals,
            'actual_prices': actual_prices,
            'predicted_prices': predicted_prices
        }
    
    def visualize_all_v2(self, num_predictions: int = 100):
        """Visualize all available models"""
        
        # Get available models
        model_files = sorted(list(self.models_dir.glob("v3_model_*.pt")))
        
        if not model_files:
            print("No models found!")
            return
        
        print(f"\nFound {len(model_files)} models")
        print(f"Strategy: Detect REVERSALS (opposite of current trend)")
        print(f"  - Strong BUY: Model predicts UP + Current downtrend (buy dips)")
        print(f"  - Strong SELL: Model predicts DOWN + Current uptrend (sell rallies)\n")
        
        all_results = {}
        
        for idx, model_file in enumerate(model_files, 1):
            pair_name = model_file.stem.replace('v3_model_', '')
            print(f"[{idx}/{len(model_files)}] {pair_name}")
            
            try:
                result = self.visualize_pair_v2(pair_name, num_predictions)
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
            strong_signals = [r['strong_signals'] for r in all_results.values()]
            
            print(f"\nAverage MAPE: {np.mean(mapes):.2f}%")
            print(f"MAPE Range: {np.min(mapes):.2f}% - {np.max(mapes):.2f}%")
            print(f"\nTotal High-Quality Reversal Signals: {sum(strong_signals)}")
            print(f"Reversals per Model: {np.mean(strong_signals):.1f} (avg)")
            
            # Best performing
            print(f"\nBest Performing Models (Lowest MAPE):")
            sorted_models = sorted(all_results.items(), key=lambda x: x[1]['mape'])
            for pair, result in sorted_models[:5]:
                print(f"  {pair}: {result['mape']:.2f}%")
        
        return all_results


if __name__ == "__main__":
    # Initialize visualizer
    visualizer = ModelVisualizerV2()
    
    # Visualize all models with improved signals
    print("\n" + "#"*80)
    print("# V3 Model Visualization V2 - Reversal Detection Strategy")
    print("#"*80)
    
    results = visualizer.visualize_all_v2(num_predictions=100)
    
    print("\nVisualization complete!")
