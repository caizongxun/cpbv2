"""
CPB V2 Production Training - Per Pair Model (Like V1 Structure)

Training Logic:
- Same as V1 training logic
- Each currency pair has its own model
- Model outputs [price, volatility] instead of just [price]
- Train 20 individual models (one per pair)
- Save as V2

Structure:
  models/
  ├── v2_model_BTC_USDT.h5
  ├── v2_model_ETH_USDT.h5
  ├── v2_model_SOL_USDT.h5
  ├── v2_model_XRP_USDT.h5
  └── v2_training_summary.json
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import os
import json
from datetime import datetime

print("="*70)
print("CPB V2 Model Training - Per Pair (Like V1 Structure)")
print("="*70)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== Data Generation ====================
def generate_training_data_for_pair(pair, num_sequences=1000, seq_length=20):
    """
    Generate synthetic OHLCV data for a specific pair
    
    Features: [Open, High, Low, Close]
    Output: [Predicted_Price, Predicted_Volatility]
    """
    print(f"   Generating {num_sequences} sequences for {pair}...")
    
    X = []
    Y_price = []
    Y_volatility = []
    
    for seq in range(num_sequences):
        # Generate random price movements
        start_price = np.random.uniform(80000, 90000)
        prices = [start_price]
        
        for _ in range(seq_length):
            # Random walk with drift
            daily_return = np.random.normal(0.0005, 0.015)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # Generate OHLCV data
        sequence_data = []
        for i in range(seq_length):
            open_price = prices[i]
            close_price = prices[i+1]
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
            
            # Normalize OHLCV
            sequence_data.append([
                (open_price - start_price) / start_price * 100,
                (high_price - start_price) / start_price * 100,
                (low_price - start_price) / start_price * 100,
                (close_price - start_price) / start_price * 100
            ])
        
        X.append(sequence_data)
        
        # Calculate next price movement (target)
        next_return = (prices[-1] - start_price) / start_price * 100
        Y_price.append([next_return])
        
        # Calculate volatility (standard deviation of returns)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * 100
        Y_volatility.append([volatility])
    
    X = np.array(X)
    Y_price = np.array(Y_price)
    Y_volatility = np.array(Y_volatility)
    
    return X, Y_price, Y_volatility

# ==================== Model Architecture ====================
def create_v2_model_for_pair(input_shape=(20, 4)):
    """
    Create V2 model for a specific pair
    
    Input: OHLCV sequences (20, 4)
    Output: [price_prediction, volatility_prediction]
    """
    model = Sequential([
        # LSTM layers
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        
        # Multi-output layer
        layers.Dense(2, activation='linear')  # [price, volatility]
    ])
    
    return model

# ==================== Training ====================
def train_v2_model_for_pair(pair, X, Y_price, Y_volatility, epochs=50, batch_size=32):
    """
    Train V2 model for a specific pair
    """
    print(f"   Training model for {pair}...")
    
    # Combine targets
    Y = np.concatenate([Y_price, Y_volatility], axis=1)
    
    # Create model
    model = create_v2_model_for_pair(input_shape=(X.shape[1], X.shape[2]))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    history = model.fit(
        X, Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    return model, history

# ==================== Model Evaluation ====================
def evaluate_model_for_pair(model, X_test, Y_price_test, Y_volatility_test):
    """
    Evaluate model performance for a pair
    """
    Y_test = np.concatenate([Y_price_test, Y_volatility_test], axis=1)
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    pred_prices = predictions[:, 0]
    pred_volatilities = predictions[:, 1]
    
    actual_prices = Y_price_test.flatten()
    actual_volatilities = Y_volatility_test.flatten()
    
    # Calculate metrics
    price_mae = np.mean(np.abs(pred_prices - actual_prices))
    price_rmse = np.sqrt(np.mean((pred_prices - actual_prices)**2))
    vol_mae = np.mean(np.abs(pred_volatilities - actual_volatilities))
    vol_rmse = np.sqrt(np.mean((pred_volatilities - actual_volatilities)**2))
    
    return {
        'price_mae': float(price_mae),
        'price_rmse': float(price_rmse),
        'vol_mae': float(vol_mae),
        'vol_rmse': float(vol_rmse)
    }

# ==================== Main Training Loop ====================
def main():
    """
    Train V2 models for each pair (Like V1 structure)
    """
    
    # Currency pairs (same structure as V1)
    pairs = [
        'BTC_USDT',
        'ETH_USDT', 
        'SOL_USDT',
        'XRP_USDT',
        'ADA_USDT',
        'BNB_USDT',
        'DOGE_USDT',
        'LINK_USDT',
        'AVAX_USDT',
        'MATIC_USDT',
        'ATOM_USDT',
        'NEAR_USDT',
        'FTM_USDT',
        'ARB_USDT',
        'OP_USDT',
        'LIT_USDT',
        'STX_USDT',
        'INJ_USDT',
        'LUNC_USDT',
        'LUNA_USDT'
    ]
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    training_results = []
    
    for idx, pair in enumerate(pairs, 1):
        print(f"\n{'='*70}")
        print(f"Training Model {idx}/{len(pairs)}: {pair}")
        print(f"{'='*70}")
        
        try:
            # Generate data for this pair
            print(f"[1] Data Generation")
            X, Y_price, Y_volatility = generate_training_data_for_pair(
                pair,
                num_sequences=1000,
                seq_length=20
            )
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            Y_price_train, Y_price_test = Y_price[:split_idx], Y_price[split_idx:]
            Y_vol_train, Y_vol_test = Y_volatility[:split_idx], Y_volatility[split_idx:]
            
            # Train model
            print(f"[2] Model Training")
            model, history = train_v2_model_for_pair(
                pair,
                X_train, Y_price_train, Y_vol_train,
                epochs=50,
                batch_size=32
            )
            print(f"      Final Loss: {history.history['loss'][-1]:.6f}")
            
            # Evaluate model
            print(f"[3] Model Evaluation")
            metrics = evaluate_model_for_pair(model, X_test, Y_price_test, Y_vol_test)
            print(f"      Price MAE: {metrics['price_mae']:.6f}")
            print(f"      Volatility MAE: {metrics['vol_mae']:.6f}")
            
            # Save model
            model_path = f'models/v2_model_{pair}.h5'
            model.save(model_path)
            print(f"[4] Model Saved: {model_path}")
            
            training_results.append({
                'pair': pair,
                'model_idx': idx,
                'metrics': metrics,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"[OK] {pair} Training Complete")
            
        except Exception as e:
            print(f"[ERROR] Training failed for {pair}: {e}")
            training_results.append({
                'pair': pair,
                'model_idx': idx,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            continue
    
    # Save training summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    
    successful = sum(1 for r in training_results if r['status'] == 'success')
    failed = sum(1 for r in training_results if r['status'] == 'failed')
    
    summary_path = 'models/v2_training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'total_models': len(pairs),
            'successful_models': successful,
            'failed_models': failed,
            'training_date': datetime.now().isoformat(),
            'model_type': 'V2_PER_PAIR',
            'model_output': '[price, volatility]',
            'training_results': training_results
        }, f, indent=2)
    
    print(f"\n[OK] Successfully trained {successful}/{len(pairs)} models")
    print(f"[OK] Training summary saved: {summary_path}")
    print(f"[OK] All models saved in: ./models/")
    
    print(f"\n{'='*70}")
    print("Model List:")
    print(f"{'='*70}")
    for result in training_results:
        if result['status'] == 'success':
            print(f"✓ v2_model_{result['pair']}.h5")
        else:
            print(f"✗ v2_model_{result['pair']}.h5 (Failed)")
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    
    return training_results

# ==================== Entry Point ====================
if __name__ == '__main__':
    results = main()
