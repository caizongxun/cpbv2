"""
CPB V2 Production Training - With Volatility Output

Training Logic:
- Same as V1 training logic
- But model outputs [price, volatility] instead of just [price]
- Train 20 models for different currency pairs
- Save as V2
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import os
import json
from datetime import datetime, timedelta

print("="*70)
print("CPB V2 Model Training - With Volatility Output")
print("="*70)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== Data Generation ====================
def generate_training_data(num_sequences=1000, seq_length=20, num_features=4):
    """
    Generate synthetic OHLCV data for training
    
    Features: [Open, High, Low, Close, Volume]
    Output: [Predicted_Price, Predicted_Volatility]
    """
    print(f"[1] Generating {num_sequences} training sequences...")
    
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
            volume = np.random.uniform(100, 1000)
            
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
    
    print(f"   Generated X shape: {X.shape}")
    print(f"   Generated Y_price shape: {Y_price.shape}")
    print(f"   Generated Y_volatility shape: {Y_volatility.shape}")
    
    return X, Y_price, Y_volatility

# ==================== Model Architecture ====================
def create_v2_model(input_shape=(20, 4)):
    """
    Create V2 model that outputs both price and volatility
    
    Input: OHLCV sequences (20, 4)
    Output: [price_prediction, volatility_prediction]
    """
    print("\n[2] Creating V2 Model Architecture...")
    
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
        # Output 1: Price prediction
        # Output 2: Volatility prediction
        layers.Dense(2, activation='linear')  # [price, volatility]
    ])
    
    print(f"   Model Summary:")
    model.summary()
    
    return model

# ==================== Training ====================
def train_v2_model(X, Y_price, Y_volatility, epochs=50, batch_size=32):
    """
    Train V2 model with combined loss function
    """
    print("\n[3] Training V2 Model...")
    
    # Combine targets
    Y = np.concatenate([Y_price, Y_volatility], axis=1)
    
    # Create model
    model = create_v2_model(input_shape=(X.shape[1], X.shape[2]))
    
    # Compile model
    # Use different loss weights for price and volatility
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
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    print("\n   Training Complete!")
    print(f"   Final Training Loss: {history.history['loss'][-1]:.6f}")
    print(f"   Final Validation Loss: {history.history['val_loss'][-1]:.6f}")
    
    return model, history

# ==================== Model Evaluation ====================
def evaluate_model(model, X_test, Y_price_test, Y_volatility_test):
    """
    Evaluate model performance
    """
    print("\n[4] Evaluating Model...")
    
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
    
    print(f"\n   Price Predictions:")
    print(f"      MAE: {price_mae:.6f}")
    print(f"      RMSE: {price_rmse:.6f}")
    print(f"\n   Volatility Predictions:")
    print(f"      MAE: {vol_mae:.6f}")
    print(f"      RMSE: {vol_rmse:.6f}")
    
    return {
        'price_mae': price_mae,
        'price_rmse': price_rmse,
        'vol_mae': vol_mae,
        'vol_rmse': vol_rmse
    }

# ==================== Main Training Loop ====================
def main():
    """
    Train V2 models for 20 different pairs
    """
    
    # Currency pairs (simulated)
    pairs = [
        'BTC_USDT', 'BTC_USDT_1', 'BTC_USDT_2', 'BTC_USDT_3', 'BTC_USDT_4',
        'ETH_USDT', 'ETH_USDT_1', 'ETH_USDT_2', 'ETH_USDT_3', 'ETH_USDT_4',
        'SOL_USDT', 'SOL_USDT_1', 'SOL_USDT_2', 'SOL_USDT_3', 'SOL_USDT_4',
        'XRP_USDT', 'XRP_USDT_1', 'XRP_USDT_2', 'XRP_USDT_3', 'XRP_USDT_4'
    ]
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    training_results = []
    
    for idx, pair in enumerate(pairs, 1):
        print(f"\n{'='*70}")
        print(f"Training Model {idx}/{len(pairs)}: {pair}")
        print(f"{'='*70}")
        
        try:
            # Generate data
            X, Y_price, Y_volatility = generate_training_data(
                num_sequences=1000,
                seq_length=20,
                num_features=4
            )
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            Y_price_train, Y_price_test = Y_price[:split_idx], Y_price[split_idx:]
            Y_vol_train, Y_vol_test = Y_volatility[:split_idx], Y_volatility[split_idx:]
            
            # Train model
            model, history = train_v2_model(
                X_train, Y_price_train, Y_vol_train,
                epochs=50,
                batch_size=32
            )
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, Y_price_test, Y_vol_test)
            
            # Save model
            model_path = f'models/v2_model_{pair}.h5'
            model.save(model_path)
            print(f"\n   [OK] Model saved: {model_path}")
            
            training_results.append({
                'pair': pair,
                'model_idx': idx,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"   [ERROR] Training failed: {e}")
            continue
    
    # Save training summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    
    summary_path = 'models/v2_training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'total_models': len(pairs),
            'successful_models': len(training_results),
            'training_date': datetime.now().isoformat(),
            'model_type': 'V2_WITH_VOLATILITY',
            'model_output': '[price, volatility]',
            'training_results': training_results
        }, f, indent=2)
    
    print(f"\n[OK] Training summary saved: {summary_path}")
    print(f"[OK] Successfully trained {len(training_results)}/{len(pairs)} models")
    print(f"[OK] All models saved in: ./models/")
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    
    return training_results

# ==================== Entry Point ====================
if __name__ == '__main__':
    results = main()
    
    print("\n[SUMMARY]")
    for result in results:
        print(f"\n{result['pair']}:")
        print(f"  Price MAE: {result['metrics']['price_mae']:.6f}")
        print(f"  Volatility MAE: {result['metrics']['vol_mae']:.6f}")
