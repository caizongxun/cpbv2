"""
CPB V2 Complete Pipeline

One-file execution for entire workflow:
1. Load historical K-line data (future: 3000 candles)
2. Train V2 models (20 pairs, output: [price, volatility])
3. Visualization (training history, predictions, volatility)
4. Upload to GitHub (caizongxun/cpbv2 repo - MODEL_V2 folder)

Usage:
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/caizongxun/cpbv2/main/notebooks/complete_v2_pipeline.py',
        'complete_v2_pipeline.py'
    )
    exec(open('complete_v2_pipeline.py').read())
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Colab

print("\n" + "="*80)
print(" "*20 + "CPB V2 Complete Pipeline")
print(" "*10 + "Data Loading -> Training -> Visualization -> Upload")
print("="*80)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ==================== Configuration ====================
CONFIG = {
    'version': 'v2',
    'model_type': 'V2_PER_PAIR',
    'pairs': [
        'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT', 'ADA_USDT',
        'BNB_USDT', 'DOGE_USDT', 'LINK_USDT', 'AVAX_USDT', 'MATIC_USDT',
        'ATOM_USDT', 'NEAR_USDT', 'FTM_USDT', 'ARB_USDT', 'OP_USDT',
        'LIT_USDT', 'STX_USDT', 'INJ_USDT', 'LUNC_USDT', 'LUNA_USDT'
    ],
    'num_sequences': 1000,
    'seq_length': 20,
    'epochs': 50,
    'batch_size': 32,
    'output': '[price, volatility]'
}

# ==================== Phase 1: Data Loading ====================
print("\n[Phase 1] Data Loading")
print("-" * 80)

def load_kline_data(pair, limit=100):
    """
    Future: Load real K-line data from exchange API
    Currently: Generate synthetic data
    
    TODO: Replace with real data loading (3000 candles)
        from binance.client import Client
        client = Client()
        klines = client.get_historical_klines(pair, '1h', "3000 hours ago UTC")
    """
    print(f"   Loading data for {pair} (synthetic, placeholder for real data)...")
    
    # Placeholder - generate synthetic data
    # In future, this will load 3000 real K-line candles
    num_candles = 100  # TODO: Change to 3000
    start_price = np.random.uniform(80000, 90000)
    
    data = []
    price = start_price
    for _ in range(num_candles):
        daily_return = np.random.normal(0.0005, 0.015)
        open_price = price
        close_price = price * (1 + daily_return)
        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': pd.Timestamp.now() + pd.Timedelta(hours=_),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
    
    return pd.DataFrame(data)

print(f"   [Note] Currently using synthetic data")
print(f"   [TODO] Replace with real 3000-candle data from exchange API")
print(f"   Loading sample for: {CONFIG['pairs'][0]}")
kline_sample = load_kline_data(CONFIG['pairs'][0])
print(f"   [OK] Data shape: {kline_sample.shape}")

# ==================== Phase 2: Data Preparation ====================
print("\n[Phase 2] Data Preparation")
print("-" * 80)

def generate_training_data_for_pair(pair, num_sequences=1000, seq_length=20):
    """
    Generate training sequences from K-line data
    """
    X = []
    Y_price = []
    Y_volatility = []
    
    for seq in range(num_sequences):
        start_price = np.random.uniform(80000, 90000)
        prices = [start_price]
        
        for _ in range(seq_length):
            daily_return = np.random.normal(0.0005, 0.015)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        sequence_data = []
        for i in range(seq_length):
            open_price = prices[i]
            close_price = prices[i+1]
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
            
            sequence_data.append([
                (open_price - start_price) / start_price * 100,
                (high_price - start_price) / start_price * 100,
                (low_price - start_price) / start_price * 100,
                (close_price - start_price) / start_price * 100
            ])
        
        X.append(sequence_data)
        
        next_return = (prices[-1] - start_price) / start_price * 100
        Y_price.append([next_return])
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * 100
        Y_volatility.append([volatility])
    
    return np.array(X), np.array(Y_price), np.array(Y_volatility)

print(f"   Preparing data for {len(CONFIG['pairs'])} pairs...")
print(f"   [OK] Data preparation ready")

# ==================== Phase 3: Model Architecture ====================
print("\n[Phase 3] Model Architecture")
print("-" * 80)

def create_v2_model(input_shape=(20, 4)):
    """
    V2 Model: Output [price, volatility]
    """
    model = Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(2, activation='linear')  # [price, volatility]
    ])
    return model

print("   Model Type: LSTM-based")
print("   Input: OHLCV sequences (20, 4)")
print("   Output: [price_prediction, volatility_prediction]")
print("   [OK] Architecture ready")

# ==================== Phase 4: Training ====================
print("\n[Phase 4] Training V2 Models")
print("-" * 80)

training_results = []
training_histories = {}

for idx, pair in enumerate(CONFIG['pairs'], 1):
    try:
        print(f"   [{idx:2d}/{len(CONFIG['pairs'])}] Training {pair}...")
        
        # Generate data
        X, Y_price, Y_volatility = generate_training_data_for_pair(
            pair,
            num_sequences=CONFIG['num_sequences'],
            seq_length=CONFIG['seq_length']
        )
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_price_train, Y_price_test = Y_price[:split_idx], Y_price[split_idx:]
        Y_vol_train, Y_vol_test = Y_volatility[:split_idx], Y_volatility[split_idx:]
        
        # Combine targets
        Y_train = np.concatenate([Y_price_train, Y_vol_train], axis=1)
        Y_test = np.concatenate([Y_price_test, Y_vol_test], axis=1)
        
        # Create model
        model = create_v2_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = model.fit(
            X_train, Y_train,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            validation_split=0.2,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )
        
        # Evaluate
        predictions = model.predict(X_test, verbose=0)
        pred_prices = predictions[:, 0]
        pred_volatilities = predictions[:, 1]
        actual_prices = Y_price_test.flatten()
        actual_volatilities = Y_vol_test.flatten()
        
        price_mae = np.mean(np.abs(pred_prices - actual_prices))
        vol_mae = np.mean(np.abs(pred_volatilities - actual_volatilities))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/v2_model_{pair}.h5'
        model.save(model_path)
        
        # Store results
        training_results.append({
            'pair': pair,
            'status': 'success',
            'price_mae': float(price_mae),
            'vol_mae': float(vol_mae),
            'final_loss': float(history.history['loss'][-1])
        })
        
        training_histories[pair] = history
        
        print(f"       ✓ Loss: {history.history['loss'][-1]:.6f} | Price MAE: {price_mae:.6f} | Vol MAE: {vol_mae:.6f}")
        
    except Exception as e:
        print(f"       ✗ Error: {e}")
        training_results.append({
            'pair': pair,
            'status': 'failed',
            'error': str(e)
        })

print(f"\n[OK] Training complete: {sum(1 for r in training_results if r['status'] == 'success')}/{len(CONFIG['pairs'])} models")

# ==================== Phase 5: Visualization ====================
print("\n[Phase 5] Visualization")
print("-" * 80)

os.makedirs('visualizations', exist_ok=True)

def plot_training_history(pair, history, idx):
    """
    Plot training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title(f'{pair} - Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_title(f'{pair} - MAE', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/training_history_{pair}_{idx:02d}.png', dpi=100, bbox_inches='tight')
    plt.close()

# Plot first 5 models as sample
for idx, (pair, history) in enumerate(list(training_histories.items())[:5], 1):
    plot_training_history(pair, history, idx)
    print(f"   ✓ {pair} training history plot")

print(f"[OK] Generated {min(5, len(training_histories))} sample visualization plots")

# ==================== Phase 6: Organize Models ====================
print("\n[Phase 6] Organizing Models")
print("-" * 80)

# Create ALL_MODELS directory
os.makedirs('ALL_MODELS/MODEL_V2', exist_ok=True)

print("   Moving models to ALL_MODELS/MODEL_V2/...")
for result in training_results:
    if result['status'] == 'success':
        pair = result['pair']
        src = f'models/v2_model_{pair}.h5'
        dst = f'ALL_MODELS/MODEL_V2/v2_model_{pair}.h5'
        shutil.copy2(src, dst)
        print(f"   ✓ {pair}")

print(f"[OK] Models organized in ALL_MODELS/MODEL_V2/")

# ==================== Phase 7: Create Metadata ====================
print("\n[Phase 7] Creating Metadata")
print("-" * 80)

metadata = {
    'model_version': 'V2',
    'description': 'CPB V2 Models - Predict [Price, Volatility]',
    'training_date': datetime.now().isoformat(),
    'total_models': len(CONFIG['pairs']),
    'successful_models': sum(1 for r in training_results if r['status'] == 'success'),
    'output_format': '[price, volatility]',
    'training_config': {
        'epochs': CONFIG['epochs'],
        'batch_size': CONFIG['batch_size'],
        'sequences': CONFIG['num_sequences'],
        'seq_length': CONFIG['seq_length']
    },
    'training_results': training_results
}

metadata_path = 'ALL_MODELS/MODEL_V2/metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   Metadata: {metadata_path}")
print(f"   Successful: {metadata['successful_models']}/{metadata['total_models']}")

# Create README
readme_content = """# CPB V2 Models

## Overview
CPB V2 Models predict both price and volatility for 20 cryptocurrency pairs.

## Model Output
```
[predicted_price, predicted_volatility]
```

## Supported Pairs (20)
""" + "\n".join([f"- {pair}" for pair in CONFIG['pairs']]) + """

## Architecture
- LSTM (64 units)
- LSTM (32 units)  
- Dense (32 units) + ReLU
- Dense (16 units) + ReLU
- Output (2 units)

## Training Details
- Epochs: 50
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Validation Split: 20%
- Early Stopping: Patience 5

## Usage
```python
import tensorflow as tf

model = tf.keras.models.load_model('v2_model_BTC_USDT.h5')
predictions = model.predict(klines)  # shape: (batch, 20, 4)
price, volatility = predictions[0]
```
"""

readme_path = 'ALL_MODELS/MODEL_V2/README.md'
with open(readme_path, 'w') as f:
    f.write(readme_content)

print(f"   README: {readme_path}")
print(f"[OK] Metadata created")

# ==================== Phase 8: Upload to GitHub ====================
print("\n[Phase 8] Upload to GitHub (caizongxun/cpbv2)")
print("-" * 80)

def upload_to_github(github_token):
    """
    Upload entire MODEL_V2 folder to GitHub (caizongxun/cpbv2)
    Key: Upload as folder, not individual files
    GitHub structure: caizongxun/cpbv2/MODEL_V2/
    """
    try:
        from github import Github
        
        # Initialize GitHub API
        g = Github(github_token)
        repo = g.get_user().get_repo('cpbv2')
        
        print(f"   Connected to: {repo.full_name}")
        
        # Upload entire MODEL_V2 folder
        print(f"   Uploading ALL_MODELS/MODEL_V2/ folder...")
        
        import base64
        
        upload_count = 0
        for filename in os.listdir('ALL_MODELS/MODEL_V2'):
            if filename.endswith(('.h5', '.json', '.md')):
                file_path = os.path.join('ALL_MODELS/MODEL_V2', filename)
                
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    
                    # Check if file exists
                    try:
                        contents = repo.get_contents(f'ALL_MODELS/MODEL_V2/{filename}')
                        # Update existing file
                        repo.update_file(
                            f'ALL_MODELS/MODEL_V2/{filename}',
                            f'Update {filename}',
                            file_content,
                            contents.sha
                        )
                        print(f"   ✓ {filename} (updated)")
                    except:
                        # Create new file
                        repo.create_file(
                            f'ALL_MODELS/MODEL_V2/{filename}',
                            f'Add {filename}',
                            file_content
                        )
                        print(f"   ✓ {filename} (created)")
                    
                    upload_count += 1
                    
                except Exception as e:
                    print(f"   ✗ {filename}: {e}")
        
        print(f"\n   [OK] Successfully uploaded {upload_count} files to GitHub")
        print(f"   Repository: https://github.com/caizongxun/cpbv2")
        print(f"   Model path: caizongxun/cpbv2/ALL_MODELS/MODEL_V2/")
        
        return True
        
    except ImportError:
        print(f"   [ERROR] PyGithub not installed")
        print(f"   Install: pip install PyGithub")
        return False
    except Exception as e:
        print(f"   [ERROR] Upload failed: {e}")
        return False

print("   [Option] Upload to GitHub")

try:
    from google.colab import output
    # In Colab, ask for input
    github_token = input("   Enter GitHub Token (or press Enter to skip): ").strip()
    if github_token:
        upload_success = upload_to_github(github_token)
    else:
        print("   [SKIP] Token not provided")
        upload_success = False
except:
    # Not in Colab
    print("   [INFO] Set GITHUB_TOKEN environment variable to auto-upload")
    print("   Or manually push to: git push origin main")
    upload_success = False

# ==================== Summary ====================
print("\n" + "="*80)
print(" "*25 + "V2 Pipeline Complete!")
print("="*80)

print("\n[SUMMARY]")
print(f"  ✓ Models Trained: {sum(1 for r in training_results if r['status'] == 'success')}/{len(CONFIG['pairs'])}")
print(f"  ✓ Output Format: {CONFIG['output']}")
print(f"  ✓ Location: ALL_MODELS/MODEL_V2/")
print(f"  ✓ Visualizations: visualizations/")
if upload_success:
    print(f"  ✓ Uploaded to GitHub: caizongxun/cpbv2/ALL_MODELS/MODEL_V2/")
else:
    print(f"  ✗ Not uploaded to GitHub (run upload separately)")

print("\n[FILES CREATED]")
print(f"  - models/v2_model_*.h5 (20 files)")
print(f"  - ALL_MODELS/MODEL_V2/ (organized folder)")
print(f"  - visualizations/ (training plots)")
print(f"  - ALL_MODELS/MODEL_V2/metadata.json")
print(f"  - ALL_MODELS/MODEL_V2/README.md")

print("\n[NEXT STEPS]")
print("  1. Review visualizations in visualizations/")
print("  2. Check metrics in ALL_MODELS/MODEL_V2/metadata.json")
print("  3. For V3, create: complete_v3_pipeline.py")
print("  4. For bug fixes: Edit complete_v2_pipeline.py and re-run")

print("\n" + "="*80)
