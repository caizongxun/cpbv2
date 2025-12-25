#!/usr/bin/env python3
"""
Upload CPB v4 Transformer Models to Hugging Face
Supports: Single repo or per-coin repos
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file, upload_folder
import json
from datetime import datetime

# ========== CONFIGURATION ==========

# Get HF token from environment or user input
HF_TOKEN = os.getenv('HF_TOKEN', None)
if not HF_TOKEN:
    print("Hugging Face Token Setup")
    print("="*60)
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Set to 'Write' permission")
    print("4. Copy the token")
    print("="*60)
    HF_TOKEN = input("\nPaste your HF token: ").strip()
    if not HF_TOKEN:
        print("Error: Token required")
        sys.exit(1)

# Detect environment
if 'google.colab' in sys.modules:
    MODELS_DIR = '/content/v4_models'
    print("Environment: Google Colab")
elif os.path.exists('/kaggle'):
    MODELS_DIR = '/kaggle/working/v4_models'
    print("Environment: Kaggle Notebook")
else:
    MODELS_DIR = './v4_models'
    print("Environment: Local")

print(f"Models directory: {MODELS_DIR}\n")

# Upload strategy
print("Upload Strategy:")
print("1. Single repo - all models in one repo")
print("2. Per-coin repos - each coin gets separate repo")
strategy = input("Choose (1 or 2): ").strip()

if strategy not in ['1', '2']:
    print("Invalid choice")
    sys.exit(1)

# Repo type
print("\nRepository Type:")
print("1. Public (anyone can access)")
print("2. Private (only you can access)")
repo_type = input("Choose (1 or 2): ").strip()

if repo_type not in ['1', '2']:
    print("Invalid choice")
    sys.exit(1)

REPO_PRIVATE = (repo_type == '2')
REPO_TYPE = 'private' if REPO_PRIVATE else 'public'

print(f"\nConfiguration:")
print(f"Strategy: {'Single repo' if strategy == '1' else 'Per-coin'}")
print(f"Type: {REPO_TYPE}")
print(f"Token: ***{HF_TOKEN[-8:]}")
print("\n" + "="*60 + "\n")


# ========== HF API SETUP ==========

api = HfApi(token=HF_TOKEN)

try:
    user_info = api.whoami()
    USERNAME = user_info['name']
    print(f"Logged in as: {USERNAME}\n")
except Exception as e:
    print(f"Error: Invalid token - {e}")
    sys.exit(1)


# ========== HELPER FUNCTIONS ==========

def get_model_files(models_dir):
    """Get all .pt model files"""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        print(f"Error: Directory not found - {models_dir}")
        return []
    
    files = sorted(list(models_dir.glob("*.pt")))
    return files


def create_readme(coin, timeframe=None):
    """Create README.md for the repo"""
    
    if timeframe:
        title = f"CPB Transformer Model - {coin} {timeframe}"
    else:
        title = "CPB Transformer Models - Complete Collection"
    
    readme = f"""# {title}

## Overview

Trained transformer-based models for crypto price prediction using CPB (Crypto Price Bot) v4.

## Model Specifications

- **Architecture**: Transformer (Encoder-Decoder with Attention)
- **Input Size**: 30-step lookback window
- **Prediction**: 10-step ahead forecast
- **Input Features**: [Open, High, Low, Close]
- **Framework**: PyTorch

## Training Details

- **Data Source**: Binance Futures
- **Normalization**: Min-Max scaling
- **Batch Size**: 128
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Loss Function**: MSE
- **Trained**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

```python
import torch
from transformers import AutoModel

# Load model
model = torch.load('model.pt')
model.eval()

# Prepare input
input_data = torch.randn(1, 30, 4)  # (batch, seq_len, features)

# Predict
with torch.no_grad():
    output = model(input_data)  # (1, 10, 4) - next 10 steps
```

## File Structure

"""
    
    if timeframe:
        readme += f"- `model.pt` - Trained weights for {coin} {timeframe}\n"
    else:
        readme += """Models included:
- BTCUSDT_15m.pt
- BTCUSDT_1h.pt
- ETHUSDT_15m.pt
- ... (40 models total)
"""
    
    readme += f"""

## Performance

Validation MSE Loss varies by coin and timeframe.
See training logs for detailed metrics.

## License

MIT License

## Author

Created with CPB v4 Transformer Training Pipeline
"""
    
    return readme


def upload_single_repo(models_dir, api, username):
    """Upload all models to a single repo"""
    
    print("Strategy: Single Repository")
    print("="*60)
    
    repo_name = 'cpb-transformer-models-v4'
    repo_id = f"{username}/{repo_name}"
    
    print(f"Creating/accessing repo: {repo_id}")
    
    try:
        # Create repo if not exists
        create_repo(
            repo_id=repo_id,
            private=REPO_PRIVATE,
            exist_ok=True
        )
    except Exception as e:
        print(f"Error creating repo: {e}")
        return False
    
    # Upload README
    readme_path = Path(models_dir) / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(create_readme(None))
    
    print(f"Uploading README...")
    try:
        upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo='README.md',
            repo_id=repo_id,
            commit_message='Add README'
        )
    except Exception as e:
        print(f"Error uploading README: {e}")
    
    # Upload models
    model_files = get_model_files(models_dir)
    print(f"\nUploading {len(model_files)} models...\n")
    
    successful = 0
    failed = 0
    
    for idx, model_file in enumerate(model_files, 1):
        model_name = model_file.name
        print(f"[{idx}/{len(model_files)}] {model_name}...", end=' ')
        sys.stdout.flush()
        
        try:
            upload_file(
                path_or_fileobj=str(model_file),
                path_in_repo=model_name,
                repo_id=repo_id,
                commit_message=f'Add {model_name}'
            )
            print("OK")
            successful += 1
        except Exception as e:
            print(f"FAIL ({str(e)[:30]}...)")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Upload Summary")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(model_files)}")
    print(f"Failed: {failed}")
    print(f"Repo: https://huggingface.co/{repo_id}")
    print(f"{'='*60}\n")
    
    return successful == len(model_files)


def upload_per_coin_repos(models_dir, api, username):
    """Upload models to per-coin repos"""
    
    print("Strategy: Per-Coin Repositories")
    print("="*60)
    
    model_files = get_model_files(models_dir)
    
    # Group by coin
    coins = {}
    for model_file in model_files:
        # BTCUSDT_15m.pt -> coin: BTCUSDT, timeframe: 15m
        parts = model_file.stem.split('_')
        coin = parts[0]
        timeframe = parts[1] if len(parts) > 1 else 'all'
        
        if coin not in coins:
            coins[coin] = []
        coins[coin].append((model_file, timeframe))
    
    print(f"Found {len(coins)} coins\n")
    
    total_uploaded = 0
    total_failed = 0
    
    for coin in sorted(coins.keys()):
        print(f"Coin: {coin}")
        repo_name = f'cpb-{coin.lower()}-v4'
        repo_id = f"{username}/{repo_name}"
        
        try:
            create_repo(
                repo_id=repo_id,
                private=REPO_PRIVATE,
                exist_ok=True
            )
        except Exception as e:
            print(f"  Error creating repo: {e}")
            continue
        
        # Upload README
        try:
            readme = create_readme(coin)
            readme_path = Path('/tmp') / f'README_{coin}.md'
            with open(readme_path, 'w') as f:
                f.write(readme)
            
            upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo='README.md',
                repo_id=repo_id,
                commit_message='Add README'
            )
        except Exception as e:
            print(f"  Error uploading README: {e}")
        
        # Upload models for this coin
        for model_file, timeframe in coins[coin]:
            model_name = model_file.name
            print(f"  [{timeframe}] {model_name}...", end=' ')
            sys.stdout.flush()
            
            try:
                upload_file(
                    path_or_fileobj=str(model_file),
                    path_in_repo=model_name,
                    repo_id=repo_id,
                    commit_message=f'Add {model_name}'
                )
                print("OK")
                total_uploaded += 1
            except Exception as e:
                print(f"FAIL")
                total_failed += 1
        
        print(f"  Repo: https://huggingface.co/{repo_id}\n")
    
    print(f"{'='*60}")
    print(f"Upload Summary")
    print(f"{'='*60}")
    print(f"Total repositories: {len(coins)}")
    print(f"Total uploaded: {total_uploaded}")
    print(f"Total failed: {total_failed}")
    print(f"{'='*60}\n")


# ========== MAIN ==========

if __name__ == "__main__":
    models_dir = Path(MODELS_DIR)
    
    if not models_dir.exists():
        print(f"Error: Models directory not found - {models_dir}")
        sys.exit(1)
    
    model_count = len(list(models_dir.glob("*.pt")))
    if model_count == 0:
        print("Error: No .pt files found in models directory")
        sys.exit(1)
    
    print(f"Found {model_count} models in {models_dir}\n")
    
    if strategy == '1':
        upload_single_repo(str(models_dir), api, USERNAME)
    else:
        upload_per_coin_repos(str(models_dir), api, USERNAME)
    
    print("Done!")
