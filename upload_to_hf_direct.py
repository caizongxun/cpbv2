#!/usr/bin/env python3
"""
Upload CPB v4 Models to zongowo111/cpb-models in model_v4/ folder
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import json

# ========== CONFIGURATION ==========

# Get HF token
HF_TOKEN = os.getenv('HF_TOKEN', None)
if not HF_TOKEN:
    print("="*60)
    print("Hugging Face Token Setup")
    print("="*60)
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Set to 'Write' permission")
    print("4. Copy the token\n")
    HF_TOKEN = input("Paste your HF token: ").strip()
    if not HF_TOKEN:
        print("Error: Token required")
        sys.exit(1)

# Target repo
TARGET_REPO = "zongowo111/cpb-models"
MODEL_FOLDER = "model_v4"

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

print(f"Models directory: {MODELS_DIR}")
print(f"Target repo: {TARGET_REPO}")
print(f"Target folder: {MODEL_FOLDER}/")
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


def create_readme():
    """Create README for model_v4 folder"""
    readme = """# CPB Transformer Models v4

## Overview

Trained Transformer-based models for cryptocurrency price prediction.

## Model Specifications

- **Architecture**: Transformer (Encoder-Decoder with Attention)
- **Input**: 30-step lookback window
- **Output**: 10-step ahead forecast  
- **Features**: [Open, High, Low, Close]
- **Framework**: PyTorch

## Training Configuration

- **Data Source**: Binance Klines (1 year historical data)
- **Normalization**: Min-Max scaling
- **Batch Size**: 128
- **Optimizer**: AdamW (lr=0.001)
- **Loss**: MSE
- **Scheduler**: CosineAnnealingLR

## Available Models

### Timeframes
- 15m (15-minute candles)
- 1h (1-hour candles)

### Coins (20 total)
- BTC/ETH/BNB/XRP/LTC
- ADA/SOL/DOGE/AVAX/LINK
- UNI/ATOM/NEAR/DYDX/ARB
- OP/PEPE/INJ/SHIB/LUNA

**Total Models**: 40 (20 coins × 2 timeframes)

## Usage Example

```python
import torch

# Load model
model = torch.load('BTCUSDT_15m.pt')
model.eval()

# Prepare input (batch_size=1, seq_len=30, features=4)
input_data = torch.randn(1, 30, 4)

# Get prediction (output: 1, 10, 4)
with torch.no_grad():
    forecast = model(input_data)
```

## Performance Notes

Validation MSE varies by:
- Coin volatility
- Timeframe (longer timeframes tend to have lower loss)
- Historical data length

For detailed metrics, see training logs.

## Files

- `BTCUSDT_15m.pt` - Bitcoin 15-minute model
- `BTCUSDT_1h.pt` - Bitcoin 1-hour model
- `ETHUSDT_15m.pt` - Ethereum 15-minute model
- ... and 37 more models

## Created

Trained with CPB v4 Transformer Training Pipeline
"""
    return readme


# ========== UPLOAD FUNCTIONS ==========

def upload_models_to_hf(api, target_repo, model_folder, models_dir):
    """Upload all models to HF repo in specified folder"""
    
    model_files = get_model_files(models_dir)
    
    if not model_files:
        print(f"Error: No .pt files found in {models_dir}")
        return False
    
    print(f"Found {len(model_files)} models to upload\n")
    
    # Create repo if doesn't exist (with continue if exists)
    print(f"Ensuring repo exists: {target_repo}")
    try:
        create_repo(
            repo_id=target_repo,
            private=False,
            exist_ok=True
        )
        print(f"Repo ready\n")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # Upload README to model folder
    print(f"Uploading README to {model_folder}/README.md...")
    try:
        readme_content = create_readme()
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo=f"{model_folder}/README.md",
            repo_id=target_repo,
            commit_message=f"Add {model_folder} README"
        )
        print(f"README uploaded\n")
    except Exception as e:
        print(f"Warning: Could not upload README - {e}\n")
    
    # Upload all models
    print(f"Uploading {len(model_files)} models to {model_folder}/...\n")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    for idx, model_file in enumerate(model_files, 1):
        model_name = model_file.name
        print(f"[{idx:2d}/{len(model_files):2d}] {model_name:25s}", end=' ')
        sys.stdout.flush()
        
        try:
            # Upload to model_v4 folder
            api.upload_file(
                path_or_fileobj=str(model_file),
                path_in_repo=f"{model_folder}/{model_name}",
                repo_id=target_repo,
                commit_message=f"Add {model_name}"
            )
            print("✓ OK")
            successful += 1
        except Exception as e:
            error_msg = str(e)[:40]
            print(f"✗ FAIL ({error_msg}...)")
            failed += 1
    
    print("-" * 60)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"UPLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Repository: {target_repo}")
    print(f"Folder: {model_folder}/")
    print(f"Successful: {successful}/{len(model_files)}")
    print(f"Failed: {failed}")
    print(f"\nAccess URL:")
    print(f"https://huggingface.co/{target_repo}/tree/main/{model_folder}")
    print(f"{'='*60}\n")
    
    return successful == len(model_files)


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
    
    print(f"Found {model_count} models\n")
    
    # Upload to HF
    success = upload_models_to_hf(api, TARGET_REPO, MODEL_FOLDER, str(models_dir))
    
    if success:
        print("✓ All models uploaded successfully!")
    else:
        print("⚠ Some models failed to upload. Check the logs above.")
        sys.exit(1)
