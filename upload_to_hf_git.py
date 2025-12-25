#!/usr/bin/env python3
"""
Upload CPB v4 Models to zongowo111/cpb-models using Git LFS
More reliable than API
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ========== CONFIGURATION ==========

# Detect environment
if 'google.colab' in sys.modules:
    MODELS_DIR = '/content/v4_models'
    WORK_DIR = '/content'
    print("Environment: Google Colab")
elif os.path.exists('/kaggle'):
    MODELS_DIR = '/kaggle/working/v4_models'
    WORK_DIR = '/kaggle/working'
    print("Environment: Kaggle Notebook")
else:
    MODELS_DIR = './v4_models'
    WORK_DIR = '.'
    print("Environment: Local")

print(f"Models directory: {MODELS_DIR}")
print(f"Work directory: {WORK_DIR}\n")

# Target
TARGET_REPO = "zongowo111/cpb-models"
MODEL_FOLDER = "model_v4"
REPO_URL = f"https://huggingface.co/datasets/{TARGET_REPO}"
CLONE_DIR = Path(WORK_DIR) / "hf_repo"

print(f"Target repo: {TARGET_REPO}")
print(f"Target folder: {MODEL_FOLDER}/")
print(f"Clone to: {CLONE_DIR}\n")

# ========== HELPER FUNCTIONS ==========

def run_cmd(cmd, cwd=None):
    """Run shell command"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def setup_git_credentials(username, token):
    """Setup git credentials for HF"""
    print("Setting up Git credentials...")
    
    # Configure git
    run_cmd('git config --global user.email "bot@huggingface.co"')
    run_cmd('git config --global user.name "HF Bot"')
    
    # Setup HF token
    os.environ['HUGGINGFACE_HUB_TOKEN'] = token
    
    print("Git credentials configured\n")


def clone_repo(token):
    """Clone HF repo with LFS support"""
    print(f"Cloning {TARGET_REPO}...")
    
    if CLONE_DIR.exists():
        print(f"Removing existing clone...")
        os.system(f"rm -rf {CLONE_DIR}")
    
    # Create clone directory
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Add token to URL
    repo_url_with_token = f"https://{token}@huggingface.co/datasets/{TARGET_REPO}.git"
    
    # Clone with LFS
    success, stdout, stderr = run_cmd(
        f"git clone --depth 1 {repo_url_with_token} .",
        cwd=CLONE_DIR
    )
    
    if not success:
        # Try without LFS first time
        print("Trying standard clone...")
        success, stdout, stderr = run_cmd(
            f"git clone {repo_url_with_token} .",
            cwd=CLONE_DIR
        )
    
    if success:
        print("✓ Repository cloned\n")
        return True
    else:
        print(f"✗ Clone failed: {stderr}")
        return False


def setup_lfs():
    """Setup Git LFS"""
    print("Setting up Git LFS...")
    
    success, stdout, stderr = run_cmd("git lfs install", cwd=CLONE_DIR)
    
    if success:
        print("✓ Git LFS installed\n")
    else:
        print(f"⚠ Git LFS setup warning: {stderr}")
        print("Continuing without LFS optimization\n")


def create_readme():
    """Create README for model folder"""
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

- **Data Source**: Binance Klines (historical data)
- **Normalization**: Min-Max scaling
- **Batch Size**: 128
- **Optimizer**: AdamW (lr=0.001)
- **Loss**: MSE
- **Scheduler**: CosineAnnealingLR

## Available Models (40 total)

### Coins
- BTC / ETH / BNB / XRP / LTC
- ADA / SOL / DOGE / AVAX / LINK
- UNI / ATOM / NEAR / DYDX / ARB
- OP / PEPE / INJ / SHIB / LUNA

### Timeframes
- 15m (15-minute candles)
- 1h (1-hour candles)

## File Format

Each `.pt` file is a PyTorch model state dict.

## Usage

```python
import torch

# Load model
model = torch.load('BTCUSDT_15m.pt')
model.eval()

# Prepare input
input_data = torch.randn(1, 30, 4)  # (batch, seq_len, features)

# Predict
with torch.no_grad():
    output = model(input_data)  # (1, 10, 4)
```

## Training Date

""" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """

## Author

Created with CPB v4 Transformer Training Pipeline
"""
    return readme


def copy_models():
    """Copy models to repo"""
    print(f"Copying models to {MODEL_FOLDER}/...")
    
    models_dir = Path(MODELS_DIR)
    target_dir = CLONE_DIR / MODEL_FOLDER
    
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        return False
    
    # Create folder
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy models
    model_files = sorted(list(models_dir.glob("*.pt")))
    
    if not model_files:
        print(f"✗ No .pt files found")
        return False
    
    print(f"Found {len(model_files)} models\n")
    
    for idx, model_file in enumerate(model_files, 1):
        target_file = target_dir / model_file.name
        print(f"[{idx:2d}/{len(model_files):2d}] {model_file.name:25s}", end=' ')
        sys.stdout.flush()
        
        try:
            # Copy file
            os.system(f"cp {model_file} {target_file}")
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            return False
    
    # Copy README
    print(f"\nCreating README...")
    readme_path = target_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(create_readme())
    print("✓ README created\n")
    
    return True


def push_to_repo():
    """Push to HF repo"""
    print("Pushing to HF...")
    print("="*60)
    
    # Add files
    print("Adding files...")
    success, stdout, stderr = run_cmd("git add -A", cwd=CLONE_DIR)
    
    if not success:
        print(f"✗ Failed to add files: {stderr}")
        return False
    
    # Check if there are changes
    success, stdout, stderr = run_cmd(
        'git diff-index --quiet HEAD --',
        cwd=CLONE_DIR
    )
    
    if success:  # No changes
        print("No changes to commit")
        return True
    
    # Commit
    print("Committing...")
    commit_msg = f"Add CPB v4 Transformer models ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
    success, stdout, stderr = run_cmd(
        f'git commit -m "{commit_msg}"',
        cwd=CLONE_DIR
    )
    
    if not success:
        print(f"✗ Commit failed: {stderr}")
        return False
    
    print("✓ Committed")
    
    # Push
    print("Pushing to server...")
    success, stdout, stderr = run_cmd("git push", cwd=CLONE_DIR)
    
    if success:
        print("✓ Pushed\n")
        return True
    else:
        print(f"✗ Push failed: {stderr}\n")
        return False


# ========== MAIN ==========

if __name__ == "__main__":
    print("="*60)
    print("HF Upload via Git")
    print("="*60 + "\n")
    
    # Get credentials
    token = os.getenv('HF_TOKEN', None)
    if not token:
        print("Option 1: Set HF_TOKEN environment variable")
        print("Option 2: Enter token now\n")
        token = input("Enter HF token (or leave blank for env var): ").strip()
        
        if not token:
            token = os.getenv('HF_TOKEN', None)
            if not token:
                print("Error: HF_TOKEN not found")
                sys.exit(1)
    
    username = "zongowo111"
    print(f"Using token: ***{token[-8:]}")
    print(f"Username: {username}\n")
    
    # Setup git
    setup_git_credentials(username, token)
    
    # Clone repo
    if not clone_repo(token):
        print("Failed to clone repo")
        sys.exit(1)
    
    # Setup LFS
    setup_lfs()
    
    # Copy models
    if not copy_models():
        print("Failed to copy models")
        sys.exit(1)
    
    # Push
    if not push_to_repo():
        print("Failed to push")
        sys.exit(1)
    
    # Success
    print("="*60)
    print("✓ UPLOAD SUCCESSFUL")
    print("="*60)
    print(f"\nRepository: https://huggingface.co/datasets/{TARGET_REPO}")
    print(f"Models folder: {MODEL_FOLDER}/")
    print(f"\nAccess URL:")
    print(f"https://huggingface.co/datasets/{TARGET_REPO}/tree/main/{MODEL_FOLDER}")
    print("\n")
