#!/usr/bin/env python3
"""
CPB Colab Quick Start

Choose your training version:

1. V4 (RECOMMENDED for Colab Free) - 2.5 hours
   - 10 coins × 2 timeframes = 20 models
   - Fits Colab free tier (3 hour 20 min limit)
   - OHLCV features only
   - Faster convergence

2. V5 (for Colab Pro) - 2-2.5 hours
   - 20 coins × 2 timeframes = 40 models
   - Requires Colab Pro or A100
   - Advanced features (RSI, MACD, Bollinger Bands, etc.)
   - Better accuracy

Quick start:

    # In Colab cell:
    import requests
    exec(requests.get('https://raw.githubusercontent.com/caizongxun/cpbv2/main/COLAB_QUICK_START.py').text)
"""

import subprocess
import sys
import os
from datetime import datetime

print("\n" + "="*70)
print("CPB Colab Quick Start")
print("="*70)
print(f"Time: {datetime.now()}")
print()

# Check GPU
print("[CHECK] Detecting GPU...")
import torch
if not torch.cuda.is_available():
    print("ERROR: No GPU detected!")
    print("Go to: Runtime > Change runtime type > GPU")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"  GPU: {gpu_name}")
print(f"  Memory: {mem_gb:.1f}GB")
print(f"  CUDA: {torch.version.cuda}")

# Determine version
print()
print("[SELECT] Which version do you want?")
print()
print("  Option 1: V4 Optimized (RECOMMENDED)")
print("    - Fastest: 20 models in 2.5 hours")
print("    - Fits Colab free tier (3h 20m limit)")
print("    - Works on T4, L4, A100")
print()
print("  Option 2: V5 Full (Better accuracy, needs Colab Pro)")
print("    - 40 models, advanced features")
print("    - Requires Colab Pro or high-tier GPU")
print()

# Auto-select based on GPU and time
if 'T4' in gpu_name or 'L4' in gpu_name:
    print("[AUTO] Detected T4/L4 GPU -> Using V4 Optimized")
    selected = 'v4'
elif 'A100' in gpu_name:
    print("[AUTO] Detected A100 GPU -> Using V5 Full")
    selected = 'v5'
else:
    print(f"[AUTO] GPU {gpu_name} -> Using V4 Optimized (safe choice)")
    selected = 'v4'

print()
print("="*70)

# Install dependencies
print("[SETUP] Installing dependencies...")
packages = [
    'torch',
    'pandas',
    'numpy',
    'scikit-learn',
    'requests'
]

for pkg in packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)

print(f"  Dependencies installed!")

# Download and run training
print()
print("[DOWNLOAD] Fetching training script...")

if selected == 'v4':
    url = 'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_train_colab_optimized.py'
    version_name = 'V4 Optimized'
    expected_time = '2.5 hours'
    expected_models = '20 models (10 coins × 2 timeframes)'
else:
    url = 'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete_fixed.py'
    version_name = 'V5 Full'
    expected_time = '2-2.5 hours'
    expected_models = '40 models (20 coins × 2 timeframes)'

import requests

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    training_code = response.text
    print(f"  Downloaded {len(training_code)} characters from {version_name}")
except Exception as e:
    print(f"ERROR: Could not download training script: {e}")
    sys.exit(1)

print()
print("="*70)
print(f"TRAINING: {version_name}")
print("="*70)
print(f"Expected time: {expected_time}")
print(f"Models to train: {expected_models}")
print()
print("IMPORTANT:")
print("  - Keep this tab open (or use Colab Pro for background)")
print("  - Do not close the browser")
print("  - Training will start below...")
print()
print("="*70)
print()

sys.stdout.flush()

# Execute training
try:
    exec(training_code, {'__name__': '__main__'})
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("Training Complete")
print("="*70)
print(f"Results saved to: /content/{selected}_models*/")
print()
print("Next steps:")
print("  1. Check results in the output above")
print("  2. Download models if needed")
print("  3. Or upload to Hugging Face:")
if selected == 'v4':
    print("     huggingface-cli upload user/model-name /content/v4_models_optimized model_v4")
else:
    print("     huggingface-cli upload user/model-name /content/all_models/model_v5 model_v5")
print("="*70)
