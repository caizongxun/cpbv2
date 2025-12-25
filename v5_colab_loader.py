#!/usr/bin/env python3
"""
CPB v5 Colab Remote Loader - FIXED VERSION

在 Google Colab 中執行:

    import requests
    exec(requests.get('https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_colab_loader.py').text)

或直接在 cell 中:

    !curl -s https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_colab_loader.py | python
"""

import os
import sys
import json
import time
import requests
from datetime import datetime

print("="*60)
print("CPB v5: Cryptocurrency Price Prediction Training")
print("="*60)
print(f"Start time: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print("="*60)

# ============================================================================
# STEP 1: 環境設置
# ============================================================================

print("\n[STEP 1/5] Installing dependencies...")

import subprocess

packages = [
    'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
    'pandas numpy scikit-learn requests huggingface-hub',
]

for package_spec in packages:
    print(f"  Installing: {package_spec}")
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q'] + package_spec.split(),
        check=False
    )

print("  Dependencies installed successfully!")

# ============================================================================
# STEP 2: 克隆倉庫
# ============================================================================

print("\n[STEP 2/5] Cloning repository...")

repo_path = '/content/cpbv2'
if not os.path.exists(repo_path):
    subprocess.run(
        ['git', 'clone', 'https://github.com/caizongxun/cpbv2.git', repo_path],
        check=False
    )
    print(f"  Cloned to {repo_path}")
else:
    print(f"  Repository already exists at {repo_path}")

os.chdir(repo_path)
print(f"  Working directory: {os.getcwd()}")

# ============================================================================
# STEP 3: 下載修正的訓練腳本
# ============================================================================

print("\n[STEP 3/5] Loading training script (FIXED VERSION)...")

try:
    # 优先使用修正版本
    training_script_url = 'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete_fixed.py'
    print(f"  Fetching from: {training_script_url}")
    
    response = requests.get(training_script_url, timeout=30)
    response.raise_for_status()
    
    training_code = response.text
    print(f"  Downloaded {len(training_code)} characters")
    print("  Script loaded successfully!")
    
except Exception as e:
    print(f"  Warning: Could not fetch fixed version: {e}")
    print("  Falling back to v5_training_complete.py")
    
    try:
        training_script_url = 'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py'
        response = requests.get(training_script_url, timeout=30)
        response.raise_for_status()
        training_code = response.text
        print(f"  Downloaded {len(training_code)} characters")
    except Exception as e2:
        print(f"  Error: Could not fetch any training script: {e2}")
        sys.exit(1)

# ============================================================================
# STEP 4: 執行訓練
# ============================================================================

print("\n[STEP 4/5] Executing training script...")
print("  This may take 2-2.5 hours...")
print("  Do not close the browser tab!")
print("-" * 60)

try:
    # 在當前命名空間中執行訓練代碼
    exec(training_code, {'__name__': '__main__'})
    print("-" * 60)
    print("\n[STEP 5/5] Training completed!")
    training_success = True
    
except KeyboardInterrupt:
    print("\n  Training interrupted by user!")
    training_success = False
    
except Exception as e:
    print(f"\n  Error during training: {e}")
    import traceback
    traceback.print_exc()
    training_success = False

# ============================================================================
# 結果總結
# ============================================================================

print("\n" + "="*60)
if training_success:
    print("TRAINING COMPLETED SUCCESSFULLY")
    
    # 嘗試讀取結果
    results_file = '/content/all_models/model_v5/training_results.json'
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                results = json.load(f)
            
            if len(results) > 0:
                mape_values = [v['mape'] for v in results.values() if 'mape' in v]
                if mape_values:
                    print(f"\nResults Summary:")
                    print(f"  Total models trained: {len(results)}")
                    print(f"  Average MAPE: {sum(mape_values)/len(mape_values):.6f}")
                    print(f"  Best MAPE: {min(mape_values):.6f}")
                    print(f"  Worst MAPE: {max(mape_values):.6f}")
                    print(f"  Models below 0.02: {sum(1 for m in mape_values if m < 0.02)} / {len(mape_values)}")
                else:
                    print(f"  Warning: No MAPE values found in results")
            else:
                print(f"  Warning: No models were successfully trained")
            
            print(f"\nModels saved to: /content/all_models/model_v5/")
            print(f"Results file: {results_file}")
        except Exception as e:
            print(f"  Error reading results: {e}")
    else:
        print(f"  Results file not found at {results_file}")
    
    print(f"\nNext step: Upload to Hugging Face")
    print(f"  huggingface-cli upload zongowo111/cpb-models \\")
    print(f"    /content/all_models/model_v5 model_v5 --repo-type model")
else:
    print("TRAINING FAILED OR INTERRUPTED")
    print(f"Check error messages above for details")

print(f"\nEnd time: {datetime.now()}")
print("="*60)
