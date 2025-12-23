"""
Complete V2 Pipeline: Train -> Organize -> Upload to Hugging Face

One-command execution for entire V2 model training and deployment pipeline
"""

import os
import sys
import urllib.request
import shutil
from pathlib import Path

print("="*80)
print(" "*20 + "CPB V2 Complete Pipeline")
print(" "*15 + "Train -> Organize -> Upload to HuggingFace")
print("="*80)

# ==================== Phase 1: Download & Train ====================
print("\n[PHASE 1] Downloading V2 Training Script...")

try:
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/caizongxun/cpbv2/main/notebooks/FINAL_PRODUCTION_V2_PER_PAIR.py',
        'train_v2_per_pair.py'
    )
    print("[OK] Training script downloaded\n")
    
    print("[PHASE 2] Training V2 Models (20 pairs)...")
    print("-" * 80)
    exec(open('train_v2_per_pair.py').read())
    print("-" * 80)
    print("[OK] Training complete\n")
    
except Exception as e:
    print(f"[ERROR] Training failed: {e}")
    sys.exit(1)

# ==================== Phase 2: Organize & Upload ====================
print("\n" + "="*80)
print("[PHASE 3] Downloading Organization & Upload Script...")

try:
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/caizongxun/cpbv2/main/notebooks/ORGANIZE_AND_UPLOAD_V2_MODELS.py',
        'organize_and_upload_v2.py'
    )
    print("[OK] Organization script downloaded\n")
    
    print("[PHASE 4] Organizing & Uploading V2 Models...")
    print("-" * 80)
    exec(open('organize_and_upload_v2.py').read())
    print("-" * 80)
    print("[OK] Organization and upload complete\n")
    
except Exception as e:
    print(f"[ERROR] Organization/Upload failed: {e}")
    print("[WARNING] Models are still available in ./models/ directory")
    sys.exit(1)

print("\n" + "="*80)
print(" "*25 + "V2 Pipeline Complete!")
print("="*80)
print("\n[SUMMARY]")
print("  ✓ V2 Models Trained: 20 pairs")
print("  ✓ Output Format: [price, volatility]")
print("  ✓ Location: ALL_MODELS/v2/")
print("  ✓ Uploaded to: Hugging Face Hub")
print("\n" + "="*80)
