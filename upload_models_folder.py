#!/usr/bin/env python3
"""
Upload entire v4_models folder to HF in single call
Avoids API rate limiting by uploading folder instead of individual files
Auto-creates repo if it doesn't exist
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import time

# ========== SETUP ==========

print("="*60)
print("Upload CPB v4 Models Folder to HuggingFace")
print("="*60 + "\n")

# Get token
token = os.getenv('HF_TOKEN', None)
if not token:
    token = input("Enter HF Token: ").strip()
    if not token:
        print("Error: Token required")
        sys.exit(1)

print(f"Token: ***{token[-8:]}")

# Environment
if 'google.colab' in sys.modules:
    MODELS_DIR = '/content/v4_models'
    print("Environment: Colab")
elif os.path.exists('/kaggle'):
    MODELS_DIR = '/kaggle/working/v4_models'
    print("Environment: Kaggle")
else:
    MODELS_DIR = './v4_models'
    print("Environment: Local")

print(f"Models folder: {MODELS_DIR}\n")

# ========== CHECK MODELS ==========

models_dir = Path(MODELS_DIR)
if not models_dir.exists():
    print(f"Error: Directory not found - {MODELS_DIR}")
    sys.exit(1)

model_files = sorted(list(models_dir.glob("*.pt")))
if not model_files:
    print("Error: No .pt files found")
    sys.exit(1)

total_size = sum(f.stat().st_size for f in model_files) / 1e9  # GB
print(f"Found {len(model_files)} models")
print(f"Total size: {total_size:.2f} GB\n")

# ========== UPLOAD ==========

REPO_ID = "zongowo111/cpb-models"
FOLDER_PATH = "model_v4"
api = HfApi(token=token)

print(f"Target repo: {REPO_ID}")
print(f"Target folder: {FOLDER_PATH}/\n")

try:
    user = api.whoami()
    print(f"Logged in as: {user['name']}\n")
except Exception as e:
    print(f"Error: Invalid token - {e}")
    sys.exit(1)

# Create repo if doesn't exist
print("Checking repository...")
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token
    )
    print(f"Repository ready: {REPO_ID}\n")
except Exception as e:
    print(f"Error creating repo: {e}")
    sys.exit(1)

print("="*60)
print("Uploading entire folder...")
print("(This may take a few minutes)")
print("="*60 + "\n")

start_time = time.time()

try:
    # Upload entire folder in one call
    result = api.upload_folder(
        folder_path=str(models_dir),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=FOLDER_PATH,
        commit_message="Add CPB v4 Transformer Models",
        token=token
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*60)
    print(f"UPLOAD SUCCESSFUL")
    print(f"="*60)
    print(f"Uploaded: {len(model_files)} models")
    print(f"Size: {total_size:.2f} GB")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"\nRepo: https://huggingface.co/datasets/{REPO_ID}")
    print(f"Files: https://huggingface.co/datasets/{REPO_ID}/tree/main/{FOLDER_PATH}")
    print("="*60 + "\n")
    print("✓ All models uploaded!")

except Exception as e:
    print(f"\nError during upload: {e}")
    print(f"\nTroubleshooting:")
    print(f"1. Check token is still valid")
    print(f"2. Ensure repo exists at https://huggingface.co/datasets/{REPO_ID}")
    print(f"3. Check network connection")
    sys.exit(1)
