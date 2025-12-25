#!/usr/bin/env python3
"""
Simple direct upload to HF using huggingface_hub
No git clone needed
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, CommitScheduler
import time

# ========== SETUP ==========

print("="*60)
print("Upload CPB v4 Models to HuggingFace")
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

print(f"Models: {MODELS_DIR}\n")

# ========== CHECK MODELS ==========

models_dir = Path(MODELS_DIR)
if not models_dir.exists():
    print(f"Error: Directory not found - {MODELS_DIR}")
    sys.exit(1)

model_files = sorted(list(models_dir.glob("*.pt")))
if not model_files:
    print("Error: No .pt files found")
    sys.exit(1)

print(f"Found {len(model_files)} models\n")

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

print("="*60)
print("Uploading...")
print("="*60 + "\n")

success_count = 0
fail_count = 0

for idx, model_file in enumerate(model_files, 1):
    filename = model_file.name
    print(f"[{idx:2d}/{len(model_files)}] {filename:25s}", end=' ', flush=True)
    
    try:
        # Upload with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                api.upload_file(
                    path_or_fileobj=str(model_file),
                    path_in_repo=f"{FOLDER_PATH}/{filename}",
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Add {filename}"
                )
                print("✓")
                success_count += 1
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n    Retrying... ({attempt+1}/{max_retries})", end='')
                    time.sleep(2)  # Wait before retry
                else:
                    raise e
    except Exception as e:
        error_msg = str(e)[:50]
        print(f"✗ {error_msg}")
        fail_count += 1

print(f"\n" + "="*60)
print(f"RESULT")
print(f"="*60)
print(f"Uploaded: {success_count}/{len(model_files)}")
print(f"Failed: {fail_count}")
print(f"\nRepo: https://huggingface.co/datasets/{REPO_ID}")
print(f"Files: https://huggingface.co/datasets/{REPO_ID}/tree/main/{FOLDER_PATH}")
print("="*60 + "\n")

if fail_count == 0:
    print("✓ All models uploaded!")
else:
    print(f"⚠ {fail_count} models failed")
