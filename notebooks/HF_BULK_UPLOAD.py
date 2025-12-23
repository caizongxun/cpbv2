#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: Bulk Upload All Models at Once
一次上傳整個 all_models 資料夾
"""

print("\n" + "="*90)
print("BULK UPLOAD ALL MODELS")
print("="*90)

import os
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

print("\n[SETUP]")
print("="*90)

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_USERNAME = os.environ.get('HF_USERNAME')

if not HF_TOKEN or not HF_USERNAME:
    print("[ERROR] Token or username not set")
    exit(1)

print(f"Token: {HF_TOKEN[:20]}...")
print(f"Username: {HF_USERNAME}")

REPO_ID = f"{HF_USERNAME}/cpbmodel"
print(f"\nRepo: {REPO_ID}")
print(f"URL: https://huggingface.co/{REPO_ID}")

print("\n[STEP 1] Finding all_models folder")
print("="*90)

# 尋找 all_models 資料夾
model_dir = Path('./hf_models')  # 或 './all_models'

if not model_dir.exists():
    # 嘗試 all_models
    model_dir = Path('./all_models')
    if not model_dir.exists():
        print(f"[ERROR] Neither 'hf_models' nor 'all_models' found")
        exit(1)

print(f"[OK] Found: {model_dir}")
print(f"[OK] Absolute path: {model_dir.absolute()}")

# 統計檔案
folders = [d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
file_count = sum(len(list(f.glob('*'))) for f in folders)

print(f"[OK] Found {len(folders)} coin folders")
print(f"[OK] Total files: {file_count}")

for folder in sorted(folders):
    files = list(folder.glob('*'))
    total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024*1024)
    print(f"     - {folder.name} ({len(files)} files, {total_size:.1f} MB)")

print("\n[STEP 2] Uploading entire folder")
print("="*90)

api = HfApi(token=HF_TOKEN)

try:
    print(f"[INFO] Starting bulk upload...")
    print(f"[INFO] This may take a few minutes...\n")
    
    # 一次上傳整個資料夾
    upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(model_dir),
        token=HF_TOKEN,
        commit_message="Bulk upload all CPB v1 coin models",
        ignore_patterns=["*.pyc", "__pycache__", ".git*"],
    )
    
    print(f"\n[SUCCESS] All models uploaded successfully!")
    print(f"\n" + "="*90)
    print("[SUMMARY]")
    print("="*90)
    
    print(f"\nRepo: https://huggingface.co/{REPO_ID}")
    print(f"\nUploaded {len(folders)} coin models:")
    for folder in sorted(folders):
        print(f"  - {folder.name}")
    
    print(f"\nAccess your models:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  ")
    print(f"  # Download a specific model")
    print(f"  model = hf_hub_download(")
    print(f"      repo_id='{REPO_ID}',")
    print(f"      filename='BTCUSDT_1h_v1/pytorch_model.bin'")
    print(f"  )")
    
    print(f"\n[NEXT] You can now manage your models on HuggingFace:")
    print(f"  1. Add model card (README)")
    print(f"  2. Add license")
    print(f"  3. Add tags and description")
    print(f"  4. Make it public/private")
    
except Exception as e:
    print(f"[ERROR] Upload failed")
    print(f"Error: {str(e)[:100]}")
    print(f"\n[TIP] Try again or use incremental upload")
    exit(1)

print("\n" + "="*90)
print("COMPLETE")
print("="*90 + "\n")
