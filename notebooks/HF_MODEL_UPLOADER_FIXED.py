#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: Hugging Face Model Uploader - FIXED VERSION
修復登入檢查問題，直接上傳模型
"""

print("\n" + "="*90)
print("HUGGING FACE MODEL UPLOADER - FIXED VERSION")
print("="*90)

import os
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from huggingface_hub import HfApi, upload_folder
    print("[OK] huggingface_hub loaded")
except ImportError:
    print("[ERROR] huggingface_hub not installed")
    print("Install: pip install huggingface-hub")
    exit(1)

print("\n[SETUP] Getting token and username from environment...")

# 從環境變數取得 Token 和 Username
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_USERNAME = os.environ.get('HF_USERNAME')

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN not set!")
    print("Please set: os.environ['HF_TOKEN'] = 'hf_...'")
    exit(1)

if not HF_USERNAME:
    print("[ERROR] HF_USERNAME not set!")
    print("Please set: os.environ['HF_USERNAME'] = 'your_username'")
    exit(1)

print(f"[OK] Token: {HF_TOKEN[:20]}...")
print(f"[OK] Username: {HF_USERNAME}")

print("\n" + "="*90)
print("[STEP 1] Checking model folders")
print("="*90)

model_dir = Path('./hf_models')

if not model_dir.exists():
    print(f"[ERROR] Model directory not found: {model_dir}")
    print("Please run ORGANIZE_MODELS.py first")
    exit(1)

model_folders = [d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

if not model_folders:
    print(f"[ERROR] No model folders found in {model_dir}")
    exit(1)

print(f"[OK] Found {len(model_folders)} model folders:")
for folder in sorted(model_folders):
    print(f"     - {folder.name}")

print("\n" + "="*90)
print("[STEP 2] Preparing upload")
print("="*90)

repo_name = 'cpbmodel'
repo_id = f"{HF_USERNAME}/{repo_name}"

print(f"[INFO] Repo ID: {repo_id}")
print(f"[INFO] Repo URL: https://huggingface.co/{repo_id}")
print(f"[NOTE] First upload will create the repo automatically")

api = HfApi(token=HF_TOKEN)

print("\n" + "="*90)
print("[STEP 3] Uploading models")
print("="*90)

upload_results = []
success_count = 0
fail_count = 0

for folder_path in sorted(model_folders):
    folder_name = folder_path.name
    
    print(f"\n[Uploading] {folder_name}")
    print("-" * 80)
    
    try:
        # 上傳整個資料夾
        upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
            commit_message=f"Add {folder_name} model"
        )
        
        print(f"[OK] Successfully uploaded {folder_name}")
        print(f"     https://huggingface.co/{repo_id}/tree/main/{folder_name}")
        
        upload_results.append({
            'folder': folder_name,
            'status': 'SUCCESS',
            'url': f"https://huggingface.co/{repo_id}/tree/main/{folder_name}"
        })
        success_count += 1
        
    except Exception as e:
        print(f"[ERROR] Failed to upload {folder_name}")
        print(f"        Error: {str(e)[:80]}")
        
        upload_results.append({
            'folder': folder_name,
            'status': 'FAILED',
            'error': str(e)[:100]
        })
        fail_count += 1

print("\n" + "="*90)
print("[STEP 4] Upload Summary")
print("="*90)

print(f"\nTotal: {len(upload_results)} models")
print(f"Success: {success_count}")
print(f"Failed: {fail_count}")

if success_count > 0:
    print(f"\n[OK] Models uploaded successfully!")
    print(f"\nYou can access them at:")
    print(f"  {repo_id}")
    print(f"  https://huggingface.co/{repo_id}")
    
    print(f"\nUploaded models:")
    for result in upload_results:
        if result['status'] == 'SUCCESS':
            print(f"  [OK] {result['folder']}")
            print(f"       {result['url']}")

if fail_count > 0:
    print(f"\n[WARNING] {fail_count} models failed to upload:")
    for result in upload_results:
        if result['status'] == 'FAILED':
            print(f"  [FAILED] {result['folder']}")
            print(f"           {result.get('error', 'Unknown error')}")

print("\n" + "="*90)
print("[NEXT STEPS]")
print("="*90)

print(f"""
1. Check your uploaded models:
   https://huggingface.co/{repo_id}

2. Load a model in Python:
   from huggingface_hub import hf_hub_download
   model_path = hf_hub_download(
       repo_id='{repo_id}',
       filename='BTCUSDT_1h_v1/pytorch_model.bin',
       token='{HF_TOKEN[:10]}...'
   )

3. Share your model:
   - Add description
   - Add model card
   - Make it public/private

""")

print("="*90)
print("COMPLETE")
print("="*90 + "\n")
