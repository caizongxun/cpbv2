#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: Bulk Upload All Models at Once
一次上傳整個 all_models 資料夾（全部16個幣種）
"""

print("\n" + "="*90)
print("BULK UPLOAD ALL MODELS (All 16 Coins)")
print("="*90)

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

print("\n[SETUP]")
print("="*90)

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_USERNAME = os.environ.get('HF_USERNAME')

if not HF_TOKEN or not HF_USERNAME:
    print("[ERROR] Token or username not set")
    print("Please set: HF_TOKEN and HF_USERNAME")
    sys.exit(1)

print(f"Token: {HF_TOKEN[:20]}...")
print(f"Username: {HF_USERNAME}")

REPO_ID = f"{HF_USERNAME}/cpbmodel"
print(f"\nRepo: {REPO_ID}")
print(f"URL: https://huggingface.co/{REPO_ID}")

print("\n[STEP 1] Finding all_models folder")
print("="*90)

# 尋找 all_models 資料夾
model_dir = Path('./all_models')

if not model_dir.exists():
    print(f"[ERROR] 'all_models' folder not found at {model_dir.absolute()}")
    print(f"\nAvailable directories:")
    for d in Path('.').iterdir():
        if d.is_dir() and 'model' in d.name.lower():
            print(f"  - {d.name}")
    sys.exit(1)

print(f"[OK] Found: {model_dir}")
print(f"[OK] Absolute path: {model_dir.absolute()}")

# 統計資料夾
folders = sorted([d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__'])
file_count = sum(len(list(f.glob('*'))) for f in folders)
total_size = sum(sum(f.stat().st_size for f in folder.glob('*') if f.is_file()) for folder in folders) / (1024*1024)

print(f"[OK] Found {len(folders)} coin folders")
print(f"[OK] Total files: {file_count}")
print(f"[OK] Total size: {total_size:.1f} MB")
print(f"\n[COINS]")
for i, folder in enumerate(folders, 1):
    files = [f for f in folder.glob('*') if f.is_file()]
    folder_size = sum(f.stat().st_size for f in files) / (1024*1024)
    print(f"  {i:2d}. {folder.name:20s} ({len(files)} files, {folder_size:5.1f} MB)")

print("\n[STEP 2] Uploading entire folder")
print("="*90)

api = HfApi(token=HF_TOKEN)

try:
    print(f"[INFO] Starting bulk upload of ALL {len(folders)} coins...")
    print(f"[INFO] This will take several minutes...")
    print(f"[INFO] Do NOT interrupt this process!\n")
    
    # 一次上傳整個資料夾（包括所有子資料夾和檔案）
    upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(model_dir),
        token=HF_TOKEN,
        commit_message="Bulk upload all 16 CPB v1 coin models",
        ignore_patterns=[
            "*.pyc",
            "__pycache__",
            ".git*",
            "*.egg-info",
            ".DS_Store",
        ],
        multi_commit=True,  # 允許多個提交（大檔案用）
        multi_commit_pr=False,
    )
    
    print(f"\n[SUCCESS] All models uploaded successfully!")
    print(f"\n" + "="*90)
    print("[SUMMARY]")
    print("="*90)
    
    print(f"\nRepo: https://huggingface.co/{REPO_ID}")
    print(f"\nUploaded {len(folders)} coin models:")
    for i, folder in enumerate(folders, 1):
        print(f"  {i:2d}. {folder.name}")
    
    print(f"\nTotal size: {total_size:.1f} MB")
    
    print(f"\n[NEXT STEPS]")
    print(f"="*90)
    print(f"\n1. Go to: https://huggingface.co/{REPO_ID}")
    print(f"2. Add/Edit Model Card (README.md)")
    print(f"3. Add License")
    print(f"4. Add Tags")
    print(f"5. Set visibility (public/private)")
    
    print(f"\n[DOWNLOAD IN FUTURE]")
    print(f"="*90)
    print(f"\nfrom huggingface_hub import hf_hub_download")
    print(f"\n# Download any coin model")
    print(f"model = hf_hub_download(")
    print(f"    repo_id='{REPO_ID}',")
    print(f"    filename='BTCUSDT_1h_v1/pytorch_model.bin'")
    print(f")")
    
except KeyboardInterrupt:
    print(f"\n\n[CANCELLED] Upload interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Upload failed")
    print(f"\nError details:")
    print(f"  {str(e)[:200]}")
    print(f"\n[TIP] Possible solutions:")
    print(f"  1. Check internet connection")
    print(f"  2. Verify HF_TOKEN is valid")
    print(f"  3. Check disk space")
    print(f"  4. Try again later")
    sys.exit(1)

print("\n" + "="*90)
print("COMPLETE")
print("="*90 + "\n")
