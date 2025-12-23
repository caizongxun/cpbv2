#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: Direct Hugging Face Upload
使用 git 方法直接上傳模型
"""

print("\n" + "="*90)
print("DIRECT HUGGING FACE UPLOAD")
print("="*90)

import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file

print("\n[SETUP]")
print("="*90)

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_USERNAME = os.environ.get('HF_USERNAME')

if not HF_TOKEN or not HF_USERNAME:
    print("[ERROR] Token or username not set")
    exit(1)

print(f"Token: {HF_TOKEN[:20]}...")
print(f"Username: {HF_USERNAME}")

api = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USERNAME}/cpbmodel"

print(f"\nRepo: {repo_id}")
print(f"URL: https://huggingface.co/{repo_id}")

print("\n[STEP 1] Listing files to upload")
print("="*90)

hf_dir = Path('./hf_models')
folders = sorted([d for d in hf_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

if not folders:
    print("[ERROR] No model folders found")
    exit(1)

print(f"[OK] Found {len(folders)} model folders\n")

for folder in folders:
    files = list(folder.glob('*'))
    print(f"{folder.name}:")
    for f in sorted(files):
        if f.is_file():
            size = f.stat().st_size / (1024*1024)
            print(f"  - {f.name} ({size:.1f} MB)")

print("\n[STEP 2] Uploading files")
print("="*90)

success_count = 0
fail_count = 0

for folder in folders:
    folder_name = folder.name
    print(f"\n[Uploading] {folder_name}")
    print("-" * 80)
    
    files = sorted([f for f in folder.glob('*') if f.is_file()])
    
    for file_path in files:
        file_name = file_path.name
        path_in_repo = f"{folder_name}/{file_name}"
        
        try:
            print(f"  Uploading: {path_in_repo}...", end=' ', flush=True)
            
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                token=HF_TOKEN,
                commit_message=f"Add {path_in_repo}"
            )
            
            print("OK")
            success_count += 1
            
        except Exception as e:
            print(f"FAILED")
            print(f"    Error: {str(e)[:70]}")
            fail_count += 1

print("\n" + "="*90)
print("[SUMMARY]")
print("="*90)

print(f"\nUploaded: {success_count} files")
print(f"Failed: {fail_count} files")

if success_count > 0:
    print(f"\n[SUCCESS] Models uploaded to Hugging Face!")
    print(f"\nAccess your models at:")
    print(f"  https://huggingface.co/{repo_id}")
    
    print(f"\nDownload example:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  model = hf_hub_download(repo_id='{repo_id}',")
    print(f"                          filename='BTCUSDT_1h_v1/pytorch_model.bin')")
else:
    print(f"\n[ERROR] Upload failed")
    exit(1)

print("\n" + "="*90)
print("COMPLETE")
print("="*90 + "\n")
