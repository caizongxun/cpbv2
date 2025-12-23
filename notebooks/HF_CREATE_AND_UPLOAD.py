#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: Create HF Repo and Upload Models
先創建 repo 然後上傳模型
"""

print("\n" + "="*90)
print("CREATE HF REPO AND UPLOAD MODELS")
print("="*90)

import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file, model_info

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

print("\n[STEP 1] Checking/Creating repo")
print("="*90)

try:
    # 嘗試取得模型信息（如果 repo 已存在）
    repo = model_info(repo_id, token=HF_TOKEN)
    print(f"[OK] Repo already exists")
    print(f"     Created: {repo.created_at}")
    print(f"     Last modified: {repo.last_modified}")
    
except Exception as e:
    # Repo 不存在，需要創建
    if "404" in str(e) or "Repository not found" in str(e):
        print(f"[INFO] Repo not found, creating...")
        
        try:
            # 創建 repo
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,
                exist_ok=True
            )
            print(f"[OK] Repo created successfully")
            print(f"     https://huggingface.co/{repo_id}")
            
        except Exception as create_error:
            print(f"[ERROR] Failed to create repo")
            print(f"        {str(create_error)[:80]}")
            
            # 嘗試 exist_ok 計數 (repo 可能已存在)
            print(f"\n[INFO] Trying with exist_ok=True...")
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    private=False,
                    exist_ok=True
                )
                print(f"[OK] Repo ready (exist_ok=True)")
            except Exception as e2:
                print(f"[ERROR] {str(e2)[:80]}")
                exit(1)
    else:
        print(f"[ERROR] {str(e)[:80]}")
        exit(1)

print("\n[STEP 2] Listing files to upload")
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

print("\n[STEP 3] Uploading files")
print("="*90)

success_count = 0
fail_count = 0
errors = []

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
            error_msg = str(e)[:70]
            print(f"FAILED")
            print(f"    Error: {error_msg}")
            errors.append((path_in_repo, error_msg))
            fail_count += 1

print("\n" + "="*90)
print("[SUMMARY]")
print("="*90)

print(f"\nUploaded: {success_count} files")
print(f"Failed: {fail_count} files")

if success_count > 0:
    print(f"\n[SUCCESS] {success_count} files uploaded to Hugging Face!")
    print(f"\nAccess your models at:")
    print(f"  https://huggingface.co/{repo_id}")
    
    if fail_count == 0:
        print(f"\n[PERFECT] All files uploaded successfully!")
    
    print(f"\nDownload example:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  model = hf_hub_download(")
    print(f"      repo_id='{repo_id}',")
    print(f"      filename='BTCUSDT_1h_v1/pytorch_model.bin'")
    print(f"  )")
    
if fail_count > 0:
    print(f"\n[WARNING] {fail_count} files failed:")
    for path, err in errors[:5]:  # Show first 5 errors
        print(f"  - {path}")
        print(f"    {err}")

print("\n" + "="*90)
print("COMPLETE")
print("="*90 + "\n")
