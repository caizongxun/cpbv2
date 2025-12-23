#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: Copy Trained Models to HF Upload Folders
從 trained_models 複製模型檔案到 hf_models
"""

print("\n" + "="*90)
print("COPY TRAINED MODELS TO HF FOLDERS")
print("="*90)

from pathlib import Path
import shutil

print("\n[STEP 1] Finding trained models")
print("="*90)

trained_dir = Path('./trained_models')
hf_dir = Path('./hf_models')

if not trained_dir.exists():
    print(f"[ERROR] trained_models directory not found: {trained_dir}")
    print("\nPlease run FINAL_PRODUCTION_V1_WITH_SAVE.py first")
    exit(1)

model_files = list(trained_dir.glob('*_model.pth'))

if not model_files:
    print(f"[ERROR] No trained models found in {trained_dir}")
    exit(1)

print(f"[OK] Found {len(model_files)} trained models:")
for mf in sorted(model_files):
    print(f"     - {mf.name}")

print("\n[STEP 2] Copying to HF folders")
print("="*90)

if not hf_dir.exists():
    print(f"[ERROR] HF folder not found: {hf_dir}")
    print("Please run ORGANIZE_MODELS.py first")
    exit(1)

hf_folders = sorted([d for d in hf_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

if not hf_folders:
    print(f"[ERROR] No HF folders found in {hf_dir}")
    exit(1)

print(f"[OK] Found {len(hf_folders)} HF folders\n")

copy_count = 0

for hf_folder in hf_folders:
    folder_name = hf_folder.name
    # 從資料夾名提取幣種 (e.g., BTCUSDT_1h_v1 -> BTCUSDT)
    coin_name = folder_name.split('_')[0]
    
    # 尋找對應的訓練模型
    trained_model = trained_dir / f"{coin_name}_model.pth"
    
    if not trained_model.exists():
        print(f"[WARNING] No trained model for {coin_name}")
        print(f"          Expected: {trained_model}")
        continue
    
    # 目標檔名
    target_file = hf_folder / 'pytorch_model.bin'
    
    try:
        # 複製檔案
        shutil.copy2(trained_model, target_file)
        
        # 檢查檔案大小
        file_size = target_file.stat().st_size / (1024*1024)
        
        print(f"[OK] {folder_name}")
        print(f"     Source: {trained_model.name}")
        print(f"     Target: {target_file.name} ({file_size:.1f} MB)")
        
        copy_count += 1
        
    except Exception as e:
        print(f"[ERROR] Failed to copy {coin_name}")
        print(f"        {str(e)[:60]}")

print("\n" + "="*90)
print(f"[SUMMARY] Copied {copy_count} model files")
print("="*90)

# 驗證所有資料夾都有 pytorch_model.bin
print("\n[STEP 3] Verifying all folders")
print("="*90)

all_complete = True
for hf_folder in hf_folders:
    folder_name = hf_folder.name
    bin_file = hf_folder / 'pytorch_model.bin'
    
    if bin_file.exists():
        file_size = bin_file.stat().st_size / (1024*1024)
        print(f"[OK] {folder_name}: pytorch_model.bin ({file_size:.1f} MB)")
    else:
        print(f"[ERROR] {folder_name}: pytorch_model.bin missing!")
        all_complete = False

if all_complete:
    print("\n" + "="*90)
    print("[SUCCESS] All models are complete and ready for upload!")
    print("="*90)
    print("\nFolder structure:")
    
    for hf_folder in hf_folders:
        folder_name = hf_folder.name
        files = sorted([f.name for f in hf_folder.glob('*')])
        print(f"\n{folder_name}/")
        for fname in files:
            print(f"  - {fname}")
    
    print("\n" + "="*90)
    print("Next: Run the upload script")
    print("="*90)
    print("\nimport urllib.request")
    print("import os")
    print("")
    print("os.environ['HF_TOKEN'] = 'your_token_here'")
    print("os.environ['HF_USERNAME'] = 'your_username'")
    print("")
    print("urllib.request.urlretrieve(")
    print("    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/notebooks/HF_MODEL_UPLOADER_FIXED.py',")
    print("    'upload.py'")
    print(")")
    print("exec(open('upload.py').read())")
else:
    print("\n[ERROR] Some models are still incomplete")
    exit(1)

print("\n")
