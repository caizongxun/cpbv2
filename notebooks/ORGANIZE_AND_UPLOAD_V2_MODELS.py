"""
Organize and Upload V2 Models to Hugging Face

Logic:
1. Move all trained V2 models from models/ to ALL_MODELS/
2. Add v2 tag to model names
3. Upload to Hugging Face Hub
4. Same upload logic as V1
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

print("="*70)
print("V2 Models Organization & Upload to Hugging Face")
print("="*70)

# ==================== Step 1: Organize Models ====================
def organize_v2_models():
    """
    Move V2 models from models/ to ALL_MODELS/
    """
    print("\n[1] Organizing V2 Models...")
    
    # Create ALL_MODELS directory if not exists
    os.makedirs('ALL_MODELS', exist_ok=True)
    os.makedirs('ALL_MODELS/v2', exist_ok=True)
    
    models_dir = 'models'
    all_models_v2_dir = 'ALL_MODELS/v2'
    
    # Find all v2 models
    v2_models = []
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.startswith('v2_model_') and filename.endswith('.h5'):
                source_path = os.path.join(models_dir, filename)
                dest_path = os.path.join(all_models_v2_dir, filename)
                
                # Move file
                shutil.copy2(source_path, dest_path)
                v2_models.append(filename)
                print(f"   ✓ {filename}")
    
    print(f"\n[OK] Moved {len(v2_models)} V2 models to ALL_MODELS/v2/")
    
    return v2_models

# ==================== Step 2: Create Metadata ====================
def create_v2_metadata(v2_models):
    """
    Create metadata for V2 models
    """
    print("\n[2] Creating V2 Metadata...")
    
    metadata = {
        'model_type': 'V2',
        'model_version': 'v2.0',
        'description': 'CPB V2 Models - Predict [Price, Volatility]',
        'output': '[predicted_price, predicted_volatility]',
        'training_date': datetime.now().isoformat(),
        'total_models': len(v2_models),
        'models': []
    }
    
    # Extract pair names from model filenames
    for model_file in v2_models:
        pair_name = model_file.replace('v2_model_', '').replace('.h5', '')
        metadata['models'].append({
            'filename': model_file,
            'pair': pair_name,
            'output_format': '[price, volatility]'
        })
    
    # Save metadata
    metadata_path = 'ALL_MODELS/v2/metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Metadata created: {metadata_path}")
    
    return metadata

# ==================== Step 3: Create README ====================
def create_v2_readme():
    """
    Create README for V2 models
    """
    print("\n[3] Creating V2 README...")
    
    readme_content = """# CPB V2 Models

## Overview
CPB V2 Models predict both price and volatility for cryptocurrency trading pairs.

## Model Output
```
[predicted_price, predicted_volatility]
```

## Supported Pairs
- BTC_USDT
- ETH_USDT
- SOL_USDT
- XRP_USDT
- ADA_USDT
- BNB_USDT
- DOGE_USDT
- LINK_USDT
- AVAX_USDT
- MATIC_USDT
- ATOM_USDT
- NEAR_USDT
- FTM_USDT
- ARB_USDT
- OP_USDT
- LIT_USDT
- STX_USDT
- INJ_USDT
- LUNC_USDT
- LUNA_USDT

## Architecture
- LSTM (64 units)
- LSTM (32 units)
- Dense (32 units)
- Dense (16 units)
- Output (2 units: price, volatility)

## Input Format
OHLCv sequences with shape (20, 4) where:
- 20 = sequence length
- 4 = [open, high, low, close]

## Usage
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('v2_model_BTC_USDT.h5')

# Make prediction
predictions = model.predict(klines_sequence)
predicted_price = predictions[0, 0]
predicted_volatility = predictions[0, 1]
```

## Training Details
- Epochs: 50
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Validation Split: 20%
- Early Stopping: Patience 5

## Files
Each model file: `v2_model_{PAIR}.h5`
Example: `v2_model_BTC_USDT.h5`
"""
    
    readme_path = 'ALL_MODELS/v2/README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"   README created: {readme_path}")

# ==================== Step 4: Upload to Hugging Face ====================
def upload_v2_to_huggingface(hf_token, hf_username):
    """
    Upload V2 models to Hugging Face Hub
    Same logic as V1 upload
    """
    print("\n[4] Uploading V2 Models to Hugging Face...")
    
    try:
        from huggingface_hub import HfApi, HfFolder
        
        # Save token
        HfFolder.save_token(hf_token)
        
        api = HfApi()
        
        # Verify token
        try:
            user_info = api.whoami(token=hf_token)
            print(f"   [OK] Logged in as: {user_info['name']}")
        except Exception as e:
            print(f"   [ERROR] Invalid token: {e}")
            return False
        
        # Repository info
        repo_id = f"{hf_username}/cpb-v2-models"
        repo_type = "model"
        
        print(f"   Target repository: {repo_id}")
        
        # Create repository if not exists
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type, token=hf_token)
            print(f"   [OK] Repository exists")
        except:
            print(f"   [INFO] Creating repository...")
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=False,
                token=hf_token
            )
            print(f"   [OK] Repository created")
        
        # Upload models
        models_dir = 'ALL_MODELS/v2'
        upload_count = 0
        
        for filename in os.listdir(models_dir):
            if filename.endswith(('.h5', '.json', '.md')):
                file_path = os.path.join(models_dir, filename)
                
                try:
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=filename,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=hf_token,
                        commit_message=f"Add {filename} to V2 models"
                    )
                    print(f"   ✓ {filename}")
                    upload_count += 1
                except Exception as e:
                    print(f"   ✗ {filename}: {e}")
        
        print(f"\n[OK] Successfully uploaded {upload_count} files to Hugging Face")
        print(f"   Repository: https://huggingface.co/{repo_id}")
        
        return True
        
    except ImportError:
        print("   [ERROR] huggingface_hub not installed")
        print("   Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"   [ERROR] Upload failed: {e}")
        return False

# ==================== Main Execution ====================
def main():
    """
    Main execution flow
    """
    
    # Step 1: Organize models
    v2_models = organize_v2_models()
    
    if not v2_models:
        print("\n[ERROR] No V2 models found in models/ directory")
        print("        Please run training first")
        return
    
    # Step 2: Create metadata
    metadata = create_v2_metadata(v2_models)
    
    # Step 3: Create README
    create_v2_readme()
    
    # Step 4: Upload to Hugging Face
    print("\n[5] Hugging Face Upload")
    
    hf_token = input("   Enter Hugging Face Token: ").strip()
    hf_username = input("   Enter Hugging Face Username: ").strip()
    
    if hf_token and hf_username:
        upload_success = upload_v2_to_huggingface(hf_token, hf_username)
        
        if upload_success:
            # Create summary
            summary = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'total_models': len(v2_models),
                'repository': f"https://huggingface.co/{hf_username}/cpb-v2-models",
                'models': metadata['models']
            }
        else:
            summary = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'total_models': len(v2_models),
                'note': 'Models organized but upload to HF failed'
            }
    else:
        print("   [INFO] Skipped Hugging Face upload")
        summary = {
            'status': 'models_organized',
            'timestamp': datetime.now().isoformat(),
            'total_models': len(v2_models),
            'note': 'Models in ALL_MODELS/v2/ ready for upload'
        }
    
    # Save summary
    summary_path = 'ALL_MODELS/v2/upload_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("Organization & Upload Summary")
    print("="*70)
    print(f"Total V2 Models: {len(v2_models)}")
    print(f"Location: ALL_MODELS/v2/")
    print(f"Summary: {summary_path}")
    print("\n[OK] V2 Models Organization Complete!")
    print("="*70)

if __name__ == '__main__':
    main()
