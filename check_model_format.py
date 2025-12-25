#!/usr/bin/env python3
"""
Debug script to inspect model file format
"""

import torch
from huggingface_hub import hf_hub_download
import sys

print("Downloading a sample model to inspect...")

try:
    model_path = hf_hub_download(
        repo_id="zongowo111/cpb-models",
        filename="model_v4/BTCUSDT_15m.pt",
        repo_type="dataset",
        cache_dir="./hf_models"
    )
    
    print(f"Model path: {model_path}\n")
    
    # Load and inspect
    data = torch.load(model_path, map_location='cpu')
    
    print(f"Type: {type(data)}")
    print(f"Data: {data}\n")
    
    if isinstance(data, dict):
        print("Keys:")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
