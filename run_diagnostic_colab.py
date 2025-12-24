#!/usr/bin/env python3
"""
Colab專用 - 蛇幾分鐘GPU診斷
直接複制整個程式碼到Colab並執行
"""

import sys
import os

print("\n" + "#"*70)
print("# Colab GPU Diagnostic - Quick Run")
print("#"*70)

print("\n[Step 1] Ensuring output is printed...")
sys.stdout.flush()
print("Output flushed successfully")
sys.stdout.flush()

# Test 1: Basic imports
print("\n[Step 2] Importing libraries...")
sys.stdout.flush()

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    sys.stdout.flush()
except ImportError as e:
    print(f"ERROR: Cannot import torch: {e}")
    sys.exit(1)

# Test 2: CUDA check
print("\n[Step 3] Checking CUDA...")
sys.stdout.flush()

if not torch.cuda.is_available():
    print("\n*** CRITICAL ERROR: CUDA NOT AVAILABLE ***")
    print("\nYou must enable GPU in Colab:")
    print("1. Go to Runtime menu (top)")
    print("2. Click 'Change runtime type'")
    print("3. Select 'GPU' from Hardware accelerator dropdown")
    print("4. Click Save")
    print("5. Re-run this cell")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
sys.stdout.flush()

# Test 3: GPU memory
print("\n[Step 4] Checking GPU memory...")
sys.stdout.flush()

props = torch.cuda.get_device_properties(0)
total_mem = props.total_memory / 1e9
print(f"Total GPU Memory: {total_mem:.2f} GB")
sys.stdout.flush()

# Test 4: Tensor operation
print("\n[Step 5] Testing GPU computation...")
sys.stdout.flush()

try:
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"Matrix multiplication test: PASSED")
    print(f"GPU Memory used: {mem:.2f} GB")
    sys.stdout.flush()
except Exception as e:
    print(f"Matrix multiplication test: FAILED - {e}")
    sys.stdout.flush()

# Test 5: LSTM
print("\n[Step 6] Testing LSTM on GPU...")
sys.stdout.flush()

try:
    torch.cuda.reset_peak_memory_stats()
    lstm = torch.nn.LSTM(4, 256, 2, batch_first=True, device='cuda')
    x = torch.randn(8, 30, 4, device='cuda')
    output, _ = lstm(x)
    torch.cuda.synchronize()
    
    mem = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"LSTM test: PASSED")
    print(f"GPU Memory used: {mem:.2f} GB")
    print(f"Peak memory: {peak:.2f} GB")
    
    if peak < 0.05:
        print("\n*** WARNING: Peak memory is < 50MB ***")
        print("This may indicate LSTM is running on CPU!")
    
    sys.stdout.flush()
except Exception as e:
    print(f"LSTM test: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()

print("\n" + "#"*70)
print("# DIAGNOSTIC COMPLETE")
print("#"*70)

print("\nNext steps:")
print("1. If all tests PASSED: GPU is properly configured")
print("2. Check GPU_FIX_GUIDE.md for training code fixes")
print("3. Apply fixes to your v4_training code")
print("4. Pre-transfer data to GPU before training loop")
print("5. Add torch.cuda.synchronize() calls")

print("\n" + "#"*70)
sys.stdout.flush()
