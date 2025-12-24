#!/usr/bin/env python3
"""
V4 GPU Diagnostic - Verbose Debug Version
確保測試輸出不丢失
"""

import torch
import subprocess
import sys
import traceback

print("="*70)
print("V4 GPU DIAGNOSTIC - VERBOSE VERSION")
print("="*70)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
sys.stdout.flush()

try:
    print("\n[1/6] Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    print(f"    CUDA Available: {cuda_available}")
    sys.stdout.flush()
    
    if not cuda_available:
        print("\n*** ERROR: CUDA not available ***")
        print("    Check: Is GPU enabled in Colab/Kaggle?")
        print("    Colab: Runtime > Change runtime type > T4 GPU")
        sys.exit(1)
    
    print(f"    Device Count: {torch.cuda.device_count()}")
    print(f"    Current Device: {torch.cuda.current_device()}")
    print(f"    Device Name: {torch.cuda.get_device_name(0)}")
    sys.stdout.flush()
    
except Exception as e:
    print(f"\n*** ERROR in step 1: {str(e)} ***")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[2/6] Checking CUDA version...")
    print(f"    CUDA Version: {torch.version.cuda}")
    print(f"    cuDNN Version: {torch.backends.cudnn.version()}")
    sys.stdout.flush()
except Exception as e:
    print(f"\n*** ERROR in step 2: {str(e)} ***")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[3/6] Checking cuDNN settings...")
    print(f"    cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"    cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"    cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    sys.stdout.flush()
    
    if not torch.backends.cudnn.enabled:
        print("\n*** WARNING: cuDNN is disabled ***")
        print("    This will impact LSTM performance")
    
except Exception as e:
    print(f"\n*** ERROR in step 3: {str(e)} ***")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[4/6] Testing basic GPU tensor operation...")
    
    device = torch.device('cuda:0')
    print(f"    Creating tensor on {device}...")
    sys.stdout.flush()
    
    x = torch.randn(100, 100, device=device)
    print(f"    Tensor created: {x.shape} on {x.device}")
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Matrix multiplication
    print("    Performing matrix multiplication...")
    y = torch.randn(100, 100, device=device)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    
    mem_used = torch.cuda.memory_allocated() / 1e6
    print(f"    GPU Memory Used: {mem_used:.2f} MB")
    print("    Matrix multiplication: SUCCESS")
    sys.stdout.flush()
    
except Exception as e:
    print(f"\n*** ERROR in step 4: {str(e)} ***")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[5/6] Testing LSTM on GPU...")
    
    torch.cuda.reset_peak_memory_stats()
    print("    Creating LSTM model...")
    
    lstm = torch.nn.LSTM(4, 256, 2, batch_first=True, device=device)
    print(f"    LSTM created on {device}")
    
    print("    Creating input tensor (batch=8, seq=30, features=4)...")
    x_lstm = torch.randn(8, 30, 4, device=device)
    print(f"    Input tensor: {x_lstm.shape} on {x_lstm.device}")
    
    print("    Running LSTM forward pass...")
    torch.cuda.synchronize()
    output, (h_n, c_n) = lstm(x_lstm)
    torch.cuda.synchronize()
    
    mem_used = torch.cuda.memory_allocated() / 1e9
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"    Output shape: {output.shape}")
    print(f"    Output device: {output.device}")
    print(f"    GPU Memory Used: {mem_used:.2f} GB")
    print(f"    Peak GPU Memory: {peak_mem:.2f} GB")
    
    if peak_mem < 0.05:
        print("\n*** WARNING: Peak memory < 50MB ***")
        print("    This suggests LSTM may be running on CPU!")
    else:
        print("    LSTM on GPU: SUCCESS")
    
    sys.stdout.flush()
    
except Exception as e:
    print(f"\n*** ERROR in step 5: {str(e)} ***")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[6/6] Checking nvidia-smi...")
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
         '--format=csv,noheader'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    if result.returncode == 0:
        print(f"    {result.stdout.strip()}")
    else:
        print("    nvidia-smi not available (this is OK in some environments)")
    
    sys.stdout.flush()
    
except Exception as e:
    print(f"    nvidia-smi error (non-critical): {str(e)}")
    sys.stdout.flush()

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print("All tests completed successfully!")
print("\nIf you see this message, GPU is properly configured.")
print("\nNext steps:")
print("1. Check your training code for data transfer issues")
print("2. Look at GPU_FIX_GUIDE.md for solutions")
print("3. Pre-transfer all data to GPU before training loop")
print("="*70)
sys.stdout.flush()
