#!/usr/bin/env python3
"""
GPU Diagnostic Tool for V4 Model
Quickly test if GPU is being used correctly
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

print("\n" + "="*70)
print("GPU DIAGNOSTIC TOOL FOR V4 MODEL")
print("="*70 + "\n")

# Step 1: Check CUDA availability
print("[1/5] Checking CUDA availability...")
if not torch.cuda.is_available():
    print("   ERROR: CUDA is not available!")
    print("   Make sure you have a GPU enabled in Colab")
    sys.exit(1)

print(f"   OK - CUDA is available")
print()

# Step 2: Check GPU properties
print("[2/5] GPU Properties")
print("-" * 70)
try:
    device_count = torch.cuda.device_count()
    print(f"   Device count: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1e9
        print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Total Memory: {total_mem:.2f}GB")
        print(f"   CUDA Capability: {props.major}.{props.minor}")
        print(f"   Max Threads: {props.max_threads_per_block}")
        print(f"   Warp Size: {props.warp_size}")
    
    current_alloc = torch.cuda.memory_allocated(0) / 1e9
    current_reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"\n   Current Allocated: {current_alloc:.3f}GB")
    print(f"   Current Reserved: {current_reserved:.3f}GB")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print()
print()

# Step 3: Check PyTorch and CUDA versions
print("[3/5] Software Versions")
print("-" * 70)
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.version.cuda}")
print(f"   cuDNN: {torch.backends.cudnn.version()}")
print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
print()
print()

# Step 4: Test manual GPU operations
print("[4/5] Testing Manual GPU Operations")
print("-" * 70)
try:
    device = torch.device('cuda:0')
    
    # Test 1: Simple tensor operations
    print("   Test 1: Creating tensors on GPU...")
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(0) / 1e9
    
    A = torch.randn(1000, 1000, device=device, dtype=torch.float32)
    B = torch.randn(1000, 1000, device=device, dtype=torch.float32)
    
    print(f"      Created tensors on {A.device}")
    print(f"      Memory after creation: {torch.cuda.memory_allocated(0) / 1e9:.3f}GB")
    
    # Test 2: Matrix multiplication
    print("   Test 2: Matrix multiplication on GPU...")
    torch.cuda.reset_peak_memory_stats(0)
    
    for _ in range(10):
        C = torch.matmul(A, B)
    
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    
    print(f"      Matrix multiplication completed on {C.device}")
    print(f"      Peak memory usage: {peak:.3f}GB")
    
    if peak < 0.01:
        print("      WARNING: Memory usage too low - computation might be on CPU!")
    else:
        print(f"      OK - Memory usage is reasonable ({peak:.3f}GB)")
    
    # Test 3: Gradient computation
    print("   Test 3: Gradient computation on GPU...")
    torch.cuda.reset_peak_memory_stats(0)
    
    X = torch.randn(100, 1000, device=device, requires_grad=True)
    Y = torch.randn(1000, 100, device=device, requires_grad=True)
    
    Z = torch.matmul(X, Y)
    loss = Z.sum()
    loss.backward()
    
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    
    print(f"      Gradient computation completed")
    print(f"      Peak memory usage: {peak:.3f}GB")
    print(f"      X.grad device: {X.grad.device}")
    
    if X.grad.device.type != 'cuda':
        print("      ERROR: Gradient is on CPU!")
    else:
        print("      OK - Gradient is on GPU")
    
    del A, B, C, X, Y, Z
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print()

# Step 5: Test v4_model_cuda_forced_v2 if available
print("[5/5] Testing v4_model_cuda_forced_v2 Model")
print("-" * 70)

try:
    # Try to import the model
    try:
        from v4_model_cuda_forced_v2 import Seq2SeqLSTMGPUv2
        print("   Successfully imported Seq2SeqLSTMGPUv2")
    except ImportError:
        print("   Model file not found - trying to download...")
        import os
        import subprocess
        
        url = 'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_model_cuda_forced_v2.py'
        result = subprocess.run(
            ['curl', '-sS', '-o', '/tmp/v4_model_cuda_forced_v2.py', url],
            capture_output=True
        )
        
        if result.returncode == 0:
            sys.path.insert(0, '/tmp')
            from v4_model_cuda_forced_v2 import Seq2SeqLSTMGPUv2
            print("   Downloaded and imported Seq2SeqLSTMGPUv2")
        else:
            print("   Failed to download model - skipping this test")
            print(f"   Error: {result.stderr.decode()}")
            raise ImportError("Could not import model")
    
    # Test model
    device = torch.device('cuda:0')
    batch_size = 32
    
    print("\n   Creating model...")
    model = Seq2SeqLSTMGPUv2(
        input_size=4, hidden_size=256, num_layers=2,
        dropout=0.3, steps_ahead=10, output_size=4,
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Check if all parameters are on GPU
    print("\n   Verifying all parameters are on GPU...")
    all_on_gpu = True
    for name, param in model.named_parameters():
        if param.device.type != 'cuda':
            print(f"   ERROR: Parameter {name} is on {param.device}")
            all_on_gpu = False
    
    if all_on_gpu:
        print("   OK - All parameters are on GPU")
    else:
        print("   CRITICAL ERROR: Some parameters are not on GPU!")
        sys.exit(1)
    
    # Test forward pass
    print("\n   Testing forward pass...")
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    
    X = torch.randn(batch_size, 30, 4, device=device)
    y = torch.randn(batch_size, 10, 4, device=device)
    
    mem_before = torch.cuda.memory_allocated(0) / 1e9
    
    with torch.no_grad():
        output = model(X, y, teacher_forcing_ratio=0.5)
    
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated(0) / 1e9
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output device: {output.device}")
    print(f"   Memory before forward: {mem_before:.3f}GB")
    print(f"   Memory after forward: {mem_after:.3f}GB")
    print(f"   Peak memory: {peak:.3f}GB")
    print(f"   Memory used: {(mem_after - mem_before)*1000:.2f}MB")
    
    if mem_after - mem_before < 0.01:
        print("\n   WARNING: Memory usage too low!")
        print("   Possible causes:")
        print("   1. Model is computing on CPU")
        print("   2. LSTM has cuDNN fallback disabled")
        print("   3. Automatic mixed precision is too aggressive")
    elif mem_after - mem_before < 0.5:
        print("\n   CAUTION: Memory usage is lower than expected (should be 1-4GB)")
    else:
        print(f"\n   OK - Memory usage is as expected")
    
    print(f"\n   GPU Utilization: {'GOOD' if peak > 0.5 else 'LOW'}")
    
except Exception as e:
    print(f"   Error testing model: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print()
print("Summary:")
print("  - If all tests passed, your GPU is configured correctly")
print("  - If you see warnings, check COLAB_V4_GPU_GUIDE.md for solutions")
print("  - For detailed troubleshooting, see gpu_diagnosis.md")
print()
