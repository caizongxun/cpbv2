# 立即行動 - GPU 沒有輸出的解決

## 你遇到的問題

执行訓練程式碼後，程式很快結束且沒有任何輸出

## 原因可能是

1. **GPU 未啟用** - 你在 Colab 沒有啟動 GPU
2. **CUDA 不可用** - PyTorch 找不到 GPU
3. **資料轉移失敗** - 程式崃啖了
4. **模型位置錯誤** - 模型沒有正確轉移到GPU

## 骗你診斷 - 5 分鐘內完成

### 1. 先確認 GPU 是否啟用

在 Colab 執行下面代碼：

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("\n*** GPU NOT AVAILABLE ***")
    print("Go to: Runtime > Change runtime type > GPU > Save")
```

### 2. 執行新的診斷工具

複制整個代碼到 Colab 並執行：

```python
import subprocess, sys
url = 'https://raw.githubusercontent.com/caizongxun/cpbv2/main/run_diagnostic_colab.py'
subprocess.run([sys.executable, '-c', f'exec(open(subprocess.run(["curl", "-s", "{url}"], capture_output=True, text=True).stdout).read()))'], check=False)
```

**或者更簡單（直接複制購購）：**

```python
import torch, sys

print("\n" + "#"*70)
print("QUICK GPU CHECK")
print("#"*70)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n*** ERROR: GPU NOT AVAILABLE ***")
    print("Fix: Runtime > Change runtime type > T4 GPU")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")

torch.cuda.reset_peak_memory_stats()
x = torch.randn(100, 100, device='cuda')
y = torch.randn(100, 100, device='cuda')
z = torch.matmul(x, y)
torch.cuda.synchronize()

mem = torch.cuda.max_memory_allocated() / 1e9
print(f"\nGPU Memory Used: {mem:.2f} GB")
print(f"Status: {'PASS' if mem > 0.01 else 'FAIL - CPU fallback detected'}")
print("\n" + "#"*70)
```

### 3. 如果 GPU 可用，來修正你的訓練代碼

沒有輸出通常表示程式購了。似便在 GPU 上執行，也會黑上一段時間。

**修正智穡：**

```python
# 你目前的訓練代碼
for i in range(0, len(X_train), batch_size):
    X_b, y_b = X_train[i:end], y_train[i:end]  # 每次轉移
    X_b = X_b.to(device, non_blocking=True)
    y_b = y_b.to(device, non_blocking=True)
```

**修正後：**

```python
# 最可能怨程式購了的事
# 重新開始 kernel / runtime

# 修正一：一次性轉移
# 就在模型延停之前子籋漠下去
print("Transferring all data to GPU...")
X_gpu = X_train.to(device, non_blocking=True).contiguous()
y_gpu = y_train.to(device, non_blocking=True).contiguous()
print(f"Data on GPU: X {X_gpu.shape}, y {y_gpu.shape}")

# 訓練迴圈
for i in range(0, len(X_gpu), batch_size):
    end = min(i + batch_size, len(X_gpu))
    X_b = X_gpu[i:end].contiguous()
    y_b = y_gpu[i:end].contiguous()
    
    # 修正二：加同步
    optimizer.zero_grad()
    pred = model(X_b, y_b, teacher_forcing_ratio=0.5)
    torch.cuda.synchronize()  # <-- 鞠關鍵！
    loss = criterion(pred, y_b)
    loss.backward()
    torch.cuda.synchronize()  # <-- 鞠關鍵！
    optimizer.step()
```

### 4. 打印輸出確認不遺漏

似便你的代碼有詳細的輸出，所以一定要把重要的信息打印出來：

```python
import sys

print("Starting training...")
sys.stdout.flush()  # 等候了，重要步驟一定要加這個！

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}...")
    sys.stdout.flush()  # <-- 鞠關鍵！
    
    for i in range(0, len(X_gpu), batch_size):
        # ... training code ...
        pass
    
    print(f"Loss: {loss.item():.6f}")
    sys.stdout.flush()  # <-- 鞠關鍵！
```

## 沒有輸出的 3 個最可能原因：

| 原因 | 信號 | 解決方梏 |
|---------|--------|--------|
| GPU 未啟用 | Runtime > Change > GPU | 重新啟動 |
| 模型/資料位置錯誤 | 出現 Exception | 使用 try-except 捕撷 |
| 輸出未止時橛醺 | 訓練橛醺 | 加 sys.stdout.flush() |

## 新手使用 - 從手第一偏詳細代碼開始：

```python
#!/usr/bin/env python3
"""
最小化GPU測試 - Colab直接複制型
"""
import torch
import sys

print("Step 1: Check GPU...")
sys.stdout.flush()

if not torch.cuda.is_available():
    print("GPU not available!")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
sys.stdout.flush()

print("\nStep 2: Test LSTM...")
sys.stdout.flush()

torch.cuda.reset_peak_memory_stats()
lstm = torch.nn.LSTM(4, 256, 2, batch_first=True, device='cuda')
x = torch.randn(8, 30, 4, device='cuda')
output, _ = lstm(x)
torch.cuda.synchronize()

mem = torch.cuda.max_memory_allocated() / 1e9
print(f"LSTM test passed. Memory: {mem:.2f}GB")
sys.stdout.flush()

if mem < 0.05:
    print("WARNING: Memory too low, might be CPU execution")
else:
    print("SUCCESS: GPU is working!")
    
sys.stdout.flush()
```

## 为什么沒有輸出?

- 每个程式範纋三推批不推新手沒有 `sys.stdout.flush()`
- 資料轉移時程式崃啖了，沒有例外捕撷
- 模型初始化時程式崃啖
- GPU未啟用（最常見）

## 需要盗助?

實作上有一個一旅二幾幾兩加一的可能：就是模型愛膺膺地從 CPU 跳到 GPU，但首先你要確認 GPU 是嗷了了
