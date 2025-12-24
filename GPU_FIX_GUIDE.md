# V4 Training GPU 問題完整解決方案

## 根本原因分析

你的 V4 訓練代碼中 GPU 未被使用，主要原因：

1. **資料在 CPU 和 GPU 之間重複同步** - 每個 batch 都索引和轉移資料
2. **LSTM 執行時的隱性 CPU 回退** - cuDNN 可能被誤禁用或無法最佳化
3. **記憶體測量不精確** - 未使用 `torch.cuda.synchronize()` 同步
4. **Seq2SeqLSTMGPUv2 內部可能有 CPU 專用操作**

## 快速診斷 (在 Colab 執行)

```python
# 直接執行診斷
import subprocess
import sys

# 下載診斷工具
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', 'requests'
], check=True)

import requests
code = requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v4_gpu_diagnostic.py'
).text

exec(code)
```

## 核心修正清單

### 修正 1：預先轉移所有資料到 GPU（最重要）

```python
# 錯誤做法（你目前的代碼）
for i in range(0, len(X_train), batch_size):
    X_b, y_b = X_train[i:end], y_train[i:end]  # 每次都轉移
    X_b = X_b.to(device, non_blocking=True)     # 重複同步開銷
    y_b = y_b.to(device, non_blocking=True)

# 正確做法
X_train_gpu = X_train.to(device, non_blocking=True).contiguous()
y_train_gpu = y_train.to(device, non_blocking=True).contiguous()

for i in range(0, len(X_train_gpu), batch_size):
    X_b = X_train_gpu[i:end].contiguous()  # 直接從GPU索引
    y_b = y_train_gpu[i:end].contiguous()
```

### 修正 2：強制 GPU 同步和精確測量

```python
# 在訓練循環中
optimizer.zero_grad()
pred = model(X_b, y_b, teacher_forcing_ratio=0.5)

# 重要：同步
torch.cuda.synchronize()

loss = criterion(pred, y_b)
loss.backward()

# 再次同步
torch.cuda.synchronize()

optimizer.step()
```

### 修正 3：檢查 Seq2SeqLSTMGPUv2 實現

確保沒有這些 CPU 專用操作：

```python
# 危險：會觸發 CPU 回退
if teacher_forcing_ratio > torch.rand(1):  # rand() 在 CPU
    ...

# 正確方法
use_teacher = torch.rand(1, device=device).item() < teacher_forcing_ratio
if use_teacher:
    ...

# 危險：CPU 張量混合
if some_cpu_tensor.item() > threshold:
    gpu_tensor = torch.where(...)  # CPU-only 操作

# 正確：全 GPU 操作
mask = some_gpu_tensor > threshold
gpu_tensor = torch.where(mask, ...)
```

### 修正 4：GPU 配置最佳化

```python
import torch

# 設定 GPU
torch.cuda.set_device(0)
device = torch.device('cuda:0')

# cuDNN 配置
torch.backends.cudnn.benchmark = True      # 自動調優
torch.backends.cudnn.enabled = True        # 啟用
torch.backends.cudnn.deterministic = False # 不需確定性

# TensorFloat32
torch.set_float32_matmul_precision('highest')

# 記憶體清理
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

## 改進的訓練循環模板

```python
def train_with_gpu_fix(model, X_train, y_train, X_val, y_val, 
                       epochs=200, batch_size=32, device='cuda:0'):
    
    # 步驟 1：預先轉移所有資料
    print("Transferring data to GPU...")
    X_train_gpu = X_train.to(device, non_blocking=True).contiguous()
    y_train_gpu = y_train.to(device, non_blocking=True).contiguous()
    X_val_gpu = X_val.to(device, non_blocking=True).contiguous()
    y_val_gpu = y_val.to(device, non_blocking=True).contiguous()
    
    torch.cuda.synchronize()  # 等待轉移完成
    print(f"Data on GPU: X {X_train_gpu.shape}, y {y_train_gpu.shape}")
    
    # 步驟 2：配置優化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 步驟 3：訓練循環
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        
        # 使用 GPU 上的張量索引
        for i in range(0, len(X_train_gpu), batch_size):
            end = min(i + batch_size, len(X_train_gpu))
            X_b = X_train_gpu[i:end].contiguous()
            y_b = y_train_gpu[i:end].contiguous()
            
            # 驗證位置
            assert X_b.device.type == 'cuda'
            assert y_b.device.type == 'cuda'
            
            # 前向傳播
            optimizer.zero_grad()\n            pred = model(X_b, y_b, teacher_forcing_ratio=0.5)
            \n            torch.cuda.synchronize()  # 同步
            \n            # 損失和反向傳播\n            loss = criterion(pred, y_b)\n            loss.backward()\n            \n            torch.cuda.synchronize()  # 再次同步\n            \n            # 優化步驟\n            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n            optimizer.step()\n            \n            train_loss += loss.item()\n            batch_count += 1\n        \n        train_loss /= batch_count\n        \n        # 驗證\n        model.eval()\n        val_loss = 0\n        val_count = 0\n        with torch.no_grad():\n            for i in range(0, len(X_val_gpu), batch_size):\n                end = min(i + batch_size, len(X_val_gpu))\n                X_b = X_val_gpu[i:end].contiguous()\n                y_b = y_val_gpu[i:end].contiguous()\n                pred = model(X_b, y_b, teacher_forcing_ratio=0)\n                loss = criterion(pred, y_b)\n                val_loss += loss.item()\n                val_count += 1\n        \n        val_loss /= val_count\n        \n        # 精確測量\n        torch.cuda.synchronize()\n        allocated = torch.cuda.memory_allocated() / 1e9\n        peak = torch.cuda.max_memory_allocated() / 1e9\n        \n        if (epoch + 1) % 20 == 0:\n            print(f\"Epoch {epoch+1:3d}/{epochs} | \"\n                  f\"Train: {train_loss:.6f} | Val: {val_loss:.6f} | \"\n                  f\"GPU: {allocated:.2f}GB (peak: {peak:.2f}GB)\")\n```\n\n## 檢查清單\n\n- [ ] 執行診斷工具確認 GPU 配置正常\n- [ ] 修改訓練循環以預先轉移資料\n- [ ] 加入 `torch.cuda.synchronize()` 在關鍵位置\n- [ ] 檢查 Seq2SeqLSTMGPUv2 沒有 CPU 專用操作\n- [ ] 確認訓練時 GPU 記憶體使用 > 100MB\n- [ ] 使用 `nvidia-smi` 監控 GPU 利用率 > 20%\n\n## 預期改進\n\n- 從 < 50MB 提升到 2-4GB GPU 記憶體使用\n- 訓練速度提升 10-50 倍\n- GPU 利用率從接近 0% 提升到 50-80%\n\n## 如果問題仍未解決\n\n1. 運行 `v4_gpu_diagnostic.py` 找出具體是哪個部分失敗\n2. 檢查 `v4_model_cuda_forced_v2.py` 中的 LSTM 實現\n3. 嘗試用簡單 LSTM 替換以確認問題不在其他地方\n