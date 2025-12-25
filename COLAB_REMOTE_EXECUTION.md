# CPB v5: Colab Remote Execution Guide

在 Google Colab 中遠端執行訓練腳本的三種方式

---

## 方式 1: 最簡單 (推薦)

在 Colab cell 中只需要一行代碼:

```python
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py').text)
```

**優點**:
- 只需一行代碼
- 自動處理所有依賴
- 自動上傳結果

**缺點**:
- 無法中途修改代碼

---

## 方式 2: 使用加載器 (推薦)

```python
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_colab_loader.py').text)
```

**特點**:
- 更詳細的進度輸出
- 自動錯誤恢復
- 顯示最終統計

**流程**:
```
[STEP 1/5] Installing dependencies...
[STEP 2/5] Cloning repository...
[STEP 3/5] Loading training script...
[STEP 4/5] Executing training script...
  (訓練進行中...)
[STEP 5/5] Training completed!
```

---

## 方式 3: 分步執行 (最靈活)

### Cell 1: 安裝依賴

```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q pandas numpy scikit-learn requests huggingface-hub
```

### Cell 2: 下載並執行

```python
import requests
import os

# 下載訓練代碼
training_script = requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py'
).text

# 保存到文件
with open('/content/v5_train.py', 'w') as f:
    f.write(training_script)

# 執行
exec(training_script, {'__name__': '__main__'})
```

或者直接執行:

```bash
!python -c "import requests; exec(requests.get('https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py').text)"
```

---

## 方式 4: 使用命令行

### 直接運行腳本

```bash
!curl -s https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py | python
```

或

```bash
!wget -q -O - https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py | python
```

---

## 完整工作流 (Cell-by-Cell)

### Cell 1: 環境設置

```python
print("Setting up environment...")

# 安裝 PyTorch
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
!pip install -q pandas numpy scikit-learn requests huggingface-hub

print("Environment ready!")
```

### Cell 2: 檢查 GPU

```python
import torch

print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Cell 3: 執行訓練

```python
import requests
import datetime

print(f"Training started at {datetime.datetime.now()}")
print("This will take 2-2.5 hours...")
print("="*60)

# 執行完整訓練
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py'
).text)

print("="*60)
print(f"Training ended at {datetime.datetime.now()}")
```

### Cell 4: 檢查結果

```python
import json
from pathlib import Path

results_file = Path('/content/all_models/model_v5/training_results.json')

if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    
    mape_values = [v['mape'] for v in results.values()]
    
    print(f"Training Results:")
    print(f"  Total models: {len(results)}")
    print(f"  Average MAPE: {sum(mape_values)/len(mape_values):.6f}")
    print(f"  Best MAPE: {min(mape_values):.6f}")
    print(f"  Worst MAPE: {max(mape_values):.6f}")
    print(f"  Models < 0.02: {sum(1 for m in mape_values if m < 0.02)}/{len(mape_values)}")
    
    print(f"\nModels saved to: {results_file.parent}")
else:
    print("Results file not found!")
```

### Cell 5: 上傳到 Hugging Face

```python
import subprocess
import getpass

print("Uploading to Hugging Face...")
print("You will need your HF token")

# 如果還沒有 login
token = getpass.getpass("Enter your Hugging Face token: ")

result = subprocess.run([
    'huggingface-cli', 'upload',
    'zongowo111/cpb-models',
    '/content/all_models/model_v5',
    'model_v5',
    '--repo-type', 'model',
    '--token', token
])

if result.returncode == 0:
    print("Upload successful!")
    print("Models available at: https://huggingface.co/zongowo111/cpb-models")
else:
    print("Upload failed!")
```

---

## 監視訓練進度

### 方式 1: 實時日誌

```python
# 訓練過程會輸出到 /content/training.log
import subprocess
import time

process = subprocess.Popen(
    ['tail', '-f', '/content/training.log'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

for line in process.stdout:
    print(line.strip())
```

### 方式 2: 定期檢查

```python
import time
import json
from pathlib import Path

while True:
    results_file = Path('/content/all_models/model_v5/training_results.json')
    
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        print(f"Models trained: {len(results)}")
    
    time.sleep(60)  # 每 60 秒檢查一次
```

---

## 常見問題

### Q: 執行 `exec()` 後發生錯誤?

A: 確保:
1. 有網路連接
2. GitHub URL 可訪問
3. 有足夠的 GPU 內存

試試這個來檢查:
```python
import requests
response = requests.get('https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py')
print(f"Status: {response.status_code}")
print(f"Content length: {len(response.text)}")
```

### Q: 訓練中途中斷怎麼辦?

A: 
1. v5 會自動保存到 `/content/all_models/model_v5/`
2. 重新運行會跳過已訓練的模型
3. 最終結果保存到 `training_results.json`

### Q: 如何修改訓練配置?

A: 下載腳本後修改 `Config` 類:

```python
import requests

code = requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py'
).text

# 修改配置
code = code.replace('EPOCHS = 100', 'EPOCHS = 50')
code = code.replace('BATCH_SIZE = 64', 'BATCH_SIZE = 32')

exec(code)
```

### Q: 如何只訓練部分幣種?

A: 修改 COINS 列表:

```python
code = requests.get(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py'
).text

# 只訓練 5 個幣種
code = code.replace(
    "'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',"
    "'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',"
    "'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT',"
    "'PEPEUSDT', 'INJUSDT', 'SHIBUSDT', 'ETCUSDT', 'LUNAUSDT'",
    "'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT'"
)

exec(code)
```

---

## 最佳實踐

### 1. 使用 Colab Pro

- 支持 A100 GPU (2倍速度)
- 支持背景運行
- 支持更長的執行時間

### 2. 保持瀏覽器打開

```python
from IPython.display import Javascript
import time

# 每 60 秒點擊一次,保持連接
for i in range(150):  # 2.5 小時
    display(Javascript('''
        document.querySelector("colab-toolbar-button#connect").click()
    '''))
    time.sleep(60)
```

### 3. 記錄日誌

訓練腳本會自動保存日誌到 `/content/training.log`

### 4. 備份結果

```python
import shutil

# 下載結果
shutil.make_archive('/content/v5_results', 'zip', '/content/all_models')
print("Results saved to /content/v5_results.zip")
```

---

## 快速開始 (一分鐘)

### 選項 A: 最簡單

```python
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py').text)
```

### 選項 B: 帶加載器

```python
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_colab_loader.py').text)
```

### 選項 C: 命令行

```bash
!curl -s https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_training_complete.py | python
```

---

## 問題排查

| 問題 | 解決方案 |
|------|----------|
| CUDA out of memory | 減少 BATCH_SIZE |
| 連接中斷 | 使用 Colab Pro 或保活腳本 |
| 超時 | 分批訓練或使用 A100 |
| 下載失敗 | 檢查網路或使用代理 |

---

**版本**: v5.0
**更新**: 2025-12-25
**狀態**: 生產就緒
