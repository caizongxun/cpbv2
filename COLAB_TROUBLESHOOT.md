# Colab 问题排查指南

## 错误问题: ModuleNotFoundError: No module named 'src'

### 原因
Colab 中的 Python 路径与仓库目录不同步，导致无法找到 `src` 模块。

### 解决方案

**使用修复版本的 Notebook** (推荐)

```
https://colab.research.google.com/github/caizongxun/cpbv2/blob/main/notebooks/train_colab_fixed.ipynb
```

这个版本包含:
- ✅ 自动 clone 仓库
- ✅ 正确的路径设置 (`sys.path.insert(0, '/tmp/cpbv2')`)
- ✅ 依赖自动安装
- ✅ 所有错误处理

---

### 如果使用原始 Notebook，手动修复:

#### 方法 1: 在 Cell 3 处添加路径修复

```python
import os
import sys

# Add repo to path
sys.path.insert(0, '/tmp/cpbv2')

# Verify
print(f'Working dir: {os.getcwd()}')
print(f'sys.path[0]: {sys.path[0]}')
```

#### 方法 2: 修改导入语句

```python
# 旧的 (错误):
from src.data_collector import BinanceDataCollector

# 新的 (正确):
import sys
sys.path.insert(0, '/tmp/cpbv2')
from src.data_collector import BinanceDataCollector
```

#### 方法 3: 使用相对导入

```python
import os
os.chdir('/tmp/cpbv2')  # Change to repo directory
sys.path.insert(0, os.getcwd())

# Now import
from src.data_collector import BinanceDataCollector
```

---

## 执行步骤 (修复版本)

### STEP 0: 验证环境
```python
import torch
print(f'GPU: {torch.cuda.is_available()}')
print(f'Python: {sys.version}')
```

### STEP 1: Clone 仓库
```python
os.chdir('/tmp')
!git clone https://github.com/caizongxun/cpbv2.git
os.chdir('/tmp/cpbv2')
```

### STEP 2: 安装依赖
```python
!pip install -q torch pandas numpy scikit-learn
!pip install -q requests ta-lib huggingface_hub
```

### STEP 3: 修复路径 (关键!)
```python
import sys
repo_path = '/tmp/cpbv2'
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

print(f'sys.path[0]: {sys.path[0]}')
```

### STEP 4: 导入模块
```python
from src.data_collector import BinanceDataCollector
from src.feature_engineer import FeatureEngineer
from src.data_preprocessor import DataPreprocessor
from src.model import LSTMModel
from src.trainer import Trainer

print('✓ All modules imported successfully!')
```

### STEP 5-8: 照常执行其他步骤

---

## 其他常见 Colab 问题

### 问题: GPU 内存不足

**症状**: `CUDA out of memory`

**解决**:
```python
# 在 config/model_params.json 中修改:
{
  "training": {
    "batch_size": 16,  # 从 32 降低到 16
    "epochs": 30       # 从 50 降低到 30
  }
}
```

### 问题: 网络超时

**症状**: `Connection timeout` 或 `Binance API error`

**解决**:
```python
# Binance API 自动重试 3 次
# 如果仍然失败，等 1-2 分钟后重新运行

collector = BinanceDataCollector()
df = collector.get_historical_klines(
    'BTCUSDT', '15m',
    max_retries=5  # 增加重试次数
)
```

### 问题: 数据验证失败

**症状**: `Data validation failed`

**解决**:
```python
# 跳过失败的币种，继续训练
if not BinanceDataCollector.validate_data(df):
    logger.warning(f'Skipping {coin}')
    continue
```

### 问题: 12 小时 Colab 时间限制

**症状**: `Disconnected`

**解决**:
1. 分批训练 (每批 12 个币种)
2. 或使用 Colab Pro (无限时间)
3. 在本地 GPU 上训练 (更快)

```python
# 分批示例
coins_batch_1 = coins[:12]
coins_batch_2 = coins[12:21]

# 第一批
for coin in coins_batch_1:
    # 训练...
    pass

# 保存检查点
torch.save(model.state_dict(), 'checkpoint_batch1.pt')
```

---

## Colab 最优实践

### ✓ 推荐做法

1. **定期检查点保存**
   ```python
   if (epoch + 1) % 10 == 0:
       torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')
   ```

2. **内存监控**
   ```python
   import psutil
   memory_usage = psutil.virtual_memory().percent
   print(f'Memory: {memory_usage}%')
   ```

3. **结果备份**
   ```python
   # 保存到 Google Drive
   !cp -r models/ /content/drive/MyDrive/cpbv2_models/
   ```

4. **实时日志**
   ```python
   # 日志实时显示
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   ```

### ❌ 避免做法

1. ❌ 一次加载所有数据到内存
2. ❌ 不检查路径
3. ❌ 不保存中间结果
4. ❌ 忽视 GPU 内存警告
5. ❌ 在 Colab 中使用 `localhost` 或本地文件

---

## 调试技巧

### 检查路径
```python
import sys
import os

print('Current directory:', os.getcwd())
print('sys.path[0]:', sys.path[0])
print('src exists:', os.path.exists('src'))
print('Files in src:', os.listdir('src'))
```

### 验证导入
```python
try:
    from src.data_collector import BinanceDataCollector
    print('✓ BinanceDataCollector imported')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    print(f'  Looking in: {sys.path[0]}')
```

### 测试 API 连接
```python
try:
    df = collector.get_historical_klines('BTCUSDT', '15m', limit=100)
    print(f'✓ API works: {len(df)} candles')
except Exception as e:
    print(f'✗ API error: {e}')
```

---

## 快速修复命令

如果遇到问题，在 Colab 中运行:

```python
# 重置并修复
import sys
import os

# 1. 清除旧数据
!rm -rf /tmp/cpbv2

# 2. 重新 Clone
!git clone https://github.com/caizongxun/cpbv2.git /tmp/cpbv2

# 3. 修复路径
os.chdir('/tmp/cpbv2')
sys.path.insert(0, '/tmp/cpbv2')

# 4. 验证
print(f'Working: {os.getcwd()}')
print(f'Path: {sys.path[0]}')

# 5. 测试导入
from src.data_collector import BinanceDataCollector
print('✓ All fixed!')
```

---

## 获取帮助

1. **检查日志**: 查看每行的输出
2. **查看文档**: README.md, QUICKSTART.md
3. **提交 Issue**: https://github.com/caizongxun/cpbv2/issues
4. **使用修复版本**: train_colab_fixed.ipynb

---

## 推荐

**强烈推荐使用修复版本**: 

https://colab.research.google.com/github/caizongxun/cpbv2/blob/main/notebooks/train_colab_fixed.ipynb

所有问题都已解决! 🎉
