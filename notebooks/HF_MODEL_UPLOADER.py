#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: Hugging Face 模型上傳器
自動創建 repo 並上傳整個資料夾
上載邏輯：一次上載整個資料夾，不有 API 限制
"""

import os
import json
import pickle
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*90)
print("HUGGING FACE MODEL UPLOADER - V1")
print("="*90)

class HFModelManager:
    def __init__(self, hf_token=None, username=None):
        """
        hf_token: Hugging Face API 令牧 (從 huggingface.co 获取)
        username: Hugging Face 用戶名 (可以缺省)
        """
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.username = username
        self.repo_name = 'cpbmodel'
        self.framework = 'pytorch'
        self.task = 'text-classification'  # 敵論：會版改为時間序列預測
        
        if not self.hf_token:
            print("⚠️  注意: HF_TOKEN 未設置，此次上傳會是簡技模式。")
            print("   設置方法: os.environ['HF_TOKEN'] = 'your_token'")
            print("   或執行: huggingface-cli login")
        else:
            self._login_hf()
    
    def _login_hf(self):
        """登錄 Hugging Face"""
        try:
            from huggingface_hub import login
            login(token=self.hf_token)
            print("✓ Hugging Face 登錄成功")
        except:
            print("⚠️  Hugging Face 登錄失敗，請確保 token 正確")
    
    def create_model_folder(self, coin, timeframe='1h', results=None):
        """
        为每個模型扵建資料夾
        """
        folder_name = f"{coin}_{timeframe}_v1"
        folder_path = Path(f"./hf_models/{folder_name}")
        folder_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Creating Model Folder: {folder_name}]")
        
        # 1. 上傳模型檔
        # 注: 此次需要從訓練後的 model 對象
        # model_path = folder_path / 'pytorch_model.bin'
        # torch.save(model.state_dict(), model_path)
        # print(f"  ✓ Model saved: {model_path}")
        
        # 2. 上傳配置檔
        config = {
            'coin': coin,
            'timeframe': timeframe,
            'version': 'v1',
            'model_type': 'LSTM',
            'input_size': 14,  # 敵論值，從实際訓練你是何得準
            'hidden_size': 256,
            'num_layers': 3,
            'dropout': 0.5,
            'output_size': 2,
            'created_at': datetime.now().isoformat(),
            'framework': 'pytorch',
        }
        
        if results:
            config.update({
                'accuracy': results.get('accuracy', 0),
                'f1_score': results.get('f1', 0),
                'epochs': results.get('epochs', 0),
            })
        
        config_path = folder_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Config saved: {config_path}")
        
        # 3. 上傳預处理器配置（缪化模型）
        preprocessor_config = {
            'scaler_type': 'StandardScaler',
            'lookback': 20,
            'features': [
                'returns', 'momentum_3', 'momentum_5', 'momentum_10',
                'sma_ratio', 'price_sma_20', 'rsi_14',
                'volatility_10', 'volatility_20'
            ],
            'target_type': 'dynamic_threshold',
            'volatility_multiplier': 0.5
        }
        
        preprocessor_path = folder_path / 'preprocessor.json'
        with open(preprocessor_path, 'w') as f:
            json.dump(preprocessor_config, f, indent=2)
        print(f"  ✓ Preprocessor config saved: {preprocessor_path}")
        
        # 4. 上傳 README.md
        readme = f"""---
library_name: pytorch
tags:
- crypto
- trading
- lstm
- prediction
- binance
---

# CPB Model: {coin} - {timeframe.upper()} - V1

## 模型詳情

**Coin:** {coin}
**Timeframe:** {timeframe}
**Version:** v1
**Framework:** PyTorch
**Architecture:** LSTM (3 layers)

## 性能

"""
        
        if results:
            readme += f"""- **Accuracy:** {results.get('accuracy', 0):.2%}
- **F1 Score:** {results.get('f1', 0):.4f}
- **Epochs:** {results.get('epochs', 0)}

"""
        
        readme += f"""## 使用方法

```python
from transformers import AutoModel

model = AutoModel.from_pretrained('your_username/cpbmodel', subfolder='{folder_name}')
```

## 模型組件

- `pytorch_model.bin` - 訓練好的模型權重
- `config.json` - 模型配置
- `preprocessor.json` - 預处理配置
- `README.md` - 模型文檔

## 特性

### 模型索齐
- Input Size: {config['input_size']}
- Hidden Size: {config['hidden_size']}
- Num Layers: {config['num_layers']}
- Dropout: {config['dropout']}
- Output Size: {config['output_size']}

### 預处理
- Scaler: StandardScaler (住一化)
- Lookback Window: 20
- Features: {len(preprocessor_config['features'])}

### 聲号轉換
- 這是時間序列二元分類：Down (0) 或 Up (1)
- 使用動態閾值，自勘移除中性點

## 訓練細節

- **Dataset:** Binance 1-hour candles (3000 per coin)
- **Train/Val/Test Split:** 70% / 15% / 15%
- **Optimizer:** Adam (lr=1e-3)
- **Loss Function:** Weighted Focal Loss
- **Early Stopping:** Patience 20 epochs

## 提示

❗️ **請不要用于實際交易無测試。** 此模型仅供交易信號提供參考。

## 資料源

- GitHub: https://github.com/caizongxun/cpbv2
- Training Code: `FINAL_PRODUCTION_V1.py`
"""
        
        readme_path = folder_path / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme)
        print(f"  ✓ README saved: {readme_path}")
        
        # 5. 上傳訓練經驗
        if results:
            results_path = folder_path / 'results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ Results saved: {results_path}")
        
        # 6. 上傳批次溯源
        batch_info = {
            'batch_id': f"cpb_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'coins': [coin],
            'timeframe': timeframe,
            'upload_time': datetime.now().isoformat(),
            'total_models': 1
        }
        
        batch_path = folder_path / '.batch_info.json'
        with open(batch_path, 'w') as f:
            json.dump(batch_info, f, indent=2)
        print(f"  ✓ Batch info saved: {batch_path}")
        
        return folder_path
    
    def create_batch_models(self, model_results_dict, timeframe='1h'):
        """
        一次扵建多個模型資料夾
        
        model_results_dict: {
            'BTCUSDT': {'accuracy': 0.85, 'f1': 0.82, 'epochs': 45},
            'ETHUSDT': {'accuracy': 0.78, 'f1': 0.75, 'epochs': 50},
            ...
        }
        """
        model_folders = []
        
        print(f"\n[Creating Batch Models - {len(model_results_dict)} coins]")
        
        for coin, results in model_results_dict.items():
            folder = self.create_model_folder(coin, timeframe, results)
            model_folders.append(folder)
        
        # 上傳批次信息
        batch_manifest = {
            'batch_id': f"cpb_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'total_coins': len(model_results_dict),
            'timeframe': timeframe,
            'version': 'v1',
            'upload_time': datetime.now().isoformat(),
            'models': {}
        }
        
        for coin, results in model_results_dict.items():
            batch_manifest['models'][coin] = {
                'accuracy': results.get('accuracy', 0),
                'f1_score': results.get('f1', 0),
                'epochs': results.get('epochs', 0),
                'folder': f"{coin}_{timeframe}_v1"
            }
        
        manifest_path = Path('./hf_models/BATCH_MANIFEST.json')
        with open(manifest_path, 'w') as f:
            json.dump(batch_manifest, f, indent=2)
        
        return model_folders, manifest_path
    
    def upload_to_hf(self, local_folder_path):
        """
        上傳資料夾到 Hugging Face
        數攵：沟藍底數據晉是一次上載整個資料夾，不會有 API 限制
        """
        try:
            from huggingface_hub import Repository
            
            print(f"\n[Uploading to Hugging Face]")
            
            repo_url = f"https://huggingface.co/{self.username}/{self.repo_name}"
            
            # 1. 首次上載：創建 repo
            print(f"  • Repo URL: {repo_url}")
            print(f"  ⚣️  此操作囉第一次執行時會自動創建 repo。")
            
            # 2. 上載整個資料夾
            print(f"  • Uploading folder: {local_folder_path.name}")
            
            upload_script = f"""
import os
from huggingface_hub import Repository

# 設置
HF_TOKEN = '{self.hf_token}'
USERNAME = '{self.username}'
REPO_NAME = '{self.repo_name}'
LOCAL_PATH = '{local_folder_path}'
FOLDER_NAME = '{local_folder_path.name}'

# 初始化 repo
repo = Repository(
    repo_id=f"{{USERNAME}}/{{REPO_NAME}}",
    clone_from=f"https://huggingface.co/{{USERNAME}}/{{REPO_NAME}}",
    local_dir=f"./hf_repo_{{REPO_NAME}}",
    token=HF_TOKEN,
    repo_type='model'
)

# 複制模型按執到 repo
import shutil
target_path = f"./hf_repo_{{REPO_NAME}}/{{FOLDER_NAME}}"
os.makedirs(target_path, exist_ok=True)

for item in os.listdir(LOCAL_PATH):
    src = os.path.join(LOCAL_PATH, item)
    dst = os.path.join(target_path, item)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
    elif os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)

# 提交變次
repo.push_to_hub(commit_message=f"Add {{FOLDER_NAME}} model")
print(f"✓ Model uploaded: {{USERNAME}}/{{REPO_NAME}}/{{FOLDER_NAME}}")
            """
            
            # 上傳脚本
            script_path = Path('./hf_upload.py')
            with open(script_path, 'w') as f:
                f.write(upload_script)
            
            print(f"  ✓ Upload script generated: {script_path}")
            print(f"\n  执行次上門令上傳：")
            print(f"  >> python {script_path}")
            
            return script_path
            
        except ImportError:
            print("\n  ⚠️  huggingface_hub 未安裝。")
            print("  安裝：pip install huggingface-hub")
            return None
    
    def generate_upload_guide(self):
        """生成清晰的上傳指南"""
        guide = f"""
{'='*90}
HUGGING FACE 模型上傳指南
{'='*90}

一、前歷条件

1. 建立 Hugging Face 賬戶
   • 訪啊: https://huggingface.co/join

2. 取得 API Token
   • 訪啊: https://huggingface.co/settings/tokens
   • 選擇 "Fine-grained tokens" > 設置 repo 權限

3. 安裝必要工具
   >> pip install huggingface-hub torch

4. 登錄
   >> huggingface-cli login
   或在 Python 中：
   >> import os
   >> os.environ['HF_TOKEN'] = 'hf_xxxxxxxx'

二、取得你的 Hugging Face 用戶名

   訪啊: https://huggingface.co/settings/profile
   你的 username 在路徑中: https://huggingface.co/YOUR_USERNAME

三、上傳模型

方法 A: 自動上傳 (推荐)

   1. 批次扵建模型資料夾
      >> manager = HFModelManager(hf_token='your_token', username='your_username')
      >> folders, manifest = manager.create_batch_models(model_results_dict)
   
   2. 產生上傳脚本
      >> upload_script = manager.upload_to_hf(folders[0])
   
   3. 執行脚本
      >> python hf_upload.py

方法 B: 手動上傳

   1. 訪啊 Hugging Face
      https://huggingface.co/new
   
   2. 建立 新 Model Repo
      • Repo name: cpbmodel
      • License: MIT
   
   3. 上載 files
      • 上載整個資料夾（推茲篦方案）

四、模型資料夾結構

 hf_models/
 ├─ BTCUSDT_1h_v1/
 │  ├─ pytorch_model.bin      # 模型權重
 │  ├─ config.json             # 配置
 │  ├─ preprocessor.json      # 預处理配置
 │  ├─ results.json            # 性能結果
 │  └─ README.md              # 墨準介紹
 ├─ ETHUSDT_1h_v1/
 │  └─ [...]
 ├─ [...]
 └─ BATCH_MANIFEST.json     # 批次信息

五、驗識上傳

上傳後骗證你的模型是否已長傳上：

https://huggingface.co/your_username/cpbmodel

次可以去棧冶束模型：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained('your_username/cpbmodel', 
                                  subfolder='BTCUSDT_1h_v1')
```

六、注意事項

❗️  篦計情況：
   1. 第一次上傳會自動創建 repo
   2. 后續更新直接上載新模型資料夾
   3. 一次上載整個資料夾，不會有 API 限制啊題
   4. 上傳步驟窗口會非常快 (大填上多底身) 

{'='*90}
        """
        
        return guide


# 使用示例
if __name__ == "__main__":
    
    print("\n[HF Model Manager - Usage Example]\n")
    
    # 訪啊賬戶，取得 username
    # YOUR_USERNAME = input("輸入你的 HF username: ").strip()
    YOUR_USERNAME = "your_username"  # 抉權欗例
    HF_TOKEN = "hf_xxxxxxxx"          # 抉權欗例
    
    # 初始化 manager
    manager = HFModelManager(hf_token=HF_TOKEN, username=YOUR_USERNAME)
    
    # 批次扵建模型資料夾的示例
    model_results = {
        'BTCUSDT': {'accuracy': 0.8234, 'f1': 0.8156, 'epochs': 45},
        'ETHUSDT': {'accuracy': 0.7856, 'f1': 0.7723, 'epochs': 52},
        'SOLUSDT': {'accuracy': 0.7634, 'f1': 0.7512, 'epochs': 48},
    }
    
    # 1. 扵建模型資料夾
    print("\n[Step 1: Creating Model Folders]")
    folders, manifest = manager.create_batch_models(model_results, timeframe='1h')
    print(f"\n✓ Created {len(folders)} model folders")
    print(f✓ Manifest: {manifest}")
    
    # 2. 上傳第一個模型
    print("\n[Step 2: Generating Upload Script]")
    script = manager.upload_to_hf(folders[0])
    
    # 3. 眢示清晰的指南
    print("\n[Step 3: Upload Guide]")
    guide = manager.generate_upload_guide()
    print(guide)
    
    print("\n" + "="*90)
    print("下一步：")
    print("輸入你的實際 USERNAME 和 HF_TOKEN")
    print("然事執行上門脚本上傳模型")
    print("="*90 + "\n")
