#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: 模型組織自動化指欿
將訓練好的模型整理到 all_models 資料夾
並重新命名添加 v1 版本標記
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import torch
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*90)
print("MODEL ORGANIZATION TOOL - V1")
print("="*90)

class ModelOrganizer:
    def __init__(self, source_dir='./trained_models', 
                 target_dir='./all_models',
                 timeframe='1h',
                 version='v1'):
        """
        source_dir: 訓練模型的來源位置
        target_dir: 整理後的目標位置
        timeframe: 時間引 (預設 '1h')
        version: 版本號 (預設 'v1')
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.timeframe = timeframe
        self.version = version
        self.organized_models = {}
        
    def find_model_files(self):
        """
        搛為所有模型檔案
        """
        print(f"\n[STEP 1] 搛為模型檔案")
        print(f"  搋晕摊: {self.source_dir}")
        
        # 搛為 .pth, .bin, .pt 檔案
        model_extensions = ['*.pth', '*.bin', '*.pt']
        model_files = []
        
        for ext in model_extensions:
            model_files.extend(self.source_dir.glob(ext))
        
        if not model_files:
            print(f"  [WARNING] 未找到模型檔案！")
            print(f"  請確保你的模型在 {self.source_dir}")
            return []
        
        print(f"  [OK] 找到 {len(model_files)} 個模型檔案")
        for f in model_files:
            print(f"    - {f.name}")
        
        return model_files
    
    def extract_coin_name(self, filename):
        """
        從檔案名提取幣種名
        例: BTCUSDT_model.pth -> BTCUSDT
        """
        stem = filename.stem.lower()
        
        # 頁為幣種名剋整成大寫
        if '_model' in stem:
            coin = stem.replace('_model', '').upper()
        elif '_lstm' in stem:
            coin = stem.replace('_lstm', '').upper()
        elif '_' in stem:
            coin = stem.split('_')[0].upper()
        else:
            coin = stem.upper()
        
        return coin
    
    def create_model_folder(self, coin_name, model_file):
        """
        為每個三幣建立資料夾並複製模型
        """
        # 建立資料夾名称
        folder_name = f"{coin_name}_{self.timeframe}_{self.version}"
        folder_path = self.target_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # 新檔案名
        new_filename = f"{coin_name}_{self.timeframe}_{self.version}{model_file.suffix}"
        new_filepath = folder_path / new_filename
        
        # 複製模型檔
        shutil.copy2(model_file, new_filepath)
        
        return folder_path, new_filepath
    
    def organize_models(self, model_results=None):
        """
        整理所有模型
        
        model_results: 可選的性能結果
        {
            'BTCUSDT': {'accuracy': 0.85, 'f1': 0.82, 'epochs': 45},
            ...
        }
        """
        print(f"\n[STEP 2] 整理模型")
        print(f"  目標摊: {self.target_dir}")
        
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        model_files = self.find_model_files()
        if not model_files:
            return False
        
        for model_file in model_files:
            coin_name = self.extract_coin_name(model_file)
            folder_path, new_filepath = self.create_model_folder(coin_name, model_file)
            
            print(f"  [OK] {coin_name}")
            print(f"       檔案: {new_filepath.name}")
            print(f"       位置: {folder_path}")
            
            self.organized_models[coin_name] = {
                'folder': str(folder_path),
                'model_file': str(new_filepath),
                'original_file': str(model_file)
            }
            
            # 新增配置檔 (Placeholder)
            self._create_config_file(folder_path, coin_name, model_results)
        
        print(f"\n  [OK] 整理完成! 共 {len(self.organized_models)} 個模型")
        return True
    
    def _create_config_file(self, folder_path, coin_name, model_results):
        """
        為每個模型建立配置檔
        """
        config = {
            'coin': coin_name,
            'timeframe': self.timeframe,
            'version': self.version,
            'organized_at': datetime.now().isoformat(),
            'model_type': 'LSTM',
            'framework': 'PyTorch'
        }
        
        if model_results and coin_name in model_results:
            config.update(model_results[coin_name])
        
        config_path = folder_path / f"{coin_name}_{self.timeframe}_{self.version}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def create_batch_manifest(self, model_results=None):
        """
        建立批次清冗
        """
        print(f"\n[STEP 3] 建立批次清増")
        
        batch_manifest = {
            'batch_id': f"cpb_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'total_coins': len(self.organized_models),
            'timeframe': self.timeframe,
            'version': self.version,
            'organized_at': datetime.now().isoformat(),
            'models': {}
        }
        
        for coin, info in self.organized_models.items():
            batch_manifest['models'][coin] = {
                'folder': info['folder'],
                'model_file': info['model_file'],
            }
            
            if model_results and coin in model_results:
                batch_manifest['models'][coin].update(model_results[coin])
        
        manifest_path = self.target_dir / 'BATCH_MANIFEST.json'
        with open(manifest_path, 'w') as f:
            json.dump(batch_manifest, f, indent=2, ensure_ascii=False)
        
        print(f"  [OK] 清劗位置: {manifest_path}")
        return manifest_path
    
    def generate_summary_report(self):
        """
        產生整理摘要
        """
        print(f"\n[STEP 4] 產生案要")
        
        report = f"""
{'='*90}
MODEL ORGANIZATION SUMMARY
{'='*90}

Organization Details:
  - Source Directory: {self.source_dir}
  - Target Directory: {self.target_dir}
  - Timeframe: {self.timeframe}
  - Version: {self.version}
  - Timestamp: {datetime.now().isoformat()}

Organized Models: {len(self.organized_models)}

"""
        
        for coin, info in sorted(self.organized_models.items()):
            report += f"  [{coin}]\n"
            report += f"    Folder: {info['folder']}\n"
            report += f"    Model:  {Path(info['model_file']).name}\n"
        
        report += f"""
{'='*90}
Folder Structure:
{'='*90}

all_models/
"""
        
        for i, coin in enumerate(sorted(self.organized_models.keys()), 1):
            prefix = "\u2514\u2500\u2500" if i == len(self.organized_models) else "\u251c\u2500\u2500"
            report += f"{prefix} {coin}_{self.timeframe}_{self.version}/\n"
            report += f"    \u251c\u2500\u2500 {coin}_{self.timeframe}_{self.version}.bin\n"
            report += f"    \u251c\u2500\u2500 {coin}_{self.timeframe}_{self.version}_config.json\n"
            report += f"    \u2514\u2500\u2500 README.md\n"
        
        report += f"\u2514\u2500\u2500 BATCH_MANIFEST.json\n"
        
        report += f"""
{'='*90}
Next Steps:
{'='*90}

1. Set HF Token:
   >> import os
   >> os.environ['HF_TOKEN'] = 'hf_xxxxxxxxxxxxxxxx'
   >> os.environ['HF_USERNAME'] = 'your_username'

2. Upload to Hugging Face:
   >> from HF_MODEL_UPLOADER import HFModelManager
   >> manager = HFModelManager(hf_token=os.environ['HF_TOKEN'],
   >>                          username=os.environ['HF_USERNAME'])
   >> folders, manifest = manager.create_batch_models(model_results, timeframe='{self.timeframe}')
   >> script = manager.upload_to_hf(folders[0])
   >> # python hf_upload.py

{'='*90}
        """
        
        print(report)
        
        # 保存報告
        report_path = self.target_dir / 'ORGANIZATION_REPORT.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[OK] 報告已保存: {report_path}")
        
        return report
    
    def run(self, model_results=None):
        """
        執行完整整理流程
        """
        print(f"\n[Starting Model Organization Process]")
        
        # 整理模型
        if not self.organize_models(model_results):
            return False
        
        # 建立清劗
        self.create_batch_manifest(model_results)
        
        # 產生報告
        self.generate_summary_report()
        
        print(f"\n[OK] 整理完成!")
        return True


if __name__ == "__main__":
    
    print("\n[Model Organizer - Usage Example]\n")
    
    # 批次結果 (來自訓練)
    example_results = {
        'BTCUSDT': {'accuracy': 0.8234, 'f1': 0.8156, 'epochs': 45},
        'ETHUSDT': {'accuracy': 0.7856, 'f1': 0.7723, 'epochs': 52},
        'SOLUSDT': {'accuracy': 0.7634, 'f1': 0.7512, 'epochs': 48},
        'BNBUSDT': {'accuracy': 0.8421, 'f1': 0.8267, 'epochs': 50},
        'AVAXUSDT': {'accuracy': 0.7923, 'f1': 0.7845, 'epochs': 46},
    }
    
    # 初始化組織器
    organizer = ModelOrganizer(
        source_dir='./trained_models',  # 你的檔案在這壽
        target_dir='./all_models',       # 整理後放到這墿
        timeframe='1h',
        version='v1'
    )
    
    # 執行整理
    organizer.run(model_results=example_results)
    
    print(f"\n[Next: Upload to Hugging Face]")
    print(f"\n1. Go to: https://huggingface.co/settings/tokens")
    print(f"2. Create new token")
    print(f"3. Copy your token and set it in Python:")
    print(f"   import os")
    print(f"   os.environ['HF_TOKEN'] = 'hf_your_token'")
    print(f"   os.environ['HF_USERNAME'] = 'your_username'")
    print(f"\n4. Then run HF_MODEL_UPLOADER.py")
    print("\n" + "="*90 + "\n")
