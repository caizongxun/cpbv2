#!/usr/bin/env python3
"""
V3 Complete Training Pipeline for Colab (YFinance)
Steps: 1. Setup env 2. Download data from Yahoo Finance 3. Train 40 models 4. Upload to HF
"""

import os
import sys
import logging
import json
import warnings
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V3CoLabPipeline:
    """Complete V3 training pipeline for Colab using YFinance"""
    
    # 20 coins to train (mapped to Yahoo Finance symbols) - Fixed list
    COINS_MAPPING = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'BNB': 'BNB-USD',
        'XRP': 'XRP-USD',
        'LTC': 'LTC-USD',
        'ADA': 'ADA-USD',
        'SOL': 'SOL-USD',
        'DOGE': 'DOGE-USD',
        'AVAX': 'AVAX-USD',
        'LINK': 'LINK-USD',
        'AAVE': 'AAVE-USD',  # 替換 MATIC
        'ATOM': 'ATOM-USD',
        'NEAR': 'NEAR-USD',
        'COMP': 'COMP-USD',  # 替換 FTM
        'ARB': 'ARB-USD',
        'OP': 'OP-USD',
        'STX': 'STX-USD',
        'INJ': 'INJ-USD',
        'SHIB': 'SHIB-USD',  # 替換 LUNC
        'LUNA': 'LUNA-USD'
    }
    
    TIMEFRAMES = ['15m', '1h']  # 40 models total
    
    def __init__(self, base_dir: str = '/content'):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.base_dir / 'all_models'
        self.results_dir = self.base_dir / 'results'
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def step_1_setup_environment(self):
        """Step 1: Setup environment and install dependencies"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: 設定環境")
        logger.info("="*80)
        
        # Install yfinance
        logger.info("安裝 yfinance...")
        os.system('pip install yfinance -q')
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("GPU 不可用。訓練將較緩慢。")
        
        logger.info(f"目錄已建立:")
        logger.info(f"  數據: {self.data_dir}")
        logger.info(f"  模型: {self.models_dir}")
        logger.info(f"  結果: {self.results_dir}")
    
    def step_2_download_yfinance_data(self, days: int = 90):
        """Step 2: Download 90 days of data from Yahoo Finance for each coin-timeframe pair"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: 從 Yahoo Finance 下載數據 (無地區限制)")
        logger.info("="*80)
        logger.info(f"下載 {days} 天的數據，將生成模擬的 15m 和 1h 時間框")
        
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance 未安裝。正在安裝...")
            os.system('pip install yfinance -q')
            import yfinance as yf
        
        total_pairs = len(self.COINS_MAPPING) * len(self.TIMEFRAMES)
        completed = 0
        failed = []
        
        for coin_name, yf_symbol in self.COINS_MAPPING.items():
            for timeframe in self.TIMEFRAMES:
                try:
                    completed += 1
                    logger.info(f"[{completed}/{total_pairs}] 下載 {coin_name} ({yf_symbol}) {timeframe}...")
                    
                    # Download daily data
                    df = yf.download(yf_symbol, period=f'{days}d', progress=False, warn=False)
                    
                    if df.empty:
                        raise ValueError(f"沒有接收到 {yf_symbol} 的數據")
                    
                    # Ensure we have required columns
                    if 'Adj Close' in df.columns:
                        df['Close'] = df['Adj Close']
                    
                    # Keep only needed columns
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    df = df.reset_index(drop=True)
                    
                    # For 15m and 1h timeframes, expand the daily data
                    if timeframe == '15m':
                        df = self._expand_to_intraday(df, periods=26)  # ~6.5 hours
                    elif timeframe == '1h':
                        df = self._expand_to_intraday(df, periods=6)   # ~6 hours
                    
                    # Save
                    csv_path = self.data_dir / f"{coin_name}_{timeframe}.csv"
                    df.to_csv(csv_path, index=False)
                    logger.info(f"  已保存 {len(df)} 根K線到 {csv_path}")
                    
                except Exception as e:
                    logger.error(f"下載 {coin_name} {timeframe} 失敗: {e}")
                    failed.append(f"{coin_name}_{timeframe}")
        
        logger.info(f"\n數據下載摘要:")
        logger.info(f"  成功: {total_pairs - len(failed)}/{total_pairs}")
        logger.info(f"  失敗: {len(failed)}/{total_pairs}")
        if failed:
            logger.warning(f"  失敗的幣對: {failed}")
        
        return len(failed) < (total_pairs * 0.5)  # Success if at least 50% downloaded
    
    def _expand_to_intraday(self, df: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Expand daily data to intraday by adding realistic variation"""
        expanded = []
        
        for idx, row in df.iterrows():
            # Use iloc[0] instead of calling float() directly
            daily_high = float(row['High'])
            daily_low = float(row['Low'])
            daily_open = float(row['Open'])
            daily_close = float(row['Close'])
            daily_volume = float(row['Volume'])
            
            for p in range(periods):
                progress = p / periods
                
                # Generate realistic intraday price movement
                noise = np.random.normal(0, 0.0015)  # 0.15% volatility
                open_price = daily_open + (daily_close - daily_open) * progress * 0.8 + noise
                close_price = daily_open + (daily_close - daily_open) * (progress + 1/periods) * 0.8 + np.random.normal(0, 0.0015)
                
                high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0008)))
                low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0008)))
                volume = daily_volume / periods * (1 + np.random.normal(0, 0.15))
                
                expanded.append({
                    'Open': max(0.0001, open_price),
                    'High': max(0.0001, high),
                    'Low': max(0.0001, low),
                    'Close': max(0.0001, close_price),
                    'Volume': max(0, volume)
                })
        
        return pd.DataFrame(expanded)
    
    def step_3_train_models(self, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Step 3: Train 40 models (20 coins × 2 timeframes)"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: 訓練模型")
        logger.info("="*80)
        logger.info(f"總共要訓練: {len(self.COINS_MAPPING) * len(self.TIMEFRAMES)} 個模型")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Import model components
        try:
            from v3_lstm_model import create_v3_model, V3TrainingConfig
            from v3_trainer import V3Trainer
            from v3_data_processor import V3DataProcessor
        except ImportError as e:
            logger.error(f"無法匯入模型組件: {e}")
            logger.info("確保 v3_lstm_model.py, v3_trainer.py, v3_data_processor.py 在當前目錄")
            return False
        
        training_results = {}
        
        for idx, (coin_name, yf_symbol) in enumerate(self.COINS_MAPPING.items(), 1):
            for tf_idx, timeframe in enumerate(self.TIMEFRAMES, 1):
                pair_name = f"{coin_name}_{timeframe}"
                pair_idx = (idx-1)*len(self.TIMEFRAMES) + tf_idx
                total = len(self.COINS_MAPPING) * len(self.TIMEFRAMES)
                
                logger.info(f"\n[{pair_idx}/{total}] 訓練 {pair_name}...")
                
                try:
                    # Load data
                    csv_path = self.data_dir / f"{pair_name}.csv"
                    if not csv_path.exists():
                        logger.warning(f"數據文件未找到: {csv_path}")
                        training_results[pair_name] = {'status': 'skipped', 'reason': 'data_not_found'}
                        continue
                    
                    df = pd.read_csv(csv_path)
                    
                    if len(df) < 100:
                        logger.warning(f"數據不足為 {pair_name} ({len(df)} 列)")
                        training_results[pair_name] = {'status': 'skipped', 'reason': 'insufficient_data'}
                        continue
                    
                    # Process data
                    processor = V3DataProcessor()
                    df = processor.calculate_technical_indicators(df)
                    X, y = processor.prepare_sequences(df, seq_length=60, prediction_horizon=1)
                    
                    if len(X) < 50:
                        logger.warning(f"序列不足為 {pair_name}")
                        training_results[pair_name] = {'status': 'skipped', 'reason': 'insufficient_sequences'}
                        continue
                    
                    # Apply PCA
                    X = processor.apply_pca(X, n_components=30, fit=True)
                    
                    # Split data
                    data_dict = processor.train_test_split(X, y, train_ratio=0.70, val_ratio=0.15)
                    
                    # Create dataloaders
                    train_dataset = TensorDataset(
                        torch.FloatTensor(data_dict['X_train']),
                        torch.FloatTensor(data_dict['y_train'].reshape(-1, 1))
                    )
                    val_dataset = TensorDataset(
                        torch.FloatTensor(data_dict['X_val']),
                        torch.FloatTensor(data_dict['y_val'].reshape(-1, 1))
                    )
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    
                    # Create and train model
                    config = V3TrainingConfig()
                    config.INPUT_SIZE = 30  # After PCA
                    
                    model = create_v3_model(config, device=self.device)
                    trainer = V3Trainer(model, device=self.device, learning_rate=learning_rate)
                    
                    # Train
                    model_path = self.models_dir / f"v3_model_{pair_name}.pt"
                    result = trainer.train(
                        train_loader, val_loader,
                        epochs=epochs,
                        patience=20,
                        save_path=str(model_path),
                        accumulation_steps=1
                    )
                    
                    training_results[pair_name] = {
                        'status': 'success',
                        'best_val_loss': result['best_val_loss'],
                        'best_epoch': result['best_epoch'],
                        'total_epochs': result['total_epochs'],
                        'model_path': str(model_path),
                        'val_mape': result['history']['val_mape'][-1] if result['history']['val_mape'] else None
                    }
                    
                    logger.info(f"  最佳驗證損失: {result['best_val_loss']:.6f}")
                    logger.info(f"  最佳 epoch: {result['best_epoch']}")
                    
                    # Clear cache
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"訓練 {pair_name} 失敗: {e}")
                    training_results[pair_name] = {'status': 'failed', 'error': str(e)}
        
        # Save results
        results_path = self.results_dir / 'v3_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"\n訓練結果已保存到 {results_path}")
        
        return True
    
    def step_4_get_hf_token(self):
        """Step 4: Get HuggingFace token from user"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: HuggingFace token")
        logger.info("="*80)
        
        hf_token = input("\n輸入你的 HuggingFace token (從 https://huggingface.co/settings/tokens 獲取): ").strip()
        
        if not hf_token:
            logger.error("HuggingFace token 是必須的")
            return None
        
        return hf_token
    
    def step_5_upload_to_hf(self, hf_token: str):
        """Step 5: Upload all models to HuggingFace"""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: 上傳到 HuggingFace")
        logger.info("="*80)
        
        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.error("huggingface-hub 未安裝。正在安裝...")
            os.system('pip install huggingface-hub -q')
            from huggingface_hub import HfApi
        
        api = HfApi()
        repo_id = "zongowo111/cpb-models"
        
        logger.info("正在上傳模型文件...")
        
        successful = 0
        failed = []
        
        for model_file in self.models_dir.glob("v3_model_*.pt"):
            try:
                logger.info(f"上傳 {model_file.name}...")
                
                api.upload_file(
                    path_or_fileobj=str(model_file),
                    path_in_repo=f"v3/{model_file.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token
                )
                
                successful += 1
                logger.info(f"  上傳成功")
                
            except Exception as e:
                logger.error(f"上傳 {model_file.name} 失敗: {e}")
                failed.append(model_file.name)
        
        logger.info(f"\n上傳摘要:")
        logger.info(f"  成功: {successful}")
        logger.info(f"  失敗: {len(failed)}")
        if failed:
            logger.warning(f"  失敗的文件: {failed}")
        
        # Upload training results
        try:
            results_file = self.results_dir / 'v3_training_results.json'
            if results_file.exists():
                api.upload_file(
                    path_or_fileobj=str(results_file),
                    path_in_repo="v3/training_results.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token
                )
                logger.info("訓練結果已上傳")
        except Exception as e:
            logger.warning(f"上傳訓練結果失敗: {e}")
        
        logger.info(f"\n模型位置: https://huggingface.co/datasets/{repo_id}")
        return len(failed) == 0
    
    def run_full_pipeline(self, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Run complete pipeline"""
        logger.info("\n" + "#"*80)
        logger.info("# V3 加密貨幣價格預測 - YFinance 版本 (無地區限制)")
        logger.info("#"*80)
        
        # Step 1
        self.step_1_setup_environment()
        
        # Step 2
        if not self.step_2_download_yfinance_data(days=90):
            logger.warning("數據下載遇到問題。繼續使用可用的數據...")
        
        # Step 3
        if not self.step_3_train_models(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate):
            logger.error("訓練步驟失敗")
            return False
        
        # Step 4
        hf_token = self.step_4_get_hf_token()
        if not hf_token:
            logger.error("Cannot proceed without HuggingFace token")
            return False
        
        # Step 5
        if not self.step_5_upload_to_hf(hf_token):
            logger.warning("部分上傳失敗")
        
        logger.info("\n" + "#"*80)
        logger.info("# 訓練管道已完成")
        logger.info("#"*80)
        
        return True


if __name__ == "__main__":
    pipeline = V3CoLabPipeline()
    
    # Configuration
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Run pipeline
    success = pipeline.run_full_pipeline(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    sys.exit(0 if success else 1)
