#!/usr/bin/env python3
"""
V3 Complete Training Pipeline for Colab
Steps: 1. Setup env 2. Download data from Binance 3. Train 40 models 4. Upload to HF
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V3CoLabPipeline:
    """Complete V3 training pipeline for Colab"""
    
    # 20 coins to train
    COINS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
        'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
        'MATICUSDT', 'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'ARBUSDT',
        'OPUSDT', 'STXUSDT', 'INJUSDT', 'LUNCUSDT', 'LUNAUSDT'
    ]
    
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
        logger.info("STEP 1: Setting up environment")
        logger.info("="*80)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("GPU not available. Training will be slower.")
        
        # Verify PyTorch and libraries
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Directories created:")
        logger.info(f"  Data: {self.data_dir}")
        logger.info(f"  Models: {self.models_dir}")
        logger.info(f"  Results: {self.results_dir}")
    
    def step_2_download_binance_data(self, limit: int = 3000):
        """Step 2: Download 3000 candles from Binance for each coin-timeframe pair"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Downloading data from Binance")
        logger.info("="*80)
        
        try:
            from binance.client import Client
        except ImportError:
            logger.error("python-binance not installed. Installing...")
            os.system('pip install python-binance -q')
            from binance.client import Client
        
        # Initialize client (public API, no key needed for historical data)
        client = Client()
        
        total_pairs = len(self.COINS) * len(self.TIMEFRAMES)
        completed = 0
        failed = []
        
        for coin in self.COINS:
            for timeframe in self.TIMEFRAMES:
                try:
                    completed += 1
                    logger.info(f"[{completed}/{total_pairs}] Downloading {coin} {timeframe}...")
                    
                    # Download klines
                    klines = client.get_historical_klines(
                        symbol=coin,
                        interval=timeframe,
                        limit=limit
                    )
                    
                    if not klines:
                        raise ValueError(f"No data received for {coin} {timeframe}")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore'
                    ])
                    
                    # Convert types
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Save
                    csv_path = self.data_dir / f"{coin}_{timeframe}.csv"
                    df[['Open', 'High', 'Low', 'Close', 'Volume']].to_csv(csv_path, index=False)
                    logger.info(f"  Saved to {csv_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to download {coin} {timeframe}: {e}")
                    failed.append(f"{coin}_{timeframe}")
        
        logger.info(f"\nData download summary:")
        logger.info(f"  Successful: {total_pairs - len(failed)}/{total_pairs}")
        logger.info(f"  Failed: {len(failed)}/{total_pairs}")
        if failed:
            logger.warning(f"  Failed pairs: {failed}")
        
        return len(failed) == 0
    
    def step_3_train_models(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """Step 3: Train 40 models (20 coins × 2 timeframes)"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Training models")
        logger.info("="*80)
        logger.info(f"Total models to train: {len(self.COINS) * len(self.TIMEFRAMES)}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Import model components
        try:
            from v3_lstm_model import create_v3_model, V3TrainingConfig
            from v3_trainer import V3Trainer
            from v3_data_processor import V3DataProcessor
        except ImportError as e:
            logger.error(f"Failed to import model components: {e}")
            logger.info("Make sure v3_lstm_model.py, v3_trainer.py, v3_data_processor.py are in current directory")
            return False
        
        training_results = {}
        
        for idx, coin in enumerate(self.COINS, 1):
            for tf_idx, timeframe in enumerate(self.TIMEFRAMES, 1):
                pair_name = f"{coin}_{timeframe}"
                pair_idx = (idx-1)*len(self.TIMEFRAMES) + tf_idx
                total = len(self.COINS) * len(self.TIMEFRAMES)
                
                logger.info(f"\n[{pair_idx}/{total}] Training {pair_name}...")
                
                try:
                    # Load data
                    csv_path = self.data_dir / f"{pair_name}.csv"
                    if not csv_path.exists():
                        logger.warning(f"Data file not found: {csv_path}")
                        training_results[pair_name] = {'status': 'skipped', 'reason': 'data_not_found'}
                        continue
                    
                    df = pd.read_csv(csv_path)
                    
                    # Process data
                    processor = V3DataProcessor()
                    df = processor.calculate_technical_indicators(df)
                    X, y = processor.prepare_sequences(df, seq_length=60, prediction_horizon=1)
                    
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
                    trainer = V3Trainer(
                        model,
                        device=self.device,
                        learning_rate=learning_rate
                    )
                    
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
                    
                    logger.info(f"  Best val loss: {result['best_val_loss']:.6f}")
                    logger.info(f"  Best epoch: {result['best_epoch']}")
                    
                    # Clear cache
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Training failed for {pair_name}: {e}")
                    training_results[pair_name] = {'status': 'failed', 'error': str(e)}
        
        # Save results
        results_path = self.results_dir / 'v3_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"\nTraining results saved to {results_path}")
        
        return True
    
    def step_4_get_hf_token(self):
        """Step 4: Get HuggingFace token from user"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: HuggingFace token")
        logger.info("="*80)
        
        hf_token = input("\nEnter your HuggingFace token (get from https://huggingface.co/settings/tokens): ").strip()
        
        if not hf_token:
            logger.error("HuggingFace token is required")
            return None
        
        return hf_token
    
    def step_5_upload_to_hf(self, hf_token: str):
        """Step 5: Upload all models to HuggingFace"""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Uploading to HuggingFace")
        logger.info("="*80)
        
        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.error("huggingface-hub not installed. Installing...")
            os.system('pip install huggingface-hub -q')
            from huggingface_hub import HfApi
        
        api = HfApi()
        repo_id = "zongowo111/cpb-models"
        
        # Create v3 folder structure
        logger.info("Uploading model files...")
        
        successful = 0
        failed = []
        
        for model_file in self.models_dir.glob("v3_model_*.pt"):
            try:
                logger.info(f"Uploading {model_file.name}...")
                
                api.upload_file(
                    path_or_fileobj=str(model_file),
                    path_in_repo=f"v3/{model_file.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token
                )
                
                successful += 1
                logger.info(f"  Uploaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to upload {model_file.name}: {e}")
                failed.append(model_file.name)
        
        logger.info(f"\nUpload summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {len(failed)}")
        if failed:
            logger.warning(f"  Failed files: {failed}")
        
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
                logger.info("Training results uploaded")
        except Exception as e:
            logger.warning(f"Failed to upload training results: {e}")
        
        logger.info(f"\nModels available at: https://huggingface.co/datasets/{repo_id}")
        return len(failed) == 0
    
    def run_full_pipeline(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """Run complete pipeline"""
        logger.info("\n" + "#"*80)
        logger.info("# V3 CRYPTOCURRENCY PRICE PREDICTION TRAINING PIPELINE")
        logger.info("#"*80)
        
        # Step 1
        self.step_1_setup_environment()
        
        # Step 2
        if not self.step_2_download_binance_data(limit=3000):
            logger.warning("Some data downloads failed. Continuing with available data...")
        
        # Step 3
        if not self.step_3_train_models(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate):
            logger.error("Training step failed")
            return False
        
        # Step 4
        hf_token = self.step_4_get_hf_token()
        if not hf_token:
            logger.error("Cannot proceed without HuggingFace token")
            return False
        
        # Step 5
        if not self.step_5_upload_to_hf(hf_token):
            logger.warning("Some uploads failed")
        
        logger.info("\n" + "#"*80)
        logger.info("# PIPELINE COMPLETED")
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
