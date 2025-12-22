#!/usr/bin/env python3
"""
Complete training pipeline for CPB v2 LSTM models.
Usage: python scripts/train_models.py
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector import BinanceDataCollector
from src.feature_engineer import FeatureEngineer
from src.data_preprocessor import DataPreprocessor
from src.model import LSTMModel
from src.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration files."""
    with open('config/coins.json', 'r') as f:
        coins_config = json.load(f)
    
    with open('config/model_params.json', 'r') as f:
        model_config = json.load(f)
    
    return coins_config, model_config


def download_data(coins, timeframes, collector):
    """Download data for all coins and timeframes."""
    logger.info(f"Downloading data for {len(coins)} coins, {len(timeframes)} timeframes...")
    
    all_data = {}
    failed = []
    
    for i, coin in enumerate(coins):
        logger.info(f"[{i+1}/{len(coins)}] {coin}")
        coin_data = {}
        
        for timeframe in timeframes:
            try:
                df = collector.get_historical_klines(
                    symbol=coin,
                    interval=timeframe,
                    limit=3000
                )
                
                if BinanceDataCollector.validate_data(df, min_candles=3000):
                    coin_data[timeframe] = df
                    
                    # Save to CSV
                    os.makedirs('data/raw', exist_ok=True)
                    df.to_csv(f'data/raw/{coin}_{timeframe}.csv', index=False)
                else:
                    logger.warning(f"Data validation failed: {coin} {timeframe}")
                    failed.append(f"{coin}_{timeframe}")
            
            except Exception as e:
                logger.error(f"Error downloading {coin} {timeframe}: {e}")
                failed.append(f"{coin}_{timeframe}")
        
        if coin_data:
            all_data[coin] = coin_data
    
    logger.info(f"Downloaded {sum([len(v) for v in all_data.values()])} datasets")
    if failed:
        logger.warning(f"Failed: {failed}")
    
    return all_data, failed


def train_model(coin_symbol, timeframe, df, model_config, device):
    """Train a single model."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Training {coin_symbol} {timeframe}")
    logger.info(f"{'='*50}")
    
    try:
        # Feature Engineering
        fe = FeatureEngineer(df)
        df_features = fe.calculate_all_indicators()
        feature_cols = fe.get_feature_columns()
        logger.info(f"Total features: {len(feature_cols)}")
        
        # Preprocessing
        preprocessor = DataPreprocessor(df_features)
        preprocessor.remove_nans()
        preprocessor.select_features(feature_cols, n_components=30)
        preprocessor.normalize_features()
        
        # Create sequences
        X, y = preprocessor.create_sequences()
        data_split = preprocessor.split_train_val_test(X, y)
        
        # Create dataloaders
        X_train = torch.FloatTensor(data_split['X_train']).to(device)
        y_train = torch.FloatTensor(data_split['y_train']).to(device)
        X_val = torch.FloatTensor(data_split['X_val']).to(device)
        y_val = torch.FloatTensor(data_split['y_val']).to(device)
        
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Build model
        model = LSTMModel.create_model(model_config, device=device)
        logger.info(f"Model parameters: {model.count_parameters():,}")
        
        # Train
        os.makedirs('models', exist_ok=True)
        save_path = f'models/{coin_symbol}_{timeframe}.pt'
        
        trainer = Trainer(model, device=device)
        history = trainer.train(
            train_loader, val_loader,
            epochs=50,
            learning_rate=0.001,
            patience=15,
            save_path=save_path
        )
        
        logger.info(f"Best validation loss: {history['best_val_loss']:.6f}")
        logger.info(f"Training completed in {history['total_epochs']} epochs")
        
        return {
            'status': 'success',
            'best_val_loss': float(history['best_val_loss']),
            'best_epoch': history['best_epoch'],
            'total_epochs': history['total_epochs'],
            'model_path': save_path,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main training pipeline."""
    start_time = time.time()
    
    # Load config
    logger.info("Loading configuration...")
    coins_config, model_config = load_config()
    
    coins = [coin['symbol'] for coin in coins_config['coins']]
    timeframes = ['15m', '1h']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Device: {device}")
    logger.info(f"Total coins: {len(coins)}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Total models: {len(coins) * len(timeframes)}")
    
    # Phase 1: Download data
    collector = BinanceDataCollector()
    all_data, failed_data = download_data(coins, timeframes, collector)
    
    # Phase 2: Train models
    training_results = {}
    
    for coin_symbol in list(all_data.keys())[:5]:  # Train first 5 coins for speed
        for timeframe in all_data[coin_symbol].keys():
            df = all_data[coin_symbol][timeframe]
            
            result = train_model(
                coin_symbol, timeframe, df,
                model_config, device
            )
            
            training_results[f"{coin_symbol}_{timeframe}"] = result
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Summary
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in training_results.values() if r['status'] == 'success')
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Total time: {elapsed_time / 60:.1f} minutes")
    logger.info(f"Successful: {successful}/{len(training_results)}")
    logger.info(f"Failed: {len(training_results) - successful}/{len(training_results)}")
    
    if successful > 0:
        best_model = min(
            [k for k, v in training_results.items() if v['status'] == 'success'],
            key=lambda k: training_results[k]['best_val_loss']
        )
        logger.info(f"Best model: {best_model} (loss: {training_results[best_model]['best_val_loss']:.6f})")
    
    logger.info(f"\nResults saved to results/training_results.json")
    
    return training_results


if __name__ == "__main__":
    main()
