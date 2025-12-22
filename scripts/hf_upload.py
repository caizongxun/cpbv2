#!/usr/bin/env python3
"""
Upload trained models to Hugging Face Hub.
Usage: HF_TOKEN=your_token python scripts/hf_upload.py
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import HfApi, login, Repository
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HFUploader:
    """Upload models to Hugging Face Hub."""
    
    def __init__(self, repo_id="caizongxun/cpb", token=None):
        self.repo_id = repo_id
        self.token = token or os.environ.get('HF_TOKEN')
        self.api = HfApi()
        
        if not self.token:
            raise ValueError("HF_TOKEN not set. Set via environment variable or pass token argument.")
        
        logger.info(f"Initialized HF uploader for repo: {repo_id}")
    
    def create_model_card(self, coin_symbol, timeframe, results):
        """Create a model card for the HF Hub."""
        model_card = f"""---
license: mit
language: en
tags:
- cryptocurrency
- lstm
- price-prediction
- binance
- {coin_symbol.lower()}
metrics:
- validation_loss
datasets:
- binance-klines
---

# CPB v2: {coin_symbol} {timeframe} LSTM Prediction Model

Bidirectional LSTM model for predicting {coin_symbol} price movements at {timeframe} timeframe.

## Model Details

- **Architecture**: Bidirectional LSTM (2 layers)
- **Input Size**: 30 features (technical indicators)
- **Lookback Period**: 60 timesteps
- **Batch Size**: 32
- **Training Time**: < 2 hours (Google Colab Free GPU)

## Hyperparameters

- **LSTM Units**: [96, 64]
- **Dropout**: 0.2 (LSTM), 0.1 (Dense)
- **Dense Units**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: MSE
- **Epochs**: 50 (with early stopping)

## Performance

- **Best Validation Loss**: {results.get('best_val_loss', 'N/A')}
- **Best Epoch**: {results.get('best_epoch', 'N/A')}
- **Total Epochs**: {results.get('total_epochs', 'N/A')}

## Technical Indicators (35+)

The model uses the following technical indicators:

- **Price & Volume**: open, high, low, close, volume, hl2, hlc3
- **Moving Averages**: SMA (10,20,50,100,200), EMA (10,20,50,100,200)
- **Momentum**: RSI (14,21), MACD, Momentum (5), ROC (12), Stochastic
- **Volatility**: Bollinger Bands, ATR (14)
- **Trend**: ADX, DI+/-, Keltner Channels, NATR
- **Volume**: OBV, CMF (20), MFI (14), VPT
- **Change Indicators**: Price Change, Volume Change, Close Change

## Data

- **Source**: Binance REST API
- **Timeframe**: {timeframe}
- **Candles**: 3000+ per dataset
- **Period**: ~3 months of historical data
- **Normalization**: MinMaxScaler (0-1 range)

## Usage

```python
import torch
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("caizongxun/cpb")
model.eval()

# Make prediction
with torch.no_grad():
    output = model(input_tensor)  # shape: (batch_size, 60, 30)
    prediction = output.numpy()
```

## Training Configuration

- **Framework**: PyTorch
- **Python**: 3.8+
- **Dependencies**: torch, pandas, numpy, scikit-learn, ta
- **Hardware**: GPU (T4 on Colab) or CPU
- **Batch Size**: 32
- **Memory**: ~2GB GPU RAM

## Limitations

- Trained on historical Binance data
- No real-time updates
- Cryptocurrency markets are highly volatile
- Past performance does not guarantee future results
- Should not be used as sole basis for trading decisions

## License

MIT License

## Citation

```
@software{{cpbv2,
  title={{CPB v2: Cryptocurrency Price Prediction LSTM}},
  author={{Cai, Zongxun}},
  year={{2025}},
  url={{https://github.com/caizongxun/cpbv2}}
}}
```

## References

- LSTM: Hochreiter & Schmidhuber (1997)
- Technical Analysis: Wilder (1978), Bollinger (1984)
- Feature Selection: Guyon & Elisseeff (2003)
- Time-series validation: Hyndman & Athanasopoulos (2021)

---

**Last Updated**: {datetime.now().isoformat()}
"""
        return model_card
    
    def upload_model(
        self,
        model_path,
        coin_symbol,
        timeframe,
        results,
        private=False
    ):
        """Upload a single model to HF Hub."""
        try:
            # Create model card
            model_card = self.create_model_card(coin_symbol, timeframe, results)
            
            logger.info(f"Uploading {coin_symbol} {timeframe}...")
            
            # Upload model weights
            self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"models/{coin_symbol}_{timeframe}.pt",
                repo_id=self.repo_id,
                token=self.token,
                private=private
            )
            
            # Upload model card
            self.api.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo=f"cards/{coin_symbol}_{timeframe}.md",
                repo_id=self.repo_id,
                token=self.token,
                private=private
            )
            
            logger.info(f"Successfully uploaded {coin_symbol} {timeframe}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upload {coin_symbol} {timeframe}: {e}")
            return False
    
    def upload_results(self, results_path):
        """Upload training results summary."""
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Create summary
            summary = {
                'total_models': len(results),
                'successful': sum(1 for r in results.values() if r.get('status') == 'success'),
                'failed': sum(1 for r in results.values() if r.get('status') == 'failed'),
                'timestamp': datetime.now().isoformat(),
                'details': results
            }
            
            # Upload
            self.api.upload_file(
                path_or_fileobj=json.dumps(summary, indent=2).encode(),
                path_in_repo="training_results.json",
                repo_id=self.repo_id,
                token=self.token
            )
            
            logger.info("Uploaded training results summary")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upload results: {e}")
            return False


def main():
    """Upload all trained models to HF Hub."""
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not installed")
        return
    
    # Initialize uploader
    uploader = HFUploader()
    
    # Load training results
    if not os.path.exists('results/training_results.json'):
        logger.error("training_results.json not found")
        return
    
    with open('results/training_results.json', 'r') as f:
        results = json.load(f)
    
    # Upload each model
    upload_count = 0
    for model_name, model_result in results.items():
        if model_result.get('status') != 'success':
            logger.warning(f"Skipping {model_name} (training failed)")
            continue
        
        model_path = model_result['model_path']
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            continue
        
        # Extract coin and timeframe
        parts = model_name.rsplit('_', 1)
        if len(parts) != 2:
            logger.warning(f"Invalid model name: {model_name}")
            continue
        
        coin_symbol, timeframe = parts
        
        if uploader.upload_model(model_path, coin_symbol, timeframe, model_result):
            upload_count += 1
    
    # Upload results summary
    uploader.upload_results('results/training_results.json')
    
    logger.info(f"\nUpload completed: {upload_count} models uploaded")


if __name__ == "__main__":
    main()
