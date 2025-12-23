# CPB Pipeline Complete Guide

## Architecture Overview

### File Structure
```
notebooks/
├── complete_v2_pipeline.py      ← V2 complete pipeline (data -> train -> viz -> upload)
├── complete_v3_pipeline.py      ← V3 complete pipeline (to be created)
├── complete_v4_pipeline.py      ← V4 complete pipeline (to be created)
└── ...

ALL_MODELS/
├── MODEL_V1/                    ← V1 models on HF
│   ├── v2_model_BTC_USDT.h5
│   ├── v2_model_ETH_USDT.h5
│   └── ...
├── MODEL_V2/                    ← V2 models on HF
│   ├── v2_model_BTC_USDT.h5
│   ├── v2_model_ETH_USDT.h5
│   ├── metadata.json
│   └── README.md
├── MODEL_V3/                    ← V3 models on HF
└── ...
```

## Execution Flow

### Run V2 Pipeline (One Cell)
```python
import urllib.request

urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/cpbv2/main/notebooks/complete_v2_pipeline.py',
    'complete_v2_pipeline.py'
)

exec(open('complete_v2_pipeline.py').read())
```

### Pipeline Phases
```
[Phase 1] Data Loading
   └─ Load/generate K-line data (future: 3000 candles)

[Phase 2] Data Preparation  
   └─ Prepare OHLCV sequences

[Phase 3] Model Architecture
   └─ Define V2 LSTM model (output: [price, volatility])

[Phase 4] Training
   └─ Train 20 models (one per pair)
   └─ Store training history

[Phase 5] Visualization
   └─ Generate training loss/MAE plots
   └─ Save to visualizations/

[Phase 6] Organize Models
   └─ Move models to ALL_MODELS/MODEL_V2/

[Phase 7] Create Metadata
   └─ metadata.json (training results)
   └─ README.md (documentation)

[Phase 8] Upload to HF
   └─ Upload entire ALL_MODELS/MODEL_V2/ folder to HF
   └─ Creates: {username}/cpb-models/MODEL_V2/
```

## Version Management

### Creating New Version (V3)
```
1. Copy complete_v2_pipeline.py to complete_v3_pipeline.py
2. Modify training logic, hyperparameters, etc.
3. Change CONFIG['version'] = 'v3'
4. Run the pipeline

Result:
   - ALL_MODELS/MODEL_V3/ created
   - Uploaded to HF: cpb-models/MODEL_V3/
```

### Fixing Bug in V2
```
1. Edit complete_v2_pipeline.py
2. Fix the bug
3. Re-run: exec(open('complete_v2_pipeline.py').read())
4. Same filename overwrites previous version
```

### Do NOT Create
```
❌ complete_v2_pipeline_fixed.py
❌ complete_v2_pipeline_v2.py
❌ complete_v2_pipeline_updated.py

✅ Just edit and save complete_v2_pipeline.py
```

## Hugging Face Structure

### HF File Overwriting
```
If you upload the same model filename twice:
  v2_model_BTC_USDT.h5 (first upload)
  v2_model_BTC_USDT.h5 (second upload with same name)
  
Result: First file is OVERWRITTEN by second file
```

### HF Folder Structure
```
Hugging Face Repository: {username}/cpb-models

├── MODEL_V1/
│   ├── v2_model_BTC_USDT.h5
│   ├── v2_model_ETH_USDT.h5
│   ├── ... (20 models)
│   ├── metadata.json
│   └── README.md
│
├── MODEL_V2/
│   ├── v2_model_BTC_USDT.h5
│   ├── v2_model_ETH_USDT.h5
│   ├── ... (20 models)
│   ├── metadata.json
│   └── README.md
│
├── MODEL_V3/
│   └── ...
```

### Upload Logic
```python
# CORRECT: Upload entire folder at once
api.upload_folder(
    folder_path='ALL_MODELS/MODEL_V2',
    path_in_repo='MODEL_V2',  # HF folder structure
    ...
)

# WRONG: Don't upload individual files (causes errors)
for file in all_files:
    api.upload_file(...)  # ❌ This can cause conflicts
```

## Workflow Example

### Day 1: Train V2
```python
# Run complete_v2_pipeline.py
# Result:
#   - models/ (local training)
#   - ALL_MODELS/MODEL_V2/ (organized)
#   - HF: cpb-models/MODEL_V2/ (uploaded)
```

### Day 2: Fix Bug in V2
```python
# Edit complete_v2_pipeline.py (fix bug)
# Run again
# Result:
#   - Same ALL_MODELS/MODEL_V2/ (overwritten with fixed models)
#   - Same HF: cpb-models/MODEL_V2/ (updated with fixed models)
```

### Day 3: Create V3 with New Features
```python
# Copy complete_v2_pipeline.py to complete_v3_pipeline.py
# Modify: Add 3000 K-candle loading, new loss function, etc.
# Run complete_v3_pipeline.py
# Result:
#   - ALL_MODELS/MODEL_V3/ (new folder)
#   - HF: cpb-models/MODEL_V3/ (new folder on HF)
```

## Important Notes

### File Naming Convention
```
✅ complete_v2_pipeline.py
✅ complete_v3_pipeline.py
✅ complete_v4_pipeline.py

❌ Do NOT name files like:
   - FINAL_PRODUCTION_V2_PER_PAIR.py
   - FINAL_PRODUCTION_V2_WITH_VOLATILITY.py
   - ORGANIZE_AND_UPLOAD_V2_MODELS.py
```

### Upload Strategy
```
✅ Always upload entire folder (MODEL_V2, MODEL_V3, etc.)
✅ One pipeline file per version
✅ Edit-and-rerun for bug fixes
✅ Copy-and-modify for new versions

❌ Don't upload individual files
❌ Don't create multiple files for same version
❌ Don't upload from multiple locations
```

### Data Loading (Future)
```python
# Current (synthetic data):
def load_kline_data(pair, limit=100):
    # Generate synthetic data
    
# TODO (3000 real candles):
def load_kline_data(pair, limit=3000):
    from binance.client import Client
    client = Client()
    klines = client.get_historical_klines(pair, '1h', "3000 hours ago UTC")
    # Process and return
```

Just change the `limit` parameter and the data loading logic when ready.

## Quick Reference

| Task | Do This |
|------|----------|
| Fix bug in V2 | Edit `complete_v2_pipeline.py`, re-run |
| Create V3 | Copy `complete_v2_pipeline.py` to `complete_v3_pipeline.py`, modify, run |
| Check HF models | Visit: https://huggingface.co/{username}/cpb-models |
| Update metadata | Edit in pipeline, re-run |
| View versions | Check `ALL_MODELS/` folder locally |

## Support

For issues or questions:
1. Check the inline comments in `complete_vX_pipeline.py`
2. Review this guide
3. Check `ALL_MODELS/MODEL_VX/metadata.json` for training results
