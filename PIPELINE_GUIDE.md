# CPB Pipeline Complete Guide

## Important Notice

**All models are uploaded to: `caizongxun/cpbv2` repository**

Never create new repositories. All versions (V1, V2, V3, etc.) go to the same repo in different folders.

---

## Architecture Overview

### File Structure (Local)
```
notebooks/
├── complete_v2_pipeline.py      ← V2 complete pipeline (data -> train -> viz -> upload)
├── complete_v3_pipeline.py      ← V3 complete pipeline (to be created)
├── complete_v4_pipeline.py      ← V4 complete pipeline (to be created)
└── ...

ALL_MODELS/
├── MODEL_V1/                    ← V1 models (20 pairs)
│   ├── v1_model_BTC_USDT.h5
│   ├── v1_model_ETH_USDT.h5
│   ├── metadata.json
│   └── README.md
├── MODEL_V2/                    ← V2 models (20 pairs)
│   ├── v2_model_BTC_USDT.h5
│   ├── v2_model_ETH_USDT.h5
│   ├── metadata.json
│   └── README.md
└── MODEL_V3/                    ← V3 models (to be created)
```

### File Structure (GitHub - caizongxun/cpbv2)
```
GitHub Repository: caizongxun/cpbv2

ALL_MODELS/
├── MODEL_V1/
│   ├── v1_model_BTC_USDT.h5
│   ├── v1_model_ETH_USDT.h5
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
└── MODEL_V3/
    └── ...
```

---

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

[Phase 8] Upload to GitHub
   └─ Upload entire ALL_MODELS/MODEL_V2/ folder to caizongxun/cpbv2
   └─ Creates: caizongxun/cpbv2/ALL_MODELS/MODEL_V2/
```

---

## Version Management

### Creating New Version (V3)
```
1. Copy complete_v2_pipeline.py to complete_v3_pipeline.py
2. Modify training logic, hyperparameters, etc.
3. Change CONFIG['version'] = 'v3'
4. Run the pipeline

Result:
   - ALL_MODELS/MODEL_V3/ created locally
   - Uploaded to GitHub: caizongxun/cpbv2/ALL_MODELS/MODEL_V3/
```

### Fixing Bug in V2
```
1. Edit complete_v2_pipeline.py
2. Fix the bug
3. Re-run: exec(open('complete_v2_pipeline.py').read())
4. Same filename overwrites previous version
5. GitHub models are updated when you push
```

### Do NOT Create
```
✗ New repositories (e.g., zongowo111/cpb-models)
✗ complete_v2_pipeline_fixed.py
✗ complete_v2_pipeline_v2.py
✗ complete_v2_pipeline_updated.py

✓ Just edit and save complete_v2_pipeline.py
✓ Create new versions: complete_v3_pipeline.py, complete_v4_pipeline.py
✓ All uploads go to caizongxun/cpbv2
```

---

## GitHub Upload Structure

### Same Repository, Different Folders
```
Repository: caizongxun/cpbv2 (ONE repo for everything)

Folders:
  ALL_MODELS/MODEL_V1/  ← V1 models
  ALL_MODELS/MODEL_V2/  ← V2 models
  ALL_MODELS/MODEL_V3/  ← V3 models
  ...
```

### File Overwriting
```
When you run complete_v2_pipeline.py again:

  First run:
    GitHub: ALL_MODELS/MODEL_V2/v2_model_BTC_USDT.h5 (new)
  
  Second run (same filename):
    GitHub: ALL_MODELS/MODEL_V2/v2_model_BTC_USDT.h5 (updated/overwritten)

Different versions stay separate:
  ALL_MODELS/MODEL_V2/v2_model_BTC_USDT.h5  ← V2
  ALL_MODELS/MODEL_V3/v3_model_BTC_USDT.h5  ← V3 (different folder)
```

### Upload Logic
```python
# Pipeline automatically uploads entire folder
api.upload_folder(
    folder_path='ALL_MODELS/MODEL_V2',
    path_in_repo='ALL_MODELS/MODEL_V2',  # GitHub path
    repo_id='caizongxun/cpbv2',
    ...
)

# This ensures:
#   ✓ All files in folder are uploaded together
#   ✓ Same-name files are updated
#   ✓ No conflicts or errors
#   ✓ Everything stays organized
```

---

## Workflow Example

### Day 1: Train V2
```python
# Run complete_v2_pipeline.py

# Result:
#   Local: models/ + ALL_MODELS/MODEL_V2/ + visualizations/
#   GitHub: caizongxun/cpbv2/ALL_MODELS/MODEL_V2/ (uploaded)
```

### Day 2: Fix Bug in V2
```python
# Edit complete_v2_pipeline.py (fix bug)
# Run again

# Result:
#   Local: same folders (overwritten with fixed models)
#   GitHub: same path (updated with fixed models)
#   V3 is untouched in MODEL_V3/
```

### Day 3: Create V3 with New Features
```python
# Copy complete_v2_pipeline.py to complete_v3_pipeline.py
# Modify: Add 3000 K-candle loading, new loss function, etc.
# Run complete_v3_pipeline.py

# Result:
#   Local: ALL_MODELS/MODEL_V3/ (new folder)
#   GitHub: caizongxun/cpbv2/ALL_MODELS/MODEL_V3/ (new folder)
#   V2 models in MODEL_V2/ remain unchanged
```

---

## Important Notes

### File Naming Convention
```
✓ complete_v2_pipeline.py
✓ complete_v3_pipeline.py
✓ complete_v4_pipeline.py

✗ Do NOT name files:
   - FINAL_PRODUCTION_V2_PER_PAIR.py
   - FINAL_PRODUCTION_V2_WITH_VOLATILITY.py
   - ORGANIZE_AND_UPLOAD_V2_MODELS.py
```

### Repository Management
```
✓ caizongxun/cpbv2 (main repository)
  ├── ALL_MODELS/MODEL_V1/
  ├── ALL_MODELS/MODEL_V2/
  ├── ALL_MODELS/MODEL_V3/
  └── notebooks/
      ├── complete_v2_pipeline.py
      ├── complete_v3_pipeline.py
      └── ...

✗ Don't create:
   - zongowo111/cpb-models
   - new_cpb_repo
   - cpb-v2
   - any other repositories
```

### Upload Strategy
```
✓ Always upload to caizongxun/cpbv2
✓ Upload entire folder (MODEL_V2, MODEL_V3, etc.)
✓ One pipeline file per version
✓ Edit-and-rerun for bug fixes
✓ Copy-and-modify for new versions

✗ Don't:
   - Create new repositories
   - Upload individual files
   - Use multiple repositories
   - Create multiple files for same version
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

---

## Quick Reference

| Task | Do This |
|------|----------|
| Fix bug in V2 | Edit `complete_v2_pipeline.py`, re-run |
| Create V3 | Copy `complete_v2_pipeline.py` to `complete_v3_pipeline.py`, modify, run |
| Check GitHub | Visit: https://github.com/caizongxun/cpbv2 |
| Update metadata | Edit in pipeline, re-run |
| View versions | Check `ALL_MODELS/` folder locally |
| Push to GitHub | Pipeline auto-uploads when you provide token |

---

## GitHub Token Setup

To auto-upload during pipeline execution:

### Option 1: Colab (Recommended)
```python
# The pipeline will prompt for token:
Enter GitHub Token (or press Enter to skip): <paste_token_here>
```

### Option 2: Environment Variable
```bash
export GITHUB_TOKEN=your_token_here
```

### Option 3: Manual Push
```bash
# If pipeline upload fails, manually push:
git add ALL_MODELS/
git commit -m "Update V2 models"
git push origin main
```

---

## Troubleshooting

### Models not uploading?
1. Check GitHub token is valid
2. Ensure PyGithub is installed: `pip install PyGithub`
3. Try manual git push instead

### Files conflicting?
1. Each version uses separate folder (MODEL_V1, MODEL_V2, etc.)
2. Same-version updates overwrite within same folder
3. Different versions never conflict

### Where are my models?
1. Local: `ALL_MODELS/MODEL_V2/` (and subfolders for other versions)
2. GitHub: `caizongxun/cpbv2/ALL_MODELS/MODEL_V2/` (and subfolders)
3. Never on Hugging Face (only GitHub)

---

## Summary

**One repo, multiple versions, organized by folders.**

✓ `caizongxun/cpbv2` is your single source of truth  
✓ Each version gets its own pipeline file  
✓ Each version gets its own folder in ALL_MODELS/  
✓ Bug fixes: edit and re-run same pipeline file  
✓ New versions: copy, modify, create new pipeline file  
✓ All uploads go to the same repo  

**Simple, clean, organized.**
