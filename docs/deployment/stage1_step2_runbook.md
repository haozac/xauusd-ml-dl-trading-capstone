# Stage 1 Step 2 Runbook

## Purpose

Stage 1 Step 2 proves that the deployment preprocessing code reproduces the official Notebook 7 M15 volume-assisted input contract before full CNN-LSTM inference is tested.

This step covers:

- Historical M15 bar integrity
- Exact 51-feature reconstruction
- Exact target construction
- Frozen feature order
- Float32-before-scaling contract
- Frozen StandardScaler transformation
- Contiguous 48-bar sequence selection
- 2024 overlay-validation endpoint alignment
- 2025 to March 2026 holdout endpoint alignment

This step does not connect to MT5 and does not place orders.

## Repository paths

The formal gate expects these local files:

```text
data/capstone_methodology/processed/
├── dukascopy_xauusd_m15_bars_with_volume.parquet
└── dukascopy_xauusd_m15_volume_assisted_relative_dataset.parquet

notebook_outputs/07_m15_cnn_lstm_final_holdout_evaluation/
├── preprocessing/
│   ├── cnn_lstm_vanilla_volume_assisted_holdout_features.json
│   └── cnn_lstm_vanilla_volume_assisted_holdout_scaler.pkl
└── tables/
    ├── overlay_validation_predictions.csv
    └── final_holdout_predictions.csv
```

The exact paths, file sizes, SHA-256 hashes, row counts, timestamp ranges, and expected partition counts are recorded in:

```text
config/stage1_step2_reference_manifest.json
```

## Installation

Activate the Python 3.11 deployment environment:

```powershell
cd C:\Users\Zac\fyp_master_starter
.\.venv-deployment\Scripts\Activate.ps1
```

Refresh the editable package after copying the Step 2 patch:

```powershell
python -m pip install -e .
```

No additional package should be required if `requirements-deployment.txt` was already installed. PyArrow is required for the parquet files.

## Tests

Run the non-TensorFlow suite:

```powershell
python -m pytest -m "not tensorflow"
```

The suite includes the formal real-data Step 2 integration test. The TensorFlow smoke test is deselected because model inference belongs to Step 3.

To run only the Step 2 integration test:

```powershell
python -m pytest -m step2
```

## Formal Step 2 gate

```powershell
python scripts\deployment\run_feature_parity.py --repo-root .
```

Expected final lines:

```text
INFO Stage 1 Step 2 status: PASS
INFO JSON report: ...\runtime\reports\stage1_step2_feature_sequence_parity.json
INFO Column report: ...\runtime\reports\stage1_step2_feature_column_parity.csv
```

## Expected reference outcomes

```text
M15 source bars: 240,636
Model-ready rows: 237,001
Frozen features: 51
Maximum accepted feature difference: 1e-12

Inner training sequences: 73,147
Inner validation sequences: 10,445
Final training sequences: 83,592
2024 overlay-validation sequences: 11,005
Final holdout sequences: 13,794
```

The feature parity report must have zero values exceeding the tolerance. Prediction endpoint timestamps and target values must align with both saved Notebook 7 prediction files.

## Outputs

Retain these files for the capstone evidence trail:

```text
runtime/reports/stage1_step2_feature_sequence_parity.json
runtime/reports/stage1_step2_feature_column_parity.csv
```

Do not edit these reports manually.

## Exit codes

```text
0 = formal Step 2 verification passed
2 = expected validation or parity failure
3 = unexpected software failure
```

## Failure routing

- Hash or file-size mismatch: confirm that the exact official parquet or Notebook 7 file is present.
- M15 bar validation failure: inspect duplicate timestamps, OHLC validity, volume, and `source_m1_bars`.
- Feature parity failure: remain in Step 2 and inspect the named feature columns.
- Sequence mismatch: inspect gap logic, date partitioning, and the 48-bar continuity rule.
- Scaling failure: confirm the frozen scaler and exact float32-before-scaling order.

Do not proceed to Step 3 until the formal JSON report returns `PASS`.
