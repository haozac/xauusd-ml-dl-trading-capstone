# Stage 1 Step 1 Runbook

## Purpose

Stage 1 Step 1 verifies the frozen Stage 0 governance files and the exact Notebook 7 model bundle before any feature-engineering, strategy, MT5, or order code is introduced.

It verifies:

1. Stage 0 freeze-manifest hashes.
2. Model A and Model B shared-artefact consistency.
3. Notebook 7 model, scaler, feature-order, parameter, epoch, and overlay hashes.
4. Python, TensorFlow, Keras, and scikit-learn compatibility.
5. Scaler type, dimensions, fitted-row count, feature order, and finite parameters.
6. Keras input/output shapes, architecture, and parameter count.
7. One actual 48-by-51 Notebook 7 holdout sequence against its saved probability.

The reference sequence is a compatibility fixture only. It is not a new evaluation and does not change the locked holdout result.

## Required local files

The following gitignored artefacts must exist locally:

```text
notebook_outputs/07_m15_cnn_lstm_final_holdout_evaluation/
├── models/
│   └── cnn_lstm_vanilla_volume_assisted_holdout_evaluation.keras
└── preprocessing/
    ├── cnn_lstm_vanilla_volume_assisted_holdout_scaler.pkl
    ├── cnn_lstm_vanilla_volume_assisted_holdout_features.json
    └── cnn_lstm_vanilla_volume_assisted_holdout_params.json
```

The selected epoch and overlay files must also remain in the corresponding `configuration/` directory.

## Windows environment setup

From the repository root:

```powershell
py -3.11 -m venv .venv-deployment
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-deployment.txt
python -m pip install -e .
```

Python 3.11 is recommended. Python 3.12 is accepted by the frozen contract. Do not use the research environment automatically if its versions differ from the deployment pins.

## Run unit and non-TensorFlow integration tests

```powershell
python -m pytest -m "not tensorflow"
```

## Run the formal Step 1 verification

```powershell
python scripts\verify_notebook7_artifacts.py --repo-root .
```

Expected terminal result:

```text
Stage 1 Step 1 status: PASS
```

The machine-readable report is written to:

```text
runtime/reports/stage1_step1_verification.json
```

## Metadata-only diagnostic

This command verifies immutable files and JSON/YAML contracts without loading the pickle or Keras model:

```powershell
python scripts\verify_notebook7_artifacts.py --repo-root . --metadata-only --non-strict-environment
```

A metadata-only result is not sufficient to pass the formal Step 1 gate.

## Failure handling

| Failure | Action |
|---|---|
| Stage 0 hash mismatch | Restore the approved Stage 0 v1.2 files. Do not edit the manifest to match an unreviewed change. |
| Notebook 7 artefact hash mismatch | Restore the exact frozen file from the audited Notebook 7 ZIP. Do not retrain or resave it. |
| Unsupported Python version | Create a Python 3.11 or 3.12 deployment virtual environment. |
| TensorFlow, Keras, or scikit-learn mismatch | Reinstall from `requirements-deployment.txt`. |
| Scaler feature mismatch | Stop. Check that the exact frozen scaler and feature JSON were extracted. |
| Model shape or parameter mismatch | Stop. Check that the exact frozen `.keras` file was extracted. |
| Reference probability mismatch | Stop. Preserve the report and inspect versions, casting, scaler loading, and model backend. Do not loosen the threshold automatically. |

## Gate

Stage 1 Step 1 passes only when the formal command returns `PASS`. A partial or non-formal pass does not authorise Step 2.
