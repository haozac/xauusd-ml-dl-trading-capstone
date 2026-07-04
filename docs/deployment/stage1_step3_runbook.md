# Stage 1 Step 3 Runbook — Full CNN-LSTM Inference Parity

## Purpose

Stage 1 Step 3 verifies that the deployment code can reproduce Notebook 7 saved CNN-LSTM probabilities for every audited sequence in the 2024 overlay-validation partition and the final untouched holdout partition.

This step is limited to offline inference parity. It does not implement Model A trading logic, Model B trading logic, MT5 connectivity, shadow trading, or demo order execution.

## Formal gate command

From the repository root:

```powershell
cd C:\Users\Zac\fyp_master_starter
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install -e .
python scripts\deployment\run_inference_parity.py --repo-root .
```

Expected final lines:

```text
INFO Stage 1 Step 3 status: PASS
INFO JSON report: ...\runtime\reports\stage1_step3_inference_parity.json
INFO Summary CSV: ...\runtime\reports\stage1_step3_probability_parity_summary.csv
INFO Diagnostic CSV: ...\runtime\reports\stage1_step3_probability_parity_diagnostics.csv
```

## Reports to retain

```text
runtime/reports/stage1_step3_inference_parity.json
runtime/reports/stage1_step3_probability_parity_summary.csv
runtime/reports/stage1_step3_probability_parity_diagnostics.csv
```

The JSON report is the formal gate evidence. The summary CSV is suitable for the methodology or deployment-validation appendix. The diagnostic CSV is intentionally compact and contains rows with the largest probability differences and rows nearest the decision thresholds.

## Formal pass rule

Step 3 is closed only if:

```text
status == PASS
formal_gate == true
checks.aggregate_inference_parity.total_threshold_flips == 0
checks.aggregate_inference_parity.maximum_absolute_difference <= probability_tolerance
```

The process exit code alone is not the evidence. The JSON status and formal-gate field must also be checked.

## Deterministic CPU controls

The formal script sets these before TensorFlow is imported:

```text
CUDA_VISIBLE_DEVICES=-1
TF_DETERMINISTIC_OPS=1
TF_CUDNN_DETERMINISTIC=1
TF_ENABLE_ONEDNN_OPTS=0
```

The purpose is to make the parity run a CPU reference check rather than a GPU-performance check. The expected numerical tolerance is `1e-5`, and decision-threshold parity is checked separately at `0.47`, `0.50`, `0.53`, and `0.55`.

## Test commands

Run the non-TensorFlow tests first:

```powershell
python -m pytest -m "not tensorflow"
```

Expected approximate result after this patch:

```text
32 passed, 2 deselected
```

Then run the formal Step 3 gate:

```powershell
python scripts\deployment\run_inference_parity.py --repo-root .
```

## What this step proves

Stage 1 Step 3 proves that the deployment inference pathway can reproduce the frozen Notebook 7 probabilities using:

1. the frozen Notebook 7 model,
2. the frozen StandardScaler,
3. the frozen 51-feature order,
4. the verified M15 feature reconstruction pipeline,
5. the exact 48-bar contiguous sequence rule,
6. the saved Notebook 7 prediction references.

This is stronger than a single smoke test because every prediction row in the two Notebook 7 inference partitions is re-generated and compared.

## What this step does not prove

Step 3 does not prove profitability, live broker compatibility, MT5 timestamp correctness, spread/slippage correctness, order execution safety, or Model B improvement. Those belong to later stages.
