# Stage 2 Step 2C Runbook — Model B Minimum-Hold Diagnostic Comparison

## Purpose

Stage 2 Step 2C compares two Model B execution variants before any broker order execution:

1. `MODEL_B_V2_CURRENT`: the already frozen long-only diagnostic overlay.
2. `MODEL_B_MIN_HOLD_3`: the same overlay, but with a fixed 3 completed-M15-bar minimum hold before normal probability-based exits.

This is diagnostic only. It does not retrain the CNN-LSTM, tune thresholds, use MT5, place orders, or create a new untouched-holdout claim.

## Preconditions

Run this only after:

- Stage 1 Step 5 Model B diagnostic replay has passed.
- Stage 2 Step 2A v1.1 timestamp-corrected live shadow logger has passed.

Required default precondition reports:

```text
runtime/reports/stage1_step5_model_b_diagnostic_replay.json
runtime/reports/stage2_step2a_v1_1_mt5_shadow_logger.json
```

## Command

```powershell
python scripts\deployment\run_model_b_min_hold_comparison.py --repo-root .
```

## Expected outputs

```text
runtime/reports/stage2_step2c_model_b_min_hold_comparison.json
runtime/reports/stage2_step2c_model_b_variant_metrics_by_cost.csv
runtime/reports/stage2_step2c_model_b_min_hold_comparison.csv
```

## Interpretation

The script compares both variants on:

- overlay validation
- final holdout diagnostic
- 0, 0.5, and 1.0 bp costs

Important metrics:

- net return
- gross return
- max drawdown
- turnover units
- round-turn equivalent trades
- active bar rate
- transaction cost burden
- successful entry count
- normal exit count
- minimum-hold blocked exits

The JSON contains decision guidance, but the script does not automatically freeze a new strategy. A human methodology decision is still required before Stage 3 order execution.
