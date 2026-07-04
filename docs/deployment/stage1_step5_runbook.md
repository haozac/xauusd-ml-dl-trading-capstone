# Stage 1 Step 5 Runbook — Model B Diagnostic Historical Replay

## Purpose

Stage 1 Step 5 replays the frozen Model B V2 long-only overlay on the same frozen Notebook 7 prediction references used by Model A. This is a diagnostic replay only. It is not a new untouched holdout, not a threshold search, not a tuning step, and not evidence of live profitability.

## Preconditions

The following gates must already have passed locally:

1. Stage 1 Step 1 — frozen artefact verification.
2. Stage 1 Step 2 — historical feature and 48-bar sequence parity.
3. Stage 1 Step 3 — full neural-network inference parity.
4. Stage 1 Step 4 — frozen Model A trading replay parity.

The script requires the formal Step 4 report at:

```text
runtime/reports/stage1_step4_model_a_replay_parity.json
```

## Command

From the repository root:

```powershell
python -m pip install -e .
python -m pytest -m "not tensorflow"
python scripts\deployment\run_model_b_diagnostic_replay.py --repo-root .
```

## Expected output

```text
INFO Stage 1 Step 5 status: PASS
INFO JSON report: ...\runtime\reports\stage1_step5_model_b_diagnostic_replay.json
INFO Metrics CSV: ...\runtime\reports\stage1_step5_model_b_metrics_by_cost.csv
INFO Comparison CSV: ...\runtime\reports\stage1_step5_model_b_vs_model_a_comparison.csv
INFO Diagnostics CSV: ...\runtime\reports\stage1_step5_model_b_diagnostics.csv
```

## Output files

```text
runtime/reports/stage1_step5_model_b_diagnostic_replay.json
runtime/reports/stage1_step5_model_b_metrics_by_cost.csv
runtime/reports/stage1_step5_model_b_vs_model_a_comparison.csv
runtime/reports/stage1_step5_model_b_diagnostics.csv
```

## Interpretation

A PASS means the frozen Model B diagnostic overlay was replayed without violating its invariants:

- no short positions;
- no entries below 0.55;
- no normal long holds below 0.50;
- maximum one successful new entry per UTC day;
- frozen risk stops were applied;
- the replay used the same frozen prediction references and prior Model A baseline from Step 4.

A PASS does not mean Model B is profitable or superior. The output comparison table gives diagnostic deltas versus Model A for turnover, active exposure, drawdown, return and transaction-cost burden.
