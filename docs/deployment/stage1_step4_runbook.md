# Stage 1 Step 4 Runbook — Frozen Model A Historical Trading Replay Parity

## Purpose

Stage 1 Step 4 verifies that the deployment trading-engine implementation reproduces the frozen Notebook 7 Model A historical overlay behaviour from prediction probabilities. This step validates trading semantics only; it does not tune Model A, test Model B, connect to MT5, or place orders.

## Preconditions

The following gates must already be passed and retained locally:

- Stage 1 Step 1: frozen artefact verification
- Stage 1 Step 2: historical M15 feature and 48-bar sequence parity
- Stage 1 Step 3: full CNN-LSTM inference parity

The formal Step 4 script checks that the Step 3 report exists and has `status == "PASS"`, `formal_gate == true`, and zero threshold flips.

## What the script verifies

The script replays Model A from the frozen prediction CSVs and applies:

- Long entry when `p_up >= 0.53`
- Short entry when `p_up <= 0.47`
- Flat-zone exit when `0.47 < p_up < 0.53`, subject to hold and cap rules
- Minimum hold of 3 eligible M15 prediction bars
- Maximum 3 policy-counted position-change events per UTC day
- Reversal counted as 1 policy event but 2 turnover units
- Gap exits at the first eligible prediction after a non-15-minute prediction gap
- Daily loss stop using Notebook 7 net log-return threshold
- Total drawdown stop using net equity relative to running peak
- Cost accounting at 0, 0.5, and 1.0 bps

The saved Notebook 7 strategy log and metrics are used only as comparison targets. They are not used as inputs to generate the replay path.

## Commands

From the repository root:

```powershell
cd C:\Users\Zac\fyp_master_starter
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest -m "not tensorflow"
python scripts\deployment\run_model_a_replay_parity.py --repo-root .
```

Expected final output:

```text
INFO Stage 1 Step 4 status: PASS
INFO JSON report: C:\Users\Zac\fyp_master_starter\runtime\reports\stage1_step4_model_a_replay_parity.json
INFO Metrics CSV: C:\Users\Zac\fyp_master_starter\runtime\reports\stage1_step4_model_a_metrics_by_cost.csv
INFO Bar comparison CSV: C:\Users\Zac\fyp_master_starter\runtime\reports\stage1_step4_bar_log_comparison.csv
```

## Reports to retain

- `runtime/reports/stage1_step4_model_a_replay_parity.json`
- `runtime/reports/stage1_step4_model_a_metrics_by_cost.csv`
- `runtime/reports/stage1_step4_bar_log_comparison.csv`

## Acceptance rule

Step 4 can only be closed when the JSON report has:

```json
{
  "status": "PASS",
  "formal_gate": true,
  "stage": 1,
  "step": 4
}
```

## Report value

If this step passes, the capstone can state that the deployment trading engine reproduces the frozen Model A historical replay semantics, not merely the model probabilities. This strengthens the deployment validation chain from artefacts to features, inference, and strategy accounting.

## Limitation

This does not prove profitability. It proves faithful reproduction of the frozen Notebook 7 Model A trading logic, including the negative holdout outcome.
