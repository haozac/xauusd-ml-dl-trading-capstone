# Stage 1 Step 6 v1.1 Runbook — Offline Runtime Simulation

## Purpose

Stage 1 Step 6 verifies the offline runtime-equivalence loop before MT5 shadow mode.

The v1.1 gate strengthens v1.0 by adding a streaming event-materialisation check:

1. The audited M15 historical bars are rebuilt into the verified feature dataframe.
2. Feature rows are scanned chronologically through a rolling 48-row buffer.
3. A runtime event is emitted only when the rolling buffer is complete and contiguous at the M15 cadence.
4. The streaming-emitted event endpoints are compared against the audited batch sequence plan.
5. The resulting event stream is then passed into stateful Model A and Model B runtime ledgers.

This still does not connect to MT5 or place orders.

## Commands

```powershell
cd C:\Users\Zac\fyp_master_starter
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest -m "not tensorflow"
python scripts\deployment\run_offline_runtime_simulation.py --repo-root .
```

## Expected outputs

```text
runtime/reports/stage1_step6_offline_runtime_simulation.json
runtime/reports/stage1_step6_runtime_summary.csv
runtime/reports/stage1_step6_runtime_event_audit_1bps.csv
runtime/reports/stage1_step6_resume_determinism.csv
```

## Formal pass conditions

The JSON report must show:

```text
status = PASS
formal_gate = true
offline_only = true
mt5_used = false
orders_enabled = false
```

Required checks:

- Step 5 precondition passed.
- Historical M15 bars rebuild to the audited feature dataframe.
- The frozen scaler and CNN-LSTM load under the strict runtime environment.
- Streaming event materialisation passes for overlay validation and final holdout.
- Recomputed probabilities have zero threshold flips against Notebook 7 references.
- Runtime Model A and Model B ledgers match their audited batch replay engines.
- Restart/resume logs match uninterrupted logs.

## Scope limitation

This is still not a full live MT5 emulator. Feature engineering is produced by the already audited batch feature builder, then event readiness is verified by a rolling chronological buffer. Full one-bar incremental indicator recomputation is intentionally deferred to avoid duplicating the feature engineering pipeline and introducing a second unvalidated implementation.
