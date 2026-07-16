# Stage 3 Step 4B Runbook

## Purpose

Validate that the two harmonised Dukascopy MT5 installations can run the same frozen CNN-LSTM pipeline concurrently while remaining fully isolated and order-free.

This gate launches two Python worker processes:

- Model A worker -> Model A terminal/account -> `runtime/model_a`
- Model B worker -> Model B terminal/account -> `runtime/model_b`

The parent coordinator stops automatically after four synchronised fresh completed M15 events, or after the 90-minute safety timeout.

## Safety boundary

Step 4B does not call `order_check` or `order_send`.

The only broker calculation methods permitted are:

- `order_calc_margin`
- `order_calc_profit`

These methods calculate hypothetical broker economics and cannot create, modify, or close an order.

## Manual prerequisites

1. Open both exact Dukascopy MT5 terminals.
2. Model A terminal must be logged into Model A account.
3. Model B terminal must be logged into Model B account.
4. Both terminals must show a live connection.
5. Enable Algo Trading in both terminals.
6. Confirm no open XAUUSD positions in either account.
7. Confirm no pending XAUUSD orders in either account.
8. Do not manually trade XAUUSD during the gate.
9. Keep the latest harmonised Step 4A PASS report in `runtime/reports`.

Expected terminal paths:

```text
C:\Program Files\Dukascopy MetaTrader 5 Model A\terminal64.exe
C:\Program Files\Dukascopy MetaTrader 5 Model B\terminal64.exe
```

## Installation

Extract the patch and copy the contents of `stage3_step4b_patch_v1_1` into the repository root.

```powershell
cd C:\Users\Zac\fyp_master_starter
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest -m "not tensorflow"
```

## Execute Step 4B

```powershell
python scripts\deployment\run_dual_terminal_shadow_sync.py `
  --repo-root . `
  --config config\dual_terminal_runtime.yaml `
  --required-synchronised-events 4 `
  --poll-seconds 30 `
  --max-runtime-minutes 90 `
  --server-time-offset-hours 3
```

Expected runtime is normally about 45 to 75 minutes depending on where the command starts within the current M15 bar. The script exits automatically.

Two TensorFlow worker processes run concurrently. Close unnecessary applications if laptop memory is limited.

## Required PASS conditions

- Step 4A prerequisite is a formal PASS and matches the current terminal paths.
- Both terminals use the same Dukascopy MT5 distribution and build.
- Four or more common fresh completed M15 events are observed.
- Event timestamps match exactly.
- Completed-rate digests match.
- Feature-frame digests match.
- Scaled 48 x 51 sequence digests match.
- Absolute probability difference is no more than `5e-7`.
- Model A and Model B overlay outputs agree across the two data feeds.
- Hypothetical margin and profit calculations match within tolerance.
- Both accounts remain flat with zero pending orders.
- No `order_check` or `order_send` call occurs.
- Model, scaler and feature-order fingerprints are identical across workers.

## Outputs to upload

```text
runtime/reports/stage3_step4b_dual_terminal_shadow_sync.json
runtime/reports/stage3_step4b_dual_terminal_shadow_sync_summary.csv
runtime/reports/stage3_step4b_synchronised_events.csv
runtime/reports/stage3_step4b_model_a_events.csv
runtime/reports/stage3_step4b_model_b_events.csv
runtime/reports/stage3_step4b_economic_sanity.csv
```

Also paste the pytest output and Step 4B console output.

Run-specific worker evidence remains preserved below:

```text
runtime/model_a/stage3_step4b/<run_id>/
runtime/model_b/stage3_step4b/<run_id>/
```

## Stop procedure

The normal process stops automatically. If an emergency requires manual interruption, press `Ctrl+C` once. The coordinator will signal both workers to stop. A manually interrupted run is not a formal PASS and must be reviewed before deciding whether a rerun is required.

## Next stage

A Step 4B PASS does not authorise the final 14-calendar-day run. The next stage is Step 4C external Windows VPS migration and launch-readiness validation.

After Step 4C readiness passes, conduct one full day of controlled VPS operation to verify real order entry, holding, exit, logging, restart and recovery behaviour before starting the final 14-calendar-day Model A versus Model B paper-trading comparison.


## v1.2 correction

Broker calculation calls use only the documented positional argument form. This avoids a MetaTrader5 extension behaviour where keyword calls can return `0.0` rather than raising an exception. The Step 4B command is unchanged.
