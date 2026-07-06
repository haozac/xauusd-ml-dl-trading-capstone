# Stage 2 Step 2A v1.1 Runbook — Single-Terminal MT5 Shadow Logger

## Purpose

This stage validates the live shadow signal pipeline using one manually logged-in MT5 demo terminal. It reads completed M15 bars only, converts Dukascopy MT5 server timestamps into canonical UTC, rebuilds the frozen feature state, runs the frozen CNN-LSTM, writes Model A and Model B shadow signals, and never sends orders.

## Why v1.1 exists

The v1.0 market-open watch run proved that the polling loop and append-only signal logging worked, but the output showed a negative latest-bar age. The latest completed bar was reported as if it was ahead of UTC. Dukascopy documents MT4/MT5 server time as GMT+3 in summer and GMT+2 in winter. The frozen research dataset is UTC-aligned, so v1.1 shifts live MT5 server timestamps back into canonical UTC before feature generation, signal logging, and age checks.

## Manual prerequisites

1. Open MT5 manually.
2. Log in to the Dukascopy DEMO account.
3. Confirm the terminal is connected.
4. Confirm XAUUSD is visible in Market Watch.
5. Keep orders disabled for shadow mode.
6. During July or other summer-time periods, keep `time.mt5_server_time_offset_hours: 3` in the runtime config. During winter, change it to `2` before running formal deployment checks.

## Apply patch

Extract this patch into the repository root:

```powershell
cd C:\Users\Zac\fyp_master_starter
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install -e .
```

## Run tests

```powershell
python -m pytest -m "not tensorflow"
```

The live MT5 integration tests are expected to remain skipped unless their environment variables are explicitly enabled.

## Short v1.1 recheck

Run this during market-open time:

```powershell
python scripts\deployment\run_mt5_shadow_logger.py --repo-root . --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe" --watch --poll-seconds 30 --max-iterations 20
```

This should run for about 10 minutes. If a new M15 bar closes during that period, it should append one new canonical UTC signal row. If no new bar closes, the latest snapshot may correctly show `duplicate_event=true` and `appended_to_signal_log=false`.

## Longer formal recheck

After the short recheck is clean, run:

```powershell
python scripts\deployment\run_mt5_shadow_logger.py --repo-root . --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe" --watch --poll-seconds 30 --max-iterations 90
```

## Output files

v1.1 writes to separate files so old v1.0 server-time logs are not mixed with canonical UTC logs:

```text
runtime/reports/stage2_step2a_v1_1_mt5_shadow_logger.json
runtime/reports/stage2_step2a_v1_1_latest_completed_m15_bars.csv
runtime/reports/stage2_step2a_v1_1_latest_shadow_signal.csv
runtime/shadow/stage2_step2a_v1_1_shadow_signals.csv
runtime/state/stage2_step2a_v1_1_shadow_state.json
```

Upload these five files after the recheck.

## Acceptance criteria

1. `status=PASS`.
2. `orders_enabled=false`.
3. `safety.order_send_called=false`.
4. `forbidden_trade_function_calls=[]`.
5. `time_normalisation.conversion_applied=true` for Dukascopy summer-time use.
6. `time_normalisation.mt5_server_time_offset_hours=3` during summer-time use.
7. `time_normalisation.latest_bar_future_minutes_after_conversion <= 2`.
8. `rates.latest_closed_bar_age_minutes` is non-negative or very close to zero.
9. Latest feature time matches the latest completed bar time.
10. Append-only log has no duplicate canonical UTC event rows.

## What not to do

Do not enable orders. Do not enable Stage 3 execution. Do not mix v1.0 and v1.1 output files for formal evidence.
