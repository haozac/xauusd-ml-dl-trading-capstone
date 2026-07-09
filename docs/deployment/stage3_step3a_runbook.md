# Stage 3 Step 3A runbook: Model B controlled execution dry-run

Purpose: run the frozen Model B current execution logic against live completed M15 MT5 signals without sending orders.

This step is a dry-run gate. It may use `order_check` for a would-enter-long request, but it must never call `order_send`.

## Before running

1. Open MT5.
2. Log in to the Dukascopy demo account.
3. Enable Algo Trading so `order_check` can be evaluated.
4. Confirm XAUUSD is visible.
5. Confirm there is no open XAUUSD position and no pending XAUUSD order.
6. Confirm Stage 3 Step 2 v1.1 has a formal PASS report.

## Recommended duration

Run for 2 hours during market-open time:

```powershell
python scripts\deployment\run_model_b_controlled_dry_run.py --repo-root . --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe" --watch --poll-seconds 30 --max-iterations 240 --reset-state
```

This polls every 30 seconds but only treats a new completed M15 bar as a new decision event. Two hours should capture about 8 completed M15 events.

A shorter minimum smoke run is possible:

```powershell
python scripts\deployment\run_model_b_controlled_dry_run.py --repo-root . --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe" --watch --poll-seconds 30 --max-iterations 120 --reset-state
```

That is about 1 hour and should capture about 4 completed M15 events.

## Outputs

- `runtime/reports/stage3_step3a_model_b_controlled_dry_run.json`
- `runtime/execution_dry_run/stage3_step3a_model_b_intents.csv`
- `runtime/reports/stage3_step3a_latest_decision.csv`
- `runtime/state/stage3_step3a_model_b_dry_run_state.json`

## Expected result

The script should report PASS after observing at least the configured minimum number of new completed M15 events, with `order_send_called = false` and no hard safety block.

If the model produces no entry signal, that is still acceptable. This stage is testing controlled execution logic and safety, not forcing a trade.
