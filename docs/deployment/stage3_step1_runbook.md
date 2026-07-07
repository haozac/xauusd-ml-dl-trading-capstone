# Stage 3 Step 1 Runbook — Order Permission and order_check Preflight

## Purpose

Validate that the current MT5 demo terminal, account, symbol, and frozen execution controls can pass broker-side `order_check` for a tiny 0.01-lot XAUUSD request. This step is the first Stage 3 gate, but it still does **not** call `order_send` and does **not** execute any trade.

## Manual prerequisites

1. Stage 2 Step 3A must be passed and `config/broker_execution_controls_frozen.yaml` must exist.
2. Open MetaTrader 5 manually.
3. Log in to the Dukascopy MT5 demo account.
4. Confirm XAUUSD is visible in Market Watch.
5. Confirm the market is open.
6. Enable the MT5 top-bar **Algo Trading** button for this preflight. Stage 2 kept it disabled; Stage 3 Step 1 needs terminal trading permission so `order_check` can be validated.
7. Do not manually place trades.

## Commands

```powershell
cd C:\Users\Zac\fyp_master_starter
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest -m "not tensorflow"
```

Run the preflight:

```powershell
python scripts\deployment\run_order_permission_preflight.py --repo-root . --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe"
```

## Outputs

- `runtime/reports/stage3_step1_order_permission_preflight.json`
- `runtime/reports/stage3_step1_order_check_results.csv`

## Pass criteria

- MT5 terminal connects.
- Demo account is active.
- Terminal trading permission is enabled.
- Account trade permission is enabled.
- XAUUSD is visible and spread is within the frozen gate.
- `order_check` passes for the requested side or sides.
- `order_send` is not called.

## If it fails

If `terminal_trade_allowed` is false, enable the MT5 Algo Trading button and rerun. If `order_check` fails with a broker retcode, do not continue to Stage 3 Step 2 until the retcode and filling mode are reviewed.


## v1.1 hotfix note

This hotfix changes the MT5 `order_check` and `order_calc_margin` calls to use named arguments first, with positional fallback. The live v1.0 run reached MT5 successfully but the MetaTrader5 module returned `(-2, "Unnamed arguments not allowed")` for the order-check calls. The hotfix also keeps human-readable filling metadata outside the actual MqlTradeRequest dictionary submitted to MT5 and fixes the report-level terminal-initialized flag after shutdown.
