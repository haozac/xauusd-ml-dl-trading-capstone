# Stage 3 Step 2 v1.1 runbook

Purpose: run one more tiny 0.01-lot XAUUSD BUY open/close demo test, this time with robust broker-history recovery.

This step calls `order_send`. It must be run only on a demo account.

## Before running

1. Open MT5.
2. Log in to the Dukascopy demo account.
3. Enable Algo Trading.
4. Confirm there is no open XAUUSD position and no pending XAUUSD order.
5. Confirm Stage 3 Step 1 has a formal PASS report.

## Command

```powershell
python scripts\deployment\run_tiny_demo_order_test.py --repo-root . --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe" --confirm-send I_UNDERSTAND_STAGE3_STEP2_SENDS_DEMO_ORDER
```

## Expected outputs

- `runtime/reports/stage3_step2_v1_1_tiny_order_test.json`
- `runtime/reports/stage3_step2_v1_1_order_send_events.csv`
- `runtime/reports/stage3_step2_v1_1_position_snapshots.csv`
- `runtime/reports/stage3_step2_v1_1_history_deals.csv`
- `runtime/reports/stage3_step2_v1_1_history_orders.csv`

After success, MT5 Trade tab should have no open XAUUSD position. MT5 History tab should show the completed demo trade.
