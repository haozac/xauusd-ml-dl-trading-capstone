# Stage 3 Step 3B v1.1 Runbook — Model B Controlled Live Execution

## Purpose

Stage 3 Step 3B is the first controlled Model B live execution gate. It may call `mt5.order_send`, but only when the frozen Model B current rules produce a valid completed-M15 entry or exit signal.

This is **not** the final unattended 14-calendar-day paper-trading run.

## Rules locked for this gate

- Model: frozen CNN-LSTM with Model B current overlay
- Direction: long-only
- Entry: `p_up >= 0.55`
- Exit: `p_up < 0.50`
- Timeframe: completed M15 bars only
- Volume: 0.01 lot only
- Symbol: XAUUSD only
- Max open position: 1
- Max successful entries per UTC day: 1
- Spread entry gate: 800 points
- Broker-side SL/TP: disabled for this gate
- End-of-run safety close: enabled by default

## v1.1 spread behaviour

- Spread above 800 while flat and below the entry threshold: continue with `HOLD_FLAT`.
- Spread above 800 when an entry signal occurs: record `BLOCK_SPREAD`, do not send an order, and continue monitoring.
- Spread above 800 while already long: do not block model exit or end-of-run safety close.
- Structural symbol or account failures remain fatal.

## Recommended runtime

Run for 4 hours while monitoring MT5 manually. At 30-second polling, use 480 iterations:

```powershell
python scripts\deployment\run_model_b_controlled_live.py --repo-root . --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe" --watch --poll-seconds 30 --max-iterations 480 --reset-state --confirm-send I_UNDERSTAND_STAGE3_STEP3B_MODEL_B_ORDER_SEND
```

Stop earlier only if one complete entry and exit cycle occurs. The script has this enabled by default.

## Manual prerequisites

1. Open MT5.
2. Log in to the Dukascopy demo account.
3. Enable Algo Trading.
4. Confirm there is no open XAUUSD position.
5. Confirm there is no pending XAUUSD order.
6. Do not manually trade XAUUSD during the run.
7. Keep the terminal visible and monitor it.

## Expected outputs

- `runtime/reports/stage3_step3b_model_b_controlled_live.json`
- `runtime/execution_live/stage3_step3b_model_b_events.csv`
- `runtime/execution_live/stage3_step3b_order_send_events.csv`
- `runtime/execution_live/stage3_step3b_position_snapshots.csv`
- `runtime/execution_live/stage3_step3b_history_deals.csv`
- `runtime/execution_live/stage3_step3b_history_orders.csv`
- `runtime/reports/stage3_step3b_latest_decision.csv`
- `runtime/state/stage3_step3b_model_b_live_state.json`

## Safety note

If the script stops unexpectedly and MT5 shows an open XAUUSD position, manually close the position immediately and preserve all reports for review. A temporary spread above 800 should no longer terminate v1.1.
