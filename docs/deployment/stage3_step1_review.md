# Stage 3 Step 1 Internal Review

## Scope

This patch performs a live MT5 order-permission and `order_check` preflight using the frozen Stage 2 Step 3A broker execution controls. It does not call `order_send`, does not open a position, and does not modify broker state except for the broker-side validation request inherent in `order_check`.

## Safety controls

- `order_send` is blocked by a guarded MT5 proxy.
- Position/history APIs are blocked in this step.
- Only `initialize`, `terminal_info`, `account_info`, `symbol_info`, `symbol_info_tick`, `order_calc_margin`, and `order_check` are allowed.
- The script writes a JSON report and CSV of `order_check` attempts.
- BUY and SELL checks are both run by default so Model B long-only and future Model A long/short execution are both preflighted.

## Broker-specific handling

The patch treats MT5 `symbol_info.filling_mode` as symbol filling permission flags, not as a direct `type_filling` value. It derives candidate order filling policies and validates them with `order_check`. For the current Dukascopy XAUUSD setup, IOC is expected to be the first candidate when `filling_mode = 2` and the symbol is market execution.

## Known limitation

A passed `order_check` does not mean an order has been sent, and does not guarantee final live fill quality. It only validates request structure, permissions, and funds/margin sufficiency in the current broker environment. Stage 3 Step 2 is still required for one tiny controlled open/close demo order.


## v1.1 hotfix note

This hotfix changes the MT5 `order_check` and `order_calc_margin` calls to use named arguments first, with positional fallback. The live v1.0 run reached MT5 successfully but the MetaTrader5 module returned `(-2, "Unnamed arguments not allowed")` for the order-check calls. The hotfix also keeps human-readable filling metadata outside the actual MqlTradeRequest dictionary submitted to MT5 and fixes the report-level terminal-initialized flag after shutdown.
