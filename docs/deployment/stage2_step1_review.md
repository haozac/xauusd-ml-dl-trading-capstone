# Stage 2 Step 1 Internal Review

## Council self-challenge

### Risk: accidentally touching live trading APIs
Mitigation: the Stage 2 readiness module uses `SafeMt5Proxy`, which exposes only read-only inspection methods and records/blocks forbidden trade and order-history methods. The script also reports `orders_enabled = false` and fails if forbidden calls are recorded.

### Risk: accidentally reading the still-forming current M15 bar
Mitigation: the script uses `copy_rates_from_pos(symbol, TIMEFRAME_M15, 1, count)`. Position 0 is the current bar in the official MT5 API, so position 1 is the latest completed bar.

### Risk: wrong broker symbol name
Mitigation: the runtime config uses ordered symbol candidates. The script checks `symbol_info`, calls `symbol_select` only when needed to make the symbol visible, and writes the checked candidates to CSV.

### Risk: real account used by mistake
Mitigation: `require_demo_account` defaults to true. The script fails if the connected account is not detected as demo.

### Risk: weekend or market-closed staleness causing false failure
Mitigation: latest bar staleness is a warning by default, not a hard failure. The readiness report still records the latest completed bar time and age.

### Risk: leaking account details
Mitigation: the report masks login and removes balance/equity/margin/profit fields.

### Limitation
This step verifies MT5 environment and historical/live terminal data-feed availability. It does not yet run the model on live bars and does not prove broker data equivalence against Dukascopy. That belongs to Stage 2 Step 2 and Step 3.
