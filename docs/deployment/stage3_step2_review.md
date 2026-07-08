# Stage 3 Step 2 v1.1 review criteria

Stage 3 Step 2 v1.1 intentionally sends one additional tiny demo order to verify broker open/close plumbing and to fix the v1.0 history-audit weakness.

It passes only if:

1. One 0.01-lot BUY order is sent and accepted.
2. One matching XAUUSD BUY position is verified after open.
3. One close SELL order is sent against the same position ticket.
4. No XAUUSD position remains after close.
5. No pre-existing XAUUSD exposure existed before open.
6. Capstone leverage review passes.
7. Broker history is recovered from `history_deals_get` and/or `history_orders_get`.
8. The JSON report shows `formal_gate = true`.

This step is still not strategy automation. It only proves controlled broker open/close plumbing plus broker-history audit capture.
