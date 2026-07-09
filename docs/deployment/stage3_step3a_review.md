# Stage 3 Step 3A review criteria

Stage 3 Step 3A passes only if:

1. Stage 3 Step 2 v1.1 is a formal PASS.
2. Frozen model/scaler/environment checks pass.
3. Live shadow inference still uses completed M15 bars only.
4. Model B current rules are preserved: entry >= 0.55, exit < 0.50, long-only, no minimum hold, max one successful entry per UTC day.
5. No actual XAUUSD position or pending XAUUSD order exists during the dry-run decision.
6. Spread is within the frozen 800-point gate for actionable events.
7. `order_check` passes whenever a dry-run ENTER_LONG intent is produced.
8. `order_send` is never called.
9. The dry-run state and intent logs are written.
10. At least the configured minimum number of new completed M15 events are observed.

This stage does not prove profitability and does not execute the strategy. It validates the live decision plumbing before Stage 3 Step 3B.
