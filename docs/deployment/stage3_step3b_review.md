# Stage 3 Step 3B v1.1 Internal Review

## Reason for hotfix

The v1.0 controlled run terminated when the live XAUUSD spread temporarily widened to 890 points, above the frozen 800-point entry gate. No order was sent and the account remained flat, but a temporary market condition should not terminate a continuous runtime.

## Council review summary

Stage 3 Step 3B v1.1 preserves the frozen strategy and broker controls. It changes only runtime handling of temporary spread widening.

Approved protections:

1. The 800-point spread gate remains unchanged.
2. A wide spread blocks a new entry only when Model B would otherwise enter.
3. A wide spread while Model B is below the entry threshold records the normal HOLD_FLAT decision.
4. A wide spread never blocks a risk-reducing exit or end-of-run safety close.
5. A spread widening between the decision snapshot and final pre-send check becomes a non-fatal BLOCK_SPREAD decision.
6. Structural symbol failures, invalid volume controls, account mismatches, permission failures, and order failures remain hard errors.
7. Completed-M15 event processing and duplicate-event prevention remain unchanged.
8. Model B thresholds remain entry 0.55 and exit 0.50.
9. Volume remains 0.01 lot and the daily successful-entry cap remains one.
10. No automatic widening of the frozen spread threshold is introduced.

## Internal verification

Focused tests cover:

- entry blocked above 800 points;
- below-threshold flat decisions continue during a wide spread;
- exits remain allowed during a wide spread;
- live symbol inspection records a wide spread without raising a fatal error;
- existing execution, preflight, and dry-run tests remain green.

## Chairman verdict

Approve v1.1 for a fresh monitored Stage 3 Step 3B rerun. Treat the earlier v1.0 termination as a safely contained runtime-handling defect, not as a trading loss or broker failure.
