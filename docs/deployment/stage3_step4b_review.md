# Stage 3 Step 4B Design Review

## Approved design

Step 4B uses two operating-system processes because the MetaTrader5 Python package maintains process-global terminal state. One process per terminal prevents cross-attachment and permits both installations to operate concurrently.

The gate deliberately separates three concerns:

1. Independent completed-M15 data and feature reconstruction
2. Shared frozen CNN-LSTM inference parity
3. Role-specific frozen overlay output

No broker order is checked or sent.

## Defensive controls

- Exact terminal executable paths inherited from the latest Step 4A PASS
- Exact masked account identities checked in every worker
- Run-specific role directories and append-only observation logs
- Atomic JSON state and latest-observation files
- Completed-bar duplicate suppression
- Canonical UTC conversion before features
- SHA-256 digests for rates, feature frames and scaled sequences
- Model, scaler and feature-order fingerprints compared across workers
- Account position and pending-order checks on every fresh event
- Calculation-only broker economics proxy that blocks `order_check` and `order_send`
- Parent stop file and controlled worker shutdown
- Hard 90-minute timeout

## Economic metadata handling

A zero `trade_tick_value` metadata field is not accepted as proof of broken economics by itself. Step 4B verifies the broker calculation engine directly through positive hypothetical margin and profit calculations on both terminals, then compares the results.

## Deferred validation

Step 4B is shadow-only and does not prove real dual-model order execution. Full live order lifecycle, watchdog, restart, reboot recovery and unattended operation are validated during Step 4C and the requested one-full-day VPS soak test before the final 14-calendar-day run.


## v1.1 portability correction

The v1.0 production worker command was correct, but one unit test compared a native Windows path against a hard-coded POSIX-style suffix. On Windows, `Path` stringification uses backslashes, so the test failed even though the worker received the exact requested path. v1.1 replaces the separator-sensitive suffix assertion with exact platform-native path assertions for the worker script, repository root, configuration, worker root and stop file. No production runtime logic, model rule, safety control or Step 4B command changed.


## v1.2 live API correction

The first live v1.1 execution stopped before the shadow loop because both workers received `0.0` from `order_calc_margin`. The MetaQuotes Python reference defines the margin and profit arguments as required unnamed parameters. v1.1 tried keyword arguments first and only fell back after `TypeError`; the live extension returned zero instead of raising. v1.2 calls both calculation functions positionally only, records the call mode in evidence, and includes `last_error()` in any remaining calculation failure.

The parent economic comparator also no longer labels a missing economic payload as a forbidden order call. A forbidden-order failure is now emitted only when there is positive evidence that `order_check`, `order_send`, or another forbidden method was actually attempted.
