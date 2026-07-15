# Stage 3 Step 3B Closeout Audit Runbook

## Purpose

This is an offline governance closeout for the monitored Model B v1.1 run that was stopped with Ctrl+C while flat. It does not connect to MT5 and cannot call `order_check` or `order_send`.

The audit isolates one explicit run ID from the append-only event CSV and checks the incremental evidence against the persisted state and latest-decision snapshot.

## Frozen run to audit

`stage3_step3b_20260713T114117Z`

## Command

```powershell
python scripts\deployment\audit_model_b_controlled_live_closeout.py `
  --repo-root . `
  --run-id stage3_step3b_20260713T114117Z
```

## Formal PASS requirements

- At least 8 fresh completed M15 events.
- Fresh events are unique and exactly 15 minutes apart.
- Frozen Model B entry threshold remains 0.55.
- Frozen spread gate remains 800 points.
- No probability reaches the entry threshold.
- Every fresh decision remains `HOLD_FLAT`.
- At least one fresh event observes spread above 800, proving v1.1 continues rather than crashes.
- No `order_check` or `order_send` call.
- All logged broker positions and pending orders are zero.
- Persisted state and latest-decision snapshot are flat and agree with the event log.
- Any stale normal report from a previous run is identified and excluded.

## Outputs

- `runtime/reports/stage3_step3b_closeout_audit.json`
- `runtime/reports/stage3_step3b_closeout_summary.csv`
- `runtime/reports/stage3_step3b_closeout_run_events.csv`
- `runtime/reports/stage3_step3b_closeout_fresh_events.csv`

## Safety

No MT5 terminal is required. No trading API is imported or invoked. This audit does not authorise the final 14-calendar-day run.
