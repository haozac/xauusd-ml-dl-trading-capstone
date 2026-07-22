# Dual-live rehearsal runtime v1.0

## Purpose

This package adds the missing runtime components needed to test Model A and Model B as independent broker-connected workers on the Windows VPS. It does not alter the frozen model, scaler, feature order, probability thresholds, holding rules, daily caps, or historical artefacts.

The first authorised runtime is **orders-disabled shadow mode**. Order-enabled rehearsal remains blocked until the shadow supervisor, restart reconciliation, logs, and closeout behaviour are reviewed.

## Architecture

```text
Windows Task Scheduler, interactive logged-on session
                    |
                    v
          dual strategy supervisor
             /               \
            v                 v
 Model A worker          Model B worker
 Model A MT5             Model B MT5
 account A               account B
```

The MetaTrader5 Python package keeps one process-global terminal connection. Therefore, each model must run in its own Python process. The supervisor never imports MetaTrader5 and cannot submit an order.

## Files

- `dual_live_state.py`: pure frozen strategy, risk, persistence, duplicate suppression, and reconciliation logic.
- `dual_live_execution.py`: guarded `order_check` and `order_send` plumbing for both roles.
- `dual_live_worker.py`: completed-M15 inference, reconciliation, decision, state, heartbeat, and optional execution.
- `dual_strategy_supervisor.py`: worker lifecycle, heartbeat monitoring, restart budget, STOP and KILL_SWITCH controls.
- `run_dual_strategy_worker.py`: one-role command-line entry point.
- `run_dual_strategy_rehearsal.py`: supervisor command-line entry point.
- `run_restart_reconciliation_test.py`: controlled orders-disabled worker termination and recovery test.
- `initialize_dual_live_local_config.ps1`: creates an ignored VPS-local config with account suffixes.
- `install_dual_rehearsal_task.ps1`: installs a manual Task Scheduler task with no automatic trigger.

## Safety invariants

1. Completed M15 bars only, through the existing Stage 2 shadow inference path.
2. MT5 is checked every 30 seconds, but full feature reconstruction and CNN-LSTM inference run only once for each newly completed M15 bar.
3. Duplicate completed-bar events are skipped and are not appended as permanent decision rows.
4. Exact terminal path and masked account suffix are checked.
5. At most one XAUUSD position and no pending XAUUSD order per role.
6. Foreign magic-number positions block the worker.
7. Model B can never open or adopt a short position.
8. Every order send is preceded by a passing `order_check`.
9. New entries obey the frozen spread gate; exits are not blocked by spread.
10. Reversals validate the entry spread before closing, then close first and open second. If the spread jumps after the confirmed close, the runtime records an explicit flat partial transition rather than pretending the old position still exists.
11. Persistent state is reconciled against actual broker state after restart. Recovered broker transitions conservatively consume the relevant daily counter so a crash cannot grant an extra entry or policy change.
12. Daily-loss, total-drawdown, and kill-switch states request immediate flattening in live mode. A detected KILL_SWITCH also asks the supervisor to stop both workers and review the run as failed.
13. STOP and KILL_SWITCH files are shared controls under the ignored runtime directory.
14. The Task Scheduler task runs only in the logged-on Windows desktop session and has no automatic trigger.
15. Supervisor heartbeats are accepted only when their role and process ID match the currently supervised worker.
16. A supervisor PASS requires both workers to exit cleanly, write PASS final reports, emit STOPPED heartbeats, and confirm broker-flat closeout state.
17. In order-enabled mode, clean stop always re-inspects and flattens the broker even when persistent state says FLAT.

## Runtime folders

Shadow and live tests must use different runtime roots. The template defaults to:

```text
runtime/dual_live_rehearsal_shadow
runtime/dual_live_rehearsal_live
```

This deliberately prevents a shadow virtual state file from being reused as live broker state.

Each role writes:

```text
state.json
heartbeat.json
decisions.csv
latest_decision.json
latest_shadow_snapshot.json
final_report.json
```

Shared output includes:

```text
supervisor_status.json
supervisor_heartbeat.json
supervisor_final_report.json
STOP
KILL_SWITCH
supervisor.lock.json
```

## Required test order

1. Apply patch and run all unit tests.
2. Review and commit the patch on `deployment/dual-live-rehearsal`.
3. Create the ignored local shadow config.
4. Run a short orders-disabled dual supervisor test directly from PowerShell.
5. Verify both heartbeats, completed M15 decisions, zero `order_send` count, correct account bindings, and flat accounts.
6. Run the restart reconciliation test once for Model A and once for Model B while orders remain disabled.
7. Stop the supervisor using the shared STOP file and verify clean reports.
8. Install and run an orders-disabled Task Scheduler soak.
9. Review logs and evidence.
10. Only after a new go/no-go decision, create a separate live config and authorise order-enabled rehearsal.

## Task Scheduler operating model

The task is registered without an automatic trigger. It also does not automatically delete an existing STOP or KILL_SWITCH file. Those files must be reviewed and cleared deliberately before a later run. Before starting it:

- log into the VPS through RDP;
- open or confirm both MT5 terminals;
- confirm the correct demo accounts and live XAUUSD feed;
- confirm both accounts are flat;
- keep the Windows user session logged on;
- start the task manually;
- verify the first supervisor and worker heartbeats;
- disconnect RDP rather than signing out.

The Azure VM must remain running and must not be stopped or deallocated during a rehearsal.

## Known limitations before the live rehearsal

- No broker-side catastrophic stop is added by this package because doing so would change the already-tested order specification. The supervisor therefore depends on worker restart, broker reconciliation, risk-triggered flattening, and emergency closeout.
- The package has not yet been exercised against the actual VPS terminals, Task Scheduler session, or rehearsal accounts. Patch application and unit tests are only the development checkpoint.
- Runtime account binding checks the dedicated terminal path and expected masked login suffix. The formal pre-run dual-account readiness gate remains responsible for proving that the two full account logins are distinct.
- Off-VM evidence backup is not automated by this package and remains a separate deployment gate.

These limitations must remain explicit in the capstone report and final go/no-go review.
