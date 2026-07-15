# Stage 3 Step 4A Runbook

## Purpose

Verify that Model A and Model B can be isolated across two MT5 installations and two distinct Dukascopy demo accounts. This gate is read-only and does not expose `order_check` or `order_send`.

## Manual preparation

1. Keep the existing installation as Model B, for example `C:\Program Files\MetaTrader 5\terminal64.exe`.
2. Run the MT5 installer again, choose **Settings**, and install a second copy into a different directory, for example `C:\Program Files\MetaTrader 5 Model A`.
3. Open both `terminal64.exe` files from their own folders.
4. Log Model A into demo account A and Model B into a different demo account B.
5. Use the same broker server, account currency and approximately equal starting balance. At least SGD 1,000 per account is required by this gate.
6. Enable Algo Trading in both terminals.
7. Confirm no XAUUSD positions and no pending XAUUSD orders in either terminal.
8. Copy `config/dual_terminal_runtime_template.yaml` to `config/dual_terminal_runtime.yaml` and correct both paths.

Do not put passwords in the YAML file.

## Commands

```powershell
Copy-Item config\dual_terminal_runtime_template.yaml config\dual_terminal_runtime.yaml
notepad config\dual_terminal_runtime.yaml
python -m pip install -e .
python -m pytest -m "not tensorflow"
python scripts\deployment\run_dual_terminal_readiness.py --repo-root . --config config/dual_terminal_runtime.yaml --server-time-offset-hours 3
```

The Python integration inspects the exact executable paths sequentially. Later simultaneous Model A and Model B runners will use two separate Python processes, one per terminal.

## Outputs

- `runtime/reports/stage3_step4a_dual_terminal_readiness.json`
- `runtime/reports/stage3_step4a_dual_terminal_readiness_summary.csv`
- `runtime/reports/stage3_step4a_terminal_inventory.csv`

## PASS criteria

- Different terminal executable paths and terminal data paths.
- Different demo account logins.
- Same Dukascopy broker/server, SGD currency, margin mode and XAUUSD contract.
- Both terminals connected with Algo Trading/Python trading API available.
- Completed M15 bars and canonical UTC conversion valid.
- No XAUUSD positions or pending orders.
- Separate runtime roots, magic numbers and comments.
- At least SGD 1,000 equity per account and estimated leverage at or below 10:1 for 0.01 lot.
- No `order_check` or `order_send` call.
