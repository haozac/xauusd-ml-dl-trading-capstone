# Stage 2 Step 3A Runbook — Broker-Specific Execution Controls Freeze

## Purpose

Freeze broker-specific execution controls before Stage 3 order preflight. This step is still no-order and read-only. It consumes the already validated Stage 2 Step 2A v1.1 MT5 shadow report.

## Prerequisites

1. Stage 2 Step 2A v1.1 is passed and closed.
2. `runtime/reports/stage2_step2a_v1_1_mt5_shadow_logger.json` exists.
3. Orders remain disabled. Do not enable Algo Trading for this stage.

## Commands

```powershell
cd C:\Users\Zac\fyp_master_starter
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest -m "not tensorflow"
```

Run the freeze. Use the actual demo account equity if known:

```powershell
python scripts\deployment\freeze_broker_execution_controls.py --repo-root . --account-equity-sgd 220
```

If you do not want to evaluate equity yet:

```powershell
python scripts\deployment\freeze_broker_execution_controls.py --repo-root .
```

## Outputs

- `runtime/reports/stage2_step3a_broker_execution_controls_freeze.json`
- `runtime/reports/stage2_step3a_broker_execution_controls_summary.csv`
- `config/broker_execution_controls_frozen.yaml`

## Pass Criteria

- Stage 2 Step 2A source report passed.
- Broker/server/symbol match the frozen capstone setup.
- XAUUSD symbol properties allow 0.01 lot.
- Canonical UTC conversion is already applied.
- Orders remain disabled.
- No order_check or order_send is called.

## Capital Warning

0.01 lot is the minimum operational lot size, but for XAUUSD it is still approximately 1 ounce of gold exposure. If the demo balance is only around SGD 200, this likely breaches the capstone 10:1 leverage cap. A larger demo balance is recommended before unattended final strategy execution.
