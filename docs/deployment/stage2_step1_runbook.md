# Stage 2 Step 1 — MT5 Environment and Data-Feed Readiness

Purpose: verify the local MetaTrader 5 terminal, demo account, broker symbol and completed M15 data feed before shadow mode. This step must not place orders.

## Safety boundary

- `orders_enabled = false`
- no calls to `order_send`, `order_check`, order history, deal history or position mutation APIs
- account must be demo by default
- `copy_rates_from_pos(..., start_pos=1, ...)` is used so the still-forming current bar is excluded
- broker/runtime config is separate from frozen model configs

## Official MT5 Python API basis

The implementation follows the official MetaTrader5 Python integration pattern:

- `initialize()` establishes the terminal connection and can auto-detect the terminal when no path is provided.
- `account_info()` returns account information as a named tuple and returns `None` on error.
- `symbol_info()` returns instrument data as a named tuple, and `symbol_select()` can make the symbol visible in Market Watch.
- `copy_rates_from_pos()` returns OHLCV bars, where position `0` is the current bar; Stage 2 Step 1 uses position `1` to fetch completed bars only.

## Commands

From repo root:

```powershell
cd C:\Users\Zac\fyp_master_starter
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install -e .
```

Install MetaTrader5 package if missing:

```powershell
python -m pip install MetaTrader5
```

Run normal tests:

```powershell
python -m pytest -m "not tensorflow"
```

Run the formal readiness gate:

```powershell
python scripts\deployment\run_mt5_environment_readiness.py --repo-root .
```

If auto-detection cannot find the terminal, pass the terminal path:

```powershell
python scripts\deployment\run_mt5_environment_readiness.py --repo-root . --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe"
```

## Outputs

- `runtime/reports/stage2_step1_mt5_readiness.json`
- `runtime/reports/stage2_step1_mt5_recent_m15_completed_bars.csv`
- `runtime/reports/stage2_step1_mt5_symbol_candidates.csv`

## Pass criteria

- JSON `status = PASS`
- `formal_gate = true`
- `mt5_used = true`
- `orders_enabled = false`
- `shutdown_called = true`
- account is demo unless explicitly changed in config later
- a valid XAUUSD/GOLD broker symbol is resolved
- at least the configured minimum number of completed M15 bars is returned
- completed bars pass OHLC, duplicate, monotonic and non-negative-volume checks
