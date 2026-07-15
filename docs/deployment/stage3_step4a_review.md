# Stage 3 Step 4A Internal Review

- Scope is limited to dual-installation, dual-account, broker, completed-bar, time-normalisation, account-capital and isolation checks.
- `order_check` and `order_send` are forbidden by the runtime proxy.
- The exact `terminal64.exe` path is supplied to `mt5.initialize` for each role.
- Connections are inspected sequentially because one MetaTrader5 Python module connection is active at a time. Simultaneous strategy operation will use separate Python processes.
- Full account numbers are not written. Only masked login values are retained.
- Model A and Model B use separate runtime roots, magic numbers and comments.
- Step 4A does not authorise the final paper-trading run. Step 4B remains a short dual-terminal shadow synchronisation gate.
