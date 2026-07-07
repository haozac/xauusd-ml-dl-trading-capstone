# Stage 2 Step 3A Internal Review

## Scope

This patch freezes broker-specific execution controls and prepares Stage 3 order preflight. It does not touch MT5 directly and does not call order APIs.

## Controls Frozen

- Broker: Dukascopy Bank SA
- Server: Dukascopy-demo-mt5-1
- Symbol: XAUUSD
- Time basis: canonical UTC after Dukascopy MT5 server-time conversion
- Initial volume: 0.01 lot
- Maximum open volume per model: 0.01 lot
- Maximum open positions per model: 1
- Entry spread gate: 800 points
- Stage 3 order request deviation: 200 points
- Filling mode: broker-reported value must be validated by Stage 3 order_check
- Broker-side SL/TP: disabled for initial tiny plumbing test; strategy/risk engine manages exits
- Magic numbers and comments for Model A and Model B

## Safety Review

Checked before release:

1. No MetaTrader5 import.
2. No order_check.
3. No order_send.
4. No live trading state change.
5. Step 2A v1.1 timestamp conversion must be present.
6. 0.01 lot must match broker volume min/step.
7. Capital/leverage warning is explicit when supplied equity is too small.

## Important Design Note

A 0.01 XAUUSD lot usually means approximately 1 ounce of gold exposure when contract size is 100. With gold around 4,100 USD in the current MT5 feed, the notional is above 4,000 USD. A SGD 200 demo balance is therefore too small for the original capstone 10:1 leverage cap. The tiny order test may still be useful as broker plumbing, but final strategy execution should use a larger demo balance or be clearly documented as a constrained operational test.
