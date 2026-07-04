# Stage 1 Step 4 Internal Council Review

## Scope reviewed

The patch implements Stage 1 Step 4 only: frozen Model A historical trading replay parity. It does not implement Model B, MT5, shadow mode, live execution, order placement, or any parameter change.

## Methodology review

The replay engine is driven by frozen rules and prediction probabilities. The saved Notebook 7 bar log is used only as a comparison target. This avoids a circular test.

The script enforces the Step 3 inference-parity report as a precondition. Therefore Step 4 can reasonably use frozen prediction CSVs to isolate trading-engine behaviour, because Step 3 already proved that the deployment model path regenerates the same probabilities.

## Rules implemented

- Model A thresholds: 0.53 long and 0.47 short
- Minimum hold: 3 eligible prediction bars
- Daily policy cap: 3 policy-counted changes per UTC day
- Reversal: 1 policy event, 2 turnover units
- Gap exit: forced flat at the first eligible prediction after a prediction-time gap
- Daily loss stop: triggered after realised row net return and active from the next eligible decision until next UTC day
- Total drawdown stop: triggered after realised row net return and active from the next eligible decision for the rest of the run
- Costs: one-way bps cost multiplied by absolute position turnover

## Error handling review

Expected failures return exit code 2 and write a JSON report. Unexpected software errors return exit code 3 and also attempt to write a JSON report.

Validation failures include:

- Missing prior Step 3 formal PASS
- Modified Stage 0 frozen files
- Modified Notebook 7 critical artefacts
- Missing or changed prediction references
- Invalid prediction timestamps or probabilities
- Missing metrics/bar-log evidence sources
- Bar-log timestamp misalignment
- Missing required comparable strategy-log columns
- Aggregate metric mismatch
- Per-row bar-log mismatch

## Code-quality review

The patch uses typed dataclasses for trading rules, replay metrics and comparison reports. It keeps trading replay in `src/capstone_trading/evaluation/trading_replay.py` and the CLI orchestration in `scripts/deployment/run_model_a_replay_parity.py`.

The replay code does not import TensorFlow and does not use model outputs directly. This is intentional because Step 3 already tested inference parity and Step 4 isolates overlay/accounting parity.

## Local review performed in this environment

The review environment does not contain the user's gitignored Notebook 7 artefacts, so the formal Step 4 gate must run on the user's machine. The following local checks were performed:

- Python compilation: PASS
- Unit tests for replay rules and comparison helpers: PASS
- Non-TensorFlow test suite with unavailable real artefacts skipped: PASS

Observed local result:

```text
36 passed, 3 skipped, 2 deselected
```

The skipped tests require the user's local gitignored Notebook 7 artefacts.

## Council verdict

| Area | Verdict |
|---|---|
| Quantitative trading semantics | Approved for local formal run |
| Risk and stop semantics | Approved for local formal run |
| Cost accounting | Approved for local formal run |
| Methodological non-circularity | Approved |
| Error handling | Approved |
| MT5 scope separation | Approved |

## Chairman verdict

The patch is approved for integration and local execution. Stage 1 Step 4 itself remains open until the user's formal JSON report returns `status = PASS` and `formal_gate = true`.
