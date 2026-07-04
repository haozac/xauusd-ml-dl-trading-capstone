# Stage 1 Step 2 Internal Council Review

## Scope reviewed

The review covers only historical M15 feature and sequence parity. It does not approve live MT5 data, model inference, strategy logic, or order execution.

## Evidence used

- Official `06_prepare_m5_m15_relative_datasets_v2.py` formulas
- Executed Notebook 7 sequence and scaling logic
- Frozen 51-feature JSON
- Frozen StandardScaler
- Audited M15 bars-with-volume parquet
- Audited M15 volume-assisted model-ready parquet
- Saved 2024 and holdout prediction tables

## Quantitative trading review

Approved. The implementation reproduces the official relative-feature definitions rather than substituting a third-party indicator library. RSI uses rolling mean gains and losses, ATR uses the rolling mean of true range, Bollinger and rolling volatility use pandas sample standard deviation, and EMA uses `adjust=False` with the original minimum periods.

## Time-series methodology review

Approved. One-bar targets are removed when the next timestamp is not exactly 15 minutes later. Sequence windows are accepted only when every internal timestamp interval is exactly 15 minutes. Sequence construction is applied separately to each chronological partition, matching Notebook 7 and preventing history from a previous partition from leaking into the next partition.

## ML deployment and MLOps review

Approved. The code validates the frozen feature order, casts the feature DataFrame to float32 before scaler transformation, returns float32 scaled data, and verifies the frozen scaler contract. Large sequence tensors are not materialised for the full dataset; sequence start and end positions are planned vectorially and later batches can be generated with bounded memory.

## Reliability and audit review

Approved. Historical inputs are verified by file size and SHA-256 before use. JSON and CSV reports are written atomically. Expected validation failures return exit code 2, while unexpected defects return exit code 3. Source parquet files are opened read-only and are never modified.

## Risk-management review

Approved for this limited step. No trading or risk decision is produced. The output is only a preprocessing and sequence parity gate.

## MT5 execution review

Not applicable to Step 2. Broker timestamps, tick volume, session rules, and forming-bar exclusion remain Stage 2 concerns. Passing historical parity must not be interpreted as proof of live feed parity.

## Independent test outcomes in the review environment

```text
Python compilation: PASS
Ruff static analysis: PASS
Non-TensorFlow tests: 27 passed, 1 deselected
Formal Step 2 script: PASS
```

The review environment uses a different scikit-learn version and therefore emits the expected model-persistence warning when opening the frozen scaler. The user's formal Python 3.11 environment is pinned to scikit-learn 1.6.1 and should not emit that warning.

## Verified parity outcomes

```text
M15 bars: 240,636
Model-ready rows: 237,001
Feature columns: 51
Target columns: 3
Maximum reconstructed numerical difference: 3.886193938266308e-15
Values exceeding 1e-12: 0
Full-dataset valid sequences: 108,391
2024 sequence endpoints: 11,005, exact timestamp and target alignment
Holdout sequence endpoints: 13,794, exact timestamp and target alignment
```

## Known limitations

1. The historical volume feature is based on the Dukascopy source. MT5 broker volume domain shift is not addressed here.
2. Full model-probability parity is intentionally deferred to Stage 1 Step 3.
3. Live EMA warm-up and restart reconstruction will be separately validated in shadow mode.
4. Passing this step proves implementation parity with the historical dataset, not profitability.

## Chairman verdict

Approved for user execution. Step 2 may be closed only after the user's pinned deployment environment produces a formal `PASS` report.
