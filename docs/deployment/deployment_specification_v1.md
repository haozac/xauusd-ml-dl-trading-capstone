# Deployment Specification v1.2

## Document control

| Field | Value |
|---|---|
| Project | From Prediction to Profit: Designing, Deploying, and Benchmarking ML/DL Trading Bots for Live XAUUSD Paper-Trading |
| Programme context | Master of Science capstone project |
| Document | Stage 0 deployment specification |
| Version | 1.2 |
| Status | Frozen after user approval and final council review |
| Prepared date | 29 June 2026 |
| Approved and frozen date | 29 June 2026 |
| Deployment environment | Windows, Python, MetaTrader 5 demo accounts |
| Architecture decision | Python modular monolith using the official MetaTrader5 package, SQLite, JSONL, and CSV exports |
| Trading scope | MT5 demo or paper trading only; no real-money trading |

## 1. Purpose

This document freezes the research and governance decisions that must be established before deployment code is written. It defines the two systems to be implemented, the common data and model contract, the stage gates, the permitted live safety controls, and the prospective comparison protocol.

The deployment has two objectives:

1. Demonstrate an auditable end-to-end MT5 demo pipeline covering completed M15 bars, feature construction, CNN-LSTM inference, strategy decisions, risk controls, execution, persistence, logging, monitoring, and restart recovery.
2. Prospectively compare the frozen Notebook 7 research system, Model A, with a predefined post-holdout overlay revision, Model B Version 2.

Operational deployment and preliminary forward comparison are the objectives. A short paper-trading period cannot establish persistent or long-term profitability.

## 2. Research-integrity position

Notebook 7 is complete and permanently locked. Its official finding is negative:

- Holdout period: January 2025 to March 2026
- Holdout ROC-AUC: approximately 0.5158
- Net return at 1.0 basis point one-way cost: approximately -13.08%
- Sharpe at 1.0 basis point: approximately -1.56
- Maximum drawdown: approximately -15.14%
- Total drawdown stop: triggered in January 2026

The deployment must not rerun, replace, or reinterpret Notebook 7 as a new holdout. Model B is explicitly post-holdout. Any historical Model B replay is diagnostic and cannot replace the official negative result.


### 2.1 Evidence inventory and traceability

The Stage 0 decisions are grounded in the following frozen Notebook 7 artefacts rather than in filename assumptions or narrative summaries alone:

| Claim or contract | Primary artefact | Use in Stage 0 |
|---|---|---|
| Holdout completion and lock | `configuration/holdout_evaluation_complete.json` | Confirms the final evaluation is complete, negative, and not to be rerun as another holdout |
| Artefact integrity | `configuration/evaluation_artefact_manifest.json` | Supplies the recorded SHA-256 values for the model, scaler, preprocessing files, predictions, metrics, and strategy log |
| Predictive metrics | `tables/final_holdout_classification_metrics.csv` | Supports the reported ROC-AUC, balanced accuracy, calibration-loss comparisons, and weak-signal conclusion |
| Trading and cost metrics | `tables/final_holdout_trading_metrics_by_cost.csv` | Supports gross return, net return, drawdown, turnover, risk-stop, and cost-sensitivity claims |
| Per-bar strategy behaviour | `tables/final_holdout_strategy_bar_log_1bps.csv` | Supports position, turnover, gap, stop, equity, drawdown, and long-versus-short contribution reconstruction |
| Frozen Model A overlay | `configuration/selected_overlay.json` and `tables/selected_final_overlay.csv` | Defines the exact Model A thresholds, hold period, daily change cap, and 2024 selection result |
| Overlay fragility | `tables/overlay_validation_grid_at_1bps.csv` | Confirms that only one of 36 configurations was viable under the recorded selection rule |
| Model B entry-score activity | `tables/overlay_validation_predictions.csv` | Supports the descriptive count of scores at or above 0.55; it is not used to claim optimality |
| Model and preprocessing contract | Frozen `.keras` model, scaler, feature-order JSON, parameter JSON, and `configuration/selected_epoch.json` | Defines the exact 48-by-51 inference pipeline and selected epoch |

Derived quantities must record their formula and source file. For example, long- and short-held gross contributions are reconstructed from the strategy bar log by grouping `gross_log_return` by held `position` and compounding each grouped log-return sum.

## 3. Common frozen model contract

Model A and Model B must use the same frozen model pipeline:

| Item | Frozen value |
|---|---|
| Model family | CNN-LSTM |
| Track | Vanilla volume-assisted |
| Timeframe | M15 |
| Sequence length | 48 contiguous M15 bars |
| Feature count | 51 |
| Feature order | Exact order in the frozen feature-order JSON |
| Scaler | Frozen Notebook 7 StandardScaler |
| Model | Frozen Notebook 7 Keras model |
| Selected epoch | 10 |
| Model parameters | 29,825 trainable parameters; architecture and training parameters recorded in the frozen parameter JSON |
| Input dtype | float32 before scaling, float32 after scaling, float32 model input |
| Sequence rule | A sequence is valid only when every consecutive timestamp difference equals 15 minutes |
| Current bar | The currently forming MT5 bar is prohibited from inference |

The artefact filenames and SHA-256 values are recorded in both YAML configurations. A hash mismatch is a hard deployment failure. The model must never be retrained or silently replaced during this deployment study.

## 4. Model A: frozen research benchmark

Model A reproduces the complete Notebook 7 research strategy:

- Long when `p_up >= 0.53`
- Short when `p_up <= 0.47`
- Flat signal when `0.47 < p_up < 0.53`
- Minimum holding period of 3 eligible M15 prediction bars after an overlay-driven change
- Maximum 3 overlay-driven position-change events per UTC day
- Daily loss stop of 2%
- Total drawdown stop of 15%
- Gap, daily-stop, total-stop, and emergency exits are never blocked by the holding rule or daily change cap
- A reversal is one policy-counted position-change event but two turnover units
- Main historical parity cost is 1.0 basis point one way per turnover unit

Model A cannot be tuned. Any change to its model, preprocessing, probability thresholds, minimum holding period, daily position-change cap, or historical risk semantics would create a different system and invalidate the research-to-deployment parity claim.

## 5. Model B Version 2: frozen post-holdout overlay hypothesis

Model B uses the same frozen probability from the common CNN-LSTM pipeline but replaces the long-short overlay with one predefined long-flat state machine:

- Flat to long entry when `p_up >= 0.55`
- Remain long while `p_up >= 0.50`
- Exit to flat when `p_up < 0.50`
- Short positions are prohibited
- Maximum one successful new long entry per UTC day
- No mandatory minimum holding period
- Exiting does not restore the daily entry allowance
- A rejected technical order does not consume the daily entry allowance; a confirmed fill establishing a long position does
- Risk, gap, session-safety, reconciliation, and emergency exits always override the overlay
- Daily loss stop exactly 2%, using the same live risk basis and action as Model A
- Total drawdown stop exactly 15%, using the same live risk basis and action as Model A
- The same broker-valid per-position sizing rule as Model A; Model B may have lower realised exposure only because it trades less

The exact rationale, evidence strength, limitations, and falsification criteria are documented in `docs/deployment/model_b_v2_rationale.md`.

Model B is not claimed to be optimal or historically proven profitable. The thresholds are frozen to avoid repeated post-holdout search. Its effectiveness must be observed prospectively.

## 6. Risk and exposure governance

### 6.1 Independent strategy risk

Each system must maintain independent:

- Intended position
- Actual position
- Realised and unrealised PnL
- Start-of-day equity
- Running equity peak
- Current drawdown
- Daily-stop state
- Total-stop state
- Entry or position-change counters

For the live comparison, both systems use the same risk thresholds and semantics: a 2% daily loss stop and a 15% total drawdown stop. The daily boundary is UTC. The daily stop compares independent strategy equity with its UTC start-of-day equity. The total stop compares independent strategy equity with its running peak. In separate accounts, broker account equity is the primary execution ledger. Under a one-account hedging fallback, strategy-specific realised and mark-to-market PnL are attributed by magic number and reconciled with the broker account. A triggered risk exit must be requested immediately and must never be blocked by overlay holding or entry rules.

### 6.2 Exposure and comparison fairness

The historical statement of up to 5% equity risk per trade is not suitable as the initial live sizing rule because the strategy has no fixed price stop from which per-trade loss can be calculated. Initial demo deployment must therefore use conservative broker-valid exposure.

For the formal comparison, Model A and Model B must use the same broker-valid per-position sizing rule, normally the minimum valid XAUUSD lot on equal-capital demo accounts. This isolates the overlay difference. Model B may exhibit lower average exposure because it enters less frequently, but it must not be assigned a smaller trade size merely to improve its drawdown. The exact lot, maximum notional-to-equity ratio, and maximum margin-use ratio remain pending until broker inspection. If the broker minimum lot breaches a frozen safety cap, the system must block trading rather than exceed the cap.

### 6.3 Emergency behaviour

New entries must be blocked when any of the following is unresolved:

- MT5 disconnection or stale tick
- Missing expected bar
- Invalid feature or insufficient warm-up
- Artefact or configuration hash mismatch
- Intended-position and actual-position mismatch
- Unreconciled prior order
- Daily stop or total stop
- Manual emergency kill switch

## 7. Research parity versus live execution safety

The deployment maintains two clearly labelled layers:

1. **Research-equivalent virtual ledger**: reproduces Notebook 7 rules and 1.0 basis point accounting for parity and comparison.
2. **Broker-executed ledger**: records actual Bid, Ask, spread, commission, swap, slippage, fills, and broker-compatible safety actions.

### 7.1 Timestamp convention

Historical dataset timestamps label the end of each M15 interval. MT5 rates identify the bar opening time. The deployment stores both `bar_open_utc` and `bar_close_utc`, where the canonical close is provisionally `bar_open_utc + 15 minutes` after empirical timestamp validation.

No broker offset is hardcoded. Timestamp alignment is measured during Stage 2 because cyclical time features make clock errors material.

### 7.2 Forming bars

MT5 position zero is the current bar. It must not be used for inference. A bar is accepted only after its expected closing time has passed and its identity has not already been processed.

### 7.3 Gap handling

Offline parity reproduces Notebook 7 exactly: a gap between eligible prediction timestamps forces the position flat at the next eligible timestamp and resets the holding state.

Live execution cannot retrospectively close at a pre-gap price. Therefore, both systems use one common broker-specific session-safety layer, to be frozen after Stage 2 inspection. A safety flattening action is logged separately from the strategy overlay and is reproduced in both systems under the same conditions. The research-equivalent virtual ledger remains available to show the difference between frozen strategy intent and executable broker safety.

### 7.4 Costs

Offline parity uses the frozen 1.0 basis point one-way cost. Live reporting records actual spread, slippage, commission, and swap. Both cost views are retained and never mixed silently.

## 8. Broker-dependent controls deferred to Stage 2

The following remain unset until the actual MT5 demo environment is inspected:

- Broker and server
- Exact XAUUSD symbol and suffix
- Account margin mode: hedging or netting
- MT5 terminal path and account mapping
- Tick volume versus real volume source
- Minimum lot, lot step, maximum lot, point, digits, and contract size
- Maximum entry spread
- Slippage or deviation limit
- Broker-supported filling mode
- Session-flatten policy
- Commission and swap treatment
- Safe technical retry rules
- Magic numbers

These controls may be selected only for safety, compatibility, and fair execution. They must not be selected by observing which values make Model B more profitable. They will be stored in a separate common runtime file, `config/mt5_runtime_frozen.yaml`, created and hashed during Stage 2. The frozen Model A and Model B strategy YAML files will not be edited to insert broker values.

## 9. Deployment stages and gates

### Stage 1: offline replay and parity

Required outcomes:

1. Artefact hashes and environment compatibility pass.
2. The 51 features and 48-bar sequences reproduce the official dataset.
3. TensorFlow probabilities reproduce saved predictions without unexplained threshold-changing differences.
4. Model A reproduces the Notebook 7 strategy log and metrics.
5. Model B obeys its frozen state machine.
6. SQLite, decision IDs, duplicate prevention, and restart recovery pass.

No MT5 connection or order submission is allowed.

### Stage 2: live MT5 shadow mode

Required outcomes:

1. Broker capabilities and timestamps are validated.
2. Only completed bars are processed.
3. Live feature, model, strategy, and virtual-ledger processing runs continuously.
4. Restart and disconnection drills pass.
5. Missing bars fail closed.
6. Broker-dependent controls are frozen.

No `order_send` call is allowed.

### Stage 3: controlled MT5 demo execution

Required outcomes before formal operation:

1. Dry-run order mapping passes.
2. Minimum-lot open and close integration tests pass.
3. Successful execution requires an acceptable MT5 result, a recorded deal, and confirmed actual position.
4. Rejected orders, duplicate orders, restarts, and reconciliation are tested.
5. The council approves the formal run.

### Stage 4: evaluation and reporting

The final evaluation reports both operational and trading outcomes while retaining the official negative holdout finding.

## 10. Prospective Model A versus Model B protocol

The formal comparison must use:

- The same forward start and end timestamps
- The same broker feed
- The same completed bars and shared model probability
- The same broker-valid per-position sizing rule, with returns additionally reported on a normalised basis if account capitals differ
- Independent strategy state and PnL
- Frozen configurations
- No strategy changes during the observation window
- At least 14 calendar days of automatic operation where feasible

### Primary outcomes

- Uptime and expected-bar coverage
- Missing and duplicate bars
- Restart and reconciliation performance
- Order acceptance, rejection, fill, and position-confirmation rates
- Turnover and active exposure
- Spread, slippage, commission, and swap
- Drawdown and risk-stop behaviour
- Strategy disagreement rate

### Trading outcomes

- Gross return
- Net return
- Maximum drawdown
- Worst daily loss
- Number of entries
- Turnover
- Sharpe and Sortino
- Daily net-return difference: Model B minus Model A

Because of the short run, all trading outcomes are preliminary and statistically underpowered.

## 11. Change control

After Stage 0 approval:

- Model A core cannot change.
- Model B probability and position rules cannot change.
- Broker-dependent controls are recorded in a separate Stage 2 runtime configuration and frozen before the formal run.
- The Stage 0 Model A and Model B YAML files remain immutable after approval.
- Their SHA-256 values are recorded externally in the Stage 0 freeze manifest or configuration-version table; a YAML file does not contain its own hash.
- Any implementation correction must be documented and tested.
- Any material strategy change creates a new version and a new observation window.
- Losing or profitable short-term results are not valid reasons to modify a frozen strategy.

## 12. Non-goals

This deployment will not include:

- Real-money trading
- Another model-training or hyperparameter study
- Another holdout evaluation
- Cloud microservices
- A large custom web platform
- Historical threshold search to manufacture profitability
- Claims of long-term profitability from the paper-trading period

## 13. Stage 0 approval checklist

- [x] Model A configuration matches Notebook 7 exactly.
- [x] Model B Version 2 state machine is accepted as the single post-holdout hypothesis.
- [x] Model B limitations are understood.
- [x] Research parity and live safety layers are distinguished.
- [x] Broker-dependent settings are clearly deferred to a separate common runtime configuration.
- [x] Prospective metrics and change-control rules are accepted.
- [x] The two YAML configurations are approved and marked frozen.

## References

1. Notebook 7 executed output and frozen artefact manifest, configuration fingerprint `55b349c6d957d26e528c0546c78fb7363bf515a93ce705c6a4c0e97862aff326`.
2. Bailey, D. H., Borwein, J. M., López de Prado, M., and Zhu, Q. J. *The Probability of Backtest Overfitting*. Journal of Computational Finance, 20(4), 39–69, 2017. DOI: 10.21314/JCF.2016.322.
3. Magill, M. J. P., and Constantinides, G. M. *Portfolio Selection with Transactions Costs*. Journal of Economic Theory, 13(2), 245–263, 1976. DOI: 10.1016/0022-0531(76)90018-1.
4. Guasoni, P., and Muhle-Karbe, J. *Portfolio Choice with Transaction Costs: A User's Guide*. Paris-Princeton Lectures on Mathematical Finance 2013, 169–201. DOI: 10.1007/978-3-319-00413-6_3.
5. MetaQuotes. *MetaTrader 5 Python Integration documentation*: bar retrieval, symbol information, order checks, order submission, positions, and deal history.
