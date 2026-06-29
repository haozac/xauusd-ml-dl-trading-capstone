# Model B Version 2 Rationale v1.2

## Document control

| Field | Value |
|---|---|
| Project | Master of Science XAUUSD trading capstone |
| Strategy | Model B Version 2 |
| Version | 1.2 |
| Status | Frozen after user approval and final council review |
| Prepared date | 29 June 2026 |
| Approved and frozen date | 29 June 2026 |
| Classification | Post-holdout overlay revision; not a new predictive model and not an untouched historical evaluation |

## 1. Decision statement

Model B Version 2 uses the exact frozen Notebook 7 CNN-LSTM probability but replaces Model A's long-short-flat overlay with one long-flat state machine:

- Enter long when `p_up >= 0.55`
- Remain long while `p_up >= 0.50`
- Exit when `p_up < 0.50`
- No short positions
- Maximum one successful new entry per UTC day
- No mandatory minimum holding period
- Risk and safety exits always permitted

This configuration is a predefined engineering hypothesis. It is not described as optimal, calibrated, or historically proven profitable.

This separate rationale document is not required by the runtime software. It is retained as an academic traceability artefact so that each post-holdout decision, its evidence strength, and its limitations can be reviewed independently in the report and viva.

## 2. Why a second strategy is methodologically useful

Notebook 7 answered the historical research question: the selected candidate failed to generalise on the untouched holdout. Deployment has a different but connected question:

> Can a predefined overlay revision reduce diagnosed execution and risk weaknesses when both systems receive the same new model probability and broker feed?

Using the same model, scaler, features, and sequence in both systems isolates the overlay. Retraining the neural network and changing the overlay simultaneously would confound the comparison.

## 3. Evidence from Notebook 7

### 3.1 Weak but non-zero ranking signal

The final holdout ROC-AUC was approximately 0.5158 and balanced accuracy was approximately 0.5081. These values do not establish a reliable trading edge. Model B therefore cannot be justified as a guaranteed profitability repair.

### 3.2 Short-side contribution was harmful

Independent reconstruction of the 1.0 basis point strategy log showed:

- Long-held bars: approximately +5.78% compounded gross contribution
- Short-held bars: approximately -8.16% compounded gross contribution

This supports removing short exposure as a targeted risk hypothesis. It does not prove that long-only trading will be profitable, because the Notebook 7 diagnosis also found that simply disabling shorts under the original overlay remained negative.

### 3.3 Transaction costs materially worsened performance

At 1.0 basis point one-way cost:

- Gross return: approximately -2.85%
- Net return: approximately -13.08%
- Turnover units: 1,112
- Physical position transitions: 1,070

Costs were therefore not the only cause of failure, because gross performance was also negative, but they materially amplified the loss. A lower-frequency overlay is a reasonable engineering response.

### 3.4 Target and mandatory-holding mismatch

The model predicts the direction of the next eligible M15 return, while Model A can require a position to remain unchanged for 3 eligible bars. The predictive horizon is therefore 15 minutes while the minimum commitment can be 45 minutes. Model B removes the mandatory hold so that an exit can occur once the model no longer favours an upward direction.

### 3.5 Overlay fragility

Only one of the 36 Notebook 7 validation overlays was viable. This indicates that the chosen historical trading rule was fragile and supports testing a simple alternative prospectively rather than claiming the original overlay is robust.


### 3.6 Claim-to-evidence matrix

| Model B justification claim | Exact evidence source | Derivation or interpretation |
|---|---|---|
| Predictive signal was weak | `tables/final_holdout_classification_metrics.csv` | Directly reads holdout ROC-AUC and balanced accuracy; does not claim a reliable trading edge |
| Short exposure was harmful | `tables/final_holdout_strategy_bar_log_1bps.csv` | Group `gross_log_return` by held `position`; compound as `exp(sum(log_returns)) - 1`, giving approximately +5.78% for long-held bars and -8.16% for short-held bars |
| Costs materially amplified losses | `tables/final_holdout_trading_metrics_by_cost.csv` | Uses the 1.0-bps row: gross return approximately -2.85%, net return approximately -13.08%, and 1,112 turnover units |
| The 0.55 score remains operationally active | `tables/overlay_validation_predictions.csv` | Counts 256 of 11,005 eligible bars and 140 of 258 prediction days with `p_up >= 0.55`; this is a selectivity check, not profitability optimisation |
| Model A overlay was fragile | `configuration/selected_overlay.json` and `tables/overlay_validation_grid_at_1bps.csv` | Reads `validation_viable_candidates = 1` from the 36-candidate grid |
| Three-bar hold mismatched the target horizon | Frozen target definition plus `configuration/selected_overlay.json` | Compares the next-eligible-M15 target with `min_hold_bars = 3`; this motivates removing a mandatory hold but does not prove profitability |
| Model B must keep the same predictive pipeline | Frozen model, scaler, feature-order JSON, parameter JSON, and selected epoch | Ensures any prospective difference begins at the overlay and execution state |

This matrix separates direct observations, derived statistics, engineering interpretations, and limitations. No row treats the Model B thresholds as empirically optimal.

## 4. Rule-by-rule justification

### 4.1 Same CNN-LSTM, scaler, features, and sequence

**Decision:** Keep the predictive pipeline unchanged.

**Reason:** This creates a controlled comparison. Any difference between Model A and Model B begins at the overlay and subsequent execution state, not at model training.

**Evidence strength:** Strong experimental-design justification.

### 4.2 Long-only

**Decision:** Model B can hold `0` or `+1`, never `-1`.

**Reason:** The short side made a substantial negative gross contribution in the holdout, while long-held bars made a positive gross contribution. Removing shorts directly addresses a diagnosed weakness.

**Limitation:** Long-only under revised rules is still not guaranteed to be profitable. It may retain weak or uneconomic long signals.

**Evidence strength:** Strong diagnostic evidence, limited prospective confirmation.

### 4.3 Entry threshold `p_up >= 0.55`

**Decision:** Require a stricter model score for a new long entry than Model A's `0.53` threshold.

**Reasons:**

1. `0.55` was already included in the predefined Notebook 7 threshold grid, so it is not an arbitrary new precision value invented after seeing the holdout.
2. A stricter entry score is consistent with the risk objective of reducing marginal entries, exposure, and turnover.
3. On the 2024 overlay-validation predictions, scores at or above `0.55` occurred on 256 of 11,005 eligible bars, approximately 2.33%, and on 140 of 258 active prediction days, approximately 54.3%. This descriptive check shows that the threshold is selective without making the system permanently inactive.

**Important interpretation:** Notebook 7 calibration was weak. The value `0.55` is a score cutoff, not evidence of a true 55% probability of an upward return.

**Limitation:** It is not proven to be the profit-maximising or statistically optimal threshold.

**Evidence strength:** Defensible governance and selectivity rationale, not optimality evidence.

### 4.4 Exit threshold `p_up < 0.50`

**Decision:** Once long, remain long while the score is at least the conventional directional boundary and exit below it.

**Reasons:**

1. The score below `0.50` means the model no longer favours the upward class.
2. Different entry and exit thresholds create hysteresis. The 0.05 buffer reduces rapid entry-exit cycling around a single threshold.
3. Transaction-cost research shows that proportional costs can justify no-trade regions rather than continuous rebalancing. Model B uses this principle as an engineering analogy, not as proof that the exact 0.55/0.50 boundaries are mathematically optimal.

**Limitation:** Poor calibration and weak signal can still make the exit timing ineffective.

**Evidence strength:** Strong behavioural rationale, exact numerical boundary remains a governance choice.

### 4.5 Maximum one successful new entry per UTC day

**Decision:** A confirmed flat-to-long entry consumes the daily allowance. Exiting does not restore it. Rejected technical requests do not consume it unless a fill established the position.

**Reasons:**

1. Directly constrains turnover and repeated re-entry.
2. Limits daily transaction-cost exposure.
3. Is simple to audit and explain.
4. Avoids an additional parameter grid under the capstone deadline.

**Limitation:** One is a conservative governance cap, not an empirically estimated optimum. It may miss legitimate later signals on the same day.

**Evidence strength:** Strong risk-control rationale, limited optimality evidence.

### 4.6 No mandatory minimum hold

**Decision:** Normal exits can occur at the next eligible decision bar when `p_up < 0.50`.

**Reasons:**

1. Aligns decisions more closely with the next-bar prediction target.
2. Avoids forcing continued exposure after the upward score has disappeared.
3. The daily entry cap limits repeated re-entry after an exit.

**Limitation:** The system may still hold for many bars if the score remains above `0.50`. This rule does not convert the target into a return-magnitude forecast.

**Evidence strength:** Strong diagnostic and design rationale.

### 4.7 Identical risk thresholds and per-position sizing

**Decision:** Model B uses the same 2% daily loss stop, the same 15% total drawdown stop, the same live risk-accounting basis, and the same broker-valid per-position sizing rule as Model A.

**Reason:** Equal thresholds and sizing isolate the effect of the overlay. Model B may have lower realised exposure because it trades less, but it must not appear safer merely because it was allocated a smaller lot. Notebook 7 also found that relaxing the drawdown stop would have worsened the historical outcome.

**Evidence strength:** Strong experimental-fairness and risk-governance justification.

## 5. Why Model B is frozen rather than tuned for profit

Model B was conceived after the Notebook 7 holdout diagnosis. The 2025 to March 2026 period has therefore already influenced the hypothesis and cannot provide a new independent holdout.

Repeatedly testing entry and exit thresholds until a positive historical return appears would increase backtest-overfitting and selection risk. Bailey et al. show that testing more strategy configurations increases the probability that the selected backtest is overfit. The capstone therefore chooses one transparent specification and evaluates it prospectively.

Historical Model B replay is allowed only to:

- Verify the state machine
- Measure activity, turnover, costs, and drawdown
- Detect accounting or risk-control defects
- Provide clearly labelled post-holdout diagnostic results

Historical profitability is reported but is not a pass condition.

## 6. Expected improvements and falsification

### 6.1 Hypothesised improvements

Relative to Model A, Model B is expected to show:

- Zero short exposure
- Fewer new entries and lower turnover
- Lower transaction-cost burden
- Lower active exposure
- Potentially lower drawdown
- A different forward net-return path

### 6.2 What would weaken or reject the hypothesis

The hypothesis is not supported if prospective evidence shows one or more of the following without compensating operational benefit:

- Turnover is not reduced
- Cost burden is not reduced
- Drawdown is materially worse
- The strategy is effectively inactive
- Operational controls create repeated unresolved state mismatches
- Net performance is materially worse under equal exposure

A negative result must be reported. It must not trigger hidden threshold changes during the formal observation window.

## 7. Evaluation status and labels

| Evidence source | Correct label |
|---|---|
| 2022–2024 analysis after Model B design | Exploratory or diagnostic |
| 2025–March 2026 replay | Post-holdout retrospective diagnostic; not untouched |
| New MT5 shadow data before freeze of broker controls | Operational calibration; not formal trading evidence |
| New MT5 data after full configuration freeze | Prospective paper-trading evidence |

## 8. Broker-specific settings

The following are not part of the probability-overlay hypothesis and remain pending until Stage 2:

- Exact common lot, notional cap, and margin-use cap
- Maximum entry spread
- Slippage or deviation limit
- Volume source
- Filling mode
- Session-safety flattening
- Technical retry policy

They may be selected only from broker mechanics, safety, and fair-exposure requirements. They cannot be selected by observed profit. They will be stored in a separate common `config/mt5_runtime_frozen.yaml` file so the frozen strategy YAML files remain unchanged.

## 9. Viva-ready summary

Model B was not created as a newly optimised profitable model. It is a predefined post-holdout overlay hypothesis based on specific weaknesses diagnosed in Model A. Long-only positioning removes the historically harmful short side. The stricter 0.55 entry score reduces marginal entries. The 0.50 exit boundary creates hysteresis and permits exit once the model no longer favours an upward move. One entry per UTC day constrains turnover and cost accumulation. Removing the mandatory three-bar hold reduces the mismatch between the next-bar prediction target and the minimum trading commitment. The underlying CNN-LSTM and preprocessing remain unchanged, allowing the forward comparison to isolate overlay behaviour. The exact settings are frozen before prospective evaluation to avoid repeated historical optimisation.

## References

1. Notebook 7 executed notebook, evaluation artefact manifest, final holdout predictions, strategy bar log, and final trading metrics.
2. Bailey, D. H., Borwein, J. M., López de Prado, M., and Zhu, Q. J. *The Probability of Backtest Overfitting*. Journal of Computational Finance, 20(4), 39–69, 2017. DOI: 10.21314/JCF.2016.322.
3. Bailey, D. H., and López de Prado, M. *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management, 40(5), 94–107, 2014. DOI: 10.3905/jpm.2014.40.5.094.
4. Magill, M. J. P., and Constantinides, G. M. *Portfolio Selection with Transactions Costs*. Journal of Economic Theory, 13(2), 245–263, 1976. DOI: 10.1016/0022-0531(76)90018-1.
5. Guasoni, P., and Muhle-Karbe, J. *Portfolio Choice with Transaction Costs: A User's Guide*. Paris-Princeton Lectures on Mathematical Finance 2013, 169–201. DOI: 10.1007/978-3-319-00413-6_3.
