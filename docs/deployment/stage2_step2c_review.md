# Stage 2 Step 2C Internal Review

## Scope reviewed

This patch adds a historical diagnostic comparison for Model B current versus a Model B minimum-hold candidate.

## Safety review

- No MT5 import is used by the Step 2C script.
- No order API is used.
- No model retraining is performed.
- No threshold search is performed.
- The minimum hold is fixed at 3 completed M15 bars.
- The final holdout remains diagnostic and is not presented as a new untouched holdout.

## Methodology review

The candidate changes only the normal probability-based exit rule. Gap exits, daily loss stops, and total drawdown stops still override minimum hold. This avoids hiding risk-control exits behind the minimum-hold rule.

## Code review notes

- `minimum_hold_bars = 3` is a constant in the script, not a search loop.
- The current Model B replay path is reused for the baseline.
- A separate `model_b_min_hold.py` module keeps the candidate overlay separate from the frozen Model B V2 replay code.
- The diagnostic invariants keep the strategy long-only and enforce the max successful entries per UTC day.
- Unit tests cover normal exit blocking, zero-min-hold equivalence, gap exit override, and daily entry cap preservation.

## Known limitations

- Step 2C is based on historical predictions, not live broker execution.
- Results must be interpreted as diagnostic comparison, not proof of profitability.
- A separate freeze decision is required before Stage 3.
