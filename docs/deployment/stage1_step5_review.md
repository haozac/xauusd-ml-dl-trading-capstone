# Stage 1 Step 5 Internal Review — Model B Diagnostic Replay

## Scope control

This patch adds Model B diagnostic historical replay only. It does not add MT5, order placement, live shadow mode, or any threshold optimisation.

## Council review

### Quantitative trading review

Approved. The Model B replay uses the frozen long-only rules from `config/model_b_v2_frozen.yaml`: entry at `p_up >= 0.55`, hold while `p_up >= 0.50`, exit when `p_up < 0.50`, no shorts, no mandatory minimum hold and at most one successful new entry per UTC day.

### Methodology review

Approved. Step 5 is explicitly labelled `diagnostic_only`. It requires the Step 4 Model A PASS report as a baseline and does not attempt to replace Notebook 7 or create a new official holdout.

### Risk review

Approved. The replay preserves the same 2 percent daily loss stop and 15 percent total drawdown stop. Risk exits override the overlay and entry cap.

### Software reliability review

Approved. The script writes JSON and CSV outputs atomically, verifies Stage 0 and prior-step preconditions, validates prediction hashes, and fails with exit code 2 for expected validation failures.

### Audit review

Approved. The patch also cleans up the Step 4 selected-overlay diagnostic comparison so the selected-overlay JSON fields are compared explicitly instead of returning zero comparisons.

## Expected gate

Stage 1 Step 5 closes only when:

```text
status == PASS
formal_gate == true
diagnostic_only == true
checks.model_b_invariants.passed == true
```
