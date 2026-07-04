# Stage 1 Step 3 Internal Review — Full CNN-LSTM Inference Parity

## Scope reviewed

The patch adds a full inference parity gate that independently rebuilds the verified historical features, creates 48-bar contiguous windows, runs the frozen CNN-LSTM, and compares generated probabilities against Notebook 7 saved prediction CSVs.

## Main controls

1. Stage 0 freeze manifest is verified before inference.
2. Notebook 7 artefact hashes are verified before scaler/model loading.
3. Historical M15 and model-ready parquet hashes are verified.
4. M15 features are rebuilt and compared to the official model-ready dataset before inference.
5. The frozen scaler is loaded only after its hash passes.
6. The frozen Keras model is loaded with `compile=False`.
7. The formal run uses CPU deterministic environment variables.
8. Every generated probability is checked for finite values and [0, 1] bounds.
9. Every saved `p_up` value is compared against a freshly generated probability.
10. Threshold flips are checked separately from numerical tolerance.

## Council observations

### Quantitative trading review

Approved. The check compares probabilities before any trading overlay is applied, which isolates model-inference reproducibility from strategy design.

### Time-series review

Approved. The script validates prediction endpoint alignment again, so probability parity cannot pass against shifted windows.

### ML deployment review

Approved. The script uses batch inference with bounded memory and checks shape, dtype, probability bounds, and batch write counts.

### Risk governance review

Approved. The script does not make profitability claims and does not alter Model A or Model B. It only verifies reproducibility of the frozen prediction source used by both overlays.

### Software reliability review

Approved with one intentional design choice: the formal Step 3 script repeats critical Step 2 preconditions instead of blindly trusting the previous report. This creates extra runtime but improves auditability.

## Known limitations

1. The probability tolerance is numerical, not economic. A pass means deployment inference matches Notebook 7; it does not mean the strategy is profitable.
2. The formal reference is CPU parity, not live MT5 latency parity.
3. The trust root still relies on committed manifests and Git review. Cryptographic signing is unnecessary for this capstone but repository review discipline remains required.

## Report value

The Step 3 JSON and summary CSV can support the report statement that the offline deployment inference pathway reproduced all frozen Notebook 7 2024 and holdout probabilities within tolerance and with zero decision-threshold flips.

## Verdict

Approved for local execution. Step 3 remains open until the user-run formal report returns `PASS` with `formal_gate=true`.
