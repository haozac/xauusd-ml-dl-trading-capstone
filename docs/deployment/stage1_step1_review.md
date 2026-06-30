# Stage 1 Step 1 Internal Review

## Scope reviewed

This review covers only environment and frozen-artefact verification. It does not implement feature engineering, strategy replay, MT5 connectivity, shadow trading, or order execution.

## Implemented controls

- Repository-relative path resolution blocks absolute paths, parent traversal, and symlink escape.
- Stage 0 files are verified against the frozen SHA-256 manifest.
- Model B shared artefact paths and hashes must equal Model A.
- All six Notebook 7 contract artefacts are verified before any pickle or Keras load.
- The scaler pickle is loaded only after its frozen hash passes.
- Scaler class, feature count, fitted-row count, finite statistics, positive scales, and feature order are checked.
- Model input/output shapes, parameter count, layer topology, activations, dropout, and L2 values are checked.
- Python and package versions are checked against the frozen contract.
- Deterministic CPU controls are enabled before formal TensorFlow model loading.
- A real 48-by-51 holdout fixture verifies scaler-plus-model inference against a saved Notebook 7 probability.
- The smoke test requires both numerical closeness and zero decision-threshold flips.
- Reports are written atomically through a temporary file and contain typed failure information.
- Metadata-only diagnostics cannot be mistaken for a formal pass.

## Failure policy

Hash, feature-order, model-shape, scaler, environment, or reference-inference failures block Step 2. The verifier never retrains, resaves, repairs, or silently replaces a frozen artefact.

## Local review evidence

- Python compilation: passed.
- Unit and non-TensorFlow integration tests: 13 passed.
- Frozen Stage 0 hashes: passed.
- Frozen Notebook 7 artefact hashes: passed.
- Scaler structural contract: passed against the uploaded scaler.
- Keras cross-backend structural smoke check: passed.
- Reference probability absolute difference in the supplemental Keras/PyTorch check: approximately 2.98e-08.
- Decision-threshold flips in the supplemental check: zero.

The formal gate still requires the user to run the verifier under Python 3.11 or 3.12 with TensorFlow 2.20.0, Keras 3.13.2, and scikit-learn 1.6.1. The review environment did not contain that exact TensorFlow environment, so no claim of a formal TensorFlow pass is made here.

## Council verdict

- Quantitative and methodology review: approved. The fixture is used only for compatibility, not for selecting or re-evaluating strategy performance.
- ML deployment review: approved subject to the pinned local TensorFlow run.
- Security and MLOps review: approved. Hash-before-pickle and immutable configuration checks are mandatory.
- Reliability review: approved. Expected failures are typed, reported, and fail closed.
- MT5 and execution review: not applicable at this step; no MT5 code is included.

**Decision:** Stage 1 Step 1 patch is approved for local integration and execution of the formal pinned-environment gate.
