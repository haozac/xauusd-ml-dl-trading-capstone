from pathlib import Path

import pytest

from capstone_trading.artifacts import verify_notebook7_artifact_bundle
from capstone_trading.config import load_model_a_config
from capstone_trading.model_loader import (
    check_runtime_environment,
    load_and_validate_model,
    load_and_validate_scaler,
    load_reference_fixture,
    run_reference_inference,
)


@pytest.mark.integration
@pytest.mark.tensorflow
def test_pinned_tensorflow_reference_inference() -> None:
    config = load_model_a_config("config/model_a_frozen.yaml")
    if not Path(config.artefact("model").relative_path).exists():
        pytest.skip("Local gitignored Notebook 7 model artefact is not available")

    check_runtime_environment(config, strict=True)
    bundle = verify_notebook7_artifact_bundle(Path.cwd(), config)
    scaler, _ = load_and_validate_scaler(bundle.scaler_path, config, bundle.feature_order)
    model, _ = load_and_validate_model(bundle.model_path, config)
    raw, metadata = load_reference_fixture(
        Path("tests/fixtures/notebook7_reference_sequence.csv"),
        Path("tests/fixtures/notebook7_reference_sequence.json"),
        bundle.feature_order,
    )
    report = run_reference_inference(
        model, scaler, raw, metadata, bundle.feature_order, tolerance=1e-5
    )
    assert report.passed
    assert report.threshold_flip_count == 0
