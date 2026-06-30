from pathlib import Path

import pytest

from capstone_trading.artifacts import verify_notebook7_artifact_bundle
from capstone_trading.config import load_model_a_config
from capstone_trading.model_loader import load_and_validate_scaler, load_reference_fixture


@pytest.mark.integration
def test_real_notebook7_bundle_hashes_and_scaler_contract() -> None:
    config = load_model_a_config("config/model_a_frozen.yaml")
    model_path = Path(config.artefact("model").relative_path)
    if not model_path.exists():
        pytest.skip("Local gitignored Notebook 7 model artefact is not available")

    bundle = verify_notebook7_artifact_bundle(Path.cwd(), config)
    assert len(bundle.integrity_results) == 6
    assert all(item.passed for item in bundle.integrity_results)
    assert len(bundle.feature_order) == 51

    scaler, report = load_and_validate_scaler(
        bundle.scaler_path,
        config,
        bundle.feature_order,
    )
    assert scaler is not None
    assert report.feature_count == 51
    assert report.fitted_rows == 184584

    raw, metadata = load_reference_fixture(
        Path("tests/fixtures/notebook7_reference_sequence.csv"),
        Path("tests/fixtures/notebook7_reference_sequence.json"),
        bundle.feature_order,
    )
    assert raw.shape == (48, 51)
    assert metadata["expected_probability"] == pytest.approx(0.3610754013061523)
