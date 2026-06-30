from pathlib import Path

import pytest

from capstone_trading.config import load_model_a_config, safe_repository_path
from capstone_trading.errors import ConfigurationError


def test_load_model_a_frozen_config() -> None:
    config = load_model_a_config(Path("config/model_a_frozen.yaml"))
    assert config.strategy_id == "MODEL_A"
    assert config.status == "FROZEN_STAGE_0"
    assert config.sequence_length == 48
    assert config.feature_count == 51
    assert config.parameter_count == 29825
    assert config.environment.python_supported == ("3.11", "3.12")
    assert {item.name for item in config.artefacts} == {
        "model",
        "scaler",
        "feature_order",
        "model_parameters",
        "selected_epoch",
        "selected_overlay",
    }


def test_safe_repository_path_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="repository-relative"):
        safe_repository_path(tmp_path, "../secret.txt", description="test")


def test_safe_repository_path_allows_future_report_path(tmp_path: Path) -> None:
    path = safe_repository_path(
        tmp_path,
        "runtime/reports/report.json",
        description="report",
        must_exist=False,
    )
    assert path == tmp_path / "runtime" / "reports" / "report.json"
