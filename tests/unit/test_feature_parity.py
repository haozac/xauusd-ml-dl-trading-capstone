from __future__ import annotations

import pandas as pd
import pytest

from capstone_trading.errors import FeatureParityError
from capstone_trading.evaluation.feature_parity import compare_model_ready_datasets


def test_compare_model_ready_datasets_accepts_machine_epsilon_difference() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")
    reference = pd.DataFrame(
        {
            "x": [0.1, 0.2],
            "is_after_gap": pd.Series([0, 1], dtype="int8", index=index),
            "target_dir": pd.Series([0, 1], dtype="int8", index=index),
            "target_class_3": pd.Series([-1, 1], dtype="int8", index=index),
        },
        index=index,
    )
    rebuilt = reference.copy()
    rebuilt["x"] += 1e-14
    report = compare_model_ready_datasets(rebuilt, reference, tolerance=1e-12)
    assert report.total_mismatch_count == 0


def test_compare_model_ready_datasets_rejects_material_difference() -> None:
    index = pd.date_range("2025-01-01", periods=1, freq="15min", tz="UTC")
    reference = pd.DataFrame({"x": [0.1]}, index=index)
    rebuilt = pd.DataFrame({"x": [0.2]}, index=index)
    with pytest.raises(FeatureParityError, match="Feature parity failed"):
        compare_model_ready_datasets(rebuilt, reference, tolerance=1e-12)
