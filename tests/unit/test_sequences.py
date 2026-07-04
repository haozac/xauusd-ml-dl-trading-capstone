from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from capstone_trading.data.sequences import (
    iter_sequence_batches,
    scale_feature_frame,
    valid_sequence_positions,
    validate_plan_contiguity,
)
from capstone_trading.errors import SequenceParityError


def test_valid_sequence_positions_excludes_windows_crossing_gap() -> None:
    index = pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC")
    index = index.delete(5)
    plan = valid_sequence_positions(index, 4)
    validate_plan_contiguity(index, plan)
    expected_starts = np.array([0, 1, 5], dtype=np.int64)
    assert np.array_equal(plan.starts, expected_starts)
    assert np.array_equal(plan.ends, expected_starts + 3)


def test_scale_feature_frame_preserves_float32_contract() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 5.0, 7.0]})
    scaler = StandardScaler().fit(frame.astype("float32"))
    values = scale_feature_frame(scaler, frame, ("a", "b"))
    assert values.dtype == np.float32
    assert values.shape == (3, 2)
    assert np.isfinite(values).all()


def test_iter_sequence_batches_rejects_wrong_row_count() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    plan = valid_sequence_positions(index, 4)
    with pytest.raises(SequenceParityError, match="row count"):
        list(
            iter_sequence_batches(
                np.zeros((7, 2), dtype=np.float32), plan, batch_size=2
            )
        )
