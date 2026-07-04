from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone_trading.data.sequences import valid_sequence_positions
from capstone_trading.errors import InferenceParityError, SequenceParityError
from capstone_trading.evaluation.inference_parity import (
    compare_probability_vectors,
    predict_probabilities_in_batches,
    validate_prediction_alignment,
)


class MeanProbabilityModel:
    def __call__(self, batch, training=False):
        assert training is False
        values = batch.mean(axis=(1, 2), keepdims=False)
        return np.clip(values, 0.0, 1.0).reshape(-1, 1).astype(np.float32)


def test_compare_probability_vectors_accepts_tiny_difference_without_flips() -> None:
    report = compare_probability_vectors(
        name="demo",
        expected_probabilities=np.array([0.20, 0.60, 0.80]),
        actual_probabilities=np.array([0.20000001, 0.59999999, 0.80000001]),
        tolerance=1e-5,
        thresholds=(0.50, 0.55),
    )
    assert report.passed
    assert report.maximum_absolute_difference < 1e-5
    assert report.total_threshold_flips == 0


def test_compare_probability_vectors_rejects_threshold_flip_inside_tolerance() -> None:
    with pytest.raises(InferenceParityError, match="threshold_flips=1"):
        compare_probability_vectors(
            name="demo",
            expected_probabilities=np.array([0.5299]),
            actual_probabilities=np.array([0.5301]),
            tolerance=1e-3,
            thresholds=(0.53,),
        )


def test_compare_probability_vectors_rejects_probability_mismatch() -> None:
    with pytest.raises(InferenceParityError, match="mismatches_above_tolerance=1"):
        compare_probability_vectors(
            name="demo",
            expected_probabilities=np.array([0.40]),
            actual_probabilities=np.array([0.41]),
            tolerance=1e-5,
            thresholds=(0.50,),
        )


def test_predict_probabilities_in_batches_preserves_sequence_count() -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
    plan = valid_sequence_positions(index, sequence_length=3)
    scaled = np.ones((6, 2), dtype=np.float32) * 0.6
    probabilities = predict_probabilities_in_batches(
        MeanProbabilityModel(),
        scaled,
        plan,
        batch_size=2,
    )
    assert probabilities.shape == (4,)
    assert np.allclose(probabilities, 0.6)


def test_validate_prediction_alignment_rejects_wrong_timestamp() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    partition = pd.DataFrame(
        {
            "target_dir": [0, 1, 0, 1],
            "target_ret_fwd": [0.1, 0.2, 0.3, 0.4],
        },
        index=index,
    )
    plan = valid_sequence_positions(index, sequence_length=2)
    predictions = pd.DataFrame(
        {
            "time": index[1:] + pd.Timedelta(minutes=15),
            "target_dir": [1, 0, 1],
            "target_ret_fwd": [0.2, 0.3, 0.4],
            "p_up": [0.5, 0.5, 0.5],
        }
    )
    with pytest.raises(SequenceParityError, match="prediction alignment failed"):
        validate_prediction_alignment(
            name="demo",
            partition=partition,
            plan=plan,
            predictions=predictions,
        )
