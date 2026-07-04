"""Full frozen CNN-LSTM inference parity checks for Stage 1 Step 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from capstone_trading.data.sequences import SequencePlan, iter_sequence_batches
from capstone_trading.errors import InferenceParityError, SequenceParityError


DEFAULT_THRESHOLDS: tuple[float, ...] = (0.47, 0.50, 0.53, 0.55)


@dataclass(frozen=True)
class ThresholdParityResult:
    threshold: float
    expected_true_count: int
    actual_true_count: int
    flip_count: int
    flip_rate: float


@dataclass(frozen=True)
class ProbabilityParityReport:
    name: str
    row_count: int
    tolerance: float
    passed: bool
    maximum_absolute_difference: float
    mean_absolute_difference: float
    median_absolute_difference: float
    p95_absolute_difference: float
    p99_absolute_difference: float
    p999_absolute_difference: float
    mismatches_above_tolerance: int
    expected_probability_min: float
    expected_probability_mean: float
    expected_probability_max: float
    actual_probability_min: float
    actual_probability_mean: float
    actual_probability_max: float
    total_threshold_flips: int
    threshold_results: tuple[ThresholdParityResult, ...]


@dataclass(frozen=True)
class PredictionAlignmentResult:
    name: str
    expected_rows: int
    actual_rows: int
    timestamp_match: bool
    target_direction_match: bool
    maximum_forward_return_difference: float
    passed: bool


def _as_probability_vector(values: Any, *, expected_rows: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim != 1:
        raise InferenceParityError(
            f"{label} probabilities must be one-dimensional or Nx1, found shape {array.shape}"
        )
    if array.shape != (expected_rows,):
        raise InferenceParityError(
            f"{label} probability count mismatch: expected {expected_rows}, found {array.shape[0]}"
        )
    if not np.isfinite(array).all():
        raise InferenceParityError(f"{label} probabilities contain NaN or infinite values")
    invalid = (array < 0.0) | (array > 1.0)
    if bool(np.any(invalid)):
        first = int(np.flatnonzero(invalid)[0])
        raise InferenceParityError(
            f"{label} probability outside [0, 1] at position {first}: {array[first]}"
        )
    return array


def _tensor_to_numpy(output: Any) -> np.ndarray:
    """Convert TensorFlow, Keras, PyTorch or NumPy model outputs safely."""

    value = output
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def predict_probabilities_in_batches(
    model: Any,
    scaled_features: np.ndarray,
    plan: SequencePlan,
    *,
    batch_size: int,
) -> np.ndarray:
    """Run the frozen model over all planned sequences with bounded memory."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    probabilities = np.empty(plan.sequence_count, dtype=np.float64)
    write_position = 0
    for batch, _endpoints in iter_sequence_batches(
        scaled_features,
        plan,
        batch_size=batch_size,
    ):
        if batch.dtype != np.float32:
            raise InferenceParityError(
                f"Model batch dtype must be float32, found {batch.dtype}"
            )
        try:
            output = model(batch, training=False)
            batch_probabilities = _tensor_to_numpy(output)
        except Exception as exc:
            raise InferenceParityError(f"Frozen model batch inference failed: {exc}") from exc

        vector = _as_probability_vector(
            batch_probabilities,
            expected_rows=len(batch),
            label="actual model output",
        )
        probabilities[write_position : write_position + len(vector)] = vector
        write_position += len(vector)

    if write_position != plan.sequence_count:
        raise InferenceParityError(
            f"Inference wrote {write_position} rows but plan requires {plan.sequence_count}"
        )
    return probabilities


def validate_prediction_alignment(
    *,
    name: str,
    partition: pd.DataFrame,
    plan: SequencePlan,
    predictions: pd.DataFrame,
    return_tolerance: float = 1e-15,
) -> PredictionAlignmentResult:
    """Ensure generated sequence endpoints refer to the same Notebook 7 rows."""

    if return_tolerance < 0:
        raise ValueError("return_tolerance must be non-negative")
    endpoints = plan.endpoint_index(partition.index)
    expected_rows = int(len(predictions))
    actual_rows = plan.sequence_count
    if actual_rows != expected_rows:
        raise SequenceParityError(
            f"{name} sequence count mismatch: expected {expected_rows}, found {actual_rows}"
        )
    prediction_times = pd.DatetimeIndex(pd.to_datetime(predictions["time"], utc=True))
    timestamp_match = np.array_equal(endpoints.asi8, prediction_times.asi8)
    aligned = partition.iloc[plan.ends]
    target_match = np.array_equal(
        aligned["target_dir"].to_numpy(dtype=np.int8),
        predictions["target_dir"].to_numpy(dtype=np.int8),
    )
    return_difference = np.abs(
        aligned["target_ret_fwd"].to_numpy(dtype=np.float64)
        - predictions["target_ret_fwd"].to_numpy(dtype=np.float64)
    )
    max_return_difference = float(return_difference.max(initial=0.0))
    passed = timestamp_match and target_match and max_return_difference <= return_tolerance
    if not passed:
        raise SequenceParityError(
            f"{name} prediction alignment failed: timestamps={timestamp_match}, "
            f"targets={target_match}, max_forward_return_difference={max_return_difference:.3e}"
        )
    return PredictionAlignmentResult(
        name=name,
        expected_rows=expected_rows,
        actual_rows=actual_rows,
        timestamp_match=timestamp_match,
        target_direction_match=target_match,
        maximum_forward_return_difference=max_return_difference,
        passed=passed,
    )


def compare_probability_vectors(
    *,
    name: str,
    expected_probabilities: Sequence[float] | np.ndarray,
    actual_probabilities: Sequence[float] | np.ndarray,
    tolerance: float,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> ProbabilityParityReport:
    """Compare generated model probabilities with Notebook 7 saved probabilities."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    expected = _as_probability_vector(
        expected_probabilities,
        expected_rows=len(expected_probabilities),
        label="expected Notebook 7",
    )
    actual = _as_probability_vector(
        actual_probabilities,
        expected_rows=len(expected),
        label="actual model output",
    )
    difference = np.abs(actual - expected)
    threshold_results: list[ThresholdParityResult] = []
    total_flips = 0
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be inside [0, 1], found {threshold}")
        expected_decision = expected >= threshold
        actual_decision = actual >= threshold
        flips = int(np.count_nonzero(expected_decision != actual_decision))
        total_flips += flips
        threshold_results.append(
            ThresholdParityResult(
                threshold=threshold,
                expected_true_count=int(np.count_nonzero(expected_decision)),
                actual_true_count=int(np.count_nonzero(actual_decision)),
                flip_count=flips,
                flip_rate=float(flips / len(expected)) if len(expected) else 0.0,
            )
        )

    mismatches = int(np.count_nonzero(difference > tolerance))
    passed = mismatches == 0 and total_flips == 0
    report = ProbabilityParityReport(
        name=name,
        row_count=int(len(expected)),
        tolerance=float(tolerance),
        passed=passed,
        maximum_absolute_difference=float(difference.max(initial=0.0)),
        mean_absolute_difference=float(difference.mean()) if len(difference) else 0.0,
        median_absolute_difference=float(np.median(difference)) if len(difference) else 0.0,
        p95_absolute_difference=float(np.quantile(difference, 0.95)) if len(difference) else 0.0,
        p99_absolute_difference=float(np.quantile(difference, 0.99)) if len(difference) else 0.0,
        p999_absolute_difference=float(np.quantile(difference, 0.999)) if len(difference) else 0.0,
        mismatches_above_tolerance=mismatches,
        expected_probability_min=float(expected.min(initial=1.0)),
        expected_probability_mean=float(expected.mean()) if len(expected) else 0.0,
        expected_probability_max=float(expected.max(initial=0.0)),
        actual_probability_min=float(actual.min(initial=1.0)),
        actual_probability_mean=float(actual.mean()) if len(actual) else 0.0,
        actual_probability_max=float(actual.max(initial=0.0)),
        total_threshold_flips=total_flips,
        threshold_results=tuple(threshold_results),
    )
    if not passed:
        raise InferenceParityError(
            f"{name} probability parity failed: max_diff={report.maximum_absolute_difference:.3e}, "
            f"mismatches_above_tolerance={mismatches}, threshold_flips={total_flips}"
        )
    return report


def build_diagnostic_rows(
    *,
    partition_name: str,
    predictions: pd.DataFrame,
    expected_probabilities: np.ndarray,
    actual_probabilities: np.ndarray,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Build compact report evidence for largest differences and decision-boundary rows."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    expected = _as_probability_vector(
        expected_probabilities,
        expected_rows=len(expected_probabilities),
        label="expected Notebook 7",
    )
    actual = _as_probability_vector(
        actual_probabilities,
        expected_rows=len(expected),
        label="actual model output",
    )
    difference = np.abs(actual - expected)
    times = pd.to_datetime(predictions["time"], utc=True)
    targets = predictions["target_dir"].to_numpy(dtype=np.int8)

    rows: list[dict[str, Any]] = []
    added: set[tuple[str, int]] = set()

    def add_row(category: str, position: int, threshold: float | None = None) -> None:
        key = (category, int(position))
        if key in added:
            return
        added.add(key)
        rows.append(
            {
                "partition": partition_name,
                "category": category,
                "row_position": int(position),
                "time_utc": times.iloc[int(position)].isoformat(),
                "target_dir": int(targets[int(position)]),
                "threshold": "" if threshold is None else float(threshold),
                "expected_p_up": float(expected[int(position)]),
                "actual_p_up": float(actual[int(position)]),
                "absolute_difference": float(difference[int(position)]),
                "decision_flip": ""
                if threshold is None
                else bool((expected[int(position)] >= threshold) != (actual[int(position)] >= threshold)),
            }
        )

    largest = np.argsort(-difference, kind="mergesort")[: min(top_n, len(difference))]
    for position in largest:
        add_row("largest_absolute_difference", int(position))

    for threshold in thresholds:
        threshold = float(threshold)
        distance = np.minimum(np.abs(expected - threshold), np.abs(actual - threshold))
        nearest = np.argsort(distance, kind="mergesort")[: min(top_n, len(distance))]
        for position in nearest:
            add_row(f"nearest_threshold_{threshold:.2f}", int(position), threshold)

    return rows


def probability_report_to_dict(report: ProbabilityParityReport) -> dict[str, Any]:
    output = asdict(report)
    output["threshold_results"] = [asdict(item) for item in report.threshold_results]
    return output


def report_to_dict(report: Any) -> Mapping[str, Any]:
    return asdict(report)
