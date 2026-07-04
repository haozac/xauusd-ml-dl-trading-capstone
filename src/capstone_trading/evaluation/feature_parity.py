"""Stage 1 Step 2 feature, scaling, and sequence parity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from capstone_trading.data.sequences import SequencePlan
from capstone_trading.errors import FeatureParityError, SequenceParityError


@dataclass(frozen=True)
class ColumnParityResult:
    column: str
    dtype_reference: str
    dtype_rebuilt: str
    maximum_absolute_difference: float
    mismatch_count: int
    passed: bool


@dataclass(frozen=True)
class DatasetParityReport:
    row_count: int
    column_count: int
    index_match: bool
    column_order_match: bool
    maximum_absolute_difference: float
    total_mismatch_count: int
    tolerance: float
    column_results: tuple[ColumnParityResult, ...]


@dataclass(frozen=True)
class PredictionAlignmentReport:
    name: str
    expected_rows: int
    actual_rows: int
    timestamp_match: bool
    target_direction_match: bool
    maximum_forward_return_difference: float
    passed: bool


def compare_model_ready_datasets(
    rebuilt: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    tolerance: float,
) -> DatasetParityReport:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    index_match = rebuilt.index.equals(reference.index)
    column_match = tuple(rebuilt.columns) == tuple(reference.columns)
    if not index_match:
        raise FeatureParityError(
            "Rebuilt dataset timestamp index differs from the reference"
        )
    if not column_match:
        raise FeatureParityError(
            "Rebuilt dataset column order differs from the reference"
        )
    if rebuilt.shape != reference.shape:
        raise FeatureParityError(
            f"Rebuilt dataset shape {rebuilt.shape} differs from reference {reference.shape}"
        )

    results: list[ColumnParityResult] = []
    overall_max = 0.0
    total_mismatches = 0
    exact_columns = {"is_after_gap", "target_dir", "target_class_3"}
    for column in reference.columns:
        ref = reference[column]
        got = rebuilt[column]
        ref_values = ref.to_numpy()
        got_values = got.to_numpy()
        if column in exact_columns:
            mismatch = ref_values != got_values
            max_diff = float(
                np.max(np.abs(ref_values.astype(float) - got_values.astype(float)))
            )
        else:
            ref_float = ref_values.astype(np.float64)
            got_float = got_values.astype(np.float64)
            difference = np.abs(ref_float - got_float)
            mismatch = difference > tolerance
            max_diff = float(difference.max(initial=0.0))
        mismatch_count = int(np.count_nonzero(mismatch))
        passed = mismatch_count == 0
        results.append(
            ColumnParityResult(
                column=column,
                dtype_reference=str(ref.dtype),
                dtype_rebuilt=str(got.dtype),
                maximum_absolute_difference=max_diff,
                mismatch_count=mismatch_count,
                passed=passed,
            )
        )
        overall_max = max(overall_max, max_diff)
        total_mismatches += mismatch_count

    report = DatasetParityReport(
        row_count=int(len(reference)),
        column_count=int(reference.shape[1]),
        index_match=index_match,
        column_order_match=column_match,
        maximum_absolute_difference=overall_max,
        total_mismatch_count=total_mismatches,
        tolerance=tolerance,
        column_results=tuple(results),
    )
    if total_mismatches:
        failed = [item.column for item in results if not item.passed]
        raise FeatureParityError(
            f"Feature parity failed for {len(failed)} columns: {failed[:10]}; "
            f"total mismatches={total_mismatches}, max difference={overall_max:.3e}"
        )
    return report


def compare_sequence_endpoints_to_predictions(
    *,
    name: str,
    partition: pd.DataFrame,
    plan: SequencePlan,
    predictions: pd.DataFrame,
) -> PredictionAlignmentReport:
    endpoints = plan.endpoint_index(partition.index)
    expected_rows = int(len(predictions))
    actual_rows = plan.sequence_count
    timestamp_match = actual_rows == expected_rows and np.array_equal(
        endpoints.asi8,
        pd.DatetimeIndex(predictions["time"]).asi8,
    )
    if actual_rows != expected_rows:
        raise SequenceParityError(
            f"{name} sequence count mismatch: expected {expected_rows}, found {actual_rows}"
        )
    aligned = partition.iloc[plan.ends]
    prediction_targets = predictions["target_dir"].to_numpy(dtype=np.int8)
    target_match = np.array_equal(
        aligned["target_dir"].to_numpy(dtype=np.int8), prediction_targets
    )
    return_difference = np.abs(
        aligned["target_ret_fwd"].to_numpy(dtype=np.float64)
        - predictions["target_ret_fwd"].to_numpy(dtype=np.float64)
    )
    max_return_difference = float(return_difference.max(initial=0.0))
    passed = timestamp_match and target_match and max_return_difference <= 1e-15
    if not passed:
        raise SequenceParityError(
            f"{name} prediction alignment failed: timestamps={timestamp_match}, "
            f"targets={target_match}, max return difference={max_return_difference:.3e}"
        )
    return PredictionAlignmentReport(
        name=name,
        expected_rows=expected_rows,
        actual_rows=actual_rows,
        timestamp_match=timestamp_match,
        target_direction_match=target_match,
        maximum_forward_return_difference=max_return_difference,
        passed=passed,
    )


def dataset_report_to_dict(report: DatasetParityReport) -> dict[str, Any]:
    output = asdict(report)
    output["column_results"] = [asdict(item) for item in report.column_results]
    return output


def report_to_dict(report: Any) -> Mapping[str, Any]:
    return asdict(report)
