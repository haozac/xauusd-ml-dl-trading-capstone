"""Exact contiguous-sequence planning and scaling for Notebook 7 parity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from capstone_trading.data.canonical_bars import M15_DELTA
from capstone_trading.errors import SequenceParityError


@dataclass(frozen=True)
class SequencePlan:
    sequence_length: int
    row_count: int
    starts: np.ndarray
    ends: np.ndarray

    @property
    def sequence_count(self) -> int:
        return int(len(self.starts))

    def endpoint_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if len(index) != self.row_count:
            raise SequenceParityError(
                f"Index length {len(index)} differs from planned row count {self.row_count}"
            )
        return pd.DatetimeIndex(index[self.ends])


@dataclass(frozen=True)
class SequencePartitionReport:
    name: str
    row_count: int
    sequence_count: int
    first_row_utc: str
    last_row_utc: str
    first_sequence_end_utc: str
    last_sequence_end_utc: str


def valid_sequence_positions(
    index: pd.DatetimeIndex,
    sequence_length: int,
) -> SequencePlan:
    """Mirror Notebook 7 contiguous sequence selection exactly."""

    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        raise SequenceParityError("Sequence timestamps must be timezone-aware")
    index = index.tz_convert("UTC")
    if index.hasnans:
        raise SequenceParityError("Sequence timestamps contain NaT")
    if index.duplicated().any():
        raise SequenceParityError("Sequence timestamps contain duplicates")
    if not index.is_monotonic_increasing:
        raise SequenceParityError("Sequence timestamps must be chronological")

    if len(index) < sequence_length:
        empty = np.array([], dtype=np.int64)
        return SequencePlan(sequence_length, len(index), empty, empty.copy())

    gap_flags = np.zeros(len(index), dtype=np.int64)
    deltas_ns = index.asi8[1:] - index.asi8[:-1]
    gap_flags[1:] = (deltas_ns != M15_DELTA.value).astype(np.int64)
    gap_cumsum = np.cumsum(gap_flags)
    starts = np.arange(0, len(index) - sequence_length + 1, dtype=np.int64)
    ends = starts + sequence_length - 1
    gaps_inside = gap_cumsum[ends] - gap_cumsum[starts]
    valid = gaps_inside == 0
    return SequencePlan(sequence_length, len(index), starts[valid], ends[valid])


def validate_plan_contiguity(index: pd.DatetimeIndex, plan: SequencePlan) -> None:
    index = pd.DatetimeIndex(index).tz_convert("UTC")
    if len(index) != plan.row_count:
        raise SequenceParityError(
            "Sequence plan row count differs from the supplied index"
        )
    if plan.sequence_count == 0:
        return
    expected_span = M15_DELTA.value * (plan.sequence_length - 1)
    observed_span = index.asi8[plan.ends] - index.asi8[plan.starts]
    invalid = np.flatnonzero(observed_span != expected_span)
    if len(invalid):
        example = int(invalid[0])
        raise SequenceParityError(
            "Sequence plan contains a non-contiguous window: "
            f"start={index[plan.starts[example]]}, end={index[plan.ends[example]]}"
        )


def scale_feature_frame(
    scaler: Any,
    frame: pd.DataFrame,
    feature_order: Sequence[str],
) -> np.ndarray:
    order = tuple(str(name) for name in feature_order)
    missing = [name for name in order if name not in frame.columns]
    if missing:
        raise SequenceParityError(f"Feature frame is missing frozen columns: {missing}")
    selected = frame.loc[:, list(order)].astype("float32")
    try:
        transformed = scaler.transform(selected)
    except Exception as exc:
        raise SequenceParityError(f"Frozen scaler transform failed: {exc}") from exc
    values = np.asarray(transformed, dtype=np.float32)
    expected_shape = (len(frame), len(order))
    if values.shape != expected_shape:
        raise SequenceParityError(
            f"Scaled feature shape mismatch: expected {expected_shape}, found {values.shape}"
        )
    if not np.isfinite(values).all():
        raise SequenceParityError("Scaled features contain NaN or infinite values")
    return values


def iter_sequence_batches(
    scaled_features: np.ndarray,
    plan: SequencePlan,
    *,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield bounded-memory sequence batches and their endpoint positions."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = np.asarray(scaled_features, dtype=np.float32)
    if values.ndim != 2:
        raise SequenceParityError("scaled_features must be a two-dimensional array")
    if values.shape[0] != plan.row_count:
        raise SequenceParityError(
            f"Scaled row count {values.shape[0]} differs from plan {plan.row_count}"
        )
    for offset in range(0, plan.sequence_count, batch_size):
        starts = plan.starts[offset : offset + batch_size]
        endpoints = plan.ends[offset : offset + batch_size]
        batch = np.stack(
            [values[start : start + plan.sequence_length] for start in starts],
            axis=0,
        ).astype(np.float32, copy=False)
        expected_shape = (len(starts), plan.sequence_length, values.shape[1])
        if batch.shape != expected_shape:
            raise SequenceParityError(
                f"Sequence batch shape mismatch: expected {expected_shape}, found {batch.shape}"
            )
        yield batch, endpoints.copy()


def materialize_sequences(
    scaled_features: np.ndarray,
    plan: SequencePlan,
    positions: Sequence[int],
) -> np.ndarray:
    """Materialise selected planned sequences for tests and diagnostics only."""

    values = np.asarray(scaled_features, dtype=np.float32)
    selected: list[np.ndarray] = []
    for position in positions:
        if position < 0 or position >= plan.sequence_count:
            raise IndexError(f"Sequence plan position out of range: {position}")
        start = int(plan.starts[position])
        selected.append(values[start : start + plan.sequence_length])
    if not selected:
        return np.empty((0, plan.sequence_length, values.shape[1]), dtype=np.float32)
    return np.stack(selected, axis=0).astype(np.float32, copy=False)


def partition_report(
    name: str, frame: pd.DataFrame, plan: SequencePlan
) -> SequencePartitionReport:
    if frame.empty or plan.sequence_count == 0:
        raise SequenceParityError(f"Partition {name} contains no valid sequences")
    endpoints = plan.endpoint_index(frame.index)
    return SequencePartitionReport(
        name=name,
        row_count=int(len(frame)),
        sequence_count=plan.sequence_count,
        first_row_utc=frame.index.min().isoformat(),
        last_row_utc=frame.index.max().isoformat(),
        first_sequence_end_utc=endpoints.min().isoformat(),
        last_sequence_end_utc=endpoints.max().isoformat(),
    )


def report_to_dict(report: SequencePartitionReport) -> dict[str, Any]:
    return asdict(report)
