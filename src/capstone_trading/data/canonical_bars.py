"""Validation and canonicalisation of historical M15 OHLCV bars."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from capstone_trading.errors import HistoricalDataError

M15_MINUTES = 15
M15_DELTA = pd.Timedelta(minutes=M15_MINUTES)
BAR_COLUMNS = ("open", "high", "low", "close", "volume", "source_m1_bars")


@dataclass(frozen=True)
class BarValidationReport:
    row_count: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    duplicate_timestamps: int
    non_monotonic: bool
    non_contiguous_gaps: int
    maximum_gap_minutes: float
    invalid_ohlc_rows: int
    negative_volume_rows: int
    incomplete_source_rows: int


def _require_utc_datetime_index(frame: pd.DataFrame, *, label: str) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise HistoricalDataError(f"{label} must use a DatetimeIndex")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise HistoricalDataError(f"{label} timestamps must be timezone-aware UTC")
    try:
        index = index.tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise HistoricalDataError(
            f"Unable to convert {label} timestamps to UTC: {exc}"
        ) from exc
    if index.hasnans:
        raise HistoricalDataError(f"{label} timestamps contain NaT values")
    return index


def validate_m15_bars(frame: pd.DataFrame) -> tuple[pd.DataFrame, BarValidationReport]:
    """Validate complete, right-labelled historical M15 OHLCV bars.

    Gaps are permitted because weekends, holidays, and maintenance closures are
    part of the research data. A bar itself is valid only when it contains all
    15 source M1 observations.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise HistoricalDataError("M15 bars must be a non-empty DataFrame")
    missing = [column for column in BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise HistoricalDataError(f"M15 bars are missing required columns: {missing}")

    bars = frame.loc[:, list(BAR_COLUMNS)].copy()
    bars.index = _require_utc_datetime_index(bars, label="M15 bars")
    bars.index.name = "time"

    duplicate_timestamps = int(bars.index.duplicated().sum())
    if duplicate_timestamps:
        raise HistoricalDataError(
            f"M15 bars contain {duplicate_timestamps} duplicate timestamps"
        )
    non_monotonic = not bars.index.is_monotonic_increasing
    if non_monotonic:
        raise HistoricalDataError("M15 bars must be strictly chronological")

    numeric = bars.apply(pd.to_numeric, errors="coerce")
    missing_values = int(numeric.isna().sum().sum())
    if missing_values:
        raise HistoricalDataError(
            f"M15 bars contain {missing_values} missing or non-numeric values"
        )
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise HistoricalDataError("M15 bars contain infinite numeric values")

    invalid_ohlc = (
        (numeric["high"] < numeric["low"])
        | (numeric["high"] < numeric["open"])
        | (numeric["high"] < numeric["close"])
        | (numeric["low"] > numeric["open"])
        | (numeric["low"] > numeric["close"])
        | (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    invalid_ohlc_rows = int(invalid_ohlc.sum())
    if invalid_ohlc_rows:
        raise HistoricalDataError(
            f"M15 bars contain {invalid_ohlc_rows} invalid OHLC rows"
        )

    negative_volume_rows = int((numeric["volume"] < 0).sum())
    if negative_volume_rows:
        raise HistoricalDataError(
            f"M15 bars contain {negative_volume_rows} negative-volume rows"
        )

    source_counts = numeric["source_m1_bars"].to_numpy(dtype=np.float64)
    incomplete_source_rows = int(np.count_nonzero(source_counts != M15_MINUTES))
    if incomplete_source_rows:
        raise HistoricalDataError(
            f"M15 bars contain {incomplete_source_rows} rows without exactly "
            f"{M15_MINUTES} source M1 bars"
        )
    numeric["source_m1_bars"] = numeric["source_m1_bars"].astype("int16")

    deltas = numeric.index.to_series().diff().dropna()
    non_contiguous = deltas != M15_DELTA
    gap_count = int(non_contiguous.sum())
    max_gap = (
        float(deltas.max().total_seconds() / 60.0)
        if len(deltas)
        else float(M15_MINUTES)
    )

    report = BarValidationReport(
        row_count=int(len(numeric)),
        first_timestamp_utc=numeric.index.min().isoformat(),
        last_timestamp_utc=numeric.index.max().isoformat(),
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic=non_monotonic,
        non_contiguous_gaps=gap_count,
        maximum_gap_minutes=max_gap,
        invalid_ohlc_rows=invalid_ohlc_rows,
        negative_volume_rows=negative_volume_rows,
        incomplete_source_rows=incomplete_source_rows,
    )
    return numeric, report


def report_to_dict(report: BarValidationReport) -> dict[str, Any]:
    return asdict(report)
