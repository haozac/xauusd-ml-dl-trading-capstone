from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone_trading.data.canonical_bars import validate_m15_bars
from capstone_trading.errors import HistoricalDataError


def make_bars(rows: int = 4) -> pd.DataFrame:
    index = pd.date_range(
        "2025-01-01", periods=rows, freq="15min", tz="UTC", name="time"
    )
    close = np.arange(rows, dtype=float) + 2000.0
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
            "volume": np.arange(rows, dtype=float) + 100.0,
            "source_m1_bars": np.full(rows, 15, dtype=np.int16),
        },
        index=index,
    )


def test_validate_m15_bars_accepts_gaps_but_counts_them() -> None:
    bars = make_bars(5).drop(make_bars(5).index[2])
    validated, report = validate_m15_bars(bars)
    assert len(validated) == 4
    assert report.non_contiguous_gaps == 1
    assert report.maximum_gap_minutes == 30.0


def test_validate_m15_bars_rejects_duplicate_timestamp() -> None:
    bars = make_bars(4)
    duplicate = pd.concat([bars, bars.iloc[[0]]]).sort_index()
    with pytest.raises(HistoricalDataError, match="duplicate"):
        validate_m15_bars(duplicate)


def test_validate_m15_bars_rejects_incomplete_source_bar() -> None:
    bars = make_bars(4)
    bars.iloc[2, bars.columns.get_loc("source_m1_bars")] = 14
    with pytest.raises(HistoricalDataError, match="exactly 15"):
        validate_m15_bars(bars)


def test_validate_m15_bars_rejects_invalid_ohlc() -> None:
    bars = make_bars(4)
    bars.iloc[1, bars.columns.get_loc("high")] = bars.iloc[1]["low"] - 1
    with pytest.raises(HistoricalDataError, match="invalid OHLC"):
        validate_m15_bars(bars)
