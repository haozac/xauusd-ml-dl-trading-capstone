from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class CleaningReport:
    raw_rows: int
    duplicate_rows_removed: int
    invalid_price_rows_removed: int
    invalid_ohlc_rows_removed: int
    invalid_volume_rows_removed: int
    final_rows: int
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None


@dataclass
class ResampleReport:
    source_m1_rows: int
    non_empty_m5_bars_before_filter: int
    incomplete_m5_bars_removed: int
    final_m5_bars: int


def _require_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe must use a DatetimeIndex named 'time'.")
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "time"
    return out


def clean_m1_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and clean Dukascopy M1 OHLCV bars without inventing missing market bars."""
    out = _require_datetime_index(df)
    raw_rows = len(out)

    missing_columns = [column for column in OHLCV_COLUMNS if column not in out.columns]
    if missing_columns:
        raise ValueError(f"Missing required M1 columns: {missing_columns}")

    out = out[OHLCV_COLUMNS].copy()
    for column in OHLCV_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    duplicate_mask = out.index.duplicated(keep="first")
    duplicate_rows_removed = int(duplicate_mask.sum())
    out = out.loc[~duplicate_mask].sort_index()

    invalid_price_mask = (
        out[["open", "high", "low", "close"]].isna().any(axis=1)
        | (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    invalid_price_rows_removed = int(invalid_price_mask.sum())
    out = out.loc[~invalid_price_mask]

    invalid_ohlc_mask = (
        (out["high"] < out["low"])
        | (out["high"] < out["open"])
        | (out["high"] < out["close"])
        | (out["low"] > out["open"])
        | (out["low"] > out["close"])
    )
    invalid_ohlc_rows_removed = int(invalid_ohlc_mask.sum())
    out = out.loc[~invalid_ohlc_mask]

    invalid_volume_mask = out["volume"].isna() | (out["volume"] < 0)
    invalid_volume_rows_removed = int(invalid_volume_mask.sum())
    out = out.loc[~invalid_volume_mask]

    report = CleaningReport(
        raw_rows=raw_rows,
        duplicate_rows_removed=duplicate_rows_removed,
        invalid_price_rows_removed=invalid_price_rows_removed,
        invalid_ohlc_rows_removed=invalid_ohlc_rows_removed,
        invalid_volume_rows_removed=invalid_volume_rows_removed,
        final_rows=len(out),
        first_timestamp_utc=str(out.index.min()) if not out.empty else None,
        last_timestamp_utc=str(out.index.max()) if not out.empty else None,
    )
    return out, report


def resample_to_m5(df_m1: pd.DataFrame) -> tuple[pd.DataFrame, ResampleReport]:
    """
    Resample M1 bars to M5 bars.

    Each M5 timestamp is the right-edge availability time: a bar labelled 00:05
    contains source observations from [00:00, 00:05). Only complete five-source-
    observation bars are retained so gaps are not silently turned into partial M5 bars.
    """
    out = _require_datetime_index(df_m1)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    bars = out.resample("5min", label="right", closed="left").agg(agg)
    counts = out["close"].resample("5min", label="right", closed="left").count()
    bars["source_m1_bars"] = counts
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    non_empty = len(bars)
    incomplete_removed = int((bars["source_m1_bars"] != 5).sum())
    bars = bars.loc[bars["source_m1_bars"] == 5].copy()
    report = ResampleReport(
        source_m1_rows=len(out),
        non_empty_m5_bars_before_filter=non_empty,
        incomplete_m5_bars_removed=incomplete_removed,
        final_m5_bars=len(bars),
    )
    return bars, report


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _require_datetime_index(df)
    idx = out.index
    out["hour_utc"] = idx.hour
    out["minute_utc"] = idx.minute
    out["day_of_week"] = idx.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_utc"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_utc"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    return out
