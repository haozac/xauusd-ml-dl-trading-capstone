"""Exact Notebook 7 M15 relative-feature reconstruction.

The formulas intentionally mirror the official dataset-preparation script.
Do not replace them with TA-Lib or another technical-analysis implementation;
indicator conventions such as RSI, ATR, rolling standard deviation, and EMA
initialisation must remain identical to the research pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from capstone_trading.data.canonical_bars import M15_DELTA, M15_MINUTES
from capstone_trading.errors import FeatureParityError, HistoricalDataError

TARGET_COLUMNS = ("target_ret_fwd", "target_dir", "target_class_3")
RAW_CONTEXT_COLUMNS = ("open", "high", "low", "close", "volume", "source_m1_bars")
SMA_EMA_VOL_WINDOWS = (5, 10, 20, 50, 100, 200)
LOG_RETURN_WINDOWS = (2, 3, 6, 12, 24, 48)
ROC_WINDOWS = (5, 10, 20)


@dataclass(frozen=True)
class FeatureBuildReport:
    bars_rows: int
    pre_dropna_rows: int
    final_rows: int
    feature_count: int
    target_count: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    dropped_rows: int
    infinite_values_before_dropna: int


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_time_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    index = pd.DatetimeIndex(out.index)
    minute_of_day = index.hour * 60 + index.minute
    out["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    out["day_of_week_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (index.month - 1) / 12)
    delta = index.to_series().diff()
    out["is_after_gap"] = (delta > M15_DELTA).astype("int8")
    out["gap_minutes_from_prev_bar"] = (
        delta.dt.total_seconds().div(60).fillna(M15_MINUTES)
    )
    return out


def add_relative_price_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(bars)
    high_low = out["high"] - out["low"]
    previous_close = out["close"].shift(1)

    out["open_rel_prev_close"] = out["open"] / previous_close - 1
    out["high_rel_prev_close"] = out["high"] / previous_close - 1
    out["low_rel_prev_close"] = out["low"] / previous_close - 1
    out["close_rel_prev_close"] = out["close"] / previous_close - 1
    out["log_ret_1"] = np.log(out["close"] / previous_close)
    for window in LOG_RETURN_WINDOWS:
        out[f"log_ret_{window}"] = np.log(out["close"] / out["close"].shift(window))

    out["open_to_close_pct"] = safe_divide(out["close"] - out["open"], out["open"])
    out["high_low_range_pct"] = safe_divide(high_low, out["close"])
    out["upper_wick_pct"] = safe_divide(
        out["high"] - out[["open", "close"]].max(axis=1), out["close"]
    )
    out["lower_wick_pct"] = safe_divide(
        out[["open", "close"]].min(axis=1) - out["low"], out["close"]
    )
    out["close_position_in_bar"] = safe_divide(out["close"] - out["low"], high_low)

    for window in SMA_EMA_VOL_WINDOWS:
        sma = out["close"].rolling(window=window, min_periods=window).mean()
        ema = out["close"].ewm(span=window, adjust=False, min_periods=window).mean()
        out[f"close_sma_{window}_ratio"] = out["close"] / sma - 1
        out[f"close_ema_{window}_ratio"] = out["close"] / ema - 1
        out[f"rolling_vol_{window}"] = (
            out["log_ret_1"].rolling(window=window, min_periods=window).std()
        )

    for window in ROC_WINDOWS:
        out[f"roc_{window}"] = out["close"] / out["close"].shift(window) - 1

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + relative_strength))

    previous_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - previous_close).abs(),
            (out["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["true_range_pct"] = safe_divide(true_range, out["close"])
    out["atr_pct_14"] = safe_divide(
        true_range.rolling(window=14, min_periods=14).mean(), out["close"]
    )

    bb_mid = out["close"].rolling(window=20, min_periods=20).mean()
    bb_std = out["close"].rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_width_pct_20"] = safe_divide(bb_upper - bb_lower, bb_mid)
    out["bb_position_20"] = safe_divide(out["close"] - bb_lower, bb_upper - bb_lower)

    volume_mean_20 = out["volume"].rolling(window=20, min_periods=20).mean()
    volume_std_20 = out["volume"].rolling(window=20, min_periods=20).std()
    out["volume_z20"] = (out["volume"] - volume_mean_20) / volume_std_20.replace(
        0, np.nan
    )
    return out


def add_targets(
    features: pd.DataFrame,
    *,
    horizon_bars: int = 1,
    neutral_threshold: float = 0.0,
) -> pd.DataFrame:
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    if neutral_threshold < 0:
        raise ValueError("neutral_threshold must be non-negative")

    out = features.copy()
    expected_delta = pd.Timedelta(minutes=M15_MINUTES * horizon_bars)
    observed_delta = out.index.to_series().shift(-horizon_bars) - out.index.to_series()
    contiguous_target = observed_delta == expected_delta
    target_return = np.log(out["close"].shift(-horizon_bars) / out["close"])
    out["target_ret_fwd"] = target_return.where(contiguous_target)
    out["target_dir"] = (out["target_ret_fwd"] > neutral_threshold).astype("float")
    out.loc[out["target_ret_fwd"].isna(), "target_dir"] = np.nan
    out["target_class_3"] = np.select(
        [
            out["target_ret_fwd"] > neutral_threshold,
            out["target_ret_fwd"] < -neutral_threshold,
        ],
        [1, -1],
        default=0,
    ).astype("float")
    out.loc[out["target_ret_fwd"].isna(), "target_class_3"] = np.nan
    return out


def build_volume_assisted_dataset(
    bars: pd.DataFrame,
    frozen_feature_order: Iterable[str],
) -> tuple[pd.DataFrame, FeatureBuildReport]:
    features = tuple(str(name) for name in frozen_feature_order)
    if not features:
        raise FeatureParityError("Frozen feature order must not be empty")
    if len(set(features)) != len(features):
        raise FeatureParityError("Frozen feature order contains duplicate names")
    if "volume_z20" not in features:
        raise FeatureParityError(
            "Volume-assisted feature order must contain volume_z20"
        )

    with_features = add_relative_price_features(bars)
    with_targets = add_targets(with_features)
    missing = [
        name
        for name in (*features, *TARGET_COLUMNS)
        if name not in with_targets.columns
    ]
    if missing:
        raise FeatureParityError(
            f"Feature builder did not produce required columns: {missing}"
        )

    pre_dropna = with_targets.loc[:, [*features, *TARGET_COLUMNS]].copy()
    numeric = pre_dropna.select_dtypes(include=[np.number])
    infinite_values_before = int(np.isinf(numeric.to_numpy(dtype=np.float64)).sum())
    final = pre_dropna.replace([np.inf, -np.inf], np.nan).dropna().copy()
    final["target_dir"] = final["target_dir"].astype("int8")
    final["target_class_3"] = final["target_class_3"].astype("int8")
    if final.empty:
        raise HistoricalDataError("Feature reconstruction produced no model-ready rows")
    if tuple(final.columns) != (*features, *TARGET_COLUMNS):
        raise FeatureParityError(
            "Final dataset column order differs from the frozen contract"
        )

    report = FeatureBuildReport(
        bars_rows=int(len(bars)),
        pre_dropna_rows=int(len(pre_dropna)),
        final_rows=int(len(final)),
        feature_count=len(features),
        target_count=len(TARGET_COLUMNS),
        first_timestamp_utc=final.index.min().isoformat(),
        last_timestamp_utc=final.index.max().isoformat(),
        dropped_rows=int(len(pre_dropna) - len(final)),
        infinite_values_before_dropna=infinite_values_before,
    )
    return final, report


def report_to_dict(report: FeatureBuildReport) -> dict[str, Any]:
    return asdict(report)
