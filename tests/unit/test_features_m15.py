from __future__ import annotations

import numpy as np
import pandas as pd

from capstone_trading.data.features_m15 import add_relative_price_features, add_targets


def make_long_bars(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01", periods=rows, freq="15min", tz="UTC", name="time"
    )
    base = 2000 + np.linspace(0, 10, rows) + np.sin(np.arange(rows) / 7)
    return pd.DataFrame(
        {
            "open": base - 0.15,
            "high": base + 0.4,
            "low": base - 0.5,
            "close": base,
            "volume": 1000 + (np.arange(rows) % 31) * 3.0,
            "source_m1_bars": np.full(rows, 15, dtype=np.int16),
        },
        index=index,
    )


def test_feature_formulas_have_expected_indicator_conventions() -> None:
    bars = make_long_bars()
    features = add_relative_price_features(bars)
    row = features.iloc[220]
    expected_ema = (
        bars["close"].ewm(span=200, adjust=False, min_periods=200).mean().iloc[220]
    )
    expected_rsi_gain = (
        bars["close"].diff().clip(lower=0).rolling(14, min_periods=14).mean().iloc[220]
    )
    expected_rsi_loss = (
        (-bars["close"].diff().clip(upper=0))
        .rolling(14, min_periods=14)
        .mean()
        .iloc[220]
    )
    expected_rsi = 100 - (100 / (1 + expected_rsi_gain / expected_rsi_loss))
    assert np.isclose(
        row["close_ema_200_ratio"], bars["close"].iloc[220] / expected_ema - 1
    )
    assert np.isclose(row["rsi_14"], expected_rsi)


def test_targets_do_not_cross_gap() -> None:
    bars = make_long_bars(30).drop(make_long_bars(30).index[15])
    features = add_relative_price_features(bars)
    targeted = add_targets(features)
    row_before_gap = bars.index[14]
    assert np.isnan(targeted.loc[row_before_gap, "target_ret_fwd"])
    assert np.isnan(targeted.loc[row_before_gap, "target_dir"])
