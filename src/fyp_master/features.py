from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_price_features(df: pd.DataFrame, include_volume_feature: bool = False) -> pd.DataFrame:
    """Add OHLC-derived features; volume is optional and excluded in the primary live-transfer experiment."""
    out = df.copy()
    close = out["close"]
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["ret_12"] = close.pct_change(12)
    out["log_ret_1"] = np.log(close / close.shift(1))
    for win in [5, 10, 20, 50]:
        out[f"sma_{win}"] = close.rolling(win).mean()
        out[f"ema_{win}"] = _ema(close, win)
        out[f"close_over_sma_{win}"] = close / out[f"sma_{win}"]
        out[f"close_over_ema_{win}"] = close / out[f"ema_{win}"]
    out["macd"] = _ema(close, 12) - _ema(close, 26)
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["rsi_14"] = _rsi(close, 14)
    out["roc_5"] = close.pct_change(5)
    out["roc_10"] = close.pct_change(10)
    out["atr_14"] = _atr(out, 14)
    out["rolling_std_12"] = out["log_ret_1"].rolling(12).std()
    out["rolling_std_24"] = out["log_ret_1"].rolling(24).std()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_mid_20"] = bb_mid
    out["bb_upper_20_2"] = bb_mid + 2 * bb_std
    out["bb_lower_20_2"] = bb_mid - 2 * bb_std
    out["bb_width_20_2"] = (out["bb_upper_20_2"] - out["bb_lower_20_2"]) / bb_mid
    out["hl_range"] = out["high"] - out["low"]
    out["body_size"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    if include_volume_feature:
        if "volume" not in out.columns:
            raise ValueError("include_volume_feature=True requires a 'volume' column.")
        out["volume_z20"] = (
            (out["volume"] - out["volume"].rolling(20).mean())
            / out["volume"].rolling(20).std()
        )
    return out


def add_targets(
    df: pd.DataFrame,
    horizon_bars: int = 1,
    neutral_threshold: float = 0.0,
    bar_minutes: int = 5,
) -> pd.DataFrame:
    """
    Create forward-return targets only when the future bar is continuously
    reachable at the intended M5 horizon.

    This prevents daily breaks, weekends and holiday closures from being
    incorrectly labelled as one-bar-ahead M5 trading opportunities.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least 1.")

    out = df.copy()

    current_time = out.index.to_series()
    future_time = current_time.shift(-horizon_bars)
    future_close = out["close"].shift(-horizon_bars)

    expected_delta = pd.Timedelta(minutes=bar_minutes * horizon_bars)
    is_contiguous_horizon = (future_time - current_time) == expected_delta

    out["target_ret_fwd"] = np.log(future_close / out["close"]).where(is_contiguous_horizon)

    if neutral_threshold <= 0:
        target = pd.Series(pd.NA, index=out.index, dtype="Int64")
        valid = out["target_ret_fwd"].notna()
        target.loc[valid] = (out.loc[valid, "target_ret_fwd"] > 0).astype(int)
        out["target_dir"] = target
    else:
        target = pd.Series(pd.NA, index=out.index, dtype="Int64")
        valid = out["target_ret_fwd"].notna()
        target.loc[valid] = np.select(
            [
                out.loc[valid, "target_ret_fwd"] > neutral_threshold,
                out.loc[valid, "target_ret_fwd"] < -neutral_threshold,
            ],
            [1, -1],
            default=0,
        )
        out["target_class_3"] = target

    return out


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna()
