from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RAW_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_COLUMNS = ["target_ret_fwd", "target_dir", "target_class_3"]
RAW_CONTEXT_COLUMNS = ["open", "high", "low", "close", "volume", "source_m1_bars"]


@dataclass
class CleaningReport:
    input_rows: int
    output_rows: int
    duplicate_timestamps: int
    missing_numeric_values: int
    invalid_ohlc_rows: int
    negative_volume_rows: int
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None


@dataclass
class ResampleReport:
    timeframe: str
    bar_minutes: int
    candidate_bars: int
    bars_with_ohlc: int
    complete_bars: int
    incomplete_bars_removed: int
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None
    non_contiguous_bar_gaps: int
    maximum_bar_gap_minutes: float


@dataclass
class DatasetReport:
    timeframe: str
    variant: str
    rows_pre_dropna: int
    rows_final: int
    columns_final: int
    feature_count: int
    target_0_rows: int
    target_1_rows: int
    target_1_rate: float
    mean_target_return_bps: float
    mean_abs_target_return_bps: float
    std_target_return_bps: float
    saved_to: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build clean M5 and M15 relative-feature datasets from the Dukascopy XAUUSD M1 master file. "
            "Outputs price-only and volume-assisted variants for timeframe feasibility analysis."
        )
    )
    parser.add_argument(
        "--input",
        default="data/raw/dukascopy_xauusd_m1_master.parquet",
        help="Path to the aggregated Dukascopy XAUUSD M1 master file.",
    )
    parser.add_argument(
        "--symbol",
        default="dukascopy_xauusd",
        help="Output symbol prefix.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        type=int,
        default=[5, 15],
        help="Timeframes in minutes to build, e.g. 5 15.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Inclusive UTC start timestamp/date. Leave empty to use the full M1 master file.",
    )
    parser.add_argument(
        "--end-exclusive",
        default=None,
        help="Exclusive UTC end timestamp/date. Leave empty to use the full M1 master file.",
    )
    parser.add_argument(
        "--horizon-bars",
        type=int,
        default=1,
        help="Prediction horizon in selected-timeframe bars.",
    )
    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=0.0,
        help="Absolute log-return threshold for target_class_3 neutral class.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/capstone_methodology",
        help="Clean output directory for processed data, reports and metadata.",
    )
    return parser


def to_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_parquet(df: pd.DataFrame, path_without_suffix: Path) -> Path:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    path = path_without_suffix.with_suffix(".parquet")
    df.to_parquet(path, index=True)
    return path


def load_m1_master(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("The timestamp column contains non-numeric values.")
        df["time"] = pd.to_datetime(df["timestamp"].round().astype("int64"), unit="ms", utc=True)
        df = df.set_index("time")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input must contain a time/timestamp column or a DatetimeIndex.")

    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "time"

    if "tick_volume" in df.columns and "volume" not in df.columns:
        raise ValueError(
            "The M1 master file contains 'tick_volume' rather than 'volume'. "
            "Use the corrected Dukascopy aggregator before building model datasets."
        )

    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Dukascopy columns: {missing}")

    return df[RAW_COLUMNS].sort_index()


def clean_m1_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    input_rows = int(len(df))
    duplicate_timestamps = int(df.index.duplicated().sum())

    if duplicate_timestamps > 0:
        raise ValueError(f"M1 master contains duplicate timestamps: {duplicate_timestamps}")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    numeric = df[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    missing_numeric_values = int(numeric.isna().sum().sum())
    if missing_numeric_values > 0:
        raise ValueError(f"M1 master contains missing/non-numeric OHLCV values: {missing_numeric_values}")

    invalid_ohlc_mask = (
        (numeric["high"] < numeric["low"])
        | (numeric["high"] < numeric["open"])
        | (numeric["high"] < numeric["close"])
        | (numeric["low"] > numeric["open"])
        | (numeric["low"] > numeric["close"])
        | (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    invalid_ohlc_rows = int(invalid_ohlc_mask.sum())
    if invalid_ohlc_rows > 0:
        raise ValueError(f"M1 master contains invalid OHLC rows: {invalid_ohlc_rows}")

    negative_volume_rows = int((numeric["volume"] < 0).sum())
    if negative_volume_rows > 0:
        raise ValueError(f"M1 master contains negative volume rows: {negative_volume_rows}")

    cleaned = numeric.copy()
    cleaned.index.name = "time"

    report = CleaningReport(
        input_rows=input_rows,
        output_rows=int(len(cleaned)),
        duplicate_timestamps=duplicate_timestamps,
        missing_numeric_values=missing_numeric_values,
        invalid_ohlc_rows=invalid_ohlc_rows,
        negative_volume_rows=negative_volume_rows,
        first_timestamp_utc=str(cleaned.index.min()) if not cleaned.empty else None,
        last_timestamp_utc=str(cleaned.index.max()) if not cleaned.empty else None,
    )
    return cleaned, report


def resample_complete_bars(m1: pd.DataFrame, bar_minutes: int) -> tuple[pd.DataFrame, ResampleReport]:
    if bar_minutes < 1:
        raise ValueError("bar_minutes must be positive.")

    rule = f"{bar_minutes}min"
    agg = m1.resample(rule, label="right", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg["source_m1_bars"] = m1["close"].resample(rule, label="right", closed="left").count()

    candidate_bars = int(len(agg))
    bars_with_ohlc_df = agg.dropna(subset=["open", "high", "low", "close"])
    bars_with_ohlc = int(len(bars_with_ohlc_df))
    complete = bars_with_ohlc_df.loc[bars_with_ohlc_df["source_m1_bars"] == bar_minutes].copy()
    complete["source_m1_bars"] = complete["source_m1_bars"].astype("int16")

    expected_delta = pd.Timedelta(minutes=bar_minutes)
    observed_deltas = complete.index.to_series().diff().dropna()
    gap_deltas = observed_deltas[observed_deltas > expected_delta]

    report = ResampleReport(
        timeframe=f"M{bar_minutes}",
        bar_minutes=bar_minutes,
        candidate_bars=candidate_bars,
        bars_with_ohlc=bars_with_ohlc,
        complete_bars=int(len(complete)),
        incomplete_bars_removed=int(bars_with_ohlc - len(complete)),
        first_timestamp_utc=str(complete.index.min()) if not complete.empty else None,
        last_timestamp_utc=str(complete.index.max()) if not complete.empty else None,
        non_contiguous_bar_gaps=int(len(gap_deltas)),
        maximum_bar_gap_minutes=(
            float(gap_deltas.max().total_seconds() / 60) if not gap_deltas.empty else float(bar_minutes)
        ),
    )
    return complete, report


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_time_features(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    out = df.copy()
    idx = out.index

    minute_of_day = idx.hour * 60 + idx.minute
    out["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    out["day_of_week_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

    expected_delta = pd.Timedelta(minutes=bar_minutes)
    delta = idx.to_series().diff()
    out["is_after_gap"] = (delta > expected_delta).astype("int8")
    out["gap_minutes_from_prev_bar"] = delta.dt.total_seconds().div(60).fillna(bar_minutes)

    return out


def add_relative_price_features(bars: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    out = add_time_features(bars, bar_minutes)

    high_low = out["high"] - out["low"]
    previous_close = out["close"].shift(1)

    # OHLC is represented as scale-normalised relationships, not raw absolute price levels.
    # This preserves candle/location information while avoiding a model that simply learns the long-run gold price level.
    out["open_rel_prev_close"] = out["open"] / previous_close - 1
    out["high_rel_prev_close"] = out["high"] / previous_close - 1
    out["low_rel_prev_close"] = out["low"] / previous_close - 1
    out["close_rel_prev_close"] = out["close"] / previous_close - 1
    out["log_ret_1"] = np.log(out["close"] / previous_close)

    for window in [2, 3, 6, 12, 24, 48]:
        out[f"log_ret_{window}"] = np.log(out["close"] / out["close"].shift(window))

    out["open_to_close_pct"] = safe_divide(out["close"] - out["open"], out["open"])
    out["high_low_range_pct"] = safe_divide(high_low, out["close"])
    out["upper_wick_pct"] = safe_divide(out["high"] - out[["open", "close"]].max(axis=1), out["close"])
    out["lower_wick_pct"] = safe_divide(out[["open", "close"]].min(axis=1) - out["low"], out["close"])
    out["close_position_in_bar"] = safe_divide(out["close"] - out["low"], high_low)

    for window in [5, 10, 20, 50, 100, 200]:
        sma = out["close"].rolling(window=window, min_periods=window).mean()
        ema = out["close"].ewm(span=window, adjust=False, min_periods=window).mean()
        out[f"close_sma_{window}_ratio"] = out["close"] / sma - 1
        out[f"close_ema_{window}_ratio"] = out["close"] / ema - 1
        out[f"rolling_vol_{window}"] = out["log_ret_1"].rolling(window=window, min_periods=window).std()

    for window in [5, 10, 20]:
        out[f"roc_{window}"] = out["close"] / out["close"].shift(window) - 1

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

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
    out["atr_pct_14"] = safe_divide(true_range.rolling(window=14, min_periods=14).mean(), out["close"])

    bb_mid = out["close"].rolling(window=20, min_periods=20).mean()
    bb_std = out["close"].rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_width_pct_20"] = safe_divide(bb_upper - bb_lower, bb_mid)
    out["bb_position_20"] = safe_divide(out["close"] - bb_lower, bb_upper - bb_lower)

    # Retain only scale-normalised volume as a candidate volume-assisted feature.
    volume_mean_20 = out["volume"].rolling(window=20, min_periods=20).mean()
    volume_std_20 = out["volume"].rolling(window=20, min_periods=20).std()
    out["volume_z20"] = (out["volume"] - volume_mean_20) / volume_std_20.replace(0, np.nan)

    return out


def add_targets(df: pd.DataFrame, bar_minutes: int, horizon_bars: int, neutral_threshold: float) -> pd.DataFrame:
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive.")
    if neutral_threshold < 0:
        raise ValueError("neutral_threshold must be non-negative.")

    out = df.copy()
    expected_delta = pd.Timedelta(minutes=bar_minutes * horizon_bars)
    observed_delta = out.index.to_series().shift(-horizon_bars) - out.index.to_series()
    contiguous_target = observed_delta == expected_delta

    target_ret = np.log(out["close"].shift(-horizon_bars) / out["close"])
    out["target_ret_fwd"] = target_ret.where(contiguous_target)
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


def finalise_dataset(pre_dropna: pd.DataFrame) -> pd.DataFrame:
    final = pre_dropna.replace([np.inf, -np.inf], np.nan).dropna().copy()
    final["target_dir"] = final["target_dir"].astype("int8")
    final["target_class_3"] = final["target_class_3"].astype("int8")
    return final


def get_price_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(RAW_CONTEXT_COLUMNS + TARGET_COLUMNS + ["volume_z20"])
    return [column for column in df.columns if column not in excluded]


def build_variant_datasets(features_with_targets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    price_feature_columns = get_price_feature_columns(features_with_targets)
    target_columns = TARGET_COLUMNS.copy()

    price_only = features_with_targets[price_feature_columns + target_columns].copy()
    volume_assisted = features_with_targets[price_feature_columns + ["volume_z20"] + target_columns].copy()

    return {
        "price_only": price_only,
        "volume_assisted": volume_assisted,
    }


def build_dataset_report(
    timeframe: str,
    variant: str,
    pre_dropna: pd.DataFrame,
    final: pd.DataFrame,
    feature_columns: Iterable[str],
    saved_to: Path,
) -> DatasetReport:
    target = final["target_dir"].astype(int)
    counts = target.value_counts()
    target_ret_bps = final["target_ret_fwd"] * 10_000

    return DatasetReport(
        timeframe=timeframe,
        variant=variant,
        rows_pre_dropna=int(len(pre_dropna)),
        rows_final=int(len(final)),
        columns_final=int(len(final.columns)),
        feature_count=int(len(list(feature_columns))),
        target_0_rows=int(counts.get(0, 0)),
        target_1_rows=int(counts.get(1, 0)),
        target_1_rate=float((target == 1).mean()) if len(final) else float("nan"),
        mean_target_return_bps=float(target_ret_bps.mean()) if len(final) else float("nan"),
        mean_abs_target_return_bps=float(target_ret_bps.abs().mean()) if len(final) else float("nan"),
        std_target_return_bps=float(target_ret_bps.std()) if len(final) else float("nan"),
        saved_to=str(saved_to),
    )


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    report_dir = output_dir / "reports" / "data_quality"
    docs_dir = output_dir / "docs"
    symbol = args.symbol.lower()

    if any(tf < 1 for tf in args.timeframes):
        raise ValueError("All timeframes must be positive minute values.")

    raw_m1 = load_m1_master(input_path)

    start_ts = to_utc_timestamp(args.start)
    end_exclusive_ts = to_utc_timestamp(args.end_exclusive)
    if start_ts is not None:
        raw_m1 = raw_m1.loc[raw_m1.index >= start_ts]
    if end_exclusive_ts is not None:
        raw_m1 = raw_m1.loc[raw_m1.index < end_exclusive_ts]
    if raw_m1.empty:
        raise SystemExit("No M1 rows available after optional date filters.")

    print(f"Loading M1 master: {input_path}")
    print(f"M1 range: {raw_m1.index.min()} -> {raw_m1.index.max()} | rows={len(raw_m1):,}")

    cleaned_m1, cleaning_report = clean_m1_bars(raw_m1)
    cleaned_m1_path = save_parquet(cleaned_m1, processed_dir / f"{symbol}_m1_clean")

    all_resample_reports: list[ResampleReport] = []
    all_dataset_reports: list[DatasetReport] = []
    metadata: dict[str, object] = {
        "purpose": "Clean relative-feature dataset build for XAUUSD timeframe feasibility analysis.",
        "input_file": str(input_path),
        "symbol": symbol,
        "base_timeframe": "M1",
        "timeframes_minutes": args.timeframes,
        "start_filter_inclusive": args.start,
        "end_filter_exclusive": args.end_exclusive,
        "target_horizon_bars": args.horizon_bars,
        "neutral_threshold": args.neutral_threshold,
        "cleaned_m1_saved_to": str(cleaned_m1_path),
        "cleaning_report": asdict(cleaning_report),
        "feature_policy": {
            "price_only": "Scale-normalised price, volatility, technical and time features only. Raw absolute OHLCV columns are excluded from model-ready datasets; OHLC information is retained through relative OHLC-derived features such as open/high/low/close relative to the previous close, candle body, wick, range, returns and technical ratios.",
            "volume_assisted": "Same as price-only plus volume_z20. Raw volume is retained only in bar files for audit and EDA.",
        },
        "resampling_rule": "right-labelled [t-bar, t) bars; only complete bars with the expected number of M1 observations are retained.",
        "outputs": {},
    }

    for bar_minutes in args.timeframes:
        timeframe = f"M{bar_minutes}"
        print(f"\nBuilding {timeframe} datasets...")

        bars, resample_report = resample_complete_bars(cleaned_m1, bar_minutes)
        all_resample_reports.append(resample_report)

        if bars.empty:
            raise ValueError(f"No complete {timeframe} bars were produced.")

        bars_path = save_parquet(bars, processed_dir / f"{symbol}_m{bar_minutes}_bars_with_volume")

        features = add_relative_price_features(bars, bar_minutes)
        features_with_targets = add_targets(
            features,
            bar_minutes=bar_minutes,
            horizon_bars=args.horizon_bars,
            neutral_threshold=args.neutral_threshold,
        )

        variant_pre_dropna = build_variant_datasets(features_with_targets)
        metadata["outputs"][timeframe] = {
            "bars_with_volume": str(bars_path),
            "resample_report": asdict(resample_report),
            "variants": {},
        }

        for variant_name, pre_dropna in variant_pre_dropna.items():
            pre_path = save_parquet(
                pre_dropna,
                processed_dir / f"{symbol}_m{bar_minutes}_{variant_name}_relative_features_pre_dropna",
            )
            final = finalise_dataset(pre_dropna)
            final_path = save_parquet(
                final,
                processed_dir / f"{symbol}_m{bar_minutes}_{variant_name}_relative_dataset",
            )

            feature_columns = [column for column in final.columns if column not in TARGET_COLUMNS]
            report = build_dataset_report(
                timeframe=timeframe,
                variant=variant_name,
                pre_dropna=pre_dropna,
                final=final,
                feature_columns=feature_columns,
                saved_to=final_path,
            )
            all_dataset_reports.append(report)

            metadata["outputs"][timeframe]["variants"][variant_name] = {
                "features_pre_dropna": str(pre_path),
                "final_dataset": str(final_path),
                "feature_columns": feature_columns,
                "target_columns": TARGET_COLUMNS,
                "row_count": int(len(final)),
                "column_count": int(len(final.columns)),
                "first_timestamp_utc": str(final.index.min()) if not final.empty else None,
                "last_timestamp_utc": str(final.index.max()) if not final.empty else None,
            }

            print(
                f"{timeframe} {variant_name}: rows={len(final):,}, "
                f"features={len(feature_columns)}, target_1_rate={report.target_1_rate:.4f}"
            )

    report_dir.mkdir(parents=True, exist_ok=True)
    resample_report_path = report_dir / f"{symbol}_m5_m15_resample_summary.csv"
    dataset_report_path = report_dir / f"{symbol}_m5_m15_dataset_summary.csv"
    pd.DataFrame([asdict(report) for report in all_resample_reports]).to_csv(resample_report_path, index=False)
    pd.DataFrame([asdict(report) for report in all_dataset_reports]).to_csv(dataset_report_path, index=False)

    metadata["resample_summary_csv"] = str(resample_report_path)
    metadata["dataset_summary_csv"] = str(dataset_report_path)
    metadata_path = docs_dir / f"{symbol}_m5_m15_relative_dataset_metadata.json"
    save_json(metadata, metadata_path)

    print("\nDone.")
    print(f"Cleaned M1:        {cleaned_m1_path}")
    print(f"Resample summary: {resample_report_path}")
    print(f"Dataset summary:  {dataset_report_path}")
    print(f"Metadata:         {metadata_path}")


if __name__ == "__main__":
    main()
