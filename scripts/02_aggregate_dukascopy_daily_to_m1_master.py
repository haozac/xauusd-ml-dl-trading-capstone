from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


FILE_PATTERN = re.compile(r"^xauusd_m1_(\d{4}-\d{2}-\d{2})\.csv$")
REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


@dataclass
class DailyValidationResult:
    file_name: str
    expected_day: str
    status: str
    rows: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    duplicate_timestamps: int
    bad_ohlc_rows: int
    negative_volume_rows: int
    gaps_gt_1m: int
    max_gap_minutes: float
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate all validated Dukascopy XAUUSD M1 daily CSV files "
            "into one historical M1 master dataset."
        )
    )
    parser.add_argument(
        "--daily-dir",
        default="data/external/dukascopy/daily",
        help="Folder containing daily Dukascopy CSV files.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/raw/dukascopy_xauusd_m1_master.csv",
        help="Output aggregated CSV file.",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/raw/dukascopy_xauusd_m1_master.parquet",
        help="Output aggregated Parquet file.",
    )
    parser.add_argument(
        "--validation-report",
        default="reports/data_quality/dukascopy_m1_aggregation_validation.csv",
        help="Per-file structural validation report.",
    )
    parser.add_argument(
        "--metadata-json",
        default="docs/dukascopy_m1_master_metadata.json",
        help="Metadata JSON output file.",
    )
    return parser.parse_args()


def get_expected_day(path: Path) -> pd.Timestamp:
    match = FILE_PATTERN.match(path.name)
    if not match:
        raise ValueError(
            f"Unexpected daily filename format: {path.name}. "
            "Expected: xauusd_m1_YYYY-MM-DD.csv"
        )
    return pd.Timestamp(match.group(1), tz="UTC")


def load_and_validate_daily_file(
    path: Path,
) -> tuple[pd.DataFrame | None, DailyValidationResult]:
    expected_day = get_expected_day(path)

    if path.stat().st_size == 0:
        return None, DailyValidationResult(
            file_name=path.name,
            expected_day=str(expected_day.date()),
            status="FAIL",
            rows=0,
            first_timestamp_utc="",
            last_timestamp_utc="",
            duplicate_timestamps=0,
            bad_ohlc_rows=0,
            negative_volume_rows=0,
            gaps_gt_1m=0,
            max_gap_minutes=0.0,
            reason="zero_byte_file",
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None, DailyValidationResult(
            file_name=path.name,
            expected_day=str(expected_day.date()),
            status="FAIL",
            rows=0,
            first_timestamp_utc="",
            last_timestamp_utc="",
            duplicate_timestamps=0,
            bad_ohlc_rows=0,
            negative_volume_rows=0,
            gaps_gt_1m=0,
            max_gap_minutes=0.0,
            reason="empty_csv_no_columns",
        )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        return None, DailyValidationResult(
            file_name=path.name,
            expected_day=str(expected_day.date()),
            status="FAIL",
            rows=len(df),
            first_timestamp_utc="",
            last_timestamp_utc="",
            duplicate_timestamps=0,
            bad_ohlc_rows=0,
            negative_volume_rows=0,
            gaps_gt_1m=0,
            max_gap_minutes=0.0,
            reason=f"missing_columns:{','.join(sorted(missing_columns))}",
        )

    numeric_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric_columns].isna().any().any():
        return None, DailyValidationResult(
            file_name=path.name,
            expected_day=str(expected_day.date()),
            status="FAIL",
            rows=len(df),
            first_timestamp_utc="",
            last_timestamp_utc="",
            duplicate_timestamps=0,
            bad_ohlc_rows=0,
            negative_volume_rows=0,
            gaps_gt_1m=0,
            max_gap_minutes=0.0,
            reason="non_numeric_or_missing_values",
        )

    # Dukascopy CLI exports timestamp in Unix milliseconds.
    df["time"] = pd.to_datetime(
        df["timestamp"].round().astype("int64"),
        unit="ms",
        utc=True,
    )

    original_monotonic = bool(df["time"].is_monotonic_increasing)
    duplicate_timestamps = int(df["time"].duplicated().sum())

    bad_ohlc_mask = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    bad_ohlc_rows = int(bad_ohlc_mask.sum())
    negative_volume_rows = int((df["volume"] < 0).sum())

    same_requested_day = bool(
        (df["time"].dt.floor("D") == expected_day).all()
    )

    df = df.sort_values("time").reset_index(drop=True)

    gaps = df["time"].diff().dropna()
    gaps_gt_1m = gaps[gaps > pd.Timedelta(minutes=1)]
    gap_count = int(len(gaps_gt_1m))
    max_gap_minutes = (
        float(gaps_gt_1m.max().total_seconds() / 60)
        if gap_count > 0
        else 1.0
    )

    fail_reasons = []
    if not original_monotonic:
        fail_reasons.append("timestamps_not_increasing")
    if duplicate_timestamps > 0:
        fail_reasons.append("duplicate_timestamps")
    if bad_ohlc_rows > 0:
        fail_reasons.append("invalid_ohlc")
    if negative_volume_rows > 0:
        fail_reasons.append("negative_volume")
    if not same_requested_day:
        fail_reasons.append("timestamp_outside_filename_day")

    status = "FAIL" if fail_reasons else "PASS"
    reason = ",".join(fail_reasons) if fail_reasons else "structural_validation_passed"

    result = DailyValidationResult(
        file_name=path.name,
        expected_day=str(expected_day.date()),
        status=status,
        rows=int(len(df)),
        first_timestamp_utc=str(df["time"].min()),
        last_timestamp_utc=str(df["time"].max()),
        duplicate_timestamps=duplicate_timestamps,
        bad_ohlc_rows=bad_ohlc_rows,
        negative_volume_rows=negative_volume_rows,
        gaps_gt_1m=gap_count,
        max_gap_minutes=max_gap_minutes,
        reason=reason,
    )

    if status == "FAIL":
        return None, result

    # Keep raw Dukascopy volume as 'volume'. It is not MT5 tick_volume.
    clean_df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    return clean_df, result


def main() -> None:
    args = parse_args()

    daily_dir = Path(args.daily_dir)
    output_csv = Path(args.output_csv)
    output_parquet = Path(args.output_parquet)
    validation_report = Path(args.validation_report)
    metadata_json = Path(args.metadata_json)

    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily data directory does not exist: {daily_dir}")

    daily_files = sorted(daily_dir.glob("xauusd_m1_*.csv"))

    if not daily_files:
        raise FileNotFoundError(f"No daily XAUUSD M1 CSV files found in: {daily_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    validation_report.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(daily_files)} daily CSV files.")
    print("Validating each daily file before aggregation...")

    frames: list[pd.DataFrame] = []
    validation_results: list[DailyValidationResult] = []

    for index, path in enumerate(daily_files, start=1):
        df_day, result = load_and_validate_daily_file(path)
        validation_results.append(result)

        if result.status == "PASS" and df_day is not None:
            frames.append(df_day)
        else:
            print(f"FAIL {path.name} | {result.reason}")

        if index % 250 == 0 or index == len(daily_files):
            print(f"Checked {index}/{len(daily_files)} files...")

    report_df = pd.DataFrame([asdict(result) for result in validation_results])
    report_df.to_csv(validation_report, index=False)

    failed_files = report_df[report_df["status"] == "FAIL"]

    if not failed_files.empty:
        print("\nAggregation stopped because structurally invalid daily files were found.")
        print(f"Failed files: {len(failed_files)}")
        print(f"Validation report saved to: {validation_report.resolve()}")
        print("\nFirst failed files:")
        print(failed_files[["file_name", "reason"]].head(20).to_string(index=False))
        raise SystemExit(
            "Resolve or re-download failed daily files before building the M1 master dataset."
        )

    if not frames:
        raise ValueError("No structurally valid daily files were available for aggregation.")

    print("Combining daily files...")
    master = pd.concat(frames, ignore_index=True)
    master = master.sort_values("time").reset_index(drop=True)

    cross_file_duplicate_count = int(master["time"].duplicated().sum())
    if cross_file_duplicate_count > 0:
        duplicates_path = validation_report.parent / "dukascopy_m1_duplicate_timestamps.csv"
        master[master["time"].duplicated(keep=False)].to_csv(duplicates_path, index=False)
        raise ValueError(
            f"Duplicate timestamps found across daily files: {cross_file_duplicate_count}. "
            f"See: {duplicates_path}"
        )

    master = master.set_index("time")

    master.to_csv(output_csv)
    master.to_parquet(output_parquet)

    global_gaps = master.index.to_series().diff().dropna()
    global_gaps_gt_1m = global_gaps[global_gaps > pd.Timedelta(minutes=1)]

    metadata = {
        "source": "Dukascopy historical XAUUSD M1 BID data retrieved via dukascopy-cli",
        "raw_daily_directory": str(daily_dir),
        "daily_files_aggregated": int(len(daily_files)),
        "daily_files_failed_structural_validation": 0,
        "rows": int(len(master)),
        "columns": list(master.columns),
        "first_timestamp_utc": str(master.index.min()),
        "last_timestamp_utc": str(master.index.max()),
        "duplicate_timestamps": 0,
        "global_gaps_greater_than_1_minute": int(len(global_gaps_gt_1m)),
        "maximum_gap_minutes": (
            float(global_gaps_gt_1m.max().total_seconds() / 60)
            if not global_gaps_gt_1m.empty
            else 1.0
        ),
        "volume_note": (
            "The 'volume' column is retained as Dukascopy-provided BID-side volume. "
            "It must not be interpreted as MT5 tick_volume."
        ),
        "output_csv": str(output_csv),
        "output_parquet": str(output_parquet),
        "validation_report": str(validation_report),
    }

    with metadata_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nDone.")
    print(f"Daily files aggregated: {len(daily_files)}")
    print(f"Rows: {len(master)}")
    print(f"Start: {master.index.min()}")
    print(f"End:   {master.index.max()}")
    print(f"Saved CSV:       {output_csv.resolve()}")
    print(f"Saved Parquet:   {output_parquet.resolve()}")
    print(f"Validation report: {validation_report.resolve()}")
    print(f"Metadata:        {metadata_json.resolve()}")


if __name__ == "__main__":
    main()