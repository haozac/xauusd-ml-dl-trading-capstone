from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


@dataclass
class AuditResult:
    day: str
    status: str
    rows: int
    first_ts: str
    last_ts: str
    duplicate_timestamps: int
    bad_ohlc_rows: int
    num_gaps_gt_1m: int
    max_gap_minutes: float
    reasons: str
    file_name: str


def resolve_npx() -> str:
    candidates = []
    if sys.platform.startswith("win"):
        candidates.extend(
            [
                r"C:\Program Files\nodejs\npx.cmd",
                r"C:\Program Files\nodejs\npx.exe",
                "npx.cmd",
                "npx",
            ]
        )
    else:
        candidates.append("npx")

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found

    raise FileNotFoundError(
        "Could not find npx/npx.cmd. Open a fresh Command Prompt and make sure Node.js is in PATH."
    )


def iter_days(start_day: date, end_day: date):
    current = start_day
    while current <= end_day:
        yield current
        current += timedelta(days=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01", help="Start date inclusive, YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-31", help="End date inclusive, YYYY-MM-DD")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing daily files")
    return parser.parse_args()


def classify_day_profile(
    expected_day: date,
    rows: int,
    first_ts: pd.Timestamp,
    last_ts: pd.Timestamp,
    num_gaps_gt_1m: int,
    max_gap_minutes: float,
) -> tuple[str, str]:
    """
    Dukascopy/spot-gold style profile in UTC:
    - Mon-Thu: 23 trading hours => ~1380 rows, one daily break (~61 min)
    - Fri: early weekly close
        winter-style often ~1320 rows ending 21:59 UTC
        summer-style often ~1260 rows ending 20:59 UTC
    - Sat: closed (handled outside this function)
    - Sun: short opening session
        winter-style often ~60 rows starting 23:00 UTC
        summer-style often ~120 rows starting 22:00 UTC
    """

    wd = expected_day.weekday()  # Mon=0 ... Sun=6

    if wd == 6:  # Sunday
        if rows == 60 and first_ts.hour == 23 and last_ts.hour == 23:
            return "PASS", "none"
        if rows == 120 and first_ts.hour == 22 and last_ts.hour == 23:
            return "PASS", "none"
        if rows in (60, 120):
            return "REVIEW", "unexpected_sunday_open_time"
        return "REVIEW", "unexpected_sunday_profile"

    if wd == 4:  # Friday
        if rows == 1320 and last_ts.hour == 21 and last_ts.minute == 59 and num_gaps_gt_1m == 0:
            return "PASS", "none"
        if rows == 1260 and last_ts.hour == 20 and last_ts.minute == 59 and num_gaps_gt_1m == 0:
            return "PASS", "none"
        return "REVIEW", "unexpected_friday_profile"

    if wd in (0, 1, 2, 3):  # Mon-Thu
        if rows == 1380 and num_gaps_gt_1m == 1 and 60 <= max_gap_minutes <= 62:
            return "PASS", "none"
        # keep weekday anomalies as REVIEW, not FAIL, because holiday schedules can vary
        if rows >= 900:
            return "REVIEW", "weekday_profile_deviation"
        return "REVIEW", "low_weekday_row_count"

    return "REVIEW", "unexpected_profile"


def load_and_audit_csv(path: Path, expected_day: date) -> AuditResult:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return AuditResult(
            day=str(expected_day),
            status="FAIL",
            rows=0,
            first_ts="",
            last_ts="",
            duplicate_timestamps=0,
            bad_ohlc_rows=0,
            num_gaps_gt_1m=0,
            max_gap_minutes=0.0,
            reasons="empty_csv_no_columns",
            file_name=path.name,
        )

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        return AuditResult(
            day=str(expected_day),
            status="FAIL",
            rows=0,
            first_ts="",
            last_ts="",
            duplicate_timestamps=0,
            bad_ohlc_rows=0,
            num_gaps_gt_1m=0,
            max_gap_minutes=0.0,
            reasons=f"missing_columns:{','.join(sorted(missing))}",
            file_name=path.name,
        )

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        return AuditResult(
            day=str(expected_day),
            status="FAIL",
            rows=0,
            first_ts="",
            last_ts="",
            duplicate_timestamps=0,
            bad_ohlc_rows=0,
            num_gaps_gt_1m=0,
            max_gap_minutes=0.0,
            reasons="non_numeric_timestamp",
            file_name=path.name,
        )

    df["time"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    rows = len(df)
    if rows == 0:
        return AuditResult(
            day=str(expected_day),
            status="FAIL",
            rows=0,
            first_ts="",
            last_ts="",
            duplicate_timestamps=0,
            bad_ohlc_rows=0,
            num_gaps_gt_1m=0,
            max_gap_minutes=0.0,
            reasons="empty_file",
            file_name=path.name,
        )

    first_ts = df["time"].iloc[0]
    last_ts = df["time"].iloc[-1]

    duplicate_timestamps = int(df["time"].duplicated().sum())
    is_monotonic = bool(df["time"].is_monotonic_increasing)

    bad_ohlc_rows = int(
        (
            (df["high"] < df[["open", "close", "low"]].max(axis=1))
            | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        ).sum()
    )

    expected_day_ts = pd.Timestamp(expected_day, tz="UTC")
    same_day = (df["time"].dt.floor("D") == expected_day_ts).all()

    gaps = df["time"].diff().dropna()
    gap_gt_1m = gaps[gaps > pd.Timedelta(minutes=1)]
    num_gaps_gt_1m = int(len(gap_gt_1m))
    max_gap_minutes = float(gap_gt_1m.max().total_seconds() / 60) if num_gaps_gt_1m > 0 else 1.0

    hard_fail_reasons = []
    if not is_monotonic:
        hard_fail_reasons.append("timestamps_not_monotonic")
    if duplicate_timestamps > 0:
        hard_fail_reasons.append("duplicate_timestamps")
    if bad_ohlc_rows > 0:
        hard_fail_reasons.append("bad_ohlc")
    if not same_day:
        hard_fail_reasons.append("timestamps_outside_requested_day")

    if hard_fail_reasons:
        return AuditResult(
            day=str(expected_day),
            status="FAIL",
            rows=rows,
            first_ts=str(first_ts),
            last_ts=str(last_ts),
            duplicate_timestamps=duplicate_timestamps,
            bad_ohlc_rows=bad_ohlc_rows,
            num_gaps_gt_1m=num_gaps_gt_1m,
            max_gap_minutes=max_gap_minutes,
            reasons=",".join(hard_fail_reasons),
            file_name=path.name,
        )

    status, reasons = classify_day_profile(
        expected_day=expected_day,
        rows=rows,
        first_ts=first_ts,
        last_ts=last_ts,
        num_gaps_gt_1m=num_gaps_gt_1m,
        max_gap_minutes=max_gap_minutes,
    )

    return AuditResult(
        day=str(expected_day),
        status=status,
        rows=rows,
        first_ts=str(first_ts),
        last_ts=str(last_ts),
        duplicate_timestamps=duplicate_timestamps,
        bad_ohlc_rows=bad_ohlc_rows,
        num_gaps_gt_1m=num_gaps_gt_1m,
        max_gap_minutes=max_gap_minutes,
        reasons=reasons if reasons else "none",
        file_name=path.name,
    )


def append_audit_row(csv_path: Path, result: AuditResult) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "day",
                    "status",
                    "rows",
                    "first_ts",
                    "last_ts",
                    "duplicate_timestamps",
                    "bad_ohlc_rows",
                    "num_gaps_gt_1m",
                    "max_gap_minutes",
                    "reasons",
                    "file_name",
                ]
            )
        writer.writerow(
            [
                result.day,
                result.status,
                result.rows,
                result.first_ts,
                result.last_ts,
                result.duplicate_timestamps,
                result.bad_ohlc_rows,
                result.num_gaps_gt_1m,
                result.max_gap_minutes,
                result.reasons if result.reasons else "none",
                result.file_name,
            ]
        )


def main() -> None:
    args = parse_args()
    start_day = date.fromisoformat(args.start)
    end_day = date.fromisoformat(args.end)

    project_root = Path(__file__).resolve().parents[1]
    temp_download_dir = project_root / "download"
    raw_daily_dir = project_root / "data" / "external" / "dukascopy" / "daily"
    log_dir = project_root / "logs"
    text_log_path = log_dir / "dukascopy_xauusd_m1_daily.log"
    audit_csv_path = log_dir / "dukascopy_xauusd_m1_daily_audit.csv"

    temp_download_dir.mkdir(parents=True, exist_ok=True)
    raw_daily_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    npx_exe = resolve_npx()

    def log(msg: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with text_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("=" * 80)
    log("Starting daily Dukascopy download + validation for XAUUSD M1")
    log(f"Range: {start_day} -> {end_day} inclusive")
    log(f"Using NPX executable: {npx_exe}")
    log("Command pattern: npx dukascopy-cli -i xauusd -from <day> -to <next_day> -t m1 -f csv -p bid -v")
    log("Cache is intentionally NOT used.")
    log("Saturday is skipped and logged as CLOSED; no synthetic empty raw CSV is created.")
    log("=" * 80)

    for day in iter_days(start_day, end_day):
        final_name = f"xauusd_m1_{day.isoformat()}.csv"
        final_path = raw_daily_dir / final_name

        # Clean temporary download folder so only today's file can appear there
        for old_csv in temp_download_dir.glob("*.csv"):
            try:
                old_csv.unlink()
            except Exception:
                pass

        # Saturday: closed market
        if day.weekday() == 5:
            closed = AuditResult(
                day=str(day),
                status="CLOSED",
                rows=0,
                first_ts="",
                last_ts="",
                duplicate_timestamps=0,
                bad_ohlc_rows=0,
                num_gaps_gt_1m=0,
                max_gap_minutes=0.0,
                reasons="saturday_market_closed",
                file_name=final_name,
            )
            append_audit_row(audit_csv_path, closed)
            log(f"CLOSED {day} | saturday_market_closed")
            continue

        if args.skip_existing and final_path.exists() and final_path.stat().st_size > 0:
            audit = load_and_audit_csv(final_path, day)
            append_audit_row(audit_csv_path, audit)
            log(
                f"SKIP   {day} | status={audit.status} | rows={audit.rows} | "
                f"first={audit.first_ts or 'NA'} | last={audit.last_ts or 'NA'} | reasons={audit.reasons}"
            )
            continue

        next_day = day + timedelta(days=1)

        cmd = [
            npx_exe,
            "dukascopy-cli",
            "-i", "xauusd",
            "-from", day.isoformat(),
            "-to", next_day.isoformat(),
            "-t", "m1",
            "-f", "csv",
            "-p", "bid",
            "-v",
        ]

        log(f"START  {day} | {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)

        if result.returncode != 0:
            stderr_text = result.stderr.strip()
            stdout_text = result.stdout.strip()

            known_empty_day_bug = "ERR_INVALID_ARG_TYPE" in stderr_text and "Downloading historical price data for:" in stdout_text

            holiday_like = (
                day.weekday() == 5
                or (day.month == 1 and day.day == 1)
                or (day.month == 12 and day.day == 25)
            )

            if known_empty_day_bug and holiday_like:
                closed = AuditResult(
                    day=str(day),
                    status="CLOSED",
                    rows=0,
                    first_ts="",
                    last_ts="",
                    duplicate_timestamps=0,
                    bad_ohlc_rows=0,
                    num_gaps_gt_1m=0,
                    max_gap_minutes=0.0,
                    reasons="holiday_market_closed_empty_response",
                    file_name=final_name,
                )
                append_audit_row(audit_csv_path, closed)
                log(f"CLOSED {day} | holiday_market_closed_empty_response")
                continue

            fail = AuditResult(
                day=str(day),
                status="FAIL",
                rows=0,
                first_ts="",
                last_ts="",
                duplicate_timestamps=0,
                bad_ohlc_rows=0,
                num_gaps_gt_1m=0,
                max_gap_minutes=0.0,
                reasons=f"download_command_failed:returncode={result.returncode}",
                file_name=final_name,
            )
            append_audit_row(audit_csv_path, fail)
            log(f"FAIL   {day} | returncode={result.returncode}")
            if stdout_text:
                log(f"STDOUT {day} | {stdout_text[:500]}")
            if stderr_text:
                log(f"STDERR {day} | {stderr_text[:500]}")
            continue

        after_files = list(temp_download_dir.glob("*.csv"))

        if len(after_files) == 0:
            fail = AuditResult(
                day=str(day),
                status="FAIL",
                rows=0,
                first_ts="",
                last_ts="",
                duplicate_timestamps=0,
                bad_ohlc_rows=0,
                num_gaps_gt_1m=0,
                max_gap_minutes=0.0,
                reasons="no_downloaded_csv_found",
                file_name=final_name,
            )
            append_audit_row(audit_csv_path, fail)
            log(f"FAIL   {day} | no CSV found in temp download dir")
            continue

        if len(after_files) > 1:
            log(f"WARN   {day} | multiple CSVs found in temp dir, using newest one")

        candidate = max(after_files, key=lambda x: x.stat().st_mtime)

        if candidate.stat().st_size == 0:
            fail = AuditResult(
                day=str(day),
                status="FAIL",
                rows=0,
                first_ts="",
                last_ts="",
                duplicate_timestamps=0,
                bad_ohlc_rows=0,
                num_gaps_gt_1m=0,
                max_gap_minutes=0.0,
                reasons="candidate_csv_0_bytes",
                file_name=final_name,
            )
            append_audit_row(audit_csv_path, fail)
            log(f"FAIL   {day} | candidate_csv_0_bytes")
            try:
                candidate.unlink()
            except Exception:
                pass
            continue

        if final_path.exists():
            final_path.unlink()

        shutil.move(str(candidate), str(final_path))

        if final_path.stat().st_size == 0:
            fail = AuditResult(
                day=str(day),
                status="FAIL",
                rows=0,
                first_ts="",
                last_ts="",
                duplicate_timestamps=0,
                bad_ohlc_rows=0,
                num_gaps_gt_1m=0,
                max_gap_minutes=0.0,
                reasons="empty_downloaded_csv_0_bytes",
                file_name=final_name,
            )
            append_audit_row(audit_csv_path, fail)
            log(f"FAIL   {day} | empty_downloaded_csv_0_bytes")
            final_path.unlink(missing_ok=True)
            continue

        try:
            audit = load_and_audit_csv(final_path, day)
        except Exception as e:
            fail = AuditResult(
                day=str(day),
                status="FAIL",
                rows=0,
                first_ts="",
                last_ts="",
                duplicate_timestamps=0,
                bad_ohlc_rows=0,
                num_gaps_gt_1m=0,
                max_gap_minutes=0.0,
                reasons=f"audit_exception:{type(e).__name__}",
                file_name=final_name,
            )
            append_audit_row(audit_csv_path, fail)
            log(f"FAIL   {day} | audit_exception:{type(e).__name__} | {e}")
            try:
                final_path.unlink()
            except Exception:
                pass
            continue

        append_audit_row(audit_csv_path, audit)

        log(
            f"{audit.status:<6} {day} | rows={audit.rows} | "
            f"first={audit.first_ts or 'NA'} | last={audit.last_ts or 'NA'} | "
            f"gaps>1m={audit.num_gaps_gt_1m} | max_gap_min={audit.max_gap_minutes:.1f} | "
            f"reasons={audit.reasons}"
        )        

    log("=" * 80)
    log("Finished daily Dukascopy download + validation.")
    log(f"Text log:  {text_log_path}")
    log(f"Audit CSV: {audit_csv_path}")
    log("=" * 80)


if __name__ == "__main__":
    main()