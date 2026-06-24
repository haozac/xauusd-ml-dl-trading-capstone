from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_day(value: str) -> str:
    value = value.strip()
    formats = ("%Y-%m-%d", "%m/%d/%Y", "%#m/%#d/%Y", "%-m/%-d/%Y")

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass

    raise ValueError(f"Unsupported day format: {value!r}")


def read_review_days(audit_path: Path, status: str) -> list[str]:
    days = []
    seen = set()
    wanted_status = status.upper()

    with audit_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required_columns = {"day", "status"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Audit CSV is missing required column(s): {missing_text}")

        for row in reader:
            row_status = (row.get("status") or "").strip().upper()
            if row_status != wanted_status:
                continue

            iso_day = parse_day(row.get("day") or "")
            if iso_day not in seen:
                days.append(iso_day)
                seen.add(iso_day)

    return days


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_audit = project_root / "logs" / "dukascopy_xauusd_m1_daily_audit_1.csv"

    parser = argparse.ArgumentParser(
        description="Rerun Dukascopy downloader for days marked REVIEW in an audit CSV."
    )
    parser.add_argument(
        "--audit-csv",
        default=str(default_audit),
        help="Path to audit CSV. Default: logs/dukascopy_xauusd_m1_daily_audit_1.csv",
    )
    parser.add_argument(
        "--status",
        default="REVIEW",
        help="Status to rerun. Default: REVIEW",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    audit_path = Path(args.audit_csv)
    if not audit_path.is_absolute():
        audit_path = project_root / audit_path

    downloader = project_root / "scripts" / "download_validate_dukascopy_xauusd_m1_daily.py"
    days = read_review_days(audit_path, args.status)

    print(f"Audit CSV: {audit_path}")
    print(f"Status filter: {args.status.upper()}")
    print(f"Matched days: {len(days)}")

    if not days:
        print("No matching days found.")
        return

    for day in days:
        cmd = [
            sys.executable,
            str(downloader),
            "--start",
            day,
            "--end",
            day,
        ]

        print()
        print("Running:", " ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode != 0:
            raise SystemExit(f"Downloader failed for {day} with return code {result.returncode}")

    print()
    print("Finished rerunning matching audit days.")


if __name__ == "__main__":
    main()
