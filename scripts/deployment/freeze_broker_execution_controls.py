from __future__ import annotations

import argparse
from pathlib import Path
import sys

from capstone_trading.runtime.broker_execution_controls import run_freeze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 Step 3A no-order broker execution control freeze")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument(
        "--step2a-report",
        default="runtime/reports/stage2_step2a_v1_1_mt5_shadow_logger.json",
        help="Path to the validated Stage 2 Step 2A v1.1 shadow report, relative to repo root",
    )
    parser.add_argument(
        "--account-equity-sgd",
        type=float,
        default=None,
        help="Optional current demo account equity in SGD for 0.01 lot leverage review. Example: 220",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    try:
        report = run_freeze(
            repo_root=repo_root,
            step2a_report_path=Path(args.step2a_report),
            account_equity_sgd=args.account_equity_sgd,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"ERROR Stage 2 Step 3A failed: {exc}", file=sys.stderr)
        return 1

    status = report.get("status")
    print(f"INFO Stage 2 Step 3A status: {status}")
    print(f"INFO JSON report: {repo_root / report['report_path']}")
    print(f"INFO Summary CSV: {repo_root / report['summary_csv_path']}")
    print(f"INFO Frozen config: {repo_root / report['frozen_config_path']}")

    capital = report.get("checks", {}).get("capital_review", {})
    if capital.get("capstone_leverage_cap_passed_if_equity_supplied") is False:
        print("WARNING 0.01 lot appears to breach the 10:1 capstone leverage cap for the supplied equity.")
    if report.get("warnings"):
        for warning in report["warnings"]:
            print(f"WARNING {warning}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
