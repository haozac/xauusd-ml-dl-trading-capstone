from __future__ import annotations

import argparse
from pathlib import Path
import sys

from capstone_trading.runtime.order_execution_probe import CONFIRM_SEND_TOKEN, run_tiny_order_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 Step 2 v1.1 tiny controlled MT5 demo order open/close test with robust history audit. This DOES call order_send."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--terminal-path", default=None, help="Full path to terminal64.exe")
    parser.add_argument(
        "--confirm-send",
        required=True,
        help=f"Required exact token to send the tiny demo order: {CONFIRM_SEND_TOKEN}",
    )
    parser.add_argument(
        "--frozen-config",
        default="config/broker_execution_controls_frozen.yaml",
        help="Frozen broker execution controls YAML, relative to repo root",
    )
    parser.add_argument(
        "--stage3-step1-report",
        default="runtime/reports/stage3_step1_order_permission_preflight.json",
        help="Stage 3 Step 1 PASS report, relative to repo root",
    )
    parser.add_argument(
        "--side",
        default="BUY",
        choices=["BUY"],
        help="Open side for Stage 3 Step 2 v1.1. Only BUY is allowed because this is Model B plumbing.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="Position verification timeout")
    parser.add_argument("--poll-seconds", type=float, default=0.5, help="Position verification poll interval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    try:
        report = run_tiny_order_test(
            repo_root=repo_root,
            terminal_path=args.terminal_path,
            confirmation=args.confirm_send,
            frozen_config_path=Path(args.frozen_config),
            step1_report_path=Path(args.stage3_step1_report),
            side=args.side,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"ERROR Stage 3 Step 2 failed before report generation: {exc}", file=sys.stderr)
        return 1

    status = report.get("status")
    print(f"INFO Stage 3 Step 2 status: {status}")
    print(f"INFO JSON report: {repo_root / report['report_path']}")
    print(f"INFO Order-send events CSV: {repo_root / report['order_events_csv_path']}")
    print(f"INFO Position snapshots CSV: {repo_root / report['position_snapshots_csv_path']}")
    print(f"INFO History deals CSV: {repo_root / report['history_deals_csv_path']}")
    if report.get("history_orders_csv_path"):
        print(f"INFO History orders CSV: {repo_root / report['history_orders_csv_path']}")
    print(f"INFO Frozen config used: {repo_root / report['frozen_config_path']}")
    if report.get("warnings"):
        for warning in report["warnings"]:
            print(f"WARNING {warning}")
    if report.get("emergency_note"):
        print(f"EMERGENCY {report['emergency_note']}", file=sys.stderr)
    if status != "PASS":
        validations = report.get("validations", {}) if isinstance(report, dict) else {}
        failed = {key: value for key, value in validations.items() if value is False}
        if failed:
            print(f"ERROR Failed validations: {failed}", file=sys.stderr)
        if report.get("error"):
            print(f"ERROR Detail: {report['error']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
