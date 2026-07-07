from __future__ import annotations

import argparse
from pathlib import Path
import sys

from capstone_trading.runtime.order_preflight import run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 Step 1 MT5 order-permission and order_check preflight. No order_send is called."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--terminal-path", default=None, help="Full path to terminal64.exe")
    parser.add_argument(
        "--frozen-config",
        default="config/broker_execution_controls_frozen.yaml",
        help="Frozen broker execution controls YAML, relative to repo root",
    )
    parser.add_argument(
        "--sides",
        default="BUY,SELL",
        help="Comma-separated order_check sides. Default BUY,SELL. Use BUY only only if explicitly testing Model B only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    sides = tuple(side.strip().upper() for side in str(args.sides).split(",") if side.strip())
    try:
        report = run_preflight(
            repo_root=repo_root,
            terminal_path=args.terminal_path,
            frozen_config_path=Path(args.frozen_config),
            sides=sides,
        )
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"ERROR Stage 3 Step 1 failed: {exc}", file=sys.stderr)
        return 1

    status = report.get("status")
    print(f"INFO Stage 3 Step 1 status: {status}")
    print(f"INFO JSON report: {repo_root / report['report_path']}")
    print(f"INFO Order-check CSV: {repo_root / report['order_check_csv_path']}")
    print(f"INFO Frozen config used: {repo_root / report['frozen_config_path']}")
    if report.get("warnings"):
        for warning in report["warnings"]:
            print(f"WARNING {warning}")
    if status != "PASS":
        validations = report.get("validations", {}) if isinstance(report, dict) else {}
        failed = {key: value for key, value in validations.items() if value is False}
        if failed:
            print(f"ERROR Failed validations: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
