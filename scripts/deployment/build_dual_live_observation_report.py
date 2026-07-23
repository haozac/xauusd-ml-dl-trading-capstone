#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from capstone_trading.runtime.live_audit_analysis import build_observation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily and consolidated reports from raw dual-live audit files."
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--expected-poll-seconds",
        type=int,
        default=30,
        help="Expected telemetry polling cadence; default: 30 seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_observation_report(
        args.runtime_root.resolve(),
        args.output_root.resolve(),
        expected_poll_seconds=args.expected_poll_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
