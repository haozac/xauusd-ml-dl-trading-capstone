#!/usr/bin/env python
"""Launch and supervise the two independent dual-rehearsal workers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from capstone_trading.runtime.dual_strategy_supervisor import (
    load_supervisor_settings,
    run_supervisor,
)

LOGGER = logging.getLogger("dual_strategy_supervisor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("shadow", "live"), default=None)
    orders = parser.add_mutually_exclusive_group(required=True)
    orders.add_argument("--orders-enabled", action="store_true")
    orders.add_argument("--orders-disabled", action="store_true")
    parser.add_argument("--confirm-live", default=None)
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--reset-control-files", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    repo_root = args.repo_root.expanduser().resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    try:
        settings = load_supervisor_settings(
            repo_root=repo_root,
            config_path=config_path,
            execution_mode_override=args.execution_mode,
            orders_enabled_override=bool(args.orders_enabled),
            duration_hours_override=args.duration_hours,
            confirmation=args.confirm_live,
        )
        report = run_supervisor(
            settings,
            reset_control_files=bool(args.reset_control_files),
        )
        LOGGER.info(
            "Dual supervisor status=%s orders_enabled=%s report=%s",
            report.get("status"),
            report.get("orders_enabled"),
            settings.paths.final_report,
        )
        return 0 if report.get("status") == "PASS" else 2
    except Exception:
        LOGGER.exception("Dual strategy supervisor failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
