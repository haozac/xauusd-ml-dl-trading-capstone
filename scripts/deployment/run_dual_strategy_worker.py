#!/usr/bin/env python
"""Run one Model A or Model B rehearsal worker.

Normally launched by ``run_dual_strategy_rehearsal.py`` rather than manually.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from capstone_trading.runtime.dual_live_worker import (
    load_worker_settings,
    run_worker,
)

LOGGER = logging.getLogger("dual_strategy_worker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=("model_a", "model_b"), required=True)
    parser.add_argument("--execution-mode", choices=("shadow", "live"), default=None)
    orders = parser.add_mutually_exclusive_group(required=True)
    orders.add_argument("--orders-enabled", action="store_true")
    orders.add_argument("--orders-disabled", action="store_true")
    parser.add_argument("--confirm-live", default=None)
    parser.add_argument("--flatten-only", action="store_true")
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
        settings = load_worker_settings(
            repo_root=repo_root,
            config_path=config_path,
            role=args.role,
            execution_mode_override=args.execution_mode,
            orders_enabled_override=bool(args.orders_enabled),
            flatten_only=bool(args.flatten_only),
            confirmation=args.confirm_live,
        )
        report = run_worker(settings)
        LOGGER.info(
            "Worker role=%s status=%s records=%s orders_enabled=%s",
            args.role,
            report.get("status"),
            report.get("state", {}).get("records_written"),
            report.get("orders_enabled"),
        )
        return 0 if report.get("status") == "PASS" else 2
    except Exception:
        LOGGER.exception("Dual strategy worker failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
