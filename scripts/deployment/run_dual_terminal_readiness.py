#!/usr/bin/env python
"""Stage 3 Step 4A dual-terminal and dual-account no-order readiness gate."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from capstone_trading.runtime.dual_terminal_readiness import (
    DualTerminalReadinessError,
    build_dual_terminal_report,
    load_dual_terminal_config,
    summary_rows,
    terminal_inventory_rows,
)
from capstone_trading.runtime.mt5_readiness import import_metatrader5_module

LOGGER = logging.getLogger("stage3_step4a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify two independent MT5 installations and demo accounts without order_check/order_send."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="config/dual_terminal_runtime.yaml")
    parser.add_argument("--model-a-terminal-path", default=None)
    parser.add_argument("--model-b-terminal-path", default=None)
    parser.add_argument("--server-time-offset-hours", type=int, choices=(2, 3), default=None)
    parser.add_argument("--report", default="runtime/reports/stage3_step4a_dual_terminal_readiness.json")
    parser.add_argument("--summary-csv", default="runtime/reports/stage3_step4a_dual_terminal_readiness_summary.csv")
    parser.add_argument("--inventory-csv", default="runtime/reports/stage3_step4a_terminal_inventory.csv")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def safe_repo_path(root: Path, raw: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DualTerminalReadinessError(f"Output/config path must stay inside repo: {resolved}") from exc
    if must_exist and not resolved.exists():
        raise DualTerminalReadinessError(f"Required file does not exist: {resolved}")
    return resolved


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0].keys()) if rows else ["scope", "check", "value"]
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s %(message)s")
    root = args.repo_root.expanduser().resolve()
    report_path = safe_repo_path(root, args.report)
    summary_path = safe_repo_path(root, args.summary_csv)
    inventory_path = safe_repo_path(root, args.inventory_csv)
    try:
        config_path = safe_repo_path(root, args.config, must_exist=True)
        config = load_dual_terminal_config(config_path)
        if args.model_a_terminal_path:
            config = replace(config, model_a=replace(config.model_a, terminal_path=args.model_a_terminal_path))
        if args.model_b_terminal_path:
            config = replace(config, model_b=replace(config.model_b, terminal_path=args.model_b_terminal_path))
        if args.server_time_offset_hours is not None:
            config = replace(config, mt5_server_time_offset_hours=args.server_time_offset_hours)
        mt5_module = import_metatrader5_module()
        report = build_dual_terminal_report(mt5_module=mt5_module, config=config)
        report["repository_root"] = str(root)
        report["config_path"] = str(config_path.relative_to(root))
        write_json_atomic(report_path, report)
        write_csv_atomic(summary_path, summary_rows(report))
        write_csv_atomic(inventory_path, terminal_inventory_rows(report))
        LOGGER.info("Stage 3 Step 4A status: %s", report["status"])
        LOGGER.info("Model A account: %s", report["cross_terminal_review"]["model_a_login_masked"])
        LOGGER.info("Model B account: %s", report["cross_terminal_review"]["model_b_login_masked"])
        LOGGER.info("Terminal paths distinct: %s", report["cross_terminal_review"]["checks"]["terminal_paths_distinct"])
        LOGGER.info("Data paths distinct: %s", report["cross_terminal_review"]["checks"]["terminal_data_paths_distinct"])
        LOGGER.info("Accounts distinct: %s", report["cross_terminal_review"]["checks"]["accounts_distinct"])
        LOGGER.info("Latest M15 difference minutes: %.1f", report["cross_terminal_review"]["latest_completed_bar_difference_minutes"])
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Summary CSV: %s", summary_path)
        LOGGER.info("Inventory CSV: %s", inventory_path)
        return 0 if report.get("formal_gate") is True else 2
    except Exception as exc:
        failure = {
            "stage": 3,
            "step": "4A",
            "status": "FAIL",
            "formal_gate": False,
            "purpose": "dual_account_dual_terminal_readiness_no_order",
            "order_check_called": False,
            "order_send_called": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            write_json_atomic(report_path, failure)
        except Exception:
            LOGGER.exception("Could not write Step 4A failure report")
        if args.debug:
            LOGGER.exception("Stage 3 Step 4A failed")
        else:
            LOGGER.error("Stage 3 Step 4A failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
