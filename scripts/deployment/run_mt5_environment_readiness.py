#!/usr/bin/env python
"""Stage 2 Step 1 MT5 environment and data-feed readiness gate."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from capstone_trading.runtime.mt5_readiness import (
    Mt5ReadinessError,
    import_metatrader5_module,
    load_mt5_runtime_config,
    rates_for_csv,
    result_to_dict,
    run_mt5_readiness_check,
)

LOGGER = logging.getLogger("stage2_step1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check MT5 terminal/account/symbol/M15 data-feed readiness without placing orders. "
            "This is a Stage 2 shadow-mode precondition, not a trading script."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runtime-config",
        default="config/mt5_runtime_template.yaml",
        help="Broker/runtime-only config. Do not put passwords or model parameters here.",
    )
    parser.add_argument(
        "--report",
        default="runtime/reports/stage2_step1_mt5_readiness.json",
    )
    parser.add_argument(
        "--bars-csv",
        default="runtime/reports/stage2_step1_mt5_recent_m15_completed_bars.csv",
    )
    parser.add_argument(
        "--symbol-candidates-csv",
        default="runtime/reports/stage2_step1_mt5_symbol_candidates.csv",
    )
    parser.add_argument(
        "--terminal-path",
        default=None,
        help="Optional terminal64.exe path override. Prefer using this over editing tracked files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    return parser.parse_args()


def safe_repository_path(
    repository_root: Path,
    raw_path: str | Path,
    *,
    description: str,
    must_exist: bool = False,
) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    resolved = candidate.expanduser().resolve()
    root = repository_root.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Mt5ReadinessError(f"{description} must stay inside repository root: {resolved}") from exc
    if must_exist and not resolved.exists():
        raise Mt5ReadinessError(f"{description} does not exist: {resolved}")
    return resolved


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_rows_csv_atomic(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    repository_root = args.repo_root.expanduser().resolve()
    report_path = safe_repository_path(
        repository_root,
        args.report,
        description="Stage 2 Step 1 report path",
        must_exist=False,
    )
    bars_csv_path = safe_repository_path(
        repository_root,
        args.bars_csv,
        description="Stage 2 Step 1 bars CSV path",
        must_exist=False,
    )
    symbol_candidates_csv_path = safe_repository_path(
        repository_root,
        args.symbol_candidates_csv,
        description="Stage 2 Step 1 symbol candidates CSV path",
        must_exist=False,
    )
    report: dict[str, Any] = {
        "stage": 2,
        "step": 1,
        "status": "RUNNING",
        "formal_gate": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "mt5_used": True,
        "orders_enabled": False,
        "safety": {
            "purpose": "environment_and_data_feed_readiness_only",
            "trade_functions_allowed": False,
            "order_send_called": False,
        },
    }
    try:
        config_path = safe_repository_path(
            repository_root,
            args.runtime_config,
            description="MT5 runtime config",
            must_exist=True,
        )
        config = load_mt5_runtime_config(config_path)
        if args.terminal_path:
            from dataclasses import replace

            config = replace(config, terminal_path=args.terminal_path)
        mt5_module = import_metatrader5_module()
        result, bars = run_mt5_readiness_check(
            mt5_module=mt5_module,
            config=config,
        )
        payload = result_to_dict(result)
        payload.update(
            {
                "repository_root": str(repository_root),
                "runtime_config": str(config_path.relative_to(repository_root)),
                "bars_csv": str(bars_csv_path.relative_to(repository_root)),
                "symbol_candidates_csv": str(symbol_candidates_csv_path.relative_to(repository_root)),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Fail defensively if shutdown failed or a trade function access was recorded.
        if payload.get("shutdown_called") is not True:
            raise Mt5ReadinessError("MT5 shutdown was not confirmed after readiness check")
        if payload.get("forbidden_trade_function_calls"):
            raise Mt5ReadinessError(
                f"Forbidden MT5 calls recorded: {payload.get('forbidden_trade_function_calls')}"
            )
        write_json_atomic(report_path, payload)
        rates_for_csv(bars).to_csv(bars_csv_path, index=False)
        candidate_rows = [
            {
                "candidate_order": index + 1,
                "symbol_candidate": item,
                "selected": item == payload["symbol_resolution"]["selected_symbol"],
            }
            for index, item in enumerate(payload["symbol_resolution"]["candidates_checked"])
        ]
        write_rows_csv_atomic(
            symbol_candidates_csv_path,
            candidate_rows,
            ["candidate_order", "symbol_candidate", "selected"],
        )
        LOGGER.info("Stage 2 Step 1 status: PASS")
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Bars CSV: %s", bars_csv_path)
        LOGGER.info("Symbol candidates CSV: %s", symbol_candidates_csv_path)
        return 0
    except (Mt5ReadinessError, ValueError, KeyError) as exc:
        report["status"] = "FAIL"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write Stage 2 Step 1 failure report")
        if args.debug:
            LOGGER.exception("Stage 2 Step 1 failed")
        else:
            LOGGER.error("Stage 2 Step 1 failed: %s", exc)
        return 2
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected Stage 2 Step 1 failure report")
        LOGGER.exception("Unexpected Stage 2 Step 1 failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
