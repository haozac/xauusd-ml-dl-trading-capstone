#!/usr/bin/env python
"""Stage 2 Step 2A v1.1 single-terminal MT5 shadow logger.

Read-only shadow mode: connects to one MT5 demo terminal, fetches completed M15
bars, rebuilds the frozen feature state, runs the frozen CNN-LSTM on the latest
valid 48-bar window, records Model A / Model B shadow signals, and exits or
watches.  v1.1 normalises Dukascopy MT5 server timestamps into canonical UTC.
It never sends orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dataclasses import replace

import pandas as pd

from capstone_trading.artifacts import verify_notebook7_artifact_bundle, verify_stage0_freeze_manifest
from capstone_trading.config import load_model_a_config, load_yaml_mapping, safe_repository_path
from capstone_trading.evaluation.model_b_replay import overlay_rules_from_model_b_config
from capstone_trading.evaluation.trading_replay import overlay_rules_from_config
from capstone_trading.model_loader import (
    check_runtime_environment,
    load_and_validate_model,
    load_and_validate_scaler,
    report_to_dict,
)
from capstone_trading.runtime.mt5_readiness import Mt5ReadinessError, import_metatrader5_module, rates_for_csv
from capstone_trading.runtime.mt5_shadow import (
    Mt5ShadowError,
    load_shadow_runtime_config,
    prepare_shadow_cpu_environment,
    run_shadow_once,
    run_watch_loop,
    shadow_signal_fieldnames,
    signal_to_csv_row,
    snapshot_to_dict,
    write_json_atomic,
)

LOGGER = logging.getLogger("stage2_step2a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-terminal MT5 shadow logger. This script reads completed M15 bars, "
            "runs frozen model inference and writes Model A / Model B shadow signals. It never sends orders."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument("--model-b-config", default="config/model_b_v2_frozen.yaml")
    parser.add_argument("--freeze-manifest", default="config/stage0_freeze_manifest.json")
    parser.add_argument("--runtime-config", default="config/mt5_shadow_runtime_template.yaml")
    parser.add_argument("--terminal-path", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one shadow snapshot and exit. Default mode.")
    mode.add_argument("--watch", action="store_true", help="Poll repeatedly and append only new shadow events.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--allow-onednn", action="store_true")
    parser.add_argument("--report", default="runtime/reports/stage2_step2a_v1_1_mt5_shadow_logger.json")
    parser.add_argument("--bars-csv", default="runtime/reports/stage2_step2a_v1_1_latest_completed_m15_bars.csv")
    parser.add_argument("--latest-signal-csv", default=None, help="Override latest-signal CSV path from runtime config.")
    parser.add_argument("--signals-csv", default=None, help="Override append-only shadow signal log path from runtime config.")
    parser.add_argument("--state", default=None, help="Override shadow state JSON path from runtime config.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def write_rows_csv_atomic(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    repository_root = args.repo_root.expanduser().resolve()
    report_path = safe_repository_path(repository_root, args.report, description="Stage 2 Step 2A report path", must_exist=False)
    bars_csv_path = safe_repository_path(repository_root, args.bars_csv, description="Stage 2 Step 2A bars CSV path", must_exist=False)

    report: dict[str, Any] = {
        "stage": 2,
        "step": "2A",
        "status": "RUNNING",
        "formal_gate": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "mt5_used": True,
        "orders_enabled": False,
        "shadow_only": True,
        "safety": {
            "manual_prerequisites": [
                "Open MT5 manually",
                "Log in to a demo account",
                "Confirm the terminal is connected",
                "Confirm XAUUSD is visible in Market Watch",
                "Keep orders disabled for shadow mode",
            ],
            "order_send_called": False,
            "trade_functions_allowed": False,
        },
    }

    try:
        prepare_shadow_cpu_environment(allow_onednn=args.allow_onednn)
        config_a_path = safe_repository_path(repository_root, args.model_a_config, description="Frozen Model A config")
        config_b_path = safe_repository_path(repository_root, args.model_b_config, description="Frozen Model B config")
        shadow_config_path = safe_repository_path(repository_root, args.runtime_config, description="MT5 shadow runtime config")
        config_a = load_model_a_config(config_a_path)
        config_b_raw = load_yaml_mapping(config_b_path)
        verify_stage0_freeze_manifest(repository_root, args.freeze_manifest)
        bundle = verify_notebook7_artifact_bundle(repository_root, config_a)
        environment = check_runtime_environment(config_a, strict=True)
        rules_a = overlay_rules_from_config(config_a.raw)
        rules_b = overlay_rules_from_model_b_config(config_b_raw)
        runtime_config = load_shadow_runtime_config(shadow_config_path)
        if args.terminal_path:
            runtime_config = replace(runtime_config, mt5=replace(runtime_config.mt5, terminal_path=args.terminal_path))
        state_path = safe_repository_path(
            repository_root,
            args.state or runtime_config.state_path,
            description="Stage 2 Step 2A shadow state path",
            must_exist=False,
        )
        signals_csv_path = safe_repository_path(
            repository_root,
            args.signals_csv or runtime_config.signals_csv_path,
            description="Stage 2 Step 2A append-only signal CSV path",
            must_exist=False,
        )
        latest_signal_csv_path = safe_repository_path(
            repository_root,
            args.latest_signal_csv or runtime_config.latest_signal_csv_path,
            description="Stage 2 Step 2A latest signal CSV path",
            must_exist=False,
        )
        scaler, scaler_report = load_and_validate_scaler(bundle.scaler_path, config_a, bundle.feature_order)
        model, model_report = load_and_validate_model(bundle.model_path, config_a)
        mt5_module = import_metatrader5_module()
        mode = "watch" if args.watch else "once"
        run_id = datetime.now(timezone.utc).strftime("stage2_step2a_%Y%m%dT%H%M%SZ")

        def run_once_callable():
            snapshot, rates, _signal = run_shadow_once(
                mt5_module=mt5_module,
                runtime_config=runtime_config,
                config_a=config_a,
                feature_order=bundle.feature_order,
                model=model,
                scaler=scaler,
                rules_a=rules_a,
                rules_b=rules_b,
                state_path=state_path,
                signals_csv_path=signals_csv_path,
                mode=mode,
                run_id=run_id,
            )
            snapshot_payload = snapshot_to_dict(snapshot)
            snapshot_payload["model_and_scaler"] = {
                "passed": True,
                "environment": report_to_dict(environment),
                "scaler": report_to_dict(scaler_report),
                "model": report_to_dict(model_report),
            }
            # Keep the most recent bar snapshot and most recent signal as easy-to-inspect files.
            rates_for_csv(rates).to_csv(bars_csv_path, index=False)
            latest_row = signal_to_csv_row(_signal, run_id=run_id, mode=mode)
            write_rows_csv_atomic(latest_signal_csv_path, [latest_row], shadow_signal_fieldnames())
            return snapshot_payload

        if mode == "watch":
            snapshots = run_watch_loop(
                run_once_callable=run_once_callable,
                poll_seconds=args.poll_seconds,
                max_iterations=args.max_iterations,
            )
            latest_snapshot = snapshots[-1]
        else:
            latest_snapshot = run_once_callable()
            snapshots = [latest_snapshot]

        latest_snapshot.update(
            {
                "status": "PASS",
                "formal_gate": True,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "repository_root": str(repository_root),
                "runtime_config": str(shadow_config_path.relative_to(repository_root)),
                "state_path": str(state_path.relative_to(repository_root)),
                "signals_csv": str(signals_csv_path.relative_to(repository_root)),
                "latest_signal_csv": str(latest_signal_csv_path.relative_to(repository_root)),
                "bars_csv": str(bars_csv_path.relative_to(repository_root)),
                "watch_iterations_executed": len(snapshots),
                "safety": report["safety"],
            }
        )
        if latest_snapshot.get("shutdown_called") is not True:
            raise Mt5ShadowError("MT5 shutdown was not confirmed after shadow snapshot")
        if latest_snapshot.get("forbidden_trade_function_calls"):
            raise Mt5ShadowError(
                f"Forbidden MT5 calls recorded: {latest_snapshot.get('forbidden_trade_function_calls')}"
            )
        if latest_snapshot.get("orders_enabled") is not False:
            raise Mt5ShadowError("Stage 2 Step 2A must have orders_enabled=false")
        write_json_atomic(report_path, latest_snapshot)
        LOGGER.info("Stage 2 Step 2A status: PASS")
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Bars CSV: %s", bars_csv_path)
        LOGGER.info("Latest signal CSV: %s", latest_signal_csv_path)
        LOGGER.info("Append-only signal CSV: %s", signals_csv_path)
        LOGGER.info("State JSON: %s", state_path)
        return 0

    except (Mt5ShadowError, Mt5ReadinessError, ValueError, KeyError) as exc:
        report["status"] = "FAIL"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write Stage 2 Step 2A failure report")
        if args.debug:
            LOGGER.exception("Stage 2 Step 2A failed")
        else:
            LOGGER.error("Stage 2 Step 2A failed: %s", exc)
        return 2
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected Stage 2 Step 2A failure report")
        LOGGER.exception("Unexpected Stage 2 Step 2A failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
