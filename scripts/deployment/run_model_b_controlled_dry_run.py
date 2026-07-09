#!/usr/bin/env python
"""Stage 3 Step 3A Model B controlled execution dry-run.

This script runs the frozen live MT5 shadow inference and converts the latest
completed M15 signal into a Model B current execution intent.  It may call
mt5.order_check for a would-enter-long request, but it never calls order_send.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from capstone_trading.artifacts import verify_notebook7_artifact_bundle, verify_stage0_freeze_manifest
from capstone_trading.config import load_model_a_config, load_yaml_mapping, safe_repository_path
from capstone_trading.evaluation.model_b_replay import overlay_rules_from_model_b_config
from capstone_trading.evaluation.trading_replay import overlay_rules_from_config
from capstone_trading.model_loader import check_runtime_environment, load_and_validate_model, load_and_validate_scaler, report_to_dict
from capstone_trading.runtime.model_b_controlled_dry_run import (
    DEFAULT_INTENTS_CSV_PATH,
    DEFAULT_LATEST_DECISION_CSV_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_STAGE3_STEP2_REPORT_PATH,
    DEFAULT_STATE_PATH,
    ModelBDryRunRules,
    Stage3Step3ADryRunError,
    append_csv_row,
    decision_fieldnames,
    decision_to_row,
    load_dry_run_state,
    load_required_stage3_step2_report,
    reset_dry_run_state,
    run_dry_run_iteration,
    summarise_decisions,
    write_csv_rows_atomic,
    write_dry_run_state,
    write_json_atomic,
)
from capstone_trading.runtime.mt5_readiness import Mt5ReadinessError, import_metatrader5_module
from capstone_trading.runtime.mt5_shadow import (
    Mt5ShadowError,
    load_shadow_runtime_config,
    prepare_shadow_cpu_environment,
    run_shadow_once,
    snapshot_to_dict,
)
from capstone_trading.runtime.order_preflight import load_frozen_controls

LOGGER = logging.getLogger("stage3_step3a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 3 Step 3A Model B controlled execution dry-run. "
            "This script NEVER sends orders."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--terminal-path", default=None)
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument("--model-b-config", default="config/model_b_v2_frozen.yaml")
    parser.add_argument("--freeze-manifest", default="config/stage0_freeze_manifest.json")
    parser.add_argument("--shadow-runtime-config", default="config/mt5_shadow_runtime_template.yaml")
    parser.add_argument("--broker-controls", default="config/broker_execution_controls_frozen.yaml")
    parser.add_argument("--stage3-step2-report", default=str(DEFAULT_STAGE3_STEP2_REPORT_PATH))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one dry-run decision and exit.")
    mode.add_argument("--watch", action="store_true", help="Poll repeatedly. Recommended for this gate.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=240)
    parser.add_argument("--min-new-events", type=int, default=4, help="Minimum unique completed M15 events required for formal PASS.")
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--reset-state", action="store_true", help="Start a fresh virtual Model B dry-run state.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--intents-csv", default=str(DEFAULT_INTENTS_CSV_PATH))
    parser.add_argument("--latest-decision-csv", default=str(DEFAULT_LATEST_DECISION_CSV_PATH))
    parser.add_argument("--disable-order-check", action="store_true", help="Do not call order_check for would-enter intents.")
    parser.add_argument("--allow-onednn", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s %(message)s")
    repo_root = args.repo_root.expanduser().resolve()
    report_path = safe_repository_path(repo_root, args.report, description="Stage 3 Step 3A report", must_exist=False)
    intents_csv_path = safe_repository_path(repo_root, args.intents_csv, description="Stage 3 Step 3A intents CSV", must_exist=False)
    latest_decision_csv_path = safe_repository_path(repo_root, args.latest_decision_csv, description="Stage 3 Step 3A latest decision CSV", must_exist=False)
    state_path = safe_repository_path(repo_root, args.state, description="Stage 3 Step 3A state", must_exist=False)

    mode = "watch" if args.watch else "once"
    run_id = datetime.now(timezone.utc).strftime("stage3_step3a_%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "stage": 3,
        "step": "3A",
        "status": "RUNNING",
        "formal_gate": False,
        "purpose": "model_b_current_controlled_execution_dry_run_no_order_send",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo_root),
        "run_id": run_id,
        "mode": mode,
        "orders_enabled": False,
        "order_send_allowed": False,
        "order_send_called": False,
        "manual_prerequisites": [
            "Open MT5 manually",
            "Log in to the Dukascopy demo account",
            "Enable Algo Trading so order_check can be evaluated",
            "Confirm no open XAUUSD position and no pending XAUUSD order",
            "Keep this as dry-run only; do not manually trade during the run",
        ],
    }

    try:
        if args.poll_seconds < 5:
            raise Stage3Step3ADryRunError("Use poll_seconds >= 5 to avoid excessive MT5 polling")
        if args.max_iterations < 1:
            raise Stage3Step3ADryRunError("max_iterations must be at least 1")
        if args.min_new_events < 0:
            raise Stage3Step3ADryRunError("min-new-events cannot be negative")

        prepare_shadow_cpu_environment(allow_onednn=args.allow_onednn)
        load_required_stage3_step2_report(repo_root, Path(args.stage3_step2_report))
        if args.reset_state:
            reset_dry_run_state(state_path)

        config_a_path = safe_repository_path(repo_root, args.model_a_config, description="Frozen Model A config")
        config_b_path = safe_repository_path(repo_root, args.model_b_config, description="Frozen Model B config")
        shadow_config_path = safe_repository_path(repo_root, args.shadow_runtime_config, description="MT5 shadow runtime config")
        controls_path = safe_repository_path(repo_root, args.broker_controls, description="Frozen broker execution controls")

        config_a = load_model_a_config(config_a_path)
        config_b_raw = load_yaml_mapping(config_b_path)
        controls = load_frozen_controls(controls_path)
        verify_stage0_freeze_manifest(repo_root, args.freeze_manifest)
        bundle = verify_notebook7_artifact_bundle(repo_root, config_a)
        environment = check_runtime_environment(config_a, strict=True)
        rules_a = overlay_rules_from_config(config_a.raw)
        rules_b_raw = overlay_rules_from_model_b_config(config_b_raw)
        rules_b = ModelBDryRunRules(
            variant="MODEL_B_V2_CURRENT",
            entry_threshold=float(rules_b_raw.entry_threshold),
            exit_threshold=float(rules_b_raw.exit_threshold),
            long_only=True,
            max_successful_entries_per_utc_day=1,
            min_hold_bars=0,
        )
        if rules_b.entry_threshold != 0.55 or rules_b.exit_threshold != 0.50:
            raise Stage3Step3ADryRunError(f"Unexpected Model B thresholds: {rules_b}")

        runtime_config = load_shadow_runtime_config(shadow_config_path)
        runtime_config = replace(runtime_config, append_signals=False, duplicate_policy="skip")
        if args.terminal_path:
            runtime_config = replace(runtime_config, mt5=replace(runtime_config.mt5, terminal_path=args.terminal_path))

        scaler, scaler_report = load_and_validate_scaler(bundle.scaler_path, config_a, bundle.feature_order)
        model, model_report = load_and_validate_model(bundle.model_path, config_a)
        mt5_module = import_metatrader5_module()
        state = load_dry_run_state(state_path)
        decisions = []
        latest_snapshot: Mapping[str, Any] | None = None
        latest_context: Mapping[str, Any] | None = None
        iterations = args.max_iterations if mode == "watch" else 1

        def shadow_once_callable() -> Mapping[str, Any]:
            snapshot, _rates, _signal = run_shadow_once(
                mt5_module=mt5_module,
                runtime_config=runtime_config,
                config_a=config_a,
                feature_order=bundle.feature_order,
                model=model,
                scaler=scaler,
                rules_a=rules_a,
                rules_b=rules_b_raw,
                state_path=state_path.with_name("stage3_step3a_shadow_unused_state.json"),
                signals_csv_path=intents_csv_path.with_name("stage3_step3a_shadow_unused_signals.csv"),
                mode=mode,
                run_id=run_id,
            )
            payload = snapshot_to_dict(snapshot)
            payload["model_and_scaler"] = {
                "passed": True,
                "environment": report_to_dict(environment),
                "scaler": report_to_dict(scaler_report),
                "model": report_to_dict(model_report),
            }
            return payload

        for idx in range(1, iterations + 1):
            decision, state, context = run_dry_run_iteration(
                mt5_module=mt5_module,
                terminal_path=args.terminal_path,
                controls=controls,
                shadow_once_callable=shadow_once_callable,
                state=state,
                rules=rules_b,
                run_id=run_id,
                iteration=idx,
                mode=mode,
                order_check_enabled=not args.disable_order_check,
            )
            decisions.append(decision)
            latest_snapshot = context.get("shadow_snapshot") if isinstance(context, Mapping) else None
            latest_context = context.get("broker_context") if isinstance(context, Mapping) else context
            write_dry_run_state(state_path, state)
            row = decision_to_row(decision)
            append_csv_row(intents_csv_path, row, decision_fieldnames())
            write_csv_rows_atomic(latest_decision_csv_path, [row], decision_fieldnames())
            LOGGER.info(
                "Iteration %s/%s event=%s p=%s action=%s reason=%s order_check=%s",
                idx,
                iterations,
                decision.event_time_utc,
                None if decision.probability_up is None else round(decision.probability_up, 6),
                decision.action,
                decision.reason,
                decision.order_check_passed,
            )
            if idx < iterations:
                time.sleep(args.poll_seconds)

        summary = summarise_decisions(decisions)
        hard_fail_actions = [
            d.action for d in decisions
            if d.action in {"BLOCK_ACTUAL_POSITION_EXISTS", "BLOCK_PENDING_ORDER_EXISTS", "BLOCK_INVALID_SIGNAL", "BLOCK_INVALID_STATE"}
        ]
        formal_gate = (
            summary["order_send_called_count"] == 0
            and not hard_fail_actions
            and summary["unique_completed_m15_events"] >= int(args.min_new_events)
        )
        report.update(
            {
                "status": "PASS" if formal_gate else "FAIL",
                "formal_gate": formal_gate,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "iterations_executed": len(decisions),
                "poll_seconds": args.poll_seconds,
                "min_new_events_required": args.min_new_events,
                "rules": rules_b.__dict__,
                "summary": summary,
                "latest_decision": decision_to_row(decisions[-1]) if decisions else None,
                "latest_shadow_snapshot": latest_snapshot,
                "latest_broker_context": latest_context,
                "state_path": str(state_path.relative_to(repo_root)),
                "intents_csv": str(intents_csv_path.relative_to(repo_root)),
                "latest_decision_csv": str(latest_decision_csv_path.relative_to(repo_root)),
                "model_and_scaler": {
                    "environment": report_to_dict(environment),
                    "scaler": report_to_dict(scaler_report),
                    "model": report_to_dict(model_report),
                },
                "safety": {
                    "order_send_called": summary["order_send_called_count"] > 0,
                    "order_send_allowed": False,
                    "dry_run_only": True,
                    "stage3_step3b_single_model_execution_allowed": formal_gate,
                },
                "decision": {
                    "stage3_step3a_dry_run_passed": formal_gate,
                    "next_step_if_pass": "Stage 3 Step 3B - Model B controlled live execution with order_send, after this dry-run report is reviewed.",
                    "do_not_start_unattended_final_run_yet": True,
                },
            }
        )
        if not formal_gate:
            report["failure_reason"] = (
                "Dry-run did not meet the formal gate. Usually this means too few new M15 events were observed, "
                "or a hard safety block occurred. Review summary/action_counts."
            )
        write_json_atomic(report_path, report)
        LOGGER.info("Stage 3 Step 3A status: %s", report["status"])
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Intent log CSV: %s", intents_csv_path)
        LOGGER.info("Latest decision CSV: %s", latest_decision_csv_path)
        LOGGER.info("State JSON: %s", state_path)
        return 0 if formal_gate else 2

    except (Stage3Step3ADryRunError, Mt5ShadowError, Mt5ReadinessError, ValueError, KeyError) as exc:
        report.update(
            {
                "status": "FAIL",
                "formal_gate": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write Stage 3 Step 3A failure report")
        if args.debug:
            LOGGER.exception("Stage 3 Step 3A failed")
        else:
            LOGGER.error("Stage 3 Step 3A failed: %s", exc)
        return 2
    except Exception as exc:
        report.update(
            {
                "status": "FAIL_UNEXPECTED",
                "formal_gate": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected Stage 3 Step 3A failure report")
        LOGGER.exception("Unexpected Stage 3 Step 3A failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
