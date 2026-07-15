#!/usr/bin/env python
"""Stage 3 Step 3B Model B controlled live execution.

This script may send real demo orders through MT5, but only under the frozen
Model B current rules and only after the explicit confirmation token is passed.
It is a monitored controlled gate, not the final unattended paper-trading run.
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
from capstone_trading.runtime.model_b_controlled_live import (
    CONFIRM_SEND_TOKEN,
    DEFAULT_EVENTS_CSV_PATH,
    DEFAULT_HISTORY_DEALS_CSV_PATH,
    DEFAULT_HISTORY_ORDERS_CSV_PATH,
    DEFAULT_LATEST_DECISION_CSV_PATH,
    DEFAULT_ORDER_EVENTS_CSV_PATH,
    DEFAULT_POSITION_SNAPSHOTS_CSV_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_STAGE3_STEP2_REPORT_PATH,
    DEFAULT_STAGE3_STEP3A_REPORT_PATH,
    DEFAULT_STATE_PATH,
    ModelBLiveState,
    Stage3Step3BLiveError,
    append_csv_row,
    decision_fieldnames,
    force_close_model_b_position,
    load_live_state,
    load_required_stage3_step2_report,
    load_required_stage3_step3a_report,
    reset_live_state,
    run_live_iteration,
    summarise_live_decisions,
    write_csv_rows_atomic,
    write_json_atomic,
    write_live_state,
)
from capstone_trading.runtime.model_b_controlled_dry_run import ModelBDryRunRules
from capstone_trading.runtime.mt5_readiness import import_metatrader5_module
from capstone_trading.runtime.mt5_shadow import (
    load_shadow_runtime_config,
    prepare_shadow_cpu_environment,
    run_shadow_once,
    snapshot_to_dict,
)
from capstone_trading.runtime.order_preflight import load_frozen_controls

LOGGER = logging.getLogger("stage3_step3b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 3 Step 3B Model B controlled live execution. "
            "This script MAY call order_send on a demo account."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--terminal-path", default=None)
    parser.add_argument("--confirm-send", required=True, help=f"Must equal {CONFIRM_SEND_TOKEN}")
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument("--model-b-config", default="config/model_b_v2_frozen.yaml")
    parser.add_argument("--freeze-manifest", default="config/stage0_freeze_manifest.json")
    parser.add_argument("--shadow-runtime-config", default="config/mt5_shadow_runtime_template.yaml")
    parser.add_argument("--broker-controls", default="config/broker_execution_controls_frozen.yaml")
    parser.add_argument("--stage3-step2-report", default=str(DEFAULT_STAGE3_STEP2_REPORT_PATH))
    parser.add_argument("--stage3-step3a-report", default=str(DEFAULT_STAGE3_STEP3A_REPORT_PATH))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one controlled live decision and exit.")
    mode.add_argument("--watch", action="store_true", help="Poll repeatedly. Recommended for this gate.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=480, help="480 at 30 seconds is about 4 hours.")
    parser.add_argument("--min-new-events", type=int, default=4, help="Minimum unique completed M15 events required for formal PASS.")
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--reset-state", action="store_true", help="Start a fresh Model B live state. Use only when no XAUUSD position exists.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--events-csv", default=str(DEFAULT_EVENTS_CSV_PATH))
    parser.add_argument("--order-events-csv", default=str(DEFAULT_ORDER_EVENTS_CSV_PATH))
    parser.add_argument("--position-snapshots-csv", default=str(DEFAULT_POSITION_SNAPSHOTS_CSV_PATH))
    parser.add_argument("--history-deals-csv", default=str(DEFAULT_HISTORY_DEALS_CSV_PATH))
    parser.add_argument("--history-orders-csv", default=str(DEFAULT_HISTORY_ORDERS_CSV_PATH))
    parser.add_argument("--latest-decision-csv", default=str(DEFAULT_LATEST_DECISION_CSV_PATH))
    parser.add_argument("--close-at-end", action="store_true", default=True, help="Safety-close any remaining Model B position when the controlled run ends. Default enabled.")
    parser.add_argument("--leave-open-at-end", action="store_true", help="Do not force-close at end. Not recommended for this gate.")
    parser.add_argument("--stop-after-entry-exit-cycle", action="store_true", default=True, help="Stop once one entry and one exit/forced close have completed. Default enabled.")
    parser.add_argument("--no-stop-after-entry-exit-cycle", action="store_true", help="Continue even after one completed cycle.")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--order-poll-seconds", type=float, default=0.5)
    parser.add_argument("--allow-onednn", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _event_time_from_snapshot(snapshot: Mapping[str, Any]) -> str | None:
    signal = snapshot.get("latest_signal", {})
    if not isinstance(signal, Mapping):
        return None
    value = signal.get("event_time_utc")
    return None if value is None else str(value)


def _safe_copy_context_list(contexts: list[Mapping[str, Any]], key: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for context in contexts:
        value = context.get(key)
        if isinstance(value, list):
            rows.extend([item for item in value if isinstance(item, Mapping)])
    return rows


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s %(message)s")
    repo_root = args.repo_root.expanduser().resolve()
    if args.confirm_send != CONFIRM_SEND_TOKEN:
        LOGGER.error("Refusing to run. Pass --confirm-send %s", CONFIRM_SEND_TOKEN)
        return 2
    if args.poll_seconds < 5:
        LOGGER.error("Use --poll-seconds >= 5")
        return 2
    if args.max_iterations < 1:
        LOGGER.error("--max-iterations must be at least 1")
        return 2
    if args.leave_open_at_end:
        close_at_end = False
    else:
        close_at_end = bool(args.close_at_end)
    stop_after_cycle = bool(args.stop_after_entry_exit_cycle and not args.no_stop_after_entry_exit_cycle)

    report_path = safe_repository_path(repo_root, args.report, description="Stage 3 Step 3B report", must_exist=False)
    events_csv_path = safe_repository_path(repo_root, args.events_csv, description="Stage 3 Step 3B events CSV", must_exist=False)
    order_events_csv_path = safe_repository_path(repo_root, args.order_events_csv, description="Stage 3 Step 3B order events CSV", must_exist=False)
    position_snapshots_csv_path = safe_repository_path(repo_root, args.position_snapshots_csv, description="Stage 3 Step 3B position snapshots CSV", must_exist=False)
    history_deals_csv_path = safe_repository_path(repo_root, args.history_deals_csv, description="Stage 3 Step 3B history deals CSV", must_exist=False)
    history_orders_csv_path = safe_repository_path(repo_root, args.history_orders_csv, description="Stage 3 Step 3B history orders CSV", must_exist=False)
    latest_decision_csv_path = safe_repository_path(repo_root, args.latest_decision_csv, description="Stage 3 Step 3B latest decision CSV", must_exist=False)
    state_path = safe_repository_path(repo_root, args.state, description="Stage 3 Step 3B state", must_exist=False)

    mode = "watch" if args.watch else "once"
    run_id = datetime.now(timezone.utc).strftime("stage3_step3b_%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "stage": 3,
        "step": "3B",
        "status": "RUNNING",
        "formal_gate": False,
        "purpose": "model_b_current_controlled_live_execution_with_order_send",
        "patch_version": "stage3_step3b_v1_1",
        "spread_policy": {
            "entry_gate_points": 800,
            "wide_spread_while_flat": "block_new_entry_and_continue_monitoring",
            "wide_spread_while_long": "do_not_block_hold_or_risk_reducing_exit",
        },
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo_root),
        "run_id": run_id,
        "mode": mode,
        "orders_enabled": True,
        "order_send_allowed": True,
        "close_at_end": close_at_end,
        "stop_after_entry_exit_cycle": stop_after_cycle,
        "manual_prerequisites": [
            "Open MT5 manually",
            "Log in to the Dukascopy demo account",
            "Enable Algo Trading",
            "Confirm no open XAUUSD position and no pending XAUUSD order before using --reset-state",
            "Monitor the terminal while this controlled gate is running",
            "Do not manually trade XAUUSD during the run",
        ],
    }

    decisions = []
    contexts: list[Mapping[str, Any]] = []
    latest_snapshot: Mapping[str, Any] | None = None
    state = ModelBLiveState()
    mt5_module: Any | None = None

    try:
        prepare_shadow_cpu_environment(allow_onednn=args.allow_onednn)
        load_required_stage3_step2_report(repo_root, Path(args.stage3_step2_report))
        load_required_stage3_step3a_report(repo_root, Path(args.stage3_step3a_report))
        if args.reset_state:
            reset_live_state(state_path)

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
            raise Stage3Step3BLiveError(f"Unexpected Model B thresholds: {rules_b}")
        if float(controls.stage3_first_order_test_volume_lots) != 0.01:
            raise Stage3Step3BLiveError("Stage 3 Step 3B requires frozen 0.01 lot volume")

        runtime_config = load_shadow_runtime_config(shadow_config_path)
        runtime_config = replace(runtime_config, append_signals=False, duplicate_policy="skip")
        if args.terminal_path:
            runtime_config = replace(runtime_config, mt5=replace(runtime_config.mt5, terminal_path=args.terminal_path))

        scaler, scaler_report = load_and_validate_scaler(bundle.scaler_path, config_a, bundle.feature_order)
        model, model_report = load_and_validate_model(bundle.model_path, config_a)
        mt5_module = import_metatrader5_module()
        state = load_live_state(state_path)
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
                state_path=state_path.with_name("stage3_step3b_shadow_unused_state.json"),
                signals_csv_path=events_csv_path.with_name("stage3_step3b_shadow_unused_signals.csv"),
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
            latest_snapshot = shadow_once_callable()
            signal = latest_snapshot.get("latest_signal", {})
            if not isinstance(signal, Mapping):
                raise Stage3Step3BLiveError("Shadow snapshot did not include latest_signal")

            decision, state, context = run_live_iteration(
                mt5_module=mt5_module,
                terminal_path=args.terminal_path,
                controls=controls,
                signal=signal,
                state=state,
                rules=rules_b,
                run_id=run_id,
                iteration=idx,
                mode=mode,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.order_poll_seconds,
            )
            decisions.append(decision)
            contexts.append(context)
            write_live_state(state_path, state)
            append_csv_row(events_csv_path, decision.__dict__, decision_fieldnames())
            write_csv_rows_atomic(latest_decision_csv_path, [decision.__dict__], decision_fieldnames())
            LOGGER.info(
                "Iteration %s/%s event=%s p=%s action=%s reason=%s order_send=%s retcode=%s position=%s",
                idx,
                iterations,
                decision.event_time_utc,
                None if decision.probability_up is None else f"{decision.probability_up:.6f}",
                decision.action,
                decision.reason,
                decision.order_send_called,
                decision.order_send_retcode,
                state.live_position_name,
            )
            if stop_after_cycle and state.completed_entry_exit_cycles >= 1:
                LOGGER.info("Stopping after one completed entry-exit cycle.")
                break
            if mode == "watch" and idx < iterations:
                time.sleep(args.poll_seconds)

        force_close_context: Mapping[str, Any] = {}
        if close_at_end and state.live_position == 1 and mt5_module is not None:
            LOGGER.warning("End of controlled run reached with Model B position open. Sending safety close.")
            close_decision, state, force_close_context = force_close_model_b_position(
                mt5_module=mt5_module,
                terminal_path=args.terminal_path,
                controls=controls,
                state=state,
                run_id=run_id,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.order_poll_seconds,
            )
            if close_decision is not None:
                decisions.append(close_decision)
                contexts.append(force_close_context)
                write_live_state(state_path, state)
                append_csv_row(events_csv_path, close_decision.__dict__, decision_fieldnames())
                write_csv_rows_atomic(latest_decision_csv_path, [close_decision.__dict__], decision_fieldnames())

        order_rows = _safe_copy_context_list(contexts, "order_send_events")
        position_rows = _safe_copy_context_list(contexts, "position_snapshots")
        history_deal_rows = _safe_copy_context_list(contexts, "history_deals_filtered")
        history_order_rows = _safe_copy_context_list(contexts, "history_orders_filtered")
        write_csv_rows_atomic(order_events_csv_path, order_rows)
        write_csv_rows_atomic(position_snapshots_csv_path, position_rows)
        write_csv_rows_atomic(history_deals_csv_path, history_deal_rows)
        write_csv_rows_atomic(history_orders_csv_path, history_order_rows)

        summary = summarise_live_decisions(decisions)
        unique_events = int(summary.get("unique_completed_m15_events", 0) or 0)
        no_open_position_after_run = state.live_position == 0
        no_failed_order_send = all((not d.order_send_called) or d.order_send_passed is True for d in decisions)
        formal_gate = (
            unique_events >= args.min_new_events
            and no_failed_order_send
            and no_open_position_after_run
            and all(not (ctx.get("forbidden_attempts") if isinstance(ctx, Mapping) else None) for ctx in contexts)
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
                "model_and_scaler": {
                    "environment": report_to_dict(environment),
                    "scaler": report_to_dict(scaler_report),
                    "model": report_to_dict(model_report),
                },
                "latest_shadow_snapshot": latest_snapshot,
                "latest_decision": decisions[-1].__dict__ if decisions else None,
                "summary": summary,
                "state": state.__dict__,
                "safety": {
                    "confirmation_token_used": True,
                    "max_lot_size": float(controls.stage3_first_order_test_volume_lots),
                    "max_successful_entries_per_utc_day": rules_b.max_successful_entries_per_utc_day,
                    "close_at_end": close_at_end,
                    "no_open_position_after_run": no_open_position_after_run,
                    "no_failed_order_send": no_failed_order_send,
                    "do_not_start_final_14_day_run_yet": True,
                },
                "validations": {
                    "minimum_new_events_observed": unique_events >= args.min_new_events,
                    "no_failed_order_send": no_failed_order_send,
                    "no_open_position_after_run": no_open_position_after_run,
                    "state_is_flat_after_run": state.live_position == 0,
                    "forbidden_attempts_absent": all(not (ctx.get("forbidden_attempts") if isinstance(ctx, Mapping) else None) for ctx in contexts),
                },
                "contexts": contexts[-5:],
                "events_csv": str(events_csv_path.relative_to(repo_root)),
                "order_events_csv": str(order_events_csv_path.relative_to(repo_root)),
                "position_snapshots_csv": str(position_snapshots_csv_path.relative_to(repo_root)),
                "history_deals_csv": str(history_deals_csv_path.relative_to(repo_root)),
                "history_orders_csv": str(history_orders_csv_path.relative_to(repo_root)),
                "latest_decision_csv": str(latest_decision_csv_path.relative_to(repo_root)),
                "state_path": str(state_path.relative_to(repo_root)),
                "decision": {
                    "stage3_step3b_controlled_live_passed": formal_gate,
                    "next_step_if_pass": "Review this report. Then either repeat a longer monitored Model B run or prepare Stage 3 Step 4 final-run packaging.",
                    "do_not_start_unattended_final_run_yet": True,
                },
            }
        )
        write_json_atomic(report_path, report)
        LOGGER.info("Stage 3 Step 3B status: %s", report["status"])
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Events CSV: %s", events_csv_path)
        LOGGER.info("Order-send events CSV: %s", order_events_csv_path)
        LOGGER.info("Position snapshots CSV: %s", position_snapshots_csv_path)
        LOGGER.info("History deals CSV: %s", history_deals_csv_path)
        LOGGER.info("History orders CSV: %s", history_orders_csv_path)
        LOGGER.info("Latest decision CSV: %s", latest_decision_csv_path)
        LOGGER.info("State JSON: %s", state_path)
        return 0 if formal_gate else 1
    except Exception as exc:
        report.update(
            {
                "status": "FAIL",
                "formal_gate": False,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
                "latest_shadow_snapshot": latest_snapshot,
                "latest_decision": decisions[-1].__dict__ if decisions else None,
                "state": state.__dict__ if isinstance(state, ModelBLiveState) else None,
                "summary": summarise_live_decisions(decisions) if decisions else {},
                "decision": {
                    "stage3_step3b_controlled_live_passed": False,
                    "manual_review_required": True,
                    "do_not_continue_live_execution_until_reviewed": True,
                },
            }
        )
        try:
            write_json_atomic(report_path, report)
        except Exception:
            pass
        LOGGER.exception("Stage 3 Step 3B failed: %s", exc)
        LOGGER.error("JSON report: %s", report_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())
