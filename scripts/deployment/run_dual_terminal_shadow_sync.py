#!/usr/bin/env python
"""Stage 3 Step 4B concurrent dual-terminal shadow synchronisation gate.

Parent mode launches two isolated worker processes, one per Dukascopy MT5
terminal. Worker mode is internal and must not be invoked manually.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from capstone_trading.runtime.dual_terminal_shadow_sync import (
    DEFAULT_PROBABILITY_TOLERANCE,
    DualTerminalShadowSyncError,
    WorkerPaths,
    append_csv_row,
    build_final_report,
    build_worker_command,
    copy_if_exists,
    inspect_flat_state_and_economics,
    latest_feature_and_sequence_digests,
    load_and_validate_step4a_report,
    normalised_path,
    observation_fieldnames,
    read_csv_rows,
    read_json,
    safe_repo_path,
    sha256_file,
    summary_rows,
    sync_comparison_rows,
    synchronised_event_comparisons,
    utc_now_iso,
    wait_for_workers_and_sync,
    write_csv_atomic,
    write_json_atomic,
)

LOGGER = logging.getLogger("stage3_step4b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage 3 Step 4B concurrent dual-terminal shadow synchronisation gate. "
            "This gate never calls order_check or order_send."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="config/dual_terminal_runtime.yaml")
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument("--model-b-config", default="config/model_b_v2_frozen.yaml")
    parser.add_argument("--freeze-manifest", default="config/stage0_freeze_manifest.json")
    parser.add_argument(
        "--step4a-report",
        default="runtime/reports/stage3_step4a_dual_terminal_readiness.json",
    )
    parser.add_argument("--required-synchronised-events", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-runtime-minutes", type=int, default=90)
    parser.add_argument(
        "--server-time-offset-hours", type=int, choices=(2, 3), default=3
    )
    parser.add_argument(
        "--probability-tolerance", type=float, default=DEFAULT_PROBABILITY_TOLERANCE
    )
    parser.add_argument("--allow-onednn", action="store_true")
    parser.add_argument(
        "--report",
        default="runtime/reports/stage3_step4b_dual_terminal_shadow_sync.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="runtime/reports/stage3_step4b_dual_terminal_shadow_sync_summary.csv",
    )
    parser.add_argument(
        "--synchronised-events-csv",
        default="runtime/reports/stage3_step4b_synchronised_events.csv",
    )
    parser.add_argument(
        "--model-a-events-csv",
        default="runtime/reports/stage3_step4b_model_a_events.csv",
    )
    parser.add_argument(
        "--model-b-events-csv",
        default="runtime/reports/stage3_step4b_model_b_events.csv",
    )
    parser.add_argument(
        "--economic-sanity-csv",
        default="runtime/reports/stage3_step4b_economic_sanity.csv",
    )
    parser.add_argument("--debug", action="store_true")

    # Internal worker arguments.  They are intentionally hidden from help.
    parser.add_argument("--worker-role", choices=("MODEL_A", "MODEL_B"), help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-root", help=argparse.SUPPRESS)
    parser.add_argument("--stop-file", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-utc", help=argparse.SUPPRESS)
    return parser.parse_args()


def worker_paths_from_root(root: Path) -> WorkerPaths:
    return WorkerPaths(
        role_root=root,
        events_csv=root / "observations.csv",
        latest_json=root / "latest_observation.json",
        status_json=root / "worker_status.json",
        state_json=root / "shadow_state.json",
        shadow_signals_csv=root / "shadow_signals.csv",
        worker_stdout_log=root / "worker_stdout.log",
        worker_stderr_log=root / "worker_stderr.log",
    )


def _worker_status_failure(
    *,
    role: str,
    run_id: str,
    paths: WorkerPaths,
    error: Exception,
) -> dict[str, Any]:
    return {
        "stage": 3,
        "step": "4B-WORKER",
        "status": "FAIL",
        "formal_gate": False,
        "role": role,
        "run_id": run_id,
        "completed_utc": utc_now_iso(),
        "role_root": str(paths.role_root),
        "order_check_called": False,
        "order_send_called": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def run_worker(args: argparse.Namespace) -> int:
    """Run one role-specific shadow worker in its own Python process."""

    from capstone_trading.artifacts import (
        verify_notebook7_artifact_bundle,
        verify_stage0_freeze_manifest,
    )
    from capstone_trading.config import load_model_a_config, load_yaml_mapping
    from capstone_trading.evaluation.model_b_replay import (
        overlay_rules_from_model_b_config,
    )
    from capstone_trading.evaluation.trading_replay import overlay_rules_from_config
    from capstone_trading.model_loader import (
        check_runtime_environment,
        load_and_validate_model,
        load_and_validate_scaler,
        report_to_dict,
    )
    from capstone_trading.runtime.dual_terminal_readiness import (
        load_dual_terminal_config,
    )
    from capstone_trading.runtime.mt5_readiness import (
        Mt5RuntimeConfig,
        import_metatrader5_module,
    )
    from capstone_trading.runtime.mt5_shadow import (
        ShadowRuntimeConfig,
        prepare_shadow_cpu_environment,
        run_shadow_once,
        snapshot_to_dict,
    )

    if not args.run_id or not args.worker_root or not args.stop_file or not args.deadline_utc:
        raise DualTerminalShadowSyncError("Internal Step 4B worker arguments are incomplete")
    role = str(args.worker_role)
    run_id = str(args.run_id)
    repo_root = args.repo_root.expanduser().resolve()
    worker_root = safe_repo_path(
        repo_root, args.worker_root, description=f"{role} worker root", must_exist=False
    )
    paths = worker_paths_from_root(worker_root)
    paths.role_root.mkdir(parents=True, exist_ok=True)
    stop_file = safe_repo_path(
        repo_root, args.stop_file, description="Step 4B stop file", must_exist=False
    )
    deadline = datetime.fromisoformat(str(args.deadline_utc).replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    started = utc_now_iso()

    try:
        if args.poll_seconds < 5:
            raise DualTerminalShadowSyncError("Worker poll interval must be at least 5 seconds")
        prepare_shadow_cpu_environment(allow_onednn=args.allow_onednn)
        dual_config_path = safe_repo_path(
            repo_root, args.config, description="Dual-terminal config", must_exist=True
        )
        config_a_path = safe_repo_path(
            repo_root, args.model_a_config, description="Frozen Model A config", must_exist=True
        )
        config_b_path = safe_repo_path(
            repo_root, args.model_b_config, description="Frozen Model B config", must_exist=True
        )
        freeze_manifest_path = safe_repo_path(
            repo_root, args.freeze_manifest, description="Stage 0 freeze manifest", must_exist=True
        )
        step4a_report_path = safe_repo_path(
            repo_root, args.step4a_report, description="Stage 3 Step 4A PASS report", must_exist=True
        )
        dual_config = load_dual_terminal_config(dual_config_path)
        dual_config = replace(
            dual_config, mt5_server_time_offset_hours=args.server_time_offset_hours
        )
        role_config = dual_config.model_a if role == "MODEL_A" else dual_config.model_b
        step4a_report = load_and_validate_step4a_report(
            report_path=step4a_report_path,
            model_a_terminal_path=dual_config.model_a.terminal_path,
            model_b_terminal_path=dual_config.model_b.terminal_path,
        )
        step4a_role_key = "model_a" if role == "MODEL_A" else "model_b"
        step4a_role = step4a_report.get(step4a_role_key, {}) or {}
        expected_login_masked = (step4a_role.get("account", {}) or {}).get("login_masked")
        expected_terminal_directory = str(Path(role_config.terminal_path).resolve().parent)

        config_a = load_model_a_config(config_a_path)
        config_b_raw = load_yaml_mapping(config_b_path)
        verify_stage0_freeze_manifest(repo_root, freeze_manifest_path.relative_to(repo_root))
        bundle = verify_notebook7_artifact_bundle(repo_root, config_a)
        environment = check_runtime_environment(config_a, strict=True)
        rules_a = overlay_rules_from_config(config_a.raw)
        rules_b = overlay_rules_from_model_b_config(config_b_raw)
        scaler, scaler_report = load_and_validate_scaler(
            bundle.scaler_path, config_a, bundle.feature_order
        )
        model, model_report = load_and_validate_model(bundle.model_path, config_a)
        mt5_module = import_metatrader5_module()

        runtime_config = ShadowRuntimeConfig(
            mt5=Mt5RuntimeConfig(
                terminal_path=role_config.terminal_path,
                symbol_candidates=(dual_config.symbol,),
                timeframe_name=dual_config.timeframe,
                bars_to_fetch=420,
                min_completed_bars=260,
                require_demo_account=True,
                allow_market_closed_stale_bar=True,
                max_latest_closed_bar_age_minutes_warning=4320,
                require_symbol_visible=True,
            ),
            bars_to_fetch=420,
            minimum_feature_rows=200,
            minimum_valid_sequences=1,
            append_signals=True,
            duplicate_policy="skip",
            state_path=str(paths.state_json),
            signals_csv_path=str(paths.shadow_signals_csv),
            latest_signal_csv_path=str(paths.role_root / "latest_shadow_signal.csv"),
            mt5_server_time_offset_hours=args.server_time_offset_hours,
            enforce_no_future_canonical_bar=True,
            max_future_canonical_bar_minutes=2,
        )

        economic_sanity = inspect_flat_state_and_economics(
            mt5_module=mt5_module,
            terminal_path=role_config.terminal_path,
            expected_terminal_directory=expected_terminal_directory,
            expected_login_masked=expected_login_masked,
            symbol=dual_config.symbol,
            volume=dual_config.test_volume_lots,
        )
        latest_flat_snapshot = economic_sanity
        poll_iteration = 0
        fresh_event_count = 0
        latest_observation: dict[str, Any] = {}

        while datetime.now(timezone.utc) < deadline and not stop_file.exists():
            poll_iteration += 1
            snapshot, rates, signal = run_shadow_once(
                mt5_module=mt5_module,
                runtime_config=runtime_config,
                config_a=config_a,
                feature_order=bundle.feature_order,
                model=model,
                scaler=scaler,
                rules_a=rules_a,
                rules_b=rules_b,
                state_path=paths.state_json,
                signals_csv_path=paths.shadow_signals_csv,
                mode="dual_terminal_shadow",
                run_id=run_id,
            )
            snapshot_payload = snapshot_to_dict(snapshot)
            signal_payload = dict(snapshot_payload.get("latest_signal", {}) or {})
            digests = latest_feature_and_sequence_digests(
                rates=rates,
                feature_order=bundle.feature_order,
                scaler=scaler,
                sequence_length=config_a.sequence_length,
            )
            if str(signal_payload.get("event_time_utc")) != str(digests.get("event_time_utc")):
                raise DualTerminalShadowSyncError(
                    "Shadow signal event does not match reconstructed sequence fingerprint event"
                )
            is_fresh = not bool(signal_payload.get("duplicate_event")) and not bool(
                signal_payload.get("stale_event_warning")
            )
            if is_fresh:
                fresh_event_count += 1
                latest_flat_snapshot = inspect_flat_state_and_economics(
                    mt5_module=mt5_module,
                    terminal_path=role_config.terminal_path,
                    expected_terminal_directory=expected_terminal_directory,
                    expected_login_masked=expected_login_masked,
                    symbol=dual_config.symbol,
                    volume=dual_config.test_volume_lots,
                )
            if int(latest_flat_snapshot.get("position_count", -1)) != 0 or int(
                latest_flat_snapshot.get("pending_order_count", -1)
            ) != 0:
                raise DualTerminalShadowSyncError(
                    f"{role} account ceased to be flat during Step 4B"
                )
            role_signal = (
                int(signal_payload.get("model_a_signal", 0))
                if role == "MODEL_A"
                else int(signal_payload.get("model_b_from_flat_signal", 0))
            )
            role_signal_name = (
                str(signal_payload.get("model_a_signal_name", "FLAT"))
                if role == "MODEL_A"
                else str(signal_payload.get("model_b_from_flat_signal_name", "FLAT"))
            )
            terminal = snapshot_payload.get("terminal", {}) or {}
            account = snapshot_payload.get("account", {}) or {}
            symbol_info = (
                (snapshot_payload.get("symbol_resolution", {}) or {}).get("symbol_info", {}) or {}
            )
            latest_observation = {
                "run_id": run_id,
                "role": role,
                "poll_iteration": poll_iteration,
                "observed_utc": utc_now_iso(),
                "event_time_utc": signal_payload.get("event_time_utc"),
                "latest_completed_bar_time_utc": signal_payload.get(
                    "latest_completed_bar_time_utc"
                ),
                "probability_up": signal_payload.get("probability_up"),
                "model_a_signal": signal_payload.get("model_a_signal"),
                "model_a_signal_name": signal_payload.get("model_a_signal_name"),
                "model_b_from_flat_signal": signal_payload.get("model_b_from_flat_signal"),
                "model_b_from_flat_signal_name": signal_payload.get(
                    "model_b_from_flat_signal_name"
                ),
                "model_b_entry_condition": signal_payload.get("model_b_entry_condition"),
                "model_b_hold_condition": signal_payload.get("model_b_hold_condition"),
                "role_shadow_signal": role_signal,
                "role_shadow_signal_name": role_signal_name,
                "duplicate_event": signal_payload.get("duplicate_event"),
                "stale_event_warning": signal_payload.get("stale_event_warning"),
                "spread_points": symbol_info.get("spread"),
                "actual_position_count": latest_flat_snapshot.get("position_count"),
                "pending_order_count": latest_flat_snapshot.get("pending_order_count"),
                "feature_rows": digests.get("feature_rows"),
                "feature_count": digests.get("feature_count"),
                "sequence_length": digests.get("sequence_length"),
                "valid_sequence_count": digests.get("valid_sequence_count"),
                "rates_digest": digests.get("rates_digest"),
                "feature_digest": digests.get("feature_digest"),
                "sequence_digest": digests.get("sequence_digest"),
                "terminal_executable": role_config.terminal_path,
                "terminal_reported_path": terminal.get("path"),
                "terminal_build": terminal.get("build"),
                "terminal_data_path": terminal.get("data_path"),
                "login_masked": account.get("login_masked"),
                "forbidden_trade_function_calls": "|".join(
                    str(item)
                    for item in snapshot_payload.get("forbidden_trade_function_calls", [])
                ),
                "order_check_called": False,
                "order_send_called": False,
            }
            if normalised_path(str(terminal.get("path", ""))) != normalised_path(
                expected_terminal_directory
            ):
                raise DualTerminalShadowSyncError(
                    f"{role} shadow worker attached to the wrong terminal directory"
                )
            if account.get("login_masked") != expected_login_masked:
                raise DualTerminalShadowSyncError(
                    f"{role} shadow worker attached to a different account"
                )
            if snapshot_payload.get("forbidden_trade_function_calls"):
                raise DualTerminalShadowSyncError(
                    f"{role} recorded forbidden MT5 calls: "
                    f"{snapshot_payload.get('forbidden_trade_function_calls')}"
                )
            append_csv_row(paths.events_csv, latest_observation, observation_fieldnames())
            write_json_atomic(paths.latest_json, latest_observation)
            LOGGER.info(
                "%s iteration=%s event=%s p=%.6f role_signal=%s duplicate=%s",
                role,
                poll_iteration,
                latest_observation["event_time_utc"],
                float(latest_observation["probability_up"]),
                role_signal_name,
                latest_observation["duplicate_event"],
            )
            if not stop_file.exists() and datetime.now(timezone.utc) < deadline:
                time.sleep(args.poll_seconds)

        final_flat = inspect_flat_state_and_economics(
            mt5_module=mt5_module,
            terminal_path=role_config.terminal_path,
            expected_terminal_directory=expected_terminal_directory,
            expected_login_masked=expected_login_masked,
            symbol=dual_config.symbol,
            volume=dual_config.test_volume_lots,
        )
        status = {
            "stage": 3,
            "step": "4B-WORKER",
            "status": "PASS",
            "formal_gate": True,
            "role": role,
            "run_id": run_id,
            "started_utc": started,
            "completed_utc": utc_now_iso(),
            "role_root": str(paths.role_root),
            "terminal_executable": role_config.terminal_path,
            "terminal_reported_path": final_flat.get("terminal_reported_path"),
            "terminal_build": latest_observation.get("terminal_build"),
            "terminal_data_path": latest_observation.get("terminal_data_path"),
            "login_masked": final_flat.get("login_masked"),
            "fresh_event_count": fresh_event_count,
            "poll_iteration_count": poll_iteration,
            "final_position_count": final_flat.get("position_count"),
            "final_pending_order_count": final_flat.get("pending_order_count"),
            "latest_event_time_utc": latest_observation.get("event_time_utc"),
            "environment": report_to_dict(environment),
            "scaler": report_to_dict(scaler_report),
            "model": report_to_dict(model_report),
            "model_artifact_sha256": sha256_file(bundle.model_path)["sha256"],
            "scaler_artifact_sha256": sha256_file(bundle.scaler_path)["sha256"],
            "feature_order_sha256": hashlib.sha256(
                json.dumps(list(bundle.feature_order), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "economic_sanity": economic_sanity,
            "final_economic_sanity": final_flat,
            "order_check_called": False,
            "order_send_called": False,
            "orders_enabled": False,
            "termination_reason": "parent_stop_file" if stop_file.exists() else "deadline_reached",
        }
        write_json_atomic(paths.status_json, status)
        return 0
    except Exception as exc:
        failure = _worker_status_failure(
            role=role, run_id=run_id, paths=paths, error=exc
        )
        write_json_atomic(paths.status_json, failure)
        LOGGER.exception("%s Step 4B worker failed", role)
        return 2


def run_parent(args: argparse.Namespace) -> int:
    from capstone_trading.runtime.dual_terminal_readiness import (
        load_dual_terminal_config,
    )

    if args.required_synchronised_events < 1:
        raise DualTerminalShadowSyncError(
            "required-synchronised-events must be at least 1"
        )
    if args.poll_seconds < 5:
        raise DualTerminalShadowSyncError("poll-seconds must be at least 5")
    if args.max_runtime_minutes < 15:
        raise DualTerminalShadowSyncError("max-runtime-minutes must be at least 15")
    if args.probability_tolerance <= 0 or args.probability_tolerance > 1e-5:
        raise DualTerminalShadowSyncError(
            "probability-tolerance must be positive and no larger than 1e-5"
        )

    repo_root = args.repo_root.expanduser().resolve()
    config_path = safe_repo_path(
        repo_root, args.config, description="Dual-terminal config", must_exist=True
    )
    model_a_config = safe_repo_path(
        repo_root, args.model_a_config, description="Frozen Model A config", must_exist=True
    )
    model_b_config = safe_repo_path(
        repo_root, args.model_b_config, description="Frozen Model B config", must_exist=True
    )
    freeze_manifest = safe_repo_path(
        repo_root, args.freeze_manifest, description="Stage 0 freeze manifest", must_exist=True
    )
    step4a_report_path = safe_repo_path(
        repo_root, args.step4a_report, description="Stage 3 Step 4A report", must_exist=True
    )
    report_path = safe_repo_path(
        repo_root, args.report, description="Stage 3 Step 4B JSON report"
    )
    summary_path = safe_repo_path(
        repo_root, args.summary_csv, description="Stage 3 Step 4B summary CSV"
    )
    synced_path = safe_repo_path(
        repo_root,
        args.synchronised_events_csv,
        description="Stage 3 Step 4B synchronised events CSV",
    )
    model_a_events_output = safe_repo_path(
        repo_root, args.model_a_events_csv, description="Step 4B Model A events CSV"
    )
    model_b_events_output = safe_repo_path(
        repo_root, args.model_b_events_csv, description="Step 4B Model B events CSV"
    )
    economic_output = safe_repo_path(
        repo_root, args.economic_sanity_csv, description="Step 4B economic sanity CSV"
    )

    dual_config = load_dual_terminal_config(config_path)
    dual_config = replace(
        dual_config, mt5_server_time_offset_hours=args.server_time_offset_hours
    )
    step4a_report = load_and_validate_step4a_report(
        report_path=step4a_report_path,
        model_a_terminal_path=dual_config.model_a.terminal_path,
        model_b_terminal_path=dual_config.model_b.terminal_path,
    )
    run_id = datetime.now(timezone.utc).strftime("stage3_step4b_%Y%m%dT%H%M%SZ")
    started_utc = utc_now_iso()
    deadline = datetime.now(timezone.utc) + timedelta(minutes=args.max_runtime_minutes)
    deadline_utc = deadline.isoformat()
    coordinator_root = safe_repo_path(
        repo_root,
        Path("runtime/coordinator/stage3_step4b") / run_id,
        description="Step 4B coordinator root",
    )
    coordinator_root.mkdir(parents=True, exist_ok=True)
    stop_file = coordinator_root / "STOP"
    role_root_a = safe_repo_path(
        repo_root,
        Path(dual_config.model_a.runtime_root) / "stage3_step4b" / run_id,
        description="Model A Step 4B runtime root",
    )
    role_root_b = safe_repo_path(
        repo_root,
        Path(dual_config.model_b.runtime_root) / "stage3_step4b" / run_id,
        description="Model B Step 4B runtime root",
    )
    paths_a = worker_paths_from_root(role_root_a)
    paths_b = worker_paths_from_root(role_root_b)
    paths_a.role_root.mkdir(parents=True, exist_ok=True)
    paths_b.role_root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    command_a = build_worker_command(
        python_executable=sys.executable,
        script_path=script_path,
        repo_root=repo_root,
        run_id=run_id,
        role="MODEL_A",
        config_path=config_path,
        model_a_config=model_a_config,
        model_b_config=model_b_config,
        freeze_manifest=freeze_manifest,
        step4a_report=step4a_report_path,
        worker_root=role_root_a,
        stop_file=stop_file,
        deadline_utc=deadline_utc,
        poll_seconds=args.poll_seconds,
        server_time_offset_hours=args.server_time_offset_hours,
        allow_onednn=args.allow_onednn,
    )
    command_b = build_worker_command(
        python_executable=sys.executable,
        script_path=script_path,
        repo_root=repo_root,
        run_id=run_id,
        role="MODEL_B",
        config_path=config_path,
        model_a_config=model_a_config,
        model_b_config=model_b_config,
        freeze_manifest=freeze_manifest,
        step4a_report=step4a_report_path,
        worker_root=role_root_b,
        stop_file=stop_file,
        deadline_utc=deadline_utc,
        poll_seconds=args.poll_seconds,
        server_time_offset_hours=args.server_time_offset_hours,
        allow_onednn=args.allow_onednn,
    )

    LOGGER.info("Starting Step 4B run_id=%s", run_id)
    LOGGER.info("Required synchronised completed M15 events: %s", args.required_synchronised_events)
    LOGGER.info("Maximum runtime: %s minutes", args.max_runtime_minutes)
    out_a = paths_a.worker_stdout_log.open("w", encoding="utf-8")
    err_a = paths_a.worker_stderr_log.open("w", encoding="utf-8")
    out_b = paths_b.worker_stdout_log.open("w", encoding="utf-8")
    err_b = paths_b.worker_stderr_log.open("w", encoding="utf-8")
    process_a: subprocess.Popen[Any] | None = None
    process_b: subprocess.Popen[Any] | None = None
    try:
        process_a = subprocess.Popen(
            command_a, cwd=repo_root, stdout=out_a, stderr=err_a, text=True
        )
        process_b = subprocess.Popen(
            command_b, cwd=repo_root, stdout=out_b, stderr=err_b, text=True
        )
        comparisons = wait_for_workers_and_sync(
            process_a=process_a,
            process_b=process_b,
            paths_a=paths_a,
            paths_b=paths_b,
            stop_file=stop_file,
            run_id=run_id,
            required_synchronised_events=args.required_synchronised_events,
            deadline=deadline,
            probability_tolerance=args.probability_tolerance,
        )
        # Workers check the stop file at the next loop boundary.  Allow one poll
        # interval plus loading overhead before using controlled termination.
        worker_wait_seconds = max(45, args.poll_seconds + 30)
        try:
            process_a.wait(timeout=worker_wait_seconds)
        except subprocess.TimeoutExpired:
            process_a.terminate()
            process_a.wait(timeout=15)
        try:
            process_b.wait(timeout=worker_wait_seconds)
        except subprocess.TimeoutExpired:
            process_b.terminate()
            process_b.wait(timeout=15)
    except KeyboardInterrupt:
        stop_file.write_text(utc_now_iso(), encoding="utf-8")
        if process_a is not None and process_a.poll() is None:
            process_a.terminate()
        if process_b is not None and process_b.poll() is None:
            process_b.terminate()
        raise
    finally:
        out_a.close()
        err_a.close()
        out_b.close()
        err_b.close()

    rows_a = read_csv_rows(paths_a.events_csv)
    rows_b = read_csv_rows(paths_b.events_csv)
    comparisons = synchronised_event_comparisons(
        rows_a,
        rows_b,
        run_id=run_id,
        probability_tolerance=args.probability_tolerance,
    )
    status_a = read_json(paths_a.status_json)
    status_b = read_json(paths_b.status_json)
    copy_if_exists(paths_a.events_csv, model_a_events_output)
    copy_if_exists(paths_b.events_csv, model_b_events_output)
    write_csv_atomic(synced_path, sync_comparison_rows(comparisons))

    economic_rows = []
    for role, status in (("MODEL_A", status_a), ("MODEL_B", status_b)):
        economic = status.get("economic_sanity", {}) or {}
        economic_rows.append(
            {
                "role": role,
                "login_masked": status.get("login_masked"),
                "buy_margin_account_currency": economic.get(
                    "buy_margin_account_currency"
                ),
                "sell_margin_account_currency": economic.get(
                    "sell_margin_account_currency"
                ),
                "buy_profit_for_positive_price_move_account_currency": economic.get(
                    "buy_profit_for_positive_price_move_account_currency"
                ),
                "sell_profit_for_positive_price_move_account_currency": economic.get(
                    "sell_profit_for_positive_price_move_account_currency"
                ),
                "trade_tick_value_metadata": economic.get("trade_tick_value_metadata"),
                "position_count": economic.get("position_count"),
                "pending_order_count": economic.get("pending_order_count"),
                "order_check_called": economic.get("order_check_called"),
                "order_send_called": economic.get("order_send_called"),
            }
        )
    write_csv_atomic(economic_output, economic_rows)

    report = build_final_report(
        run_id=run_id,
        started_utc=started_utc,
        completed_utc=utc_now_iso(),
        required_synchronised_events=args.required_synchronised_events,
        poll_seconds=args.poll_seconds,
        max_runtime_minutes=args.max_runtime_minutes,
        comparisons=comparisons,
        worker_status_a=status_a,
        worker_status_b=status_b,
        step4a_report=step4a_report,
        source_paths={
            "model_a_events": paths_a.events_csv,
            "model_b_events": paths_b.events_csv,
            "model_a_worker_status": paths_a.status_json,
            "model_b_worker_status": paths_b.status_json,
            "step4a_prerequisite_report": step4a_report_path,
        },
        probability_tolerance=args.probability_tolerance,
    )
    report["repository_root"] = str(repo_root)
    report["config_path"] = str(config_path.relative_to(repo_root))
    report["outputs"] = {
        "report_json": str(report_path.relative_to(repo_root)),
        "summary_csv": str(summary_path.relative_to(repo_root)),
        "synchronised_events_csv": str(synced_path.relative_to(repo_root)),
        "model_a_events_csv": str(model_a_events_output.relative_to(repo_root)),
        "model_b_events_csv": str(model_b_events_output.relative_to(repo_root)),
        "economic_sanity_csv": str(economic_output.relative_to(repo_root)),
        "model_a_worker_root": str(role_root_a.relative_to(repo_root)),
        "model_b_worker_root": str(role_root_b.relative_to(repo_root)),
    }
    write_json_atomic(report_path, report)
    write_csv_atomic(summary_path, summary_rows(report))

    LOGGER.info("Stage 3 Step 4B status: %s", report["status"])
    LOGGER.info("Synchronised completed M15 events: %s", len(comparisons))
    LOGGER.info(
        "Maximum probability difference: %s",
        report.get("summary", {}).get("maximum_probability_absolute_difference"),
    )
    LOGGER.info(
        "Economic calculation parity: %s",
        report.get("economic_comparison", {}).get("passed"),
    )
    LOGGER.info("order_check called: False")
    LOGGER.info("order_send called: False")
    LOGGER.info("JSON report: %s", report_path)
    LOGGER.info("Summary CSV: %s", summary_path)
    LOGGER.info("Synchronised events CSV: %s", synced_path)
    return 0 if report.get("formal_gate") is True else 2



def write_parent_terminal_report(args: argparse.Namespace, *, status: str, error: BaseException | None = None) -> None:
    if args.worker_role:
        return
    try:
        repo_root = args.repo_root.expanduser().resolve()
        report_path = safe_repo_path(
            repo_root, args.report, description="Stage 3 Step 4B failure report"
        )
        payload = {
            "stage": 3,
            "step": "4B",
            "status": status,
            "formal_gate": False,
            "purpose": "concurrent_dual_terminal_shadow_synchronisation_no_order",
            "completed_utc": utc_now_iso(),
            "orders_enabled": False,
            "order_check_called": False,
            "order_send_called": False,
            "decision": {
                "stage3_step4b_passed": False,
                "final_14_day_run_authorised": False,
                "manual_review_required": True,
            },
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)
        write_json_atomic(report_path, payload)
    except Exception:
        LOGGER.exception("Unable to write Step 4B terminal report")

def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        if args.worker_role:
            return run_worker(args)
        return run_parent(args)
    except KeyboardInterrupt as exc:
        write_parent_terminal_report(args, status="INTERRUPTED", error=exc)
        LOGGER.error("Stage 3 Step 4B interrupted. Workers were asked to stop; final gate not passed.")
        return 130
    except Exception as exc:
        write_parent_terminal_report(args, status="FAIL", error=exc)
        LOGGER.exception("Stage 3 Step 4B failed") if args.debug else LOGGER.error(
            "Stage 3 Step 4B failed: %s", exc
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
