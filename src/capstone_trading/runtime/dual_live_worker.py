"""One restart-safe Model A or Model B dual-rehearsal worker.

The worker deliberately runs one role per operating-system process because the
MetaTrader5 Python package has one process-global terminal connection.  Each
iteration first executes the existing completed-M15 shadow inference pipeline,
then reconciles persistent state against the broker, applies the frozen strategy
and risk rules, and finally either records a virtual transition (shadow mode) or
uses the guarded execution module (live mode).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from types import SimpleNamespace
import json
import os
import time
import traceback

import pandas as pd

from capstone_trading.artifacts import (
    verify_notebook7_artifact_bundle,
    verify_stage0_freeze_manifest,
)
from capstone_trading.config import (
    load_model_a_config,
    load_yaml_mapping,
    safe_repository_path,
)
from capstone_trading.evaluation.model_b_replay import overlay_rules_from_model_b_config
from capstone_trading.evaluation.trading_replay import overlay_rules_from_config
from capstone_trading.model_loader import (
    check_runtime_environment,
    load_and_validate_model,
    load_and_validate_scaler,
)
from capstone_trading.runtime.dual_live_execution import (
    EntrySpreadBlocked,
    collect_broker_history,
    execute_transition,
    flatten_position,
    inspect_broker,
)
from capstone_trading.runtime.dual_live_state import (
    DualLiveState,
    RiskRules,
    StrategyDecision,
    StrategyRules,
    advance_blocked_or_hold_state,
    apply_transition,
    decide_strategy_transition,
    decision_to_mapping,
    heartbeat_payload,
    load_state,
    reconcile_state,
    session_gap_lockout_status,
    summarise_decisions,
    update_decision_execution,
    update_risk_state,
    utc_now_iso,
    write_json_atomic,
    write_state,
)
from capstone_trading.runtime.live_audit import (
    BROKER_DEAL_FIELDS,
    BROKER_ORDER_FIELDS,
    DECISION_FIELDS,
    ORDER_EVENT_FIELDS,
    RUNTIME_EVENT_FIELDS,
    TELEMETRY_FIELDS,
    append_csv_row,
    append_unique_rows,
    broker_deal_row,
    broker_order_row,
    completed_bar_context,
    decision_audit_row,
    execution_audit_rows,
    runtime_event_audit_row,
    spread_points_from_tick,
    telemetry_audit_row,
)
from capstone_trading.runtime.mt5_readiness import (
    SafeMt5Proxy,
    fetch_completed_m15_rates,
    import_metatrader5_module,
    initialise_terminal as initialise_read_only_terminal,
    resolve_symbol,
    resolve_timeframe,
)
from capstone_trading.runtime.mt5_shadow import (
    load_shadow_runtime_config,
    prepare_shadow_cpu_environment,
    run_shadow_once,
    snapshot_to_dict,
)
from capstone_trading.runtime.order_preflight import load_frozen_controls


LIVE_CONFIRMATION_TOKEN = "I_UNDERSTAND_DUAL_REHEARSAL_SENDS_DEMO_ORDERS"


class DualLiveWorkerError(RuntimeError):
    """Raised when a worker cannot continue safely."""


@dataclass(frozen=True)
class WorkerPaths:
    role_root: Path
    state: Path
    heartbeat: Path
    decisions_csv: Path
    telemetry_csv: Path
    order_events_csv: Path
    broker_deals_csv: Path
    broker_orders_csv: Path
    runtime_events_csv: Path
    audit_session: Path
    latest_decision: Path
    latest_shadow_snapshot: Path
    final_report: Path
    stop_file: Path
    kill_switch_file: Path
    shadow_unused_state: Path
    shadow_unused_signals: Path


@dataclass(frozen=True)
class WorkerSettings:
    repo_root: Path
    role: str
    terminal_path: str
    expected_login_suffix: str
    execution_mode: str
    orders_enabled: bool
    poll_seconds: int
    max_iterations: int
    history_refresh_seconds: int
    model_a_config: Path
    model_b_config: Path
    freeze_manifest: Path
    shadow_runtime_config: Path
    broker_controls: Path
    paths: WorkerPaths
    flatten_on_clean_stop: bool
    flatten_only: bool
    confirmation: str | None
    session_gap_lockout_enabled: bool = True
    session_gap_lockout_start_utc_minutes: int = 20 * 60 + 30
    session_gap_lockout_daily_end_utc_minutes: int = 22 * 60
    session_gap_lockout_weekend_end_utc_minutes: int = 22 * 60
    allow_onednn: bool = False


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise DualLiveWorkerError(f"Config section {key!r} must be a mapping")
    return value


def _parse_utc_hhmm(value: Any, *, field_name: str) -> int:
    text = str(value).strip()
    try:
        hour_text, minute_text = text.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (TypeError, ValueError) as exc:
        raise DualLiveWorkerError(
            f"{field_name} must use HH:MM UTC format"
        ) from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise DualLiveWorkerError(
            f"{field_name} must use a valid HH:MM UTC time"
        )
    return hour * 60 + minute


def load_worker_settings(
    *,
    repo_root: Path,
    config_path: Path,
    role: str,
    execution_mode_override: str | None = None,
    orders_enabled_override: bool | None = None,
    flatten_only: bool = False,
    confirmation: str | None = None,
) -> WorkerSettings:
    if role not in {"model_a", "model_b"}:
        raise DualLiveWorkerError(f"Unsupported role: {role}")
    raw = load_yaml_mapping(config_path)
    runtime = _mapping(raw, "runtime")
    market_session_safety = _mapping(raw, "market_session_safety")
    models = _mapping(raw, "models")
    paths = _mapping(raw, "paths")
    role_config = _mapping(models, role)
    execution_mode = str(
        execution_mode_override or runtime.get("execution_mode", "shadow")
    ).strip().lower()
    if execution_mode not in {"shadow", "live"}:
        raise DualLiveWorkerError("execution_mode must be shadow or live")
    config_orders_enabled = bool(runtime.get("orders_enabled", False))
    orders_enabled = (
        config_orders_enabled
        if orders_enabled_override is None
        else bool(orders_enabled_override)
    )
    if execution_mode == "shadow" and orders_enabled:
        raise DualLiveWorkerError("orders_enabled cannot be true in shadow mode")
    if orders_enabled and confirmation != LIVE_CONFIRMATION_TOKEN:
        raise DualLiveWorkerError(
            "Live order mode requires the exact dual-rehearsal confirmation token"
        )
    expected_suffix = str(role_config.get("expected_login_suffix", "")).strip()
    if len(expected_suffix) != 4 or not expected_suffix.isdigit():
        raise DualLiveWorkerError(
            f"{role} expected_login_suffix must contain exactly four digits"
        )
    terminal_path = str(role_config.get("terminal_path", "")).strip()
    if not terminal_path:
        raise DualLiveWorkerError(f"{role} terminal_path is required")
    runtime_root_raw = str(paths.get("runtime_root", "runtime/dual_live_rehearsal"))
    runtime_root = safe_repository_path(
        repo_root,
        runtime_root_raw,
        description="dual live runtime root",
        must_exist=False,
    )
    role_root = runtime_root / role
    shared_root = runtime_root / "shared"
    worker_paths = WorkerPaths(
        role_root=role_root,
        state=role_root / "state.json",
        heartbeat=role_root / "heartbeat.json",
        decisions_csv=role_root / "decisions.csv",
        telemetry_csv=role_root / "telemetry.csv",
        order_events_csv=role_root / "order_events.csv",
        broker_deals_csv=role_root / "broker_deals.csv",
        broker_orders_csv=role_root / "broker_orders.csv",
        runtime_events_csv=role_root / "runtime_events.csv",
        audit_session=role_root / "audit_session.json",
        latest_decision=role_root / "latest_decision.json",
        latest_shadow_snapshot=role_root / "latest_shadow_snapshot.json",
        final_report=role_root / "final_report.json",
        stop_file=shared_root / "STOP",
        kill_switch_file=shared_root / "KILL_SWITCH",
        shadow_unused_state=role_root / "shadow_internal_state.json",
        shadow_unused_signals=role_root / "shadow_internal_signals.csv",
    )
    lockout_start = _parse_utc_hhmm(
        market_session_safety.get("pre_break_flatten_utc", "20:30"),
        field_name="market_session_safety.pre_break_flatten_utc",
    )
    lockout_end = _parse_utc_hhmm(
        market_session_safety.get("daily_reopen_utc", "22:00"),
        field_name="market_session_safety.daily_reopen_utc",
    )
    weekend_lockout_end = _parse_utc_hhmm(
        market_session_safety.get("weekend_reopen_utc", "22:00"),
        field_name="market_session_safety.weekend_reopen_utc",
    )
    if lockout_start >= lockout_end:
        raise DualLiveWorkerError(
            "market_session_safety.pre_break_flatten_utc must be before "
            "daily_reopen_utc"
        )
    return WorkerSettings(
        repo_root=repo_root,
        role=role,
        terminal_path=terminal_path,
        expected_login_suffix=expected_suffix,
        execution_mode=execution_mode,
        orders_enabled=orders_enabled,
        poll_seconds=max(5, int(runtime.get("poll_seconds", 30))),
        max_iterations=max(0, int(runtime.get("max_iterations", 0))),
        history_refresh_seconds=max(
            60, int(runtime.get("history_refresh_seconds", 300))
        ),
        model_a_config=Path(str(paths.get("model_a_config", "config/model_a_frozen.yaml"))),
        model_b_config=Path(str(paths.get("model_b_config", "config/model_b_v2_frozen.yaml"))),
        freeze_manifest=Path(str(paths.get("freeze_manifest", "config/stage0_freeze_manifest.json"))),
        shadow_runtime_config=Path(
            str(paths.get("shadow_runtime_config", "config/mt5_shadow_runtime_template.yaml"))
        ),
        broker_controls=Path(
            str(
                paths.get(
                    "broker_controls",
                    "runtime/frozen/vps_rehearsal_broker_execution_controls_frozen.yaml",
                )
            )
        ),
        paths=worker_paths,
        flatten_on_clean_stop=bool(runtime.get("flatten_on_clean_stop", True)),
        flatten_only=bool(flatten_only),
        confirmation=confirmation,
        session_gap_lockout_enabled=bool(
            market_session_safety.get("enabled", True)
        ),
        session_gap_lockout_start_utc_minutes=lockout_start,
        session_gap_lockout_daily_end_utc_minutes=lockout_end,
        session_gap_lockout_weekend_end_utc_minutes=weekend_lockout_end,
        allow_onednn=bool(runtime.get("allow_onednn", False)),
    )


def probe_latest_completed_event_time(
    *,
    mt5_module: Any,
    runtime_config: Any,
) -> str:
    """Read only the latest completed M15 timestamp without model inference.

    MT5 bar epochs on this Dukascopy setup encode broker server time.  The same
    frozen +3-hour conversion used by the shadow pipeline is applied before the
    timestamp is compared with persistent strategy state.
    """

    proxy = SafeMt5Proxy(mt5_module)
    initialized = False
    try:
        timeframe = resolve_timeframe(proxy, runtime_config.mt5.timeframe_name)
        initialise_read_only_terminal(proxy, runtime_config.mt5)
        initialized = True
        symbol = resolve_symbol(proxy, runtime_config.mt5).selected_symbol
        rates = fetch_completed_m15_rates(
            proxy,
            symbol=symbol,
            timeframe_value=timeframe,
            count=1,
        )
        raw = pd.Timestamp(rates["time"].iloc[-1])
        if raw.tzinfo is None:
            raw = raw.tz_localize("UTC")
        else:
            raw = raw.tz_convert("UTC")
        canonical = raw - pd.Timedelta(
            hours=int(runtime_config.mt5_server_time_offset_hours)
        )
        return canonical.isoformat()
    finally:
        if initialized:
            proxy.shutdown()
        if proxy.forbidden_attempts:
            raise DualLiveWorkerError(
                f"Latest-bar probe attempted forbidden APIs: {proxy.forbidden_attempts}"
            )


def _rules_from_configs(config_a: Any, config_b_raw: Mapping[str, Any], role: str) -> tuple[StrategyRules, RiskRules, Any, Any]:
    rules_a = overlay_rules_from_config(config_a.raw)
    rules_b = overlay_rules_from_model_b_config(config_b_raw)
    live_risk = config_a.raw.get("live_risk_semantics", {})
    if not isinstance(live_risk, Mapping):
        live_risk = {}
    risk = RiskRules(
        daily_loss_stop_simple_return=float(
            live_risk.get("daily_loss_stop_simple_return", -0.02)
        ),
        total_drawdown_stop=float(live_risk.get("total_drawdown_stop", -0.15)),
    )
    if role == "model_a":
        strategy = StrategyRules(
            role=role,
            long_threshold=float(rules_a.long_threshold),
            short_threshold=float(rules_a.short_threshold),
            exit_threshold=None,
            minimum_hold_bars=int(rules_a.minimum_hold_bars),
            max_policy_changes_per_utc_day=int(rules_a.max_policy_changes_per_day),
            max_successful_entries_per_utc_day=None,
            long_only=False,
            reversal_policy_event_units=int(rules_a.reversal_policy_event_units),
            allow_risk_reducing_exit_when_capped=True,
        )
    else:
        strategy = StrategyRules(
            role=role,
            long_threshold=float(rules_b.entry_threshold),
            short_threshold=None,
            exit_threshold=float(rules_b.exit_threshold),
            minimum_hold_bars=0,
            max_policy_changes_per_utc_day=None,
            max_successful_entries_per_utc_day=int(
                rules_b.max_successful_entries_per_day
            ),
            long_only=True,
        )
    strategy.validate()
    risk.validate()
    return strategy, risk, rules_a, rules_b


def _load_or_create_observation_start(
    settings: WorkerSettings,
    *,
    run_id: str,
    worker_started_utc: str,
) -> datetime:
    path = settings.paths.audit_session
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, Mapping):
                raise TypeError("audit session JSON must contain an object")
            role = str(payload.get("role", ""))
            execution_mode = str(payload.get("execution_mode", ""))
            started_raw = str(payload.get("observation_started_utc", ""))
            started = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
        except Exception as exc:
            raise DualLiveWorkerError(
                f"Invalid existing audit session file {path}: {exc}"
            ) from exc
        if role != settings.role or execution_mode != settings.execution_mode:
            raise DualLiveWorkerError(
                "Audit session role/execution mode does not match worker settings"
            )
        if started.tzinfo is None:
            raise DualLiveWorkerError(
                "Audit session observation_started_utc must be timezone-aware"
            )
        return started.astimezone(timezone.utc)

    started = datetime.fromisoformat(
        worker_started_utc.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    write_json_atomic(
        path,
        {
            "schema_version": "1.0",
            "role": settings.role,
            "execution_mode": settings.execution_mode,
            "observation_started_utc": started.isoformat(),
            "created_by_run_id": run_id,
            "created_by_worker_pid": os.getpid(),
        },
    )
    return started


def _write_telemetry(
    settings: WorkerSettings,
    *,
    run_id: str,
    iteration: int,
    snapshot_phase: str,
    state: DualLiveState,
    broker_inspection: Any,
    latest_completed_event_time_utc: str | None,
    latest_decision: Any,
    server_time_offset_hours: int = 0,
) -> None:
    append_csv_row(
        settings.paths.telemetry_csv,
        telemetry_audit_row(
            role=settings.role,
            run_id=run_id,
            iteration=iteration,
            worker_pid=os.getpid(),
            snapshot_phase=snapshot_phase,
            execution_mode=settings.execution_mode,
            orders_enabled=settings.orders_enabled,
            latest_completed_event_time_utc=latest_completed_event_time_utc,
            latest_decision=latest_decision,
            state=state,
            broker_inspection=broker_inspection,
            stop_file_exists=settings.paths.stop_file.exists(),
            kill_switch_file_exists=settings.paths.kill_switch_file.exists(),
            server_time_offset_hours=server_time_offset_hours,
        ),
        fieldnames=TELEMETRY_FIELDS,
    )


def _write_runtime_event(
    settings: WorkerSettings,
    *,
    run_id: str,
    iteration: int,
    event_type: str,
    event_reason: str,
    severity: str,
    state: DualLiveState,
    event_time_utc: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    append_csv_row(
        settings.paths.runtime_events_csv,
        runtime_event_audit_row(
            role=settings.role,
            run_id=run_id,
            iteration=iteration,
            worker_pid=os.getpid(),
            event_type=event_type,
            event_reason=event_reason,
            severity=severity,
            state=state,
            event_time_utc=event_time_utc,
            details=details,
        ),
        fieldnames=RUNTIME_EVENT_FIELDS,
    )


def _refresh_broker_history(
    settings: WorkerSettings,
    *,
    mt5_module: Any,
    controls: Any,
    run_id: str,
    observation_started_at: datetime,
    expected_order_tickets: set[int] | None = None,
    expected_deal_tickets: set[int] | None = None,
    retries: int = 1,
    retry_sleep_seconds: float = 0.5,
    server_time_offset_hours: int = 0,
) -> tuple[int, int]:
    expected_orders = set(expected_order_tickets or ())
    expected_deals = set(expected_deal_tickets or ())
    attempts = max(1, int(retries))
    total_deals_added = 0
    total_orders_added = 0
    last_found_orders: set[int] = set()
    last_found_deals: set[int] = set()

    for attempt in range(1, attempts + 1):
        audit = collect_broker_history(
            mt5_module=mt5_module,
            terminal_path=settings.terminal_path,
            controls=controls,
            role=settings.role,
            expected_login_suffix=settings.expected_login_suffix,
            started_utc=observation_started_at,
        )
        deal_rows = [
            broker_deal_row(
                row=row,
                role=settings.role,
                run_id=run_id,
                captured_utc=audit.captured_utc,
                server_time_offset_hours=server_time_offset_hours,
            )
            for row in audit.deals
        ]
        order_rows = [
            broker_order_row(
                row=row,
                role=settings.role,
                run_id=run_id,
                captured_utc=audit.captured_utc,
                server_time_offset_hours=server_time_offset_hours,
            )
            for row in audit.orders
        ]
        total_deals_added += append_unique_rows(
            settings.paths.broker_deals_csv,
            deal_rows,
            fieldnames=BROKER_DEAL_FIELDS,
            key_field="history_key",
        )
        total_orders_added += append_unique_rows(
            settings.paths.broker_orders_csv,
            order_rows,
            fieldnames=BROKER_ORDER_FIELDS,
            key_field="history_key",
        )
        last_found_deals = {
            int(row["ticket"])
            for row in deal_rows
            if row.get("ticket") not in (None, "")
        }
        last_found_orders = {
            int(row["ticket"])
            for row in order_rows
            if row.get("ticket") not in (None, "")
        }
        if expected_deals.issubset(last_found_deals) and expected_orders.issubset(
            last_found_orders
        ):
            return total_deals_added, total_orders_added
        if attempt < attempts:
            time.sleep(max(0.0, float(retry_sleep_seconds)))

    missing_deals = sorted(expected_deals - last_found_deals)
    missing_orders = sorted(expected_orders - last_found_orders)
    raise DualLiveWorkerError(
        "Broker history did not expose all expected tickets after "
        f"{attempts} attempts; missing_deals={missing_deals}, "
        f"missing_orders={missing_orders}"
    )


def _execution_ticket_sets(execution: Any) -> tuple[set[int], set[int]]:
    order_tickets = {
        int(leg.order_ticket)
        for leg in execution.legs
        if leg.order_ticket not in (None, 0)
    }
    deal_tickets = {
        int(leg.deal_ticket)
        for leg in execution.legs
        if leg.deal_ticket not in (None, 0)
    }
    return order_tickets, deal_tickets


def _print_poll_status(
    settings: WorkerSettings,
    *,
    iteration: int,
    state: DualLiveState,
    decision: StrategyDecision,
    broker_inspection: Any,
) -> None:
    print(
        " | ".join(
            (
                utc_now_iso(),
                f"role={settings.role}",
                f"iteration={iteration}",
                f"event={decision.event_time_utc}",
                f"action={decision.action}",
                f"position={state.virtual_position}",
                f"broker_position={state.broker_position}",
                f"equity={broker_inspection.account.get('equity')}",
                f"spread={spread_points_from_tick(broker_inspection.tick, broker_inspection.symbol_info)}",
                f"reconciliation={state.reconciliation_status}",
            )
        ),
        flush=True,
    )


def _heartbeat(
    settings: WorkerSettings,
    *,
    state: DualLiveState,
    run_id: str,
    started_utc: str,
    status: str,
    message: str,
    last_decision: StrategyDecision | None,
) -> None:
    write_json_atomic(
        settings.paths.heartbeat,
        heartbeat_payload(
            state=state,
            run_id=run_id,
            pid=os.getpid(),
            status=status,
            message=message,
            last_decision=last_decision,
            started_utc=started_utc,
            orders_enabled=settings.orders_enabled,
        ),
    )


def _report(
    *,
    settings: WorkerSettings,
    run_id: str,
    started_utc: str,
    state: DualLiveState,
    decisions: list[StrategyDecision],
    status: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "role": settings.role,
        "status": status,
        "formal_gate": status == "PASS",
        "started_utc": started_utc,
        "completed_utc": utc_now_iso(),
        "execution_mode": settings.execution_mode,
        "orders_enabled": settings.orders_enabled,
        "terminal_path": settings.terminal_path,
        "expected_login_suffix_masked": f"****{settings.expected_login_suffix}",
        "state": asdict(state),
        "summary": summarise_decisions(decisions),
        "error": error,
        "safety": {
            "completed_m15_only": True,
            "duplicate_event_policy": "skip",
            "order_check_before_every_order_send": True,
            "model_b_short_prohibited": settings.role == "model_b",
            "account_suffix_checked_every_iteration": True,
            "broker_reconciliation_before_decision": True,
            "telemetry_append_only_every_poll": True,
            "completed_m15_decision_audit": True,
            "order_and_broker_history_audit": True,
            "performance_metrics_calculated_offline": True,
            "market_event_clock_drives_transition_safety": True,
            "canonical_broker_audit_timestamps": True,
            "session_gap_lockout_enabled": (
                settings.session_gap_lockout_enabled
            ),
            "session_gap_lockout_start_utc_minutes": (
                settings.session_gap_lockout_start_utc_minutes
            ),
            "session_gap_lockout_daily_end_utc_minutes": (
                settings.session_gap_lockout_daily_end_utc_minutes
            ),
            "session_gap_lockout_weekend_end_utc_minutes": (
                settings.session_gap_lockout_weekend_end_utc_minutes
            ),
        },
    }


def run_worker(settings: WorkerSettings) -> dict[str, Any]:
    """Run one role until stopped, max iterations reached, or a hard error occurs."""

    prepare_shadow_cpu_environment(allow_onednn=settings.allow_onednn)
    run_id = datetime.now(timezone.utc).strftime(
        f"dual_{settings.role}_%Y%m%dT%H%M%SZ"
    )
    started_utc = utc_now_iso()
    settings.paths.role_root.mkdir(parents=True, exist_ok=True)
    observation_started_at = _load_or_create_observation_start(
        settings,
        run_id=run_id,
        worker_started_utc=started_utc,
    )
    state = load_state(
        settings.paths.state,
        role=settings.role,
        execution_mode=settings.execution_mode,
        worker_pid=os.getpid(),
    )
    decisions: list[StrategyDecision] = []
    last_decision: StrategyDecision | None = None
    _heartbeat(
        settings,
        state=state,
        run_id=run_id,
        started_utc=started_utc,
        status="STARTING",
        message="loading frozen artefacts",
        last_decision=None,
    )
    try:
        _write_runtime_event(
            settings,
            run_id=run_id,
            iteration=0,
            event_type="WORKER_STARTED",
            event_reason="worker_process_started",
            severity="INFO",
            state=state,
        )
        config_a_path = safe_repository_path(
            settings.repo_root,
            settings.model_a_config,
            description="frozen Model A config",
        )
        config_b_path = safe_repository_path(
            settings.repo_root,
            settings.model_b_config,
            description="frozen Model B config",
        )
        shadow_config_path = safe_repository_path(
            settings.repo_root,
            settings.shadow_runtime_config,
            description="shadow runtime config",
        )
        controls_path = safe_repository_path(
            settings.repo_root,
            settings.broker_controls,
            description="frozen broker controls",
        )
        config_a = load_model_a_config(config_a_path)
        config_b_raw = load_yaml_mapping(config_b_path)
        verify_stage0_freeze_manifest(settings.repo_root, str(settings.freeze_manifest))
        bundle = verify_notebook7_artifact_bundle(settings.repo_root, config_a)
        check_runtime_environment(config_a, strict=True)
        scaler, _scaler_report = load_and_validate_scaler(
            bundle.scaler_path,
            config_a,
            bundle.feature_order,
        )
        model, _model_report = load_and_validate_model(bundle.model_path, config_a)
        strategy_rules, risk_rules, rules_a, rules_b = _rules_from_configs(
            config_a,
            config_b_raw,
            settings.role,
        )
        controls = load_frozen_controls(controls_path)
        runtime_config = load_shadow_runtime_config(shadow_config_path)
        if str(controls.timeframe).upper() != str(runtime_config.mt5.timeframe_name).upper():
            raise DualLiveWorkerError(
                "Frozen broker timeframe differs from the shadow runtime timeframe"
            )
        if int(controls.mt5_server_time_offset_hours_current) != int(
            runtime_config.mt5_server_time_offset_hours
        ):
            raise DualLiveWorkerError(
                "Frozen broker-time offset differs from the shadow runtime offset"
            )
        runtime_config = replace(
            runtime_config,
            append_signals=False,
            duplicate_policy="skip",
            mt5=replace(runtime_config.mt5, terminal_path=settings.terminal_path),
        )
        mt5_module = import_metatrader5_module()
        startup_broker = inspect_broker(
            mt5_module=mt5_module,
            terminal_path=settings.terminal_path,
            controls=controls,
            expected_login_suffix=settings.expected_login_suffix,
            require_trading_permissions=settings.orders_enabled,
        )
        _write_telemetry(
            settings,
            run_id=run_id,
            iteration=0,
            snapshot_phase="STARTUP",
            state=state,
            broker_inspection=startup_broker,
            latest_completed_event_time_utc=state.last_event_time_utc,
            latest_decision=None,
            server_time_offset_hours=int(
                runtime_config.mt5_server_time_offset_hours
            ),
        )

        if settings.flatten_only:
            if not settings.orders_enabled:
                raise DualLiveWorkerError("flatten-only requires orders_enabled")
            execution = flatten_position(
                mt5_module=mt5_module,
                terminal_path=settings.terminal_path,
                controls=controls,
                role=settings.role,
                expected_login_suffix=settings.expected_login_suffix,
                event_time_utc=state.last_event_time_utc,
            )
            state = replace(
                state,
                virtual_position=0,
                broker_position=0,
                broker_position_ticket=None,
                broker_position_identifier=None,
                broker_open_order_ticket=None,
                reconciliation_status="EMERGENCY_FLAT_CONFIRMED",
                updated_utc=utc_now_iso(),
                order_send_calls=state.order_send_calls + execution.order_send_calls,
                successful_order_sends=(
                    state.successful_order_sends + execution.successful_order_sends
                ),
            )
            flatten_decision = SimpleNamespace(
                role=settings.role,
                run_id=run_id,
                iteration=0,
                event_time_utc=state.last_event_time_utc,
                action="FLATTEN_ONLY",
                reason="explicit_flatten_only_execution",
                audit_trigger_type="CONTROL_FLATTEN_ONLY",
                audit_trigger_id=(
                    f"CONTROL_FLATTEN_ONLY:{settings.role}:{run_id}:0"
                ),
            )
            if execution.legs:
                append_unique_rows(
                    settings.paths.order_events_csv,
                    execution_audit_rows(
                        execution=execution,
                        decision=flatten_decision,
                    ),
                    fieldnames=ORDER_EVENT_FIELDS,
                    key_field="execution_id",
                )
            expected_orders, expected_deals = _execution_ticket_sets(execution)
            _refresh_broker_history(
                settings,
                mt5_module=mt5_module,
                controls=controls,
                run_id=run_id,
                observation_started_at=observation_started_at,
                expected_order_tickets=expected_orders,
                expected_deal_tickets=expected_deals,
                retries=10,
                server_time_offset_hours=int(
                    runtime_config.mt5_server_time_offset_hours
                ),
            )
            final_broker = inspect_broker(
                mt5_module=mt5_module,
                terminal_path=settings.terminal_path,
                controls=controls,
                expected_login_suffix=settings.expected_login_suffix,
                require_trading_permissions=True,
            )
            _write_telemetry(
                settings,
                run_id=run_id,
                iteration=0,
                snapshot_phase="FLATTEN_ONLY_FINAL",
                state=state,
                broker_inspection=final_broker,
                latest_completed_event_time_utc=state.last_event_time_utc,
                latest_decision=flatten_decision,
                server_time_offset_hours=int(
                    runtime_config.mt5_server_time_offset_hours
                ),
            )
            _write_runtime_event(
                settings,
                run_id=run_id,
                iteration=0,
                event_type="FLATTEN_ONLY_EXECUTION",
                event_reason="explicit_flatten_only_completed",
                severity="WARNING",
                state=state,
                event_time_utc=state.last_event_time_utc,
                details={
                    "order_send_calls": execution.order_send_calls,
                    "successful_order_sends": execution.successful_order_sends,
                },
            )
            write_state(settings.paths.state, state)
            _write_runtime_event(
                settings,
                run_id=run_id,
                iteration=0,
                event_type="WORKER_STOPPED",
                event_reason="flatten_only_completed",
                severity="INFO",
                state=state,
                event_time_utc=state.last_event_time_utc,
            )
            _heartbeat(
                settings,
                state=state,
                run_id=run_id,
                started_utc=started_utc,
                status="STOPPED",
                message="flatten-only completed",
                last_decision=None,
            )
            report = _report(
                settings=settings,
                run_id=run_id,
                started_utc=started_utc,
                state=state,
                decisions=[],
                status="PASS",
                error=None,
            )
            report["flatten_execution"] = asdict(execution)
            write_json_atomic(settings.paths.final_report, report)
            return report

        iteration = 0
        cached_signal: Any | None = None
        cached_bar_context: dict[str, Any] = {}
        cached_signal_context: dict[str, Any] = {}
        cached_snapshot_context: dict[str, Any] = {}
        last_inference_probe_event_time: str | None = None
        last_history_refresh_monotonic = 0.0
        previous_session_gap_lockout: bool | None = None
        while True:
            iteration += 1
            if settings.paths.stop_file.exists():
                break
            if settings.max_iterations and iteration > settings.max_iterations:
                break

            # Poll MT5 every configured interval, but rebuild features and run the
            # CNN-LSTM only when a new completed M15 event is available.  A first
            # inference is deliberately performed after every worker start so the
            # process owns a fresh in-memory signal before restart reconciliation.
            latest_completed_event_time = probe_latest_completed_event_time(
                mt5_module=mt5_module,
                runtime_config=runtime_config,
            )
            inference_performed = bool(
                cached_signal is None
                or latest_completed_event_time
                != last_inference_probe_event_time
            )
            if inference_performed:
                snapshot, rates, signal = run_shadow_once(
                    mt5_module=mt5_module,
                    runtime_config=runtime_config,
                    config_a=config_a,
                    feature_order=bundle.feature_order,
                    model=model,
                    scaler=scaler,
                    rules_a=rules_a,
                    rules_b=rules_b,
                    state_path=settings.paths.shadow_unused_state,
                    signals_csv_path=settings.paths.shadow_unused_signals,
                    mode=settings.execution_mode,
                    run_id=run_id,
                )
                cached_signal = signal
                cached_bar_context = completed_bar_context(rates)
                cached_signal_context = asdict(signal)
                last_inference_probe_event_time = latest_completed_event_time
                snapshot_payload = snapshot_to_dict(snapshot)
                cached_snapshot_context = {
                    "feature_report": snapshot_payload.get("feature_report", {}),
                    "time_normalisation": snapshot_payload.get(
                        "time_normalisation", {}
                    ),
                    "rates": snapshot_payload.get("rates", {}),
                    "strategy_rules": asdict(strategy_rules),
                    "risk_rules": asdict(risk_rules),
                }
                selected_symbol = str(
                    snapshot_payload.get("symbol_resolution", {}).get(
                        "selected_symbol", ""
                    )
                )
                if selected_symbol.upper() != str(controls.symbol).upper():
                    raise DualLiveWorkerError(
                        "Inference symbol differs from the frozen execution symbol: "
                        f"inference={selected_symbol!r}, execution={controls.symbol!r}"
                    )
                write_json_atomic(
                    settings.paths.latest_shadow_snapshot,
                    snapshot_payload,
                )
            else:
                signal = cached_signal

            if signal is None:  # Defensive; inference above must populate it.
                raise DualLiveWorkerError("No completed-M15 signal is available")

            # Broker and account risk are inspected on every poll, including
            # between M15 closes.  This lets kill switches and drawdown stops
            # flatten immediately without turning each 30-second poll into a new
            # model event or a duplicate decision-log row.
            broker_inspection = inspect_broker(
                mt5_module=mt5_module,
                terminal_path=settings.terminal_path,
                controls=controls,
                expected_login_suffix=settings.expected_login_suffix,
                require_trading_permissions=settings.orders_enabled,
            )
            # Reset UTC-day counters and update risk first.  Reconciliation
            # then conservatively counts any broker transition adopted after a
            # crash, so a day reset cannot erase that recovered entry/change.
            risk_update = update_risk_state(
                state,
                equity=broker_inspection.snapshot.account_equity,
                event_time_utc=latest_completed_event_time,
                rules=risk_rules,
                kill_switch_active=settings.paths.kill_switch_file.exists(),
            )
            state = risk_update.state
            if risk_update.daily_stop_triggered_now:
                _write_runtime_event(
                    settings,
                    run_id=run_id,
                    iteration=iteration,
                    event_type="DAILY_STOP_TRIGGERED",
                    event_reason="live_account_daily_loss_threshold_reached",
                    severity="WARNING",
                    state=state,
                    event_time_utc=latest_completed_event_time,
                )
            if risk_update.total_stop_triggered_now:
                _write_runtime_event(
                    settings,
                    run_id=run_id,
                    iteration=iteration,
                    event_type="TOTAL_STOP_TRIGGERED",
                    event_reason="live_account_total_drawdown_threshold_reached",
                    severity="CRITICAL",
                    state=state,
                    event_time_utc=latest_completed_event_time,
                )
            reconciliation = reconcile_state(
                state,
                broker_inspection.snapshot,
                expected_magic=(
                    controls.model_a_magic_number
                    if settings.role == "model_a"
                    else controls.model_b_magic_number
                ),
                execution_mode=settings.execution_mode,
            )
            state = reconciliation.state
            if reconciliation.incident:
                _write_runtime_event(
                    settings,
                    run_id=run_id,
                    iteration=iteration,
                    event_type="RECONCILIATION_INCIDENT",
                    event_reason=reconciliation.reason,
                    severity="WARNING",
                    state=state,
                    event_time_utc=latest_completed_event_time,
                    details={"status": reconciliation.status},
                )
            state_before_decision = replace(state)
            broker_before = broker_inspection
            execution = None
            session_lockout_active, session_lockout_reason = (
                session_gap_lockout_status(
                    datetime.now(timezone.utc),
                    enabled=settings.session_gap_lockout_enabled,
                    start_utc_minutes=(
                        settings.session_gap_lockout_start_utc_minutes
                    ),
                    daily_end_utc_minutes=(
                        settings.session_gap_lockout_daily_end_utc_minutes
                    ),
                    weekend_end_utc_minutes=(
                        settings.session_gap_lockout_weekend_end_utc_minutes
                    ),
                )
            )
            if (
                previous_session_gap_lockout is None
                and session_lockout_active
            ) or (
                previous_session_gap_lockout is not None
                and previous_session_gap_lockout != session_lockout_active
            ):
                _write_runtime_event(
                    settings,
                    run_id=run_id,
                    iteration=iteration,
                    event_type=(
                        "SESSION_GAP_LOCKOUT_STARTED"
                        if session_lockout_active
                        else "SESSION_GAP_LOCKOUT_ENDED"
                    ),
                    event_reason=session_lockout_reason,
                    severity="WARNING" if session_lockout_active else "INFO",
                    state=state,
                    event_time_utc=latest_completed_event_time,
                    details={
                        "pre_break_flatten_utc_minutes": (
                            settings.session_gap_lockout_start_utc_minutes
                        ),
                        "daily_reopen_utc_minutes": (
                            settings.session_gap_lockout_daily_end_utc_minutes
                        ),
                        "weekend_reopen_utc_minutes": (
                            settings.session_gap_lockout_weekend_end_utc_minutes
                        ),
                    },
                )
            previous_session_gap_lockout = session_lockout_active
            decision = decide_strategy_transition(
                state,
                rules=strategy_rules,
                run_id=run_id,
                iteration=iteration,
                # Transition safety follows the broker's latest completed M15
                # event, not the latest valid model-sequence endpoint.  Around
                # session/weekend gaps the model signal can remain anchored to
                # an older contiguous window for up to 48 bars; using that stale
                # timestamp would misclassify every new broker bar as a duplicate
                # and delay mandatory gap flattening.
                event_time_utc=latest_completed_event_time,
                # A probability whose endpoint is older than the current broker
                # event is not a current prediction.  Keep the original signal
                # context for diagnosis, but pass no probability into the
                # transition/audit row while the frozen contiguous sequence is
                # unavailable.
                probability_up=(
                    None
                    if bool(signal.stale_event_warning)
                    else float(signal.probability_up)
                ),
                stale_event_warning=bool(signal.stale_event_warning),
                reconciliation_blocked=reconciliation.blocked,
                reconciliation_reason=reconciliation.reason,
                session_gap_lockout_active=session_lockout_active,
                session_gap_lockout_reason=session_lockout_reason,
            )
            if decision.duplicate_event:
                # Duplicate M15 events do not advance hold/cooldown counters and
                # are not appended to the permanent decision log.  Updated risk
                # and reconciliation state is still persisted below.
                pass
            elif settings.orders_enabled and (
                decision.target_position != decision.position_before
            ):
                try:
                    execution = execute_transition(
                        mt5_module=mt5_module,
                        terminal_path=settings.terminal_path,
                        controls=controls,
                        role=settings.role,
                        expected_login_suffix=settings.expected_login_suffix,
                        target_position=decision.target_position,
                        event_time_utc=decision.event_time_utc,
                    )
                except EntrySpreadBlocked:
                    decision = replace(
                        decision,
                        target_position=decision.position_before,
                        action="BLOCK_SPREAD",
                        reason="spread_above_frozen_entry_gate",
                    )
                    state = advance_blocked_or_hold_state(state, decision)
                else:
                    if not execution.completed_target:
                        decision = replace(
                            decision,
                            target_position=execution.broker_position_after,
                            action="PARTIAL_REVERSAL_FLAT",
                            reason=(
                                "frozen_overlay_transition_partial_flat_due_to_spread"
                            ),
                        )
                    decision = update_decision_execution(
                        decision,
                        order_check_called=execution.order_check_called,
                        order_check_passed=True,
                        order_send_called=execution.order_send_called,
                        order_send_passed=(
                            execution.successful_order_sends
                            == execution.order_send_calls
                        ),
                        broker_position_after=execution.broker_position_after,
                        broker_position_ticket_after=(
                            execution.broker_position_ticket_after
                        ),
                    )
                    state = apply_transition(
                        state,
                        decision,
                        confirmed_position=execution.broker_position_after,
                        broker_ticket=execution.broker_position_ticket_after,
                        order_send_calls=execution.order_send_calls,
                        successful_order_sends=execution.successful_order_sends,
                    )
            else:
                # Shadow mode applies the exact frozen virtual transition.  Live
                # mode with no transition simply advances the held-state counters.
                confirmed = (
                    decision.target_position
                    if settings.execution_mode == "shadow"
                    else decision.position_before
                )
                state = apply_transition(
                    state,
                    decision,
                    confirmed_position=confirmed,
                    broker_ticket=state.broker_position_ticket,
                )

            if execution is not None:
                append_unique_rows(
                    settings.paths.order_events_csv,
                    execution_audit_rows(execution=execution, decision=decision),
                    fieldnames=ORDER_EVENT_FIELDS,
                    key_field="execution_id",
                )
                _write_runtime_event(
                    settings,
                    run_id=run_id,
                    iteration=iteration,
                    event_type="ORDER_EXECUTION",
                    event_reason=decision.action,
                    severity="INFO",
                    state=state,
                    event_time_utc=decision.event_time_utc,
                    details={
                        "order_send_calls": execution.order_send_calls,
                        "successful_order_sends": execution.successful_order_sends,
                        "completed_target": execution.completed_target,
                        "partial_reason": execution.partial_reason,
                    },
                )
                broker_after = inspect_broker(
                    mt5_module=mt5_module,
                    terminal_path=settings.terminal_path,
                    controls=controls,
                    expected_login_suffix=settings.expected_login_suffix,
                    require_trading_permissions=settings.orders_enabled,
                )
            else:
                broker_after = broker_before

            last_decision = decision
            write_state(settings.paths.state, state)
            if not decision.duplicate_event:
                decisions.append(decision)
                append_csv_row(
                    settings.paths.decisions_csv,
                    decision_audit_row(
                        decision=decision,
                        state_before=state_before_decision,
                        state_after=state,
                        bar_context=cached_bar_context,
                        signal_context=cached_signal_context,
                        snapshot_context=cached_snapshot_context,
                        broker_before=broker_before,
                        broker_after=broker_after,
                    ),
                    fieldnames=DECISION_FIELDS,
                )
                write_json_atomic(
                    settings.paths.latest_decision,
                    decision_to_mapping(decision),
                )

            _write_telemetry(
                settings,
                run_id=run_id,
                iteration=iteration,
                snapshot_phase="POLL",
                state=state,
                broker_inspection=broker_after,
                latest_completed_event_time_utc=latest_completed_event_time,
                latest_decision=decision,
                server_time_offset_hours=int(
                    runtime_config.mt5_server_time_offset_hours
                ),
            )

            history_due = (
                execution is not None
                or time.monotonic() - last_history_refresh_monotonic
                >= settings.history_refresh_seconds
            )
            if history_due:
                expected_orders: set[int] = set()
                expected_deals: set[int] = set()
                if execution is not None:
                    expected_orders, expected_deals = _execution_ticket_sets(
                        execution
                    )
                try:
                    deals_added, orders_added = _refresh_broker_history(
                        settings,
                        mt5_module=mt5_module,
                        controls=controls,
                        run_id=run_id,
                        observation_started_at=observation_started_at,
                        expected_order_tickets=expected_orders,
                        expected_deal_tickets=expected_deals,
                        retries=(6 if execution is not None else 1),
                        server_time_offset_hours=int(
                            runtime_config.mt5_server_time_offset_hours
                        ),
                    )
                except Exception as history_exc:
                    # Back off to the configured refresh cadence after a failed
                    # read.  Immediate post-order reads already retry several
                    # times, so polling again every 30 seconds would only add
                    # broker/API load and repetitive warning rows.
                    last_history_refresh_monotonic = time.monotonic()
                    _write_runtime_event(
                        settings,
                        run_id=run_id,
                        iteration=iteration,
                        event_type="BROKER_HISTORY_REFRESH_FAILED",
                        event_reason=str(history_exc),
                        severity="WARNING",
                        state=state,
                        event_time_utc=decision.event_time_utc,
                    )
                else:
                    last_history_refresh_monotonic = time.monotonic()
                    if deals_added or orders_added:
                        _write_runtime_event(
                            settings,
                            run_id=run_id,
                            iteration=iteration,
                            event_type="BROKER_HISTORY_REFRESHED",
                            event_reason="new_broker_history_rows_persisted",
                            severity="INFO",
                            state=state,
                            event_time_utc=decision.event_time_utc,
                            details={
                                "deals_added": deals_added,
                                "orders_added": orders_added,
                            },
                        )
            _print_poll_status(
                settings,
                iteration=iteration,
                state=state,
                decision=decision,
                broker_inspection=broker_after,
            )
            message = (
                f"iteration {iteration}: completed M15 event processed"
                if inference_performed and not decision.duplicate_event
                else f"iteration {iteration}: broker/risk heartbeat completed"
            )
            _heartbeat(
                settings,
                state=state,
                run_id=run_id,
                started_utc=started_utc,
                status="RUNNING",
                message=message,
                last_decision=decision,
            )
            if settings.paths.stop_file.exists():
                break
            time.sleep(settings.poll_seconds)

        if settings.flatten_on_clean_stop and settings.orders_enabled:
            # Always inspect and flatten the broker on a clean stop, even when
            # persistent state says FLAT.  This closes the crash window where an
            # order reached MT5 but the worker died before state.json was updated.
            execution = flatten_position(
                mt5_module=mt5_module,
                terminal_path=settings.terminal_path,
                controls=controls,
                role=settings.role,
                expected_login_suffix=settings.expected_login_suffix,
                event_time_utc=state.last_event_time_utc,
            )
            state = replace(
                state,
                virtual_position=0,
                broker_position=0,
                broker_position_ticket=None,
                broker_position_identifier=None,
                broker_open_order_ticket=None,
                reconciliation_status="CLEAN_STOP_FLAT_CONFIRMED",
                order_send_calls=state.order_send_calls + execution.order_send_calls,
                successful_order_sends=(
                    state.successful_order_sends + execution.successful_order_sends
                ),
                updated_utc=utc_now_iso(),
            )
            if execution.legs:
                close_decision = SimpleNamespace(
                    role=settings.role,
                    run_id=run_id,
                    iteration=iteration + 1,
                    event_time_utc=state.last_event_time_utc,
                    action="CLEAN_STOP_FLATTEN",
                    reason="clean_stop_control_flatten",
                    audit_trigger_type="CONTROL_CLEAN_STOP",
                    audit_trigger_id=(
                        f"CONTROL_CLEAN_STOP:{settings.role}:{run_id}:"
                        f"{iteration + 1}"
                    ),
                )
                append_unique_rows(
                    settings.paths.order_events_csv,
                    execution_audit_rows(
                        execution=execution,
                        decision=close_decision,
                    ),
                    fieldnames=ORDER_EVENT_FIELDS,
                    key_field="execution_id",
                )
            write_state(settings.paths.state, state)
            final_expected_orders, final_expected_deals = _execution_ticket_sets(
                execution
            )
            try:
                _refresh_broker_history(
                    settings,
                    mt5_module=mt5_module,
                    controls=controls,
                    run_id=run_id,
                    observation_started_at=observation_started_at,
                    expected_order_tickets=final_expected_orders,
                    expected_deal_tickets=final_expected_deals,
                    retries=10,
                    server_time_offset_hours=int(
                        runtime_config.mt5_server_time_offset_hours
                    ),
                )
            except Exception as history_exc:
                _write_runtime_event(
                    settings,
                    run_id=run_id,
                    iteration=iteration,
                    event_type="FINAL_BROKER_HISTORY_REFRESH_FAILED",
                    event_reason=str(history_exc),
                    severity="CRITICAL",
                    state=state,
                    event_time_utc=state.last_event_time_utc,
                )
                raise
        final_broker = inspect_broker(
            mt5_module=mt5_module,
            terminal_path=settings.terminal_path,
            controls=controls,
            expected_login_suffix=settings.expected_login_suffix,
            require_trading_permissions=settings.orders_enabled,
        )
        _write_telemetry(
            settings,
            run_id=run_id,
            iteration=iteration + 1,
            snapshot_phase="FINAL",
            state=state,
            broker_inspection=final_broker,
            latest_completed_event_time_utc=state.last_event_time_utc,
            latest_decision=last_decision,
            server_time_offset_hours=int(
                runtime_config.mt5_server_time_offset_hours
            ),
        )
        _write_runtime_event(
            settings,
            run_id=run_id,
            iteration=iteration,
            event_type="WORKER_STOPPED",
            event_reason="worker_stopped_cleanly",
            severity="INFO",
            state=state,
            event_time_utc=state.last_event_time_utc,
        )
        _heartbeat(
            settings,
            state=state,
            run_id=run_id,
            started_utc=started_utc,
            status="STOPPED",
            message="worker stopped cleanly",
            last_decision=last_decision,
        )
        report = _report(
            settings=settings,
            run_id=run_id,
            started_utc=started_utc,
            state=state,
            decisions=decisions,
            status="PASS",
            error=None,
        )
        write_json_atomic(settings.paths.final_report, report)
        return report
    except Exception as exc:
        error_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        try:
            _write_runtime_event(
                settings,
                run_id=run_id,
                iteration=locals().get("iteration", 0),
                event_type="WORKER_ERROR",
                event_reason=str(exc),
                severity="CRITICAL",
                state=state,
                event_time_utc=state.last_event_time_utc,
                details={"traceback": error_text},
            )
            _heartbeat(
                settings,
                state=state,
                run_id=run_id,
                started_utc=started_utc,
                status="ERROR",
                message=str(exc),
                last_decision=last_decision,
            )
            report = _report(
                settings=settings,
                run_id=run_id,
                started_utc=started_utc,
                state=state,
                decisions=decisions,
                status="FAIL",
                error=error_text,
            )
            write_json_atomic(settings.paths.final_report, report)
        except Exception:
            pass
        raise
