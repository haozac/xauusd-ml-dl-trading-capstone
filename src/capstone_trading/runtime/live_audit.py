"""Append-only raw audit logging for dual-model live observation runs.

This module performs no performance analysis.  It only persists stable,
structured raw evidence so daily and consolidated metrics can be calculated
offline after the observation window.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import csv
import json
import math
import os


class LiveAuditError(RuntimeError):
    """Raised when an audit file is malformed or cannot be appended safely."""


TELEMETRY_FIELDS: tuple[str, ...] = (
    "schema_version", "snapshot_id", "snapshot_utc", "snapshot_phase",
    "role", "run_id", "iteration", "worker_pid", "execution_mode",
    "orders_enabled", "full_context_captured",
    "terminal_connected", "terminal_trade_allowed", "tradeapi_disabled",
    "account_login_masked", "account_currency", "account_leverage",
    "account_company", "account_server", "account_trade_allowed",
    "account_expert_allowed", "symbol", "latest_completed_event_time_utc",
    "latest_decision_event_time_utc", "seconds_since_latest_completed_event",
    "tick_time_utc", "tick_time_msc", "bid", "ask", "last",
    "spread_points", "symbol_reported_spread_points", "point",
    "balance", "equity", "floating_profit",
    "margin", "free_margin", "margin_level", "broker_position",
    "position_ticket", "position_identifier", "position_order_ticket",
    "position_side", "position_volume", "position_open_time_utc",
    "position_open_price", "position_current_price", "position_profit",
    "position_swap", "position_magic", "position_comment",
    "pending_order_count", "account_json", "terminal_json",
    "symbol_info_json", "tick_json", "capital_review_json",
    "positions_json", "pending_orders_json",
    "virtual_position", "hold_bars", "flat_bars_since_exit",
    "policy_changes_today", "successful_entries_today", "daily_return",
    "total_drawdown", "daily_stop_active", "total_stop_active",
    "kill_switch_active", "reconciliation_status",
    "reconciliation_incidents", "stop_file_exists",
    "kill_switch_file_exists", "last_action", "last_reason",
)

DECISION_FIELDS: tuple[str, ...] = (
    "schema_version", "decision_id", "role", "run_id", "iteration",
    "event_time_utc", "decision_utc", "execution_mode", "probability_up",
    "model_prediction_available", "model_prediction_event_time_utc",
    "model_unavailable_reason", "broker_event_disposition",
    "signal_window_start_utc", "signal_window_end_utc",
    "latest_completed_bar_time_utc", "selected_symbol",
    "model_a_signal", "model_b_from_flat_signal",
    "model_b_hold_condition", "model_b_entry_condition",
    "sequence_length", "feature_count", "valid_sequence_count",
    "event_is_latest_feature", "event_is_latest_completed_bar",
    "feature_report_json", "time_normalisation_json", "rates_report_json",
    "strategy_rules_json", "risk_rules_json",
    "m15_open", "m15_high", "m15_low", "m15_close", "m15_tick_volume",
    "m15_spread", "m15_real_volume", "position_before",
    "position_before_name", "desired_position", "desired_position_name",
    "target_position", "target_position_name", "action", "reason",
    "duplicate_event", "gap_from_previous_event", "stale_event_warning",
    "requested_policy_event_units", "policy_event_units",
    "policy_changes_today_before", "policy_changes_today_after",
    "successful_entries_today_before", "successful_entries_today_after",
    "policy_cap_reached", "entry_blocked_by_policy_cap",
    "exit_allowed_when_capped", "close_only_reversal", "hold_bars_before",
    "hold_bars_after", "flat_bars_since_exit_before",
    "flat_bars_since_exit_after", "current_utc_date_before",
    "current_utc_date_after", "day_start_equity_before",
    "day_start_equity_after", "peak_equity_before", "peak_equity_after",
    "daily_return_before", "daily_return_after", "total_drawdown_before",
    "total_drawdown_after", "daily_stop_active_before",
    "daily_stop_active_after", "total_stop_active_before",
    "total_stop_active_after", "kill_switch_active_before",
    "kill_switch_active_after", "reconciliation_status_before",
    "reconciliation_status_after", "reconciliation_incidents_before",
    "reconciliation_incidents_after", "daily_stop_active",
    "total_stop_active", "kill_switch_active", "reconciliation_status",
    "bid_at_decision",
    "ask_at_decision", "spread_points_at_decision",
    "symbol_reported_spread_points_at_decision", "balance_before",
    "equity_before", "floating_profit_before", "broker_position_before",
    "broker_position_ticket_before", "broker_position_open_price_before",
    "broker_position_current_price_before", "broker_position_profit_before",
    "balance_after", "equity_after", "floating_profit_after",
    "broker_position_after_inspection", "broker_position_ticket_after_inspection",
    "broker_position_open_price_after", "broker_position_current_price_after",
    "broker_position_profit_after", "order_check_called",
    "order_check_passed", "order_send_called", "order_send_passed",
    "broker_position_after", "broker_position_ticket_after",
    "broker_before_json", "broker_after_json",
)

ORDER_EVENT_FIELDS: tuple[str, ...] = (
    "schema_version", "execution_id", "decision_id", "role", "run_id",
    "trigger_type", "event_time_utc", "completed_utc", "leg_index", "purpose", "side",
    "position_before", "position_after", "requested_volume",
    "requested_price", "bid_before", "ask_before", "spread_points_before",
    "symbol_reported_spread_points_before", "symbol_point", "order_check_passed", "order_check_retcode",
    "order_check_comment", "order_send_passed", "order_send_retcode",
    "order_send_comment", "broker_result_price", "slippage_points_signed",
    "slippage_points_adverse", "margin_required_account_currency",
    "order_ticket", "deal_ticket", "position_ticket", "request_position_ticket",
    "magic", "comment", "filling_name", "filling_value", "request_json",
    "check_result_json", "send_result_json", "last_error_json",
)

RUNTIME_EVENT_FIELDS: tuple[str, ...] = (
    "schema_version", "runtime_event_id", "timestamp_utc", "role", "run_id",
    "iteration", "worker_pid", "event_type", "event_reason", "severity",
    "event_time_utc", "virtual_position", "broker_position",
    "reconciliation_status", "restart_count", "details_json",
)

BROKER_DEAL_FIELDS: tuple[str, ...] = (
    "schema_version", "history_key", "captured_utc", "role", "run_id",
    "ticket", "order", "position_id", "time", "time_msc", "time_utc",
    "symbol", "type", "entry", "reason", "magic", "volume", "price",
    "commission", "swap", "profit", "fee", "comment", "external_id",
    "raw_json",
)

BROKER_ORDER_FIELDS: tuple[str, ...] = (
    "schema_version", "history_key", "captured_utc", "role", "run_id",
    "ticket", "position_id", "position_by_id", "time_setup",
    "time_setup_msc", "time_setup_utc", "time_done", "time_done_msc",
    "time_done_utc", "symbol", "type", "state", "reason", "magic",
    "volume_initial", "volume_current", "price_open", "price_current",
    "price_stoplimit", "sl", "tp", "comment", "external_id", "raw_json",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if is_dataclass(value):
        return json.dumps(asdict(value), sort_keys=True, default=str)
    if isinstance(value, Mapping) or isinstance(value, (list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _normalise_row(row: Mapping[str, Any], fieldnames: Sequence[str]) -> dict[str, Any]:
    unknown = sorted(set(row) - set(fieldnames))
    if unknown:
        raise LiveAuditError(f"Audit row contains unknown fields: {unknown}")
    return {name: _normalise_scalar(row.get(name)) for name in fieldnames}


def append_csv_row(
    path: Path,
    row: Mapping[str, Any],
    *,
    fieldnames: Sequence[str],
) -> None:
    """Append one fixed-schema row and force it to disk.

    Each worker is the sole writer for its role directory, so an append-only
    writer is both faster and safer than rewriting the whole CSV every poll.
    The header is validated on restart to prevent silent schema drift.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(str(item) for item in fieldnames)
    if not fields or len(fields) != len(set(fields)):
        raise LiveAuditError("Audit fieldnames must be non-empty and unique")
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            existing = tuple(next(csv.reader(handle), ()))
        if existing != fields:
            raise LiveAuditError(
                f"Audit schema mismatch for {path}: expected {fields}, found {existing}"
            )
    normalised = _normalise_row(row, fields)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow(normalised)
        handle.flush()
        os.fsync(handle.fileno())


def existing_keys(path: Path, key_field: str) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if key_field not in (reader.fieldnames or []):
            raise LiveAuditError(f"{path} is missing key field {key_field!r}")
        return {str(row.get(key_field, "")) for row in reader if row.get(key_field)}


def append_unique_rows(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    key_field: str,
) -> int:
    seen = existing_keys(path, key_field)
    appended = 0
    for row in rows:
        key = str(row.get(key_field, ""))
        if not key:
            raise LiveAuditError(f"Unique audit row is missing {key_field!r}")
        if key in seen:
            continue
        append_csv_row(path, row, fieldnames=fieldnames)
        seen.add(key)
        appended += 1
    return appended


def iso_from_epoch(
    seconds: Any = None,
    milliseconds: Any = None,
    *,
    server_time_offset_hours: int = 0,
) -> str | None:
    """Convert an MT5 epoch-like value into canonical UTC.

    Dukascopy MT5 exposes bar, tick, position, order, and deal timestamps using
    its broker-server clock while the Python package presents the raw numeric
    value as an epoch.  The raw number is retained elsewhere in every audit
    row; this helper only corrects derived ``*_utc`` labels by subtracting the
    frozen broker-server offset.
    """

    try:
        if milliseconds not in (None, ""):
            value = float(milliseconds) / 1000.0
        elif seconds not in (None, ""):
            value = float(seconds)
        else:
            return None
        converted = datetime.fromtimestamp(value, tz=timezone.utc) - timedelta(
            hours=int(server_time_offset_hours)
        )
        return converted.isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def history_key(kind: str, row: Mapping[str, Any]) -> str:
    if kind == "deal":
        identity = row.get("ticket") or row.get("deal")
        fallback = (row.get("order"), row.get("position_id"), row.get("time_msc"), row.get("type"))
    elif kind == "order":
        identity = row.get("ticket") or row.get("order")
        fallback = (row.get("position_id"), row.get("time_setup_msc"), row.get("type"), row.get("state"))
    else:
        raise LiveAuditError(f"Unsupported history kind: {kind!r}")
    return f"{kind}:{identity if identity not in (None, '') else json.dumps(fallback, default=str)}"


def broker_deal_row(
    *,
    row: Mapping[str, Any],
    role: str,
    run_id: str,
    captured_utc: str,
    server_time_offset_hours: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "history_key": history_key("deal", row),
        "captured_utc": captured_utc, "role": role, "run_id": run_id,
        "ticket": row.get("ticket") or row.get("deal"), "order": row.get("order"),
        "position_id": row.get("position_id"), "time": row.get("time"),
        "time_msc": row.get("time_msc"),
        "time_utc": iso_from_epoch(
            row.get("time"),
            row.get("time_msc"),
            server_time_offset_hours=server_time_offset_hours,
        ),
        "symbol": row.get("symbol"), "type": row.get("type"),
        "entry": row.get("entry"), "reason": row.get("reason"),
        "magic": row.get("magic"), "volume": row.get("volume"),
        "price": row.get("price"), "commission": row.get("commission"),
        "swap": row.get("swap"), "profit": row.get("profit"), "fee": row.get("fee"),
        "comment": row.get("comment"), "external_id": row.get("external_id"),
        "raw_json": dict(row),
    }


def broker_order_row(
    *,
    row: Mapping[str, Any],
    role: str,
    run_id: str,
    captured_utc: str,
    server_time_offset_hours: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "history_key": history_key("order", row),
        "captured_utc": captured_utc, "role": role, "run_id": run_id,
        "ticket": row.get("ticket") or row.get("order"),
        "position_id": row.get("position_id"), "position_by_id": row.get("position_by_id"),
        "time_setup": row.get("time_setup"), "time_setup_msc": row.get("time_setup_msc"),
        "time_setup_utc": iso_from_epoch(
            row.get("time_setup"),
            row.get("time_setup_msc"),
            server_time_offset_hours=server_time_offset_hours,
        ),
        "time_done": row.get("time_done"), "time_done_msc": row.get("time_done_msc"),
        "time_done_utc": iso_from_epoch(
            row.get("time_done"),
            row.get("time_done_msc"),
            server_time_offset_hours=server_time_offset_hours,
        ),
        "symbol": row.get("symbol"), "type": row.get("type"), "state": row.get("state"),
        "reason": row.get("reason"), "magic": row.get("magic"),
        "volume_initial": row.get("volume_initial"), "volume_current": row.get("volume_current"),
        "price_open": row.get("price_open"), "price_current": row.get("price_current"),
        "price_stoplimit": row.get("price_stoplimit"), "sl": row.get("sl"), "tp": row.get("tp"),
        "comment": row.get("comment"), "external_id": row.get("external_id"),
        "raw_json": dict(row),
    }


def _value(mapping: Mapping[str, Any] | None, key: str) -> Any:
    return None if mapping is None else mapping.get(key)


def completed_bar_context(rates: Any) -> dict[str, Any]:
    """Return the latest completed M15 OHLCV row from a pandas-like frame."""

    if rates is None or len(rates) == 0:
        return {}
    latest = rates.iloc[-1]
    return {
        "event_time_utc": str(latest.get("time")),
        "m15_open": latest.get("open"),
        "m15_high": latest.get("high"),
        "m15_low": latest.get("low"),
        "m15_close": latest.get("close"),
        "m15_tick_volume": latest.get("tick_volume"),
        "m15_spread": latest.get("spread"),
        "m15_real_volume": latest.get("real_volume"),
    }


def decision_identifier(
    role: str,
    event_time_utc: str | None,
    *,
    run_id: str | None = None,
    iteration: int | None = None,
) -> str:
    event = str(event_time_utc or "NO_EVENT").replace(":", "").replace("+", "_")
    return f"{role}:{run_id or 'NO_RUN'}:{iteration if iteration is not None else 'NA'}:{event}"


def spread_points_from_tick(
    tick: Mapping[str, Any],
    symbol_info: Mapping[str, Any],
) -> float | int | None:
    """Return the observed bid/ask spread in points, with MT5 fallback."""

    try:
        bid = float(tick.get("bid"))
        ask = float(tick.get("ask"))
        point = float(symbol_info.get("point"))
        if all(math.isfinite(value) for value in (bid, ask, point)) and point > 0.0:
            return (ask - bid) / point
    except (TypeError, ValueError):
        pass
    return symbol_info.get("spread")


def broker_context_payload(inspection: Any) -> dict[str, Any]:
    """Preserve complete read-only MT5 inspection context for offline audit."""

    return {
        "snapshot": asdict(inspection.snapshot),
        "package": dict(inspection.package),
        "terminal": dict(inspection.terminal),
        "account": dict(inspection.account),
        "symbol_info": dict(inspection.symbol_info),
        "tick": dict(inspection.tick),
        "positions": [dict(item) for item in inspection.positions_raw],
        "pending_orders": [dict(item) for item in inspection.pending_orders_raw],
        "capital_review": dict(inspection.capital_review),
        "mt5_calls": list(inspection.mt5_calls),
        "forbidden_attempts": list(inspection.forbidden_attempts),
        "shutdown_called": bool(inspection.shutdown_called),
    }


def telemetry_audit_row(
    *,
    role: str,
    run_id: str,
    iteration: int,
    worker_pid: int,
    snapshot_phase: str,
    execution_mode: str,
    orders_enabled: bool,
    latest_completed_event_time_utc: str | None,
    latest_decision: Any,
    state: Any,
    broker_inspection: Any,
    stop_file_exists: bool,
    kill_switch_file_exists: bool,
    server_time_offset_hours: int = 0,
) -> dict[str, Any]:
    captured = datetime.now(timezone.utc)
    latest_event = None
    try:
        if latest_completed_event_time_utc:
            latest_event = datetime.fromisoformat(
                str(latest_completed_event_time_utc).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
    except ValueError:
        latest_event = None
    position = (
        dict(broker_inspection.positions_raw[0])
        if broker_inspection.positions_raw
        else {}
    )
    account = broker_inspection.account
    tick = broker_inspection.tick
    symbol_info = broker_inspection.symbol_info
    snapshot = broker_inspection.snapshot
    broker_position = snapshot.positions[0].position if snapshot.positions else 0
    observed_spread = spread_points_from_tick(tick, symbol_info)
    full_context = str(snapshot_phase).upper() in {
        "STARTUP", "FINAL", "FLATTEN_ONLY_FINAL"
    }
    return {
        "schema_version": "1.0",
        "snapshot_id": f"{role}:{run_id}:{iteration}:{captured.isoformat()}",
        "snapshot_utc": captured.isoformat(),
        "snapshot_phase": str(snapshot_phase).upper(),
        "role": role,
        "run_id": run_id,
        "iteration": iteration,
        "worker_pid": worker_pid,
        "execution_mode": execution_mode,
        "orders_enabled": orders_enabled,
        "full_context_captured": full_context,
        "terminal_connected": snapshot.connected,
        "terminal_trade_allowed": snapshot.terminal_trade_allowed,
        "tradeapi_disabled": snapshot.trade_api_disabled,
        "account_login_masked": snapshot.account_login_masked,
        "account_currency": account.get("currency"),
        "account_leverage": account.get("leverage"),
        "account_company": account.get("company"),
        "account_server": account.get("server"),
        "account_trade_allowed": snapshot.account_trade_allowed,
        "account_expert_allowed": snapshot.account_expert_allowed,
        "symbol": snapshot.symbol,
        "latest_completed_event_time_utc": latest_completed_event_time_utc,
        "latest_decision_event_time_utc": getattr(latest_decision, "event_time_utc", None),
        "seconds_since_latest_completed_event": (
            None if latest_event is None else (captured - latest_event).total_seconds()
        ),
        "tick_time_utc": iso_from_epoch(
            tick.get("time"),
            tick.get("time_msc"),
            server_time_offset_hours=server_time_offset_hours,
        ),
        "tick_time_msc": tick.get("time_msc"),
        "bid": tick.get("bid"),
        "ask": tick.get("ask"),
        "last": tick.get("last"),
        "spread_points": observed_spread,
        "symbol_reported_spread_points": symbol_info.get("spread"),
        "point": symbol_info.get("point"),
        "balance": account.get("balance"),
        "equity": account.get("equity"),
        "floating_profit": account.get("profit"),
        "margin": account.get("margin"),
        "free_margin": account.get("margin_free"),
        "margin_level": account.get("margin_level"),
        "broker_position": broker_position,
        "position_ticket": position.get("ticket"),
        "position_identifier": position.get("identifier"),
        "position_order_ticket": position.get("order"),
        "position_side": position.get("type"),
        "position_volume": position.get("volume"),
        "position_open_time_utc": iso_from_epoch(
            position.get("time"),
            position.get("time_msc"),
            server_time_offset_hours=server_time_offset_hours,
        ),
        "position_open_price": position.get("price_open"),
        "position_current_price": position.get("price_current"),
        "position_profit": position.get("profit"),
        "position_swap": position.get("swap"),
        "position_magic": position.get("magic"),
        "position_comment": position.get("comment"),
        "pending_order_count": snapshot.pending_order_count,
        "account_json": dict(account) if full_context else None,
        "terminal_json": (
            dict(broker_inspection.terminal) if full_context else None
        ),
        "symbol_info_json": dict(symbol_info) if full_context else None,
        "tick_json": dict(tick),
        "capital_review_json": (
            dict(broker_inspection.capital_review) if full_context else None
        ),
        "positions_json": [dict(item) for item in broker_inspection.positions_raw],
        "pending_orders_json": [
            dict(item) for item in broker_inspection.pending_orders_raw
        ],
        "virtual_position": state.virtual_position,
        "hold_bars": state.hold_bars,
        "flat_bars_since_exit": state.flat_bars_since_exit,
        "policy_changes_today": state.policy_changes_today,
        "successful_entries_today": state.successful_entries_today,
        "daily_return": state.daily_return,
        "total_drawdown": state.total_drawdown,
        "daily_stop_active": state.daily_stop_active,
        "total_stop_active": state.total_stop_active,
        "kill_switch_active": state.kill_switch_active,
        "reconciliation_status": state.reconciliation_status,
        "reconciliation_incidents": state.reconciliation_incidents,
        "stop_file_exists": stop_file_exists,
        "kill_switch_file_exists": kill_switch_file_exists,
        "last_action": getattr(latest_decision, "action", None),
        "last_reason": getattr(latest_decision, "reason", None),
    }


def decision_audit_row(
    *,
    decision: Any,
    state_before: Any,
    state_after: Any,
    bar_context: Mapping[str, Any],
    signal_context: Mapping[str, Any],
    snapshot_context: Mapping[str, Any],
    broker_before: Any,
    broker_after: Any,
) -> dict[str, Any]:
    before_account = broker_before.account
    after_account = broker_after.account
    before_tick = broker_before.tick
    before_symbol = broker_before.symbol_info
    before_position = (
        dict(broker_before.positions_raw[0])
        if broker_before.positions_raw
        else {}
    )
    after_position = (
        dict(broker_after.positions_raw[0])
        if broker_after.positions_raw
        else {}
    )
    model_prediction_event_time = signal_context.get("event_time_utc")

    def _normalised_event_time(value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        try:
            parsed = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    model_prediction_available = bool(
        decision.probability_up is not None
        and not decision.stale_event_warning
        and _normalised_event_time(model_prediction_event_time)
        == _normalised_event_time(decision.event_time_utc)
    )
    model_unavailable_reason = (
        None
        if model_prediction_available
        else (
            decision.reason
            if decision.stale_event_warning
            or decision.action.startswith("MODEL_UNAVAILABLE")
            or decision.action.startswith("CONTROL_MODEL_UNAVAILABLE")
            else None
        )
    )
    return {
        "schema_version": "1.0",
        "decision_id": decision_identifier(
            decision.role,
            decision.event_time_utc,
            run_id=decision.run_id,
            iteration=decision.iteration,
        ),
        "role": decision.role,
        "run_id": decision.run_id,
        "iteration": decision.iteration,
        "event_time_utc": decision.event_time_utc,
        "decision_utc": decision.decision_utc,
        "execution_mode": decision.execution_mode,
        "probability_up": (
            decision.probability_up if model_prediction_available else None
        ),
        "model_prediction_available": model_prediction_available,
        "model_prediction_event_time_utc": model_prediction_event_time,
        "model_unavailable_reason": model_unavailable_reason,
        "broker_event_disposition": decision.action,
        "signal_window_start_utc": signal_context.get("window_start_utc"),
        "signal_window_end_utc": signal_context.get("window_end_utc"),
        "latest_completed_bar_time_utc": signal_context.get(
            "latest_completed_bar_time_utc"
        ),
        "selected_symbol": signal_context.get("selected_symbol"),
        "model_a_signal": (
            signal_context.get("model_a_signal")
            if model_prediction_available
            else None
        ),
        "model_b_from_flat_signal": (
            signal_context.get("model_b_from_flat_signal")
            if model_prediction_available
            else None
        ),
        "model_b_hold_condition": (
            signal_context.get("model_b_hold_condition")
            if model_prediction_available
            else None
        ),
        "model_b_entry_condition": (
            signal_context.get("model_b_entry_condition")
            if model_prediction_available
            else None
        ),
        "sequence_length": signal_context.get("sequence_length"),
        "feature_count": signal_context.get("feature_count"),
        "valid_sequence_count": signal_context.get("valid_sequence_count"),
        "event_is_latest_feature": signal_context.get("event_is_latest_feature"),
        "event_is_latest_completed_bar": signal_context.get(
            "event_is_latest_completed_bar"
        ),
        "feature_report_json": snapshot_context.get("feature_report"),
        "time_normalisation_json": snapshot_context.get("time_normalisation"),
        "rates_report_json": snapshot_context.get("rates"),
        "strategy_rules_json": snapshot_context.get("strategy_rules"),
        "risk_rules_json": snapshot_context.get("risk_rules"),
        "m15_open": bar_context.get("m15_open"),
        "m15_high": bar_context.get("m15_high"),
        "m15_low": bar_context.get("m15_low"),
        "m15_close": bar_context.get("m15_close"),
        "m15_tick_volume": bar_context.get("m15_tick_volume"),
        "m15_spread": bar_context.get("m15_spread"),
        "m15_real_volume": bar_context.get("m15_real_volume"),
        "position_before": decision.position_before,
        "position_before_name": {-1: "SHORT", 0: "FLAT", 1: "LONG"}.get(
            int(decision.position_before)
        ),
        "desired_position": decision.desired_position,
        "desired_position_name": {-1: "SHORT", 0: "FLAT", 1: "LONG"}.get(
            int(decision.desired_position)
        ),
        "target_position": decision.target_position,
        "target_position_name": {-1: "SHORT", 0: "FLAT", 1: "LONG"}.get(
            int(decision.target_position)
        ),
        "action": decision.action,
        "reason": decision.reason,
        "duplicate_event": decision.duplicate_event,
        "gap_from_previous_event": decision.gap_from_previous_event,
        "stale_event_warning": decision.stale_event_warning,
        "requested_policy_event_units": decision.requested_policy_event_units,
        "policy_event_units": decision.policy_event_units,
        "policy_changes_today_before": decision.policy_changes_today_before,
        "policy_changes_today_after": state_after.policy_changes_today,
        "successful_entries_today_before": decision.successful_entries_today_before,
        "successful_entries_today_after": state_after.successful_entries_today,
        "policy_cap_reached": decision.policy_cap_reached,
        "entry_blocked_by_policy_cap": decision.entry_blocked_by_policy_cap,
        "exit_allowed_when_capped": decision.exit_allowed_when_capped,
        "close_only_reversal": decision.close_only_reversal,
        "hold_bars_before": state_before.hold_bars,
        "hold_bars_after": state_after.hold_bars,
        "flat_bars_since_exit_before": state_before.flat_bars_since_exit,
        "flat_bars_since_exit_after": state_after.flat_bars_since_exit,
        "current_utc_date_before": state_before.current_utc_date,
        "current_utc_date_after": state_after.current_utc_date,
        "day_start_equity_before": state_before.day_start_equity,
        "day_start_equity_after": state_after.day_start_equity,
        "peak_equity_before": state_before.running_peak_equity,
        "peak_equity_after": state_after.running_peak_equity,
        "daily_return_before": state_before.daily_return,
        "daily_return_after": state_after.daily_return,
        "total_drawdown_before": state_before.total_drawdown,
        "total_drawdown_after": state_after.total_drawdown,
        "daily_stop_active_before": state_before.daily_stop_active,
        "daily_stop_active_after": state_after.daily_stop_active,
        "total_stop_active_before": state_before.total_stop_active,
        "total_stop_active_after": state_after.total_stop_active,
        "kill_switch_active_before": state_before.kill_switch_active,
        "kill_switch_active_after": state_after.kill_switch_active,
        "reconciliation_status_before": state_before.reconciliation_status,
        "reconciliation_status_after": state_after.reconciliation_status,
        "reconciliation_incidents_before": state_before.reconciliation_incidents,
        "reconciliation_incidents_after": state_after.reconciliation_incidents,
        "daily_stop_active": decision.daily_stop_active,
        "total_stop_active": decision.total_stop_active,
        "kill_switch_active": decision.kill_switch_active,
        "reconciliation_status": decision.reconciliation_status,
        "bid_at_decision": before_tick.get("bid"),
        "ask_at_decision": before_tick.get("ask"),
        "spread_points_at_decision": spread_points_from_tick(
            before_tick, before_symbol
        ),
        "symbol_reported_spread_points_at_decision": before_symbol.get("spread"),
        "balance_before": before_account.get("balance"),
        "equity_before": before_account.get("equity"),
        "floating_profit_before": before_account.get("profit"),
        "broker_position_before": broker_before.snapshot.positions[0].position
        if broker_before.snapshot.positions
        else 0,
        "broker_position_ticket_before": before_position.get("ticket"),
        "broker_position_open_price_before": before_position.get("price_open"),
        "broker_position_current_price_before": before_position.get("price_current"),
        "broker_position_profit_before": before_position.get("profit"),
        "balance_after": after_account.get("balance"),
        "equity_after": after_account.get("equity"),
        "floating_profit_after": after_account.get("profit"),
        "broker_position_after_inspection": broker_after.snapshot.positions[0].position
        if broker_after.snapshot.positions
        else 0,
        "broker_position_ticket_after_inspection": after_position.get("ticket"),
        "broker_position_open_price_after": after_position.get("price_open"),
        "broker_position_current_price_after": after_position.get("price_current"),
        "broker_position_profit_after": after_position.get("profit"),
        "order_check_called": decision.order_check_called,
        "order_check_passed": decision.order_check_passed,
        "order_send_called": decision.order_send_called,
        "order_send_passed": decision.order_send_passed,
        "broker_position_after": decision.broker_position_after,
        "broker_position_ticket_after": decision.broker_position_ticket_after,
        "broker_before_json": broker_context_payload(broker_before),
        "broker_after_json": broker_context_payload(broker_after),
    }


def execution_audit_rows(*, execution: Any, decision: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    action = str(getattr(decision, "action", "")).strip().upper()
    inferred_control_type = (
        action
        if action.startswith("CONTROL_")
        else (
            "CONTROL_SESSION_GAP_LOCKOUT"
            if action == "SESSION_GAP_LOCKOUT_FLATTEN"
            else None
        )
    )
    trigger_type = str(
        getattr(
            decision,
            "audit_trigger_type",
            inferred_control_type or "STRATEGY_DECISION",
        )
    ).strip().upper()
    explicit_trigger_id = getattr(decision, "audit_trigger_id", None)
    if explicit_trigger_id:
        decision_id = str(explicit_trigger_id)
    elif trigger_type.startswith("CONTROL_"):
        event_token = str(decision.event_time_utc or "NO_EVENT").replace(
            ":", ""
        )
        decision_id = (
            f"{trigger_type}:{decision.role}:{decision.run_id}:"
            f"{getattr(decision, 'iteration', 0)}:{event_token}"
        )
    else:
        decision_id = decision_identifier(
            decision.role,
            decision.event_time_utc,
            run_id=decision.run_id,
            iteration=getattr(decision, "iteration", None),
        )
    for index, leg in enumerate(execution.legs, start=1):
        order_event = dict(leg.order_event)
        request = dict(order_event.get("request") or {})
        check_result = dict(order_event.get("check_result") or {})
        send_result = dict(order_event.get("send_result") or {})
        execution_id = f"{decision_id}:{index}:{leg.purpose}"
        rows.append({
            "schema_version": "1.0", "execution_id": execution_id,
            "decision_id": decision_id, "role": decision.role,
            "run_id": decision.run_id, "trigger_type": trigger_type,
            "event_time_utc": decision.event_time_utc,
            "completed_utc": leg.completed_utc, "leg_index": index,
            "purpose": leg.purpose, "side": leg.side,
            "position_before": leg.position_before, "position_after": leg.position_after,
            "requested_volume": request.get("volume"),
            "requested_price": leg.requested_price,
            "bid_before": leg.bid_before, "ask_before": leg.ask_before,
            "spread_points_before": leg.spread_points_before,
            "symbol_reported_spread_points_before": (
                leg.symbol_reported_spread_points_before
            ),
            "symbol_point": leg.symbol_point,
            "order_check_passed": leg.order_check_passed,
            "order_check_retcode": check_result.get("retcode"),
            "order_check_comment": check_result.get("comment"),
            "order_send_passed": leg.order_send_passed,
            "order_send_retcode": order_event.get("retcode"),
            "order_send_comment": order_event.get("comment"),
            "broker_result_price": leg.broker_result_price,
            "slippage_points_signed": leg.slippage_points_signed,
            "slippage_points_adverse": leg.slippage_points_adverse,
            "margin_required_account_currency": order_event.get("margin_required_account_currency"),
            "order_ticket": leg.order_ticket, "deal_ticket": leg.deal_ticket,
            "position_ticket": leg.position_ticket,
            "request_position_ticket": leg.request_position_ticket,
            "magic": request.get("magic"), "comment": request.get("comment"),
            "filling_name": order_event.get("filling_name"),
            "filling_value": order_event.get("filling_value"),
            "request_json": request, "check_result_json": check_result,
            "send_result_json": send_result,
            "last_error_json": order_event.get("last_error"),
        })
    return rows


def runtime_event_audit_row(
    *,
    role: str,
    run_id: str,
    iteration: int,
    worker_pid: int,
    event_type: str,
    event_reason: str,
    severity: str,
    state: Any,
    event_time_utc: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = utc_now_iso()
    return {
        "schema_version": "1.0",
        "runtime_event_id": f"{role}:{run_id}:{iteration}:{event_type}:{timestamp}",
        "timestamp_utc": timestamp,
        "role": role,
        "run_id": run_id,
        "iteration": iteration,
        "worker_pid": worker_pid,
        "event_type": event_type,
        "event_reason": event_reason,
        "severity": severity,
        "event_time_utc": event_time_utc,
        "virtual_position": state.virtual_position,
        "broker_position": state.broker_position,
        "reconciliation_status": state.reconciliation_status,
        "restart_count": state.restart_count,
        "details_json": dict(details or {}),
    }
