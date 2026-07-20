"""Stage 3 Step 3A Model B controlled execution dry-run.

This module bridges the read-only Stage 2 shadow signal with the Stage 3
broker execution controls, but it deliberately does not call order_send.

Stage 3 Step 3A answers: "Given the latest completed M15 signal, what would
Model B current do, and would the broker accept the intended tiny 0.01-lot
request through order_check?"  It is a dry-run gate, not strategy automation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
import csv
import json
import math
import time

DEFAULT_STAGE3_STEP2_REPORT_PATH = Path("runtime/reports/stage3_step2_v1_1_tiny_order_test.json")
DEFAULT_REPORT_PATH = Path("runtime/reports/stage3_step3a_model_b_controlled_dry_run.json")
DEFAULT_INTENTS_CSV_PATH = Path("runtime/execution_dry_run/stage3_step3a_model_b_intents.csv")
DEFAULT_LATEST_DECISION_CSV_PATH = Path("runtime/reports/stage3_step3a_latest_decision.csv")
DEFAULT_STATE_PATH = Path("runtime/state/stage3_step3a_model_b_dry_run_state.json")

MODEL_B_ENTRY_THRESHOLD = 0.55
MODEL_B_EXIT_THRESHOLD = 0.50


class Stage3Step3ADryRunError(RuntimeError):
    """Raised when Stage 3 Step 3A cannot proceed safely."""


@dataclass(frozen=True)
class ModelBDryRunRules:
    variant: str = "MODEL_B_V2_CURRENT"
    entry_threshold: float = MODEL_B_ENTRY_THRESHOLD
    exit_threshold: float = MODEL_B_EXIT_THRESHOLD
    long_only: bool = True
    max_successful_entries_per_utc_day: int = 1
    min_hold_bars: int = 0


@dataclass
class ModelBDryRunState:
    schema_version: int = 1
    virtual_position: int = 0
    virtual_position_name: str = "FLAT"
    last_event_time_utc: str | None = None
    successful_entry_dates: dict[str, int] = field(default_factory=dict)
    records_written: int = 0
    updated_utc: str | None = None


@dataclass(frozen=True)
class DryRunDecision:
    run_id: str
    iteration: int
    mode: str
    event_time_utc: str | None
    probability_up: float | None
    action: str
    reason: str
    virtual_position_before: int
    virtual_position_after: int
    duplicate_event: bool
    stale_event_warning: bool
    spread_points: int | None
    spread_gate_points: int | None
    actual_position_count: int | None
    pending_order_count: int | None
    order_check_called: bool
    order_check_passed: bool | None
    order_check_retcode: int | None
    order_check_comment: str | None
    order_check_margin_required: float | None
    order_send_called: bool
    would_send_order_in_stage3b: bool
    decision_utc: str


class GuardedMt5DryRunProxy:
    """MT5 proxy for Step 3A.

    Allows inspection, position/order reads, margin calculation and order_check.
    Blocks order_send and history mutation/execution-style calls.  Stage 3 Step
    3A must not open, close, or modify any broker position.
    """

    allowed_methods = {
        "initialize",
        "shutdown",
        "last_error",
        "version",
        "terminal_info",
        "account_info",
        "symbol_info",
        "symbol_info_tick",
        "positions_get",
        "positions_total",
        "orders_get",
        "orders_total",
        "order_calc_margin",
        "order_check",
    }
    forbidden_methods = {
        "order_send",
        "history_deals_get",
        "history_orders_get",
        "history_deals_total",
        "history_orders_total",
        "order_calc_profit",
    }

    def __init__(self, mt5_module: Any):
        self._mt5 = mt5_module
        self.calls: list[str] = []
        self.forbidden_attempts: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name in self.forbidden_methods:
            self.forbidden_attempts.append(name)
            raise Stage3Step3ADryRunError(f"Forbidden MT5 method accessed in Stage 3 Step 3A: {name}")
        if name in self.allowed_methods:
            attr = getattr(self._mt5, name)
            if callable(attr):
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    self.calls.append(name)
                    return attr(*args, **kwargs)
                return wrapped
            return attr
        if name.startswith("ORDER_") or name.startswith("TRADE_") or name.startswith("ACCOUNT_") or name.startswith("SYMBOL_"):
            return getattr(self._mt5, name)
        if name in {"__author__", "__version__"}:
            return getattr(self._mt5, name, None)
        raise AttributeError(name)


def position_name(value: int) -> str:
    return {-1: "SHORT", 0: "FLAT", 1: "LONG"}.get(int(value), f"UNKNOWN_{value}")


def load_required_stage3_step2_report(repo_root: Path, path: Path = DEFAULT_STAGE3_STEP2_REPORT_PATH) -> dict[str, Any]:
    full_path = repo_root / path
    if not full_path.exists():
        raise Stage3Step3ADryRunError(f"Stage 3 Step 2 v1.1 PASS report not found: {full_path}")
    report = json.loads(full_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("formal_gate") is not True:
        raise Stage3Step3ADryRunError("Stage 3 Step 2 v1.1 report is not a formal PASS")
    if report.get("open_close_completed") is not True or report.get("orders_executed") is not True:
        raise Stage3Step3ADryRunError("Stage 3 Step 2 v1.1 report did not prove a complete open-close execution")
    validations = report.get("validations", {}) if isinstance(report.get("validations"), Mapping) else {}
    if validations.get("history_records_recovered") is not True:
        raise Stage3Step3ADryRunError("Stage 3 Step 2 v1.1 report did not recover broker history records")
    if validations.get("no_position_after_close") is not True:
        raise Stage3Step3ADryRunError("Stage 3 Step 2 v1.1 report did not verify no position after close")
    return report


def load_dry_run_state(path: Path) -> ModelBDryRunState:
    if not path.exists():
        return ModelBDryRunState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Stage3Step3ADryRunError(f"Unable to read dry-run state {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise Stage3Step3ADryRunError(f"Dry-run state must be a JSON object: {path}")
    state = ModelBDryRunState(
        schema_version=int(raw.get("schema_version", 1)),
        virtual_position=int(raw.get("virtual_position", 0)),
        virtual_position_name=position_name(int(raw.get("virtual_position", 0))),
        last_event_time_utc=raw.get("last_event_time_utc"),
        successful_entry_dates={str(k): int(v) for k, v in dict(raw.get("successful_entry_dates", {})).items()},
        records_written=int(raw.get("records_written", 0)),
        updated_utc=raw.get("updated_utc"),
    )
    if state.virtual_position not in {0, 1}:
        raise Stage3Step3ADryRunError("Model B dry-run state must be FLAT or LONG only")
    return state


def write_dry_run_state(path: Path, state: ModelBDryRunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def reset_dry_run_state(path: Path) -> None:
    if path.exists():
        path.unlink()


def _event_date(event_time_utc: str | None) -> str | None:
    if not event_time_utc:
        return None
    try:
        return datetime.fromisoformat(event_time_utc.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return str(event_time_utc)[:10]


def decide_model_b_action(
    *,
    signal: Mapping[str, Any],
    state: ModelBDryRunState,
    rules: ModelBDryRunRules,
    spread_points: int | None,
    spread_gate_points: int,
    actual_position_count: int,
    pending_order_count: int,
    run_id: str = "unit",
    iteration: int = 1,
    mode: str = "unit",
) -> DryRunDecision:
    """Apply Model B current entry/exit logic to one completed M15 signal.

    This function is pure and does not touch MT5.  It produces a would-send
    decision only; later runtime code may optionally validate the request with
    order_check, but Stage 3 Step 3A still never sends an order.
    """
    event_time = _str_or_none(signal.get("event_time_utc"))
    probability = _float_or_none(signal.get("probability_up"))
    stale = bool(signal.get("stale_event_warning", False))
    duplicate = bool(event_time and state.last_event_time_utc == event_time)
    before = int(state.virtual_position)
    action = "HOLD_FLAT" if before == 0 else "HOLD_LONG"
    reason = "no_action"
    would_send = False

    if duplicate:
        action = "DUPLICATE_SKIP"
        reason = "event_already_processed"
    elif event_time is None or probability is None:
        action = "BLOCK_INVALID_SIGNAL"
        reason = "missing_event_time_or_probability"
    elif stale:
        action = "BLOCK_STALE_EVENT"
        reason = "latest_valid_window_not_aligned_to_latest_completed_bar"
    elif actual_position_count > 0:
        action = "BLOCK_ACTUAL_POSITION_EXISTS"
        reason = "dry_run_requires_no_actual_xauusd_position"
    elif pending_order_count > 0:
        action = "BLOCK_PENDING_ORDER_EXISTS"
        reason = "dry_run_requires_no_pending_xauusd_order"
    elif spread_points is None or spread_points < 0:
        action = "BLOCK_INVALID_SPREAD"
        reason = "spread_unavailable"
    elif spread_points > spread_gate_points:
        action = "BLOCK_SPREAD"
        reason = "spread_above_frozen_entry_gate"
    elif before == 0:
        date_key = _event_date(event_time)
        entries_today = int(state.successful_entry_dates.get(date_key or "", 0))
        if probability >= rules.entry_threshold:
            if entries_today >= rules.max_successful_entries_per_utc_day:
                action = "BLOCK_DAILY_ENTRY_CAP"
                reason = "max_one_successful_entry_per_utc_day_already_used"
            else:
                action = "ENTER_LONG"
                reason = "p_up_at_or_above_model_b_entry_threshold"
                would_send = True
        else:
            action = "HOLD_FLAT"
            reason = "p_up_below_model_b_entry_threshold"
    elif before == 1:
        if probability < rules.exit_threshold:
            action = "EXIT_LONG"
            reason = "p_up_below_model_b_exit_threshold"
            would_send = True
        else:
            action = "HOLD_LONG"
            reason = "p_up_at_or_above_model_b_exit_threshold"
    else:
        action = "BLOCK_INVALID_STATE"
        reason = "model_b_state_must_be_flat_or_long"

    after = before
    if action == "ENTER_LONG":
        after = 1
    elif action == "EXIT_LONG":
        after = 0

    return DryRunDecision(
        run_id=run_id,
        iteration=int(iteration),
        mode=mode,
        event_time_utc=event_time,
        probability_up=probability,
        action=action,
        reason=reason,
        virtual_position_before=before,
        virtual_position_after=after,
        duplicate_event=duplicate,
        stale_event_warning=stale,
        spread_points=spread_points,
        spread_gate_points=spread_gate_points,
        actual_position_count=int(actual_position_count),
        pending_order_count=int(pending_order_count),
        order_check_called=False,
        order_check_passed=None,
        order_check_retcode=None,
        order_check_comment=None,
        order_check_margin_required=None,
        order_send_called=False,
        would_send_order_in_stage3b=would_send,
        decision_utc=datetime.now(timezone.utc).isoformat(),
    )


def apply_decision_to_state(state: ModelBDryRunState, decision: DryRunDecision) -> ModelBDryRunState:
    if decision.duplicate_event:
        return state
    next_entries = dict(state.successful_entry_dates)
    if decision.action == "ENTER_LONG":
        date_key = _event_date(decision.event_time_utc)
        if date_key:
            next_entries[date_key] = int(next_entries.get(date_key, 0)) + 1
    return ModelBDryRunState(
        schema_version=1,
        virtual_position=int(decision.virtual_position_after),
        virtual_position_name=position_name(int(decision.virtual_position_after)),
        last_event_time_utc=decision.event_time_utc or state.last_event_time_utc,
        successful_entry_dates=next_entries,
        records_written=state.records_written + 1,
        updated_utc=datetime.now(timezone.utc).isoformat(),
    )


def with_order_check_result(
    decision: DryRunDecision,
    *,
    called: bool,
    passed: bool | None,
    retcode: int | None,
    comment: str | None,
    margin_required: float | None,
) -> DryRunDecision:
    payload = asdict(decision)
    payload.update(
        {
            "order_check_called": bool(called),
            "order_check_passed": passed,
            "order_check_retcode": retcode,
            "order_check_comment": comment,
            "order_check_margin_required": margin_required,
            "order_send_called": False,
        }
    )
    return DryRunDecision(**payload)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _object_list_to_plain(value: Any, object_to_plain_dict: Callable[[Any], dict[str, Any]]) -> list[dict[str, Any]]:
    if value is None:
        return []
    return [object_to_plain_dict(item) for item in list(value)]


def _get_positions(proxy: GuardedMt5DryRunProxy, symbol: str, object_to_plain_dict: Callable[[Any], dict[str, Any]]) -> list[dict[str, Any]]:
    positions = proxy.positions_get(symbol=symbol)
    if positions is None:
        raise Stage3Step3ADryRunError("positions_get returned None during dry-run")
    return _object_list_to_plain(positions, object_to_plain_dict)


def _get_orders(proxy: GuardedMt5DryRunProxy, symbol: str, object_to_plain_dict: Callable[[Any], dict[str, Any]]) -> list[dict[str, Any]]:
    orders = proxy.orders_get(symbol=symbol)
    if orders is None:
        raise Stage3Step3ADryRunError("orders_get returned None during dry-run")
    return _object_list_to_plain(orders, object_to_plain_dict)


def _build_entry_order_check_request(proxy: GuardedMt5DryRunProxy, controls: Any, tick: Mapping[str, Any], filling_value: int, run_id: str) -> dict[str, Any]:
    comment_suffix = run_id[-10:].replace("T", "")[-8:]
    comment = f"CP_S3P3A_B_{comment_suffix}"[:31]
    return {
        "action": int(getattr(proxy, "TRADE_ACTION_DEAL")),
        "symbol": controls.symbol,
        "volume": float(controls.stage3_first_order_test_volume_lots),
        "type": int(getattr(proxy, "ORDER_TYPE_BUY")),
        "price": float(tick["ask"]),
        "deviation": int(controls.max_deviation_points_for_stage3_order_request),
        "magic": int(controls.model_b_magic_number),
        "comment": comment,
        "type_time": int(getattr(proxy, "ORDER_TIME_GTC")),
        "type_filling": int(filling_value),
    }


def run_dry_run_iteration(
    *,
    mt5_module: Any,
    terminal_path: str | None,
    controls: Any,
    shadow_once_callable: Callable[[], Mapping[str, Any]],
    state: ModelBDryRunState,
    rules: ModelBDryRunRules,
    run_id: str,
    iteration: int,
    mode: str,
    order_check_enabled: bool = True,
) -> tuple[DryRunDecision, ModelBDryRunState, dict[str, Any]]:
    """Run one Model B dry-run iteration.

    The shadow callable performs model inference on the latest completed M15 bar.
    This function then inspects broker context, applies Model B current rules,
    optionally performs order_check for ENTER_LONG intent, and writes no orders.
    """
    from capstone_trading.runtime.order_preflight import (  # type: ignore
        call_order_calc_margin_compat,
        call_order_check_compat,
        capital_review_from_live_account,
        choose_filling_candidates,
        inspect_account_for_trading,
        inspect_symbol_for_order_check,
        inspect_terminal_for_trading,
        inspect_tick_for_order_check,
        initialise_terminal,
        normalise_check_result,
    )
    from capstone_trading.runtime.mt5_readiness import object_to_plain_dict, safe_last_error  # type: ignore

    shadow_snapshot = dict(shadow_once_callable())
    signal = shadow_snapshot.get("latest_signal", {})
    if not isinstance(signal, Mapping):
        raise Stage3Step3ADryRunError("Shadow snapshot did not include latest_signal")

    # Duplicate events are skipped without reconnecting for broker order_check.
    event_time = _str_or_none(signal.get("event_time_utc"))
    if event_time and state.last_event_time_utc == event_time:
        decision = decide_model_b_action(
            signal=signal,
            state=state,
            rules=rules,
            spread_points=None,
            spread_gate_points=int(controls.max_spread_points_for_entry),
            actual_position_count=0,
            pending_order_count=0,
            run_id=run_id,
            iteration=iteration,
            mode=mode,
        )
        return decision, state, {"shadow_snapshot": shadow_snapshot, "broker_context_skipped": "duplicate_event"}

    proxy = GuardedMt5DryRunProxy(mt5_module)
    initialized = False
    shutdown_called = False
    broker_context: dict[str, Any] = {}
    try:
        initialise_terminal(proxy, terminal_path)
        initialized = True
        terminal = inspect_terminal_for_trading(proxy)
        account = inspect_account_for_trading(proxy, controls)
        symbol_info = inspect_symbol_for_order_check(proxy, controls)
        tick = inspect_tick_for_order_check(proxy, controls.symbol)
        positions = _get_positions(proxy, controls.symbol, object_to_plain_dict)
        orders = _get_orders(proxy, controls.symbol, object_to_plain_dict)
        capital_review = capital_review_from_live_account(controls=controls, account=account, symbol_info=symbol_info)
        spread = int(symbol_info.get("spread", -1))
        decision = decide_model_b_action(
            signal=signal,
            state=state,
            rules=rules,
            spread_points=spread,
            spread_gate_points=int(controls.max_spread_points_for_entry),
            actual_position_count=len(positions),
            pending_order_count=len(orders),
            run_id=run_id,
            iteration=iteration,
            mode=mode,
        )

        order_check_payload: dict[str, Any] | None = None
        if order_check_enabled and decision.action == "ENTER_LONG":
            filling_candidates = choose_filling_candidates(proxy, symbol_info)
            filling_name, filling_value = filling_candidates[0]
            request = _build_entry_order_check_request(proxy, controls, tick, filling_value, run_id)
            margin = call_order_calc_margin_compat(
                proxy,
                action=int(request["type"]),
                symbol=str(request["symbol"]),
                volume=float(request["volume"]),
                price=float(request["price"]),
            )
            check = normalise_check_result(call_order_check_compat(proxy, request))
            retcode = None
            comment = None
            passed = False
            if check is not None:
                try:
                    retcode = int(check.get("retcode")) if check.get("retcode") is not None else None
                except Exception:
                    retcode = None
                comment = None if check.get("comment") is None else str(check.get("comment"))
                passed = retcode == 0
            order_check_payload = {
                "request": request,
                "result": check,
                "filling_name": filling_name,
                "filling_value": filling_value,
                "last_error": safe_last_error(proxy),
            }
            decision = with_order_check_result(
                decision,
                called=True,
                passed=passed,
                retcode=retcode,
                comment=comment,
                margin_required=margin,
            )

        next_state = apply_decision_to_state(state, decision)
        proxy.shutdown()
        shutdown_called = True
        initialized = False
        broker_context = {
            "terminal": terminal,
            "account": account,
            "symbol_info": symbol_info,
            "tick": tick,
            "actual_positions": positions,
            "pending_orders": orders,
            "capital_review": capital_review,
            "order_check": order_check_payload,
            "mt5_calls": tuple(proxy.calls),
            "forbidden_attempts": tuple(proxy.forbidden_attempts),
            "shutdown_called": shutdown_called,
        }
        if proxy.forbidden_attempts:
            raise Stage3Step3ADryRunError(f"Forbidden MT5 calls in dry-run: {proxy.forbidden_attempts}")
        if "order_send" in proxy.calls:
            raise Stage3Step3ADryRunError("order_send was called during dry-run, which is forbidden")
        return decision, next_state, {"shadow_snapshot": shadow_snapshot, "broker_context": broker_context}
    finally:
        if initialized:
            try:
                proxy.shutdown()
            except Exception:
                pass


def decision_to_row(decision: DryRunDecision) -> dict[str, Any]:
    return asdict(decision)


def append_csv_row(path: Path, row: Mapping[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))


def write_csv_rows_atomic(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
    tmp.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def decision_fieldnames() -> list[str]:
    return list(asdict(DryRunDecision(
        run_id="",
        iteration=0,
        mode="",
        event_time_utc=None,
        probability_up=None,
        action="",
        reason="",
        virtual_position_before=0,
        virtual_position_after=0,
        duplicate_event=False,
        stale_event_warning=False,
        spread_points=None,
        spread_gate_points=None,
        actual_position_count=None,
        pending_order_count=None,
        order_check_called=False,
        order_check_passed=None,
        order_check_retcode=None,
        order_check_comment=None,
        order_check_margin_required=None,
        order_send_called=False,
        would_send_order_in_stage3b=False,
        decision_utc="",
    )).keys())


def summarise_decisions(decisions: list[DryRunDecision]) -> dict[str, Any]:
    unique_events = {d.event_time_utc for d in decisions if d.event_time_utc and not d.duplicate_event}
    actions: dict[str, int] = {}
    for decision in decisions:
        actions[decision.action] = actions.get(decision.action, 0) + 1
    return {
        "decision_count": len(decisions),
        "unique_completed_m15_events": len(unique_events),
        "action_counts": actions,
        "would_send_order_count_stage3b": sum(1 for d in decisions if d.would_send_order_in_stage3b),
        "order_check_called_count": sum(1 for d in decisions if d.order_check_called),
        "order_check_passed_count": sum(1 for d in decisions if d.order_check_passed is True),
        "order_send_called_count": sum(1 for d in decisions if d.order_send_called),
        "final_virtual_position": decisions[-1].virtual_position_after if decisions else None,
        "final_virtual_position_name": position_name(decisions[-1].virtual_position_after) if decisions else None,
    }
