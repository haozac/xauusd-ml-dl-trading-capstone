"""Stage 3 Step 3B Model B controlled live execution.

This module is the first controlled Model B runtime that may call
``mt5.order_send`` when the frozen Model B current rules produce a valid live
entry or exit signal.  It is deliberately narrow and safety-first:

* Model B current only: long-only, entry p_up >= 0.55, exit p_up < 0.50.
* XAUUSD only, 0.01 lot only, frozen broker controls only.
* Completed M15 bars only, inherited from the Stage 2/3A shadow pipeline.
* One Model B position maximum.
* One successful entry per UTC day maximum.
* No broker-side SL/TP in this controlled gate.
* A manual confirmation token is required before order_send is allowed.

This is not the final unattended 14-calendar-day paper-trading run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import csv
import json
import math
import time

from capstone_trading.runtime.model_b_controlled_dry_run import (
    MODEL_B_ENTRY_THRESHOLD,
    MODEL_B_EXIT_THRESHOLD,
    ModelBDryRunRules,
    position_name,
)
from capstone_trading.runtime.order_execution_probe import (  # type: ignore
    GuardedMt5TinyOrderProxy,
    OrderSendEvent,
    SendAuthorisation,
    build_deal_request,
    build_send_event,
    collect_history_rows,
    get_orders_for_symbol,
    get_positions_for_symbol,
    run_order_check_for_request,
    snapshot_positions,
    wait_for_no_position,
    wait_for_position,
    call_order_send_compat,
)
from capstone_trading.runtime.order_preflight import (  # type: ignore
    FrozenBrokerControls,
    Stage3OrderPreflightError,
    build_package_snapshot,
    capital_review_from_live_account,
    choose_filling_candidates,
    inspect_account_for_trading,
    inspect_terminal_for_trading,
    inspect_tick_for_order_check,
    initialise_terminal,
)
from capstone_trading.runtime.mt5_readiness import object_to_plain_dict

DEFAULT_STAGE3_STEP2_REPORT_PATH = Path("runtime/reports/stage3_step2_v1_1_tiny_order_test.json")
DEFAULT_STAGE3_STEP3A_REPORT_PATH = Path("runtime/reports/stage3_step3a_model_b_controlled_dry_run.json")
DEFAULT_REPORT_PATH = Path("runtime/reports/stage3_step3b_model_b_controlled_live.json")
DEFAULT_EVENTS_CSV_PATH = Path("runtime/execution_live/stage3_step3b_model_b_events.csv")
DEFAULT_ORDER_EVENTS_CSV_PATH = Path("runtime/execution_live/stage3_step3b_order_send_events.csv")
DEFAULT_POSITION_SNAPSHOTS_CSV_PATH = Path("runtime/execution_live/stage3_step3b_position_snapshots.csv")
DEFAULT_HISTORY_DEALS_CSV_PATH = Path("runtime/execution_live/stage3_step3b_history_deals.csv")
DEFAULT_HISTORY_ORDERS_CSV_PATH = Path("runtime/execution_live/stage3_step3b_history_orders.csv")
DEFAULT_LATEST_DECISION_CSV_PATH = Path("runtime/reports/stage3_step3b_latest_decision.csv")
DEFAULT_STATE_PATH = Path("runtime/state/stage3_step3b_model_b_live_state.json")

CONFIRM_SEND_TOKEN = "I_UNDERSTAND_STAGE3_STEP3B_MODEL_B_ORDER_SEND"
MODEL_B_MAGIC_COMMENT_PREFIX = "CP_S3P3B_B"


class Stage3Step3BLiveError(RuntimeError):
    """Raised when Stage 3 Step 3B cannot proceed safely."""


class EntrySpreadBlockedError(Stage3Step3BLiveError):
    """Raised when the entry spread widens above the frozen gate before send.

    This is a non-fatal market condition.  The caller must convert it into a
    BLOCK_SPREAD decision and continue monitoring rather than terminate the
    controlled runtime.
    """

    def __init__(self, spread_points: int | None, spread_gate_points: int):
        self.spread_points = spread_points
        self.spread_gate_points = spread_gate_points
        super().__init__(
            f"Entry spread {spread_points} exceeds frozen gate {spread_gate_points}"
        )


@dataclass
class ModelBLiveState:
    schema_version: int = 1
    live_position: int = 0
    live_position_name: str = "FLAT"
    position_ticket: int | None = None
    position_identifier: int | None = None
    open_order_ticket: int | None = None
    open_event_time_utc: str | None = None
    last_event_time_utc: str | None = None
    successful_entry_dates: dict[str, int] = field(default_factory=dict)
    records_written: int = 0
    completed_entry_exit_cycles: int = 0
    updated_utc: str | None = None


@dataclass(frozen=True)
class LiveDecision:
    run_id: str
    iteration: int
    mode: str
    event_time_utc: str | None
    probability_up: float | None
    action: str
    reason: str
    live_position_before: int
    live_position_after: int
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
    order_send_passed: bool | None
    order_send_retcode: int | None
    order_send_comment: str | None
    broker_position_ticket: int | None
    order_ticket: int | None
    decision_utc: str


def load_required_json_report(repo_root: Path, path: Path, *, stage_name: str) -> dict[str, Any]:
    full_path = repo_root / path
    if not full_path.exists():
        raise Stage3Step3BLiveError(f"{stage_name} report not found: {full_path}")
    report = json.loads(full_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("formal_gate") is not True:
        raise Stage3Step3BLiveError(f"{stage_name} report is not a formal PASS")
    return report


def load_required_stage3_step2_report(repo_root: Path, path: Path = DEFAULT_STAGE3_STEP2_REPORT_PATH) -> dict[str, Any]:
    report = load_required_json_report(repo_root, path, stage_name="Stage 3 Step 2 v1.1")
    if report.get("open_close_completed") is not True or report.get("orders_executed") is not True:
        raise Stage3Step3BLiveError("Stage 3 Step 2 v1.1 did not prove a complete open-close execution")
    validations = report.get("validations", {}) if isinstance(report.get("validations"), Mapping) else {}
    if validations.get("history_records_recovered") is not True:
        raise Stage3Step3BLiveError("Stage 3 Step 2 v1.1 did not recover broker history records")
    if validations.get("no_position_after_close") is not True:
        raise Stage3Step3BLiveError("Stage 3 Step 2 v1.1 did not verify no position after close")
    return report


def load_required_stage3_step3a_report(repo_root: Path, path: Path = DEFAULT_STAGE3_STEP3A_REPORT_PATH) -> dict[str, Any]:
    report = load_required_json_report(repo_root, path, stage_name="Stage 3 Step 3A")
    if report.get("order_send_called") is not False:
        raise Stage3Step3BLiveError("Stage 3 Step 3A must be a dry-run report with no order_send")
    safety = report.get("safety", {}) if isinstance(report.get("safety"), Mapping) else {}
    if safety.get("stage3_step3b_single_model_execution_allowed") is not True:
        raise Stage3Step3BLiveError("Stage 3 Step 3A did not allow Step 3B controlled live execution")
    summary = report.get("summary", {}) if isinstance(report.get("summary"), Mapping) else {}
    if int(summary.get("unique_completed_m15_events", 0) or 0) < 4:
        raise Stage3Step3BLiveError("Stage 3 Step 3A did not observe enough completed M15 events")
    return report


def load_live_state(path: Path) -> ModelBLiveState:
    if not path.exists():
        return ModelBLiveState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Stage3Step3BLiveError(f"Unable to read live state {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise Stage3Step3BLiveError(f"Live state must be a JSON object: {path}")
    live_position = int(raw.get("live_position", 0))
    if live_position not in {0, 1}:
        raise Stage3Step3BLiveError("Model B live state must be FLAT or LONG only")
    return ModelBLiveState(
        schema_version=int(raw.get("schema_version", 1)),
        live_position=live_position,
        live_position_name=position_name(live_position),
        position_ticket=_int_or_none(raw.get("position_ticket")),
        position_identifier=_int_or_none(raw.get("position_identifier")),
        open_order_ticket=_int_or_none(raw.get("open_order_ticket")),
        open_event_time_utc=_str_or_none(raw.get("open_event_time_utc")),
        last_event_time_utc=_str_or_none(raw.get("last_event_time_utc")),
        successful_entry_dates={str(k): int(v) for k, v in dict(raw.get("successful_entry_dates", {})).items()},
        records_written=int(raw.get("records_written", 0)),
        completed_entry_exit_cycles=int(raw.get("completed_entry_exit_cycles", 0)),
        updated_utc=_str_or_none(raw.get("updated_utc")),
    )


def write_live_state(path: Path, state: ModelBLiveState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def reset_live_state(path: Path) -> None:
    if path.exists():
        path.unlink()


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


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _event_date(event_time_utc: str | None) -> str | None:
    if not event_time_utc:
        return None
    try:
        return datetime.fromisoformat(event_time_utc.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return str(event_time_utc)[:10]


def compact_comment(prefix: str, run_id: str, event_time_utc: str | None, suffix: str = "") -> str:
    """Build an MT5-safe comment no longer than 31 characters."""
    event_piece = "NA"
    if event_time_utc:
        digits = "".join(ch for ch in str(event_time_utc) if ch.isdigit())
        if len(digits) >= 12:
            event_piece = digits[4:12]  # MMDDHHMM
    run_piece = "".join(ch for ch in str(run_id) if ch.isdigit())[-4:] or "0000"
    text = f"{prefix}_{event_piece}_{run_piece}{suffix}"
    return text[:31]


def decision_fieldnames() -> list[str]:
    return list(asdict(LiveDecision(
        run_id="",
        iteration=0,
        mode="",
        event_time_utc=None,
        probability_up=None,
        action="",
        reason="",
        live_position_before=0,
        live_position_after=0,
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
        order_send_passed=None,
        order_send_retcode=None,
        order_send_comment=None,
        broker_position_ticket=None,
        order_ticket=None,
        decision_utc="",
    )).keys())


def inspect_symbol_for_live_runtime(
    mt5: Any,
    controls: FrozenBrokerControls,
    *,
    enforce_entry_spread: bool = False,
) -> dict[str, Any]:
    """Inspect XAUUSD without treating a temporary wide spread as fatal.

    Structural symbol failures remain hard errors.  The spread is recorded on
    every iteration, but it is enforced only for a new entry.  Existing
    positions must remain manageable even when the spread is temporarily wide.
    """
    info = mt5.symbol_info(controls.symbol)
    if info is None:
        raise Stage3Step3BLiveError(f"symbol_info({controls.symbol!r}) returned None")
    snapshot = object_to_plain_dict(info)
    checks = {
        "visible": bool(snapshot.get("visible", False)),
        "trade_mode_recorded": "trade_mode" in snapshot,
        "volume_min_allows_test_volume": float(snapshot.get("volume_min", math.inf))
        <= controls.stage3_first_order_test_volume_lots,
        "volume_step_allows_test_volume": _volume_step_valid(
            controls.stage3_first_order_test_volume_lots,
            _float_or_none(snapshot.get("volume_min")),
            _float_or_none(snapshot.get("volume_step")),
        ),
    }
    if not all(checks.values()):
        raise Stage3Step3BLiveError(
            f"Symbol failed structural Stage 3 live checks: {checks}"
        )

    spread = _int_or_none(snapshot.get("spread"))
    spread_within_gate = (
        spread is not None
        and spread >= 0
        and spread <= int(controls.max_spread_points_for_entry)
    )
    checks["spread_within_entry_gate"] = spread_within_gate
    snapshot["live_runtime_checks"] = checks

    if enforce_entry_spread and not spread_within_gate:
        raise EntrySpreadBlockedError(
            spread_points=spread,
            spread_gate_points=int(controls.max_spread_points_for_entry),
        )
    return snapshot


def _volume_step_valid(
    volume: float,
    volume_min: float | None,
    volume_step: float | None,
    eps: float = 1e-9,
) -> bool:
    if volume_min is None or volume_step is None or volume_step <= 0:
        return False
    if volume < volume_min - eps:
        return False
    steps = (volume - volume_min) / volume_step
    return abs(steps - round(steps)) <= eps


def decide_model_b_live_action(
    *,
    signal: Mapping[str, Any],
    state: ModelBLiveState,
    rules: ModelBDryRunRules,
    spread_points: int | None,
    spread_gate_points: int,
    actual_position_count: int,
    pending_order_count: int,
    run_id: str = "unit",
    iteration: int = 1,
    mode: str = "unit",
) -> LiveDecision:
    event_time = _str_or_none(signal.get("event_time_utc"))
    probability = _float_or_none(signal.get("probability_up"))
    stale = bool(signal.get("stale_event_warning", False))
    duplicate = bool(event_time and state.last_event_time_utc == event_time)
    before = int(state.live_position)
    after = before
    action = "HOLD_FLAT" if before == 0 else "HOLD_LONG"
    reason = "no_action"

    if duplicate:
        action = "DUPLICATE_SKIP"
        reason = "event_already_processed"
    elif event_time is None or probability is None:
        action = "BLOCK_INVALID_SIGNAL"
        reason = "missing_event_time_or_probability"
    elif stale:
        action = "BLOCK_STALE_EVENT"
        reason = "latest_valid_window_not_aligned_to_latest_completed_bar"
    elif pending_order_count > 0:
        action = "BLOCK_PENDING_ORDER_EXISTS"
        reason = "live_execution_requires_no_pending_xauusd_order"
    elif actual_position_count > 1:
        action = "BLOCK_TOO_MANY_POSITIONS"
        reason = "one_model_b_position_maximum"
    elif before == 0 and actual_position_count > 0:
        action = "BLOCK_ACTUAL_POSITION_STATE_MISMATCH"
        reason = "state_flat_but_actual_xauusd_position_exists"
    elif before == 1 and actual_position_count != 1:
        action = "BLOCK_ACTUAL_POSITION_STATE_MISMATCH"
        reason = "state_long_but_actual_xauusd_position_missing"
    elif before == 0:
        date_key = _event_date(event_time)
        entries_today = int(state.successful_entry_dates.get(date_key or "", 0))
        if probability >= rules.entry_threshold:
            if spread_points is None or spread_points < 0:
                action = "BLOCK_INVALID_SPREAD"
                reason = "spread_unavailable_for_new_entry"
            elif spread_points > spread_gate_points:
                action = "BLOCK_SPREAD"
                reason = "spread_above_frozen_entry_gate"
            elif entries_today >= rules.max_successful_entries_per_utc_day:
                action = "BLOCK_DAILY_ENTRY_CAP"
                reason = "max_one_successful_entry_per_utc_day_already_used"
            else:
                action = "ENTER_LONG"
                reason = "p_up_at_or_above_model_b_entry_threshold"
                after = 1
        else:
            action = "HOLD_FLAT"
            reason = "p_up_below_model_b_entry_threshold"
    elif before == 1:
        # The spread gate is an entry-cost control, not an exit blocker.
        # Existing positions must remain closable during a spread spike.
        if probability < rules.exit_threshold:
            action = "EXIT_LONG"
            reason = "p_up_below_model_b_exit_threshold"
            after = 0
        else:
            action = "HOLD_LONG"
            reason = "p_up_at_or_above_model_b_exit_threshold"
    else:
        action = "BLOCK_INVALID_STATE"
        reason = "model_b_state_must_be_flat_or_long"

    return LiveDecision(
        run_id=run_id,
        iteration=int(iteration),
        mode=mode,
        event_time_utc=event_time,
        probability_up=probability,
        action=action,
        reason=reason,
        live_position_before=before,
        live_position_after=after,
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
        order_send_passed=None,
        order_send_retcode=None,
        order_send_comment=None,
        broker_position_ticket=state.position_ticket,
        order_ticket=None,
        decision_utc=datetime.now(timezone.utc).isoformat(),
    )


def update_decision_after_send(
    decision: LiveDecision,
    *,
    check_result: Mapping[str, Any] | None,
    margin_required: float | None,
    send_event: OrderSendEvent,
    broker_position_ticket: int | None,
) -> LiveDecision:
    payload = asdict(decision)
    check_retcode = None
    check_comment = None
    if check_result is not None:
        check_retcode = _int_or_none(check_result.get("retcode"))
        check_comment = None if check_result.get("comment") is None else str(check_result.get("comment"))
    order_ticket = None
    if send_event.send_result is not None:
        order_ticket = _int_or_none(send_event.send_result.get("order"))
    payload.update(
        {
            "order_check_called": True,
            "order_check_passed": check_retcode == 0,
            "order_check_retcode": check_retcode,
            "order_check_comment": check_comment,
            "order_check_margin_required": margin_required,
            "order_send_called": True,
            "order_send_passed": bool(send_event.passed),
            "order_send_retcode": send_event.retcode,
            "order_send_comment": send_event.comment,
            "broker_position_ticket": broker_position_ticket,
            "order_ticket": order_ticket,
        }
    )
    return LiveDecision(**payload)


def apply_live_decision_to_state(
    state: ModelBLiveState,
    decision: LiveDecision,
    *,
    opened_position: Mapping[str, Any] | None = None,
    closed_position: bool = False,
) -> ModelBLiveState:
    if decision.duplicate_event:
        return state
    entries = dict(state.successful_entry_dates)
    position_ticket = state.position_ticket
    position_identifier = state.position_identifier
    open_order_ticket = state.open_order_ticket
    open_event_time_utc = state.open_event_time_utc
    completed_cycles = state.completed_entry_exit_cycles

    if decision.action == "ENTER_LONG" and decision.order_send_passed:
        if opened_position is None:
            raise Stage3Step3BLiveError("Opened position snapshot is required after successful ENTER_LONG")
        position_ticket = _int_or_none(opened_position.get("ticket")) or _int_or_none(opened_position.get("identifier"))
        position_identifier = _int_or_none(opened_position.get("identifier"))
        open_order_ticket = decision.order_ticket
        open_event_time_utc = decision.event_time_utc
        date_key = _event_date(decision.event_time_utc)
        if date_key:
            entries[date_key] = int(entries.get(date_key, 0)) + 1
    elif decision.action in {"EXIT_LONG", "FORCED_CLOSE_END_OF_RUN"} and decision.order_send_passed and closed_position:
        position_ticket = None
        position_identifier = None
        open_order_ticket = None
        open_event_time_utc = None
        completed_cycles += 1

    return ModelBLiveState(
        schema_version=1,
        live_position=int(decision.live_position_after),
        live_position_name=position_name(int(decision.live_position_after)),
        position_ticket=position_ticket,
        position_identifier=position_identifier,
        open_order_ticket=open_order_ticket,
        open_event_time_utc=open_event_time_utc,
        last_event_time_utc=decision.event_time_utc or state.last_event_time_utc,
        successful_entry_dates=entries,
        records_written=state.records_written + 1,
        completed_entry_exit_cycles=completed_cycles,
        updated_utc=datetime.now(timezone.utc).isoformat(),
    )


def _position_side_name(mt5: Any, position: Mapping[str, Any]) -> str:
    pos_type = int(position.get("type", -999))
    if pos_type == int(getattr(mt5, "POSITION_TYPE_BUY", 0)):
        return "BUY"
    if pos_type == int(getattr(mt5, "POSITION_TYPE_SELL", 1)):
        return "SELL"
    return f"UNKNOWN_{pos_type}"


def _matching_model_b_position(positions: list[dict[str, Any]], controls: FrozenBrokerControls) -> dict[str, Any] | None:
    if len(positions) != 1:
        return None
    position = positions[0]
    if str(position.get("symbol", "")).upper() != controls.symbol.upper():
        return None
    magic = _int_or_none(position.get("magic"))
    # Some brokers may omit magic from positions in netting mode.  If there is
    # exactly one XAUUSD position and this controlled script owns the state, it
    # is still safe to use the position ticket for closing.
    if magic not in {None, 0, int(controls.model_b_magic_number)}:
        return None
    return position


def execute_model_b_order(
    *,
    proxy: GuardedMt5TinyOrderProxy,
    controls: FrozenBrokerControls,
    decision: LiveDecision,
    run_id: str,
    current_position: Mapping[str, Any] | None,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[LiveDecision, OrderSendEvent, list[dict[str, Any]], dict[str, Any] | None]:
    """Execute the ENTER_LONG or EXIT_LONG decision and verify broker state."""
    symbol_info = inspect_symbol_for_live_runtime(
        proxy,
        controls,
        enforce_entry_spread=decision.action == "ENTER_LONG",
    )
    filling_name, filling_value = choose_filling_candidates(proxy, symbol_info)[0]
    tick = inspect_tick_for_order_check(proxy, controls.symbol)

    if decision.action == "ENTER_LONG":
        side = "BUY"
        comment = compact_comment(MODEL_B_MAGIC_COMMENT_PREFIX, run_id, decision.event_time_utc, "_O")
        request = build_deal_request(
            mt5=proxy,
            controls=controls,
            side=side,
            tick=tick,
            filling_value=filling_value,
            magic=controls.model_b_magic_number,
            comment=comment,
        )
        require_position = False
        expected_position_side = "BUY"
        purpose = "ENTER_LONG"
    elif decision.action in {"EXIT_LONG", "FORCED_CLOSE_END_OF_RUN"}:
        if current_position is None:
            raise Stage3Step3BLiveError("Cannot close because no current Model B position was provided")
        side = "SELL"
        ticket = _int_or_none(current_position.get("ticket")) or _int_or_none(current_position.get("identifier"))
        if ticket is None or ticket <= 0:
            raise Stage3Step3BLiveError(f"Cannot close because position ticket is unavailable: {current_position}")
        volume = _float_or_none(current_position.get("volume")) or controls.stage3_first_order_test_volume_lots
        comment = compact_comment(MODEL_B_MAGIC_COMMENT_PREFIX, run_id, decision.event_time_utc, "_C")
        request = build_deal_request(
            mt5=proxy,
            controls=controls,
            side=side,
            tick=tick,
            filling_value=filling_value,
            magic=controls.model_b_magic_number,
            comment=comment,
            position_ticket=ticket,
            volume=volume,
        )
        require_position = True
        expected_position_side = "BUY"
        purpose = decision.action
    else:
        raise Stage3Step3BLiveError(f"Decision is not executable: {decision.action}")

    check_result, margin_required = run_order_check_for_request(proxy, request)
    check_retcode = _int_or_none((check_result or {}).get("retcode"))
    if check_retcode != 0:
        raise Stage3Step3BLiveError(f"{purpose} order_check failed before send: {check_result}")

    auth = SendAuthorisation(
        purpose=purpose,
        expected_side=side,
        expected_symbol=controls.symbol,
        expected_volume=float(request["volume"]),
        expected_magic=controls.model_b_magic_number,
        comment_prefix=MODEL_B_MAGIC_COMMENT_PREFIX,
        require_position=require_position,
    )
    send_result = call_order_send_compat(proxy, request, auth)
    event = build_send_event(
        mt5=proxy,
        purpose=purpose,
        side=side,
        request=request,
        check_result=check_result,
        send_result_raw=send_result,
        margin_required=margin_required,
        filling_name=filling_name,
        filling_value=filling_value,
    )
    if not event.passed:
        raise Stage3Step3BLiveError(f"{purpose} order_send failed: {asdict(event)}")

    if decision.action == "ENTER_LONG":
        positions = wait_for_position(
            proxy,
            controls=controls,
            expected_side=expected_position_side,
            expected_volume=float(request["volume"]),
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        opened = positions[0]
        updated = update_decision_after_send(
            decision,
            check_result=check_result,
            margin_required=margin_required,
            send_event=event,
            broker_position_ticket=_int_or_none(opened.get("ticket")) or _int_or_none(opened.get("identifier")),
        )
        return updated, event, positions, opened

    positions_after_close = wait_for_no_position(
        proxy,
        controls=controls,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    updated = update_decision_after_send(
        decision,
        check_result=check_result,
        margin_required=margin_required,
        send_event=event,
        broker_position_ticket=None,
    )
    return updated, event, positions_after_close, None


def run_live_iteration(
    *,
    mt5_module: Any,
    terminal_path: str | None,
    controls: FrozenBrokerControls,
    signal: Mapping[str, Any],
    state: ModelBLiveState,
    rules: ModelBDryRunRules,
    run_id: str,
    iteration: int,
    mode: str,
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.5,
) -> tuple[LiveDecision, ModelBLiveState, dict[str, Any]]:
    """Run one controlled live iteration for a single completed M15 signal."""
    event_time = _str_or_none(signal.get("event_time_utc"))
    if event_time and state.last_event_time_utc == event_time:
        decision = decide_model_b_live_action(
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
        return decision, state, {"broker_context_skipped": "duplicate_event"}

    proxy = GuardedMt5TinyOrderProxy(mt5_module, controls)
    initialized = False
    shutdown_called = False
    order_events: list[OrderSendEvent] = []
    position_snapshots: list[dict[str, Any]] = []
    history_audit: dict[str, Any] = {}
    history_deals: list[dict[str, Any]] = []
    history_orders: list[dict[str, Any]] = []
    start_utc = datetime.now(timezone.utc)
    try:
        initialise_terminal(proxy, terminal_path)
        initialized = True
        package = build_package_snapshot(proxy)
        terminal = inspect_terminal_for_trading(proxy)
        account = inspect_account_for_trading(proxy, controls)
        symbol_info = inspect_symbol_for_live_runtime(proxy, controls)
        tick = inspect_tick_for_order_check(proxy, controls.symbol)
        positions = get_positions_for_symbol(proxy, controls.symbol)
        pending_orders = get_orders_for_symbol(proxy, controls.symbol)
        capital_review = capital_review_from_live_account(controls=controls, account=account, symbol_info=symbol_info)
        if capital_review.get("capstone_10x_leverage_cap_passed") is not True:
            raise Stage3Step3BLiveError("Live account does not pass capstone 10x leverage cap for 0.01 lot")

        position_snapshots.extend(snapshot_positions(label="before_decision", positions=positions))
        current_position = _matching_model_b_position(positions, controls) if positions else None
        spread = int(symbol_info.get("spread", -1))
        decision = decide_model_b_live_action(
            signal=signal,
            state=state,
            rules=rules,
            spread_points=spread,
            spread_gate_points=int(controls.max_spread_points_for_entry),
            actual_position_count=len(positions),
            pending_order_count=len(pending_orders),
            run_id=run_id,
            iteration=iteration,
            mode=mode,
        )

        opened_position: Mapping[str, Any] | None = None
        closed_position = False
        if decision.action in {"ENTER_LONG", "EXIT_LONG"}:
            try:
                decision, event, post_positions, opened_position = execute_model_b_order(
                    proxy=proxy,
                    controls=controls,
                    decision=decision,
                    run_id=run_id,
                    current_position=current_position,
                    timeout_seconds=timeout_seconds,
                    poll_seconds=poll_seconds,
                )
            except EntrySpreadBlockedError as exc:
                # The spread may widen between the decision snapshot and the
                # final pre-send check.  Block this entry and keep monitoring.
                decision = replace(
                    decision,
                    action="BLOCK_SPREAD",
                    reason="spread_above_frozen_entry_gate_before_send",
                    live_position_after=decision.live_position_before,
                    spread_points=exc.spread_points,
                    order_check_called=False,
                    order_check_passed=None,
                    order_check_retcode=None,
                    order_check_comment=None,
                    order_check_margin_required=None,
                    order_send_called=False,
                    order_send_passed=None,
                    order_send_retcode=None,
                    order_send_comment=None,
                    broker_position_ticket=None,
                    order_ticket=None,
                    decision_utc=datetime.now(timezone.utc).isoformat(),
                )
            else:
                order_events.append(event)
                label = "after_enter" if decision.action == "ENTER_LONG" else "after_exit"
                position_snapshots.extend(snapshot_positions(label=label, positions=post_positions))
                closed_position = decision.action == "EXIT_LONG" and event.passed
                order_ticket = _int_or_none((event.send_result or {}).get("order"))
                order_tickets = {ticket for ticket in {order_ticket, state.open_order_ticket} if ticket is not None}
                pos_ticket = (
                    _int_or_none((opened_position or {}).get("ticket"))
                    or _int_or_none((opened_position or {}).get("identifier"))
                    or state.position_ticket
                )
                history_audit = collect_history_rows(
                    proxy,
                    controls=controls,
                    start_utc=start_utc,
                    position_ticket=pos_ticket,
                    order_tickets=order_tickets,
                    retries=3,
                    retry_sleep_seconds=0.5,
                )
                history_deals = list(history_audit.get("history_deals_filtered", []))
                history_orders = list(history_audit.get("history_orders_filtered", []))

        next_state = apply_live_decision_to_state(
            state,
            decision,
            opened_position=opened_position,
            closed_position=closed_position,
        )
        proxy.shutdown()
        shutdown_called = True
        initialized = False
        context = {
            "package": package,
            "terminal": terminal,
            "account": account,
            "symbol_info": symbol_info,
            "tick": tick,
            "capital_review": capital_review,
            "positions_before": positions,
            "pending_orders_before": pending_orders,
            "position_snapshots": position_snapshots,
            "order_send_events": [asdict(event) for event in order_events],
            "history_deals_filtered": history_deals,
            "history_orders_filtered": history_orders,
            "history_audit": history_audit,
            "mt5_calls": tuple(proxy.calls),
            "forbidden_attempts": tuple(proxy.forbidden_attempts),
            "shutdown_called": shutdown_called,
        }
        if proxy.forbidden_attempts:
            raise Stage3Step3BLiveError(f"Forbidden MT5 calls in live execution: {proxy.forbidden_attempts}")
        return decision, next_state, context
    finally:
        if initialized:
            try:
                proxy.shutdown()
            except Exception:
                pass


def force_close_model_b_position(
    *,
    mt5_module: Any,
    terminal_path: str | None,
    controls: FrozenBrokerControls,
    state: ModelBLiveState,
    run_id: str,
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.5,
    reason: str = "end_of_controlled_run_safety_close",
) -> tuple[LiveDecision | None, ModelBLiveState, dict[str, Any]]:
    """Close a single remaining Model B XAUUSD position at the end of Step 3B."""
    proxy = GuardedMt5TinyOrderProxy(mt5_module, controls)
    initialized = False
    try:
        initialise_terminal(proxy, terminal_path)
        initialized = True
        positions = get_positions_for_symbol(proxy, controls.symbol)
        current_position = _matching_model_b_position(positions, controls) if positions else None
        if not current_position:
            proxy.shutdown()
            return None, state, {"force_close_skipped": "no_model_b_position"}
        event_time = datetime.now(timezone.utc).isoformat()
        decision = LiveDecision(
            run_id=run_id,
            iteration=-1,
            mode="force_close",
            event_time_utc=event_time,
            probability_up=None,
            action="FORCED_CLOSE_END_OF_RUN",
            reason=reason,
            live_position_before=1,
            live_position_after=0,
            duplicate_event=False,
            stale_event_warning=False,
            spread_points=None,
            spread_gate_points=int(controls.max_spread_points_for_entry),
            actual_position_count=len(positions),
            pending_order_count=len(get_orders_for_symbol(proxy, controls.symbol)),
            order_check_called=False,
            order_check_passed=None,
            order_check_retcode=None,
            order_check_comment=None,
            order_check_margin_required=None,
            order_send_called=False,
            order_send_passed=None,
            order_send_retcode=None,
            order_send_comment=None,
            broker_position_ticket=_int_or_none(current_position.get("ticket")) or _int_or_none(current_position.get("identifier")),
            order_ticket=None,
            decision_utc=datetime.now(timezone.utc).isoformat(),
        )
        decision, event, post_positions, _opened = execute_model_b_order(
            proxy=proxy,
            controls=controls,
            decision=decision,
            run_id=run_id,
            current_position=current_position,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        next_state = apply_live_decision_to_state(state, decision, closed_position=True)
        order_ticket = _int_or_none((event.send_result or {}).get("order"))
        history_audit = collect_history_rows(
            proxy,
            controls=controls,
            start_utc=datetime.now(timezone.utc),
            position_ticket=state.position_ticket or decision.broker_position_ticket,
            order_tickets={ticket for ticket in {order_ticket, state.open_order_ticket} if ticket is not None},
            retries=3,
            retry_sleep_seconds=0.5,
        )
        proxy.shutdown()
        initialized = False
        return decision, next_state, {
            "order_send_events": [asdict(event)],
            "position_snapshots": snapshot_positions(label="after_force_close", positions=post_positions),
            "history_deals_filtered": list(history_audit.get("history_deals_filtered", [])),
            "history_orders_filtered": list(history_audit.get("history_orders_filtered", [])),
            "history_audit": history_audit,
            "mt5_calls": tuple(proxy.calls),
            "forbidden_attempts": tuple(proxy.forbidden_attempts),
            "force_close_executed": True,
        }
    finally:
        if initialized:
            try:
                proxy.shutdown()
            except Exception:
                pass


def append_csv_row(path: Path, row: Mapping[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))


def write_csv_rows_atomic(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [_flatten_mapping(row) for row in rows]
    if fieldnames is None:
        fieldnames = []
        for row in flattened:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"], extrasaction="ignore")
        writer.writeheader()
        if flattened:
            writer.writerows(flattened)
    tmp.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _flatten_mapping(row: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        flat_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            out.update(_flatten_mapping(value, flat_key))
        elif isinstance(value, (list, tuple)):
            out[flat_key] = json.dumps(value, default=str)
        else:
            out[flat_key] = value
    return out


def summarise_live_decisions(decisions: list[LiveDecision]) -> dict[str, Any]:
    unique_events = {d.event_time_utc for d in decisions if d.event_time_utc and not d.duplicate_event and d.iteration > 0}
    actions: dict[str, int] = {}
    for decision in decisions:
        actions[decision.action] = actions.get(decision.action, 0) + 1
    return {
        "decision_count": len(decisions),
        "unique_completed_m15_events": len(unique_events),
        "action_counts": actions,
        "order_check_called_count": sum(1 for d in decisions if d.order_check_called),
        "order_send_called_count": sum(1 for d in decisions if d.order_send_called),
        "order_send_passed_count": sum(1 for d in decisions if d.order_send_passed is True),
        "enter_long_count": sum(1 for d in decisions if d.action == "ENTER_LONG"),
        "exit_long_count": sum(1 for d in decisions if d.action == "EXIT_LONG"),
        "forced_close_count": sum(1 for d in decisions if d.action == "FORCED_CLOSE_END_OF_RUN"),
        "final_live_position": decisions[-1].live_position_after if decisions else None,
        "final_live_position_name": position_name(decisions[-1].live_position_after) if decisions else None,
    }
