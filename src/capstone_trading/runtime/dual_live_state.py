"""Restart-safe strategy state and decision logic for the dual MT5 rehearsal.

The module contains no MetaTrader dependency.  It is intentionally pure so the
most important safety rules can be unit tested without a broker terminal:

* completed M15 events only;
* duplicate-event suppression;
* Model A frozen thresholds, minimum hold and daily policy-change cap;
* Model B frozen long-only thresholds and one-entry-per-UTC-day cap;
* gap exits;
* daily-loss, total-drawdown and emergency kill-switch flattening;
* atomic state and heartbeat persistence;
* explicit reconciliation between persistent strategy state and broker state.

Broker order construction and submission live in ``dual_live_execution.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import json
import math

from capstone_trading.policy.position_transition import (
    policy_event_units as shared_policy_event_units,
    resolve_position_transition,
)

M15_DELTA = timedelta(minutes=15)
POSITION_NAMES: Mapping[int, str] = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
VALID_POSITIONS = frozenset(POSITION_NAMES)


class DualLiveStateError(RuntimeError):
    """Raised when live state or a transition is unsafe or malformed."""


@dataclass(frozen=True)
class StrategyRules:
    role: str
    long_threshold: float
    short_threshold: float | None
    exit_threshold: float | None
    minimum_hold_bars: int
    max_policy_changes_per_utc_day: int | None
    max_successful_entries_per_utc_day: int | None
    long_only: bool
    reversal_policy_event_units: int = 1
    allow_risk_reducing_exit_when_capped: bool = True

    def validate(self) -> None:
        if self.role not in {"model_a", "model_b"}:
            raise DualLiveStateError(f"Unsupported role: {self.role!r}")
        if not 0.0 <= float(self.long_threshold) <= 1.0:
            raise DualLiveStateError("long_threshold must be in [0, 1]")
        if self.role == "model_a":
            if self.short_threshold is None:
                raise DualLiveStateError("Model A requires short_threshold")
            if not 0.0 <= float(self.short_threshold) < float(self.long_threshold) <= 1.0:
                raise DualLiveStateError("Invalid Model A threshold ordering")
            if self.minimum_hold_bars < 1:
                raise DualLiveStateError("Model A minimum_hold_bars must be positive")
            if self.max_policy_changes_per_utc_day is None or self.max_policy_changes_per_utc_day < 0:
                raise DualLiveStateError("Model A requires a non-negative daily policy cap")
        else:
            if self.exit_threshold is None:
                raise DualLiveStateError("Model B requires exit_threshold")
            if not 0.0 <= float(self.exit_threshold) <= float(self.long_threshold) <= 1.0:
                raise DualLiveStateError("Invalid Model B threshold ordering")
            if not self.long_only:
                raise DualLiveStateError("Model B must remain long-only")
            if self.minimum_hold_bars != 0:
                raise DualLiveStateError("Frozen Model B current requires minimum_hold_bars=0")
            if (
                self.max_successful_entries_per_utc_day is None
                or self.max_successful_entries_per_utc_day < 1
            ):
                raise DualLiveStateError("Model B requires a positive daily entry cap")


@dataclass(frozen=True)
class RiskRules:
    daily_loss_stop_simple_return: float = -0.02
    total_drawdown_stop: float = -0.15

    def validate(self) -> None:
        if not -1.0 < self.daily_loss_stop_simple_return < 0.0:
            raise DualLiveStateError("daily loss stop must be between -1 and 0")
        if not -1.0 < self.total_drawdown_stop < 0.0:
            raise DualLiveStateError("total drawdown stop must be between -1 and 0")


@dataclass
class DualLiveState:
    schema_version: int = 1
    role: str = ""
    execution_mode: str = "shadow"
    virtual_position: int = 0
    broker_position: int = 0
    broker_position_ticket: int | None = None
    broker_position_identifier: int | None = None
    broker_open_order_ticket: int | None = None
    open_event_time_utc: str | None = None
    last_event_time_utc: str | None = None
    hold_bars: int = 0
    flat_bars_since_exit: int = 1_000_000_000
    current_utc_date: str | None = None
    policy_changes_today: int = 0
    successful_entries_today: int = 0
    start_equity: float | None = None
    day_start_equity: float | None = None
    running_peak_equity: float | None = None
    latest_equity: float | None = None
    daily_return: float | None = None
    total_drawdown: float | None = None
    daily_stop_active: bool = False
    total_stop_active: bool = False
    kill_switch_active: bool = False
    reconciliation_status: str = "UNINITIALISED"
    reconciliation_incidents: int = 0
    records_written: int = 0
    order_send_calls: int = 0
    successful_order_sends: int = 0
    completed_entry_exit_cycles: int = 0
    restart_count: int = 0
    last_worker_pid: int | None = None
    updated_utc: str | None = None

    @property
    def position_name(self) -> str:
        return position_name(self.virtual_position)


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    position: int
    ticket: int | None
    identifier: int | None
    order_ticket: int | None
    volume: float | None
    magic: int | None
    symbol: str


@dataclass(frozen=True)
class BrokerSnapshot:
    account_login_masked: str
    account_equity: float
    account_balance: float
    symbol: str
    positions: tuple[BrokerPositionSnapshot, ...]
    pending_order_count: int
    connected: bool
    terminal_trade_allowed: bool
    account_trade_allowed: bool
    account_expert_allowed: bool
    trade_api_disabled: bool


@dataclass(frozen=True)
class ReconciliationResult:
    state: DualLiveState
    broker_position: int
    position_ticket: int | None
    blocked: bool
    status: str
    reason: str
    incident: bool


@dataclass(frozen=True)
class StrategyDecision:
    role: str
    run_id: str
    iteration: int
    event_time_utc: str | None
    probability_up: float | None
    execution_mode: str
    position_before: int
    desired_position: int
    target_position: int
    action: str
    reason: str
    duplicate_event: bool
    gap_from_previous_event: bool
    stale_event_warning: bool
    policy_event_units: int
    policy_changes_today_before: int
    successful_entries_today_before: int
    daily_stop_active: bool
    total_stop_active: bool
    kill_switch_active: bool
    reconciliation_status: str
    policy_cap_reached: bool = False
    entry_blocked_by_policy_cap: bool = False
    exit_allowed_when_capped: bool = False
    close_only_reversal: bool = False
    requested_policy_event_units: int = 0
    order_check_called: bool = False
    order_check_passed: bool | None = None
    order_send_called: bool = False
    order_send_passed: bool | None = None
    broker_position_after: int | None = None
    broker_position_ticket_after: int | None = None
    decision_utc: str = ""


@dataclass(frozen=True)
class RiskUpdate:
    state: DualLiveState
    day_reset: bool
    daily_stop_triggered_now: bool
    total_stop_triggered_now: bool


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def position_name(value: int) -> str:
    return POSITION_NAMES.get(int(value), f"UNKNOWN_{value}")


def parse_event_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DualLiveStateError(f"Invalid event timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def mask_login(value: Any) -> str:
    text = str(value or "").strip()
    return "*" * max(len(text) - 4, 0) + text[-4:]


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DualLiveStateError(f"{field_name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise DualLiveStateError(f"{field_name} is not finite: {value!r}")
    return number


def validate_state(state: DualLiveState) -> None:
    if state.schema_version != 1:
        raise DualLiveStateError(f"Unsupported state schema: {state.schema_version}")
    if state.role not in {"model_a", "model_b"}:
        raise DualLiveStateError(f"Invalid state role: {state.role!r}")
    if state.execution_mode not in {"shadow", "live"}:
        raise DualLiveStateError(f"Invalid execution mode: {state.execution_mode!r}")
    if state.virtual_position not in VALID_POSITIONS:
        raise DualLiveStateError("virtual_position must be -1, 0, or 1")
    if state.broker_position not in VALID_POSITIONS:
        raise DualLiveStateError("broker_position must be -1, 0, or 1")
    if state.role == "model_b" and state.virtual_position == -1:
        raise DualLiveStateError("Model B state cannot be short")
    for name in (
        "hold_bars",
        "flat_bars_since_exit",
        "policy_changes_today",
        "successful_entries_today",
        "reconciliation_incidents",
        "records_written",
        "order_send_calls",
        "successful_order_sends",
        "completed_entry_exit_cycles",
        "restart_count",
    ):
        if int(getattr(state, name)) < 0:
            raise DualLiveStateError(f"{name} must be non-negative")


def initial_state(role: str, *, execution_mode: str, worker_pid: int | None = None) -> DualLiveState:
    state = DualLiveState(
        role=role,
        execution_mode=execution_mode,
        last_worker_pid=worker_pid,
        updated_utc=utc_now_iso(),
    )
    validate_state(state)
    return state


def state_from_mapping(raw: Mapping[str, Any]) -> DualLiveState:
    allowed = {field_name for field_name in DualLiveState.__dataclass_fields__}
    payload = {key: value for key, value in raw.items() if key in allowed}
    state = DualLiveState(**payload)
    validate_state(state)
    return state


def load_state(path: Path, *, role: str, execution_mode: str, worker_pid: int | None = None) -> DualLiveState:
    if not path.exists():
        return initial_state(role, execution_mode=execution_mode, worker_pid=worker_pid)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise DualLiveStateError(f"Unable to read state file {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise DualLiveStateError(f"State file is not a JSON object: {path}")
    state = state_from_mapping(raw)
    if state.role != role:
        raise DualLiveStateError(
            f"State role mismatch. Expected {role!r}, found {state.role!r}: {path}"
        )
    if state.execution_mode != execution_mode:
        raise DualLiveStateError(
            f"State execution mode mismatch. Expected {execution_mode!r}, "
            f"found {state.execution_mode!r}: {path}"
        )
    restart_count = state.restart_count
    if worker_pid is not None and state.last_worker_pid not in {None, worker_pid}:
        restart_count += 1
    state = replace(
        state,
        restart_count=restart_count,
        last_worker_pid=worker_pid,
        updated_utc=utc_now_iso(),
    )
    validate_state(state)
    return state


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_state(path: Path, state: DualLiveState) -> None:
    validate_state(state)
    write_json_atomic(path, asdict(state))


def append_csv_atomic_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append a row after constructing a stable union of existing columns.

    Runtime decision files can gain fields between patch versions.  Rewriting
    through a temporary file avoids partial rows after a process interruption.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = [dict(item) for item in reader]
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(dict(row))
    temporary.replace(path)


def broker_position_from_plain(
    position: Mapping[str, Any],
    *,
    symbol: str,
    buy_type: int = 0,
    sell_type: int = 1,
) -> BrokerPositionSnapshot:
    position_symbol = str(position.get("symbol", ""))
    if position_symbol.upper() != symbol.upper():
        raise DualLiveStateError(
            f"Position symbol mismatch: expected {symbol}, found {position_symbol}"
        )
    position_type = int(position.get("type", -999))
    if position_type == int(buy_type):
        side = 1
    elif position_type == int(sell_type):
        side = -1
    else:
        raise DualLiveStateError(f"Unsupported broker position type: {position_type}")
    return BrokerPositionSnapshot(
        position=side,
        ticket=_int_or_none(position.get("ticket")),
        identifier=_int_or_none(position.get("identifier")),
        order_ticket=_int_or_none(position.get("order")),
        volume=_float_or_none(position.get("volume")),
        magic=_int_or_none(position.get("magic")),
        symbol=position_symbol,
    )


def reconcile_state(
    state: DualLiveState,
    broker: BrokerSnapshot,
    *,
    expected_magic: int,
    execution_mode: str,
) -> ReconciliationResult:
    """Reconcile persistent state with the actual broker position.

    In shadow mode the virtual strategy state is independent and the broker must
    stay flat.  In live mode the broker is authoritative after a restart.  A
    single position with the expected magic, or with broker-omitted magic 0/None,
    can be adopted.  Multiple positions, pending orders, a foreign magic number,
    or a Model B short position are hard blocks.
    """

    if execution_mode not in {"shadow", "live"}:
        raise DualLiveStateError(f"Unsupported execution mode: {execution_mode}")
    positions = tuple(broker.positions)
    incident = False
    if not broker.connected:
        return ReconciliationResult(
            state=replace(state, reconciliation_status="BLOCK_DISCONNECTED"),
            broker_position=state.broker_position,
            position_ticket=state.broker_position_ticket,
            blocked=True,
            status="BLOCK_DISCONNECTED",
            reason="terminal_not_connected",
            incident=True,
        )
    if broker.pending_order_count > 0:
        return ReconciliationResult(
            state=replace(
                state,
                reconciliation_status="BLOCK_PENDING_ORDER",
                reconciliation_incidents=state.reconciliation_incidents + 1,
                updated_utc=utc_now_iso(),
            ),
            broker_position=state.broker_position,
            position_ticket=state.broker_position_ticket,
            blocked=True,
            status="BLOCK_PENDING_ORDER",
            reason="pending_xauusd_order_exists",
            incident=True,
        )
    if len(positions) > 1:
        return ReconciliationResult(
            state=replace(
                state,
                reconciliation_status="BLOCK_MULTIPLE_POSITIONS",
                reconciliation_incidents=state.reconciliation_incidents + 1,
                updated_utc=utc_now_iso(),
            ),
            broker_position=state.broker_position,
            position_ticket=state.broker_position_ticket,
            blocked=True,
            status="BLOCK_MULTIPLE_POSITIONS",
            reason="more_than_one_xauusd_position_exists",
            incident=True,
        )
    actual_position = 0
    ticket = None
    identifier = None
    order_ticket = None
    if positions:
        item = positions[0]
        if item.magic not in {None, 0, int(expected_magic)}:
            return ReconciliationResult(
                state=replace(
                    state,
                    reconciliation_status="BLOCK_FOREIGN_MAGIC",
                    reconciliation_incidents=state.reconciliation_incidents + 1,
                    updated_utc=utc_now_iso(),
                ),
                broker_position=item.position,
                position_ticket=item.ticket or item.identifier,
                blocked=True,
                status="BLOCK_FOREIGN_MAGIC",
                reason=f"unexpected_position_magic_{item.magic}",
                incident=True,
            )
        actual_position = int(item.position)
        ticket = item.ticket or item.identifier
        identifier = item.identifier
        order_ticket = item.order_ticket
    if state.role == "model_b" and actual_position == -1:
        return ReconciliationResult(
            state=replace(
                state,
                reconciliation_status="BLOCK_MODEL_B_SHORT",
                reconciliation_incidents=state.reconciliation_incidents + 1,
                updated_utc=utc_now_iso(),
            ),
            broker_position=actual_position,
            position_ticket=ticket,
            blocked=True,
            status="BLOCK_MODEL_B_SHORT",
            reason="model_b_short_position_is_forbidden",
            incident=True,
        )

    if execution_mode == "shadow":
        if actual_position != 0:
            return ReconciliationResult(
                state=replace(
                    state,
                    broker_position=actual_position,
                    broker_position_ticket=ticket,
                    broker_position_identifier=identifier,
                    broker_open_order_ticket=order_ticket,
                    reconciliation_status="BLOCK_EXPOSURE_IN_SHADOW",
                    reconciliation_incidents=state.reconciliation_incidents + 1,
                    updated_utc=utc_now_iso(),
                ),
                broker_position=actual_position,
                position_ticket=ticket,
                blocked=True,
                status="BLOCK_EXPOSURE_IN_SHADOW",
                reason="orders_disabled_but_broker_position_exists",
                incident=True,
            )
        next_state = replace(
            state,
            broker_position=0,
            broker_position_ticket=None,
            broker_position_identifier=None,
            broker_open_order_ticket=None,
            reconciliation_status="PASS_SHADOW_BROKER_FLAT",
            updated_utc=utc_now_iso(),
        )
        return ReconciliationResult(
            state=next_state,
            broker_position=0,
            position_ticket=None,
            blocked=False,
            status="PASS_SHADOW_BROKER_FLAT",
            reason="broker_flat_virtual_strategy_state_preserved",
            incident=False,
        )

    virtual_before = int(state.virtual_position)
    if virtual_before != actual_position:
        incident = True
    conservative_policy_changes = state.policy_changes_today
    conservative_successful_entries = state.successful_entries_today
    if incident:
        # A broker/state mismatch can be the crash window after a successful
        # order_send but before the atomic state write.  Conservatively count
        # the adopted transition so a restart can never grant an extra daily
        # policy change or an extra Model B entry.
        conservative_policy_changes += 1
        if virtual_before == 0 and actual_position != 0:
            conservative_successful_entries += 1

    next_state = replace(
        state,
        virtual_position=actual_position,
        broker_position=actual_position,
        broker_position_ticket=ticket,
        broker_position_identifier=identifier,
        broker_open_order_ticket=order_ticket,
        open_event_time_utc=(
            state.open_event_time_utc if actual_position != 0 else None
        ),
        hold_bars=(state.hold_bars if actual_position == virtual_before else 0),
        flat_bars_since_exit=(
            0
            if actual_position == 0 and virtual_before != 0
            else (
                state.flat_bars_since_exit
                if actual_position == 0
                else 1_000_000_000
            )
        ),
        policy_changes_today=int(conservative_policy_changes),
        successful_entries_today=int(conservative_successful_entries),
        reconciliation_status=(
            "PASS_STATE_MATCHES_BROKER"
            if not incident
            else "PASS_BROKER_STATE_ADOPTED"
        ),
        reconciliation_incidents=(
            state.reconciliation_incidents + (1 if incident else 0)
        ),
        updated_utc=utc_now_iso(),
    )
    return ReconciliationResult(
        state=next_state,
        broker_position=actual_position,
        position_ticket=ticket,
        blocked=False,
        status=next_state.reconciliation_status,
        reason=(
            "persistent_state_matches_broker"
            if not incident
            else "broker_position_adopted_after_restart_or_external_change"
        ),
        incident=incident,
    )


def update_risk_state(
    state: DualLiveState,
    *,
    equity: float,
    event_time_utc: str | None,
    rules: RiskRules,
    kill_switch_active: bool,
) -> RiskUpdate:
    rules.validate()
    equity_value = _finite_float(equity, field_name="account equity")
    if equity_value <= 0.0:
        raise DualLiveStateError("Account equity must be positive")
    event = parse_event_time(event_time_utc) or datetime.now(timezone.utc)
    day_key = event.date().isoformat()
    day_reset = state.current_utc_date != day_key
    start_equity = state.start_equity or equity_value
    day_start_equity = equity_value if day_reset or state.day_start_equity is None else state.day_start_equity
    running_peak = max(state.running_peak_equity or equity_value, equity_value)
    daily_return = equity_value / day_start_equity - 1.0
    total_drawdown = equity_value / running_peak - 1.0
    prior_daily = False if day_reset else state.daily_stop_active
    daily_active = prior_daily or daily_return <= rules.daily_loss_stop_simple_return
    total_active = state.total_stop_active or total_drawdown <= rules.total_drawdown_stop
    next_state = replace(
        state,
        current_utc_date=day_key,
        policy_changes_today=(0 if day_reset else state.policy_changes_today),
        successful_entries_today=(0 if day_reset else state.successful_entries_today),
        start_equity=float(start_equity),
        day_start_equity=float(day_start_equity),
        running_peak_equity=float(running_peak),
        latest_equity=float(equity_value),
        daily_return=float(daily_return),
        total_drawdown=float(total_drawdown),
        daily_stop_active=bool(daily_active),
        total_stop_active=bool(total_active),
        kill_switch_active=bool(kill_switch_active),
        updated_utc=utc_now_iso(),
    )
    return RiskUpdate(
        state=next_state,
        day_reset=day_reset,
        daily_stop_triggered_now=bool(daily_active and not prior_daily),
        total_stop_triggered_now=bool(total_active and not state.total_stop_active),
    )


def desired_position_from_probability(
    probability_up: float,
    *,
    current_position: int,
    rules: StrategyRules,
) -> int:
    rules.validate()
    probability = _finite_float(probability_up, field_name="probability_up")
    if not 0.0 <= probability <= 1.0:
        raise DualLiveStateError("probability_up must be in [0, 1]")
    if rules.role == "model_a":
        assert rules.short_threshold is not None
        if probability >= rules.long_threshold:
            return 1
        if probability <= rules.short_threshold:
            return -1
        return 0
    assert rules.exit_threshold is not None
    if current_position == 1:
        return 1 if probability >= rules.exit_threshold else 0
    return 1 if probability >= rules.long_threshold else 0


def policy_event_units(previous: int, target: int, rules: StrategyRules) -> int:
    return shared_policy_event_units(
        previous,
        target,
        reversal_policy_event_units=rules.reversal_policy_event_units,
    )


def _decision(
    *,
    state: DualLiveState,
    rules: StrategyRules,
    run_id: str,
    iteration: int,
    event_time_utc: str | None,
    probability_up: float | None,
    desired: int,
    target: int,
    action: str,
    reason: str,
    duplicate: bool,
    gap: bool,
    stale: bool,
    units: int = 0,
    requested_units: int | None = None,
    policy_cap_reached: bool = False,
    entry_blocked_by_policy_cap: bool = False,
    exit_allowed_when_capped: bool = False,
    close_only_reversal: bool = False,
) -> StrategyDecision:
    return StrategyDecision(
        role=state.role,
        run_id=run_id,
        iteration=int(iteration),
        event_time_utc=event_time_utc,
        probability_up=probability_up,
        execution_mode=state.execution_mode,
        position_before=int(state.virtual_position),
        desired_position=int(desired),
        target_position=int(target),
        action=action,
        reason=reason,
        duplicate_event=bool(duplicate),
        gap_from_previous_event=bool(gap),
        stale_event_warning=bool(stale),
        policy_event_units=int(units),
        policy_changes_today_before=int(state.policy_changes_today),
        successful_entries_today_before=int(state.successful_entries_today),
        daily_stop_active=bool(state.daily_stop_active),
        total_stop_active=bool(state.total_stop_active),
        kill_switch_active=bool(state.kill_switch_active),
        reconciliation_status=str(state.reconciliation_status),
        policy_cap_reached=bool(policy_cap_reached),
        entry_blocked_by_policy_cap=bool(entry_blocked_by_policy_cap),
        exit_allowed_when_capped=bool(exit_allowed_when_capped),
        close_only_reversal=bool(close_only_reversal),
        requested_policy_event_units=int(
            units if requested_units is None else requested_units
        ),
        decision_utc=utc_now_iso(),
    )


def decide_strategy_transition(
    state: DualLiveState,
    *,
    rules: StrategyRules,
    run_id: str,
    iteration: int,
    event_time_utc: str | None,
    probability_up: float | None,
    stale_event_warning: bool,
    reconciliation_blocked: bool,
    reconciliation_reason: str = "",
) -> StrategyDecision:
    """Apply frozen strategy and safety rules to one completed M15 event.

    Safety exits are evaluated before duplicate suppression.  This is critical:
    a kill switch, daily stop, total stop, or broker-state incident may become
    active between M15 closes and must be able to flatten an existing position
    even when the latest model event has already been processed.
    """

    validate_state(state)
    rules.validate()
    event = parse_event_time(event_time_utc)
    previous_event = parse_event_time(state.last_event_time_utc)
    duplicate = bool(
        event is not None
        and previous_event is not None
        and event == previous_event
    )
    gap = bool(
        event is not None
        and previous_event is not None
        and event - previous_event != M15_DELTA
    )
    current = int(state.virtual_position)

    if reconciliation_blocked:
        return _decision(
            state=state,
            rules=rules,
            run_id=run_id,
            iteration=iteration,
            event_time_utc=event_time_utc,
            probability_up=probability_up,
            desired=current,
            target=current,
            action="BLOCK_RECONCILIATION",
            reason=reconciliation_reason or "broker_state_unresolved",
            duplicate=duplicate,
            gap=gap,
            stale=stale_event_warning,
        )
    if state.kill_switch_active:
        target = 0
        return _decision(
            state=state,
            rules=rules,
            run_id=run_id,
            iteration=iteration,
            event_time_utc=event_time_utc,
            probability_up=probability_up,
            desired=0,
            target=target,
            action="KILL_SWITCH_FLATTEN" if current != 0 else "KILL_SWITCH_BLOCK",
            reason="emergency_kill_switch_active",
            duplicate=(duplicate and current == 0),
            gap=gap,
            stale=stale_event_warning,
        )
    if state.total_stop_active:
        return _decision(
            state=state,
            rules=rules,
            run_id=run_id,
            iteration=iteration,
            event_time_utc=event_time_utc,
            probability_up=probability_up,
            desired=0,
            target=0,
            action="TOTAL_STOP_FLATTEN" if current != 0 else "TOTAL_STOP_BLOCK",
            reason="total_drawdown_stop_active",
            duplicate=(duplicate and current == 0),
            gap=gap,
            stale=stale_event_warning,
        )
    if state.daily_stop_active:
        return _decision(
            state=state,
            rules=rules,
            run_id=run_id,
            iteration=iteration,
            event_time_utc=event_time_utc,
            probability_up=probability_up,
            desired=0,
            target=0,
            action="DAILY_STOP_FLATTEN" if current != 0 else "DAILY_STOP_BLOCK",
            reason="daily_loss_stop_active_until_next_utc_day",
            duplicate=(duplicate and current == 0),
            gap=gap,
            stale=stale_event_warning,
        )
    if event is None or probability_up is None:
        return _decision(
            state=state,
            rules=rules,
            run_id=run_id,
            iteration=iteration,
            event_time_utc=event_time_utc,
            probability_up=probability_up,
            desired=current,
            target=current,
            action="BLOCK_INVALID_SIGNAL",
            reason="missing_event_time_or_probability",
            duplicate=False,
            gap=gap,
            stale=stale_event_warning,
        )
    if duplicate:
        return _decision(
            state=state,
            rules=rules,
            run_id=run_id,
            iteration=iteration,
            event_time_utc=event_time_utc,
            probability_up=probability_up,
            desired=current,
            target=current,
            action="DUPLICATE_SKIP",
            reason="event_already_processed",
            duplicate=True,
            gap=False,
            stale=stale_event_warning,
        )
    if gap:
        return _decision(
            state=state,
            rules=rules,
            run_id=run_id,
            iteration=iteration,
            event_time_utc=event_time_utc,
            probability_up=probability_up,
            desired=0,
            target=0,
            action="GAP_FLATTEN" if current != 0 else "GAP_BLOCK",
            reason="non_contiguous_completed_m15_event",
            duplicate=False,
            gap=True,
            stale=stale_event_warning,
        )
    if stale_event_warning:
        return _decision(
            state=state,
            rules=rules,
            run_id=run_id,
            iteration=iteration,
            event_time_utc=event_time_utc,
            probability_up=probability_up,
            desired=current,
            target=current,
            action="BLOCK_STALE_EVENT",
            reason="latest_valid_sequence_not_aligned_to_latest_completed_bar",
            duplicate=False,
            gap=False,
            stale=True,
        )

    desired = desired_position_from_probability(
        float(probability_up),
        current_position=current,
        rules=rules,
    )
    if desired == current:
        return _decision(
            state=state,
            rules=rules,
            run_id=run_id,
            iteration=iteration,
            event_time_utc=event_time_utc,
            probability_up=float(probability_up),
            desired=desired,
            target=current,
            action="HOLD_FLAT" if current == 0 else f"HOLD_{position_name(current)}",
            reason="probability_preserves_current_position",
            duplicate=False,
            gap=False,
            stale=False,
        )

    if rules.role == "model_a":
        active_hold_blocked = current != 0 and state.hold_bars <= rules.minimum_hold_bars
        flat_cooldown_blocked = (
            current == 0
            and desired != 0
            and state.flat_bars_since_exit < rules.minimum_hold_bars
        )
        if active_hold_blocked or flat_cooldown_blocked:
            return _decision(
                state=state,
                rules=rules,
                run_id=run_id,
                iteration=iteration,
                event_time_utc=event_time_utc,
                probability_up=float(probability_up),
                desired=desired,
                target=current,
                action="BLOCK_MINIMUM_HOLD",
                reason="minimum_hold_or_flat_cooldown_active",
                duplicate=False,
                gap=False,
                stale=False,
            )
        assert rules.max_policy_changes_per_utc_day is not None
        resolution = resolve_position_transition(
            current_position=current,
            desired_position=desired,
            policy_changes_today=state.policy_changes_today,
            max_policy_changes_per_day=rules.max_policy_changes_per_utc_day,
            reversal_policy_event_units=rules.reversal_policy_event_units,
            allow_risk_reducing_exit_when_capped=(
                rules.allow_risk_reducing_exit_when_capped
            ),
        )
        units = int(resolution.consumed_policy_units)
        if not resolution.transition_allowed:
            return _decision(
                state=state,
                rules=rules,
                run_id=run_id,
                iteration=iteration,
                event_time_utc=event_time_utc,
                probability_up=float(probability_up),
                desired=desired,
                target=current,
                action="BLOCK_DAILY_POLICY_CAP",
                reason=resolution.reason,
                duplicate=False,
                gap=False,
                stale=False,
                units=0,
                requested_units=resolution.requested_policy_units,
                policy_cap_reached=resolution.cap_reached,
                entry_blocked_by_policy_cap=resolution.entry_blocked,
                exit_allowed_when_capped=resolution.exit_allowed,
                close_only_reversal=resolution.close_only_reversal,
            )
        if resolution.close_only_reversal:
            return _decision(
                state=state,
                rules=rules,
                run_id=run_id,
                iteration=iteration,
                event_time_utc=event_time_utc,
                probability_up=float(probability_up),
                desired=desired,
                target=resolution.effective_target_position,
                action="CLOSE_ONLY_DAILY_POLICY_CAP",
                reason=resolution.reason,
                duplicate=False,
                gap=False,
                stale=False,
                units=units,
                requested_units=resolution.requested_policy_units,
                policy_cap_reached=True,
                entry_blocked_by_policy_cap=True,
                exit_allowed_when_capped=True,
                close_only_reversal=True,
            )
        if resolution.cap_reached and resolution.exit_allowed:
            return _decision(
                state=state,
                rules=rules,
                run_id=run_id,
                iteration=iteration,
                event_time_utc=event_time_utc,
                probability_up=float(probability_up),
                desired=desired,
                target=resolution.effective_target_position,
                action="EXIT_POSITION_CAP_REACHED",
                reason=resolution.reason,
                duplicate=False,
                gap=False,
                stale=False,
                units=units,
                requested_units=resolution.requested_policy_units,
                policy_cap_reached=True,
                exit_allowed_when_capped=True,
            )
    else:
        units = 1 if current != desired else 0
        if current == 0 and desired == 1:
            assert rules.max_successful_entries_per_utc_day is not None
            if state.successful_entries_today >= rules.max_successful_entries_per_utc_day:
                return _decision(
                    state=state,
                    rules=rules,
                    run_id=run_id,
                    iteration=iteration,
                    event_time_utc=event_time_utc,
                    probability_up=float(probability_up),
                    desired=desired,
                    target=current,
                    action="BLOCK_DAILY_ENTRY_CAP",
                    reason="maximum_successful_new_entries_per_utc_day_reached",
                    duplicate=False,
                    gap=False,
                    stale=False,
                )

    if current == 0 and desired == 1:
        action = "ENTER_LONG"
    elif current == 0 and desired == -1:
        action = "ENTER_SHORT"
    elif current != 0 and desired == 0:
        action = "EXIT_POSITION"
    elif current == 1 and desired == -1:
        action = "REVERSE_LONG_TO_SHORT"
    elif current == -1 and desired == 1:
        action = "REVERSE_SHORT_TO_LONG"
    else:
        action = "CHANGE_POSITION"
    return _decision(
        state=state,
        rules=rules,
        run_id=run_id,
        iteration=iteration,
        event_time_utc=event_time_utc,
        probability_up=float(probability_up),
        desired=desired,
        target=desired,
        action=action,
        reason="frozen_overlay_transition",
        duplicate=False,
        gap=False,
        stale=False,
        units=units,
    )


def transition_requires_order(decision: StrategyDecision) -> bool:
    return decision.target_position != decision.position_before


def apply_transition(
    state: DualLiveState,
    decision: StrategyDecision,
    *,
    confirmed_position: int,
    broker_ticket: int | None,
    broker_identifier: int | None = None,
    broker_order_ticket: int | None = None,
    order_send_calls: int = 0,
    successful_order_sends: int = 0,
) -> DualLiveState:
    """Persist a processed event after a virtual or broker-confirmed transition."""

    if decision.duplicate_event:
        return state
    if confirmed_position not in VALID_POSITIONS:
        raise DualLiveStateError(f"Invalid confirmed position: {confirmed_position}")
    if state.role == "model_b" and confirmed_position == -1:
        raise DualLiveStateError("Model B cannot confirm a short position")
    previous = int(state.virtual_position)
    target = int(confirmed_position)
    forced_gap = decision.action == "GAP_FLATTEN"
    if target == 0:
        next_hold = 0
        if previous == 0:
            next_flat = min(state.flat_bars_since_exit + 1, 1_000_000_000)
        elif forced_gap:
            next_flat = 1_000_000_000
        else:
            next_flat = 0
    elif target == previous:
        next_hold = state.hold_bars + 1 if previous != 0 else 1
        next_flat = 1_000_000_000
    else:
        next_hold = 1
        next_flat = 1_000_000_000

    policy_changes = state.policy_changes_today
    successful_entries = state.successful_entries_today
    completed_cycles = state.completed_entry_exit_cycles
    if target != previous and int(decision.policy_event_units) > 0:
        policy_changes += int(decision.policy_event_units)
    if previous == 0 and target != 0:
        successful_entries += 1
    if previous != 0 and target == 0:
        completed_cycles += 1
    open_event = state.open_event_time_utc
    if previous == 0 and target != 0:
        open_event = decision.event_time_utc
    elif target == 0:
        open_event = None

    next_state = replace(
        state,
        virtual_position=target,
        broker_position=(target if state.execution_mode == "live" else state.broker_position),
        broker_position_ticket=(broker_ticket if target != 0 else None),
        broker_position_identifier=(broker_identifier if target != 0 else None),
        broker_open_order_ticket=(broker_order_ticket if target != 0 else None),
        open_event_time_utc=open_event,
        last_event_time_utc=decision.event_time_utc or state.last_event_time_utc,
        hold_bars=int(next_hold),
        flat_bars_since_exit=int(next_flat),
        policy_changes_today=int(policy_changes),
        successful_entries_today=int(successful_entries),
        records_written=state.records_written + 1,
        order_send_calls=state.order_send_calls + int(order_send_calls),
        successful_order_sends=state.successful_order_sends + int(successful_order_sends),
        completed_entry_exit_cycles=int(completed_cycles),
        updated_utc=utc_now_iso(),
    )
    validate_state(next_state)
    return next_state


def advance_blocked_or_hold_state(
    state: DualLiveState,
    decision: StrategyDecision,
) -> DualLiveState:
    """Advance counters for a processed non-transition event."""

    return apply_transition(
        state,
        decision,
        confirmed_position=state.virtual_position,
        broker_ticket=state.broker_position_ticket,
        broker_identifier=state.broker_position_identifier,
        broker_order_ticket=state.broker_open_order_ticket,
    )


def decision_to_mapping(decision: StrategyDecision) -> dict[str, Any]:
    payload = asdict(decision)
    payload["position_before_name"] = position_name(decision.position_before)
    payload["desired_position_name"] = position_name(decision.desired_position)
    payload["target_position_name"] = position_name(decision.target_position)
    return payload


def update_decision_execution(
    decision: StrategyDecision,
    *,
    order_check_called: bool,
    order_check_passed: bool | None,
    order_send_called: bool,
    order_send_passed: bool | None,
    broker_position_after: int,
    broker_position_ticket_after: int | None,
) -> StrategyDecision:
    return replace(
        decision,
        order_check_called=bool(order_check_called),
        order_check_passed=order_check_passed,
        order_send_called=bool(order_send_called),
        order_send_passed=order_send_passed,
        broker_position_after=int(broker_position_after),
        broker_position_ticket_after=broker_position_ticket_after,
    )


def heartbeat_payload(
    *,
    state: DualLiveState,
    run_id: str,
    pid: int,
    status: str,
    message: str,
    last_decision: StrategyDecision | None,
    started_utc: str,
    orders_enabled: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "role": state.role,
        "pid": int(pid),
        "status": status,
        "message": message,
        "started_utc": started_utc,
        "updated_utc": utc_now_iso(),
        "orders_enabled": bool(orders_enabled),
        "execution_mode": state.execution_mode,
        "virtual_position": int(state.virtual_position),
        "virtual_position_name": position_name(state.virtual_position),
        "broker_position": int(state.broker_position),
        "broker_position_name": position_name(state.broker_position),
        "last_event_time_utc": state.last_event_time_utc,
        "records_written": int(state.records_written),
        "restart_count": int(state.restart_count),
        "daily_stop_active": bool(state.daily_stop_active),
        "total_stop_active": bool(state.total_stop_active),
        "kill_switch_active": bool(state.kill_switch_active),
        "reconciliation_status": state.reconciliation_status,
        "last_action": None if last_decision is None else last_decision.action,
        "last_reason": None if last_decision is None else last_decision.reason,
    }


def summarise_decisions(decisions: Iterable[StrategyDecision]) -> dict[str, Any]:
    items = list(decisions)
    action_counts: dict[str, int] = {}
    unique_events: set[str] = set()
    for item in items:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
        if item.event_time_utc and not item.duplicate_event:
            unique_events.add(item.event_time_utc)
    return {
        "decision_count": len(items),
        "unique_completed_m15_events": len(unique_events),
        "action_counts": action_counts,
        "order_check_called_count": sum(1 for item in items if item.order_check_called),
        "order_send_called_count": sum(1 for item in items if item.order_send_called),
        "order_send_passed_count": sum(1 for item in items if item.order_send_passed is True),
        "duplicate_event_count": sum(1 for item in items if item.duplicate_event),
        "gap_event_count": sum(1 for item in items if item.gap_from_previous_event),
    }


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
