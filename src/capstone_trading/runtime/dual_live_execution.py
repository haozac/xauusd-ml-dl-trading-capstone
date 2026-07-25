"""Broker inspection and guarded order execution for the dual rehearsal.

The module deliberately reuses the repositories existing Stage 3 guarded MT5
plumbing.  It does not load the prediction model and it does not decide whether
a trade should occur.  It receives an already-audited target position and then:

* validates the exact demo account suffix;
* validates terminal/account trading permissions;
* inspects XAUUSD positions and pending orders;
* calls ``order_check`` before every ``order_send``;
* limits each order to the frozen 0.01-lot controls;
* closes before opening during a reversal;
* blocks new entries when the spread exceeds the frozen entry gate;
* never blocks a risk-reducing close because of a wide spread;
* confirms the final broker position after every transition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
import json
import math

from capstone_trading.runtime.dual_live_state import (
    BrokerSnapshot,
    broker_position_from_plain,
)
from capstone_trading.runtime.mt5_readiness import object_to_plain_dict, safe_last_error
from capstone_trading.runtime.order_execution_probe import (
    GuardedMt5TinyOrderProxy,
    OrderSendEvent,
    SendAuthorisation,
    build_deal_request,
    build_send_event,
    call_order_send_compat,
    get_orders_for_symbol,
    object_list_to_plain,
    get_positions_for_symbol,
    run_order_check_for_request,
    wait_for_no_position,
    wait_for_position,
)
from capstone_trading.runtime.order_preflight import (
    FrozenBrokerControls,
    capital_review_from_live_account,
    choose_filling_candidates,
    inspect_account_for_trading,
    inspect_terminal_for_trading,
    inspect_tick_for_order_check,
    initialise_terminal,
)


class DualLiveExecutionError(RuntimeError):
    """Raised when broker inspection or a guarded transition is unsafe."""


class EntrySpreadBlocked(DualLiveExecutionError):
    """A normal, non-fatal block when a new entry spread is too wide."""

    def __init__(self, spread_points: int | None, gate_points: int):
        self.spread_points = spread_points
        self.gate_points = int(gate_points)
        super().__init__(
            f"Entry spread {spread_points!r} exceeds frozen gate {gate_points}"
        )


@dataclass(frozen=True)
class BrokerInspection:
    snapshot: BrokerSnapshot
    package: Mapping[str, Any]
    terminal: Mapping[str, Any]
    account: Mapping[str, Any]
    symbol_info: Mapping[str, Any]
    tick: Mapping[str, Any]
    positions_raw: tuple[Mapping[str, Any], ...]
    pending_orders_raw: tuple[Mapping[str, Any], ...]
    capital_review: Mapping[str, Any]
    mt5_calls: tuple[str, ...]
    forbidden_attempts: tuple[str, ...]
    shutdown_called: bool


@dataclass(frozen=True)
class ExecutionLeg:
    purpose: str
    side: str
    position_before: int
    position_after: int
    order_check_passed: bool
    order_send_passed: bool
    order_check: Mapping[str, Any] | None
    order_event: Mapping[str, Any]
    position_ticket: int | None
    completed_utc: str
    requested_price: float | None = None
    broker_result_price: float | None = None
    bid_before: float | None = None
    ask_before: float | None = None
    spread_points_before: float | None = None
    symbol_reported_spread_points_before: int | None = None
    symbol_point: float | None = None
    slippage_points_signed: float | None = None
    slippage_points_adverse: float | None = None
    order_ticket: int | None = None
    deal_ticket: int | None = None
    request_position_ticket: int | None = None


@dataclass(frozen=True)
class TransitionExecution:
    role: str
    target_position: int
    completed_target: bool
    partial_reason: str | None
    broker_position_before: int
    broker_position_after: int
    broker_position_ticket_after: int | None
    legs: tuple[ExecutionLeg, ...]
    order_check_called: bool
    order_send_called: bool
    order_send_calls: int
    successful_order_sends: int
    account_login_masked: str
    shutdown_called: bool
    mt5_calls: tuple[str, ...]
    forbidden_attempts: tuple[str, ...]


@dataclass(frozen=True)
class BrokerHistoryAudit:
    role: str
    account_login_masked: str
    started_utc: str
    captured_utc: str
    deals: tuple[Mapping[str, Any], ...]
    orders: tuple[Mapping[str, Any], ...]
    mt5_calls: tuple[str, ...]
    forbidden_attempts: tuple[str, ...]
    shutdown_called: bool


def _package_snapshot(mt5: Any) -> dict[str, Any]:
    try:
        version = mt5.version()
    except Exception:
        version = None
    return {
        "module_author": getattr(mt5, "__author__", None),
        "module_version": getattr(mt5, "__version__", None),
        "terminal_version": str(version),
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _magic_for_role(controls: FrozenBrokerControls, role: str) -> int:
    if role == "model_a":
        return int(controls.model_a_magic_number)
    if role == "model_b":
        return int(controls.model_b_magic_number)
    raise DualLiveExecutionError(f"Unsupported role: {role!r}")


def _comment_prefix(role: str) -> str:
    return "CP_DUAL_A" if role == "model_a" else "CP_DUAL_B"


def _compact_comment(role: str, purpose: str, event_time_utc: str | None) -> str:
    digits = "".join(ch for ch in str(event_time_utc or "") if ch.isdigit())
    event_piece = digits[4:12] if len(digits) >= 12 else "NA"
    purpose_piece = {
        "OPEN_LONG": "OL",
        "OPEN_SHORT": "OS",
        "CLOSE_LONG": "CL",
        "CLOSE_SHORT": "CS",
        "EMERGENCY_CLOSE_LONG": "EL",
        "EMERGENCY_CLOSE_SHORT": "ES",
    }.get(purpose, "XX")
    return f"{_comment_prefix(role)}_{purpose_piece}_{event_piece}"[:31]


def _position_side(mt5: Any, plain: Mapping[str, Any]) -> int:
    raw = int(plain.get("type", -999))
    if raw == int(getattr(mt5, "POSITION_TYPE_BUY", 0)):
        return 1
    if raw == int(getattr(mt5, "POSITION_TYPE_SELL", 1)):
        return -1
    raise DualLiveExecutionError(f"Unsupported MT5 position type: {raw}")


def _symbol_snapshot(mt5: Any, controls: FrozenBrokerControls) -> dict[str, Any]:
    info = mt5.symbol_info(controls.symbol)
    if info is None:
        raise DualLiveExecutionError(
            f"symbol_info({controls.symbol!r}) returned None: {safe_last_error(mt5)}"
        )
    snapshot = object_to_plain_dict(info)
    if snapshot.get("visible") is not True:
        raise DualLiveExecutionError(f"{controls.symbol} is not visible in MT5")
    volume_min = _float_or_none(snapshot.get("volume_min"))
    volume_step = _float_or_none(snapshot.get("volume_step"))
    volume = float(controls.order_volume_lots)
    if volume_min is None or volume_step is None or volume_step <= 0:
        raise DualLiveExecutionError("Invalid XAUUSD volume_min/volume_step")
    if volume < volume_min - 1e-9:
        raise DualLiveExecutionError(
            f"Frozen volume {volume} is below broker minimum {volume_min}"
        )
    steps = (volume - volume_min) / volume_step
    if abs(steps - round(steps)) > 1e-9:
        raise DualLiveExecutionError(
            f"Frozen volume {volume} is not aligned to broker step {volume_step}"
        )
    return snapshot


def _plain_broker_snapshot(
    *,
    mt5: Any,
    controls: FrozenBrokerControls,
    account: Mapping[str, Any],
    terminal: Mapping[str, Any],
    positions: list[dict[str, Any]],
    pending_orders: list[dict[str, Any]],
) -> BrokerSnapshot:
    parsed_positions = tuple(
        broker_position_from_plain(
            item,
            symbol=controls.symbol,
            buy_type=int(getattr(mt5, "POSITION_TYPE_BUY", 0)),
            sell_type=int(getattr(mt5, "POSITION_TYPE_SELL", 1)),
        )
        for item in positions
    )
    return BrokerSnapshot(
        account_login_masked=str(account.get("login_masked") or ""),
        account_equity=float(account.get("equity", 0.0) or 0.0),
        account_balance=float(account.get("balance", 0.0) or 0.0),
        symbol=controls.symbol,
        positions=parsed_positions,
        pending_order_count=len(pending_orders),
        connected=bool(terminal.get("connected")),
        terminal_trade_allowed=bool(terminal.get("trade_allowed")),
        account_trade_allowed=bool(account.get("trade_allowed")),
        account_expert_allowed=bool(account.get("trade_expert")),
        trade_api_disabled=bool(terminal.get("tradeapi_disabled")),
    )


def inspect_broker(
    *,
    mt5_module: Any,
    terminal_path: str,
    controls: FrozenBrokerControls,
    expected_login_suffix: str,
    require_trading_permissions: bool,
) -> BrokerInspection:
    """Inspect one terminal and return a broker snapshot without sending orders."""

    proxy = GuardedMt5TinyOrderProxy(mt5_module, controls)
    initialized = False
    shutdown_called = False
    try:
        initialise_terminal(proxy, terminal_path)
        initialized = True
        package = _package_snapshot(proxy)
        terminal = inspect_terminal_for_trading(proxy)
        account = inspect_account_for_trading(proxy, controls)
        if str(account.get("currency", "")).upper() != "SGD":
            raise DualLiveExecutionError(
                f"Account currency must remain SGD, found {account.get('currency')!r}"
            )
        account_equity = _float_or_none(account.get("equity"))
        if (
            account_equity is None
            or account_equity < controls.minimum_demo_equity_recommendation_sgd
        ):
            raise DualLiveExecutionError(
                "Account equity is below the frozen minimum demo-equity gate"
            )
        login_masked = str(account.get("login_masked") or "")
        if not login_masked.endswith(str(expected_login_suffix)):
            raise DualLiveExecutionError(
                f"Account mismatch for {terminal_path}. Expected suffix "
                f"{expected_login_suffix}, found {login_masked}"
            )
        if require_trading_permissions:
            permission_checks = {
                "terminal_trade_allowed": terminal.get("trade_allowed") is True,
                "tradeapi_enabled": terminal.get("tradeapi_disabled") is False,
                "account_trade_allowed": account.get("trade_allowed") is True,
                "account_expert_allowed": account.get("trade_expert") is True,
            }
            if not all(permission_checks.values()):
                raise DualLiveExecutionError(
                    f"Trading permissions are not ready: {permission_checks}"
                )
        symbol_info = _symbol_snapshot(proxy, controls)
        tick = inspect_tick_for_order_check(proxy, controls.symbol)
        positions = get_positions_for_symbol(proxy, controls.symbol)
        pending_orders = get_orders_for_symbol(proxy, controls.symbol)
        capital = capital_review_from_live_account(
            controls=controls,
            account=account,
            symbol_info=symbol_info,
        )
        if capital.get("capstone_10x_leverage_cap_passed") is not True:
            raise DualLiveExecutionError(
                "Live account does not pass the frozen 10:1 leverage cap"
            )
        broker = _plain_broker_snapshot(
            mt5=proxy,
            controls=controls,
            account=account,
            terminal=terminal,
            positions=positions,
            pending_orders=pending_orders,
        )
        return BrokerInspection(
            snapshot=broker,
            package=package,
            terminal=terminal,
            account=account,
            symbol_info=symbol_info,
            tick=tick,
            positions_raw=tuple(dict(item) for item in positions),
            pending_orders_raw=tuple(dict(item) for item in pending_orders),
            capital_review=capital,
            mt5_calls=tuple(proxy.calls),
            forbidden_attempts=tuple(proxy.forbidden_attempts),
            shutdown_called=True,
        )
    finally:
        if initialized:
            try:
                proxy.shutdown()
                shutdown_called = True
            except Exception:
                shutdown_called = False
        # The dataclass returned above cannot be mutated from finally.  Callers
        # use the MT5 call list only for audit; a failed shutdown raises below.
        if initialized and not shutdown_called:
            raise DualLiveExecutionError(
                f"MT5 shutdown was not confirmed for {terminal_path}"
            )


def _single_position_plain(
    *,
    mt5: Any,
    controls: FrozenBrokerControls,
    role: str,
    positions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not positions:
        return None
    if len(positions) != 1:
        raise DualLiveExecutionError(
            f"Expected at most one {controls.symbol} position, found {len(positions)}"
        )
    position = positions[0]
    magic = _int_or_none(position.get("magic"))
    expected_magic = _magic_for_role(controls, role)
    if magic not in {None, 0, expected_magic}:
        raise DualLiveExecutionError(
            f"Position has foreign magic {magic}; expected {expected_magic}"
        )
    if role == "model_b" and _position_side(mt5, position) == -1:
        raise DualLiveExecutionError("Model B has a forbidden short position")
    return position


def _execute_leg(
    *,
    proxy: GuardedMt5TinyOrderProxy,
    controls: FrozenBrokerControls,
    role: str,
    purpose: str,
    side: str,
    event_time_utc: str | None,
    current_position: Mapping[str, Any] | None,
    enforce_entry_spread: bool,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[ExecutionLeg, list[dict[str, Any]]]:
    symbol_info = _symbol_snapshot(proxy, controls)
    reported_spread = _int_or_none(symbol_info.get("spread"))
    if enforce_entry_spread and (
        reported_spread is None
        or reported_spread < 0
        or reported_spread > int(controls.max_spread_points_for_entry)
    ):
        raise EntrySpreadBlocked(
            reported_spread, controls.max_spread_points_for_entry
        )
    tick = inspect_tick_for_order_check(proxy, controls.symbol)
    filling_name, filling_value = choose_filling_candidates(proxy, symbol_info)[0]
    expected_magic = _magic_for_role(controls, role)
    comment = _compact_comment(role, purpose, event_time_utc)
    position_ticket = None
    volume = float(controls.order_volume_lots)
    position_before = 0
    expected_after = 1 if side.upper() == "BUY" else -1
    if current_position is not None:
        position_before = _position_side(proxy, current_position)
        position_ticket = (
            _int_or_none(current_position.get("ticket"))
            or _int_or_none(current_position.get("identifier"))
        )
        if position_ticket is None:
            raise DualLiveExecutionError("Close position ticket is unavailable")
        volume = _float_or_none(current_position.get("volume")) or volume
        expected_after = 0
    request = build_deal_request(
        mt5=proxy,
        controls=controls,
        side=side,
        tick=tick,
        filling_value=filling_value,
        magic=expected_magic,
        comment=comment,
        position_ticket=position_ticket,
        volume=volume,
    )
    check_result, margin_required = run_order_check_for_request(proxy, request)
    check_retcode = _int_or_none((check_result or {}).get("retcode"))
    if check_retcode != 0:
        raise DualLiveExecutionError(
            f"{purpose} order_check failed: {check_result}"
        )
    authorisation = SendAuthorisation(
        purpose=purpose,
        expected_side=side.upper(),
        expected_symbol=controls.symbol,
        expected_volume=float(request["volume"]),
        expected_magic=expected_magic,
        comment_prefix=_comment_prefix(role),
        require_position=current_position is not None,
    )
    send_result = call_order_send_compat(proxy, request, authorisation)
    event: OrderSendEvent = build_send_event(
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
        raise DualLiveExecutionError(f"{purpose} order_send failed: {asdict(event)}")
    if current_position is None:
        post_positions = wait_for_position(
            proxy,
            controls=controls,
            expected_side=side,
            expected_volume=float(request["volume"]),
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        ticket = (
            _int_or_none(post_positions[0].get("ticket"))
            or _int_or_none(post_positions[0].get("identifier"))
        )
    else:
        post_positions = wait_for_no_position(
            proxy,
            controls=controls,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        ticket = None
    requested_price = _float_or_none(request.get("price"))
    send_result = event.send_result or {}
    broker_result_price = _float_or_none(send_result.get("price"))
    if broker_result_price is not None and broker_result_price <= 0.0:
        broker_result_price = None
    point = _float_or_none(symbol_info.get("point"))
    bid_before = _float_or_none(tick.get("bid"))
    ask_before = _float_or_none(tick.get("ask"))
    observed_spread_points = None
    if (
        bid_before is not None
        and ask_before is not None
        and point is not None
        and point > 0.0
    ):
        observed_spread_points = (ask_before - bid_before) / point
    slippage_signed = None
    slippage_adverse = None
    if (
        requested_price is not None
        and broker_result_price is not None
        and point is not None
        and point > 0.0
    ):
        slippage_signed = (broker_result_price - requested_price) / point
        slippage_adverse = (
            slippage_signed if side.upper() == "BUY" else -slippage_signed
        )
    leg = ExecutionLeg(
        purpose=purpose,
        side=side.upper(),
        position_before=position_before,
        position_after=expected_after,
        order_check_passed=True,
        order_send_passed=True,
        order_check=check_result,
        order_event=asdict(event),
        position_ticket=ticket,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        requested_price=requested_price,
        broker_result_price=broker_result_price,
        bid_before=bid_before,
        ask_before=ask_before,
        spread_points_before=observed_spread_points,
        symbol_reported_spread_points_before=reported_spread,
        symbol_point=point,
        slippage_points_signed=slippage_signed,
        slippage_points_adverse=slippage_adverse,
        order_ticket=_int_or_none(send_result.get("order")),
        deal_ticket=_int_or_none(send_result.get("deal")),
        request_position_ticket=_int_or_none(request.get("position")),
    )
    return leg, post_positions


def execute_transition(
    *,
    mt5_module: Any,
    terminal_path: str,
    controls: FrozenBrokerControls,
    role: str,
    expected_login_suffix: str,
    target_position: int,
    event_time_utc: str | None,
    timeout_seconds: float = 15.0,
    poll_seconds: float = 0.5,
) -> TransitionExecution:
    """Move one account to ``target_position`` using guarded 0.01-lot orders."""

    if target_position not in {-1, 0, 1}:
        raise DualLiveExecutionError(f"Invalid target position: {target_position}")
    if role == "model_b" and target_position == -1:
        raise DualLiveExecutionError("Model B target cannot be short")
    proxy = GuardedMt5TinyOrderProxy(mt5_module, controls)
    initialized = False
    shutdown_called = False
    legs: list[ExecutionLeg] = []
    account_masked = ""
    try:
        initialise_terminal(proxy, terminal_path)
        initialized = True
        terminal = inspect_terminal_for_trading(proxy)
        account = inspect_account_for_trading(proxy, controls)
        if str(account.get("currency", "")).upper() != "SGD":
            raise DualLiveExecutionError(
                f"Account currency must remain SGD, found {account.get('currency')!r}"
            )
        account_equity = _float_or_none(account.get("equity"))
        if (
            account_equity is None
            or account_equity < controls.minimum_demo_equity_recommendation_sgd
        ):
            raise DualLiveExecutionError(
                "Account equity is below the frozen minimum demo-equity gate"
            )
        account_masked = str(account.get("login_masked") or "")
        if not account_masked.endswith(str(expected_login_suffix)):
            raise DualLiveExecutionError(
                f"Wrong account. Expected suffix {expected_login_suffix}, "
                f"found {account_masked}"
            )
        permission_checks = {
            "terminal_trade_allowed": terminal.get("trade_allowed") is True,
            "tradeapi_enabled": terminal.get("tradeapi_disabled") is False,
            "account_trade_allowed": account.get("trade_allowed") is True,
            "account_expert_allowed": account.get("trade_expert") is True,
        }
        if not all(permission_checks.values()):
            raise DualLiveExecutionError(
                f"Trading permissions are not ready: {permission_checks}"
            )
        symbol_info = _symbol_snapshot(proxy, controls)
        capital = capital_review_from_live_account(
            controls=controls,
            account=account,
            symbol_info=symbol_info,
        )
        if capital.get("capstone_10x_leverage_cap_passed") is not True:
            raise DualLiveExecutionError("Account violates the frozen leverage cap")
        pending = get_orders_for_symbol(proxy, controls.symbol)
        if pending:
            raise DualLiveExecutionError(
                f"Pending {controls.symbol} order exists; refusing transition"
            )
        positions = get_positions_for_symbol(proxy, controls.symbol)
        current = _single_position_plain(
            mt5=proxy,
            controls=controls,
            role=role,
            positions=positions,
        )
        before = 0 if current is None else _position_side(proxy, current)
        if before == target_position:
            ticket = None
            if current is not None:
                ticket = (
                    _int_or_none(current.get("ticket"))
                    or _int_or_none(current.get("identifier"))
                )
            return TransitionExecution(
                role=role,
                target_position=target_position,
                completed_target=True,
                partial_reason=None,
                broker_position_before=before,
                broker_position_after=before,
                broker_position_ticket_after=ticket,
                legs=tuple(),
                order_check_called=False,
                order_send_called=False,
                order_send_calls=0,
                successful_order_sends=0,
                account_login_masked=account_masked,
                shutdown_called=True,
                mt5_calls=tuple(proxy.calls),
                forbidden_attempts=tuple(proxy.forbidden_attempts),
            )
        # For a reversal, validate the entry spread before reducing the
        # existing position.  The spread is checked again immediately before
        # the new entry to cover a last-moment market change.
        is_reversal = bool(
            current is not None
            and target_position != 0
            and target_position != before
        )
        if is_reversal:
            reversal_symbol = _symbol_snapshot(proxy, controls)
            reversal_spread = _int_or_none(reversal_symbol.get("spread"))
            if (
                reversal_spread is None
                or reversal_spread < 0
                or reversal_spread > int(controls.max_spread_points_for_entry)
            ):
                raise EntrySpreadBlocked(
                    reversal_spread,
                    controls.max_spread_points_for_entry,
                )

        # Close before opening.  This is mandatory for reversals.
        if current is not None:
            close_side = "SELL" if before == 1 else "BUY"
            purpose = "CLOSE_LONG" if before == 1 else "CLOSE_SHORT"
            leg, positions = _execute_leg(
                proxy=proxy,
                controls=controls,
                role=role,
                purpose=purpose,
                side=close_side,
                event_time_utc=event_time_utc,
                current_position=current,
                enforce_entry_spread=False,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
            )
            legs.append(leg)
            current = None
        if target_position != 0:
            open_side = "BUY" if target_position == 1 else "SELL"
            purpose = "OPEN_LONG" if target_position == 1 else "OPEN_SHORT"
            try:
                leg, positions = _execute_leg(
                    proxy=proxy,
                    controls=controls,
                    role=role,
                    purpose=purpose,
                    side=open_side,
                    event_time_utc=event_time_utc,
                    current_position=None,
                    enforce_entry_spread=True,
                    timeout_seconds=timeout_seconds,
                    poll_seconds=poll_seconds,
                )
            except EntrySpreadBlocked:
                if not legs or before == 0:
                    raise
                # A reversal can encounter a spread jump in the milliseconds
                # between its close and open legs.  Preserve the successful
                # close as an explicit partial transition instead of pretending
                # the old position still exists.
                final_positions = get_positions_for_symbol(
                    proxy, controls.symbol
                )
                if final_positions:
                    raise DualLiveExecutionError(
                        "Reversal entry was blocked after close, but the "
                        "account is not flat"
                    )
                return TransitionExecution(
                    role=role,
                    target_position=target_position,
                    completed_target=False,
                    partial_reason=(
                        "reversal_entry_spread_blocked_after_confirmed_close"
                    ),
                    broker_position_before=before,
                    broker_position_after=0,
                    broker_position_ticket_after=None,
                    legs=tuple(legs),
                    order_check_called=True,
                    order_send_called=True,
                    order_send_calls=len(legs),
                    successful_order_sends=sum(
                        1 for item in legs if item.order_send_passed
                    ),
                    account_login_masked=account_masked,
                    shutdown_called=True,
                    mt5_calls=tuple(proxy.calls),
                    forbidden_attempts=tuple(proxy.forbidden_attempts),
                )
            else:
                legs.append(leg)
        final_positions = get_positions_for_symbol(proxy, controls.symbol)
        final_plain = _single_position_plain(
            mt5=proxy,
            controls=controls,
            role=role,
            positions=final_positions,
        )
        after = 0 if final_plain is None else _position_side(proxy, final_plain)
        if after != target_position:
            raise DualLiveExecutionError(
                f"Final broker position {after} does not match target {target_position}"
            )
        ticket_after = None
        if final_plain is not None:
            ticket_after = (
                _int_or_none(final_plain.get("ticket"))
                or _int_or_none(final_plain.get("identifier"))
            )
        if proxy.forbidden_attempts:
            raise DualLiveExecutionError(
                f"Forbidden MT5 attempts recorded: {proxy.forbidden_attempts}"
            )
        return TransitionExecution(
            role=role,
            target_position=target_position,
            completed_target=True,
            partial_reason=None,
            broker_position_before=before,
            broker_position_after=after,
            broker_position_ticket_after=ticket_after,
            legs=tuple(legs),
            order_check_called=any(leg.order_check_passed for leg in legs),
            order_send_called=bool(legs),
            order_send_calls=len(legs),
            successful_order_sends=sum(1 for leg in legs if leg.order_send_passed),
            account_login_masked=account_masked,
            shutdown_called=True,
            mt5_calls=tuple(proxy.calls),
            forbidden_attempts=tuple(proxy.forbidden_attempts),
        )
    finally:
        if initialized:
            try:
                proxy.shutdown()
                shutdown_called = True
            except Exception:
                shutdown_called = False
        if initialized and not shutdown_called:
            raise DualLiveExecutionError(
                f"MT5 shutdown was not confirmed for {terminal_path}"
            )


def _dedupe_plain_rows(rows: list[dict[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    seen: set[str] = set()
    output: list[Mapping[str, Any]] = []

    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(row))
    return tuple(output)


def _history_row_time_utc(
    row: Mapping[str, Any],
    *,
    kind: str,
    server_time_offset_hours: int = 0,
) -> datetime | None:
    if kind == "deal":
        milliseconds = row.get("time_msc")
        seconds = row.get("time")
    elif kind == "order":
        milliseconds = row.get("time_done_msc") or row.get("time_setup_msc")
        seconds = row.get("time_done") or row.get("time_setup")
    else:
        raise DualLiveExecutionError(f"Unsupported history kind: {kind!r}")
    try:
        if milliseconds not in (None, "", 0, "0"):
            return datetime.fromtimestamp(
                float(milliseconds) / 1000.0,
                tz=timezone.utc,
            ) - timedelta(hours=int(server_time_offset_hours))
        if seconds not in (None, "", 0, "0"):
            return datetime.fromtimestamp(
                float(seconds), tz=timezone.utc
            ) - timedelta(hours=int(server_time_offset_hours))
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    return None


def _history_rows_for_role(
    rows: list[dict[str, Any]],
    *,
    controls: FrozenBrokerControls,
    role: str,
    kind: str,
    observation_started_utc: datetime,
    known_order_tickets: set[int] | None = None,
    known_position_ids: set[int] | None = None,
    server_time_offset_hours: int = 0,
) -> tuple[Mapping[str, Any], ...]:
    """Filter broker history to this role and observation session.

    Some MT5 brokers emit commission or fee deals with a blank symbol.  Those
    rows are retained when they link to a role-owned order or position, so the
    offline PnL ledger is complete without admitting unrelated account history.
    """

    expected_magic = _magic_for_role(controls, role)
    prefix = _comment_prefix(role)
    lower_bound = observation_started_utc.astimezone(timezone.utc) - timedelta(
        seconds=1
    )
    linked_orders = set(known_order_tickets or ())
    linked_positions = set(known_position_ids or ())
    expected_symbol = str(controls.symbol).upper()
    matched: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol", "") or "").upper()
        magic = _int_or_none(row.get("magic"))
        comment = str(row.get("comment", "") or "")
        row_time = _history_row_time_utc(
            row,
            kind=kind,
            server_time_offset_hours=server_time_offset_hours,
        )
        order_ticket = _int_or_none(row.get("order")) or _int_or_none(
            row.get("ticket") if kind == "order" else None
        )
        position_id = _int_or_none(row.get("position_id")) or _int_or_none(
            row.get("position")
        )
        if row_time is None or row_time < lower_bound:
            continue
        role_marker = magic == expected_magic or comment.startswith(prefix)
        linked_to_role = (
            order_ticket in linked_orders
            or position_id in linked_positions
        )
        if kind == "order":
            include = symbol == expected_symbol and role_marker
        else:
            include = (symbol == expected_symbol and role_marker) or linked_to_role
            if role_marker and symbol in {"", expected_symbol}:
                include = True
        if include:
            matched.append(dict(row))
    return _dedupe_plain_rows(matched)


def _required_history_call(
    proxy: GuardedMt5TinyOrderProxy,
    method_name: str,
    *args: Any,
) -> list[dict[str, Any]]:
    method = getattr(proxy, method_name)
    result = method(*args)
    if result is None:
        raise DualLiveExecutionError(
            f"{method_name} returned None: {safe_last_error(proxy)}"
        )
    return object_list_to_plain(result)


def collect_broker_history(
    *,
    mt5_module: Any,
    terminal_path: str,
    controls: FrozenBrokerControls,
    role: str,
    expected_login_suffix: str,
    started_utc: datetime,
    server_time_offset_hours: int = 0,
) -> BrokerHistoryAudit:
    """Read all role-owned broker deals and orders for the current run window."""

    proxy = GuardedMt5TinyOrderProxy(mt5_module, controls)
    initialized = False
    shutdown_called = False
    captured = datetime.now(timezone.utc)
    try:
        initialise_terminal(proxy, terminal_path)
        initialized = True
        account = inspect_account_for_trading(proxy, controls)
        login_masked = str(account.get("login_masked") or "")
        if not login_masked.endswith(str(expected_login_suffix)):
            raise DualLiveExecutionError(
                f"Account mismatch for {terminal_path}. Expected suffix "
                f"{expected_login_suffix}, found {login_masked}"
            )
        observation_started = started_utc.astimezone(timezone.utc)
        # A wide broker query is retained because some MT5 builds return empty
        # results for narrow windows.  Rows are then filtered back to the exact
        # observation session so an earlier rehearsal cannot contaminate the
        # current run.
        date_from = observation_started - timedelta(days=2)
        date_to = captured + timedelta(days=2)
        deal_rows = _required_history_call(
            proxy,
            "history_deals_get",
            date_from,
            date_to,
        )
        order_rows = _required_history_call(
            proxy,
            "history_orders_get",
            date_from,
            date_to,
        )
        orders = _history_rows_for_role(
            order_rows,
            controls=controls,
            role=role,
            kind="order",
            observation_started_utc=observation_started,
            server_time_offset_hours=server_time_offset_hours,
        )
        known_order_tickets = {
            ticket
            for row in orders
            if (ticket := _int_or_none(row.get("ticket") or row.get("order")))
            is not None
        }
        known_position_ids = {
            position_id
            for row in orders
            if (position_id := _int_or_none(row.get("position_id")))
            is not None
        }
        deals = _history_rows_for_role(
            deal_rows,
            controls=controls,
            role=role,
            kind="deal",
            observation_started_utc=observation_started,
            known_order_tickets=known_order_tickets,
            known_position_ids=known_position_ids,
            server_time_offset_hours=server_time_offset_hours,
        )
        return BrokerHistoryAudit(
            role=role,
            account_login_masked=login_masked,
            started_utc=started_utc.astimezone(timezone.utc).isoformat(),
            captured_utc=captured.isoformat(),
            deals=deals,
            orders=orders,
            mt5_calls=tuple(proxy.calls),
            forbidden_attempts=tuple(proxy.forbidden_attempts),
            shutdown_called=True,
        )
    finally:
        if initialized:
            try:
                proxy.shutdown()
                shutdown_called = True
            except Exception:
                shutdown_called = False
        if initialized and not shutdown_called:
            raise DualLiveExecutionError(
                f"MT5 shutdown was not confirmed for {terminal_path}"
            )


def flatten_position(
    *,
    mt5_module: Any,
    terminal_path: str,
    controls: FrozenBrokerControls,
    role: str,
    expected_login_suffix: str,
    event_time_utc: str | None = None,
) -> TransitionExecution:
    """Emergency-close any owned XAUUSD position and confirm flat state."""

    return execute_transition(
        mt5_module=mt5_module,
        terminal_path=terminal_path,
        controls=controls,
        role=role,
        expected_login_suffix=expected_login_suffix,
        target_position=0,
        event_time_utc=event_time_utc,
    )
