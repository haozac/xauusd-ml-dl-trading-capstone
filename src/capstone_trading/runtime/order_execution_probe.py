"""Stage 3 Step 2 tiny controlled MT5 demo order open/close test v1.1.

This module is intentionally narrow.  It opens one 0.01-lot demo BUY order
for XAUUSD, verifies the resulting position, immediately closes the same
position, and verifies that no XAUUSD position remains.

It is not a strategy runner.  It must not be used for unattended trading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
import csv
import json
import math
import time

from capstone_trading.runtime.order_preflight import (
    DEFAULT_FROZEN_CONFIG_PATH,
    FrozenBrokerControls,
    Stage3OrderPreflightError,
    build_package_snapshot,
    call_order_calc_margin_compat,
    call_order_check_compat,
    capital_review_from_live_account,
    check_result_has_empty_request,
    choose_filling_candidates,
    inspect_account_for_trading,
    inspect_symbol_for_order_check,
    inspect_terminal_for_trading,
    inspect_tick_for_order_check,
    initialise_terminal,
    load_frozen_controls,
    normalise_check_result,
)
from capstone_trading.runtime.mt5_readiness import (
    import_metatrader5_module,
    object_to_plain_dict,
    safe_last_error,
)

DEFAULT_STAGE3_STEP1_REPORT_PATH = Path("runtime/reports/stage3_step1_order_permission_preflight.json")
DEFAULT_REPORT_PATH = Path("runtime/reports/stage3_step2_v1_1_tiny_order_test.json")
DEFAULT_ORDER_EVENTS_CSV_PATH = Path("runtime/reports/stage3_step2_v1_1_order_send_events.csv")
DEFAULT_POSITION_SNAPSHOTS_CSV_PATH = Path("runtime/reports/stage3_step2_v1_1_position_snapshots.csv")
DEFAULT_HISTORY_DEALS_CSV_PATH = Path("runtime/reports/stage3_step2_v1_1_history_deals.csv")
DEFAULT_HISTORY_ORDERS_CSV_PATH = Path("runtime/reports/stage3_step2_v1_1_history_orders.csv")

CONFIRM_SEND_TOKEN = "I_UNDERSTAND_STAGE3_STEP2_SENDS_DEMO_ORDER"
SUCCESS_SEND_RETCODE_NAMES = ("TRADE_RETCODE_DONE", "TRADE_RETCODE_DONE_PARTIAL")

ALLOWED_MT5_METHODS: tuple[str, ...] = (
    "initialize",
    "shutdown",
    "last_error",
    "version",
    "terminal_info",
    "account_info",
    "symbol_info",
    "symbol_info_tick",
    "order_calc_margin",
    "order_check",
    "order_send",
    "positions_get",
    "positions_total",
    "orders_get",
    "orders_total",
    "history_deals_get",
    "history_orders_get",
)

FORBIDDEN_MT5_METHODS: tuple[str, ...] = (
    "order_calc_profit",
    "history_orders_total",
)


@dataclass(frozen=True)
class SendAuthorisation:
    purpose: str
    expected_side: str
    expected_symbol: str
    expected_volume: float
    expected_magic: int
    comment_prefix: str
    require_position: bool


@dataclass(frozen=True)
class OrderSendEvent:
    purpose: str
    side: str
    request: Mapping[str, Any]
    check_result: Mapping[str, Any] | None
    send_result: Mapping[str, Any] | None
    margin_required_account_currency: float | None
    retcode: int | None
    comment: str | None
    passed: bool
    last_error: Any
    filling_name: str
    filling_value: int


class GuardedMt5TinyOrderProxy:
    """Guard MT5 so Stage 3 Step 2 can only send authorised tiny requests."""

    def __init__(self, mt5_module: Any, controls: FrozenBrokerControls):
        self._mt5 = mt5_module
        self._controls = controls
        self.calls: list[str] = []
        self.forbidden_attempts: list[str] = []
        self._next_send_authorisation: SendAuthorisation | None = None
        self.order_send_count = 0

    def authorise_next_order_send(self, authorisation: SendAuthorisation) -> None:
        if self._next_send_authorisation is not None:
            raise Stage3OrderPreflightError("A previous order_send authorisation is still pending")
        self._next_send_authorisation = authorisation

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_MT5_METHODS:
            self.forbidden_attempts.append(name)
            raise Stage3OrderPreflightError(f"Forbidden MT5 function accessed in Stage 3 Step 2: {name}")
        if name == "order_send":
            return self._guarded_order_send
        if name in ALLOWED_MT5_METHODS:
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

    def _guarded_order_send(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("order_send")
        auth = self._next_send_authorisation
        if auth is None:
            self.forbidden_attempts.append("unauthorised_order_send")
            raise Stage3OrderPreflightError("order_send attempted without explicit one-shot Stage 3 Step 2 authorisation")
        if self.order_send_count >= 4:
            self.forbidden_attempts.append("too_many_order_send_calls")
            raise Stage3OrderPreflightError("Too many order_send calls attempted for one tiny order test")
        request = _extract_request_from_send_args(args=args, kwargs=kwargs)
        self._validate_authorised_request(request, auth)
        self._next_send_authorisation = None
        self.order_send_count += 1
        return self._mt5.order_send(*args, **kwargs)

    def _validate_authorised_request(self, request: Mapping[str, Any], auth: SendAuthorisation) -> None:
        side = _side_from_order_type(self, int(request.get("type", -999)))
        if side != auth.expected_side:
            raise Stage3OrderPreflightError(f"Authorised {auth.expected_side} order_send, got {side}")
        if request.get("symbol") != auth.expected_symbol:
            raise Stage3OrderPreflightError(f"order_send symbol mismatch: {request.get('symbol')!r}")
        volume = float(request.get("volume", -1.0))
        if abs(volume - auth.expected_volume) > 1e-9:
            raise Stage3OrderPreflightError(f"order_send volume mismatch: {volume}")
        if int(request.get("magic", -1)) != auth.expected_magic:
            raise Stage3OrderPreflightError(f"order_send magic mismatch: {request.get('magic')!r}")
        if not str(request.get("comment", "")).startswith(auth.comment_prefix):
            raise Stage3OrderPreflightError(f"order_send comment prefix mismatch: {request.get('comment')!r}")
        if auth.require_position and not request.get("position"):
            raise Stage3OrderPreflightError("Close order_send must include the position ticket")
        if not auth.require_position and request.get("position"):
            raise Stage3OrderPreflightError("Open order_send must not include a position ticket")
        if float(request.get("sl", 0.0) or 0.0) != 0.0 or float(request.get("tp", 0.0) or 0.0) != 0.0:
            raise Stage3OrderPreflightError("Stage 3 Step 2 tiny plumbing test must not set broker-side SL/TP")
        if int(request.get("action", -999)) != int(getattr(self, "TRADE_ACTION_DEAL")):
            raise Stage3OrderPreflightError("Stage 3 Step 2 only allows TRADE_ACTION_DEAL")
        if float(request.get("price", 0.0) or 0.0) <= 0.0:
            raise Stage3OrderPreflightError("order_send price must be positive")


def _extract_request_from_send_args(*, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    if len(args) == 1 and not kwargs and isinstance(args[0], Mapping):
        return args[0]
    if set(kwargs) == {"request"} and isinstance(kwargs["request"], Mapping):
        return kwargs["request"]
    # Some MT5 builds accept expanded MqlTradeRequest fields as keyword args.
    if kwargs and not args:
        return kwargs
    raise Stage3OrderPreflightError("Unsupported order_send argument shape")


def load_required_stage3_step1_report(repo_root: Path, path: Path) -> dict[str, Any]:
    full_path = repo_root / path
    if not full_path.exists():
        raise Stage3OrderPreflightError(f"Stage 3 Step 1 PASS report not found: {full_path}")
    report = json.loads(full_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("formal_gate") is not True:
        raise Stage3OrderPreflightError("Stage 3 Step 1 report is not a formal PASS")
    if report.get("order_send_called") is not False or report.get("orders_executed") is not False:
        raise Stage3OrderPreflightError("Stage 3 Step 1 report is not clean: expected no order_send and no orders executed")
    decision = report.get("decision", {}) if isinstance(report.get("decision"), Mapping) else {}
    if decision.get("stage3_step2_tiny_order_test_allowed") is not True:
        raise Stage3OrderPreflightError("Stage 3 Step 1 did not allow Stage 3 Step 2 tiny order test")
    return report


def object_list_to_plain(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    return [object_to_plain_dict(item) for item in list(value)]


def get_positions_for_symbol(mt5: GuardedMt5TinyOrderProxy, symbol: str) -> list[dict[str, Any]]:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        raise Stage3OrderPreflightError(f"positions_get({symbol!r}) returned None: {safe_last_error(mt5)}")
    return object_list_to_plain(positions)


def get_orders_for_symbol(mt5: GuardedMt5TinyOrderProxy, symbol: str) -> list[dict[str, Any]]:
    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        raise Stage3OrderPreflightError(f"orders_get({symbol!r}) returned None: {safe_last_error(mt5)}")
    return object_list_to_plain(orders)


def assert_no_existing_exposure(mt5: GuardedMt5TinyOrderProxy, controls: FrozenBrokerControls) -> None:
    positions = get_positions_for_symbol(mt5, controls.symbol)
    orders = get_orders_for_symbol(mt5, controls.symbol)
    if positions:
        raise Stage3OrderPreflightError(
            f"Refusing tiny order test because existing open {controls.symbol} positions were found: {positions}"
        )
    if orders:
        raise Stage3OrderPreflightError(
            f"Refusing tiny order test because existing active {controls.symbol} orders were found: {orders}"
        )


def _side_from_order_type(mt5: Any, order_type: int) -> str:
    if order_type == int(getattr(mt5, "ORDER_TYPE_BUY")):
        return "BUY"
    if order_type == int(getattr(mt5, "ORDER_TYPE_SELL")):
        return "SELL"
    return f"UNKNOWN_{order_type}"


def _position_type_to_side(mt5: Any, position_type: int) -> str:
    if position_type == int(getattr(mt5, "POSITION_TYPE_BUY", 0)):
        return "BUY"
    if position_type == int(getattr(mt5, "POSITION_TYPE_SELL", 1)):
        return "SELL"
    return f"UNKNOWN_{position_type}"


def _success_send_retcodes(mt5: Any) -> set[int]:
    values: set[int] = set()
    for name in SUCCESS_SEND_RETCODE_NAMES:
        value = getattr(mt5, name, None)
        if value is not None:
            values.add(int(value))
    # Common MetaTrader retcodes, used only as fallback when constants are absent in fakes.
    values.update({10009, 10010})
    return values


def normalise_send_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    plain = object_to_plain_dict(result)
    req = plain.get("request")
    if req is not None and not isinstance(req, (str, int, float, bool)):
        plain["request"] = object_to_plain_dict(req)
    return plain


def send_result_has_empty_request(result: Any) -> bool:
    plain = normalise_send_result(result)
    if plain is None:
        return False
    request = plain.get("request")
    if isinstance(request, Mapping):
        symbol = str(request.get("symbol", "") or "")
        volume = _float_or_none(request.get("volume"))
        price = _float_or_none(request.get("price"))
        try:
            action = int(request.get("action") or 0)
        except Exception:
            action = None
        return symbol == "" and volume == 0.0 and price == 0.0 and action == 0
    if isinstance(request, str):
        return "symbol=''" in request and "volume=0.0" in request and "price=0.0" in request
    return False


def call_order_send_compat(
    mt5: GuardedMt5TinyOrderProxy,
    request: Mapping[str, Any],
    authorisation: SendAuthorisation,
) -> Any:
    """Call MT5 order_send once safely, with compatibility fallbacks.

    The function may retry only when the previous call clearly bound an empty
    request or raised TypeError before reaching the broker.  It must not retry
    after a real broker response to the intended request.
    """
    clean_request = dict(request)

    # Attempt 1: keyword request=<dict>. Some builds return retcode 10013 with
    # an empty request; that is safe to retry because it did not submit the real order.
    mt5.authorise_next_order_send(authorisation)
    try:
        result = mt5.order_send(request=clean_request)
    except TypeError:
        result = None
        mt5._next_send_authorisation = None
    if result is not None and not send_result_has_empty_request(result):
        return result

    # Attempt 2: expanded keyword fields. This is the style needed by some
    # MetaTrader5 package builds observed in Stage 3 Step 1.
    mt5.authorise_next_order_send(authorisation)
    try:
        result = mt5.order_send(**clean_request)
    except TypeError:
        result = None
        mt5._next_send_authorisation = None
    if result is not None and not send_result_has_empty_request(result):
        return result

    # Attempt 3: official positional request dictionary style.
    mt5.authorise_next_order_send(authorisation)
    try:
        return mt5.order_send(clean_request)
    except TypeError:
        mt5._next_send_authorisation = None
        return result


def build_deal_request(
    *,
    mt5: Any,
    controls: FrozenBrokerControls,
    side: str,
    tick: Mapping[str, Any],
    filling_value: int,
    magic: int,
    comment: str,
    position_ticket: int | None = None,
    volume: float | None = None,
) -> dict[str, Any]:
    side_upper = side.upper()
    if side_upper not in {"BUY", "SELL"}:
        raise ValueError(f"Unsupported side: {side}")
    price = float(tick["ask"] if side_upper == "BUY" else tick["bid"])
    request: dict[str, Any] = {
        "action": int(getattr(mt5, "TRADE_ACTION_DEAL")),
        "symbol": controls.symbol,
        "volume": controls.stage3_first_order_test_volume_lots if volume is None else float(volume),
        "type": int(getattr(mt5, f"ORDER_TYPE_{side_upper}")),
        "price": price,
        "deviation": controls.max_deviation_points_for_stage3_order_request,
        "magic": int(magic),
        "comment": comment,
        "type_time": int(getattr(mt5, "ORDER_TIME_GTC")),
        "type_filling": int(filling_value),
    }
    if position_ticket is not None:
        request["position"] = int(position_ticket)
    return request


def run_order_check_for_request(mt5: GuardedMt5TinyOrderProxy, request: Mapping[str, Any]) -> tuple[dict[str, Any] | None, float | None]:
    margin = call_order_calc_margin_compat(
        mt5,
        action=int(request["type"]),
        symbol=str(request["symbol"]),
        volume=float(request["volume"]),
        price=float(request["price"]),
    )
    check = call_order_check_compat(mt5, request)
    return normalise_check_result(check), margin


def build_send_event(
    *,
    mt5: GuardedMt5TinyOrderProxy,
    purpose: str,
    side: str,
    request: Mapping[str, Any],
    check_result: Mapping[str, Any] | None,
    send_result_raw: Any,
    margin_required: float | None,
    filling_name: str,
    filling_value: int,
) -> OrderSendEvent:
    send_result = normalise_send_result(send_result_raw)
    retcode = None
    comment = None
    if send_result is not None:
        raw_retcode = send_result.get("retcode")
        try:
            retcode = int(raw_retcode) if raw_retcode is not None else None
        except Exception:
            retcode = None
        comment = None if send_result.get("comment") is None else str(send_result.get("comment"))
    passed = retcode in _success_send_retcodes(mt5)
    return OrderSendEvent(
        purpose=purpose,
        side=side.upper(),
        request=dict(request),
        check_result=check_result,
        send_result=send_result,
        margin_required_account_currency=margin_required,
        retcode=retcode,
        comment=comment,
        passed=passed,
        last_error=safe_last_error(mt5),
        filling_name=filling_name,
        filling_value=filling_value,
    )


def wait_for_position(
    mt5: GuardedMt5TinyOrderProxy,
    *,
    controls: FrozenBrokerControls,
    expected_side: str,
    expected_volume: float,
    timeout_seconds: float,
    poll_seconds: float,
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_seconds
    last_positions: list[dict[str, Any]] = []
    while time.time() <= deadline:
        positions = get_positions_for_symbol(mt5, controls.symbol)
        last_positions = positions
        if len(positions) == 1:
            position = positions[0]
            side = _position_type_to_side(mt5, int(position.get("type", -999)))
            volume = float(position.get("volume", 0.0) or 0.0)
            if side == expected_side.upper() and abs(volume - expected_volume) <= 1e-9:
                return positions
        time.sleep(poll_seconds)
    raise Stage3OrderPreflightError(f"Position did not appear as expected. Last positions: {last_positions}")


def wait_for_no_position(
    mt5: GuardedMt5TinyOrderProxy,
    *,
    controls: FrozenBrokerControls,
    timeout_seconds: float,
    poll_seconds: float,
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_seconds
    last_positions: list[dict[str, Any]] = []
    while time.time() <= deadline:
        positions = get_positions_for_symbol(mt5, controls.symbol)
        last_positions = positions
        if not positions:
            return []
        time.sleep(poll_seconds)
    raise Stage3OrderPreflightError(f"Position still exists after close attempt. Last positions: {last_positions}")


def snapshot_positions(
    *,
    label: str,
    positions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ts = datetime.now(timezone.utc).isoformat()
    if not positions:
        return [{"snapshot_label": label, "snapshot_utc": ts, "position_count": 0}]
    rows = []
    for position in positions:
        row = {"snapshot_label": label, "snapshot_utc": ts, "position_count": len(positions)}
        row.update(position)
        rows.append(row)
    return rows


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


def _history_row_matches_probe(
    row: Mapping[str, Any],
    *,
    controls: FrozenBrokerControls,
    order_tickets: set[int],
    position_tickets: set[int],
) -> bool:
    symbol = str(row.get("symbol", "") or "").upper()
    comment = str(row.get("comment", "") or "")
    magic = _int_or_none(row.get("magic"))
    order = _int_or_none(row.get("order"))
    ticket = _int_or_none(row.get("ticket"))
    position_id = _int_or_none(row.get("position_id")) or _int_or_none(row.get("position"))
    allowed_magic = {controls.model_b_magic_number, controls.model_a_magic_number}
    return (
        symbol == controls.symbol.upper()
        or comment.startswith("CP_S3P2_")
        or magic in allowed_magic
        or order in order_tickets
        or ticket in order_tickets
        or position_id in position_tickets
    )


def _dedupe_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _safe_history_call(method: Any, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    try:
        result = method(*args, **kwargs)
    except TypeError:
        return []
    except Exception:
        return []
    if result is None:
        return []
    return object_list_to_plain(result)


def collect_history_rows(
    mt5: GuardedMt5TinyOrderProxy,
    *,
    controls: FrozenBrokerControls,
    start_utc: datetime,
    position_ticket: int | None,
    order_tickets: set[int],
    retries: int = 10,
    retry_sleep_seconds: float = 0.5,
) -> dict[str, Any]:
    """Recover broker history for the tiny open/close probe.

    v1.0 used one narrow grouped history_deals_get call and Dukascopy returned
    an empty result although the MT5 History tab showed the completed trade.
    v1.1 uses a wider time window, retries after the close, and checks both
    history_deals_get and history_orders_get.  This function never sends an
    order; it only reads broker history.
    """
    date_from = start_utc - timedelta(days=2)
    date_to = datetime.now(timezone.utc) + timedelta(days=2)
    position_tickets = {int(position_ticket)} if position_ticket else set()
    attempts: list[dict[str, Any]] = []
    filtered_deals: list[dict[str, Any]] = []
    filtered_orders: list[dict[str, Any]] = []

    for attempt_idx in range(1, retries + 1):
        deal_candidates: list[dict[str, Any]] = []
        order_candidates: list[dict[str, Any]] = []

        # Position-specific retrieval can work even when grouped retrieval does not.
        for position in sorted(position_tickets):
            deal_candidates.extend(_safe_history_call(mt5.history_deals_get, position=position))

        # Ticket-specific retrieval for order history.
        if hasattr(mt5, "history_orders_get"):
            for order_ticket in sorted(order_tickets):
                order_candidates.extend(_safe_history_call(mt5.history_orders_get, ticket=order_ticket))

        # Wide time-window retrieval without group filter, then symbol/comment/magic filtering in Python.
        deal_candidates.extend(_safe_history_call(mt5.history_deals_get, date_from, date_to))
        deal_candidates.extend(_safe_history_call(mt5.history_deals_get, date_from, date_to, group=f"*{controls.symbol}*"))
        if hasattr(mt5, "history_orders_get"):
            order_candidates.extend(_safe_history_call(mt5.history_orders_get, date_from, date_to))
            order_candidates.extend(_safe_history_call(mt5.history_orders_get, date_from, date_to, group=f"*{controls.symbol}*"))

        filtered_deals = _dedupe_history_rows([
            row for row in deal_candidates
            if _history_row_matches_probe(row, controls=controls, order_tickets=order_tickets, position_tickets=position_tickets)
        ])
        filtered_orders = _dedupe_history_rows([
            row for row in order_candidates
            if _history_row_matches_probe(row, controls=controls, order_tickets=order_tickets, position_tickets=position_tickets)
        ])
        attempts.append({
            "attempt": attempt_idx,
            "deal_candidate_count": len(deal_candidates),
            "filtered_deal_count": len(filtered_deals),
            "order_candidate_count": len(order_candidates),
            "filtered_order_count": len(filtered_orders),
        })
        if filtered_deals or filtered_orders:
            break
        if attempt_idx < retries:
            time.sleep(retry_sleep_seconds)

    return {
        "history_deals_filtered": filtered_deals,
        "history_orders_filtered": filtered_orders,
        "history_recovery_attempts": attempts,
        "history_records_recovered": bool(filtered_deals or filtered_orders),
        "history_window_from_utc": date_from.isoformat(),
        "history_window_to_utc": date_to.isoformat(),
        "history_order_tickets_used": sorted(order_tickets),
        "history_position_tickets_used": sorted(position_tickets),
    }


def run_stage3_tiny_order_test(
    *,
    mt5_module: Any,
    controls: FrozenBrokerControls,
    terminal_path: str | None,
    confirmation: str,
    side: str = "BUY",
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.5,
) -> dict[str, Any]:
    if confirmation != CONFIRM_SEND_TOKEN:
        raise Stage3OrderPreflightError(
            f"Refusing to send demo order. Pass --confirm-send {CONFIRM_SEND_TOKEN!r} to acknowledge Stage 3 Step 2."
        )
    side_upper = side.upper()
    if side_upper != "BUY":
        raise Stage3OrderPreflightError("Stage 3 Step 2 v1.0 only allows BUY open/SELL close for Model B plumbing")
    proxy = GuardedMt5TinyOrderProxy(mt5_module, controls)
    initialized = False
    shutdown_called = False
    order_events: list[OrderSendEvent] = []
    position_snapshots: list[dict[str, Any]] = []
    history_deals: list[dict[str, Any]] = []
    history_orders: list[dict[str, Any]] = []
    history_audit: dict[str, Any] = {}
    start_utc = datetime.now(timezone.utc)
    status = "FAIL"
    formal_gate = False
    warnings: list[str] = []
    emergency_note = None
    try:
        initialise_terminal(proxy, terminal_path)
        initialized = True
        package = build_package_snapshot(proxy)
        terminal = inspect_terminal_for_trading(proxy)
        account = inspect_account_for_trading(proxy, controls)
        symbol_info = inspect_symbol_for_order_check(proxy, controls)
        tick = inspect_tick_for_order_check(proxy, controls.symbol)
        capital_review = capital_review_from_live_account(controls=controls, account=account, symbol_info=symbol_info)
        if capital_review.get("capstone_10x_leverage_cap_passed") is not True:
            raise Stage3OrderPreflightError("Live account does not pass capstone 10x leverage cap for 0.01 lot")

        assert_no_existing_exposure(proxy, controls)
        position_snapshots.extend(snapshot_positions(label="before_open", positions=get_positions_for_symbol(proxy, controls.symbol)))
        filling_candidates = choose_filling_candidates(proxy, symbol_info)
        filling_name, filling_value = filling_candidates[0]

        open_request = build_deal_request(
            mt5=proxy,
            controls=controls,
            side="BUY",
            tick=tick,
            filling_value=filling_value,
            magic=controls.model_b_magic_number,
            comment="CP_S3P2_OPEN_B",
        )
        open_check, open_margin = run_order_check_for_request(proxy, open_request)
        open_retcode = int(open_check.get("retcode")) if open_check and open_check.get("retcode") is not None else None
        if open_retcode != 0:
            raise Stage3OrderPreflightError(f"Open order_check failed before send: {open_check}")
        open_result = call_order_send_compat(
            proxy,
            open_request,
            SendAuthorisation(
                purpose="OPEN",
                expected_side="BUY",
                expected_symbol=controls.symbol,
                expected_volume=controls.stage3_first_order_test_volume_lots,
                expected_magic=controls.model_b_magic_number,
                comment_prefix="CP_S3P2_OPEN_B",
                require_position=False,
            ),
        )
        open_event = build_send_event(
            mt5=proxy,
            purpose="OPEN",
            side="BUY",
            request=open_request,
            check_result=open_check,
            send_result_raw=open_result,
            margin_required=open_margin,
            filling_name=filling_name,
            filling_value=filling_value,
        )
        order_events.append(open_event)
        if not open_event.passed:
            # No close should be attempted unless a real position appears.
            positions_after_failed_open = get_positions_for_symbol(proxy, controls.symbol)
            position_snapshots.extend(snapshot_positions(label="after_failed_open", positions=positions_after_failed_open))
            if positions_after_failed_open:
                emergency_note = "Open send did not return success but a position exists; manual review required."
            raise Stage3OrderPreflightError(f"Open order_send did not pass: {asdict(open_event)}")

        open_positions = wait_for_position(
            proxy,
            controls=controls,
            expected_side="BUY",
            expected_volume=controls.stage3_first_order_test_volume_lots,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        position_snapshots.extend(snapshot_positions(label="after_open", positions=open_positions))
        open_position = open_positions[0]
        position_ticket = int(open_position.get("ticket") or open_position.get("identifier") or 0)
        if position_ticket <= 0:
            raise Stage3OrderPreflightError(f"Could not identify opened position ticket: {open_position}")
        close_volume = float(open_position.get("volume", controls.stage3_first_order_test_volume_lots))

        close_success = False
        close_failures: list[dict[str, Any]] = []
        for attempt_idx, (close_filling_name, close_filling_value) in enumerate(filling_candidates, start=1):
            close_tick = inspect_tick_for_order_check(proxy, controls.symbol)
            close_request = build_deal_request(
                mt5=proxy,
                controls=controls,
                side="SELL",
                tick=close_tick,
                filling_value=close_filling_value,
                magic=controls.model_b_magic_number,
                comment=f"CP_S3P2_CLOSE_B_{attempt_idx}",
                position_ticket=position_ticket,
                volume=close_volume,
            )
            close_check, close_margin = run_order_check_for_request(proxy, close_request)
            close_retcode = int(close_check.get("retcode")) if close_check and close_check.get("retcode") is not None else None
            if close_retcode != 0:
                close_failures.append({"attempt": attempt_idx, "stage": "order_check", "result": close_check})
                continue
            close_result = call_order_send_compat(
                proxy,
                close_request,
                SendAuthorisation(
                    purpose="CLOSE",
                    expected_side="SELL",
                    expected_symbol=controls.symbol,
                    expected_volume=close_volume,
                    expected_magic=controls.model_b_magic_number,
                    comment_prefix="CP_S3P2_CLOSE_B",
                    require_position=True,
                ),
            )
            close_event = build_send_event(
                mt5=proxy,
                purpose="CLOSE",
                side="SELL",
                request=close_request,
                check_result=close_check,
                send_result_raw=close_result,
                margin_required=close_margin,
                filling_name=close_filling_name,
                filling_value=close_filling_value,
            )
            order_events.append(close_event)
            if close_event.passed:
                close_success = True
                break
            close_failures.append({"attempt": attempt_idx, "stage": "order_send", "event": asdict(close_event)})
        if not close_success:
            emergency_note = "Close failed; immediately close the XAUUSD demo position manually in MT5 and preserve logs."
            raise Stage3OrderPreflightError(f"Close attempts failed: {close_failures}")

        final_positions = wait_for_no_position(
            proxy,
            controls=controls,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        position_snapshots.extend(snapshot_positions(label="after_close", positions=final_positions))
        open_order_ticket = _int_or_none((open_event.send_result or {}).get("order"))
        close_order_tickets = {
            ticket for ticket in (
                _int_or_none((event.send_result or {}).get("order"))
                for event in order_events
                if event.purpose == "CLOSE"
            )
            if ticket is not None
        }
        order_tickets = {ticket for ticket in {open_order_ticket, *close_order_tickets} if ticket is not None}
        history_audit = collect_history_rows(
            proxy,
            controls=controls,
            start_utc=start_utc,
            position_ticket=position_ticket,
            order_tickets=order_tickets,
        )
        history_deals = list(history_audit.get("history_deals_filtered", []))
        history_orders = list(history_audit.get("history_orders_filtered", []))

        validations = {
            "terminal_connected": terminal.get("connected") is True,
            "terminal_trade_allowed": terminal.get("trade_allowed") is True,
            "terminal_tradeapi_enabled": terminal.get("tradeapi_disabled") is False,
            "account_is_demo": account.get("trade_mode_name") == "DEMO",
            "symbol_spread_within_entry_gate": int(symbol_info.get("spread", 999999)) <= controls.max_spread_points_for_entry,
            "capstone_10x_leverage_cap_passed": capital_review.get("capstone_10x_leverage_cap_passed") is True,
            "no_existing_position_before_open": True,
            "open_order_send_passed": open_event.passed,
            "position_verified_after_open": True,
            "close_order_send_passed": close_success,
            "no_position_after_close": True,
            "max_two_successful_send_events": len([event for event in order_events if event.passed]) <= 2,
            "history_records_recovered": bool(history_deals or history_orders),
        }
        formal_gate = all(validations.values())
        status = "PASS" if formal_gate else "FAIL"

        try:
            proxy.shutdown()
            shutdown_called = True
            initialized = False
        except Exception:
            shutdown_called = False

        return {
            "stage": 3,
            "step": 2,
            "status": status,
            "formal_gate": formal_gate,
            "purpose": "tiny_controlled_demo_order_open_close_test",
            "mt5_used": True,
            "order_send_called": "order_send" in proxy.calls,
            "orders_executed": bool(order_events),
            "open_close_completed": formal_gate,
            "started_utc": start_utc.isoformat(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "shutdown_called": shutdown_called,
            "controls": asdict(controls),
            "package": package,
            "terminal": terminal,
            "account": account,
            "symbol_info": symbol_info,
            "tick_at_open_decision": tick,
            "filling_candidates": [{"name": name, "value": value} for name, value in filling_candidates],
            "order_send_events": [asdict(event) for event in order_events],
            "position_snapshots": position_snapshots,
            "history_deals_filtered": history_deals,
            "history_orders_filtered": history_orders,
            "history_audit": history_audit,
            "capital_review": capital_review,
            "validations": validations,
            "read_write_methods_used": tuple(proxy.calls),
            "forbidden_function_attempts": tuple(proxy.forbidden_attempts),
            "warnings": warnings,
            "emergency_note": emergency_note,
            "decision": {
                "stage3_step2_tiny_order_test_passed": formal_gate,
                "stage3_step3_single_model_execution_allowed": False,
                "next_step_if_pass": "Stage 3 Step 3 design/review only; do not start strategy automation until this report is reviewed.",
            },
        }
    except Exception as exc:
        # Best-effort position snapshot for diagnostics.  Do not hide the failure.
        failure_positions: list[dict[str, Any]] = []
        try:
            failure_positions = get_positions_for_symbol(proxy, controls.symbol)
            position_snapshots.extend(snapshot_positions(label="failure_snapshot", positions=failure_positions))
        except Exception:
            pass
        try:
            history_audit = collect_history_rows(
                proxy,
                controls=controls,
                start_utc=start_utc,
                position_ticket=None,
                order_tickets={
                    ticket for ticket in (
                        _int_or_none((event.send_result or {}).get("order"))
                        for event in order_events
                    )
                    if ticket is not None
                },
                retries=1,
                retry_sleep_seconds=0.0,
            )
            history_deals = list(history_audit.get("history_deals_filtered", []))
            history_orders = list(history_audit.get("history_orders_filtered", []))
        except Exception:
            history_deals = []
            history_orders = []
            history_audit = {}
        validations = {
            "open_close_completed": False,
            "no_position_after_failure": len(failure_positions) == 0,
        }
        return {
            "stage": 3,
            "step": 2,
            "status": "FAIL",
            "formal_gate": False,
            "purpose": "tiny_controlled_demo_order_open_close_test",
            "mt5_used": True,
            "order_send_called": "order_send" in proxy.calls,
            "orders_executed": bool(order_events),
            "open_close_completed": False,
            "started_utc": start_utc.isoformat(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "shutdown_called": shutdown_called,
            "controls": asdict(controls),
            "order_send_events": [asdict(event) for event in order_events],
            "position_snapshots": position_snapshots,
            "history_deals_filtered": history_deals,
            "history_orders_filtered": history_orders,
            "history_audit": history_audit,
            "validations": validations,
            "read_write_methods_used": tuple(proxy.calls),
            "forbidden_function_attempts": tuple(proxy.forbidden_attempts),
            "warnings": warnings,
            "emergency_note": emergency_note,
            "error": str(exc),
            "decision": {
                "stage3_step2_tiny_order_test_passed": False,
                "manual_intervention_required": len(failure_positions) > 0,
                "stage3_step3_single_model_execution_allowed": False,
            },
        }
    finally:
        if initialized:
            try:
                proxy.shutdown()
            except Exception:
                pass


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def write_rows_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [_flatten_mapping(row) for row in rows]
    fieldnames: list[str] = []
    for row in flattened:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        if flattened:
            writer.writerows(flattened)


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


def run_tiny_order_test(
    *,
    repo_root: Path,
    terminal_path: str | None = None,
    confirmation: str,
    frozen_config_path: Path = DEFAULT_FROZEN_CONFIG_PATH,
    step1_report_path: Path = DEFAULT_STAGE3_STEP1_REPORT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    order_events_csv_path: Path = DEFAULT_ORDER_EVENTS_CSV_PATH,
    position_snapshots_csv_path: Path = DEFAULT_POSITION_SNAPSHOTS_CSV_PATH,
    history_deals_csv_path: Path = DEFAULT_HISTORY_DEALS_CSV_PATH,
    history_orders_csv_path: Path = DEFAULT_HISTORY_ORDERS_CSV_PATH,
    mt5_module: Any | None = None,
    side: str = "BUY",
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.5,
) -> dict[str, Any]:
    load_required_stage3_step1_report(repo_root, step1_report_path)
    controls = load_frozen_controls(repo_root / frozen_config_path)
    if mt5_module is None:
        mt5_module = import_metatrader5_module()
    report = run_stage3_tiny_order_test(
        mt5_module=mt5_module,
        controls=controls,
        terminal_path=terminal_path,
        confirmation=confirmation,
        side=side,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    report["report_path"] = str(report_path)
    report["order_events_csv_path"] = str(order_events_csv_path)
    report["position_snapshots_csv_path"] = str(position_snapshots_csv_path)
    report["history_deals_csv_path"] = str(history_deals_csv_path)
    report["history_orders_csv_path"] = str(history_orders_csv_path)
    report["frozen_config_path"] = str(frozen_config_path)
    report["stage3_step1_report_path"] = str(step1_report_path)
    write_json(repo_root / report_path, report)
    write_rows_csv(repo_root / order_events_csv_path, report.get("order_send_events", []))
    write_rows_csv(repo_root / position_snapshots_csv_path, report.get("position_snapshots", []))
    write_rows_csv(repo_root / history_deals_csv_path, report.get("history_deals_filtered", []))
    write_rows_csv(repo_root / history_orders_csv_path, report.get("history_orders_filtered", []))
    return report
