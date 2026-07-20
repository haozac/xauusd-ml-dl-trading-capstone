"""Stage 3 Step 1 MT5 order-permission and order_check preflight.

This module is deliberately limited to broker permission inspection, margin
estimation, and mt5.order_check().  It must not call mt5.order_send().

Stage 3 Step 1 answers: "Can this terminal/account/symbol validate a tiny
0.01-lot demo order request under the frozen broker execution controls?"
It does not open, close, or modify any broker position.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json
import math

try:  # Reuse utilities from the Stage 2 MT5 readiness module when available.
    from capstone_trading.runtime.mt5_readiness import (  # type: ignore
        import_metatrader5_module,
        object_to_plain_dict,
        safe_last_error,
        trade_mode_to_name,
        mask_login,
    )
except Exception:  # pragma: no cover - fallback for isolated linting only
    import_metatrader5_module = None  # type: ignore

    def object_to_plain_dict(value: Any) -> dict[str, Any]:  # type: ignore
        if value is None:
            return {}
        if hasattr(value, "_asdict"):
            return dict(value._asdict())
        if isinstance(value, Mapping):
            return dict(value)
        return {
            name: getattr(value, name)
            for name in dir(value)
            if not name.startswith("_") and not callable(getattr(value, name))
        }

    def safe_last_error(mt5: Any) -> Any:  # type: ignore
        try:
            return mt5.last_error()
        except Exception:
            return None

    def trade_mode_to_name(mt5: Any, trade_mode: Any) -> str:  # type: ignore
        try:
            value = int(trade_mode)
        except Exception:
            return "UNKNOWN"
        mapping = {
            int(getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)): "DEMO",
            int(getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1)): "CONTEST",
            int(getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2)): "REAL",
        }
        return mapping.get(value, f"UNKNOWN_{value}")

    def mask_login(login: Any) -> str | None:  # type: ignore
        if login is None:
            return None
        text = str(login)
        return ("*" * max(0, len(text) - 4)) + text[-4:]


DEFAULT_FROZEN_CONFIG_PATH = Path("config/broker_execution_controls_frozen.yaml")
DEFAULT_REPORT_PATH = Path("runtime/reports/stage3_step1_order_permission_preflight.json")
DEFAULT_ORDER_CHECK_CSV_PATH = Path("runtime/reports/stage3_step1_order_check_results.csv")

# Stage 3 Step 1 deliberately allows order_check and order_calc_margin, but
# still blocks order_send and all order/history mutation APIs.
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
)

FORBIDDEN_MT5_METHODS: tuple[str, ...] = (
    "order_send",
    "orders_get",
    "orders_total",
    "history_orders_get",
    "history_orders_total",
    "history_deals_get",
    "history_deals_total",
    "positions_get",
    "positions_total",
)

SUCCESS_ORDER_CHECK_RETCODES: set[int] = {0}


class Stage3OrderPreflightError(RuntimeError):
    """Raised when Stage 3 Step 1 preflight cannot pass safely."""


@dataclass(frozen=True)
class FrozenBrokerControls:
    broker_company_expected: str
    server_expected: str
    symbol: str
    timeframe: str
    order_volume_lots: float
    max_open_volume_lots_per_model: float
    max_positions_per_model: int
    max_spread_points_for_entry: int
    max_deviation_points_for_stage3_order_request: int
    capstone_leverage_cap: float
    minimum_demo_equity_recommendation_sgd: float
    stage3_first_order_test_volume_lots: float
    stage3_order_check_required_before_order_send: bool
    model_a_magic_number: int
    model_b_magic_number: int
    model_a_order_comment: str
    model_b_order_comment: str
    model_b_variant_for_first_controlled_execution: str
    review_usd_sgd_rate_assumption: float
    mt5_server_time_offset_hours_current: int
    controls_version: str


@dataclass(frozen=True)
class OrderCheckAttempt:
    side: str
    request: Mapping[str, Any]
    check_result: Mapping[str, Any] | None
    margin_required_account_currency: float | None
    last_error: Any
    passed: bool
    retcode: int | None
    comment: str | None
    filling_name: str
    filling_value: int


class GuardedMt5OrderCheckProxy:
    """Small MT5 proxy for Stage 3 Step 1.

    This proxy exposes order_check and order_calc_margin only.  order_send and
    position/history functions are blocked and recorded if accidentally accessed.
    """

    def __init__(self, mt5_module: Any):
        self._mt5 = mt5_module
        self.calls: list[str] = []
        self.forbidden_attempts: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_MT5_METHODS:
            self.forbidden_attempts.append(name)
            raise Stage3OrderPreflightError(f"Forbidden MT5 execution/history function accessed: {name}")
        if name in ALLOWED_MT5_METHODS:
            attr = getattr(self._mt5, name)
            if callable(attr):
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    self.calls.append(name)
                    return attr(*args, **kwargs)

                return wrapped
            return attr
        # Constants used to build order_check requests are safe to read.
        if name.startswith("ORDER_") or name.startswith("TRADE_") or name.startswith("ACCOUNT_") or name.startswith("SYMBOL_"):
            return getattr(self._mt5, name)
        if name in {"__author__", "__version__"}:
            return getattr(self._mt5, name, None)
        raise AttributeError(name)


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1].replace('\\"', '"')
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.lower() == "null":
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_simple_yaml_mapping(path: Path) -> dict[str, dict[str, Any]]:
    """Load the simple nested YAML emitted by Stage 2 Step 3A.

    The project already writes a very small config structure.  This parser avoids
    taking a hard dependency on PyYAML inside the deployment preflight script.
    """
    if not path.exists():
        raise Stage3OrderPreflightError(f"Frozen broker controls config not found: {path}")
    data: dict[str, dict[str, Any]] = {}
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            current_section = stripped[:-1]
            data[current_section] = {}
            continue
        if current_section is None or not line.startswith("  ") or ":" not in stripped:
            raise Stage3OrderPreflightError(f"Unsupported frozen config line: {raw_line!r}")
        key, value = stripped.split(":", 1)
        data[current_section][key.strip()] = parse_scalar(value)
    return data


def load_frozen_controls(path: Path) -> FrozenBrokerControls:
    raw = load_simple_yaml_mapping(path)
    metadata = raw.get("metadata", {})
    broker = raw.get("broker", {})
    time = raw.get("time", {})
    limits = raw.get("execution_limits", {})
    order = raw.get("order_policy", {})
    identifiers = raw.get("identifiers", {})
    model = raw.get("model_policy", {})
    assumptions = raw.get("review_assumptions", {})
    required_paths = {
        "broker.symbol": broker.get("symbol"),
        "execution_limits.order_volume_lots": limits.get("order_volume_lots"),
        "order_policy.stage3_order_check_required_before_order_send": order.get("stage3_order_check_required_before_order_send"),
    }
    missing = [key for key, value in required_paths.items() if value is None]
    if missing:
        raise Stage3OrderPreflightError(f"Frozen broker config missing required keys: {missing}")
    return FrozenBrokerControls(
        broker_company_expected=str(broker.get("broker_company_expected", "Dukascopy Bank SA")),
        server_expected=str(broker.get("server_expected", "Dukascopy-demo-mt5-1")),
        symbol=str(broker.get("symbol")),
        timeframe=str(broker.get("timeframe", "M15")),
        order_volume_lots=float(limits.get("order_volume_lots")),
        max_open_volume_lots_per_model=float(limits.get("max_open_volume_lots_per_model", limits.get("order_volume_lots"))),
        max_positions_per_model=int(limits.get("max_positions_per_model", 1)),
        max_spread_points_for_entry=int(limits.get("max_spread_points_for_entry", 800)),
        max_deviation_points_for_stage3_order_request=int(limits.get("max_deviation_points_for_stage3_order_request", 200)),
        capstone_leverage_cap=float(limits.get("capstone_leverage_cap", 10.0)),
        minimum_demo_equity_recommendation_sgd=float(limits.get("minimum_demo_equity_recommendation_sgd", 1000.0)),
        stage3_first_order_test_volume_lots=float(order.get("stage3_first_order_test_volume_lots", limits.get("order_volume_lots"))),
        stage3_order_check_required_before_order_send=bool(order.get("stage3_order_check_required_before_order_send", True)),
        model_a_magic_number=int(identifiers.get("model_a_magic_number", 26070101)),
        model_b_magic_number=int(identifiers.get("model_b_magic_number", 26070102)),
        model_a_order_comment=str(identifiers.get("model_a_order_comment", "CAPSTONE_MODEL_A")),
        model_b_order_comment=str(identifiers.get("model_b_order_comment", "CAPSTONE_MODEL_B")),
        model_b_variant_for_first_controlled_execution=str(model.get("model_b_variant_for_first_controlled_execution", "MODEL_B_V2_CURRENT")),
        review_usd_sgd_rate_assumption=float(assumptions.get("review_usd_sgd_rate_assumption", 1.35)),
        mt5_server_time_offset_hours_current=int(time.get("mt5_server_time_offset_hours_current", 3)),
        controls_version=str(metadata.get("controls_version", "unknown")),
    )


def initialise_terminal(mt5: GuardedMt5OrderCheckProxy, terminal_path: str | None) -> None:
    if terminal_path:
        ok = bool(mt5.initialize(terminal_path))
    else:
        ok = bool(mt5.initialize())
    if not ok:
        raise Stage3OrderPreflightError(f"mt5.initialize() failed: {safe_last_error(mt5)}")


def inspect_terminal_for_trading(mt5: GuardedMt5OrderCheckProxy) -> dict[str, Any]:
    info = mt5.terminal_info()
    if info is None:
        raise Stage3OrderPreflightError(f"mt5.terminal_info() returned None: {safe_last_error(mt5)}")
    snapshot = object_to_plain_dict(info)
    if snapshot.get("connected") is not True:
        raise Stage3OrderPreflightError("MT5 terminal is not connected")
    return snapshot


def inspect_account_for_trading(mt5: GuardedMt5OrderCheckProxy, controls: FrozenBrokerControls) -> dict[str, Any]:
    info = mt5.account_info()
    if info is None:
        raise Stage3OrderPreflightError(f"mt5.account_info() returned None: {safe_last_error(mt5)}")
    snapshot = object_to_plain_dict(info)
    snapshot["trade_mode_name"] = trade_mode_to_name(mt5, snapshot.get("trade_mode"))
    snapshot["login_masked"] = mask_login(snapshot.get("login"))
    snapshot.pop("login", None)
    if snapshot.get("trade_mode_name") != "DEMO":
        raise Stage3OrderPreflightError(f"Connected account is not DEMO: {snapshot.get('trade_mode_name')}")
    if snapshot.get("company") != controls.broker_company_expected:
        raise Stage3OrderPreflightError(
            f"Broker mismatch. Expected {controls.broker_company_expected!r}, got {snapshot.get('company')!r}"
        )
    if snapshot.get("server") != controls.server_expected:
        raise Stage3OrderPreflightError(
            f"Server mismatch. Expected {controls.server_expected!r}, got {snapshot.get('server')!r}"
        )
    return snapshot


def inspect_symbol_for_order_check(mt5: GuardedMt5OrderCheckProxy, controls: FrozenBrokerControls) -> dict[str, Any]:
    info = mt5.symbol_info(controls.symbol)
    if info is None:
        raise Stage3OrderPreflightError(f"symbol_info({controls.symbol!r}) returned None")
    snapshot = object_to_plain_dict(info)
    checks = {
        "visible": bool(snapshot.get("visible", False)),
        "trade_mode_recorded": "trade_mode" in snapshot,
        "volume_min_allows_test_volume": float(snapshot.get("volume_min", math.inf)) <= controls.stage3_first_order_test_volume_lots,
        "volume_step_allows_test_volume": is_volume_step_valid(
            controls.stage3_first_order_test_volume_lots,
            float(snapshot.get("volume_min", math.nan)),
            float(snapshot.get("volume_step", math.nan)),
        ),
        "spread_within_gate": 0 <= int(snapshot.get("spread", -1)) <= controls.max_spread_points_for_entry,
    }
    if not all(checks.values()):
        raise Stage3OrderPreflightError(f"Symbol failed Stage 3 preflight checks: {checks}")
    snapshot["preflight_checks"] = checks
    return snapshot


def is_volume_step_valid(volume: float, volume_min: float, volume_step: float, eps: float = 1e-9) -> bool:
    if not math.isfinite(volume_min) or not math.isfinite(volume_step):
        return False
    if volume_step <= 0 or volume < volume_min - eps:
        return False
    steps = (volume - volume_min) / volume_step
    return abs(steps - round(steps)) <= eps


def inspect_tick_for_order_check(mt5: GuardedMt5OrderCheckProxy, symbol: str) -> dict[str, Any]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise Stage3OrderPreflightError(f"symbol_info_tick({symbol!r}) returned None: {safe_last_error(mt5)}")
    snapshot = object_to_plain_dict(tick)
    ask = float(snapshot.get("ask", 0.0) or 0.0)
    bid = float(snapshot.get("bid", 0.0) or 0.0)
    if ask <= 0 or bid <= 0 or ask < bid:
        raise Stage3OrderPreflightError(f"Invalid tick prices for order_check: bid={bid}, ask={ask}")
    snapshot["available"] = True
    return snapshot


def choose_filling_candidates(mt5: Any, symbol_info: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Return filling candidates in a conservative order.

    MT5 symbol_info.filling_mode is treated as broker symbol permission flags,
    not blindly as the order request type_filling.  If IOC is allowed, try IOC
    first.  If FOK is allowed, try FOK.  Finally try RETURN only when the symbol
    is not market-execution, because RETURN is not allowed for market execution.
    """
    filling_flags = int(symbol_info.get("filling_mode", 0) or 0)
    trade_exemode = int(symbol_info.get("trade_exemode", -999) or -999)
    market_execution = trade_exemode == int(getattr(mt5, "SYMBOL_TRADE_EXECUTION_MARKET", -9999))

    candidates: list[tuple[str, int]] = []
    # SYMBOL_FILLING_IOC is normally bit 2, but use MT5 constant when exposed.
    symbol_filling_ioc = int(getattr(mt5, "SYMBOL_FILLING_IOC", 2))
    symbol_filling_fok = int(getattr(mt5, "SYMBOL_FILLING_FOK", 1))
    if filling_flags & symbol_filling_ioc:
        candidates.append(("ORDER_FILLING_IOC", int(getattr(mt5, "ORDER_FILLING_IOC"))))
    if filling_flags & symbol_filling_fok:
        candidates.append(("ORDER_FILLING_FOK", int(getattr(mt5, "ORDER_FILLING_FOK"))))
    if not market_execution:
        candidates.append(("ORDER_FILLING_RETURN", int(getattr(mt5, "ORDER_FILLING_RETURN"))))

    # Deduplicate while preserving order.
    seen: set[int] = set()
    unique: list[tuple[str, int]] = []
    for name, value in candidates:
        if value not in seen:
            seen.add(value)
            unique.append((name, value))
    if not unique:
        raise Stage3OrderPreflightError(
            f"No safe filling candidate could be derived from filling_mode={filling_flags}, trade_exemode={trade_exemode}"
        )
    return unique


def build_order_request(
    *,
    mt5: Any,
    controls: FrozenBrokerControls,
    side: str,
    tick: Mapping[str, Any],
    filling_value: int,
) -> dict[str, Any]:
    """Build a clean MqlTradeRequest dictionary for MT5.

    Only official MT5 request fields are returned.  Human-readable metadata such
    as the filling policy name is kept outside this dictionary so it can never
    be accidentally submitted to mt5.order_check().
    """
    side_upper = side.upper()
    if side_upper not in {"BUY", "SELL"}:
        raise ValueError(f"Unsupported side: {side}")
    order_type = int(getattr(mt5, f"ORDER_TYPE_{side_upper}"))
    price = float(tick["ask"] if side_upper == "BUY" else tick["bid"])
    magic = controls.model_b_magic_number if side_upper == "BUY" else controls.model_a_magic_number
    comment = "CP_S3P1_CHECK_B" if side_upper == "BUY" else "CP_S3P1_CHECK_A"
    return {
        "action": int(getattr(mt5, "TRADE_ACTION_DEAL")),
        "symbol": controls.symbol,
        "volume": controls.stage3_first_order_test_volume_lots,
        "type": order_type,
        "price": price,
        "deviation": controls.max_deviation_points_for_stage3_order_request,
        "magic": magic,
        "comment": comment,
        "type_time": int(getattr(mt5, "ORDER_TIME_GTC")),
        "type_filling": filling_value,
    }


def normalise_check_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    plain = object_to_plain_dict(result)
    # Convert nested request namedtuple if needed.
    req = plain.get("request")
    if req is not None and not isinstance(req, (str, int, float, bool)):
        plain["request"] = object_to_plain_dict(req)
    return plain


def check_result_has_empty_request(result: Any) -> bool:
    """Detect MT5 builds that silently bind order_check keyword args as empty."""
    plain = normalise_check_result(result)
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


def call_order_calc_margin_compat(
    mt5: GuardedMt5OrderCheckProxy,
    *,
    action: int,
    symbol: str,
    volume: float,
    price: float,
) -> float | None:
    """Call order_calc_margin using the argument style accepted by this MT5 build.

    Some MetaTrader5 Python builds reject positional, unnamed arguments for
    trading-related functions and return RES_E_INVALID_PARAMS /
    "Unnamed arguments not allowed".  Use named arguments first, with a
    positional fallback for older builds.  This function does not execute an
    order.
    """
    try:
        margin = mt5.order_calc_margin(action=action, symbol=symbol, volume=volume, price=price)
    except TypeError:
        margin = mt5.order_calc_margin(action, symbol, volume, price)
    return float(margin) if margin is not None else None


def call_order_check_compat(mt5: GuardedMt5OrderCheckProxy, request: Mapping[str, Any]) -> Any:
    """Call order_check using the argument style accepted by this MT5 build.

    Some MetaTrader5 Python builds reject positional calls with "Unnamed
    arguments not allowed"; others accept the keyword call but bind an empty
    request, which comes back as broker retcode 10013 / "Invalid request".
    Try the keyword request style first, then expanded request fields for
    builds that require named trade-request fields.  Keep positional as the
    last fallback for older builds.
    """
    clean_request = dict(request)
    try:
        result = mt5.order_check(request=clean_request)
    except TypeError:
        result = None
    if result is not None and not check_result_has_empty_request(result):
        return result

    try:
        result = mt5.order_check(**clean_request)
    except TypeError:
        result = None
    if result is not None and not check_result_has_empty_request(result):
        return result

    try:
        return mt5.order_check(clean_request)
    except TypeError:
        return result
    return result


def run_order_check_attempt(
    *,
    mt5: GuardedMt5OrderCheckProxy,
    controls: FrozenBrokerControls,
    side: str,
    tick: Mapping[str, Any],
    filling_name: str,
    filling_value: int,
) -> OrderCheckAttempt:
    request = build_order_request(
        mt5=mt5,
        controls=controls,
        side=side,
        tick=tick,
        filling_value=filling_value,
    )
    price = float(request["price"])
    order_type = int(request["type"])
    margin_value = call_order_calc_margin_compat(
        mt5,
        action=order_type,
        symbol=controls.symbol,
        volume=controls.stage3_first_order_test_volume_lots,
        price=price,
    )
    check = call_order_check_compat(mt5, request)
    check_result = normalise_check_result(check)
    retcode = None
    comment = None
    passed = False
    if check_result is not None:
        raw_retcode = check_result.get("retcode")
        try:
            retcode = int(raw_retcode) if raw_retcode is not None else None
        except Exception:
            retcode = None
        comment = None if check_result.get("comment") is None else str(check_result.get("comment"))
        passed = retcode in SUCCESS_ORDER_CHECK_RETCODES
    return OrderCheckAttempt(
        side=side.upper(),
        request=request,
        check_result=check_result,
        margin_required_account_currency=margin_value,
        last_error=safe_last_error(mt5),
        passed=passed,
        retcode=retcode,
        comment=comment,
        filling_name=filling_name,
        filling_value=filling_value,
    )


def build_package_snapshot(mt5: Any) -> dict[str, Any]:
    try:
        terminal_version = mt5.version()
    except Exception:
        terminal_version = None
    return {
        "module_author": getattr(mt5, "__author__", None),
        "module_version": getattr(mt5, "__version__", None),
        "terminal_version": str(terminal_version),
    }


def capital_review_from_live_account(
    *,
    controls: FrozenBrokerControls,
    account: Mapping[str, Any],
    symbol_info: Mapping[str, Any],
) -> dict[str, Any]:
    equity = _float_or_none(account.get("equity"))
    currency = str(account.get("currency", ""))
    ask = _float_or_none(symbol_info.get("ask")) or _float_or_none(symbol_info.get("bid"))
    contract_size = _float_or_none(symbol_info.get("trade_contract_size")) or 100.0
    notional_profit_currency = (
        ask * contract_size * controls.stage3_first_order_test_volume_lots if ask is not None else None
    )
    notional_account_currency_estimate = None
    effective_leverage_estimate = None
    capstone_passed = None
    required_equity_estimate = None
    if notional_profit_currency is not None:
        if currency.upper() == "SGD":
            notional_account_currency_estimate = notional_profit_currency * controls.review_usd_sgd_rate_assumption
        else:
            notional_account_currency_estimate = notional_profit_currency
        required_equity_estimate = notional_account_currency_estimate / controls.capstone_leverage_cap
        if equity is not None and equity > 0:
            effective_leverage_estimate = notional_account_currency_estimate / equity
            capstone_passed = effective_leverage_estimate <= controls.capstone_leverage_cap + 1e-12
    return {
        "account_currency": currency,
        "equity": equity,
        "order_volume_lots": controls.stage3_first_order_test_volume_lots,
        "contract_size": contract_size,
        "price_reference": ask,
        "notional_symbol_profit_currency": notional_profit_currency,
        "notional_account_currency_estimate": notional_account_currency_estimate,
        "review_usd_sgd_rate_assumption": controls.review_usd_sgd_rate_assumption,
        "effective_leverage_estimate": effective_leverage_estimate,
        "required_equity_estimate_at_10x_cap": required_equity_estimate,
        "capstone_10x_leverage_cap_passed": capstone_passed,
        "minimum_demo_equity_recommendation_sgd": controls.minimum_demo_equity_recommendation_sgd,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def pick_best_attempt(attempts: list[OrderCheckAttempt], side: str) -> OrderCheckAttempt | None:
    side_attempts = [attempt for attempt in attempts if attempt.side == side.upper()]
    passed = [attempt for attempt in side_attempts if attempt.passed]
    return passed[0] if passed else (side_attempts[0] if side_attempts else None)


def run_stage3_order_preflight(
    *,
    mt5_module: Any,
    controls: FrozenBrokerControls,
    terminal_path: str | None,
    sides: tuple[str, ...] = ("BUY", "SELL"),
) -> dict[str, Any]:
    if not controls.stage3_order_check_required_before_order_send:
        raise Stage3OrderPreflightError("Frozen controls do not require order_check before order_send; refusing Stage 3.")
    proxy = GuardedMt5OrderCheckProxy(mt5_module)
    initialized = False
    terminal_initialized_success = False
    shutdown_called = False
    attempts: list[OrderCheckAttempt] = []
    warnings: list[str] = []
    hard_gate_passed = False
    try:
        initialise_terminal(proxy, terminal_path)
        initialized = True
        terminal_initialized_success = True
        package = build_package_snapshot(proxy)
        terminal = inspect_terminal_for_trading(proxy)
        account = inspect_account_for_trading(proxy, controls)
        symbol_info = inspect_symbol_for_order_check(proxy, controls)
        tick = inspect_tick_for_order_check(proxy, controls.symbol)

        filling_candidates = choose_filling_candidates(proxy, symbol_info)
        for side in sides:
            side_upper = side.upper()
            for filling_name, filling_value in filling_candidates:
                attempt = run_order_check_attempt(
                    mt5=proxy,
                    controls=controls,
                    side=side_upper,
                    tick=tick,
                    filling_name=filling_name,
                    filling_value=filling_value,
                )
                attempts.append(attempt)
                if attempt.passed:
                    break

        best_buy = pick_best_attempt(attempts, "BUY")
        best_sell = pick_best_attempt(attempts, "SELL")
        side_pass = {side.upper(): bool(pick_best_attempt(attempts, side) and pick_best_attempt(attempts, side).passed) for side in sides}
        capital_review = capital_review_from_live_account(controls=controls, account=account, symbol_info=symbol_info)

        validations = {
            "terminal_initialized": terminal_initialized_success,
            "terminal_connected": terminal.get("connected") is True,
            "terminal_trade_allowed": terminal.get("trade_allowed") is True,
            "terminal_tradeapi_enabled": terminal.get("tradeapi_disabled") is False,
            "account_is_demo": account.get("trade_mode_name") == "DEMO",
            "account_trade_allowed": account.get("trade_allowed") is True,
            "account_expert_trading_allowed": account.get("trade_expert") is True,
            "symbol_matches_frozen_config": symbol_info.get("name") == controls.symbol,
            "symbol_spread_within_entry_gate": int(symbol_info.get("spread", 999999)) <= controls.max_spread_points_for_entry,
            "order_check_called": "order_check" in proxy.calls,
            "order_send_not_called": "order_send" not in proxy.calls and "order_send" not in proxy.forbidden_attempts,
            "buy_order_check_passed": side_pass.get("BUY") if "BUY" in [s.upper() for s in sides] else None,
            "sell_order_check_passed": side_pass.get("SELL") if "SELL" in [s.upper() for s in sides] else None,
            "capstone_10x_leverage_cap_passed": capital_review.get("capstone_10x_leverage_cap_passed"),
        }
        required = [
            "terminal_initialized",
            "terminal_connected",
            "terminal_trade_allowed",
            "terminal_tradeapi_enabled",
            "account_is_demo",
            "account_trade_allowed",
            "account_expert_trading_allowed",
            "symbol_matches_frozen_config",
            "symbol_spread_within_entry_gate",
            "order_check_called",
            "order_send_not_called",
            "buy_order_check_passed",
        ]
        # SELL is required for Model A future compatibility only when requested.
        if "SELL" in [s.upper() for s in sides]:
            required.append("sell_order_check_passed")
        hard_gate_passed = all(validations.get(key) is True for key in required)

        if terminal.get("trade_allowed") is not True:
            warnings.append("Terminal trading is disabled. Enable the MT5 Algo Trading button before Stage 3 Step 1 can pass.")
        if capital_review.get("capstone_10x_leverage_cap_passed") is False:
            warnings.append(
                "The live account equity does not satisfy the capstone 10:1 leverage cap for 0.01 XAUUSD lot. This can still be used for order plumbing only, not final strategy execution."
            )
        if best_buy and best_buy.passed and best_sell and not best_sell.passed:
            warnings.append("BUY order_check passed but SELL did not. Model B plumbing can proceed, but Model A short execution must remain blocked.")

        if initialized:
            try:
                proxy.shutdown()
                shutdown_called = True
                initialized = False
            except Exception:
                shutdown_called = False

        report = {
            "stage": 3,
            "step": 1,
            "status": "PASS" if hard_gate_passed else "FAIL",
            "formal_gate": hard_gate_passed,
            "purpose": "order_permission_and_order_check_preflight_no_order_send",
            "mt5_used": True,
            "order_check_called": "order_check" in proxy.calls,
            "order_send_called": False,
            "orders_executed": False,
            "terminal_initialized": terminal_initialized_success,
            "shutdown_called": shutdown_called,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "controls": asdict(controls),
            "package": package,
            "terminal": terminal,
            "account": account,
            "symbol_info": symbol_info,
            "tick": tick,
            "filling_candidates": [
                {"name": name, "value": value} for name, value in filling_candidates
            ],
            "order_check_attempts": [asdict(attempt) for attempt in attempts],
            "selected_order_check": {
                "BUY": asdict(best_buy) if best_buy else None,
                "SELL": asdict(best_sell) if best_sell else None,
            },
            "capital_review": capital_review,
            "validations": validations,
            "required_gate_keys": required,
            "read_only_plus_order_check_methods_used": tuple(proxy.calls),
            "forbidden_function_attempts": tuple(proxy.forbidden_attempts),
            "warnings": warnings,
            "decision": {
                "stage3_step2_tiny_order_test_allowed": hard_gate_passed,
                "stage3_step2_volume_lots": controls.stage3_first_order_test_volume_lots,
                "model_b_first_controlled_execution_candidate": controls.model_b_variant_for_first_controlled_execution,
                "order_send_remains_blocked_until_stage3_step2": True,
            },
        }
        return report
    finally:
        if initialized:
            try:
                proxy.shutdown()
                shutdown_called = True
            except Exception:
                shutdown_called = False
        # The report is normally returned before finally ends, so set through
        # mutable update is not possible here.  run_preflight() updates this field.
        _ = shutdown_called


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def write_order_check_csv(path: Path, attempts: list[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        check_result = attempt.get("check_result") or {}
        rows.append(
            {
                "side": attempt.get("side"),
                "passed": attempt.get("passed"),
                "retcode": attempt.get("retcode"),
                "comment": attempt.get("comment"),
                "filling_name": attempt.get("filling_name"),
                "filling_value": attempt.get("filling_value"),
                "margin_required_account_currency": attempt.get("margin_required_account_currency"),
                "volume": (attempt.get("request") or {}).get("volume"),
                "price": (attempt.get("request") or {}).get("price"),
                "check_balance": check_result.get("balance"),
                "check_equity": check_result.get("equity"),
                "check_margin": check_result.get("margin"),
                "check_margin_free": check_result.get("margin_free"),
                "check_margin_level": check_result.get("margin_level"),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "side",
                "passed",
                "retcode",
                "comment",
                "filling_name",
                "filling_value",
                "margin_required_account_currency",
                "volume",
                "price",
                "check_balance",
                "check_equity",
                "check_margin",
                "check_margin_free",
                "check_margin_level",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_preflight(
    *,
    repo_root: Path,
    terminal_path: str | None = None,
    frozen_config_path: Path = DEFAULT_FROZEN_CONFIG_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    order_check_csv_path: Path = DEFAULT_ORDER_CHECK_CSV_PATH,
    sides: tuple[str, ...] = ("BUY", "SELL"),
    mt5_module: Any | None = None,
) -> dict[str, Any]:
    controls = load_frozen_controls(repo_root / frozen_config_path)
    if mt5_module is None:
        if import_metatrader5_module is None:  # pragma: no cover
            raise Stage3OrderPreflightError("MetaTrader5 import helper is unavailable")
        mt5_module = import_metatrader5_module()
    report = run_stage3_order_preflight(
        mt5_module=mt5_module,
        controls=controls,
        terminal_path=terminal_path,
        sides=sides,
    )
    report["report_path"] = str(report_path)
    report["order_check_csv_path"] = str(order_check_csv_path)
    report["frozen_config_path"] = str(frozen_config_path)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    # mt5.shutdown has already been called in run_stage3_order_preflight finally.
    report["shutdown_called"] = "shutdown" in report.get("read_only_plus_order_check_methods_used", [])
    write_json(repo_root / report_path, report)
    write_order_check_csv(repo_root / order_check_csv_path, report.get("order_check_attempts", []))
    return report
