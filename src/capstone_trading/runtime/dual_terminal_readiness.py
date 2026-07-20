"""Stage 3 Step 4A dual-terminal and dual-account readiness gate.

This gate is intentionally no-order. It sequentially attaches the MetaTrader5
Python package to two explicitly supplied terminal64.exe installations, verifies
that they are independent, and records broker/account/XAUUSD/M15 readiness for
Model A and Model B. order_check and order_send are never exposed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import math
import os

import pandas as pd

from capstone_trading.runtime.mt5_readiness import (
    Mt5ReadinessError,
    Mt5RuntimeConfig,
    analyse_rates,
    build_package_snapshot,
    fetch_completed_m15_rates,
    inspect_tick,
    mask_login,
    object_to_plain_dict,
    resolve_symbol,
    resolve_timeframe,
    safe_last_error,
    trade_mode_to_name,
)


class DualTerminalReadinessError(RuntimeError):
    """Raised when Stage 3 Step 4A cannot establish independent readiness."""

@dataclass(frozen=True)
class TimeNormalisationReport:
    policy: str
    mt5_server_time_offset_hours: int
    raw_first_bar_server_time: str | None
    raw_latest_bar_server_time: str | None
    canonical_first_bar_time_utc: str | None
    canonical_latest_bar_time_utc: str | None
    latest_bar_age_minutes_after_conversion: float | None
    latest_bar_age_minutes_before_conversion: float | None
    latest_bar_future_minutes_after_conversion: float
    raw_tick_server_time: str | None
    canonical_tick_time_utc: str | None
    tick_age_minutes_after_conversion: float | None
    conversion_applied: bool
    server_time_note: str


def _raw_server_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        if isinstance(value, pd.Timestamp):
            return value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
        if isinstance(value, (int, float)):
            return pd.Timestamp(datetime.fromtimestamp(float(value), tz=timezone.utc))
        parsed = pd.Timestamp(value)
        return parsed.tz_convert("UTC") if parsed.tzinfo is not None else parsed.tz_localize("UTC")
    except Exception:
        return None


def normalise_mt5_server_times(
    rates: pd.DataFrame,
    tick: Mapping[str, Any],
    *,
    server_time_offset_hours: int,
    now_utc: datetime | None = None,
    enforce_no_future_canonical_bar: bool = True,
    max_future_canonical_bar_minutes: int = 2,
) -> tuple[pd.DataFrame, TimeNormalisationReport]:
    if "time" not in rates.columns:
        raise DualTerminalReadinessError("MT5 rates are missing the time column")
    if server_time_offset_hours not in {0, 2, 3}:
        raise DualTerminalReadinessError("MT5 server time offset must be 0, 2, or 3 hours")
    frame = rates.copy()
    raw_times = pd.to_datetime(frame["time"], utc=True)
    offset = pd.Timedelta(hours=int(server_time_offset_hours))
    canonical_times = raw_times - offset
    frame["time"] = canonical_times
    frame = frame.sort_values("time").reset_index(drop=True)
    now = pd.Timestamp(now_utc or datetime.now(timezone.utc)).tz_convert("UTC")
    raw_latest = pd.Timestamp(raw_times.iloc[-1]).tz_convert("UTC") if len(raw_times) else None
    canonical_latest = pd.Timestamp(canonical_times.iloc[-1]).tz_convert("UTC") if len(canonical_times) else None
    age_before = float((now - raw_latest).total_seconds() / 60.0) if raw_latest is not None else None
    age_after = float((now - canonical_latest).total_seconds() / 60.0) if canonical_latest is not None else None
    future_minutes = max(0.0, -age_after) if age_after is not None else 0.0
    if enforce_no_future_canonical_bar and future_minutes > max_future_canonical_bar_minutes:
        raise DualTerminalReadinessError(
            "Latest completed MT5 bar remains in the future after UTC conversion: "
            f"future_minutes={future_minutes:.2f}, offset_hours={server_time_offset_hours}"
        )
    raw_tick = _raw_server_timestamp(tick.get("time")) if tick else None
    canonical_tick = raw_tick - offset if raw_tick is not None else None
    tick_age = float((now - canonical_tick).total_seconds() / 60.0) if canonical_tick is not None else None
    report = TimeNormalisationReport(
        policy="fixed_broker_server_offset_to_utc",
        mt5_server_time_offset_hours=int(server_time_offset_hours),
        raw_first_bar_server_time=pd.Timestamp(raw_times.iloc[0]).tz_convert("UTC").isoformat() if len(raw_times) else None,
        raw_latest_bar_server_time=raw_latest.isoformat() if raw_latest is not None else None,
        canonical_first_bar_time_utc=pd.Timestamp(canonical_times.iloc[0]).tz_convert("UTC").isoformat() if len(canonical_times) else None,
        canonical_latest_bar_time_utc=canonical_latest.isoformat() if canonical_latest is not None else None,
        latest_bar_age_minutes_after_conversion=age_after,
        latest_bar_age_minutes_before_conversion=age_before,
        latest_bar_future_minutes_after_conversion=float(future_minutes),
        raw_tick_server_time=raw_tick.isoformat() if raw_tick is not None else None,
        canonical_tick_time_utc=canonical_tick.isoformat() if canonical_tick is not None else None,
        tick_age_minutes_after_conversion=tick_age,
        conversion_applied=bool(server_time_offset_hours != 0),
        server_time_note="Dukascopy MT5 server timestamps are converted to canonical UTC before comparison.",
    )
    return frame, report



FORBIDDEN_METHODS: tuple[str, ...] = (
    "order_check",
    "order_send",
    "history_orders_get",
    "history_deals_get",
)

READ_ONLY_METHODS: tuple[str, ...] = (
    "initialize",
    "shutdown",
    "last_error",
    "version",
    "terminal_info",
    "account_info",
    "symbol_info",
    "symbol_info_tick",
    "symbol_select",
    "copy_rates_from_pos",
    "positions_get",
    "orders_get",
)


@dataclass(frozen=True)
class TerminalRoleConfig:
    role: str
    terminal_path: str
    runtime_root: str
    magic_number: int
    order_comment: str


@dataclass(frozen=True)
class DualTerminalConfig:
    model_a: TerminalRoleConfig
    model_b: TerminalRoleConfig
    broker_company_expected: str = "Dukascopy Bank SA"
    server_expected: str = "Dukascopy-demo-mt5-1"
    symbol: str = "XAUUSD"
    timeframe: str = "M15"
    bars_to_fetch: int = 260
    min_completed_bars: int = 200
    mt5_server_time_offset_hours: int = 3
    require_demo_account: bool = True
    require_terminal_trade_allowed: bool = True
    require_trade_api_enabled: bool = True
    require_account_trade_allowed: bool = True
    require_account_expert_allowed: bool = True
    require_distinct_accounts: bool = True
    require_distinct_terminal_paths: bool = True
    require_distinct_data_paths: bool = True
    require_no_open_symbol_positions: bool = True
    require_no_pending_symbol_orders: bool = True
    require_same_currency: bool = True
    require_same_margin_mode: bool = True
    require_same_symbol_contract: bool = True
    max_latest_bar_difference_minutes: float = 15.0
    min_equity_per_account_sgd: float = 1000.0
    max_starting_balance_difference_percent: float = 1.0
    capstone_leverage_cap: float = 10.0
    usd_sgd_review_rate: float = 1.35
    test_volume_lots: float = 0.01


class DualTerminalSafeProxy:
    """Narrow no-order MT5 proxy used by the dual-terminal gate."""

    def __init__(self, mt5_module: Any):
        self._mt5 = mt5_module
        self.calls: list[str] = []
        self.forbidden_attempts: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_METHODS:
            self.forbidden_attempts.append(name)
            raise DualTerminalReadinessError(f"Forbidden MT5 method accessed: {name}")
        if name in READ_ONLY_METHODS or name.startswith("TIMEFRAME_") or name.startswith(
            "ACCOUNT_TRADE_MODE_"
        ):
            attr = getattr(self._mt5, name)
            if callable(attr) and name in READ_ONLY_METHODS:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    self.calls.append(name)
                    return attr(*args, **kwargs)

                return wrapped
            return attr
        if name in {"__author__", "__version__"}:
            return getattr(self._mt5, name, None)
        raise AttributeError(name)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise DualTerminalReadinessError("PyYAML is required for the Step 4A config") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise DualTerminalReadinessError("Step 4A config must be a YAML mapping")
    return raw


def _role_from_mapping(name: str, raw: Mapping[str, Any]) -> TerminalRoleConfig:
    terminal_path = str(raw.get("terminal_path", "")).strip()
    runtime_root = str(raw.get("runtime_root", "")).strip()
    if not terminal_path:
        raise DualTerminalReadinessError(f"{name}.terminal_path is required")
    if not runtime_root:
        raise DualTerminalReadinessError(f"{name}.runtime_root is required")
    return TerminalRoleConfig(
        role=name,
        terminal_path=terminal_path,
        runtime_root=runtime_root,
        magic_number=int(raw.get("magic_number")),
        order_comment=str(raw.get("order_comment", "")).strip(),
    )


def load_dual_terminal_config(path: Path) -> DualTerminalConfig:
    raw = _load_yaml_mapping(path)
    roles = raw.get("roles", {}) or {}
    broker = raw.get("broker", {}) or {}
    data = raw.get("data", {}) or {}
    safety = raw.get("safety", {}) or {}
    capital = raw.get("capital", {}) or {}
    if not all(isinstance(x, Mapping) for x in (roles, broker, data, safety, capital)):
        raise DualTerminalReadinessError("Invalid Step 4A config structure")
    model_a_raw = roles.get("model_a", {}) or {}
    model_b_raw = roles.get("model_b", {}) or {}
    if not isinstance(model_a_raw, Mapping) or not isinstance(model_b_raw, Mapping):
        raise DualTerminalReadinessError("roles.model_a and roles.model_b must be mappings")
    return DualTerminalConfig(
        model_a=_role_from_mapping("MODEL_A", model_a_raw),
        model_b=_role_from_mapping("MODEL_B", model_b_raw),
        broker_company_expected=str(broker.get("company", "Dukascopy Bank SA")),
        server_expected=str(broker.get("server", "Dukascopy-demo-mt5-1")),
        symbol=str(broker.get("symbol", "XAUUSD")),
        timeframe=str(data.get("timeframe", "M15")).upper(),
        bars_to_fetch=int(data.get("bars_to_fetch", 260)),
        min_completed_bars=int(data.get("min_completed_bars", 200)),
        mt5_server_time_offset_hours=int(data.get("mt5_server_time_offset_hours", 3)),
        require_demo_account=bool(safety.get("require_demo_account", True)),
        require_terminal_trade_allowed=bool(safety.get("require_terminal_trade_allowed", True)),
        require_trade_api_enabled=bool(safety.get("require_trade_api_enabled", True)),
        require_account_trade_allowed=bool(safety.get("require_account_trade_allowed", True)),
        require_account_expert_allowed=bool(safety.get("require_account_expert_allowed", True)),
        require_distinct_accounts=bool(safety.get("require_distinct_accounts", True)),
        require_distinct_terminal_paths=bool(safety.get("require_distinct_terminal_paths", True)),
        require_distinct_data_paths=bool(safety.get("require_distinct_data_paths", True)),
        require_no_open_symbol_positions=bool(safety.get("require_no_open_symbol_positions", True)),
        require_no_pending_symbol_orders=bool(safety.get("require_no_pending_symbol_orders", True)),
        require_same_currency=bool(safety.get("require_same_currency", True)),
        require_same_margin_mode=bool(safety.get("require_same_margin_mode", True)),
        require_same_symbol_contract=bool(safety.get("require_same_symbol_contract", True)),
        max_latest_bar_difference_minutes=float(safety.get("max_latest_bar_difference_minutes", 15.0)),
        min_equity_per_account_sgd=float(capital.get("min_equity_per_account_sgd", 1000.0)),
        max_starting_balance_difference_percent=float(
            capital.get("max_starting_balance_difference_percent", 1.0)
        ),
        capstone_leverage_cap=float(capital.get("capstone_leverage_cap", 10.0)),
        usd_sgd_review_rate=float(capital.get("usd_sgd_review_rate", 1.35)),
        test_volume_lots=float(capital.get("test_volume_lots", 0.01)),
    )


def _normalised_windows_path(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).expanduser().resolve())))


def _terminal_executable_path(role: TerminalRoleConfig) -> Path:
    path = Path(role.terminal_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise DualTerminalReadinessError(f"{role.role} terminal executable does not exist: {path}")
    if path.name.lower() not in {"terminal64.exe", "terminal.exe", "metatrader64.exe", "metatrader.exe"}:
        raise DualTerminalReadinessError(
            f"{role.role} path must identify an MT5 terminal executable, received: {path.name}"
        )
    return path


def _account_snapshot(proxy: Any, *, require_demo: bool) -> tuple[dict[str, Any], Any]:
    info = proxy.account_info()
    if info is None:
        raise DualTerminalReadinessError(f"account_info returned None: {safe_last_error(proxy)}")
    raw = object_to_plain_dict(info)
    login = raw.get("login")
    trade_mode_name = trade_mode_to_name(proxy, raw.get("trade_mode"))
    raw["trade_mode_name"] = trade_mode_name
    raw["login_masked"] = mask_login(login)
    raw.pop("login", None)
    if require_demo and trade_mode_name != "DEMO":
        raise DualTerminalReadinessError(f"Expected DEMO account, detected {trade_mode_name}")
    return raw, login


def _terminal_snapshot(proxy: Any) -> dict[str, Any]:
    info = proxy.terminal_info()
    if info is None:
        raise DualTerminalReadinessError(f"terminal_info returned None: {safe_last_error(proxy)}")
    data = object_to_plain_dict(info)
    if data.get("connected") is not True:
        raise DualTerminalReadinessError("MT5 terminal is not connected")
    return data


def _safe_count(items: Any, *, label: str) -> int:
    if items is None:
        raise DualTerminalReadinessError(f"{label} returned None")
    return len(tuple(items))


def _capital_review(
    *, account: Mapping[str, Any], symbol: Mapping[str, Any], config: DualTerminalConfig
) -> dict[str, Any]:
    equity = float(account.get("equity", 0.0) or 0.0)
    currency = str(account.get("currency", ""))
    ask = float(symbol.get("ask", 0.0) or 0.0)
    bid = float(symbol.get("bid", 0.0) or 0.0)
    price = ask if ask > 0 else bid
    contract_size = float(symbol.get("trade_contract_size", 0.0) or 0.0)
    notional_usd = price * contract_size * config.test_volume_lots
    notional_sgd = notional_usd * config.usd_sgd_review_rate
    effective_leverage = notional_sgd / equity if equity > 0 else math.inf
    return {
        "account_currency": currency,
        "account_balance": float(account.get("balance", 0.0) or 0.0),
        "account_equity": equity,
        "test_volume_lots": config.test_volume_lots,
        "estimated_notional_usd": notional_usd,
        "estimated_notional_sgd": notional_sgd,
        "effective_leverage_estimate": effective_leverage,
        "capstone_leverage_cap": config.capstone_leverage_cap,
        "capstone_leverage_cap_passed": effective_leverage <= config.capstone_leverage_cap,
        "minimum_equity_required_sgd": config.min_equity_per_account_sgd,
        "minimum_equity_passed": currency == "SGD" and equity >= config.min_equity_per_account_sgd,
    }


def _contract_signature(symbol_info: Mapping[str, Any]) -> tuple[Any, ...]:
    keys = (
        "name", "digits", "point", "trade_contract_size", "volume_min", "volume_step",
        "volume_max", "currency_base", "currency_profit", "trade_calc_mode", "filling_mode",
    )
    return tuple(symbol_info.get(key) for key in keys)


def inspect_terminal_role(
    *, mt5_module: Any, role: TerminalRoleConfig, config: DualTerminalConfig,
    now_utc: datetime | None = None,
) -> tuple[dict[str, Any], Any]:
    terminal_exe = _terminal_executable_path(role)
    proxy = DualTerminalSafeProxy(mt5_module)
    initialized = False
    shutdown_called = False
    snapshot: dict[str, Any] | None = None
    raw_login: Any = None
    primary_error: Exception | None = None
    try:
        timeframe_value = resolve_timeframe(proxy, config.timeframe)
        if not bool(proxy.initialize(str(terminal_exe))):
            raise DualTerminalReadinessError(
                f"{role.role} mt5.initialize failed: {safe_last_error(proxy)}"
            )
        initialized = True
        package = build_package_snapshot(proxy)
        terminal = _terminal_snapshot(proxy)
        account, raw_login = _account_snapshot(proxy, require_demo=config.require_demo_account)
        mt5_config = Mt5RuntimeConfig(
            terminal_path=str(terminal_exe),
            symbol_candidates=(config.symbol,),
            timeframe_name=config.timeframe,
            bars_to_fetch=config.bars_to_fetch,
            min_completed_bars=config.min_completed_bars,
            require_demo_account=config.require_demo_account,
            allow_market_closed_stale_bar=True,
            max_latest_closed_bar_age_minutes_warning=4320,
            require_symbol_visible=True,
        )
        symbol_resolution = resolve_symbol(proxy, mt5_config)
        symbol_info = dict(symbol_resolution.symbol_info)
        tick = inspect_tick(proxy, symbol_resolution.selected_symbol)
        raw_rates = fetch_completed_m15_rates(
            proxy,
            symbol=symbol_resolution.selected_symbol,
            timeframe_value=timeframe_value,
            count=config.bars_to_fetch,
        )
        canonical_rates, time_report = normalise_mt5_server_times(
            raw_rates,
            tick,
            server_time_offset_hours=config.mt5_server_time_offset_hours,
            now_utc=now_utc,
            enforce_no_future_canonical_bar=True,
            max_future_canonical_bar_minutes=2,
        )
        rates_report = analyse_rates(
            canonical_rates,
            symbol=symbol_resolution.selected_symbol,
            timeframe_name=config.timeframe,
            timeframe_value=timeframe_value,
            requested_bars=config.bars_to_fetch,
            min_completed_bars=config.min_completed_bars,
            max_latest_closed_bar_age_minutes_warning=4320,
            allow_market_closed_stale_bar=True,
            now_utc=now_utc,
        )
        positions = proxy.positions_get(symbol=symbol_resolution.selected_symbol)
        orders = proxy.orders_get(symbol=symbol_resolution.selected_symbol)
        position_count = _safe_count(positions, label="positions_get")
        pending_order_count = _safe_count(orders, label="orders_get")
        expected_dir = _normalised_windows_path(terminal_exe.parent)
        reported_dir = _normalised_windows_path(str(terminal.get("path", "")))
        checks = {
            "terminal_executable_exists": True,
            "terminal_reported_path_matches_requested": reported_dir == expected_dir,
            "terminal_connected": terminal.get("connected") is True,
            "terminal_trade_allowed": terminal.get("trade_allowed") is True,
            "terminal_trade_api_enabled": terminal.get("tradeapi_disabled") is False,
            "account_is_demo": account.get("trade_mode_name") == "DEMO",
            "broker_company_matches": account.get("company") == config.broker_company_expected,
            "server_matches": account.get("server") == config.server_expected,
            "account_trade_allowed": account.get("trade_allowed") is True,
            "account_expert_allowed": account.get("trade_expert") is True,
            "symbol_matches": symbol_resolution.selected_symbol == config.symbol,
            "completed_m15_bars_used": rates_report.uses_completed_bars_only is True,
            "canonical_utc_conversion_applied": time_report.conversion_applied is True,
            "canonical_latest_bar_not_future": time_report.latest_bar_future_minutes_after_conversion <= 0,
            "no_open_symbol_positions": position_count == 0,
            "no_pending_symbol_orders": pending_order_count == 0,
            "runtime_root_present": bool(role.runtime_root),
            "magic_number_positive": role.magic_number > 0,
            "order_comment_present": bool(role.order_comment),
            "forbidden_trade_methods_not_called": not proxy.forbidden_attempts,
        }
        capital = _capital_review(account=account, symbol=symbol_info, config=config)
        snapshot = {
            "role": role.role,
            "terminal_executable": str(terminal_exe),
            "runtime_root": role.runtime_root,
            "magic_number": role.magic_number,
            "order_comment": role.order_comment,
            "package": package,
            "terminal": terminal,
            "account": account,
            "symbol_resolution": asdict(symbol_resolution),
            "tick": tick,
            "rates": asdict(rates_report),
            "time_normalisation": asdict(time_report),
            "position_count": position_count,
            "pending_order_count": pending_order_count,
            "capital_review": capital,
            "checks": checks,
            "read_only_api_methods_used": tuple(proxy.calls),
            "forbidden_trade_function_calls": tuple(proxy.forbidden_attempts),
            "order_check_called": False,
            "order_send_called": False,
        }
    except Exception as exc:
        primary_error = exc
    finally:
        if initialized:
            try:
                proxy.shutdown()
                shutdown_called = True
            except Exception:
                shutdown_called = False

    if primary_error is not None:
        if isinstance(primary_error, DualTerminalReadinessError):
            raise primary_error
        if isinstance(primary_error, (Mt5ReadinessError, ValueError, KeyError)):
            raise DualTerminalReadinessError(f"{role.role} readiness failed: {primary_error}") from primary_error
        raise primary_error
    if snapshot is None:
        raise DualTerminalReadinessError(f"{role.role} readiness produced no snapshot")
    snapshot["shutdown_called"] = shutdown_called
    snapshot["checks"]["shutdown_confirmed"] = shutdown_called
    snapshot["read_only_api_methods_used"] = tuple(proxy.calls)
    if not shutdown_called:
        raise DualTerminalReadinessError(f"{role.role} MT5 shutdown was not confirmed")
    return snapshot, raw_login


def _balance_difference_percent(a: float, b: float) -> float:
    denominator = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denominator * 100.0


def build_dual_terminal_report(
    *, mt5_module: Any, config: DualTerminalConfig, now_utc: datetime | None = None
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    path_a = _normalised_windows_path(config.model_a.terminal_path)
    path_b = _normalised_windows_path(config.model_b.terminal_path)
    if config.require_distinct_terminal_paths and path_a == path_b:
        raise DualTerminalReadinessError("Model A and Model B terminal paths must be different")
    if _normalised_windows_path(config.model_a.runtime_root) == _normalised_windows_path(
        config.model_b.runtime_root
    ):
        raise DualTerminalReadinessError("Model A and Model B runtime roots must be different")
    if config.model_a.magic_number == config.model_b.magic_number:
        raise DualTerminalReadinessError("Model A and Model B magic numbers must be different")
    if config.model_a.order_comment == config.model_b.order_comment:
        raise DualTerminalReadinessError("Model A and Model B order comments must be different")

    snapshot_a, login_a = inspect_terminal_role(
        mt5_module=mt5_module, role=config.model_a, config=config, now_utc=now_utc
    )
    snapshot_b, login_b = inspect_terminal_role(
        mt5_module=mt5_module, role=config.model_b, config=config, now_utc=now_utc
    )

    account_a = snapshot_a["account"]
    account_b = snapshot_b["account"]
    terminal_a = snapshot_a["terminal"]
    terminal_b = snapshot_b["terminal"]
    symbol_a = snapshot_a["symbol_resolution"]["symbol_info"]
    symbol_b = snapshot_b["symbol_resolution"]["symbol_info"]
    latest_a = pd.Timestamp(snapshot_a["time_normalisation"]["canonical_latest_bar_time_utc"])
    latest_b = pd.Timestamp(snapshot_b["time_normalisation"]["canonical_latest_bar_time_utc"])
    latest_diff_minutes = abs((latest_a - latest_b).total_seconds()) / 60.0
    balance_a = float(account_a.get("balance", 0.0) or 0.0)
    balance_b = float(account_b.get("balance", 0.0) or 0.0)
    balance_diff_pct = _balance_difference_percent(balance_a, balance_b)
    data_path_a = _normalised_windows_path(str(terminal_a.get("data_path", "")))
    data_path_b = _normalised_windows_path(str(terminal_b.get("data_path", "")))

    cross_checks = {
        "terminal_paths_distinct": path_a != path_b,
        "terminal_data_paths_distinct": bool(data_path_a) and bool(data_path_b) and data_path_a != data_path_b,
        "accounts_distinct": login_a is not None and login_b is not None and str(login_a) != str(login_b),
        "account_currency_matches": account_a.get("currency") == account_b.get("currency"),
        "account_margin_mode_matches": account_a.get("margin_mode") == account_b.get("margin_mode"),
        "broker_company_matches_between_roles": account_a.get("company") == account_b.get("company"),
        "server_matches_between_roles": account_a.get("server") == account_b.get("server"),
        "symbol_contract_matches": _contract_signature(symbol_a) == _contract_signature(symbol_b),
        "latest_completed_bar_difference_within_gate": latest_diff_minutes <= config.max_latest_bar_difference_minutes,
        "starting_balance_difference_within_gate": balance_diff_pct <= config.max_starting_balance_difference_percent,
        "model_a_minimum_equity_passed": snapshot_a["capital_review"]["minimum_equity_passed"] is True,
        "model_b_minimum_equity_passed": snapshot_b["capital_review"]["minimum_equity_passed"] is True,
        "model_a_leverage_cap_passed": snapshot_a["capital_review"]["capstone_leverage_cap_passed"] is True,
        "model_b_leverage_cap_passed": snapshot_b["capital_review"]["capstone_leverage_cap_passed"] is True,
        "magic_numbers_distinct": config.model_a.magic_number != config.model_b.magic_number,
        "order_comments_distinct": config.model_a.order_comment != config.model_b.order_comment,
        "runtime_roots_distinct": _normalised_windows_path(config.model_a.runtime_root)
        != _normalised_windows_path(config.model_b.runtime_root),
    }

    role_required = [
        "terminal_executable_exists", "terminal_reported_path_matches_requested", "terminal_connected",
        "account_is_demo", "broker_company_matches", "server_matches", "symbol_matches",
        "completed_m15_bars_used", "canonical_utc_conversion_applied",
        "canonical_latest_bar_not_future", "runtime_root_present", "magic_number_positive",
        "order_comment_present", "forbidden_trade_methods_not_called", "shutdown_confirmed",
    ]
    if config.require_terminal_trade_allowed:
        role_required.append("terminal_trade_allowed")
    if config.require_trade_api_enabled:
        role_required.append("terminal_trade_api_enabled")
    if config.require_account_trade_allowed:
        role_required.append("account_trade_allowed")
    if config.require_account_expert_allowed:
        role_required.append("account_expert_allowed")
    if config.require_no_open_symbol_positions:
        role_required.append("no_open_symbol_positions")
    if config.require_no_pending_symbol_orders:
        role_required.append("no_pending_symbol_orders")

    cross_required = [
        "broker_company_matches_between_roles", "server_matches_between_roles",
        "latest_completed_bar_difference_within_gate", "starting_balance_difference_within_gate",
        "model_a_minimum_equity_passed", "model_b_minimum_equity_passed",
        "model_a_leverage_cap_passed", "model_b_leverage_cap_passed",
        "magic_numbers_distinct", "order_comments_distinct", "runtime_roots_distinct",
    ]
    if config.require_distinct_terminal_paths:
        cross_required.append("terminal_paths_distinct")
    if config.require_distinct_data_paths:
        cross_required.append("terminal_data_paths_distinct")
    if config.require_distinct_accounts:
        cross_required.append("accounts_distinct")
    if config.require_same_currency:
        cross_required.append("account_currency_matches")
    if config.require_same_margin_mode:
        cross_required.append("account_margin_mode_matches")
    if config.require_same_symbol_contract:
        cross_required.append("symbol_contract_matches")

    role_a_passed = all(snapshot_a["checks"].get(k) is True for k in role_required)
    role_b_passed = all(snapshot_b["checks"].get(k) is True for k in role_required)
    cross_passed = all(cross_checks.get(k) is True for k in cross_required)
    formal_gate = role_a_passed and role_b_passed and cross_passed

    return {
        "stage": 3,
        "step": "4A",
        "status": "PASS" if formal_gate else "FAIL",
        "formal_gate": formal_gate,
        "patch_version": "stage3_step4a_v1_0",
        "purpose": "dual_account_dual_terminal_readiness_no_order",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "mt5_used": True,
        "orders_enabled": False,
        "order_check_called": False,
        "order_send_called": False,
        "inspection_mode": "sequential_exact_terminal_path",
        "model_a": snapshot_a,
        "model_b": snapshot_b,
        "cross_terminal_review": {
            "checks": cross_checks,
            "required_checks": cross_required,
            "latest_completed_bar_difference_minutes": latest_diff_minutes,
            "starting_balance_difference_percent": balance_diff_pct,
            "model_a_login_masked": account_a.get("login_masked"),
            "model_b_login_masked": account_b.get("login_masked"),
        },
        "gate_review": {
            "role_required_checks": role_required,
            "model_a_passed": role_a_passed,
            "model_b_passed": role_b_passed,
            "cross_terminal_passed": cross_passed,
        },
        "decision": {
            "stage3_step4a_passed": formal_gate,
            "final_14_day_run_authorised": False,
            "next_step_if_pass": "Stage 3 Step 4B - short dual-terminal shadow synchronisation test; no order_send.",
        },
    }


def summary_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role_key in ("model_a", "model_b"):
        role = report.get(role_key, {}) or {}
        for name, value in (role.get("checks", {}) or {}).items():
            rows.append({"scope": role_key, "check": name, "value": value})
        capital = role.get("capital_review", {}) or {}
        for name, value in capital.items():
            rows.append({"scope": f"{role_key}_capital", "check": name, "value": value})
    cross = report.get("cross_terminal_review", {}) or {}
    for name, value in (cross.get("checks", {}) or {}).items():
        rows.append({"scope": "cross_terminal", "check": name, "value": value})
    rows.extend(
        [
            {"scope": "gate", "check": "status", "value": report.get("status")},
            {"scope": "gate", "check": "formal_gate", "value": report.get("formal_gate")},
        ]
    )
    return rows


def terminal_inventory_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role_key in ("model_a", "model_b"):
        role = report.get(role_key, {}) or {}
        terminal = role.get("terminal", {}) or {}
        account = role.get("account", {}) or {}
        time_norm = role.get("time_normalisation", {}) or {}
        rows.append(
            {
                "role": role.get("role"),
                "terminal_executable": role.get("terminal_executable"),
                "terminal_data_path": terminal.get("data_path"),
                "terminal_connected": terminal.get("connected"),
                "terminal_trade_allowed": terminal.get("trade_allowed"),
                "tradeapi_disabled": terminal.get("tradeapi_disabled"),
                "login_masked": account.get("login_masked"),
                "company": account.get("company"),
                "server": account.get("server"),
                "currency": account.get("currency"),
                "balance": account.get("balance"),
                "equity": account.get("equity"),
                "margin_mode": account.get("margin_mode"),
                "position_count": role.get("position_count"),
                "pending_order_count": role.get("pending_order_count"),
                "latest_completed_bar_utc": time_norm.get("canonical_latest_bar_time_utc"),
                "runtime_root": role.get("runtime_root"),
                "magic_number": role.get("magic_number"),
                "order_comment": role.get("order_comment"),
            }
        )
    return rows
