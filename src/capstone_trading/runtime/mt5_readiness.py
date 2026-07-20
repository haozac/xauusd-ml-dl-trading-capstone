"""MT5 environment and data-feed readiness checks for Stage 2 Step 1.

This module is intentionally read-only.  It exposes only the subset of the
MetaTrader5 Python API needed for terminal/account/symbol/bar inspection and
keeps trade functions such as order_send unavailable through the safety proxy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import math
import numpy as np
import pandas as pd

FORBIDDEN_TRADE_FUNCTIONS: tuple[str, ...] = (
    "order_send",
    "order_check",
    "orders_get",
    "orders_total",
    "history_orders_get",
    "history_orders_total",
    "history_deals_get",
    "history_deals_total",
)

READ_ONLY_API_METHODS: tuple[str, ...] = (
    "initialize",
    "shutdown",
    "last_error",
    "version",
    "terminal_info",
    "account_info",
    "symbols_get",
    "symbol_info",
    "symbol_info_tick",
    "symbol_select",
    "copy_rates_from_pos",
)

M15_SECONDS = 15 * 60


class Mt5ReadinessError(RuntimeError):
    """Raised when the Stage 2 Step 1 readiness gate fails."""


@dataclass(frozen=True)
class Mt5RuntimeConfig:
    terminal_path: str | None = None
    symbol_candidates: tuple[str, ...] = (
        "XAUUSD",
        "XAUUSDm",
        "XAUUSD.",
        "XAUUSD.pro",
        "GOLD",
    )
    timeframe_name: str = "M15"
    bars_to_fetch: int = 200
    min_completed_bars: int = 120
    require_demo_account: bool = True
    allow_market_closed_stale_bar: bool = True
    max_latest_closed_bar_age_minutes_warning: int = 4320
    require_symbol_visible: bool = True


@dataclass(frozen=True)
class SymbolResolution:
    selected_symbol: str
    candidates_checked: tuple[str, ...]
    symbol_select_called: bool
    symbol_visible_after_select: bool
    symbol_info: Mapping[str, Any]


@dataclass(frozen=True)
class RatesReadinessReport:
    symbol: str
    timeframe_name: str
    timeframe_value: int
    start_pos: int
    requested_bars: int
    returned_bars: int
    uses_completed_bars_only: bool
    first_bar_time_utc: str | None
    latest_closed_bar_time_utc: str | None
    latest_closed_bar_age_minutes: float | None
    latest_closed_bar_stale_warning: bool
    duplicate_timestamps: int
    non_monotonic: bool
    invalid_ohlc_rows: int
    non_positive_price_rows: int
    negative_tick_volume_rows: int
    negative_spread_rows: int
    negative_real_volume_rows: int
    non_contiguous_gap_count: int
    maximum_gap_minutes: float | None
    passed: bool


@dataclass(frozen=True)
class Mt5ReadinessResult:
    status: str
    formal_gate: bool
    stage: int
    step: int
    offline_only_until_shadow: bool
    mt5_used: bool
    orders_enabled: bool
    terminal_initialized: bool
    shutdown_called: bool
    read_only_api_methods_used: tuple[str, ...]
    forbidden_trade_function_calls: tuple[str, ...]
    package: Mapping[str, Any]
    terminal: Mapping[str, Any]
    account: Mapping[str, Any]
    symbol_resolution: Mapping[str, Any]
    tick: Mapping[str, Any]
    rates: Mapping[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def load_mt5_runtime_config(path: Path) -> Mt5RuntimeConfig:
    """Load the Stage 2 runtime config from YAML or JSON.

    The config is intentionally broker/runtime-only.  It must not contain frozen
    research-model parameters or account passwords.
    """

    if not path.exists():
        raise Mt5ReadinessError(f"MT5 runtime config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        raw = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            raise Mt5ReadinessError(
                "PyYAML is required to read YAML runtime config files. "
                "Install pyyaml or provide a JSON config."
            ) from exc
        raw = yaml.safe_load(text)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise Mt5ReadinessError("MT5 runtime config must be a mapping/object")

    terminal = raw.get("terminal", {}) or {}
    symbol = raw.get("symbol", {}) or {}
    data = raw.get("data", {}) or {}
    safety = raw.get("safety", {}) or {}
    if not isinstance(terminal, Mapping) or not isinstance(symbol, Mapping):
        raise Mt5ReadinessError("Invalid MT5 runtime config structure")
    if not isinstance(data, Mapping) or not isinstance(safety, Mapping):
        raise Mt5ReadinessError("Invalid MT5 runtime config structure")

    candidates_raw = symbol.get("candidates", Mt5RuntimeConfig.symbol_candidates)
    if isinstance(candidates_raw, str):
        candidates = (candidates_raw,)
    else:
        candidates = tuple(str(item).strip() for item in candidates_raw if str(item).strip())
    if not candidates:
        raise Mt5ReadinessError("At least one MT5 symbol candidate is required")

    return Mt5RuntimeConfig(
        terminal_path=_none_if_blank(terminal.get("path")),
        symbol_candidates=candidates,
        timeframe_name=str(data.get("timeframe", "M15")).upper(),
        bars_to_fetch=int(data.get("bars_to_fetch", 200)),
        min_completed_bars=int(data.get("min_completed_bars", 120)),
        require_demo_account=bool(safety.get("require_demo_account", True)),
        allow_market_closed_stale_bar=bool(data.get("allow_market_closed_stale_bar", True)),
        max_latest_closed_bar_age_minutes_warning=int(
            data.get("max_latest_closed_bar_age_minutes_warning", 4320)
        ),
        require_symbol_visible=bool(symbol.get("require_visible", True)),
    )


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def object_to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert MetaTrader5 named tuples and simple objects to JSON-safe dicts."""

    if value is None:
        return {}
    if hasattr(value, "_asdict"):
        raw = dict(value._asdict())
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = {
            name: getattr(value, name)
            for name in dir(value)
            if not name.startswith("_") and not callable(getattr(value, name))
        }
    return {str(key): _json_safe(item) for key, item in raw.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    return str(value)


class SafeMt5Proxy:
    """Read-only wrapper around the MetaTrader5 module.

    Only documented read/inspection methods used by Stage 2 Step 1 are exposed.
    Accessing a forbidden trading/history method raises immediately and records
    the attempted method name.
    """

    def __init__(self, mt5_module: Any):
        self._mt5 = mt5_module
        self.calls: list[str] = []
        self.forbidden_attempts: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_TRADE_FUNCTIONS:
            self.forbidden_attempts.append(name)
            raise Mt5ReadinessError(f"Forbidden MT5 trade/history function accessed: {name}")
        if name in READ_ONLY_API_METHODS or name.startswith("TIMEFRAME_") or name.startswith(
            "ACCOUNT_TRADE_MODE_"
        ):
            attr = getattr(self._mt5, name)
            if callable(attr) and name in READ_ONLY_API_METHODS:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    self.calls.append(name)
                    return attr(*args, **kwargs)

                return wrapped
            return attr
        if name in {"__author__", "__version__"}:
            return getattr(self._mt5, name, None)
        raise AttributeError(name)


def import_metatrader5_module() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise Mt5ReadinessError(
            "MetaTrader5 Python package is not importable in this environment. "
            "Install it in the deployment venv before Stage 2 Step 1."
        ) from exc
    return mt5


def resolve_timeframe(mt5: Any, timeframe_name: str) -> int:
    name = timeframe_name.strip().upper()
    constant_name = f"TIMEFRAME_{name}"
    if not hasattr(mt5, constant_name):
        raise Mt5ReadinessError(f"MetaTrader5 module has no timeframe constant {constant_name}")
    value = int(getattr(mt5, constant_name))
    if name != "M15":
        raise Mt5ReadinessError(
            f"Stage 2 Step 1 currently expects M15 only; config requested {timeframe_name}"
        )
    return value


def initialise_terminal(mt5: SafeMt5Proxy, config: Mt5RuntimeConfig) -> None:
    kwargs: dict[str, Any] = {}
    if config.terminal_path is not None:
        ok = bool(mt5.initialize(config.terminal_path, **kwargs))
    else:
        ok = bool(mt5.initialize(**kwargs))
    if not ok:
        raise Mt5ReadinessError(f"mt5.initialize() failed: {safe_last_error(mt5)}")


def safe_last_error(mt5: Any) -> Any:
    try:
        return mt5.last_error()
    except Exception:
        return None


def build_package_snapshot(mt5: Any) -> dict[str, Any]:
    version_value: Any = None
    try:
        version_value = mt5.version()
    except Exception:
        version_value = None
    return {
        "module_author": getattr(mt5, "__author__", None),
        "module_version": getattr(mt5, "__version__", None),
        "terminal_version": _json_safe(version_value),
    }


def inspect_terminal(mt5: Any) -> dict[str, Any]:
    info = mt5.terminal_info()
    if info is None:
        raise Mt5ReadinessError(f"mt5.terminal_info() returned None: {safe_last_error(mt5)}")
    snapshot = object_to_plain_dict(info)
    connected = snapshot.get("connected")
    if connected is False:
        raise Mt5ReadinessError("MT5 terminal_info.connected is False")
    return snapshot


def inspect_account(mt5: Any, *, require_demo: bool) -> dict[str, Any]:
    info = mt5.account_info()
    if info is None:
        raise Mt5ReadinessError(f"mt5.account_info() returned None: {safe_last_error(mt5)}")
    snapshot = object_to_plain_dict(info)
    trade_mode = snapshot.get("trade_mode")
    trade_mode_name = trade_mode_to_name(mt5, trade_mode)
    snapshot["trade_mode_name"] = trade_mode_name
    snapshot["login_masked"] = mask_login(snapshot.get("login"))
    snapshot.pop("login", None)
    for sensitive in ("balance", "equity", "margin", "margin_free", "profit"):
        snapshot.pop(sensitive, None)
    if require_demo and trade_mode_name != "DEMO":
        raise Mt5ReadinessError(
            f"Connected account is not DEMO. Detected trade_mode={trade_mode!r} ({trade_mode_name})."
        )
    return snapshot


def trade_mode_to_name(mt5: Any, trade_mode: Any) -> str:
    if trade_mode is None:
        return "UNKNOWN"
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


def mask_login(login: Any) -> str | None:
    if login is None:
        return None
    text = str(login)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{'*' * (len(text) - 4)}{text[-4:]}"


def resolve_symbol(mt5: Any, config: Mt5RuntimeConfig) -> SymbolResolution:
    checked: list[str] = []
    for candidate in config.symbol_candidates:
        checked.append(candidate)
        info = mt5.symbol_info(candidate)
        if info is None:
            continue
        info_dict = object_to_plain_dict(info)
        visible_before = bool(info_dict.get("visible", False))
        select_called = False
        if config.require_symbol_visible and not visible_before:
            select_called = True
            if not bool(mt5.symbol_select(candidate, True)):
                continue
            info = mt5.symbol_info(candidate)
            if info is None:
                continue
            info_dict = object_to_plain_dict(info)
        visible_after = bool(info_dict.get("visible", visible_before))
        validate_symbol_info(candidate, info_dict)
        return SymbolResolution(
            selected_symbol=candidate,
            candidates_checked=tuple(checked),
            symbol_select_called=select_called,
            symbol_visible_after_select=visible_after,
            symbol_info=info_dict,
        )
    raise Mt5ReadinessError(
        "Unable to resolve an MT5 XAUUSD/GOLD symbol from candidates: "
        + ", ".join(config.symbol_candidates)
    )


def validate_symbol_info(symbol: str, info: Mapping[str, Any]) -> None:
    digits = _float_or_none(info.get("digits"))
    point = _float_or_none(info.get("point"))
    volume_min = _float_or_none(info.get("volume_min"))
    volume_max = _float_or_none(info.get("volume_max"))
    volume_step = _float_or_none(info.get("volume_step"))
    if digits is not None and digits < 0:
        raise Mt5ReadinessError(f"Symbol {symbol} has invalid digits={digits}")
    if point is not None and point <= 0:
        raise Mt5ReadinessError(f"Symbol {symbol} has non-positive point={point}")
    if volume_min is not None and volume_min <= 0:
        raise Mt5ReadinessError(f"Symbol {symbol} has non-positive volume_min={volume_min}")
    if volume_max is not None and volume_min is not None and volume_max < volume_min:
        raise Mt5ReadinessError(f"Symbol {symbol} has volume_max < volume_min")
    if volume_step is not None and volume_step <= 0:
        raise Mt5ReadinessError(f"Symbol {symbol} has non-positive volume_step={volume_step}")


def inspect_tick(mt5: Any, symbol: str) -> dict[str, Any]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"available": False, "last_error": _json_safe(safe_last_error(mt5))}
    snapshot = object_to_plain_dict(tick)
    snapshot["available"] = True
    for key in ("time", "time_msc"):
        if key in snapshot and isinstance(snapshot[key], (int, float)):
            try:
                divisor = 1000.0 if key.endswith("msc") else 1.0
                snapshot[f"{key}_utc"] = datetime.fromtimestamp(
                    float(snapshot[key]) / divisor,
                    tz=timezone.utc,
                ).isoformat()
            except Exception:
                pass
    return snapshot


def fetch_completed_m15_rates(
    mt5: Any,
    *,
    symbol: str,
    timeframe_value: int,
    count: int,
) -> pd.DataFrame:
    # start_pos=1 intentionally excludes the still-forming zero bar.  This is
    # critical for completed-bar-only deployment semantics.
    rates = mt5.copy_rates_from_pos(symbol, timeframe_value, 1, count)
    if rates is None:
        raise Mt5ReadinessError(f"copy_rates_from_pos() returned None: {safe_last_error(mt5)}")
    frame = pd.DataFrame(rates)
    if frame.empty:
        raise Mt5ReadinessError("copy_rates_from_pos() returned an empty rate table")
    required = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise Mt5ReadinessError(f"MT5 rates missing columns: {missing}")
    frame = frame.loc[:, required].copy()
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.sort_values("time").reset_index(drop=True)
    return frame


def analyse_rates(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe_name: str,
    timeframe_value: int,
    requested_bars: int,
    min_completed_bars: int,
    max_latest_closed_bar_age_minutes_warning: int,
    allow_market_closed_stale_bar: bool,
    now_utc: datetime | None = None,
) -> RatesReadinessReport:
    if len(frame) < min_completed_bars:
        raise Mt5ReadinessError(
            f"Only {len(frame)} completed M15 bars returned; minimum required is {min_completed_bars}."
        )
    now = pd.Timestamp(now_utc or datetime.now(timezone.utc)).tz_convert("UTC")
    duplicate_count = int(frame["time"].duplicated().sum())
    non_monotonic = not bool(frame["time"].is_monotonic_increasing)
    open_ = pd.to_numeric(frame["open"], errors="raise")
    high = pd.to_numeric(frame["high"], errors="raise")
    low = pd.to_numeric(frame["low"], errors="raise")
    close = pd.to_numeric(frame["close"], errors="raise")
    tick_volume = pd.to_numeric(frame["tick_volume"], errors="raise")
    spread = pd.to_numeric(frame["spread"], errors="raise")
    real_volume = pd.to_numeric(frame["real_volume"], errors="raise")

    invalid_ohlc = (high < pd.concat([open_, close], axis=1).max(axis=1)) | (
        low > pd.concat([open_, close], axis=1).min(axis=1)
    )
    non_positive_price = (open_ <= 0) | (high <= 0) | (low <= 0) | (close <= 0)
    negative_tick_volume = tick_volume < 0
    negative_spread = spread < 0
    negative_real_volume = real_volume < 0
    gaps_seconds = frame["time"].diff().dt.total_seconds().dropna()
    non_contiguous_gap_count = int((gaps_seconds != M15_SECONDS).sum())
    maximum_gap_minutes = float(gaps_seconds.max() / 60.0) if not gaps_seconds.empty else None
    latest_time = pd.Timestamp(frame["time"].iloc[-1]).tz_convert("UTC")
    age_minutes = float((now - latest_time).total_seconds() / 60.0)
    stale_warning = age_minutes > max_latest_closed_bar_age_minutes_warning

    hard_failures = {
        "duplicate_timestamps": duplicate_count,
        "non_monotonic": int(non_monotonic),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "non_positive_price_rows": int(non_positive_price.sum()),
        "negative_tick_volume_rows": int(negative_tick_volume.sum()),
        "negative_spread_rows": int(negative_spread.sum()),
        "negative_real_volume_rows": int(negative_real_volume.sum()),
    }
    failed = {key: value for key, value in hard_failures.items() if value}
    if failed:
        raise Mt5ReadinessError(f"MT5 completed-bar sanity checks failed: {failed}")
    if stale_warning and not allow_market_closed_stale_bar:
        raise Mt5ReadinessError(
            f"Latest completed M15 bar is stale by {age_minutes:.1f} minutes and stale bars are not allowed"
        )

    return RatesReadinessReport(
        symbol=symbol,
        timeframe_name=timeframe_name,
        timeframe_value=int(timeframe_value),
        start_pos=1,
        requested_bars=int(requested_bars),
        returned_bars=int(len(frame)),
        uses_completed_bars_only=True,
        first_bar_time_utc=pd.Timestamp(frame["time"].iloc[0]).isoformat() if len(frame) else None,
        latest_closed_bar_time_utc=latest_time.isoformat(),
        latest_closed_bar_age_minutes=age_minutes,
        latest_closed_bar_stale_warning=bool(stale_warning),
        duplicate_timestamps=duplicate_count,
        non_monotonic=bool(non_monotonic),
        invalid_ohlc_rows=int(invalid_ohlc.sum()),
        non_positive_price_rows=int(non_positive_price.sum()),
        negative_tick_volume_rows=int(negative_tick_volume.sum()),
        negative_spread_rows=int(negative_spread.sum()),
        negative_real_volume_rows=int(negative_real_volume.sum()),
        non_contiguous_gap_count=non_contiguous_gap_count,
        maximum_gap_minutes=maximum_gap_minutes,
        passed=True,
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def rates_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out


def run_mt5_readiness_check(
    *,
    mt5_module: Any,
    config: Mt5RuntimeConfig,
    now_utc: datetime | None = None,
) -> tuple[Mt5ReadinessResult, pd.DataFrame]:
    proxy = SafeMt5Proxy(mt5_module)
    initialized = False
    shutdown_called = False
    warnings: list[str] = []
    result: Mt5ReadinessResult | None = None
    rates_frame: pd.DataFrame | None = None
    try:
        timeframe_value = resolve_timeframe(proxy, config.timeframe_name)
        initialise_terminal(proxy, config)
        initialized = True
        package = build_package_snapshot(proxy)
        terminal = inspect_terminal(proxy)
        account = inspect_account(proxy, require_demo=config.require_demo_account)
        symbol_resolution = resolve_symbol(proxy, config)
        tick = inspect_tick(proxy, symbol_resolution.selected_symbol)
        rates_frame = fetch_completed_m15_rates(
            proxy,
            symbol=symbol_resolution.selected_symbol,
            timeframe_value=timeframe_value,
            count=config.bars_to_fetch,
        )
        rates_report = analyse_rates(
            rates_frame,
            symbol=symbol_resolution.selected_symbol,
            timeframe_name=config.timeframe_name,
            timeframe_value=timeframe_value,
            requested_bars=config.bars_to_fetch,
            min_completed_bars=config.min_completed_bars,
            max_latest_closed_bar_age_minutes_warning=config.max_latest_closed_bar_age_minutes_warning,
            allow_market_closed_stale_bar=config.allow_market_closed_stale_bar,
            now_utc=now_utc,
        )
        if rates_report.latest_closed_bar_stale_warning:
            warnings.append(
                "Latest completed M15 bar is older than configured warning threshold; this can be normal "
                "while the market is closed, but should be checked before shadow mode."
            )
        result = Mt5ReadinessResult(
            status="PASS",
            formal_gate=True,
            stage=2,
            step=1,
            offline_only_until_shadow=True,
            mt5_used=True,
            orders_enabled=False,
            terminal_initialized=initialized,
            shutdown_called=False,
            read_only_api_methods_used=tuple(proxy.calls),
            forbidden_trade_function_calls=tuple(proxy.forbidden_attempts),
            package=package,
            terminal=terminal,
            account=account,
            symbol_resolution=asdict(symbol_resolution),
            tick=tick,
            rates=asdict(rates_report),
            warnings=tuple(warnings),
        )
    finally:
        if initialized:
            try:
                proxy.shutdown()
                shutdown_called = True
            except Exception:
                shutdown_called = False
    if result is None or rates_frame is None:
        raise Mt5ReadinessError("MT5 readiness did not produce a result")
    result = mark_shutdown(result, shutdown_called=shutdown_called)
    if proxy.forbidden_attempts:
        raise Mt5ReadinessError(
            f"Forbidden MT5 API methods were accessed: {proxy.forbidden_attempts}"
        )
    return result, rates_frame

def mark_shutdown(result: Mt5ReadinessResult, *, shutdown_called: bool) -> Mt5ReadinessResult:
    payload = asdict(result)
    payload["shutdown_called"] = bool(shutdown_called)
    return Mt5ReadinessResult(**payload)


def result_to_dict(result: Mt5ReadinessResult) -> dict[str, Any]:
    return asdict(result)
