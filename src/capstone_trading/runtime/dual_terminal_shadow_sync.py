"""Stage 3 Step 4B concurrent dual-terminal shadow synchronisation gate.

The gate launches one Python worker per explicitly configured MT5 terminal.
Each worker loads the same frozen CNN-LSTM bundle, reconstructs completed-M15
features independently from its assigned Dukascopy terminal, and records
shadow-only observations.  The parent process then verifies that both workers
observe the same completed bars, feature state, scaled 48x51 sequence, model
probability, and frozen overlay signals.

Safety design
-------------
* One operating-system process is used per MT5 terminal because the MetaTrader5
  Python package maintains process-global terminal state.
* order_check and order_send are not exposed in this gate.
* order_calc_margin and order_calc_profit are permitted only in a narrow,
  calculation-only proxy to validate broker economic metadata.  They cannot
  place, modify, or close an order.
* Each role writes to a run-specific directory below its frozen runtime root.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("stage3_step4b_sync")


class DualTerminalShadowSyncError(RuntimeError):
    """Raised when Step 4B cannot establish a safe synchronised shadow run."""


DEFAULT_PROBABILITY_TOLERANCE = 5e-7
DEFAULT_ECONOMIC_ABSOLUTE_TOLERANCE = 0.05
DEFAULT_ECONOMIC_RELATIVE_TOLERANCE = 1e-6
DEFAULT_PRICE_MOVE_USD = 1.0

FORBIDDEN_TRADE_METHODS: tuple[str, ...] = (
    "order_check",
    "order_send",
    "history_orders_get",
    "history_deals_get",
)


@dataclass(frozen=True)
class WorkerPaths:
    role_root: Path
    events_csv: Path
    latest_json: Path
    status_json: Path
    state_json: Path
    shadow_signals_csv: Path
    worker_stdout_log: Path
    worker_stderr_log: Path


@dataclass(frozen=True)
class SyncComparison:
    event_time_utc: str
    model_a_probability_up: float
    model_b_probability_up: float
    probability_absolute_difference: float
    probability_tolerance: float
    rates_digest_matches: bool
    feature_digest_matches: bool
    sequence_digest_matches: bool
    model_a_signal_matches: bool
    model_b_signal_matches: bool
    model_b_entry_condition_matches: bool
    model_b_hold_condition_matches: bool
    model_a_role_signal: int
    model_b_role_signal: int
    both_roles_flat: bool
    both_roles_no_pending_orders: bool
    both_roles_no_forbidden_calls: bool
    passed: bool
    failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EconomicComparison:
    buy_margin_absolute_difference: float
    sell_margin_absolute_difference: float
    buy_profit_absolute_difference: float
    sell_profit_absolute_difference: float
    absolute_tolerance: float
    relative_tolerance: float
    model_a_calculations_valid: bool
    model_b_calculations_valid: bool
    values_match: bool
    passed: bool
    failures: tuple[str, ...] = field(default_factory=tuple)


class CalculationOnlyMt5Proxy:
    """Narrow MT5 proxy for no-order broker economic calculations.

    order_calc_margin and order_calc_profit are pure broker calculations.  The
    execution functions order_check and order_send are intentionally blocked.
    """

    allowed_methods = {
        "initialize",
        "shutdown",
        "last_error",
        "terminal_info",
        "account_info",
        "symbol_info",
        "symbol_info_tick",
        "positions_get",
        "orders_get",
        "order_calc_margin",
        "order_calc_profit",
    }
    forbidden_methods = set(FORBIDDEN_TRADE_METHODS)

    def __init__(self, mt5_module: Any):
        self._mt5 = mt5_module
        self.calls: list[str] = []
        self.forbidden_attempts: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name in self.forbidden_methods:
            self.forbidden_attempts.append(name)
            raise DualTerminalShadowSyncError(
                f"Forbidden MT5 method accessed in Step 4B calculation session: {name}"
            )
        if name in self.allowed_methods:
            attr = getattr(self._mt5, name)
            if callable(attr):
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    self.calls.append(name)
                    return attr(*args, **kwargs)

                return wrapped
            return attr
        if name.startswith(("ORDER_", "ACCOUNT_", "SYMBOL_", "TRADE_")):
            return getattr(self._mt5, name)
        if name in {"__author__", "__version__"}:
            return getattr(self._mt5, name, None)
        raise AttributeError(name)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalised_path(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).expanduser().resolve())))


def safe_repo_path(
    repo_root: Path,
    raw_path: str | Path,
    *,
    description: str,
    must_exist: bool = False,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise DualTerminalShadowSyncError(
            f"{description} must stay inside the repository: {resolved}"
        ) from exc
    if must_exist and not resolved.exists():
        raise DualTerminalShadowSyncError(f"{description} does not exist: {resolved}")
    return resolved


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str]
    if rows:
        fields = list(rows[0].keys())
    else:
        fields = ["status"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
    tmp.replace(path)


def append_csv_row(path: Path, row: Mapping[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DualTerminalShadowSyncError(f"Unable to read JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DualTerminalShadowSyncError(f"Expected JSON object in {path}")
    return raw


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception as exc:
        raise DualTerminalShadowSyncError(f"Unable to read CSV {path}: {exc}") from exc


def mask_login(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    return "*" * (len(text) - 4) + text[-4:]


def sha256_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _canonical_float_bytes(values: np.ndarray) -> bytes:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    if not np.isfinite(array).all():
        raise DualTerminalShadowSyncError("Cannot hash non-finite numeric values")
    return array.tobytes(order="C")


def digest_numpy(values: np.ndarray, *, prefix: bytes = b"") -> str:
    digest = hashlib.sha256()
    digest.update(prefix)
    digest.update(_canonical_float_bytes(values))
    return digest.hexdigest()


def digest_feature_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        raise DualTerminalShadowSyncError("Cannot hash an empty feature frame")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(index.asi8.astype("<i8")).tobytes())
    digest.update("\x1f".join(str(col) for col in frame.columns).encode("utf-8"))
    digest.update(_canonical_float_bytes(frame.to_numpy(dtype=np.float64, copy=False)))
    return digest.hexdigest()


def digest_completed_rates(rates: pd.DataFrame) -> str:
    required = ["time", "open", "high", "low", "close", "tick_volume"]
    missing = [column for column in required if column not in rates.columns]
    if missing:
        raise DualTerminalShadowSyncError(f"Completed-rate digest missing columns: {missing}")
    subset = rates.loc[:, required].copy()
    times = pd.to_datetime(subset.pop("time"), utc=True)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(times.astype("int64").to_numpy(dtype="<i8")).tobytes())
    digest.update(_canonical_float_bytes(subset.to_numpy(dtype=np.float64, copy=False)))
    return digest.hexdigest()


def latest_feature_and_sequence_digests(
    *,
    rates: pd.DataFrame,
    feature_order: Sequence[str],
    scaler: Any,
    sequence_length: int,
) -> dict[str, Any]:
    """Rebuild and fingerprint the latest live feature state and scaled sequence."""

    from capstone_trading.data.sequences import (
        scale_feature_frame,
        valid_sequence_positions,
        validate_plan_contiguity,
    )
    from capstone_trading.runtime.mt5_shadow import (
        build_live_feature_frame,
        convert_mt5_rates_to_feature_bars,
    )

    bars = convert_mt5_rates_to_feature_bars(rates)
    feature_frame, feature_report = build_live_feature_frame(bars, feature_order)
    scaled = scale_feature_frame(scaler, feature_frame, feature_order)
    plan = valid_sequence_positions(pd.DatetimeIndex(feature_frame.index), sequence_length)
    validate_plan_contiguity(pd.DatetimeIndex(feature_frame.index), plan)
    if plan.sequence_count < 1:
        raise DualTerminalShadowSyncError(
            f"No contiguous {sequence_length}-bar sequence available for fingerprinting"
        )
    start = int(plan.starts[-1])
    end = int(plan.ends[-1])
    sequence = np.asarray(scaled[start : end + 1], dtype=np.float64)
    expected_shape = (int(sequence_length), len(feature_order))
    if sequence.shape != expected_shape:
        raise DualTerminalShadowSyncError(
            f"Latest sequence shape mismatch: expected {expected_shape}, found {sequence.shape}"
        )
    event_time = pd.Timestamp(feature_frame.index[end]).tz_convert("UTC").isoformat()
    return {
        "rates_digest": digest_completed_rates(rates),
        "feature_digest": digest_feature_frame(feature_frame),
        "sequence_digest": digest_numpy(sequence, prefix=b"scaled_sequence_v1"),
        "event_time_utc": event_time,
        "feature_rows": int(len(feature_frame)),
        "feature_count": int(len(feature_order)),
        "sequence_length": int(sequence_length),
        "valid_sequence_count": int(plan.sequence_count),
        "feature_report": asdict(feature_report),
    }


def _call_order_calc_margin_compat(
    proxy: CalculationOnlyMt5Proxy,
    *,
    action: int,
    symbol: str,
    volume: float,
    price: float,
) -> float:
    """Call MT5 margin calculation using its required unnamed parameters.

    The MetaTrader5 Python API documents all four parameters as required
    unnamed parameters.  Passing keyword arguments can return an invalid zero
    result with some terminal/package combinations instead of raising a
    TypeError, so Step 4B always uses the documented positional form.
    """

    value = proxy.order_calc_margin(action, symbol, volume, price)
    last_error = proxy.last_error()
    if value is None:
        raise DualTerminalShadowSyncError(
            f"order_calc_margin returned None using positional arguments: {last_error}"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise DualTerminalShadowSyncError(
            "Invalid calculated margin using positional arguments: "
            f"value={result}, last_error={last_error}"
        )
    return result


def _call_order_calc_profit_compat(
    proxy: CalculationOnlyMt5Proxy,
    *,
    action: int,
    symbol: str,
    volume: float,
    price_open: float,
    price_close: float,
) -> float:
    """Call MT5 profit calculation using its required unnamed parameters."""

    value = proxy.order_calc_profit(action, symbol, volume, price_open, price_close)
    last_error = proxy.last_error()
    if value is None:
        raise DualTerminalShadowSyncError(
            f"order_calc_profit returned None using positional arguments: {last_error}"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise DualTerminalShadowSyncError(
            "Invalid calculated positive profit using positional arguments: "
            f"value={result}, last_error={last_error}"
        )
    return result


def inspect_flat_state_and_economics(
    *,
    mt5_module: Any,
    terminal_path: str,
    expected_terminal_directory: str,
    expected_login_masked: str | None,
    symbol: str,
    volume: float,
    price_move_usd: float = DEFAULT_PRICE_MOVE_USD,
) -> dict[str, Any]:
    """Inspect flat broker state and run pure economic calculations.

    No order request is checked or sent.  The calculation session also verifies
    that the exact terminal path and expected account remain attached.
    """

    from capstone_trading.runtime.mt5_readiness import object_to_plain_dict, safe_last_error

    proxy = CalculationOnlyMt5Proxy(mt5_module)
    initialized = False
    try:
        if not bool(proxy.initialize(str(terminal_path))):
            raise DualTerminalShadowSyncError(
                f"mt5.initialize failed for calculation session: {safe_last_error(proxy)}"
            )
        initialized = True
        terminal_info = proxy.terminal_info()
        account_info = proxy.account_info()
        symbol_info = proxy.symbol_info(symbol)
        tick_info = proxy.symbol_info_tick(symbol)
        if terminal_info is None or account_info is None or symbol_info is None or tick_info is None:
            raise DualTerminalShadowSyncError(
                f"Calculation session received incomplete MT5 metadata: {safe_last_error(proxy)}"
            )
        terminal = object_to_plain_dict(terminal_info)
        account = object_to_plain_dict(account_info)
        symbol_snapshot = object_to_plain_dict(symbol_info)
        tick = object_to_plain_dict(tick_info)
        reported_dir = normalised_path(str(terminal.get("path", "")))
        expected_dir = normalised_path(expected_terminal_directory)
        if reported_dir != expected_dir:
            raise DualTerminalShadowSyncError(
                f"Calculation session attached to wrong terminal. Expected {expected_dir}, got {reported_dir}"
            )
        login_masked = mask_login(account.get("login"))
        if expected_login_masked and login_masked != expected_login_masked:
            raise DualTerminalShadowSyncError(
                f"Calculation session attached to wrong account. Expected {expected_login_masked}, got {login_masked}"
            )
        positions = proxy.positions_get(symbol=symbol)
        orders = proxy.orders_get(symbol=symbol)
        if positions is None or orders is None:
            raise DualTerminalShadowSyncError(
                f"positions_get/orders_get failed: {safe_last_error(proxy)}"
            )
        position_count = len(tuple(positions))
        pending_order_count = len(tuple(orders))
        if position_count != 0 or pending_order_count != 0:
            raise DualTerminalShadowSyncError(
                f"Step 4B requires a flat account. positions={position_count}, pending_orders={pending_order_count}"
            )
        ask = float(tick.get("ask", 0.0) or 0.0)
        bid = float(tick.get("bid", 0.0) or 0.0)
        if ask <= 0 or bid <= 0 or ask < bid:
            raise DualTerminalShadowSyncError(f"Invalid calculation tick: bid={bid}, ask={ask}")
        buy_type = int(getattr(proxy, "ORDER_TYPE_BUY"))
        sell_type = int(getattr(proxy, "ORDER_TYPE_SELL"))
        buy_margin = _call_order_calc_margin_compat(
            proxy, action=buy_type, symbol=symbol, volume=volume, price=ask
        )
        sell_margin = _call_order_calc_margin_compat(
            proxy, action=sell_type, symbol=symbol, volume=volume, price=bid
        )
        buy_profit = _call_order_calc_profit_compat(
            proxy,
            action=buy_type,
            symbol=symbol,
            volume=volume,
            price_open=ask,
            price_close=ask + float(price_move_usd),
        )
        sell_profit = _call_order_calc_profit_compat(
            proxy,
            action=sell_type,
            symbol=symbol,
            volume=volume,
            price_open=bid,
            price_close=bid - float(price_move_usd),
        )
        result = {
            "status": "PASS",
            "terminal_path": str(terminal_path),
            "terminal_reported_path": terminal.get("path"),
            "login_masked": login_masked,
            "symbol": symbol,
            "volume_lots": float(volume),
            "price_move_usd": float(price_move_usd),
            "calculation_call_mode": "documented_required_unnamed_positional_parameters",
            "bid": bid,
            "ask": ask,
            "spread_points": int(symbol_snapshot.get("spread", -1) or -1),
            "position_count": position_count,
            "pending_order_count": pending_order_count,
            "buy_margin_account_currency": buy_margin,
            "sell_margin_account_currency": sell_margin,
            "buy_profit_for_positive_price_move_account_currency": buy_profit,
            "sell_profit_for_positive_price_move_account_currency": sell_profit,
            "account_currency": account.get("currency"),
            "trade_tick_value_metadata": symbol_snapshot.get("trade_tick_value"),
            "trade_tick_value_profit_metadata": symbol_snapshot.get("trade_tick_value_profit"),
            "trade_tick_value_loss_metadata": symbol_snapshot.get("trade_tick_value_loss"),
            "api_calls": tuple(proxy.calls),
            "forbidden_trade_function_calls": tuple(proxy.forbidden_attempts),
            "order_check_called": "order_check" in proxy.calls,
            "order_send_called": "order_send" in proxy.calls,
        }
        if result["forbidden_trade_function_calls"]:
            raise DualTerminalShadowSyncError(
                f"Forbidden calls recorded in calculation session: {result['forbidden_trade_function_calls']}"
            )
        return result
    finally:
        if initialized:
            try:
                proxy.shutdown()
            except Exception:
                pass


def observation_fieldnames() -> list[str]:
    return [
        "run_id",
        "role",
        "poll_iteration",
        "observed_utc",
        "event_time_utc",
        "latest_completed_bar_time_utc",
        "probability_up",
        "model_a_signal",
        "model_a_signal_name",
        "model_b_from_flat_signal",
        "model_b_from_flat_signal_name",
        "model_b_entry_condition",
        "model_b_hold_condition",
        "role_shadow_signal",
        "role_shadow_signal_name",
        "duplicate_event",
        "stale_event_warning",
        "spread_points",
        "actual_position_count",
        "pending_order_count",
        "feature_rows",
        "feature_count",
        "sequence_length",
        "valid_sequence_count",
        "rates_digest",
        "feature_digest",
        "sequence_digest",
        "terminal_executable",
        "terminal_reported_path",
        "terminal_build",
        "terminal_data_path",
        "login_masked",
        "forbidden_trade_function_calls",
        "order_check_called",
        "order_send_called",
    ]


def _role_signal(signal: Mapping[str, Any], role: str) -> tuple[int, str]:
    if role == "MODEL_A":
        return int(signal.get("model_a_signal", 0)), str(signal.get("model_a_signal_name", "FLAT"))
    if role == "MODEL_B":
        return int(signal.get("model_b_from_flat_signal", 0)), str(
            signal.get("model_b_from_flat_signal_name", "FLAT")
        )
    raise DualTerminalShadowSyncError(f"Unknown worker role: {role}")


def build_worker_paths(role_runtime_root: Path, run_id: str) -> WorkerPaths:
    role_root = role_runtime_root / "stage3_step4b" / run_id
    return WorkerPaths(
        role_root=role_root,
        events_csv=role_root / "observations.csv",
        latest_json=role_root / "latest_observation.json",
        status_json=role_root / "worker_status.json",
        state_json=role_root / "shadow_state.json",
        shadow_signals_csv=role_root / "shadow_signals.csv",
        worker_stdout_log=role_root / "worker_stdout.log",
        worker_stderr_log=role_root / "worker_stderr.log",
    )


def load_and_validate_step4a_report(
    *,
    report_path: Path,
    model_a_terminal_path: str,
    model_b_terminal_path: str,
) -> dict[str, Any]:
    report = read_json(report_path)
    if report.get("status") != "PASS" or report.get("formal_gate") is not True:
        raise DualTerminalShadowSyncError("Stage 3 Step 4A report is not a formal PASS")
    if report.get("order_check_called") is not False or report.get("order_send_called") is not False:
        raise DualTerminalShadowSyncError("Stage 3 Step 4A report recorded a forbidden order call")
    report_a = str((report.get("model_a", {}) or {}).get("terminal_executable", ""))
    report_b = str((report.get("model_b", {}) or {}).get("terminal_executable", ""))
    if normalised_path(report_a) != normalised_path(model_a_terminal_path):
        raise DualTerminalShadowSyncError(
            "Step 4A Model A terminal path does not match the current dual-terminal configuration"
        )
    if normalised_path(report_b) != normalised_path(model_b_terminal_path):
        raise DualTerminalShadowSyncError(
            "Step 4A Model B terminal path does not match the current dual-terminal configuration"
        )
    role_a = report.get("model_a", {}) or {}
    role_b = report.get("model_b", {}) or {}
    package_a = role_a.get("package", {}) or {}
    package_b = role_b.get("package", {}) or {}
    terminal_a = role_a.get("terminal", {}) or {}
    terminal_b = role_b.get("terminal", {}) or {}
    if package_a.get("terminal_version") != package_b.get("terminal_version"):
        raise DualTerminalShadowSyncError("Step 4A terminals do not use the same MT5 build")
    if terminal_a.get("name") != terminal_b.get("name"):
        raise DualTerminalShadowSyncError("Step 4A terminals do not use the same MT5 distribution")
    cross = report.get("cross_terminal_review", {}) or {}
    checks = cross.get("checks", {}) or {}
    required = (
        "terminal_paths_distinct",
        "terminal_data_paths_distinct",
        "accounts_distinct",
        "symbol_contract_matches",
        "latest_completed_bar_difference_within_gate",
    )
    if not all(checks.get(key) is True for key in required):
        raise DualTerminalShadowSyncError(
            f"Step 4A prerequisite checks are incomplete: { {key: checks.get(key) for key in required} }"
        )
    return report


def _fresh_event_map(rows: Iterable[Mapping[str, Any]], *, run_id: str) -> dict[str, Mapping[str, Any]]:
    events: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("run_id")) != run_id:
            continue
        event_time = str(row.get("event_time_utc", "")).strip()
        if not event_time:
            continue
        if _to_bool(row.get("duplicate_event")) or _to_bool(row.get("stale_event_warning")):
            continue
        events[event_time] = row
    return events


def compare_synchronised_event(
    row_a: Mapping[str, Any],
    row_b: Mapping[str, Any],
    *,
    probability_tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
) -> SyncComparison:
    event_a = str(row_a.get("event_time_utc", ""))
    event_b = str(row_b.get("event_time_utc", ""))
    failures: list[str] = []
    if event_a != event_b:
        failures.append("event_time_mismatch")
    p_a = _to_float(row_a.get("probability_up"))
    p_b = _to_float(row_b.get("probability_up"))
    p_diff = abs(p_a - p_b) if math.isfinite(p_a) and math.isfinite(p_b) else math.inf
    if p_diff > probability_tolerance:
        failures.append("probability_mismatch")
    rates_match = str(row_a.get("rates_digest")) == str(row_b.get("rates_digest"))
    features_match = str(row_a.get("feature_digest")) == str(row_b.get("feature_digest"))
    sequence_match = str(row_a.get("sequence_digest")) == str(row_b.get("sequence_digest"))
    if not rates_match:
        failures.append("rates_digest_mismatch")
    if not features_match:
        failures.append("feature_digest_mismatch")
    if not sequence_match:
        failures.append("sequence_digest_mismatch")
    model_a_signal_match = _to_int(row_a.get("model_a_signal")) == _to_int(
        row_b.get("model_a_signal")
    )
    model_b_signal_match = _to_int(row_a.get("model_b_from_flat_signal")) == _to_int(
        row_b.get("model_b_from_flat_signal")
    )
    model_b_entry_match = _to_bool(row_a.get("model_b_entry_condition")) == _to_bool(
        row_b.get("model_b_entry_condition")
    )
    model_b_hold_match = _to_bool(row_a.get("model_b_hold_condition")) == _to_bool(
        row_b.get("model_b_hold_condition")
    )
    if not model_a_signal_match:
        failures.append("model_a_overlay_signal_mismatch")
    if not model_b_signal_match:
        failures.append("model_b_overlay_signal_mismatch")
    if not model_b_entry_match:
        failures.append("model_b_entry_condition_mismatch")
    if not model_b_hold_match:
        failures.append("model_b_hold_condition_mismatch")
    both_flat = _to_int(row_a.get("actual_position_count"), -1) == 0 and _to_int(
        row_b.get("actual_position_count"), -1
    ) == 0
    both_no_pending = _to_int(row_a.get("pending_order_count"), -1) == 0 and _to_int(
        row_b.get("pending_order_count"), -1
    ) == 0
    if not both_flat:
        failures.append("broker_position_detected")
    if not both_no_pending:
        failures.append("pending_order_detected")
    no_forbidden = (
        not str(row_a.get("forbidden_trade_function_calls", "")).strip(" []()'\"")
        and not str(row_b.get("forbidden_trade_function_calls", "")).strip(" []()'\"")
        and not _to_bool(row_a.get("order_check_called"))
        and not _to_bool(row_b.get("order_check_called"))
        and not _to_bool(row_a.get("order_send_called"))
        and not _to_bool(row_b.get("order_send_called"))
    )
    if not no_forbidden:
        failures.append("forbidden_trade_call_detected")
    return SyncComparison(
        event_time_utc=event_a or event_b,
        model_a_probability_up=p_a,
        model_b_probability_up=p_b,
        probability_absolute_difference=p_diff,
        probability_tolerance=float(probability_tolerance),
        rates_digest_matches=rates_match,
        feature_digest_matches=features_match,
        sequence_digest_matches=sequence_match,
        model_a_signal_matches=model_a_signal_match,
        model_b_signal_matches=model_b_signal_match,
        model_b_entry_condition_matches=model_b_entry_match,
        model_b_hold_condition_matches=model_b_hold_match,
        model_a_role_signal=_to_int(row_a.get("role_shadow_signal")),
        model_b_role_signal=_to_int(row_b.get("role_shadow_signal")),
        both_roles_flat=both_flat,
        both_roles_no_pending_orders=both_no_pending,
        both_roles_no_forbidden_calls=no_forbidden,
        passed=not failures,
        failures=tuple(failures),
    )


def synchronised_event_comparisons(
    rows_a: Iterable[Mapping[str, Any]],
    rows_b: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    probability_tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
) -> list[SyncComparison]:
    events_a = _fresh_event_map(rows_a, run_id=run_id)
    events_b = _fresh_event_map(rows_b, run_id=run_id)
    common = sorted(set(events_a) & set(events_b))
    return [
        compare_synchronised_event(
            events_a[event_time],
            events_b[event_time],
            probability_tolerance=probability_tolerance,
        )
        for event_time in common
    ]


def _close_enough(
    a: float,
    b: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    return abs(a - b) <= max(absolute_tolerance, relative_tolerance * max(abs(a), abs(b), 1.0))


def compare_economic_calculations(
    model_a: Mapping[str, Any],
    model_b: Mapping[str, Any],
    *,
    absolute_tolerance: float = DEFAULT_ECONOMIC_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_ECONOMIC_RELATIVE_TOLERANCE,
) -> EconomicComparison:
    fields = (
        "buy_margin_account_currency",
        "sell_margin_account_currency",
        "buy_profit_for_positive_price_move_account_currency",
        "sell_profit_for_positive_price_move_account_currency",
    )
    values_a = {field: _to_float(model_a.get(field)) for field in fields}
    values_b = {field: _to_float(model_b.get(field)) for field in fields}
    valid_a = all(math.isfinite(value) and value > 0 for value in values_a.values())
    valid_b = all(math.isfinite(value) and value > 0 for value in values_b.values())
    failures: list[str] = []
    if not valid_a:
        failures.append("model_a_economic_calculation_invalid")
    if not valid_b:
        failures.append("model_b_economic_calculation_invalid")
    diffs = {field: abs(values_a[field] - values_b[field]) for field in fields}
    values_match = all(
        _close_enough(
            values_a[field],
            values_b[field],
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for field in fields
    )
    if not values_match:
        failures.append("economic_calculation_mismatch")
    forbidden_call_detected = any(
        _to_bool(value)
        for value in (
            model_a.get("order_check_called"),
            model_a.get("order_send_called"),
            model_b.get("order_check_called"),
            model_b.get("order_send_called"),
        )
    ) or bool(model_a.get("forbidden_trade_function_calls")) or bool(
        model_b.get("forbidden_trade_function_calls")
    )
    if forbidden_call_detected:
        failures.append("forbidden_order_call_in_economic_check")
    return EconomicComparison(
        buy_margin_absolute_difference=diffs["buy_margin_account_currency"],
        sell_margin_absolute_difference=diffs["sell_margin_account_currency"],
        buy_profit_absolute_difference=diffs[
            "buy_profit_for_positive_price_move_account_currency"
        ],
        sell_profit_absolute_difference=diffs[
            "sell_profit_for_positive_price_move_account_currency"
        ],
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
        model_a_calculations_valid=valid_a,
        model_b_calculations_valid=valid_b,
        values_match=values_match,
        passed=not failures,
        failures=tuple(failures),
    )


def sync_comparison_rows(comparisons: Sequence[SyncComparison]) -> list[dict[str, Any]]:
    return [asdict(item) for item in comparisons]


def summary_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = report.get("summary", {}) or {}
    for key, value in summary.items():
        rows.append({"scope": "summary", "metric": key, "value": value})
    gate = report.get("validations", {}) or {}
    for key, value in gate.items():
        rows.append({"scope": "validation", "metric": key, "value": value})
    economics = report.get("economic_comparison", {}) or {}
    for key, value in economics.items():
        if key != "failures":
            rows.append({"scope": "economic", "metric": key, "value": value})
    return rows


def build_final_report(
    *,
    run_id: str,
    started_utc: str,
    completed_utc: str,
    required_synchronised_events: int,
    poll_seconds: int,
    max_runtime_minutes: int,
    comparisons: Sequence[SyncComparison],
    worker_status_a: Mapping[str, Any],
    worker_status_b: Mapping[str, Any],
    step4a_report: Mapping[str, Any],
    source_paths: Mapping[str, Path],
    probability_tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
) -> dict[str, Any]:
    economic_a = worker_status_a.get("economic_sanity", {}) or {}
    economic_b = worker_status_b.get("economic_sanity", {}) or {}
    economic_comparison = compare_economic_calculations(economic_a, economic_b)
    passing_comparisons = [item for item in comparisons if item.passed]
    enough_sync = len(comparisons) >= int(required_synchronised_events)
    all_sync_pass = enough_sync and all(item.passed for item in comparisons)
    workers_pass = worker_status_a.get("status") == "PASS" and worker_status_b.get("status") == "PASS"
    workers_no_orders = (
        worker_status_a.get("order_check_called") is False
        and worker_status_a.get("order_send_called") is False
        and worker_status_b.get("order_check_called") is False
        and worker_status_b.get("order_send_called") is False
    )
    workers_flat = (
        int(worker_status_a.get("final_position_count", -1)) == 0
        and int(worker_status_b.get("final_position_count", -1)) == 0
        and int(worker_status_a.get("final_pending_order_count", -1)) == 0
        and int(worker_status_b.get("final_pending_order_count", -1)) == 0
    )
    artifacts_match = (
        bool(worker_status_a.get("model_artifact_sha256"))
        and worker_status_a.get("model_artifact_sha256") == worker_status_b.get("model_artifact_sha256")
        and worker_status_a.get("scaler_artifact_sha256") == worker_status_b.get("scaler_artifact_sha256")
        and worker_status_a.get("feature_order_sha256") == worker_status_b.get("feature_order_sha256")
    )
    identities_match_step4a = (
        worker_status_a.get("login_masked")
        == ((step4a_report.get("model_a", {}) or {}).get("account", {}) or {}).get("login_masked")
        and worker_status_b.get("login_masked")
        == ((step4a_report.get("model_b", {}) or {}).get("account", {}) or {}).get("login_masked")
    )
    validations = {
        "step4a_prerequisite_formal_pass": step4a_report.get("formal_gate") is True,
        "both_workers_completed_cleanly": workers_pass,
        "minimum_synchronised_events_observed": enough_sync,
        "all_synchronised_event_comparisons_passed": all_sync_pass,
        "probability_tolerance_preserved": probability_tolerance
        == DEFAULT_PROBABILITY_TOLERANCE,
        "economic_calculation_parity_passed": economic_comparison.passed,
        "no_order_check_called": workers_no_orders,
        "no_order_send_called": workers_no_orders,
        "both_accounts_flat_at_close": workers_flat,
        "worker_accounts_match_step4a": identities_match_step4a,
        "frozen_model_scaler_and_feature_order_match": artifacts_match,
        "worker_runtime_roots_distinct": normalised_path(
            str(worker_status_a.get("role_root", ""))
        )
        != normalised_path(str(worker_status_b.get("role_root", ""))),
    }
    formal_gate = all(validations.values())
    source_fingerprints: dict[str, Any] = {}
    for label, path in source_paths.items():
        if path.exists() and path.is_file():
            source_fingerprints[label] = {"path": str(path), **sha256_file(path)}
    return {
        "stage": 3,
        "step": "4B",
        "status": "PASS" if formal_gate else "FAIL",
        "formal_gate": formal_gate,
        "patch_version": "stage3_step4b_v1_2",
        "purpose": "concurrent_dual_terminal_shadow_synchronisation_no_order",
        "run_id": run_id,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "orders_enabled": False,
        "order_check_called": False,
        "order_send_called": False,
        "configuration": {
            "required_synchronised_events": int(required_synchronised_events),
            "poll_seconds": int(poll_seconds),
            "max_runtime_minutes": int(max_runtime_minutes),
            "probability_tolerance": float(probability_tolerance),
            "economic_absolute_tolerance": DEFAULT_ECONOMIC_ABSOLUTE_TOLERANCE,
            "economic_relative_tolerance": DEFAULT_ECONOMIC_RELATIVE_TOLERANCE,
        },
        "summary": {
            "synchronised_event_count": len(comparisons),
            "passing_synchronised_event_count": len(passing_comparisons),
            "first_synchronised_event_utc": comparisons[0].event_time_utc if comparisons else None,
            "last_synchronised_event_utc": comparisons[-1].event_time_utc if comparisons else None,
            "maximum_probability_absolute_difference": max(
                (item.probability_absolute_difference for item in comparisons), default=None
            ),
            "model_a_worker_fresh_events": worker_status_a.get("fresh_event_count"),
            "model_b_worker_fresh_events": worker_status_b.get("fresh_event_count"),
            "model_a_final_position_count": worker_status_a.get("final_position_count"),
            "model_b_final_position_count": worker_status_b.get("final_position_count"),
            "model_a_final_pending_order_count": worker_status_a.get("final_pending_order_count"),
            "model_b_final_pending_order_count": worker_status_b.get("final_pending_order_count"),
        },
        "validations": validations,
        "economic_comparison": asdict(economic_comparison),
        "model_a_worker": dict(worker_status_a),
        "model_b_worker": dict(worker_status_b),
        "synchronised_events": sync_comparison_rows(comparisons),
        "source_fingerprints": source_fingerprints,
        "decision": {
            "stage3_step4b_passed": formal_gate,
            "final_14_day_run_authorised": False,
            "next_step_if_pass": (
                "Stage 3 Step 4C - external Windows VPS migration and launch-readiness gate, "
                "followed by a one-full-day controlled VPS soak test before the final 14-calendar-day run."
            ),
        },
    }


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _terminate_process(process: subprocess.Popen[Any], *, grace_seconds: int = 20) -> None:
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def build_worker_command(
    *,
    python_executable: str,
    script_path: Path,
    repo_root: Path,
    run_id: str,
    role: str,
    config_path: Path,
    model_a_config: Path,
    model_b_config: Path,
    freeze_manifest: Path,
    step4a_report: Path,
    worker_root: Path,
    stop_file: Path,
    deadline_utc: str,
    poll_seconds: int,
    server_time_offset_hours: int,
    allow_onednn: bool,
) -> list[str]:
    command = [
        python_executable,
        str(script_path),
        "--worker-role",
        role,
        "--repo-root",
        str(repo_root),
        "--run-id",
        run_id,
        "--config",
        str(config_path),
        "--model-a-config",
        str(model_a_config),
        "--model-b-config",
        str(model_b_config),
        "--freeze-manifest",
        str(freeze_manifest),
        "--step4a-report",
        str(step4a_report),
        "--worker-root",
        str(worker_root),
        "--stop-file",
        str(stop_file),
        "--deadline-utc",
        deadline_utc,
        "--poll-seconds",
        str(poll_seconds),
        "--server-time-offset-hours",
        str(server_time_offset_hours),
    ]
    if allow_onednn:
        command.append("--allow-onednn")
    return command


def wait_for_workers_and_sync(
    *,
    process_a: subprocess.Popen[Any],
    process_b: subprocess.Popen[Any],
    paths_a: WorkerPaths,
    paths_b: WorkerPaths,
    stop_file: Path,
    run_id: str,
    required_synchronised_events: int,
    deadline: datetime,
    probability_tolerance: float,
    parent_poll_seconds: int = 5,
) -> list[SyncComparison]:
    latest_comparisons: list[SyncComparison] = []
    last_reported_count = -1
    while datetime.now(timezone.utc) < deadline:
        rows_a = read_csv_rows(paths_a.events_csv)
        rows_b = read_csv_rows(paths_b.events_csv)
        latest_comparisons = synchronised_event_comparisons(
            rows_a,
            rows_b,
            run_id=run_id,
            probability_tolerance=probability_tolerance,
        )
        if len(latest_comparisons) != last_reported_count:
            last_reported_count = len(latest_comparisons)
            LOGGER.info(
                "Step 4B progress: %s/%s synchronised completed M15 events",
                len(latest_comparisons),
                required_synchronised_events,
            )
        if len(latest_comparisons) >= required_synchronised_events:
            stop_file.parent.mkdir(parents=True, exist_ok=True)
            stop_file.write_text(utc_now_iso(), encoding="utf-8")
            break
        if process_a.poll() is not None or process_b.poll() is not None:
            break
        time.sleep(max(1, parent_poll_seconds))
    if not stop_file.exists():
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.write_text(utc_now_iso(), encoding="utf-8")
    return latest_comparisons
