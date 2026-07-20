"""Single-terminal MT5 shadow logger for Stage 2 Step 2A.

This module is intentionally read-only with respect to MT5.  It connects to a
single manually logged-in MT5 demo terminal, fetches completed M15 bars only,
rebuilds the frozen feature state, runs the frozen CNN-LSTM on the latest valid
contiguous 48-bar window, and appends a shadow signal/audit row.  It does not
send orders and does not expose trading functions through the MT5 safety proxy.

Step 2A is a shadow logger, not a live execution engine.  It records the latest
prediction and intended Model A / Model B directions from the current feature
window.  Full broker order handling and live PnL accounting belong to later
Stage 3 execution gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import math
import os
import time

import numpy as np
import pandas as pd

from capstone_trading.config import ModelAConfig, load_yaml_mapping, safe_repository_path
from capstone_trading.data.canonical_bars import M15_DELTA
from capstone_trading.data.features_m15 import add_relative_price_features
from capstone_trading.data.sequences import (
    scale_feature_frame,
    valid_sequence_positions,
    validate_plan_contiguity,
)
from capstone_trading.evaluation.model_b_replay import ModelBOverlayRules
from capstone_trading.evaluation.trading_replay import ModelAOverlayRules
from capstone_trading.model_loader import (
    load_and_validate_model,
    load_and_validate_scaler,
    report_to_dict,
)
from capstone_trading.runtime.mt5_readiness import (
    Mt5ReadinessError,
    Mt5RuntimeConfig,
    SafeMt5Proxy,
    analyse_rates,
    build_package_snapshot,
    fetch_completed_m15_rates,
    initialise_terminal,
    inspect_account,
    inspect_terminal,
    inspect_tick,
    rates_for_csv,
    resolve_symbol,
    resolve_timeframe,
)


class Mt5ShadowError(RuntimeError):
    """Raised when the Stage 2 Step 2A shadow logger fails."""


@dataclass(frozen=True)
class ShadowRuntimeConfig:
    """Broker/runtime-only config for the single-terminal shadow logger."""

    mt5: Mt5RuntimeConfig
    bars_to_fetch: int = 420
    minimum_feature_rows: int = 260
    minimum_valid_sequences: int = 1
    append_signals: bool = True
    duplicate_policy: str = "skip"  # skip or append
    state_path: str = "runtime/state/stage2_step2a_v1_1_shadow_state.json"
    signals_csv_path: str = "runtime/shadow/stage2_step2a_v1_1_shadow_signals.csv"
    latest_signal_csv_path: str = "runtime/reports/stage2_step2a_v1_1_latest_shadow_signal.csv"
    mt5_server_time_offset_hours: int = 3
    enforce_no_future_canonical_bar: bool = True
    max_future_canonical_bar_minutes: int = 2


@dataclass(frozen=True)
class LiveFeatureReport:
    bars_rows: int
    pre_dropna_rows: int
    feature_rows: int
    feature_count: int
    dropped_rows: int
    first_feature_time_utc: str | None
    latest_feature_time_utc: str | None
    infinite_values_before_dropna: int
    latest_feature_matches_latest_completed_bar: bool


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


@dataclass(frozen=True)
class LatestShadowSignal:
    event_time_utc: str
    window_start_utc: str
    window_end_utc: str
    latest_completed_bar_time_utc: str
    selected_symbol: str
    probability_up: float
    model_a_signal: int
    model_a_signal_name: str
    model_b_from_flat_signal: int
    model_b_from_flat_signal_name: str
    model_b_hold_condition: bool
    model_b_entry_condition: bool
    sequence_length: int
    feature_count: int
    valid_sequence_count: int
    event_is_latest_feature: bool
    event_is_latest_completed_bar: bool
    stale_event_warning: bool
    duplicate_event: bool
    appended_to_signal_log: bool
    orders_enabled: bool = False


@dataclass(frozen=True)
class ShadowSnapshot:
    status: str
    formal_gate: bool
    stage: int
    step: str
    mode: str
    mt5_used: bool
    orders_enabled: bool
    shadow_only: bool
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
    time_normalisation: Mapping[str, Any]
    feature_report: Mapping[str, Any]
    model_and_scaler: Mapping[str, Any]
    latest_signal: Mapping[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def load_shadow_runtime_config(path: Path) -> ShadowRuntimeConfig:
    """Load Stage 2 Step 2A broker/runtime config.

    This file must not contain passwords or frozen research parameters.  It may
    contain terminal path, symbol candidates, data-fetch sizes, and output paths.
    """

    raw = load_yaml_mapping(path)
    terminal = raw.get("terminal", {}) or {}
    symbol = raw.get("symbol", {}) or {}
    data = raw.get("data", {}) or {}
    safety = raw.get("safety", {}) or {}
    shadow = raw.get("shadow", {}) or {}
    time_config = raw.get("time", {}) or {}
    if not all(isinstance(item, Mapping) for item in (terminal, symbol, data, safety, shadow, time_config)):
        raise Mt5ShadowError("Invalid MT5 shadow runtime config structure")

    candidates_raw = symbol.get("candidates", Mt5RuntimeConfig.symbol_candidates)
    if isinstance(candidates_raw, str):
        candidates = (candidates_raw,)
    else:
        candidates = tuple(str(item).strip() for item in candidates_raw if str(item).strip())
    if not candidates:
        raise Mt5ShadowError("At least one symbol candidate is required")

    bars_to_fetch = int(data.get("bars_to_fetch", shadow.get("bars_to_fetch", 420)))
    if bars_to_fetch < 260:
        raise Mt5ShadowError("Stage 2 Step 2A should fetch at least 260 completed M15 bars")
    duplicate_policy = str(shadow.get("duplicate_policy", "skip")).strip().lower()
    if duplicate_policy not in {"skip", "append"}:
        raise Mt5ShadowError("shadow.duplicate_policy must be 'skip' or 'append'")

    mt5_config = Mt5RuntimeConfig(
        terminal_path=_none_if_blank(terminal.get("path")),
        symbol_candidates=candidates,
        timeframe_name=str(data.get("timeframe", "M15")).upper(),
        bars_to_fetch=bars_to_fetch,
        min_completed_bars=int(data.get("min_completed_bars", 260)),
        require_demo_account=bool(safety.get("require_demo_account", True)),
        allow_market_closed_stale_bar=bool(data.get("allow_market_closed_stale_bar", True)),
        max_latest_closed_bar_age_minutes_warning=int(
            data.get("max_latest_closed_bar_age_minutes_warning", 4320)
        ),
        require_symbol_visible=bool(symbol.get("require_visible", True)),
    )
    return ShadowRuntimeConfig(
        mt5=mt5_config,
        bars_to_fetch=bars_to_fetch,
        minimum_feature_rows=int(shadow.get("minimum_feature_rows", 260)),
        minimum_valid_sequences=int(shadow.get("minimum_valid_sequences", 1)),
        append_signals=bool(shadow.get("append_signals", True)),
        duplicate_policy=duplicate_policy,
        state_path=str(shadow.get("state_path", "runtime/state/stage2_step2a_v1_1_shadow_state.json")),
        signals_csv_path=str(shadow.get("signals_csv_path", "runtime/shadow/stage2_step2a_v1_1_shadow_signals.csv")),
        latest_signal_csv_path=str(
            shadow.get("latest_signal_csv_path", "runtime/reports/stage2_step2a_v1_1_latest_shadow_signal.csv")
        ),
        mt5_server_time_offset_hours=int(time_config.get("mt5_server_time_offset_hours", 3)),
        enforce_no_future_canonical_bar=bool(time_config.get("enforce_no_future_canonical_bar", True)),
        max_future_canonical_bar_minutes=int(time_config.get("max_future_canonical_bar_minutes", 2)),
    )


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def position_name(value: int) -> str:
    return {-1: "SHORT", 0: "FLAT", 1: "LONG"}.get(int(value), f"UNKNOWN_{value}")


def model_a_signal_from_probability(probability: float, rules: ModelAOverlayRules) -> int:
    if probability >= rules.long_threshold:
        return 1
    if probability <= rules.short_threshold:
        return -1
    return 0


def model_b_from_flat_signal(probability: float, rules: ModelBOverlayRules) -> int:
    return 1 if probability >= rules.entry_threshold else 0


def _raw_server_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        if isinstance(value, pd.Timestamp):
            return value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
        if isinstance(value, (int, float, np.integer, np.floating)):
            return pd.Timestamp(datetime.fromtimestamp(float(value), tz=timezone.utc))
        return pd.Timestamp(value).tz_convert("UTC")
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
    """Convert MT5 server-time bars into canonical UTC timestamps.

    Dukascopy documents MT4/MT5 server time as GMT+3 during summer and GMT+2
    during winter.  The historical research dataset and all frozen feature
    contracts are UTC-aligned, so the live MT5 timestamps are shifted back by
    the configured server offset before feature generation, signal logging, and
    age checks.
    """

    if "time" not in rates.columns:
        raise Mt5ShadowError("MT5 rates are missing the time column required for time normalisation")
    if server_time_offset_hours not in {0, 2, 3}:
        raise Mt5ShadowError(
            "mt5_server_time_offset_hours must be one of 0, 2, or 3. "
            "Dukascopy MT4/MT5 is normally GMT+3 in summer and GMT+2 in winter."
        )
    frame = rates.copy()
    raw_times = pd.to_datetime(frame["time"], utc=True)
    offset = pd.Timedelta(hours=int(server_time_offset_hours))
    canonical_times = raw_times - offset
    frame["time"] = canonical_times
    frame = frame.sort_values("time").reset_index(drop=True)

    now = pd.Timestamp(now_utc or datetime.now(timezone.utc)).tz_convert("UTC")
    raw_latest = pd.Timestamp(raw_times.iloc[-1]).tz_convert("UTC") if len(raw_times) else None
    canonical_latest = pd.Timestamp(canonical_times.iloc[-1]).tz_convert("UTC") if len(canonical_times) else None
    latest_age_after = None
    latest_age_before = None
    future_minutes = 0.0
    if raw_latest is not None:
        latest_age_before = float((now - raw_latest).total_seconds() / 60.0)
    if canonical_latest is not None:
        latest_age_after = float((now - canonical_latest).total_seconds() / 60.0)
        future_minutes = max(0.0, -latest_age_after)
        if enforce_no_future_canonical_bar and future_minutes > float(max_future_canonical_bar_minutes):
            raise Mt5ShadowError(
                "Latest completed MT5 bar is still in the future after configured server-time conversion: "
                f"canonical_latest={canonical_latest.isoformat()}, now_utc={now.isoformat()}, "
                f"future_minutes={future_minutes:.2f}, offset_hours={server_time_offset_hours}. "
                "Check whether the broker is currently using GMT+3 or GMT+2."
            )

    raw_tick = _raw_server_timestamp(tick.get("time")) if tick else None
    canonical_tick = raw_tick - offset if raw_tick is not None else None
    tick_age_after = float((now - canonical_tick).total_seconds() / 60.0) if canonical_tick is not None else None

    report = TimeNormalisationReport(
        policy="fixed_broker_server_offset_to_utc",
        mt5_server_time_offset_hours=int(server_time_offset_hours),
        raw_first_bar_server_time=pd.Timestamp(raw_times.iloc[0]).tz_convert("UTC").isoformat() if len(raw_times) else None,
        raw_latest_bar_server_time=raw_latest.isoformat() if raw_latest is not None else None,
        canonical_first_bar_time_utc=pd.Timestamp(canonical_times.iloc[0]).tz_convert("UTC").isoformat() if len(canonical_times) else None,
        canonical_latest_bar_time_utc=canonical_latest.isoformat() if canonical_latest is not None else None,
        latest_bar_age_minutes_after_conversion=latest_age_after,
        latest_bar_age_minutes_before_conversion=latest_age_before,
        latest_bar_future_minutes_after_conversion=float(future_minutes),
        raw_tick_server_time=raw_tick.isoformat() if raw_tick is not None else None,
        canonical_tick_time_utc=canonical_tick.isoformat() if canonical_tick is not None else None,
        tick_age_minutes_after_conversion=tick_age_after,
        conversion_applied=bool(server_time_offset_hours != 0),
        server_time_note=(
            "Dukascopy documents MT4/MT5 server time as GMT+3 in summer and GMT+2 in winter. "
            "This hotfix converts MT5 server timestamps into canonical UTC before feature generation."
        ),
    )
    return frame, report


def convert_mt5_rates_to_feature_bars(rates: pd.DataFrame) -> pd.DataFrame:
    """Convert MT5 rates into the feature-builder bar format.

    MT5 provides ``tick_volume``.  The frozen volume-assisted pipeline uses a
    ``volume`` column, so the shadow feed maps tick volume to volume explicitly.
    """

    required = {"time", "open", "high", "low", "close", "tick_volume"}
    missing = sorted(required - set(rates.columns))
    if missing:
        raise Mt5ShadowError(f"MT5 rates are missing columns required for features: {missing}")
    frame = rates.loc[:, ["time", "open", "high", "low", "close", "tick_volume"]].copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    bars = pd.DataFrame(index=pd.DatetimeIndex(frame["time"], tz="UTC"))
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(frame[column].to_numpy(), errors="raise").astype(float)
    bars["volume"] = pd.to_numeric(frame["tick_volume"].to_numpy(), errors="raise").astype(float)
    bars["source_m1_bars"] = 15
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise Mt5ShadowError("Converted MT5 feature bars are not unique and chronological")
    return bars


def build_live_feature_frame(
    bars: pd.DataFrame,
    feature_order: Sequence[str],
) -> tuple[pd.DataFrame, LiveFeatureReport]:
    order = tuple(str(item) for item in feature_order)
    with_features = add_relative_price_features(bars)
    missing = [name for name in order if name not in with_features.columns]
    if missing:
        raise Mt5ShadowError(f"Live feature builder did not produce frozen features: {missing}")
    pre_dropna = with_features.loc[:, list(order)].copy()
    numeric = pre_dropna.select_dtypes(include=[np.number])
    infinite_values_before = int(np.isinf(numeric.to_numpy(dtype=np.float64)).sum())
    final = pre_dropna.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if final.empty:
        raise Mt5ShadowError("Live feature reconstruction produced no usable feature rows")
    if tuple(final.columns) != order:
        raise Mt5ShadowError("Live feature column order differs from frozen feature order")
    values = final.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise Mt5ShadowError("Live feature frame contains non-finite values after dropna")
    latest_completed = pd.Timestamp(bars.index[-1]).tz_convert("UTC")
    latest_feature = pd.Timestamp(final.index[-1]).tz_convert("UTC")
    report = LiveFeatureReport(
        bars_rows=int(len(bars)),
        pre_dropna_rows=int(len(pre_dropna)),
        feature_rows=int(len(final)),
        feature_count=len(order),
        dropped_rows=int(len(pre_dropna) - len(final)),
        first_feature_time_utc=pd.Timestamp(final.index[0]).tz_convert("UTC").isoformat(),
        latest_feature_time_utc=latest_feature.isoformat(),
        infinite_values_before_dropna=infinite_values_before,
        latest_feature_matches_latest_completed_bar=latest_feature == latest_completed,
    )
    return final, report


def _predict_single_probability(model: Any, sequence: np.ndarray) -> float:
    batch = np.asarray(sequence, dtype=np.float32)[np.newaxis, ...]
    try:
        output = model(batch, training=False)
        if hasattr(output, "detach"):
            output = output.detach()
            if hasattr(output, "cpu"):
                output = output.cpu()
        if hasattr(output, "numpy"):
            output = output.numpy()
        value = float(np.asarray(output).reshape(-1)[0])
    except Exception as exc:
        raise Mt5ShadowError(f"Frozen model shadow inference failed: {exc}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise Mt5ShadowError(f"Frozen model produced invalid shadow probability: {value}")
    return value


def load_shadow_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "records_written": 0, "last_event_time_utc": None}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Mt5ShadowError(f"Unable to read shadow state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Mt5ShadowError(f"Shadow state must be a JSON object: {path}")
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_signal_row(path: Path, row: Mapping[str, Any], *, fieldnames: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))


def signal_to_csv_row(signal: LatestShadowSignal, *, run_id: str, mode: str) -> dict[str, Any]:
    payload = asdict(signal)
    payload.update({"run_id": run_id, "mode": mode, "logged_utc": datetime.now(timezone.utc).isoformat()})
    return payload


def shadow_signal_fieldnames() -> list[str]:
    return [
        "run_id",
        "mode",
        "logged_utc",
        "event_time_utc",
        "window_start_utc",
        "window_end_utc",
        "latest_completed_bar_time_utc",
        "selected_symbol",
        "probability_up",
        "model_a_signal",
        "model_a_signal_name",
        "model_b_from_flat_signal",
        "model_b_from_flat_signal_name",
        "model_b_hold_condition",
        "model_b_entry_condition",
        "sequence_length",
        "feature_count",
        "valid_sequence_count",
        "event_is_latest_feature",
        "event_is_latest_completed_bar",
        "stale_event_warning",
        "duplicate_event",
        "appended_to_signal_log",
        "orders_enabled",
    ]


def create_latest_shadow_signal(
    *,
    feature_frame: pd.DataFrame,
    scaled_features: np.ndarray,
    model: Any,
    config: ModelAConfig,
    selected_symbol: str,
    latest_completed_bar_time: pd.Timestamp,
    rules_a: ModelAOverlayRules,
    rules_b: ModelBOverlayRules,
) -> LatestShadowSignal:
    plan = valid_sequence_positions(pd.DatetimeIndex(feature_frame.index), config.sequence_length)
    validate_plan_contiguity(pd.DatetimeIndex(feature_frame.index), plan)
    if plan.sequence_count < 1:
        raise Mt5ShadowError(
            f"No contiguous {config.sequence_length}-bar window is available from current MT5 completed bars"
        )
    end = int(plan.ends[-1])
    start = int(plan.starts[-1])
    sequence = scaled_features[start : end + 1]
    expected_shape = (config.sequence_length, config.feature_count)
    if sequence.shape != expected_shape:
        raise Mt5ShadowError(f"Latest shadow sequence shape mismatch: expected {expected_shape}, found {sequence.shape}")
    p_up = _predict_single_probability(model, sequence)
    event_time = pd.Timestamp(feature_frame.index[end]).tz_convert("UTC")
    window_start = pd.Timestamp(feature_frame.index[start]).tz_convert("UTC")
    window_end = pd.Timestamp(feature_frame.index[end]).tz_convert("UTC")
    latest_feature_time = pd.Timestamp(feature_frame.index[-1]).tz_convert("UTC")
    latest_completed = pd.Timestamp(latest_completed_bar_time).tz_convert("UTC")
    a_signal = model_a_signal_from_probability(p_up, rules_a)
    b_signal = model_b_from_flat_signal(p_up, rules_b)
    b_hold = bool(p_up >= rules_b.exit_threshold)
    b_entry = bool(p_up >= rules_b.entry_threshold)
    stale_event_warning = event_time != latest_feature_time or event_time != latest_completed
    return LatestShadowSignal(
        event_time_utc=event_time.isoformat(),
        window_start_utc=window_start.isoformat(),
        window_end_utc=window_end.isoformat(),
        latest_completed_bar_time_utc=latest_completed.isoformat(),
        selected_symbol=selected_symbol,
        probability_up=float(p_up),
        model_a_signal=int(a_signal),
        model_a_signal_name=position_name(a_signal),
        model_b_from_flat_signal=int(b_signal),
        model_b_from_flat_signal_name=position_name(b_signal),
        model_b_hold_condition=b_hold,
        model_b_entry_condition=b_entry,
        sequence_length=int(config.sequence_length),
        feature_count=int(config.feature_count),
        valid_sequence_count=int(plan.sequence_count),
        event_is_latest_feature=event_time == latest_feature_time,
        event_is_latest_completed_bar=event_time == latest_completed,
        stale_event_warning=bool(stale_event_warning),
        duplicate_event=False,
        appended_to_signal_log=False,
        orders_enabled=False,
    )


def _with_signal_flags(
    signal: LatestShadowSignal,
    *,
    duplicate_event: bool,
    appended: bool,
) -> LatestShadowSignal:
    return replace(signal, duplicate_event=bool(duplicate_event), appended_to_signal_log=bool(appended))


def run_shadow_once(
    *,
    mt5_module: Any,
    runtime_config: ShadowRuntimeConfig,
    config_a: ModelAConfig,
    feature_order: Sequence[str],
    model: Any,
    scaler: Any,
    rules_a: ModelAOverlayRules,
    rules_b: ModelBOverlayRules,
    state_path: Path,
    signals_csv_path: Path,
    mode: str,
    run_id: str,
    now_utc: datetime | None = None,
) -> tuple[ShadowSnapshot, pd.DataFrame, LatestShadowSignal]:
    proxy = SafeMt5Proxy(mt5_module)
    initialized = False
    shutdown_called = False
    warnings: list[str] = []
    snapshot: ShadowSnapshot | None = None
    rates: pd.DataFrame | None = None
    signal: LatestShadowSignal | None = None
    try:
        timeframe_value = resolve_timeframe(proxy, runtime_config.mt5.timeframe_name)
        initialise_terminal(proxy, runtime_config.mt5)
        initialized = True
        package = build_package_snapshot(proxy)
        terminal = inspect_terminal(proxy)
        account = inspect_account(proxy, require_demo=runtime_config.mt5.require_demo_account)
        symbol_resolution = resolve_symbol(proxy, runtime_config.mt5)
        tick = inspect_tick(proxy, symbol_resolution.selected_symbol)
        raw_rates = fetch_completed_m15_rates(
            proxy,
            symbol=symbol_resolution.selected_symbol,
            timeframe_value=timeframe_value,
            count=runtime_config.bars_to_fetch,
        )
        rates, time_normalisation_report = normalise_mt5_server_times(
            raw_rates,
            tick,
            server_time_offset_hours=runtime_config.mt5_server_time_offset_hours,
            now_utc=now_utc,
            enforce_no_future_canonical_bar=runtime_config.enforce_no_future_canonical_bar,
            max_future_canonical_bar_minutes=runtime_config.max_future_canonical_bar_minutes,
        )
        if time_normalisation_report.latest_bar_age_minutes_before_conversion is not None and (
            time_normalisation_report.latest_bar_age_minutes_before_conversion < -float(
                runtime_config.max_future_canonical_bar_minutes
            )
        ):
            warnings.append(
                "Raw MT5 bar time was ahead of system UTC before broker-time conversion; "
                "canonical UTC conversion was applied before feature generation."
            )
        rates_report = analyse_rates(
            rates,
            symbol=symbol_resolution.selected_symbol,
            timeframe_name=runtime_config.mt5.timeframe_name,
            timeframe_value=timeframe_value,
            requested_bars=runtime_config.bars_to_fetch,
            min_completed_bars=runtime_config.mt5.min_completed_bars,
            max_latest_closed_bar_age_minutes_warning=runtime_config.mt5.max_latest_closed_bar_age_minutes_warning,
            allow_market_closed_stale_bar=runtime_config.mt5.allow_market_closed_stale_bar,
            now_utc=now_utc,
        )
        if rates_report.latest_closed_bar_stale_warning:
            warnings.append(
                "Latest completed M15 bar is older than configured warning threshold; this is usually normal over weekends."
            )
        bars = convert_mt5_rates_to_feature_bars(rates)
        feature_frame, feature_report = build_live_feature_frame(bars, feature_order)
        if len(feature_frame) < runtime_config.minimum_feature_rows:
            raise Mt5ShadowError(
                f"Only {len(feature_frame)} live feature rows are available; minimum configured is "
                f"{runtime_config.minimum_feature_rows}. Increase bars_to_fetch or wait for more data."
            )
        scaled = scale_feature_frame(scaler, feature_frame, feature_order)
        latest_completed = pd.Timestamp(rates["time"].iloc[-1]).tz_convert("UTC")
        signal = create_latest_shadow_signal(
            feature_frame=feature_frame,
            scaled_features=scaled,
            model=model,
            config=config_a,
            selected_symbol=symbol_resolution.selected_symbol,
            latest_completed_bar_time=latest_completed,
            rules_a=rules_a,
            rules_b=rules_b,
        )
        if signal.valid_sequence_count < runtime_config.minimum_valid_sequences:
            raise Mt5ShadowError(
                f"Only {signal.valid_sequence_count} valid sequence(s) available; minimum configured is "
                f"{runtime_config.minimum_valid_sequences}."
            )
        if signal.stale_event_warning:
            warnings.append(
                "Latest valid inference event does not end exactly at the latest completed MT5 bar. "
                "This can happen around session gaps; no order is placed."
            )
        state = load_shadow_state(state_path)
        duplicate_event = state.get("last_event_time_utc") == signal.event_time_utc
        should_append = runtime_config.append_signals and (
            runtime_config.duplicate_policy == "append" or not duplicate_event
        )
        signal = _with_signal_flags(signal, duplicate_event=duplicate_event, appended=should_append)
        if should_append:
            append_signal_row(
                signals_csv_path,
                signal_to_csv_row(signal, run_id=run_id, mode=mode),
                fieldnames=shadow_signal_fieldnames(),
            )
            records_written = int(state.get("records_written", 0)) + 1
            state = {
                "schema_version": 1,
                "last_event_time_utc": signal.event_time_utc,
                "last_probability_up": signal.probability_up,
                "last_model_a_signal": signal.model_a_signal,
                "last_model_b_from_flat_signal": signal.model_b_from_flat_signal,
                "records_written": records_written,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "selected_symbol": symbol_resolution.selected_symbol,
            }
            write_json_atomic(state_path, state)
        snapshot = ShadowSnapshot(
            status="PASS",
            formal_gate=True,
            stage=2,
            step="2A",
            mode=mode,
            mt5_used=True,
            orders_enabled=False,
            shadow_only=True,
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
            time_normalisation=asdict(time_normalisation_report),
            feature_report=asdict(feature_report),
            model_and_scaler={},
            latest_signal=asdict(signal),
            warnings=tuple(warnings),
        )
    finally:
        if initialized:
            try:
                proxy.shutdown()
                shutdown_called = True
            except Exception:
                shutdown_called = False
    if snapshot is None or rates is None or signal is None:
        raise Mt5ShadowError("Shadow snapshot did not produce a result")
    snapshot = replace(snapshot, shutdown_called=shutdown_called)
    if proxy.forbidden_attempts:
        raise Mt5ShadowError(f"Forbidden MT5 API methods were accessed: {proxy.forbidden_attempts}")
    return snapshot, rates, signal


def snapshot_to_dict(snapshot: ShadowSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def run_watch_loop(
    *,
    run_once_callable: Any,
    poll_seconds: int,
    max_iterations: int,
) -> list[ShadowSnapshot]:
    if poll_seconds < 1:
        raise Mt5ShadowError("poll_seconds must be positive")
    if max_iterations < 1:
        raise Mt5ShadowError("max_iterations must be positive")
    snapshots: list[ShadowSnapshot] = []
    for idx in range(max_iterations):
        snapshots.append(run_once_callable())
        if idx < max_iterations - 1:
            time.sleep(poll_seconds)
    return snapshots


def prepare_shadow_cpu_environment(*, allow_onednn: bool = False) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    if not allow_onednn:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
