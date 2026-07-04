"""Frozen Model B diagnostic historical replay for Stage 1 Step 5.

Model B is a post-holdout engineering diagnostic.  It deliberately reuses the
frozen Notebook 7 model probabilities and applies the already frozen long-only
risk-oriented overlay from ``config/model_b_v2_frozen.yaml``.  The code in this
module must not optimise, search, or tune thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from capstone_trading.data.canonical_bars import M15_DELTA
from capstone_trading.errors import TradingReplayError
from capstone_trading.evaluation.trading_replay import (
    ReplayMetrics,
    compute_replay_metrics,
    validate_prediction_frame,
)


@dataclass(frozen=True)
class ModelBOverlayRules:
    """Frozen Model B V2 long-only overlay and risk rules."""

    entry_threshold: float
    exit_threshold: float
    max_successful_entries_per_day: int
    daily_loss_log_threshold: float
    total_drawdown_stop: float


@dataclass(frozen=True)
class ModelBDiagnostics:
    """Model B-specific diagnostic counters and invariant results."""

    successful_entry_count: int
    normal_exit_count: int
    daily_entry_cap_block_count: int
    gap_block_count: int
    gap_exit_events: int
    daily_stop_exit_events: int
    total_stop_exit_events: int
    short_position_count: int
    entry_below_threshold_count: int
    active_below_exit_threshold_count: int
    max_successful_entries_in_utc_day: int
    worst_daily_net_return: float
    transaction_cost_log_sum: float
    transaction_cost_burden: float
    invariant_passed: bool
    invariant_failures: tuple[str, ...]


def overlay_rules_from_model_b_config(config_raw: Mapping[str, Any]) -> ModelBOverlayRules:
    """Parse the frozen Model B configuration without changing its values."""

    strategy_id = config_raw.get("strategy_id")
    status = config_raw.get("status")
    if strategy_id != "MODEL_B_V2":
        raise TradingReplayError(f"Expected Model B strategy_id MODEL_B_V2, found {strategy_id!r}")
    if status != "FROZEN_STAGE_0":
        raise TradingReplayError(f"Model B configuration is not frozen: {status!r}")

    overlay = config_raw.get("overlay")
    risk = config_raw.get("risk_governance")
    if not isinstance(overlay, Mapping) or not isinstance(risk, Mapping):
        raise TradingReplayError("Model B configuration is missing overlay or risk_governance")
    if overlay.get("short_positions_allowed") is not False:
        raise TradingReplayError("Model B must remain long-only; short_positions_allowed must be false")
    if int(overlay.get("minimum_hold_eligible_bars", -1)) != 0:
        raise TradingReplayError("Model B minimum_hold_eligible_bars must remain 0")

    entry = overlay.get("entry")
    normal_exit = overlay.get("normal_exit")
    if not isinstance(entry, Mapping) or not isinstance(normal_exit, Mapping):
        raise TradingReplayError("Model B overlay entry and normal_exit mappings are required")

    try:
        entry_threshold = float(entry["threshold"])
        exit_threshold = float(normal_exit["threshold"])
        max_entries = int(overlay["maximum_successful_new_entries_per_utc_day"])
        daily_simple = float(risk["daily_loss_stop_simple_return"])
        total_drawdown_stop = float(risk["total_drawdown_stop"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TradingReplayError(f"Missing or invalid frozen Model B field: {exc}") from exc

    if not 0.0 <= exit_threshold < entry_threshold <= 1.0:
        raise TradingReplayError("Invalid Model B threshold ordering")
    if max_entries < 0:
        raise TradingReplayError("Model B max successful entries per day must be non-negative")
    if not -1.0 < daily_simple < 0.0:
        raise TradingReplayError("Model B daily loss stop must be a negative simple return greater than -100%")
    if not -1.0 < total_drawdown_stop < 0.0:
        raise TradingReplayError("Model B total drawdown stop must be a negative drawdown greater than -100%")

    return ModelBOverlayRules(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        max_successful_entries_per_day=max_entries,
        daily_loss_log_threshold=float(np.log1p(daily_simple)),
        total_drawdown_stop=total_drawdown_stop,
    )


def _reason(
    *,
    previous_position: int,
    next_position: int,
    forced_reason: str,
    blocked_reason: str,
) -> str:
    if blocked_reason:
        return blocked_reason
    if forced_reason:
        if previous_position == next_position:
            return f"{forced_reason}_block"
        return forced_reason
    if previous_position == next_position:
        return "hold"
    if previous_position == 0 and next_position == 1:
        return "policy_entry"
    if previous_position == 1 and next_position == 0:
        return "policy_exit"
    return "policy_change"


def replay_model_b(
    predictions: pd.DataFrame,
    rules: ModelBOverlayRules,
    *,
    cost_bps: float,
) -> tuple[pd.DataFrame, ReplayMetrics, ModelBDiagnostics]:
    """Replay the frozen Model B V2 long-only diagnostic overlay."""

    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    frame = validate_prediction_frame(predictions)
    one_way_cost = float(cost_bps) / 10000.0

    current_position = 0
    current_day: pd.Timestamp | None = None
    successful_entries_today = 0
    daily_log_return = 0.0
    daily_stop_active = False
    total_stop_active = False
    total_stop_triggered = False
    first_total_stop_trigger: pd.Timestamp | None = None
    cumulative_gross_log = 0.0
    cumulative_net_log = 0.0
    running_peak_net_equity = 1.0
    last_time: pd.Timestamp | None = None

    rows: list[dict[str, Any]] = []
    successful_entry_count = 0
    normal_exit_count = 0
    daily_entry_cap_block_count = 0
    gap_exit_events = 0
    gap_block_count = 0
    daily_stop_exit_events = 0
    daily_stop_trigger_count = 0
    total_stop_exit_events = 0
    daily_stop_dates: set[str] = set()
    entries_by_day: dict[str, int] = {}
    daily_net_log_by_day: dict[str, float] = {}

    for row_number, row in enumerate(frame.itertuples(index=False)):
        timestamp = pd.Timestamp(row.time).tz_convert("UTC")
        utc_date = timestamp.date().isoformat()
        day_key = pd.Timestamp(timestamp.date(), tz="UTC")
        if current_day is None or day_key != current_day:
            current_day = day_key
            successful_entries_today = 0
            daily_log_return = 0.0
            daily_stop_active = False

        gap_from_previous = bool(last_time is not None and timestamp - last_time != M15_DELTA)
        probability = float(row.p_up)
        previous_position = int(current_position)
        next_position = previous_position
        forced_reason = ""
        blocked_reason = ""
        signal = 1 if probability >= rules.entry_threshold else 0
        successful_entry_event = 0
        normal_exit_event = 0

        if total_stop_active:
            next_position = 0
            forced_reason = "total_drawdown_stop"
            if previous_position != 0:
                total_stop_exit_events += 1
        elif daily_stop_active:
            next_position = 0
            forced_reason = "daily_loss_stop"
            if previous_position != 0:
                daily_stop_exit_events += 1
        elif gap_from_previous:
            next_position = 0
            if previous_position != 0:
                forced_reason = "gap_exit"
                gap_exit_events += 1
            else:
                blocked_reason = "gap_block"
                gap_block_count += 1
        elif previous_position == 0:
            if probability >= rules.entry_threshold:
                if successful_entries_today < rules.max_successful_entries_per_day:
                    next_position = 1
                    successful_entries_today += 1
                    successful_entry_count += 1
                    successful_entry_event = 1
                    entries_by_day[utc_date] = entries_by_day.get(utc_date, 0) + 1
                else:
                    next_position = 0
                    blocked_reason = "daily_entry_cap_active"
                    daily_entry_cap_block_count += 1
            else:
                next_position = 0
        else:
            # Long position is held only while the model still favours the long
            # side at or above the frozen 0.50 hold threshold.
            if probability >= rules.exit_threshold:
                next_position = 1
            else:
                next_position = 0
                normal_exit_count += 1
                normal_exit_event = 1

        turnover_units = float(abs(next_position - previous_position))
        gross_log_return = float(next_position) * float(row.target_ret_fwd)
        cost_log_return = turnover_units * one_way_cost
        net_log_return = gross_log_return - cost_log_return
        cumulative_gross_log += gross_log_return
        cumulative_net_log += net_log_return
        daily_log_return += net_log_return
        daily_net_log_by_day[utc_date] = daily_log_return
        gross_equity = float(np.exp(cumulative_gross_log))
        net_equity = float(np.exp(cumulative_net_log))
        running_peak_net_equity = max(running_peak_net_equity, net_equity)
        net_drawdown = float(net_equity / running_peak_net_equity - 1.0)

        daily_stop_triggered_now = False
        if not daily_stop_active and daily_log_return <= rules.daily_loss_log_threshold:
            daily_stop_active = True
            daily_stop_triggered_now = True
            daily_stop_trigger_count += 1
            daily_stop_dates.add(utc_date)

        total_stop_triggered_now = False
        if not total_stop_active and net_drawdown <= rules.total_drawdown_stop:
            total_stop_active = True
            total_stop_triggered = True
            total_stop_triggered_now = True
            if first_total_stop_trigger is None:
                first_total_stop_trigger = timestamp

        rows.append(
            {
                "time": timestamp.isoformat(),
                "row_number": row_number,
                "p_up": probability,
                "target_dir": int(row.target_dir),
                "target_ret_fwd": float(row.target_ret_fwd),
                "signal": signal,
                "previous_position": previous_position,
                "position": int(next_position),
                "position_before": previous_position,
                "position_after": int(next_position),
                "successful_entries_today": int(successful_entries_today),
                "successful_entry_event": int(successful_entry_event),
                "normal_exit_event": int(normal_exit_event),
                "turnover": turnover_units,
                "turnover_units": turnover_units,
                "gross_log_return": gross_log_return,
                "cost_log_return": cost_log_return,
                "net_log_return": net_log_return,
                "gross_equity": gross_equity,
                "net_equity": net_equity,
                "net_drawdown": net_drawdown,
                "drawdown": net_drawdown,
                "daily_log_return": float(daily_log_return),
                "gap_from_previous_prediction": gap_from_previous,
                "daily_stop_active_after": bool(daily_stop_active),
                "total_stop_active_after": bool(total_stop_active),
                "daily_stop_triggered": bool(daily_stop_triggered_now),
                "total_stop_triggered": bool(total_stop_triggered_now),
                "change_reason": _reason(
                    previous_position=previous_position,
                    next_position=int(next_position),
                    forced_reason=forced_reason,
                    blocked_reason=blocked_reason,
                ),
            }
        )
        current_position = int(next_position)
        last_time = timestamp

    log = pd.DataFrame(rows)
    metrics = compute_replay_metrics(
        log,
        cost_bps=float(cost_bps),
        policy_change_events=successful_entry_count + normal_exit_count,
        gap_exit_events=gap_exit_events,
        daily_stop_exit_events=daily_stop_exit_events,
        daily_stop_trigger_count=daily_stop_trigger_count,
        total_stop_exit_events=total_stop_exit_events,
        total_stop_triggered=total_stop_triggered,
        first_total_stop_trigger=first_total_stop_trigger,
        daily_stop_dates=tuple(sorted(daily_stop_dates)),
    )
    diagnostics = compute_model_b_diagnostics(
        log,
        rules=rules,
        successful_entry_count=successful_entry_count,
        normal_exit_count=normal_exit_count,
        daily_entry_cap_block_count=daily_entry_cap_block_count,
        gap_block_count=gap_block_count,
        gap_exit_events=gap_exit_events,
        daily_stop_exit_events=daily_stop_exit_events,
        total_stop_exit_events=total_stop_exit_events,
        entries_by_day=entries_by_day,
        daily_net_log_by_day=daily_net_log_by_day,
    )
    return log, metrics, diagnostics


def compute_model_b_diagnostics(
    log: pd.DataFrame,
    *,
    rules: ModelBOverlayRules,
    successful_entry_count: int,
    normal_exit_count: int,
    daily_entry_cap_block_count: int,
    gap_block_count: int,
    gap_exit_events: int,
    daily_stop_exit_events: int,
    total_stop_exit_events: int,
    entries_by_day: Mapping[str, int],
    daily_net_log_by_day: Mapping[str, float],
) -> ModelBDiagnostics:
    if log.empty:
        raise TradingReplayError("Model B replay log is empty")
    short_position_count = int(np.count_nonzero(log["position"].to_numpy(dtype=np.int8) < 0))
    entry_rows = log["successful_entry_event"].to_numpy(dtype=np.int8) == 1
    entry_probabilities = log.loc[entry_rows, "p_up"].to_numpy(dtype=np.float64)
    entry_below_threshold_count = int(np.count_nonzero(entry_probabilities < rules.entry_threshold))
    active_below_exit_threshold_count = int(
        np.count_nonzero(
            (log["position"].to_numpy(dtype=np.int8) == 1)
            & (log["p_up"].to_numpy(dtype=np.float64) < rules.exit_threshold)
        )
    )
    max_entries = max(entries_by_day.values(), default=0)
    daily_simple_returns = [float(np.expm1(value)) for value in daily_net_log_by_day.values()]
    worst_daily_net_return = min(daily_simple_returns, default=0.0)
    transaction_cost_log_sum = float(log["cost_log_return"].sum())
    final_gross = float(log["gross_equity"].iloc[-1])
    final_net = float(log["net_equity"].iloc[-1])
    transaction_cost_burden = final_gross - final_net

    failures: list[str] = []
    if short_position_count != 0:
        failures.append(f"short_position_count={short_position_count}")
    if entry_below_threshold_count != 0:
        failures.append(f"entry_below_threshold_count={entry_below_threshold_count}")
    if active_below_exit_threshold_count != 0:
        failures.append(f"active_below_exit_threshold_count={active_below_exit_threshold_count}")
    if max_entries > rules.max_successful_entries_per_day:
        failures.append(
            f"max_successful_entries_in_utc_day={max_entries} exceeds {rules.max_successful_entries_per_day}"
        )

    return ModelBDiagnostics(
        successful_entry_count=int(successful_entry_count),
        normal_exit_count=int(normal_exit_count),
        daily_entry_cap_block_count=int(daily_entry_cap_block_count),
        gap_block_count=int(gap_block_count),
        gap_exit_events=int(gap_exit_events),
        daily_stop_exit_events=int(daily_stop_exit_events),
        total_stop_exit_events=int(total_stop_exit_events),
        short_position_count=short_position_count,
        entry_below_threshold_count=entry_below_threshold_count,
        active_below_exit_threshold_count=active_below_exit_threshold_count,
        max_successful_entries_in_utc_day=int(max_entries),
        worst_daily_net_return=worst_daily_net_return,
        transaction_cost_log_sum=transaction_cost_log_sum,
        transaction_cost_burden=transaction_cost_burden,
        invariant_passed=not failures,
        invariant_failures=tuple(failures),
    )


def model_b_metrics_row(
    *,
    partition: str,
    cost_bps: float,
    metrics: ReplayMetrics,
    diagnostics: ModelBDiagnostics,
) -> dict[str, Any]:
    return {
        "partition": partition,
        **asdict(metrics),
        **asdict(diagnostics),
        "cost_bps": float(cost_bps),
    }


def compare_model_b_to_model_a(
    *,
    partition: str,
    cost_bps: float,
    model_a_metrics: Mapping[str, Any],
    model_b_metrics: ReplayMetrics,
    model_b_diagnostics: ModelBDiagnostics,
) -> dict[str, Any]:
    """Return diagnostic deltas, not pass/fail profitability criteria."""

    def get_float(name: str, default: float = 0.0) -> float:
        value = model_a_metrics.get(name, default)
        if value is None:
            return default
        return float(value)

    a_net = get_float("net_total_return")
    a_gross = get_float("gross_total_return")
    a_turnover = get_float("turnover_units")
    a_active = get_float("active_bar_rate")
    a_mdd = get_float("max_drawdown")
    a_cost_burden = get_float("final_gross_equity") - get_float("final_net_equity")
    b_cost_burden = model_b_diagnostics.transaction_cost_burden

    return {
        "partition": partition,
        "cost_bps": float(cost_bps),
        "model_a_net_total_return": a_net,
        "model_b_net_total_return": float(model_b_metrics.net_total_return),
        "delta_net_total_return_b_minus_a": float(model_b_metrics.net_total_return - a_net),
        "model_a_gross_total_return": a_gross,
        "model_b_gross_total_return": float(model_b_metrics.gross_total_return),
        "delta_gross_total_return_b_minus_a": float(model_b_metrics.gross_total_return - a_gross),
        "model_a_max_drawdown": a_mdd,
        "model_b_max_drawdown": float(model_b_metrics.max_drawdown),
        "drawdown_reduction_positive_means_less_negative": float(model_b_metrics.max_drawdown - a_mdd),
        "model_a_turnover_units": a_turnover,
        "model_b_turnover_units": float(model_b_metrics.turnover_units),
        "turnover_reduction_units": float(a_turnover - model_b_metrics.turnover_units),
        "model_a_active_bar_rate": a_active,
        "model_b_active_bar_rate": float(model_b_metrics.active_bar_rate),
        "active_rate_reduction": float(a_active - model_b_metrics.active_bar_rate),
        "model_a_transaction_cost_burden": a_cost_burden,
        "model_b_transaction_cost_burden": b_cost_burden,
        "transaction_cost_burden_reduction": float(a_cost_burden - b_cost_burden),
        "model_b_successful_entry_count": int(model_b_diagnostics.successful_entry_count),
        "model_b_short_position_count": int(model_b_diagnostics.short_position_count),
        "model_b_invariant_passed": bool(model_b_diagnostics.invariant_passed),
    }


def diagnostics_to_dict(item: ModelBDiagnostics) -> dict[str, Any]:
    return asdict(item)
