"""Model B minimum-hold diagnostic overlay for Stage 2 Step 2C.

This module does not tune thresholds or retrain a model.  It takes the already
frozen Model B V2 long-only rules and applies one pre-declared candidate
execution refinement: a three completed-M15-bar minimum hold before normal
probability-based exits.  Gap exits, daily loss stops, and total drawdown stops
remain allowed to override the minimum hold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from capstone_trading.errors import TradingReplayError

M15_DELTA = pd.Timedelta(minutes=15)
from capstone_trading.evaluation.model_b_replay import ModelBOverlayRules
from capstone_trading.evaluation.trading_replay import (
    ReplayMetrics,
    compute_replay_metrics,
    validate_prediction_frame,
)


@dataclass(frozen=True)
class ModelBMinHoldRules:
    """Model B candidate rules with a fixed normal-exit minimum hold."""

    base_rules: ModelBOverlayRules
    minimum_hold_bars: int = 3

    @property
    def entry_threshold(self) -> float:
        return self.base_rules.entry_threshold

    @property
    def exit_threshold(self) -> float:
        return self.base_rules.exit_threshold

    @property
    def max_successful_entries_per_day(self) -> int:
        return self.base_rules.max_successful_entries_per_day

    @property
    def daily_loss_log_threshold(self) -> float:
        return self.base_rules.daily_loss_log_threshold

    @property
    def total_drawdown_stop(self) -> float:
        return self.base_rules.total_drawdown_stop


@dataclass(frozen=True)
class ModelBMinHoldDiagnostics:
    """Diagnostic counters and invariant results for Model B minimum-hold candidate."""

    successful_entry_count: int
    normal_exit_count: int
    min_hold_blocked_exit_count: int
    daily_entry_cap_block_count: int
    gap_block_count: int
    gap_exit_events: int
    daily_stop_exit_events: int
    total_stop_exit_events: int
    short_position_count: int
    entry_below_threshold_count: int
    active_below_exit_threshold_count: int
    active_below_exit_threshold_after_eligible_count: int
    max_successful_entries_in_utc_day: int
    max_hold_bars_completed: int
    worst_daily_net_return: float
    transaction_cost_log_sum: float
    transaction_cost_burden: float
    invariant_passed: bool
    invariant_failures: tuple[str, ...]


def create_min_hold_rules(
    base_rules: ModelBOverlayRules,
    *,
    minimum_hold_bars: int = 3,
) -> ModelBMinHoldRules:
    """Create the fixed Stage 2 Step 2C Model B minimum-hold candidate."""

    if int(minimum_hold_bars) < 0:
        raise TradingReplayError("minimum_hold_bars must be non-negative")
    return ModelBMinHoldRules(base_rules=base_rules, minimum_hold_bars=int(minimum_hold_bars))


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


def replay_model_b_min_hold(
    predictions: pd.DataFrame,
    rules: ModelBMinHoldRules,
    *,
    cost_bps: float,
) -> tuple[pd.DataFrame, ReplayMetrics, ModelBMinHoldDiagnostics]:
    """Replay Model B with a fixed minimum hold before normal exits.

    Semantics:
    - Entry remains unchanged: enter long from flat when p_up >= entry_threshold.
    - Normal probability exit is blocked until the long has been active for at
      least ``minimum_hold_bars`` completed M15 bars.  The entry decision bar is
      counted as the first held bar.
    - Gap exits, daily loss stops, and total drawdown stops override the minimum
      hold because they are safety controls rather than normal exits.
    - The strategy remains long-only and keeps the same one-entry-per-day cap.
    """

    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if rules.minimum_hold_bars < 0:
        raise TradingReplayError("minimum_hold_bars must be non-negative")

    frame = validate_prediction_frame(predictions)
    one_way_cost = float(cost_bps) / 10000.0

    current_position = 0
    hold_bars_completed = 0
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
    min_hold_blocked_exit_count = 0
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
        previous_hold_bars = int(hold_bars_completed)
        minimum_hold_eligible = bool(previous_position == 1 and previous_hold_bars >= rules.minimum_hold_bars)
        next_position = previous_position
        forced_reason = ""
        blocked_reason = ""
        signal = 1 if probability >= rules.entry_threshold else 0
        successful_entry_event = 0
        normal_exit_event = 0
        min_hold_blocked_exit_event = 0

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
            hold_bars_completed = 0
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
            if probability >= rules.exit_threshold:
                next_position = 1
            elif not minimum_hold_eligible:
                next_position = 1
                blocked_reason = "minimum_hold_exit_block"
                min_hold_blocked_exit_count += 1
                min_hold_blocked_exit_event = 1
            else:
                next_position = 0
                normal_exit_count += 1
                normal_exit_event = 1

        if next_position == 1:
            if previous_position == 1:
                next_hold_bars = previous_hold_bars + 1
            else:
                next_hold_bars = 1
        else:
            next_hold_bars = 0

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
                "hold_bars_completed_before": int(previous_hold_bars),
                "hold_bars_completed_after": int(next_hold_bars),
                "minimum_hold_bars": int(rules.minimum_hold_bars),
                "minimum_hold_eligible_before": bool(minimum_hold_eligible),
                "min_hold_blocked_exit_event": int(min_hold_blocked_exit_event),
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
        hold_bars_completed = int(next_hold_bars)
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
    diagnostics = compute_model_b_min_hold_diagnostics(
        log,
        rules=rules,
        successful_entry_count=successful_entry_count,
        normal_exit_count=normal_exit_count,
        min_hold_blocked_exit_count=min_hold_blocked_exit_count,
        daily_entry_cap_block_count=daily_entry_cap_block_count,
        gap_block_count=gap_block_count,
        gap_exit_events=gap_exit_events,
        daily_stop_exit_events=daily_stop_exit_events,
        total_stop_exit_events=total_stop_exit_events,
        entries_by_day=entries_by_day,
        daily_net_log_by_day=daily_net_log_by_day,
    )
    return log, metrics, diagnostics


def compute_model_b_min_hold_diagnostics(
    log: pd.DataFrame,
    *,
    rules: ModelBMinHoldRules,
    successful_entry_count: int,
    normal_exit_count: int,
    min_hold_blocked_exit_count: int,
    daily_entry_cap_block_count: int,
    gap_block_count: int,
    gap_exit_events: int,
    daily_stop_exit_events: int,
    total_stop_exit_events: int,
    entries_by_day: Mapping[str, int],
    daily_net_log_by_day: Mapping[str, float],
) -> ModelBMinHoldDiagnostics:
    if log.empty:
        raise TradingReplayError("Model B minimum-hold replay log is empty")

    position = log["position"].to_numpy(dtype=np.int8)
    probability = log["p_up"].to_numpy(dtype=np.float64)
    short_position_count = int(np.count_nonzero(position < 0))
    entry_rows = log["successful_entry_event"].to_numpy(dtype=np.int8) == 1
    entry_probabilities = log.loc[entry_rows, "p_up"].to_numpy(dtype=np.float64)
    entry_below_threshold_count = int(np.count_nonzero(entry_probabilities < rules.entry_threshold))
    active_below_exit_threshold_count = int(
        np.count_nonzero((position == 1) & (probability < rules.exit_threshold))
    )
    eligible_before = log["minimum_hold_eligible_before"].to_numpy(dtype=bool)
    active_below_exit_threshold_after_eligible_count = int(
        np.count_nonzero((position == 1) & (probability < rules.exit_threshold) & eligible_before)
    )
    max_entries = max(entries_by_day.values(), default=0)
    max_hold_bars_completed = int(log["hold_bars_completed_after"].max())
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
    if active_below_exit_threshold_after_eligible_count != 0:
        failures.append(
            "active_below_exit_threshold_after_eligible_count="
            f"{active_below_exit_threshold_after_eligible_count}"
        )
    if max_entries > rules.max_successful_entries_per_day:
        failures.append(
            f"max_successful_entries_in_utc_day={max_entries} exceeds "
            f"{rules.max_successful_entries_per_day}"
        )

    return ModelBMinHoldDiagnostics(
        successful_entry_count=int(successful_entry_count),
        normal_exit_count=int(normal_exit_count),
        min_hold_blocked_exit_count=int(min_hold_blocked_exit_count),
        daily_entry_cap_block_count=int(daily_entry_cap_block_count),
        gap_block_count=int(gap_block_count),
        gap_exit_events=int(gap_exit_events),
        daily_stop_exit_events=int(daily_stop_exit_events),
        total_stop_exit_events=int(total_stop_exit_events),
        short_position_count=short_position_count,
        entry_below_threshold_count=entry_below_threshold_count,
        active_below_exit_threshold_count=active_below_exit_threshold_count,
        active_below_exit_threshold_after_eligible_count=active_below_exit_threshold_after_eligible_count,
        max_successful_entries_in_utc_day=int(max_entries),
        max_hold_bars_completed=max_hold_bars_completed,
        worst_daily_net_return=worst_daily_net_return,
        transaction_cost_log_sum=transaction_cost_log_sum,
        transaction_cost_burden=transaction_cost_burden,
        invariant_passed=not failures,
        invariant_failures=tuple(failures),
    )


def model_b_min_hold_metrics_row(
    *,
    variant: str,
    partition: str,
    cost_bps: float,
    metrics: ReplayMetrics,
    diagnostics: ModelBMinHoldDiagnostics,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "partition": partition,
        "cost_bps": float(cost_bps),
        **asdict(metrics),
        **asdict(diagnostics),
    }


def diagnostics_to_dict(item: ModelBMinHoldDiagnostics) -> dict[str, Any]:
    return asdict(item)
