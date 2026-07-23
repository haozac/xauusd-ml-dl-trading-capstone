"""Frozen Model A historical trading replay for Stage 1 Step 4.

The functions in this module are intentionally independent from the saved
Notebook 7 strategy bar log.  They consume only prediction timestamps,
probabilities and forward returns, apply the frozen overlay and risk rules, and
then compare the generated trading path with saved Notebook 7 reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from capstone_trading.data.canonical_bars import M15_DELTA
from capstone_trading.errors import TradingReplayError
from capstone_trading.policy.position_transition import resolve_position_transition

DEFAULT_COSTS_BPS: tuple[float, ...] = (0.0, 0.5, 1.0)
POSITION_VALUES: tuple[int, ...] = (-1, 0, 1)


@dataclass(frozen=True)
class ModelAOverlayRules:
    long_threshold: float
    short_threshold: float
    minimum_hold_bars: int
    max_policy_changes_per_day: int
    daily_loss_log_threshold: float
    total_drawdown_stop: float
    count_gap_exits_against_cap: bool = False
    count_risk_exits_against_cap: bool = False
    reversal_policy_event_units: int = 1
    allow_risk_reducing_exit_when_capped: bool = False


@dataclass(frozen=True)
class ReplayMetrics:
    cost_bps: float
    row_count: int
    active_bar_count: int
    active_bar_rate: float
    turnover_units: float
    round_turn_equivalent_trades: float
    policy_change_events: int
    gap_exit_events: int
    daily_stop_exit_events: int
    daily_stop_trigger_count: int
    total_stop_exit_events: int
    total_stop_triggered: bool
    first_total_stop_trigger_utc: str | None
    final_gross_equity: float
    final_net_equity: float
    gross_total_return: float
    net_total_return: float
    max_drawdown: float
    gross_log_return_sum: float
    net_log_return_sum: float
    daily_stop_dates: tuple[str, ...]


@dataclass(frozen=True)
class MetricComparison:
    reference_column: str
    generated_field: str
    expected_value: float
    actual_value: float
    absolute_difference: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class BarLogComparison:
    column: str
    compared_rows: int
    maximum_absolute_difference: float
    mismatch_count: int
    tolerance: float
    passed: bool


def overlay_rules_from_config(config_raw: Mapping[str, Any]) -> ModelAOverlayRules:
    overlay = config_raw.get("overlay")
    risk = config_raw.get("historical_risk_semantics")
    if not isinstance(overlay, Mapping) or not isinstance(risk, Mapping):
        raise TradingReplayError("Frozen Model A configuration is missing overlay or risk semantics")
    daily_loss_threshold = risk.get(
        "daily_loss_stop_log_threshold",
        risk.get("daily_loss_log_threshold"),
    )
    try:
        rules = ModelAOverlayRules(
            long_threshold=float(overlay["long_when_p_up_gte"]),
            short_threshold=float(overlay["short_when_p_up_lte"]),
            minimum_hold_bars=int(overlay["minimum_hold_eligible_bars"]),
            max_policy_changes_per_day=int(
                overlay["maximum_overlay_position_change_events_per_utc_day"]
            ),
            daily_loss_log_threshold=float(daily_loss_threshold),
            total_drawdown_stop=float(risk["total_drawdown_stop"]),
            count_gap_exits_against_cap=bool(
                overlay.get("gap_exits_count_against_daily_change_cap", False)
            ),
            count_risk_exits_against_cap=bool(
                overlay.get("risk_exits_count_against_daily_change_cap", False)
            ),
            reversal_policy_event_units=int(overlay.get("reversal_policy_event_units", 1)),
            allow_risk_reducing_exit_when_capped=bool(
                overlay.get("allow_risk_reducing_exit_when_capped", False)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TradingReplayError(f"Missing frozen overlay/risk field: {exc}") from exc
    if not 0.0 <= rules.short_threshold < rules.long_threshold <= 1.0:
        raise TradingReplayError("Invalid Model A threshold ordering")
    if rules.minimum_hold_bars < 1:
        raise TradingReplayError("minimum_hold_bars must be positive")
    if rules.max_policy_changes_per_day < 0:
        raise TradingReplayError("daily change cap must be non-negative")
    return rules


def validate_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "p_up", "target_ret_fwd", "target_dir"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise TradingReplayError(f"Prediction frame is missing columns: {missing}")
    frame = predictions.loc[:, ["time", "p_up", "target_ret_fwd", "target_dir"]].copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    if frame["time"].isna().any():
        raise TradingReplayError("Prediction timestamps contain NaT")
    if frame["time"].duplicated().any():
        raise TradingReplayError("Prediction timestamps contain duplicates")
    if not frame["time"].is_monotonic_increasing:
        raise TradingReplayError("Prediction timestamps must be chronological")
    for column in ("p_up", "target_ret_fwd"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            raise TradingReplayError(f"Prediction column {column} contains NaN or infinite values")
    invalid_probability = (frame["p_up"] < 0.0) | (frame["p_up"] > 1.0)
    if bool(invalid_probability.any()):
        first = int(np.flatnonzero(invalid_probability.to_numpy())[0])
        raise TradingReplayError(f"p_up outside [0, 1] at row {first}")
    frame["target_dir"] = pd.to_numeric(frame["target_dir"], errors="raise").astype("int8")
    return frame


def _signal_from_probability(probability: float, rules: ModelAOverlayRules) -> int:
    if probability >= rules.long_threshold:
        return 1
    if probability <= rules.short_threshold:
        return -1
    return 0



def _position_change_reason(
    *,
    previous_position: int,
    next_position: int,
    forced_reason: str,
    blocked_reason: str,
) -> str:
    if blocked_reason:
        return blocked_reason
    if previous_position == next_position:
        return "hold"
    if forced_reason:
        return forced_reason
    if previous_position == 0 and next_position != 0:
        return "policy_entry"
    if previous_position != 0 and next_position == 0:
        return "policy_exit"
    if previous_position != 0 and next_position != 0 and previous_position != next_position:
        return "policy_reversal"
    return "policy_change"


def replay_model_a(
    predictions: pd.DataFrame,
    rules: ModelAOverlayRules,
    *,
    cost_bps: float,
) -> tuple[pd.DataFrame, ReplayMetrics]:
    """Replay the frozen Model A overlay and historical risk semantics.

    Position decisions are made at each eligible prediction timestamp and applied
    to that row's forward log return.  Daily and total drawdown stops are triggered
    after a row's net return is realised and take effect from the next eligible
    decision, mirroring the audited Notebook 7 semantics.
    """

    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    frame = validate_prediction_frame(predictions)
    one_way_cost = float(cost_bps) / 10000.0

    current_position = 0
    hold_bars = 0
    # Notebook 7 treats min_hold_bars=3 as three completed bars after entry
    # before another position change is allowed.  It also applies a symmetric
    # flat-state cooldown after normal policy/risk exits.  Forced gap exits are
    # exempt because the gap bar itself is non-tradable and the next eligible
    # bar may enter again if the signal and daily cap allow it.
    flat_bars_since_exit = 10**9
    current_day: pd.Timestamp | None = None
    policy_changes_today = 0
    daily_log_return = 0.0
    daily_stop_active = False
    total_stop_active = False
    total_stop_triggered = False
    first_total_stop_trigger: pd.Timestamp | None = None
    cumulative_gross_log = 0.0
    cumulative_net_log = 0.0
    running_peak_net_equity = 1.0
    last_time: pd.Timestamp | None = None
    daily_stop_dates: set[str] = set()

    rows: list[dict[str, Any]] = []
    policy_change_events = 0
    gap_exit_events = 0
    daily_stop_exit_events = 0
    daily_stop_trigger_count = 0
    total_stop_exit_events = 0

    for row_number, row in enumerate(frame.itertuples(index=False)):
        timestamp = pd.Timestamp(row.time).tz_convert("UTC")
        utc_date = timestamp.date().isoformat()
        day_key = pd.Timestamp(timestamp.date(), tz="UTC")
        if current_day is None or day_key != current_day:
            current_day = day_key
            policy_changes_today = 0
            daily_log_return = 0.0
            daily_stop_active = False

        gap_from_previous = bool(
            last_time is not None and timestamp - last_time != M15_DELTA
        )
        probability = float(row.p_up)
        signal = _signal_from_probability(probability, rules)
        previous_position = int(current_position)
        desired_position = int(signal)
        next_position = previous_position
        forced_reason = ""
        blocked_reason = ""
        policy_event_units = 0

        if total_stop_active:
            desired_position = 0
            next_position = 0
            forced_reason = "total_drawdown_stop"
            if previous_position != 0:
                total_stop_exit_events += 1
        elif daily_stop_active:
            desired_position = 0
            next_position = 0
            forced_reason = "daily_loss_stop"
            if previous_position != 0:
                daily_stop_exit_events += 1
        elif gap_from_previous:
            # A non-contiguous prediction timestamp is never tradable in the
            # frozen Notebook 7 strategy path.  If a position is open, the gap
            # row performs a forced exit.  If already flat, the row remains flat
            # even when the raw signal is long or short.
            desired_position = 0
            next_position = 0
            if previous_position != 0:
                forced_reason = "gap_exit"
                gap_exit_events += 1
            else:
                blocked_reason = "gap_block"
        else:
            if desired_position != previous_position:
                active_min_hold_blocked = (
                    previous_position != 0
                    and hold_bars <= rules.minimum_hold_bars
                )
                flat_cooldown_blocked = (
                    previous_position == 0
                    and desired_position != 0
                    and flat_bars_since_exit < rules.minimum_hold_bars
                )
                if active_min_hold_blocked or flat_cooldown_blocked:
                    next_position = previous_position
                    blocked_reason = "minimum_hold_active"
                else:
                    resolution = resolve_position_transition(
                        current_position=previous_position,
                        desired_position=desired_position,
                        policy_changes_today=policy_changes_today,
                        max_policy_changes_per_day=rules.max_policy_changes_per_day,
                        reversal_policy_event_units=(
                            rules.reversal_policy_event_units
                        ),
                        allow_risk_reducing_exit_when_capped=(
                            rules.allow_risk_reducing_exit_when_capped
                        ),
                    )
                    next_position = int(resolution.effective_target_position)
                    policy_event_units = int(resolution.consumed_policy_units)
                    if next_position != previous_position:
                        policy_changes_today += policy_event_units
                        policy_change_events += policy_event_units
                        if resolution.close_only_reversal:
                            blocked_reason = (
                                "daily_change_cap_close_only_reversal"
                            )
                        elif resolution.cap_reached and resolution.exit_allowed:
                            blocked_reason = "daily_change_cap_exit_allowed"
                    else:
                        blocked_reason = "daily_change_cap_active"
            else:
                next_position = previous_position

        turnover_units = float(abs(next_position - previous_position))
        gross_log_return = float(next_position) * float(row.target_ret_fwd)
        cost_log_return = turnover_units * one_way_cost
        net_log_return = gross_log_return - cost_log_return
        cumulative_gross_log += gross_log_return
        cumulative_net_log += net_log_return
        daily_log_return += net_log_return
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

        if next_position == 0:
            next_hold_bars = 0
            if previous_position == 0:
                next_flat_bars_since_exit = min(flat_bars_since_exit + 1, 10**9)
            elif forced_reason == "gap_exit":
                next_flat_bars_since_exit = 10**9
            else:
                next_flat_bars_since_exit = 0
        elif next_position == previous_position:
            next_hold_bars = hold_bars + 1 if previous_position != 0 else 1
            next_flat_bars_since_exit = 10**9
        else:
            next_hold_bars = 1
            next_flat_bars_since_exit = 10**9

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
                "hold_bars_before": int(hold_bars),
                "hold_bars_after": int(next_hold_bars),
                "flat_bars_since_exit_before": int(flat_bars_since_exit),
                "flat_bars_since_exit_after": int(next_flat_bars_since_exit),
                "policy_change_events_today": int(policy_changes_today),
                "policy_event_units": int(policy_event_units),
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
                "daily_stop_triggered": daily_stop_triggered_now,
                "total_stop_triggered": total_stop_triggered_now,
                "change_reason": _position_change_reason(
                    previous_position=previous_position,
                    next_position=int(next_position),
                    forced_reason=forced_reason,
                    blocked_reason=blocked_reason,
                ),
            }
        )
        current_position = int(next_position)
        hold_bars = int(next_hold_bars)
        flat_bars_since_exit = int(next_flat_bars_since_exit)
        last_time = timestamp

    log = pd.DataFrame(rows)
    metrics = compute_replay_metrics(
        log,
        cost_bps=float(cost_bps),
        policy_change_events=policy_change_events,
        gap_exit_events=gap_exit_events,
        daily_stop_exit_events=daily_stop_exit_events,
        daily_stop_trigger_count=daily_stop_trigger_count,
        total_stop_exit_events=total_stop_exit_events,
        total_stop_triggered=total_stop_triggered,
        first_total_stop_trigger=first_total_stop_trigger,
        daily_stop_dates=tuple(sorted(daily_stop_dates)),
    )
    return log, metrics


def compute_replay_metrics(
    log: pd.DataFrame,
    *,
    cost_bps: float,
    policy_change_events: int,
    gap_exit_events: int,
    daily_stop_exit_events: int,
    daily_stop_trigger_count: int,
    total_stop_exit_events: int,
    total_stop_triggered: bool,
    first_total_stop_trigger: pd.Timestamp | None,
    daily_stop_dates: Sequence[str],
) -> ReplayMetrics:
    if log.empty:
        raise TradingReplayError("Replay log is empty")
    active = log["position"].to_numpy(dtype=np.int8) != 0
    turnover = float(log["turnover"].sum())
    final_gross_equity = float(log["gross_equity"].iloc[-1])
    final_net_equity = float(log["net_equity"].iloc[-1])
    return ReplayMetrics(
        cost_bps=float(cost_bps),
        row_count=int(len(log)),
        active_bar_count=int(np.count_nonzero(active)),
        active_bar_rate=float(np.mean(active)),
        turnover_units=turnover,
        round_turn_equivalent_trades=turnover / 2.0,
        policy_change_events=int(policy_change_events),
        gap_exit_events=int(gap_exit_events),
        daily_stop_exit_events=int(daily_stop_exit_events),
        daily_stop_trigger_count=int(daily_stop_trigger_count),
        total_stop_exit_events=int(total_stop_exit_events),
        total_stop_triggered=bool(total_stop_triggered),
        first_total_stop_trigger_utc=(
            first_total_stop_trigger.isoformat() if first_total_stop_trigger is not None else None
        ),
        final_gross_equity=final_gross_equity,
        final_net_equity=final_net_equity,
        gross_total_return=final_gross_equity - 1.0,
        net_total_return=final_net_equity - 1.0,
        max_drawdown=float(log["net_drawdown"].min()),
        gross_log_return_sum=float(log["gross_log_return"].sum()),
        net_log_return_sum=float(log["net_log_return"].sum()),
        daily_stop_dates=tuple(daily_stop_dates),
    )


def load_strategy_bar_log(path: Any) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise TradingReplayError("Saved strategy bar log is empty")
    time_column = find_column(frame, ("time", "timestamp", "time_utc", "bar_time"), required=True)
    assert time_column is not None
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    if frame[time_column].duplicated().any() or not frame[time_column].is_monotonic_increasing:
        raise TradingReplayError("Saved strategy bar log timestamps are duplicated or unordered")
    return frame


def find_column(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = False,
) -> str | None:
    lower_to_actual = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        found = lower_to_actual.get(alias.lower())
        if found is not None:
            return found
    if required:
        raise TradingReplayError(f"Missing required column; tried aliases {tuple(aliases)}")
    return None


BAR_LOG_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "target_ret_fwd": ("target_ret_fwd", "forward_return", "fwd_return", "ret_fwd"),
    "target_dir": ("target_dir", "target_direction", "y_true"),
    "position": ("position", "held_position", "strategy_position", "position_after"),
    "turnover": ("turnover", "turnover_units", "position_turnover"),
    "gross_log_return": ("gross_log_return", "gross_ret", "gross_return", "gross_strategy_log_return"),
    "net_log_return": ("net_log_return", "net_ret", "net_return", "net_strategy_log_return"),
    "cost_log_return": ("cost_log_return", "cost", "transaction_cost", "turnover_cost"),
    "gross_equity": ("gross_equity", "equity_gross", "gross_curve"),
    "net_equity": ("net_equity", "equity", "equity_net", "strategy_equity", "net_curve"),
    "net_drawdown": ("net_drawdown", "drawdown", "dd", "strategy_drawdown"),
}


def compare_replayed_bar_log(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    tolerance: float,
) -> tuple[BarLogComparison, ...]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    generated_time = pd.DatetimeIndex(pd.to_datetime(generated["time"], utc=True))
    reference_time_column = find_column(
        reference,
        ("time", "timestamp", "time_utc", "bar_time"),
        required=True,
    )
    assert reference_time_column is not None
    reference_time = pd.DatetimeIndex(pd.to_datetime(reference[reference_time_column], utc=True))
    if len(generated_time) != len(reference_time) or not np.array_equal(
        generated_time.asi8,
        reference_time.asi8,
    ):
        raise TradingReplayError(
            f"Strategy bar-log timestamp alignment failed: generated={len(generated_time)}, "
            f"reference={len(reference_time)}"
        )

    comparisons: list[BarLogComparison] = []
    for generated_column, aliases in BAR_LOG_COLUMN_ALIASES.items():
        reference_column = find_column(reference, aliases)
        if reference_column is None or generated_column not in generated.columns:
            continue
        expected = pd.to_numeric(reference[reference_column], errors="raise").to_numpy(dtype=np.float64)
        actual = pd.to_numeric(generated[generated_column], errors="raise").to_numpy(dtype=np.float64)
        difference = np.abs(actual - expected)
        mismatches = int(np.count_nonzero(difference > tolerance))
        comparisons.append(
            BarLogComparison(
                column=generated_column,
                compared_rows=int(len(difference)),
                maximum_absolute_difference=float(difference.max(initial=0.0)),
                mismatch_count=mismatches,
                tolerance=float(tolerance),
                passed=mismatches == 0,
            )
        )
    required = {"position", "turnover", "gross_log_return", "net_log_return", "net_equity"}
    compared = {item.column for item in comparisons}
    missing_required = sorted(required - compared)
    if missing_required:
        raise TradingReplayError(
            "Saved strategy bar log does not expose the required comparable columns: "
            f"{missing_required}. Available reference columns: {list(reference.columns)}"
        )
    failed = [item for item in comparisons if not item.passed]
    if failed:
        detail = ", ".join(
            f"{item.column}: max_diff={item.maximum_absolute_difference:.3e}, mismatches={item.mismatch_count}"
            for item in failed[:8]
        )
        raise TradingReplayError(f"Model A bar-log parity failed: {detail}")
    return tuple(comparisons)


METRIC_FIELD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "cost_bps": ("cost_bps", "cost", "one_way_cost_bps", "transaction_cost_bps"),
    "net_total_return": ("net_total_return", "total_return", "net_return", "final_net_return", "total_return_net", "net_total_ret", "return_net"),
    "gross_total_return": ("gross_total_return", "gross_return", "final_gross_return", "total_return_gross", "gross_total_ret", "return_gross"),
    "final_net_equity": ("final_net_equity", "net_equity", "ending_net_equity"),
    "final_gross_equity": ("final_gross_equity", "gross_equity", "ending_gross_equity"),
    "max_drawdown": ("max_drawdown", "max_dd", "minimum_drawdown", "net_max_drawdown", "max_net_drawdown", "mdd"),
    "turnover_units": ("turnover_units", "turnover", "total_turnover", "turnover_unit_count"),
    "round_turn_equivalent_trades": (
        "round_turn_equivalent_trades",
        "round_turn_trades",
        "trades",
        "trade_count",
    ),
    "active_bar_rate": ("active_bar_rate", "active_rate", "exposure_rate", "market_exposure", "active_fraction"),
    "total_stop_triggered": ("total_stop_triggered", "hit_total_stop", "total_drawdown_stop_triggered"),
}


def _normalise_metric_table(reference: pd.DataFrame) -> pd.DataFrame:
    if reference.empty:
        raise TradingReplayError("Reference metrics table is empty")
    frame = reference.copy()
    cost_column = find_column(frame, METRIC_FIELD_ALIASES["cost_bps"])
    if cost_column is None:
        raise TradingReplayError(
            f"Reference metrics table has no cost column; available columns: {list(frame.columns)}"
        )
    frame[cost_column] = pd.to_numeric(frame[cost_column], errors="raise")
    # Some notebooks keep validation and holdout rows in the same table.  When a
    # split/evaluation column exists, select the final-holdout rows.
    for split_column in ("split", "partition", "period", "evaluation", "dataset"):
        actual = find_column(frame, (split_column,))
        if actual is None:
            continue
        lower_values = frame[actual].astype(str).str.lower()
        mask = lower_values.str.contains("holdout") | lower_values.str.contains("final")
        if mask.any():
            frame = frame.loc[mask].copy()
            break
    return frame


def compare_metrics_to_reference(
    metrics_by_cost: Mapping[float, ReplayMetrics],
    reference_metrics: pd.DataFrame,
    *,
    tolerance: float,
) -> tuple[MetricComparison, ...]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    frame = _normalise_metric_table(reference_metrics)
    cost_column = find_column(frame, METRIC_FIELD_ALIASES["cost_bps"], required=True)
    assert cost_column is not None
    comparisons: list[MetricComparison] = []
    for cost_bps, metrics in metrics_by_cost.items():
        cost_values = pd.to_numeric(frame[cost_column], errors="raise").to_numpy(dtype=np.float64)
        row_positions = np.flatnonzero(np.isclose(cost_values, float(cost_bps), atol=1e-12))
        if len(row_positions) == 0:
            raise TradingReplayError(f"Reference metrics table has no row for cost_bps={cost_bps}")
        row = frame.iloc[int(row_positions[0])]
        metric_dict = asdict(metrics)
        for generated_field, aliases in METRIC_FIELD_ALIASES.items():
            if generated_field == "cost_bps":
                continue
            reference_column = find_column(frame, aliases)
            if reference_column is None:
                continue
            expected_raw = row[reference_column]
            actual_raw = metric_dict[generated_field]
            if isinstance(actual_raw, bool):
                expected_bool = str(expected_raw).strip().lower() in {"true", "1", "yes"}
                passed = bool(actual_raw) == expected_bool
                comparisons.append(
                    MetricComparison(
                        reference_column=reference_column,
                        generated_field=generated_field,
                        expected_value=float(expected_bool),
                        actual_value=float(bool(actual_raw)),
                        absolute_difference=0.0 if passed else 1.0,
                        tolerance=0.0,
                        passed=passed,
                    )
                )
                continue
            try:
                expected = float(expected_raw)
                actual = float(actual_raw)
            except (TypeError, ValueError):
                continue
            difference = abs(actual - expected)
            comparisons.append(
                MetricComparison(
                    reference_column=reference_column,
                    generated_field=generated_field,
                    expected_value=expected,
                    actual_value=actual,
                    absolute_difference=difference,
                    tolerance=float(tolerance),
                    passed=difference <= tolerance,
                )
            )
    required_fields = {"net_total_return", "max_drawdown", "turnover_units"}
    compared_fields = {item.generated_field for item in comparisons}
    missing_required = sorted(required_fields - compared_fields)
    if missing_required:
        raise TradingReplayError(
            "Reference metrics table does not expose the minimum required comparable fields: "
            f"{missing_required}. Available columns: {list(reference_metrics.columns)}"
        )
    failed = [item for item in comparisons if not item.passed]
    if failed:
        detail = ", ".join(
            f"{item.generated_field}/{item.reference_column}: diff={item.absolute_difference:.3e}"
            for item in failed[:8]
        )
        raise TradingReplayError(f"Model A metrics parity failed: {detail}")
    return tuple(comparisons)


def compare_overlay_selection_metrics(
    validation_metrics: ReplayMetrics,
    selected_overlay: Mapping[str, Any],
    *,
    tolerance: float,
) -> tuple[MetricComparison, ...]:
    """Compare validation replay with any numeric metrics exposed by selected_overlay.json.

    The Notebook 7 selected-overlay JSON is treated as an optional comparison
    target because the exact metric key names are Notebook artefact dependent.
    Frozen overlay parameters are already verified by Step 1; this function adds
    metric parity when the relevant metric keys are present.
    """

    aliases = {
        "net_total_return": (
            "validation_selected_net_return",
            "validation_net_return",
            "validation_net_total_return",
            "net_return",
            "net_total_return",
        ),
        "max_drawdown": (
            "validation_selected_max_drawdown",
            "validation_max_drawdown",
            "max_drawdown",
            "validation_net_max_drawdown",
        ),
        "turnover_units": (
            "validation_selected_turnover_units",
            "validation_turnover",
            "turnover",
            "turnover_units",
        ),
        "round_turn_equivalent_trades": (
            "validation_selected_trade_count",
            "validation_round_turn_trades",
            "round_turn_equivalent_trades",
            "trade_count",
            "trades",
        ),
        "active_bar_rate": (
            "validation_selected_active_rate",
            "validation_active_rate",
            "active_bar_rate",
            "active_rate",
        ),
    }
    metric_dict = asdict(validation_metrics)
    comparisons: list[MetricComparison] = []
    lower_to_key = {str(key).lower(): str(key) for key in selected_overlay.keys()}
    for field, candidates in aliases.items():
        selected_key = None
        for candidate in candidates:
            if candidate.lower() in lower_to_key:
                selected_key = lower_to_key[candidate.lower()]
                break
        if selected_key is None:
            continue
        try:
            expected = float(selected_overlay[selected_key])
            actual = float(metric_dict[field])
        except (TypeError, ValueError):
            continue
        difference = abs(actual - expected)
        comparisons.append(
            MetricComparison(
                reference_column=selected_key,
                generated_field=field,
                expected_value=expected,
                actual_value=actual,
                absolute_difference=difference,
                tolerance=float(tolerance),
                passed=difference <= tolerance,
            )
        )
    failed = [item for item in comparisons if not item.passed]
    if failed:
        detail = ", ".join(
            f"{item.generated_field}/{item.reference_column}: diff={item.absolute_difference:.3e}"
            for item in failed[:8]
        )
        raise TradingReplayError(f"Overlay-validation metric parity failed: {detail}")
    return tuple(comparisons)


def metrics_to_dict(metrics: ReplayMetrics) -> dict[str, Any]:
    return asdict(metrics)


def comparison_to_dict(item: Any) -> dict[str, Any]:
    return asdict(item)
