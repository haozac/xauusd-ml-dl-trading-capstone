"""Event-driven offline runtime simulation for Stage 1 Step 6.

This module deliberately separates runtime event processing from the historical
batch replay functions used in Steps 4 and 5.  It consumes a chronological stream
of prediction events and updates stateful virtual ledgers one event at a time.
The generated ledgers are later compared with the audited batch replay outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from capstone_trading.data.canonical_bars import M15_DELTA
from capstone_trading.errors import Step4TradingReplayError, TradingReplayError
from capstone_trading.evaluation.model_b_replay import (
    ModelBDiagnostics,
    ModelBOverlayRules,
    compute_model_b_diagnostics,
    replay_model_b,
)
from capstone_trading.policy.position_transition import resolve_position_transition
from capstone_trading.evaluation.trading_replay import (
    ModelAOverlayRules,
    ReplayMetrics,
    compute_replay_metrics,
    replay_model_a,
    validate_prediction_frame,
)

POSITION_VALUES: tuple[int, ...] = (-1, 0, 1)
STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RuntimeLogComparison:
    model_id: str
    partition: str
    cost_bps: float
    column: str
    compared_rows: int
    maximum_absolute_difference: float
    mismatch_count: int
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class RuntimeResumeComparison:
    model_id: str
    partition: str
    cost_bps: float
    row_count: int
    comparable_columns: int
    maximum_absolute_difference: float
    mismatch_count: int
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class StreamingEventMaterialisationReport:
    partition: str
    feature_rows: int
    sequence_length: int
    batch_event_count: int
    streaming_event_count: int
    first_streaming_event_time_utc: str | None
    last_streaming_event_time_utc: str | None
    endpoint_position_mismatch_count: int
    timestamp_mismatch_count: int
    target_direction_mismatch_count: int
    maximum_target_return_difference: float
    passed: bool


def streaming_endpoint_positions_from_features(
    feature_index: pd.Index,
    *,
    sequence_length: int,
) -> np.ndarray:
    """Emit sequence endpoint positions from a rolling chronological feature buffer.

    This function intentionally mirrors live event readiness rather than the
    vectorised training helper.  It scans one feature timestamp at a time, keeps
    only the latest ``sequence_length`` timestamps, and emits the current row
    position only when the rolling buffer is complete and strictly contiguous at
    the M15 cadence.
    """

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    timestamps = pd.DatetimeIndex(pd.to_datetime(feature_index, utc=True))
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise TradingReplayError("Streaming feature index must be unique and chronological")
    endpoint_positions: list[int] = []
    buffer: list[pd.Timestamp] = []
    for row_position, timestamp in enumerate(timestamps):
        buffer.append(pd.Timestamp(timestamp))
        if len(buffer) > sequence_length:
            buffer.pop(0)
        if len(buffer) < sequence_length:
            continue
        contiguous = all(
            buffer[idx] - buffer[idx - 1] == M15_DELTA
            for idx in range(1, len(buffer))
        )
        if contiguous:
            endpoint_positions.append(row_position)
    return np.asarray(endpoint_positions, dtype=np.int64)


def verify_streaming_event_materialisation(
    *,
    partition: pd.DataFrame,
    endpoint_positions: Sequence[int] | np.ndarray,
    batch_events: pd.DataFrame,
    sequence_length: int,
    partition_name: str,
    target_return_tolerance: float = 1e-12,
) -> StreamingEventMaterialisationReport:
    """Verify live-like event readiness against the audited batch sequence plan.

    The feature dataframe is streamed chronologically through a rolling 48-row
    readiness buffer.  The emitted endpoint rows are then compared with the
    batch sequence endpoints and the already constructed runtime event table.
    The function does not recompute technical indicators incrementally; that
    remains covered by the Step 2 feature parity gate.
    """

    if target_return_tolerance < 0:
        raise ValueError("target_return_tolerance must be non-negative")
    if {"target_ret_fwd", "target_dir"} - set(partition.columns):
        raise TradingReplayError("Partition is missing target columns required for streaming event verification")

    expected_positions = np.asarray(endpoint_positions, dtype=np.int64)
    streaming_positions = streaming_endpoint_positions_from_features(
        partition.index,
        sequence_length=sequence_length,
    )
    endpoint_mismatch_count = 0
    if len(expected_positions) != len(streaming_positions):
        endpoint_mismatch_count = abs(len(expected_positions) - len(streaming_positions))
    elif not np.array_equal(expected_positions, streaming_positions):
        endpoint_mismatch_count = int(np.count_nonzero(expected_positions != streaming_positions))

    streaming_events = prediction_events_from_partition(
        partition=partition,
        endpoint_positions=streaming_positions,
        probabilities=np.zeros(len(streaming_positions), dtype=np.float64),
    )
    events = _ensure_event_frame(batch_events)

    timestamp_mismatch_count = abs(len(streaming_events) - len(events))
    target_direction_mismatch_count = abs(len(streaming_events) - len(events))
    maximum_target_return_difference = float("inf") if len(streaming_events) != len(events) else 0.0
    if len(streaming_events) == len(events):
        stream_time = pd.DatetimeIndex(pd.to_datetime(streaming_events["time"], utc=True))
        event_time = pd.DatetimeIndex(pd.to_datetime(events["time"], utc=True))
        timestamp_mismatch_count = int(np.count_nonzero(stream_time.asi8 != event_time.asi8))
        stream_dir = streaming_events["target_dir"].to_numpy(dtype=np.int8)
        event_dir = events["target_dir"].to_numpy(dtype=np.int8)
        target_direction_mismatch_count = int(np.count_nonzero(stream_dir != event_dir))
        target_diff = np.abs(
            streaming_events["target_ret_fwd"].to_numpy(dtype=np.float64)
            - events["target_ret_fwd"].to_numpy(dtype=np.float64)
        )
        maximum_target_return_difference = float(target_diff.max(initial=0.0))

    first_time = None
    last_time = None
    if not streaming_events.empty:
        first_time = pd.Timestamp(streaming_events["time"].iloc[0]).tz_convert("UTC").isoformat()
        last_time = pd.Timestamp(streaming_events["time"].iloc[-1]).tz_convert("UTC").isoformat()

    passed = (
        endpoint_mismatch_count == 0
        and timestamp_mismatch_count == 0
        and target_direction_mismatch_count == 0
        and maximum_target_return_difference <= target_return_tolerance
    )
    report = StreamingEventMaterialisationReport(
        partition=partition_name,
        feature_rows=int(len(partition)),
        sequence_length=int(sequence_length),
        batch_event_count=int(len(events)),
        streaming_event_count=int(len(streaming_events)),
        first_streaming_event_time_utc=first_time,
        last_streaming_event_time_utc=last_time,
        endpoint_position_mismatch_count=int(endpoint_mismatch_count),
        timestamp_mismatch_count=int(timestamp_mismatch_count),
        target_direction_mismatch_count=int(target_direction_mismatch_count),
        maximum_target_return_difference=maximum_target_return_difference,
        passed=bool(passed),
    )
    if not report.passed:
        raise TradingReplayError(
            "Streaming event materialisation failed for "
            f"{partition_name}: endpoint_mismatches={endpoint_mismatch_count}, "
            f"timestamp_mismatches={timestamp_mismatch_count}, "
            f"target_dir_mismatches={target_direction_mismatch_count}, "
            f"max_target_return_diff={maximum_target_return_difference:.3e}"
        )
    return report


def _iso_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).tz_convert("UTC").isoformat()


def _timestamp_from_iso(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value).tz_convert("UTC")


def _day_key(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(timestamp.date(), tz="UTC")


def _ensure_event_frame(events: pd.DataFrame) -> pd.DataFrame:
    return validate_prediction_frame(events)


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


def _model_a_signal(probability: float, rules: ModelAOverlayRules) -> int:
    if probability >= rules.long_threshold:
        return 1
    if probability <= rules.short_threshold:
        return -1
    return 0



@dataclass
class ModelARuntimeState:
    """Stateful Model A virtual ledger for completed eligible prediction events."""

    rules: ModelAOverlayRules
    cost_bps: float
    current_position: int = 0
    hold_bars: int = 0
    flat_bars_since_exit: int = 10**9
    current_day_iso: str | None = None
    policy_changes_today: int = 0
    daily_log_return: float = 0.0
    daily_stop_active: bool = False
    total_stop_active: bool = False
    total_stop_triggered: bool = False
    first_total_stop_trigger_utc: str | None = None
    cumulative_gross_log: float = 0.0
    cumulative_net_log: float = 0.0
    running_peak_net_equity: float = 1.0
    last_time_utc: str | None = None
    row_number: int = 0
    policy_change_events: int = 0
    gap_exit_events: int = 0
    daily_stop_exit_events: int = 0
    daily_stop_trigger_count: int = 0
    total_stop_exit_events: int = 0
    daily_stop_dates: tuple[str, ...] = ()
    rows: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.rows is None:
            self.rows = []

    @property
    def one_way_cost(self) -> float:
        return float(self.cost_bps) / 10000.0

    def snapshot(self) -> dict[str, Any]:
        """Return restart state only; persisted audit rows live separately."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "model_id": "MODEL_A",
            "cost_bps": float(self.cost_bps),
            "current_position": int(self.current_position),
            "hold_bars": int(self.hold_bars),
            "flat_bars_since_exit": int(self.flat_bars_since_exit),
            "current_day_iso": self.current_day_iso,
            "policy_changes_today": int(self.policy_changes_today),
            "daily_log_return": float(self.daily_log_return),
            "daily_stop_active": bool(self.daily_stop_active),
            "total_stop_active": bool(self.total_stop_active),
            "total_stop_triggered": bool(self.total_stop_triggered),
            "first_total_stop_trigger_utc": self.first_total_stop_trigger_utc,
            "cumulative_gross_log": float(self.cumulative_gross_log),
            "cumulative_net_log": float(self.cumulative_net_log),
            "running_peak_net_equity": float(self.running_peak_net_equity),
            "last_time_utc": self.last_time_utc,
            "row_number": int(self.row_number),
            "policy_change_events": int(self.policy_change_events),
            "gap_exit_events": int(self.gap_exit_events),
            "daily_stop_exit_events": int(self.daily_stop_exit_events),
            "daily_stop_trigger_count": int(self.daily_stop_trigger_count),
            "total_stop_exit_events": int(self.total_stop_exit_events),
            "daily_stop_dates": list(self.daily_stop_dates),
        }

    @classmethod
    def from_snapshot(cls, rules: ModelAOverlayRules, snapshot: Mapping[str, Any]) -> "ModelARuntimeState":
        if int(snapshot.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
            raise TradingReplayError("Unsupported Model A runtime-state schema version")
        if snapshot.get("model_id") != "MODEL_A":
            raise TradingReplayError("Snapshot is not for MODEL_A")
        return cls(
            rules=rules,
            cost_bps=float(snapshot["cost_bps"]),
            current_position=int(snapshot["current_position"]),
            hold_bars=int(snapshot["hold_bars"]),
            flat_bars_since_exit=int(snapshot["flat_bars_since_exit"]),
            current_day_iso=(None if snapshot.get("current_day_iso") is None else str(snapshot["current_day_iso"])),
            policy_changes_today=int(snapshot["policy_changes_today"]),
            daily_log_return=float(snapshot["daily_log_return"]),
            daily_stop_active=bool(snapshot["daily_stop_active"]),
            total_stop_active=bool(snapshot["total_stop_active"]),
            total_stop_triggered=bool(snapshot["total_stop_triggered"]),
            first_total_stop_trigger_utc=(None if snapshot.get("first_total_stop_trigger_utc") is None else str(snapshot["first_total_stop_trigger_utc"])),
            cumulative_gross_log=float(snapshot["cumulative_gross_log"]),
            cumulative_net_log=float(snapshot["cumulative_net_log"]),
            running_peak_net_equity=float(snapshot["running_peak_net_equity"]),
            last_time_utc=(None if snapshot.get("last_time_utc") is None else str(snapshot["last_time_utc"])),
            row_number=int(snapshot["row_number"]),
            policy_change_events=int(snapshot["policy_change_events"]),
            gap_exit_events=int(snapshot["gap_exit_events"]),
            daily_stop_exit_events=int(snapshot["daily_stop_exit_events"]),
            daily_stop_trigger_count=int(snapshot["daily_stop_trigger_count"]),
            total_stop_exit_events=int(snapshot["total_stop_exit_events"]),
            daily_stop_dates=tuple(str(item) for item in snapshot.get("daily_stop_dates", [])),
        )

    def process_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = pd.Timestamp(event["time"]).tz_convert("UTC")
        utc_date = timestamp.date().isoformat()
        day_key = _day_key(timestamp).date().isoformat()
        if self.current_day_iso is None or day_key != self.current_day_iso:
            self.current_day_iso = day_key
            self.policy_changes_today = 0
            self.daily_log_return = 0.0
            self.daily_stop_active = False

        last_time = _timestamp_from_iso(self.last_time_utc)
        gap_from_previous = bool(last_time is not None and timestamp - last_time != M15_DELTA)
        probability = float(event["p_up"])
        target_ret_fwd = float(event["target_ret_fwd"])
        target_dir = int(event["target_dir"])
        signal = _model_a_signal(probability, self.rules)
        previous_position = int(self.current_position)
        desired_position = int(signal)
        next_position = previous_position
        forced_reason = ""
        blocked_reason = ""
        policy_event_units = 0

        if self.total_stop_active:
            desired_position = 0
            next_position = 0
            forced_reason = "total_drawdown_stop"
            if previous_position != 0:
                self.total_stop_exit_events += 1
        elif self.daily_stop_active:
            desired_position = 0
            next_position = 0
            forced_reason = "daily_loss_stop"
            if previous_position != 0:
                self.daily_stop_exit_events += 1
        elif gap_from_previous:
            desired_position = 0
            next_position = 0
            if previous_position != 0:
                forced_reason = "gap_exit"
                self.gap_exit_events += 1
            else:
                blocked_reason = "gap_block"
        else:
            if desired_position != previous_position:
                active_min_hold_blocked = (
                    previous_position != 0 and self.hold_bars <= self.rules.minimum_hold_bars
                )
                flat_cooldown_blocked = (
                    previous_position == 0
                    and desired_position != 0
                    and self.flat_bars_since_exit < self.rules.minimum_hold_bars
                )
                if active_min_hold_blocked or flat_cooldown_blocked:
                    next_position = previous_position
                    blocked_reason = "minimum_hold_active"
                else:
                    resolution = resolve_position_transition(
                        current_position=previous_position,
                        desired_position=desired_position,
                        policy_changes_today=self.policy_changes_today,
                        max_policy_changes_per_day=(
                            self.rules.max_policy_changes_per_day
                        ),
                        reversal_policy_event_units=(
                            self.rules.reversal_policy_event_units
                        ),
                        allow_risk_reducing_exit_when_capped=(
                            self.rules.allow_risk_reducing_exit_when_capped
                        ),
                    )
                    next_position = int(resolution.effective_target_position)
                    policy_event_units = int(resolution.consumed_policy_units)
                    if next_position != previous_position:
                        self.policy_changes_today += policy_event_units
                        self.policy_change_events += policy_event_units
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
        gross_log_return = float(next_position) * target_ret_fwd
        cost_log_return = turnover_units * self.one_way_cost
        net_log_return = gross_log_return - cost_log_return
        self.cumulative_gross_log += gross_log_return
        self.cumulative_net_log += net_log_return
        self.daily_log_return += net_log_return
        gross_equity = float(np.exp(self.cumulative_gross_log))
        net_equity = float(np.exp(self.cumulative_net_log))
        self.running_peak_net_equity = max(self.running_peak_net_equity, net_equity)
        net_drawdown = float(net_equity / self.running_peak_net_equity - 1.0)

        daily_stop_triggered_now = False
        if not self.daily_stop_active and self.daily_log_return <= self.rules.daily_loss_log_threshold:
            self.daily_stop_active = True
            daily_stop_triggered_now = True
            self.daily_stop_trigger_count += 1
            self.daily_stop_dates = tuple(sorted({*self.daily_stop_dates, utc_date}))

        total_stop_triggered_now = False
        if not self.total_stop_active and net_drawdown <= self.rules.total_drawdown_stop:
            self.total_stop_active = True
            self.total_stop_triggered = True
            total_stop_triggered_now = True
            if self.first_total_stop_trigger_utc is None:
                self.first_total_stop_trigger_utc = timestamp.isoformat()

        if next_position == 0:
            next_hold_bars = 0
            if previous_position == 0:
                next_flat_bars_since_exit = min(self.flat_bars_since_exit + 1, 10**9)
            elif forced_reason == "gap_exit":
                next_flat_bars_since_exit = 10**9
            else:
                next_flat_bars_since_exit = 0
        elif next_position == previous_position:
            next_hold_bars = self.hold_bars + 1 if previous_position != 0 else 1
            next_flat_bars_since_exit = 10**9
        else:
            next_hold_bars = 1
            next_flat_bars_since_exit = 10**9

        row = {
            "time": timestamp.isoformat(),
            "row_number": int(self.row_number),
            "p_up": probability,
            "target_dir": target_dir,
            "target_ret_fwd": target_ret_fwd,
            "signal": signal,
            "previous_position": previous_position,
            "position": int(next_position),
            "position_before": previous_position,
            "position_after": int(next_position),
            "hold_bars_before": int(self.hold_bars),
            "hold_bars_after": int(next_hold_bars),
            "flat_bars_since_exit_before": int(self.flat_bars_since_exit),
            "flat_bars_since_exit_after": int(next_flat_bars_since_exit),
            "policy_change_events_today": int(self.policy_changes_today),
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
            "daily_log_return": float(self.daily_log_return),
            "gap_from_previous_prediction": gap_from_previous,
            "daily_stop_active_after": bool(self.daily_stop_active),
            "total_stop_active_after": bool(self.total_stop_active),
            "daily_stop_triggered": bool(daily_stop_triggered_now),
            "total_stop_triggered": bool(total_stop_triggered_now),
            "change_reason": _position_change_reason(
                previous_position=previous_position,
                next_position=int(next_position),
                forced_reason=forced_reason,
                blocked_reason=blocked_reason,
            ),
        }
        assert self.rows is not None
        self.rows.append(row)
        self.current_position = int(next_position)
        self.hold_bars = int(next_hold_bars)
        self.flat_bars_since_exit = int(next_flat_bars_since_exit)
        self.last_time_utc = timestamp.isoformat()
        self.row_number += 1
        return row

    def to_frame(self) -> pd.DataFrame:
        assert self.rows is not None
        return pd.DataFrame(self.rows)

    def metrics(self) -> ReplayMetrics:
        return compute_replay_metrics(
            self.to_frame(),
            cost_bps=float(self.cost_bps),
            policy_change_events=self.policy_change_events,
            gap_exit_events=self.gap_exit_events,
            daily_stop_exit_events=self.daily_stop_exit_events,
            daily_stop_trigger_count=self.daily_stop_trigger_count,
            total_stop_exit_events=self.total_stop_exit_events,
            total_stop_triggered=self.total_stop_triggered,
            first_total_stop_trigger=_timestamp_from_iso(self.first_total_stop_trigger_utc),
            daily_stop_dates=self.daily_stop_dates,
        )


@dataclass
class ModelBRuntimeState:
    """Stateful Model B virtual ledger for completed eligible prediction events."""

    rules: ModelBOverlayRules
    cost_bps: float
    current_position: int = 0
    current_day_iso: str | None = None
    successful_entries_today: int = 0
    daily_log_return: float = 0.0
    daily_stop_active: bool = False
    total_stop_active: bool = False
    total_stop_triggered: bool = False
    first_total_stop_trigger_utc: str | None = None
    cumulative_gross_log: float = 0.0
    cumulative_net_log: float = 0.0
    running_peak_net_equity: float = 1.0
    last_time_utc: str | None = None
    row_number: int = 0
    successful_entry_count: int = 0
    normal_exit_count: int = 0
    daily_entry_cap_block_count: int = 0
    gap_block_count: int = 0
    gap_exit_events: int = 0
    daily_stop_exit_events: int = 0
    daily_stop_trigger_count: int = 0
    total_stop_exit_events: int = 0
    daily_stop_dates: tuple[str, ...] = ()
    entries_by_day: dict[str, int] | None = None
    daily_net_log_by_day: dict[str, float] | None = None
    rows: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.rows is None:
            self.rows = []
        if self.entries_by_day is None:
            self.entries_by_day = {}
        if self.daily_net_log_by_day is None:
            self.daily_net_log_by_day = {}

    @property
    def one_way_cost(self) -> float:
        return float(self.cost_bps) / 10000.0

    def snapshot(self) -> dict[str, Any]:
        assert self.entries_by_day is not None and self.daily_net_log_by_day is not None
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "model_id": "MODEL_B_V2",
            "cost_bps": float(self.cost_bps),
            "current_position": int(self.current_position),
            "current_day_iso": self.current_day_iso,
            "successful_entries_today": int(self.successful_entries_today),
            "daily_log_return": float(self.daily_log_return),
            "daily_stop_active": bool(self.daily_stop_active),
            "total_stop_active": bool(self.total_stop_active),
            "total_stop_triggered": bool(self.total_stop_triggered),
            "first_total_stop_trigger_utc": self.first_total_stop_trigger_utc,
            "cumulative_gross_log": float(self.cumulative_gross_log),
            "cumulative_net_log": float(self.cumulative_net_log),
            "running_peak_net_equity": float(self.running_peak_net_equity),
            "last_time_utc": self.last_time_utc,
            "row_number": int(self.row_number),
            "successful_entry_count": int(self.successful_entry_count),
            "normal_exit_count": int(self.normal_exit_count),
            "daily_entry_cap_block_count": int(self.daily_entry_cap_block_count),
            "gap_block_count": int(self.gap_block_count),
            "gap_exit_events": int(self.gap_exit_events),
            "daily_stop_exit_events": int(self.daily_stop_exit_events),
            "daily_stop_trigger_count": int(self.daily_stop_trigger_count),
            "total_stop_exit_events": int(self.total_stop_exit_events),
            "daily_stop_dates": list(self.daily_stop_dates),
            "entries_by_day": dict(self.entries_by_day),
            "daily_net_log_by_day": dict(self.daily_net_log_by_day),
        }

    @classmethod
    def from_snapshot(cls, rules: ModelBOverlayRules, snapshot: Mapping[str, Any]) -> "ModelBRuntimeState":
        if int(snapshot.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
            raise TradingReplayError("Unsupported Model B runtime-state schema version")
        if snapshot.get("model_id") != "MODEL_B_V2":
            raise TradingReplayError("Snapshot is not for MODEL_B_V2")
        return cls(
            rules=rules,
            cost_bps=float(snapshot["cost_bps"]),
            current_position=int(snapshot["current_position"]),
            current_day_iso=(None if snapshot.get("current_day_iso") is None else str(snapshot["current_day_iso"])),
            successful_entries_today=int(snapshot["successful_entries_today"]),
            daily_log_return=float(snapshot["daily_log_return"]),
            daily_stop_active=bool(snapshot["daily_stop_active"]),
            total_stop_active=bool(snapshot["total_stop_active"]),
            total_stop_triggered=bool(snapshot["total_stop_triggered"]),
            first_total_stop_trigger_utc=(None if snapshot.get("first_total_stop_trigger_utc") is None else str(snapshot["first_total_stop_trigger_utc"])),
            cumulative_gross_log=float(snapshot["cumulative_gross_log"]),
            cumulative_net_log=float(snapshot["cumulative_net_log"]),
            running_peak_net_equity=float(snapshot["running_peak_net_equity"]),
            last_time_utc=(None if snapshot.get("last_time_utc") is None else str(snapshot["last_time_utc"])),
            row_number=int(snapshot["row_number"]),
            successful_entry_count=int(snapshot["successful_entry_count"]),
            normal_exit_count=int(snapshot["normal_exit_count"]),
            daily_entry_cap_block_count=int(snapshot["daily_entry_cap_block_count"]),
            gap_block_count=int(snapshot["gap_block_count"]),
            gap_exit_events=int(snapshot["gap_exit_events"]),
            daily_stop_exit_events=int(snapshot["daily_stop_exit_events"]),
            daily_stop_trigger_count=int(snapshot["daily_stop_trigger_count"]),
            total_stop_exit_events=int(snapshot["total_stop_exit_events"]),
            daily_stop_dates=tuple(str(item) for item in snapshot.get("daily_stop_dates", [])),
            entries_by_day={str(key): int(value) for key, value in dict(snapshot.get("entries_by_day", {})).items()},
            daily_net_log_by_day={str(key): float(value) for key, value in dict(snapshot.get("daily_net_log_by_day", {})).items()},
        )

    def process_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = pd.Timestamp(event["time"]).tz_convert("UTC")
        utc_date = timestamp.date().isoformat()
        day_key = _day_key(timestamp).date().isoformat()
        if self.current_day_iso is None or day_key != self.current_day_iso:
            self.current_day_iso = day_key
            self.successful_entries_today = 0
            self.daily_log_return = 0.0
            self.daily_stop_active = False

        last_time = _timestamp_from_iso(self.last_time_utc)
        gap_from_previous = bool(last_time is not None and timestamp - last_time != M15_DELTA)
        probability = float(event["p_up"])
        target_ret_fwd = float(event["target_ret_fwd"])
        target_dir = int(event["target_dir"])
        previous_position = int(self.current_position)
        next_position = previous_position
        forced_reason = ""
        blocked_reason = ""
        signal = 1 if probability >= self.rules.entry_threshold else 0
        successful_entry_event = 0
        normal_exit_event = 0

        if self.total_stop_active:
            next_position = 0
            forced_reason = "total_drawdown_stop"
            if previous_position != 0:
                self.total_stop_exit_events += 1
        elif self.daily_stop_active:
            next_position = 0
            forced_reason = "daily_loss_stop"
            if previous_position != 0:
                self.daily_stop_exit_events += 1
        elif gap_from_previous:
            next_position = 0
            if previous_position != 0:
                forced_reason = "gap_exit"
                self.gap_exit_events += 1
            else:
                blocked_reason = "gap_block"
                self.gap_block_count += 1
        elif previous_position == 0:
            if probability >= self.rules.entry_threshold:
                if self.successful_entries_today < self.rules.max_successful_entries_per_day:
                    next_position = 1
                    self.successful_entries_today += 1
                    self.successful_entry_count += 1
                    successful_entry_event = 1
                    assert self.entries_by_day is not None
                    self.entries_by_day[utc_date] = self.entries_by_day.get(utc_date, 0) + 1
                else:
                    next_position = 0
                    blocked_reason = "daily_entry_cap_active"
                    self.daily_entry_cap_block_count += 1
            else:
                next_position = 0
        else:
            if probability >= self.rules.exit_threshold:
                next_position = 1
            else:
                next_position = 0
                self.normal_exit_count += 1
                normal_exit_event = 1

        turnover_units = float(abs(next_position - previous_position))
        gross_log_return = float(next_position) * target_ret_fwd
        cost_log_return = turnover_units * self.one_way_cost
        net_log_return = gross_log_return - cost_log_return
        self.cumulative_gross_log += gross_log_return
        self.cumulative_net_log += net_log_return
        self.daily_log_return += net_log_return
        assert self.daily_net_log_by_day is not None
        self.daily_net_log_by_day[utc_date] = self.daily_log_return
        gross_equity = float(np.exp(self.cumulative_gross_log))
        net_equity = float(np.exp(self.cumulative_net_log))
        self.running_peak_net_equity = max(self.running_peak_net_equity, net_equity)
        net_drawdown = float(net_equity / self.running_peak_net_equity - 1.0)

        daily_stop_triggered_now = False
        if not self.daily_stop_active and self.daily_log_return <= self.rules.daily_loss_log_threshold:
            self.daily_stop_active = True
            daily_stop_triggered_now = True
            self.daily_stop_trigger_count += 1
            self.daily_stop_dates = tuple(sorted({*self.daily_stop_dates, utc_date}))

        total_stop_triggered_now = False
        if not self.total_stop_active and net_drawdown <= self.rules.total_drawdown_stop:
            self.total_stop_active = True
            self.total_stop_triggered = True
            total_stop_triggered_now = True
            if self.first_total_stop_trigger_utc is None:
                self.first_total_stop_trigger_utc = timestamp.isoformat()

        row = {
            "time": timestamp.isoformat(),
            "row_number": int(self.row_number),
            "p_up": probability,
            "target_dir": target_dir,
            "target_ret_fwd": target_ret_fwd,
            "signal": signal,
            "previous_position": previous_position,
            "position": int(next_position),
            "position_before": previous_position,
            "position_after": int(next_position),
            "successful_entries_today": int(self.successful_entries_today),
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
            "daily_log_return": float(self.daily_log_return),
            "gap_from_previous_prediction": gap_from_previous,
            "daily_stop_active_after": bool(self.daily_stop_active),
            "total_stop_active_after": bool(self.total_stop_active),
            "daily_stop_triggered": bool(daily_stop_triggered_now),
            "total_stop_triggered": bool(total_stop_triggered_now),
            "change_reason": _model_b_reason(
                previous_position=previous_position,
                next_position=int(next_position),
                forced_reason=forced_reason,
                blocked_reason=blocked_reason,
            ),
        }
        assert self.rows is not None
        self.rows.append(row)
        self.current_position = int(next_position)
        self.last_time_utc = timestamp.isoformat()
        self.row_number += 1
        return row

    def to_frame(self) -> pd.DataFrame:
        assert self.rows is not None
        return pd.DataFrame(self.rows)

    def metrics(self) -> ReplayMetrics:
        return compute_replay_metrics(
            self.to_frame(),
            cost_bps=float(self.cost_bps),
            policy_change_events=self.successful_entry_count + self.normal_exit_count,
            gap_exit_events=self.gap_exit_events,
            daily_stop_exit_events=self.daily_stop_exit_events,
            daily_stop_trigger_count=self.daily_stop_trigger_count,
            total_stop_exit_events=self.total_stop_exit_events,
            total_stop_triggered=self.total_stop_triggered,
            first_total_stop_trigger=_timestamp_from_iso(self.first_total_stop_trigger_utc),
            daily_stop_dates=self.daily_stop_dates,
        )

    def diagnostics(self) -> ModelBDiagnostics:
        assert self.entries_by_day is not None and self.daily_net_log_by_day is not None
        return compute_model_b_diagnostics(
            self.to_frame(),
            rules=self.rules,
            successful_entry_count=self.successful_entry_count,
            normal_exit_count=self.normal_exit_count,
            daily_entry_cap_block_count=self.daily_entry_cap_block_count,
            gap_block_count=self.gap_block_count,
            gap_exit_events=self.gap_exit_events,
            daily_stop_exit_events=self.daily_stop_exit_events,
            total_stop_exit_events=self.total_stop_exit_events,
            entries_by_day=self.entries_by_day,
            daily_net_log_by_day=self.daily_net_log_by_day,
        )


def _model_b_reason(
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


def run_model_a_runtime(events: pd.DataFrame, rules: ModelAOverlayRules, *, cost_bps: float) -> tuple[pd.DataFrame, ReplayMetrics]:
    frame = _ensure_event_frame(events)
    engine = ModelARuntimeState(rules=rules, cost_bps=cost_bps)
    for event in frame.to_dict("records"):
        engine.process_event(event)
    return engine.to_frame(), engine.metrics()


def run_model_b_runtime(events: pd.DataFrame, rules: ModelBOverlayRules, *, cost_bps: float) -> tuple[pd.DataFrame, ReplayMetrics, ModelBDiagnostics]:
    frame = _ensure_event_frame(events)
    engine = ModelBRuntimeState(rules=rules, cost_bps=cost_bps)
    for event in frame.to_dict("records"):
        engine.process_event(event)
    return engine.to_frame(), engine.metrics(), engine.diagnostics()


def run_model_a_runtime_with_resume(
    events: pd.DataFrame,
    rules: ModelAOverlayRules,
    *,
    cost_bps: float,
    split_at: int,
) -> tuple[pd.DataFrame, ReplayMetrics]:
    frame = _ensure_event_frame(events)
    if not 0 < split_at < len(frame):
        raise ValueError("split_at must be inside the event frame")
    first = ModelARuntimeState(rules=rules, cost_bps=cost_bps)
    for event in frame.iloc[:split_at].to_dict("records"):
        first.process_event(event)
    persisted_rows = first.to_frame().copy()
    resumed = ModelARuntimeState.from_snapshot(rules, first.snapshot())
    for event in frame.iloc[split_at:].to_dict("records"):
        resumed.process_event(event)
    combined = pd.concat([persisted_rows, resumed.to_frame()], ignore_index=True)
    metrics = compute_replay_metrics(
        combined,
        cost_bps=float(cost_bps),
        policy_change_events=resumed.policy_change_events,
        gap_exit_events=resumed.gap_exit_events,
        daily_stop_exit_events=resumed.daily_stop_exit_events,
        daily_stop_trigger_count=resumed.daily_stop_trigger_count,
        total_stop_exit_events=resumed.total_stop_exit_events,
        total_stop_triggered=resumed.total_stop_triggered,
        first_total_stop_trigger=_timestamp_from_iso(resumed.first_total_stop_trigger_utc),
        daily_stop_dates=resumed.daily_stop_dates,
    )
    return combined, metrics


def run_model_b_runtime_with_resume(
    events: pd.DataFrame,
    rules: ModelBOverlayRules,
    *,
    cost_bps: float,
    split_at: int,
) -> tuple[pd.DataFrame, ReplayMetrics, ModelBDiagnostics]:
    frame = _ensure_event_frame(events)
    if not 0 < split_at < len(frame):
        raise ValueError("split_at must be inside the event frame")
    first = ModelBRuntimeState(rules=rules, cost_bps=cost_bps)
    for event in frame.iloc[:split_at].to_dict("records"):
        first.process_event(event)
    persisted_rows = first.to_frame().copy()
    resumed = ModelBRuntimeState.from_snapshot(rules, first.snapshot())
    for event in frame.iloc[split_at:].to_dict("records"):
        resumed.process_event(event)
    combined = pd.concat([persisted_rows, resumed.to_frame()], ignore_index=True)
    metrics = compute_replay_metrics(
        combined,
        cost_bps=float(cost_bps),
        policy_change_events=resumed.successful_entry_count + resumed.normal_exit_count,
        gap_exit_events=resumed.gap_exit_events,
        daily_stop_exit_events=resumed.daily_stop_exit_events,
        daily_stop_trigger_count=resumed.daily_stop_trigger_count,
        total_stop_exit_events=resumed.total_stop_exit_events,
        total_stop_triggered=resumed.total_stop_triggered,
        first_total_stop_trigger=_timestamp_from_iso(resumed.first_total_stop_trigger_utc),
        daily_stop_dates=resumed.daily_stop_dates,
    )
    diagnostics = compute_model_b_diagnostics(
        combined,
        rules=rules,
        successful_entry_count=resumed.successful_entry_count,
        normal_exit_count=resumed.normal_exit_count,
        daily_entry_cap_block_count=resumed.daily_entry_cap_block_count,
        gap_block_count=resumed.gap_block_count,
        gap_exit_events=resumed.gap_exit_events,
        daily_stop_exit_events=resumed.daily_stop_exit_events,
        total_stop_exit_events=resumed.total_stop_exit_events,
        entries_by_day=resumed.entries_by_day or {},
        daily_net_log_by_day=resumed.daily_net_log_by_day or {},
    )
    return combined, metrics, diagnostics


def prediction_events_from_partition(
    *,
    partition: pd.DataFrame,
    endpoint_positions: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    positions = np.asarray(endpoint_positions, dtype=np.int64)
    p_up = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(positions) != len(p_up):
        raise TradingReplayError(
            f"Event count mismatch: endpoints={len(positions)}, probabilities={len(p_up)}"
        )
    if len(positions) == 0:
        raise TradingReplayError("Runtime event stream is empty")
    aligned = partition.iloc[positions]
    events = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(aligned.index).tz_convert("UTC"),
            "p_up": p_up,
            "target_ret_fwd": aligned["target_ret_fwd"].to_numpy(dtype=np.float64),
            "target_dir": aligned["target_dir"].to_numpy(dtype=np.int8),
        }
    )
    return _ensure_event_frame(events)


def compare_runtime_log(
    *,
    model_id: str,
    partition: str,
    cost_bps: float,
    runtime_log: pd.DataFrame,
    reference_log: pd.DataFrame,
    columns: Sequence[str],
    tolerance: float,
) -> tuple[RuntimeLogComparison, ...]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    runtime_time = pd.DatetimeIndex(pd.to_datetime(runtime_log["time"], utc=True))
    reference_time = pd.DatetimeIndex(pd.to_datetime(reference_log["time"], utc=True))
    if len(runtime_time) != len(reference_time) or not np.array_equal(runtime_time.asi8, reference_time.asi8):
        raise Step4TradingReplayError(
            f"{model_id} runtime timestamp mismatch for {partition} cost={cost_bps}"
        )
    comparisons: list[RuntimeLogComparison] = []
    for column in columns:
        if column not in runtime_log.columns or column not in reference_log.columns:
            raise Step4TradingReplayError(f"Cannot compare missing runtime column {column}")
        expected = pd.to_numeric(reference_log[column], errors="raise").to_numpy(dtype=np.float64)
        actual = pd.to_numeric(runtime_log[column], errors="raise").to_numpy(dtype=np.float64)
        diff = np.abs(actual - expected)
        mismatches = int(np.count_nonzero(diff > tolerance))
        comparisons.append(
            RuntimeLogComparison(
                model_id=model_id,
                partition=partition,
                cost_bps=float(cost_bps),
                column=column,
                compared_rows=int(len(diff)),
                maximum_absolute_difference=float(diff.max(initial=0.0)),
                mismatch_count=mismatches,
                tolerance=float(tolerance),
                passed=mismatches == 0,
            )
        )
    failures = [item for item in comparisons if not item.passed]
    if failures:
        detail = ", ".join(
            f"{item.model_id}/{item.partition}/{item.cost_bps}/{item.column}: max_diff={item.maximum_absolute_difference:.3e}, mismatches={item.mismatch_count}"
            for item in failures[:8]
        )
        raise Step4TradingReplayError(f"Runtime-vs-batch log parity failed: {detail}")
    return tuple(comparisons)


def compare_full_vs_resumed(
    *,
    model_id: str,
    partition: str,
    cost_bps: float,
    full_log: pd.DataFrame,
    resumed_log: pd.DataFrame,
    columns: Sequence[str],
    tolerance: float,
) -> RuntimeResumeComparison:
    comparisons = compare_runtime_log(
        model_id=model_id,
        partition=partition,
        cost_bps=cost_bps,
        runtime_log=resumed_log,
        reference_log=full_log,
        columns=columns,
        tolerance=tolerance,
    )
    return RuntimeResumeComparison(
        model_id=model_id,
        partition=partition,
        cost_bps=float(cost_bps),
        row_count=int(len(full_log)),
        comparable_columns=len(columns),
        maximum_absolute_difference=max((item.maximum_absolute_difference for item in comparisons), default=0.0),
        mismatch_count=sum(item.mismatch_count for item in comparisons),
        tolerance=float(tolerance),
        passed=all(item.passed for item in comparisons),
    )


def metric_dict(metrics: ReplayMetrics) -> dict[str, Any]:
    return asdict(metrics)


def diagnostic_dict(diagnostics: ModelBDiagnostics) -> dict[str, Any]:
    return asdict(diagnostics)


def comparison_dict(item: Any) -> dict[str, Any]:
    return asdict(item)


def streaming_report_dict(item: StreamingEventMaterialisationReport) -> dict[str, Any]:
    return asdict(item)


def compact_audit_row(
    *,
    partition: str,
    row_position: int,
    event: Mapping[str, Any],
    model_a_row: Mapping[str, Any],
    model_b_row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "partition": partition,
        "row_position": int(row_position),
        "time": str(event["time"]),
        "p_up": float(event["p_up"]),
        "target_dir": int(event["target_dir"]),
        "target_ret_fwd": float(event["target_ret_fwd"]),
        "model_a_position": int(model_a_row["position"]),
        "model_a_turnover": float(model_a_row["turnover"]),
        "model_a_net_equity": float(model_a_row["net_equity"]),
        "model_a_change_reason": str(model_a_row["change_reason"]),
        "model_b_position": int(model_b_row["position"]),
        "model_b_turnover": float(model_b_row["turnover"]),
        "model_b_net_equity": float(model_b_row["net_equity"]),
        "model_b_change_reason": str(model_b_row["change_reason"]),
    }
