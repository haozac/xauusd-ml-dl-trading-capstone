"""Offline daily and consolidated analysis for dual live-observation audits.

The live workers persist raw evidence only.  This module reads that evidence
and builds daily partitions, trade ledgers, operational-quality checks, and
small-sample descriptive metrics after the observation has stopped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math

import pandas as pd

from capstone_trading.runtime.live_audit import (
    completed_broker_event_identifier,
    decision_identifier,
    historical_backfill_identifier,
)


class LiveAuditAnalysisError(RuntimeError):
    """Raised when raw observation evidence is missing or malformed."""


@dataclass(frozen=True)
class RoleObservationSummary:
    role: str
    formal_audit_gate: bool
    operational_acceptance_status: str
    audit_gate_failures: tuple[str, ...]
    limited_recovery_reasons: tuple[str, ...]
    telemetry_rows: int
    decision_rows: int
    invalid_decision_evidence_count: int
    decision_evidence_issue_count: int
    decision_evidence_issue_reasons: tuple[str, ...]
    unique_completed_event_count: int
    completed_broker_event_ledger_rows: int
    completed_broker_event_ledger_present: bool
    duplicate_broker_event_key_count: int
    decision_without_broker_event_count: int
    broker_event_disposition_coverage_ratio: float | None
    completed_event_coverage_ratio: float | None
    missing_completed_event_decision_count: int
    model_prediction_count: int
    model_unavailable_event_count: int
    model_prediction_coverage_ratio: float | None
    model_availability_status: str
    model_prediction_endpoint_mismatch_count: int
    model_snapshot_mismatch_runtime_event_count: int
    broker_event_with_multiple_dispositions_count: int
    allowed_same_event_safety_override_count: int
    unexpected_multiple_disposition_event_count: int
    maximum_dispositions_per_broker_event: int
    contiguity_warmup_event_count: int
    historical_backfill_event_count: int
    historical_backfill_exposure_observed_count: int
    historical_backfill_order_count: int
    model_unavailable_exposure_after_disposition_count: int
    maximum_gap_control_processing_delay_seconds: float | None
    maximum_completed_to_decision_lag_minutes: float | None
    maximum_current_broker_event_to_model_prediction_lag_minutes: float | None
    maximum_broker_event_to_model_prediction_lag_minutes: float | None
    stale_completed_event_count: int
    gap_decision_count: int
    order_event_rows: int
    control_execution_count: int
    strategy_execution_missing_decision_link_count: int
    invalid_control_execution_link_count: int
    unknown_execution_trigger_count: int
    maximum_broker_order_time_alignment_seconds: float | None
    broker_deal_rows: int
    broker_order_rows: int
    runtime_event_rows: int
    first_snapshot_utc: str | None
    last_snapshot_utc: str | None
    observed_hours: float | None
    expected_poll_seconds: int
    median_telemetry_interval_seconds: float | None
    maximum_telemetry_gap_seconds: float | None
    telemetry_gap_count_over_threshold: int
    telemetry_coverage_ratio: float | None
    terminal_connected_snapshot_rate: float | None
    broker_exposure_snapshot_rate: float | None
    worker_run_count: int
    worker_pid_count: int
    inferred_worker_restart_count: int
    initial_broker_position: int | None
    initial_pending_order_count: int | None
    starting_balance: float | None
    ending_balance: float | None
    starting_equity: float | None
    ending_equity: float | None
    balance_change: float | None
    net_equity_return: float | None
    maximum_equity_drawdown: float | None
    balance_pnl_reconciliation_difference: float | None
    final_equity_balance_difference: float | None
    realised_profit: float
    commission: float
    swap: float
    fee: float
    realised_net_pnl: float
    completed_trade_count: int
    winning_trade_count: int
    losing_trade_count: int
    breakeven_trade_count: int
    win_rate: float | None
    average_winning_trade: float | None
    average_losing_trade: float | None
    profit_factor: float | None
    total_order_volume_lots: float
    average_adverse_slippage_points: float | None
    maximum_adverse_slippage_points: float | None
    average_order_spread_points: float | None
    maximum_order_spread_points: float | None
    average_telemetry_spread_points: float | None
    maximum_telemetry_spread_points: float | None
    distinct_decision_actions: int
    blocked_decision_count: int
    policy_cap_block_count: int
    capped_exit_allowed_count: int
    close_only_reversal_count: int
    reconciliation_incident_count: int
    reconciliation_nonpass_snapshot_count: int
    daily_stop_trigger_count: int
    total_stop_trigger_count: int
    worker_error_count: int
    successful_order_event_count: int
    successful_order_event_missing_order_ticket_count: int
    successful_order_event_missing_deal_ticket_count: int
    successful_order_missing_fill_price_count: int
    broker_fill_price_recovered_count: int
    missing_broker_order_link_count: int
    missing_broker_deal_link_count: int
    broker_order_missing_execution_link_count: int
    broker_deal_missing_execution_link_count: int
    recovered_deal_link_by_order_count: int
    duplicate_snapshot_id_count: int
    duplicate_decision_id_count: int
    duplicate_execution_id_count: int
    duplicate_broker_deal_key_count: int
    duplicate_broker_order_key_count: int
    expected_decision_rows_from_state: int | None
    decision_record_count_mismatch: int | None
    expected_order_send_calls_from_state: int | None
    order_event_count_mismatch: int | None
    expected_successful_order_sends_from_state: int | None
    successful_order_event_count_mismatch: int | None
    final_broker_position: int | None
    final_pending_order_count: int | None
    final_worker_status: str | None
    final_worker_formal_gate: bool | None
    daily_return_observations: int
    descriptive_daily_sharpe_annualised_252: float | None
    sharpe_limitation: str


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise LiveAuditAnalysisError(f"Unable to read {path}: {exc}") from exc


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise LiveAuditAnalysisError(f"Unable to read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise LiveAuditAnalysisError(f"Expected JSON object in {path}")
    return raw


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise LiveAuditAnalysisError(f"{source} is missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="boolean")
    values = frame[column]
    if str(values.dtype) in {"bool", "boolean"}:
        return values.astype("boolean")
    mapped = values.astype(str).str.strip().str.lower().map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
            "none": pd.NA,
            "nan": pd.NA,
            "": pd.NA,
        }
    )
    return mapped.astype("boolean")


def _first_last(series: pd.Series) -> tuple[float | None, float | None]:
    clean = series.dropna()
    if clean.empty:
        return None, None
    return float(clean.iloc[0]), float(clean.iloc[-1])


def _last_int(series: pd.Series) -> int | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return None if clean.empty else int(clean.iloc[-1])


def _maximum_drawdown(equity: pd.Series) -> float | None:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if values.empty or bool((values <= 0).any()):
        return None
    running_peak = values.cummax()
    return float((values / running_peak - 1.0).min())


def _daily_equity_returns(telemetry: pd.DataFrame) -> pd.Series:
    """Return one first-to-last equity return per observed UTC date."""

    if telemetry.empty:
        return pd.Series(dtype="float64")
    _require_columns(telemetry, ("snapshot_utc", "equity"), source="telemetry.csv")
    frame = telemetry.loc[:, ["snapshot_utc", "equity"]].copy()
    frame["snapshot_utc"] = pd.to_datetime(
        frame["snapshot_utc"], utc=True, errors="coerce", format="mixed"
    )
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_utc", "equity"]).sort_values(
        "snapshot_utc"
    )
    if frame.empty:
        return pd.Series(dtype="float64")
    frame["utc_date"] = frame["snapshot_utc"].dt.strftime("%Y-%m-%d")
    daily = frame.groupby("utc_date", sort=True)["equity"].agg(["first", "last"])
    daily = daily[daily["first"] > 0.0]
    returns = daily["last"] / daily["first"] - 1.0
    returns.name = "daily_equity_return"
    return returns.astype("float64")


def _descriptive_sharpe(daily_returns: pd.Series) -> float | None:
    values = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(values) < 2:
        return None
    standard_deviation = float(values.std(ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return None
    return float(values.mean() / standard_deviation * math.sqrt(252.0))


def _duplicate_count(frame: pd.DataFrame, key: str) -> int:
    if frame.empty or key not in frame.columns:
        return 0
    values = frame[key].fillna("").astype(str)
    values = values[values != ""]
    return int(values.duplicated().sum())


def _ticket_set(frame: pd.DataFrame, column: str) -> set[int]:
    values = _numeric(frame, column).dropna()
    return {int(value) for value in values if int(value) != 0}


_ALLOWED_SAME_EVENT_FLATTEN_ACTIONS = frozenset(
    {
        "KILL_SWITCH_FLATTEN",
        "TOTAL_STOP_FLATTEN",
        "DAILY_STOP_FLATTEN",
        "SESSION_GAP_LOCKOUT_FLATTEN",
    }
)
_ALLOWED_SAME_EVENT_PRECEDING_EXPOSED_ACTIONS = frozenset(
    {
        "ENTER_LONG",
        "ENTER_SHORT",
        "HOLD_LONG",
        "HOLD_SHORT",
        "REVERSE_LONG_TO_SHORT",
        "REVERSE_SHORT_TO_LONG",
        "BLOCK_MINIMUM_HOLD",
        "BLOCK_DAILY_POLICY_CAP",
        "BLOCK_INVALID_SIGNAL",
        "BLOCK_SPREAD",
        "BLOCK_RECONCILIATION",
    }
)
_HISTORICAL_BACKFILL_ACTION = "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL"


def _same_event_disposition_counts(
    decisions: pd.DataFrame,
) -> tuple[int, int, int, int]:
    """Classify multiple decisions sharing one completed broker event.

    One later safety-control flatten is legitimate when it closes exposure on
    an already-processed event between bar completions. Virtual continuity is
    required in both modes. Broker continuity is mode-aware: live evidence
    must show the same exposure before the flatten and zero afterward, while
    shadow evidence must remain explicitly broker-flat throughout.
    """

    if decisions.empty or "event_time_utc" not in decisions.columns:
        return 0, 0, 0, 0
    work = decisions.copy()
    work["_event_time"] = pd.to_datetime(
        work["event_time_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    work = work.dropna(subset=["_event_time"])
    if work.empty:
        return 0, 0, 0, 0
    work["_decision_time"] = pd.to_datetime(
        work.get(
            "decision_utc",
            pd.Series(index=work.index, dtype="object"),
        ),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    work["_decision_id"] = (
        work.get(
            "decision_id",
            pd.Series("", index=work.index, dtype="object"),
        )
        .fillna("")
        .astype(str)
    )
    work["_action"] = (
        work.get(
            "action",
            pd.Series("", index=work.index, dtype="object"),
        )
        .fillna("")
        .astype(str)
        .str.upper()
    )
    work["_execution_mode"] = (
        work.get(
            "execution_mode",
            pd.Series("", index=work.index, dtype="object"),
        )
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    work["_position_before"] = pd.to_numeric(
        work.get(
            "position_before",
            pd.Series(index=work.index, dtype="float64"),
        ),
        errors="coerce",
    )
    work["_target_position"] = pd.to_numeric(
        work.get(
            "target_position",
            pd.Series(index=work.index, dtype="float64"),
        ),
        errors="coerce",
    )
    work["_broker_before"] = pd.to_numeric(
        work.get(
            "broker_position_before",
            pd.Series(index=work.index, dtype="float64"),
        ),
        errors="coerce",
    )
    work["_broker_after"] = pd.to_numeric(
        work.get(
            "broker_position_after_inspection",
            pd.Series(index=work.index, dtype="float64"),
        ),
        errors="coerce",
    )
    multiple_event_count = 0
    allowed_override_count = 0
    unexpected_event_count = 0
    maximum_dispositions = 0

    for _, group in work.groupby("_event_time", sort=True):
        count = int(len(group))
        maximum_dispositions = max(maximum_dispositions, count)
        if count <= 1:
            continue

        multiple_event_count += 1
        if count != 2:
            unexpected_event_count += 1
            continue
        ordered = group.sort_values(
            ["_decision_time"],
            kind="stable",
            na_position="last",
        )
        first = ordered.iloc[0]
        later = ordered.iloc[1]
        ids_are_distinct = bool(
            first["_decision_id"]
            and later["_decision_id"]
            and first["_decision_id"] != later["_decision_id"]
        )
        times_are_ordered = bool(
            pd.notna(first["_decision_time"])
            and pd.notna(later["_decision_time"])
            and later["_decision_time"] > first["_decision_time"]
        )
        execution_mode = str(first["_execution_mode"])
        execution_modes_are_valid_and_equal = bool(
            execution_mode in {"live", "shadow"}
            and str(later["_execution_mode"]) == execution_mode
        )
        recognised_later_flatten = bool(
            later["_action"] in _ALLOWED_SAME_EVENT_FLATTEN_ACTIONS
        )
        recognised_first_exposed_action = bool(
            first["_action"] in _ALLOWED_SAME_EVENT_PRECEDING_EXPOSED_ACTIONS
        )
        no_backfill_collision = bool(
            first["_action"] != _HISTORICAL_BACKFILL_ACTION
            and later["_action"] != _HISTORICAL_BACKFILL_ACTION
        )
        later_position_is_valid_exposure = bool(
            pd.notna(later["_position_before"])
            and float(later["_position_before"]) in {-1.0, 1.0}
        )
        virtual_position_continuity = bool(
            later_position_is_valid_exposure
            and pd.notna(first["_target_position"])
            and float(first["_target_position"])
            == float(later["_position_before"])
        )
        closes_existing_virtual_exposure = bool(
            later_position_is_valid_exposure
            and pd.notna(later["_target_position"])
            and float(later["_target_position"]) == 0.0
        )
        broker_continuity_is_valid = False
        if execution_modes_are_valid_and_equal:
            broker_values_present = bool(
                pd.notna(first["_broker_after"])
                and pd.notna(later["_broker_before"])
                and pd.notna(later["_broker_after"])
            )
            if broker_values_present and execution_mode == "live":
                exposure = float(later["_position_before"])
                broker_continuity_is_valid = bool(
                    float(first["_broker_after"]) == exposure
                    and float(later["_broker_before"]) == exposure
                    and float(later["_broker_after"]) == 0.0
                )
            elif broker_values_present and execution_mode == "shadow":
                first_broker_before_present = pd.notna(first["_broker_before"])
                broker_continuity_is_valid = bool(
                    first_broker_before_present
                    and float(first["_broker_before"]) == 0.0
                    and float(first["_broker_after"]) == 0.0
                    and float(later["_broker_before"]) == 0.0
                    and float(later["_broker_after"]) == 0.0
                )
        if all(
            (
                ids_are_distinct,
                times_are_ordered,
                execution_modes_are_valid_and_equal,
                recognised_later_flatten,
                recognised_first_exposed_action,
                no_backfill_collision,
                virtual_position_continuity,
                closes_existing_virtual_exposure,
                broker_continuity_is_valid,
            )
        ):
            allowed_override_count += 1
        else:
            unexpected_event_count += 1
    return (
        multiple_event_count,
        allowed_override_count,
        unexpected_event_count,
        maximum_dispositions,
    )




_CURRENT_DECISION_SCHEMA_VERSION = "1.0"
_CURRENT_AUDIT_SCHEMA_VERSION = "1.0"
_VALID_DECISION_ROLES = frozenset({"model_a", "model_b"})
_VALID_EXECUTION_MODES = frozenset({"live", "shadow"})
_VALID_POSITIONS = frozenset({-1, 0, 1})
_M15_DELTA = pd.Timedelta(minutes=15)
_REQUIRED_DECISION_EVIDENCE_COLUMNS = (
    "schema_version",
    "decision_id",
    "role",
    "run_id",
    "iteration",
    "event_time_utc",
    "decision_utc",
    "execution_mode",
    "probability_up",
    "model_prediction_available",
    "model_prediction_event_time_utc",
    "model_unavailable_reason",
    "broker_event_disposition",
    "latest_completed_bar_time_utc",
    "event_is_latest_feature",
    "event_is_latest_completed_bar",
    "action",
    "reason",
    "position_before",
    "desired_position",
    "target_position",
    "duplicate_event",
    "gap_from_previous_event",
    "stale_event_warning",
    "policy_cap_reached",
    "entry_blocked_by_policy_cap",
    "exit_allowed_when_capped",
    "close_only_reversal",
    "daily_stop_active",
    "total_stop_active",
    "kill_switch_active",
    "reconciliation_status",
    "broker_position_before",
    "broker_position_after_inspection",
    "order_check_called",
    "order_check_passed",
    "order_send_called",
    "order_send_passed",
    "broker_position_after",
)
_OPTIONAL_DECISION_TIMESTAMP_COLUMNS = (
    "model_prediction_event_time_utc",
    "signal_window_start_utc",
    "signal_window_end_utc",
    "latest_completed_bar_time_utc",
)
_KNOWN_DECISION_ACTIONS = frozenset(
    {
        "ENTER_LONG",
        "ENTER_SHORT",
        "HOLD_FLAT",
        "HOLD_LONG",
        "HOLD_SHORT",
        "EXIT_POSITION",
        "REVERSE_LONG_TO_SHORT",
        "REVERSE_SHORT_TO_LONG",
        "BLOCK_MINIMUM_HOLD",
        "BLOCK_DAILY_POLICY_CAP",
        "CLOSE_ONLY_DAILY_POLICY_CAP",
        "EXIT_POSITION_CAP_REACHED",
        "BLOCK_DAILY_ENTRY_CAP",
        "BLOCK_RECONCILIATION",
        "BLOCK_INVALID_SIGNAL",
        "BLOCK_SPREAD",
        "PARTIAL_REVERSAL_FLAT",
        "KILL_SWITCH_FLATTEN",
        "KILL_SWITCH_BLOCK",
        "TOTAL_STOP_FLATTEN",
        "TOTAL_STOP_BLOCK",
        "DAILY_STOP_FLATTEN",
        "DAILY_STOP_BLOCK",
        "SESSION_GAP_LOCKOUT_FLATTEN",
        "BLOCK_SESSION_GAP_LOCKOUT",
        "CONTROL_FRESH_START_FLATTEN",
        "CONTROL_FRESH_START_BLOCK",
        "CONTROL_GAP_FLATTEN",
        "CONTROL_GAP_BLOCK",
        "CONTROL_MODEL_SNAPSHOT_MISMATCH_FLATTEN",
        "CONTROL_MODEL_SNAPSHOT_MISMATCH_BLOCK",
        "CONTROL_MODEL_UNAVAILABLE_FLATTEN",
        "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP",
        "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL",
    }
)
_MODEL_A_ONLY_ACTIONS = frozenset(
    {
        "ENTER_SHORT",
        "HOLD_SHORT",
        "REVERSE_LONG_TO_SHORT",
        "REVERSE_SHORT_TO_LONG",
        "BLOCK_MINIMUM_HOLD",
        "BLOCK_DAILY_POLICY_CAP",
        "CLOSE_ONLY_DAILY_POLICY_CAP",
        "EXIT_POSITION_CAP_REACHED",
    }
)
_MODEL_B_ONLY_ACTIONS = frozenset({"BLOCK_DAILY_ENTRY_CAP"})
_RISK_CONTROL_FLAG_BY_ACTION = {
    "KILL_SWITCH_FLATTEN": "kill_switch_active",
    "KILL_SWITCH_BLOCK": "kill_switch_active",
    "TOTAL_STOP_FLATTEN": "total_stop_active",
    "TOTAL_STOP_BLOCK": "total_stop_active",
    "DAILY_STOP_FLATTEN": "daily_stop_active",
    "DAILY_STOP_BLOCK": "daily_stop_active",
}
_CONTROL_TRIGGER_BY_ACTION = {
    "KILL_SWITCH_FLATTEN": "CONTROL_KILL_SWITCH",
    "TOTAL_STOP_FLATTEN": "CONTROL_TOTAL_STOP",
    "DAILY_STOP_FLATTEN": "CONTROL_DAILY_STOP",
    "SESSION_GAP_LOCKOUT_FLATTEN": "CONTROL_SESSION_GAP_LOCKOUT",
}
_CONTROL_REASON_BY_ACTION = {
    "KILL_SWITCH_FLATTEN": "emergency_kill_switch_active",
    "KILL_SWITCH_BLOCK": "emergency_kill_switch_active",
    "TOTAL_STOP_FLATTEN": "total_drawdown_stop_active",
    "TOTAL_STOP_BLOCK": "total_drawdown_stop_active",
    "DAILY_STOP_FLATTEN": "daily_loss_stop_active_until_next_utc_day",
    "DAILY_STOP_BLOCK": "daily_loss_stop_active_until_next_utc_day",
    "CONTROL_FRESH_START_FLATTEN": "fresh_runtime_first_broker_event_adopt_only",
    "CONTROL_FRESH_START_BLOCK": "fresh_runtime_first_broker_event_adopt_only",
    "CONTROL_GAP_FLATTEN": "non_contiguous_completed_m15_broker_event",
    "CONTROL_GAP_BLOCK": "non_contiguous_completed_m15_broker_event",
    "CONTROL_MODEL_SNAPSHOT_MISMATCH_FLATTEN": (
        "broker_and_model_completed_event_endpoints_differ"
    ),
    "CONTROL_MODEL_SNAPSHOT_MISMATCH_BLOCK": (
        "broker_and_model_completed_event_endpoints_differ"
    ),
    "CONTROL_MODEL_UNAVAILABLE_FLATTEN": (
        "frozen_48_bar_contiguous_sequence_unavailable"
    ),
    "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP": (
        "frozen_48_bar_contiguous_sequence_unavailable"
    ),
    "BLOCK_DAILY_ENTRY_CAP": (
        "maximum_successful_new_entries_per_utc_day_reached"
    ),
}
_SESSION_GAP_REASONS = frozenset(
    {
        "expected_broker_session_gap_lockout",
        "weekend_market_lockout_saturday",
        "weekend_market_lockout_before_sunday_reopen",
        "weekend_market_lockout_friday_preclose",
        "daily_market_break_lockout",
    }
)
_MODEL_UNAVAILABLE_ACTIONS = frozenset(
    {
        "CONTROL_MODEL_SNAPSHOT_MISMATCH_FLATTEN",
        "CONTROL_MODEL_SNAPSHOT_MISMATCH_BLOCK",
        "CONTROL_MODEL_UNAVAILABLE_FLATTEN",
        "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP",
        "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL",
        "CONTROL_FRESH_START_FLATTEN",
        "CONTROL_FRESH_START_BLOCK",
    }
)
_NORMAL_MODEL_ACTIONS = frozenset(
    {
        "ENTER_LONG",
        "ENTER_SHORT",
        "HOLD_FLAT",
        "HOLD_LONG",
        "HOLD_SHORT",
        "EXIT_POSITION",
        "REVERSE_LONG_TO_SHORT",
        "REVERSE_SHORT_TO_LONG",
        "BLOCK_MINIMUM_HOLD",
        "BLOCK_DAILY_POLICY_CAP",
        "CLOSE_ONLY_DAILY_POLICY_CAP",
        "EXIT_POSITION_CAP_REACHED",
        "BLOCK_DAILY_ENTRY_CAP",
        "BLOCK_SPREAD",
        "PARTIAL_REVERSAL_FLAT",
    }
)


@dataclass(frozen=True)
class DecisionEvidenceValidation:
    invalid_row_count: int
    issue_count: int
    reason_counts: tuple[str, ...]


def _audit_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _audit_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    ):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    text = _audit_text(value).lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    return None


def _audit_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _audit_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _audit_timestamp(value: Any) -> pd.Timestamp | None:
    text = _audit_text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _is_m15_grid_timestamp(value: pd.Timestamp) -> bool:
    utc = value.tz_convert("UTC")
    return bool(
        utc.minute % 15 == 0
        and utc.second == 0
        and utc.microsecond == 0
        and utc.nanosecond == 0
    )


def _decision_action_contract_is_valid(
    *,
    action: str,
    role: str,
    before: int,
    desired: int,
    target: int,
) -> bool:
    if action == "ENTER_LONG":
        return (before, desired, target) == (0, 1, 1)
    if action == "ENTER_SHORT":
        return role == "model_a" and (before, desired, target) == (0, -1, -1)
    if action == "HOLD_FLAT":
        return (before, desired, target) == (0, 0, 0)
    if action == "HOLD_LONG":
        return (before, desired, target) == (1, 1, 1)
    if action == "HOLD_SHORT":
        return role == "model_a" and (before, desired, target) == (-1, -1, -1)
    if action == "EXIT_POSITION":
        return before in {-1, 1} and desired == 0 and target == 0
    if action == "REVERSE_LONG_TO_SHORT":
        return role == "model_a" and (before, desired, target) == (1, -1, -1)
    if action == "REVERSE_SHORT_TO_LONG":
        return role == "model_a" and (before, desired, target) == (-1, 1, 1)
    if action in {"BLOCK_MINIMUM_HOLD", "BLOCK_DAILY_POLICY_CAP"}:
        return role == "model_a" and desired != before and target == before
    if action == "CLOSE_ONLY_DAILY_POLICY_CAP":
        return role == "model_a" and before in {-1, 1} and desired == -before and target == 0
    if action == "EXIT_POSITION_CAP_REACHED":
        return role == "model_a" and before in {-1, 1} and desired == 0 and target == 0
    if action == "BLOCK_DAILY_ENTRY_CAP":
        return role == "model_b" and (before, desired, target) == (0, 1, 0)
    if action in {"BLOCK_RECONCILIATION", "BLOCK_INVALID_SIGNAL"}:
        return desired == before and target == before
    if action == "BLOCK_SPREAD":
        return desired != before and target == before
    if action == "PARTIAL_REVERSAL_FLAT":
        return before in {-1, 1} and desired == -before and target == 0
    if action in {
        "KILL_SWITCH_FLATTEN",
        "TOTAL_STOP_FLATTEN",
        "DAILY_STOP_FLATTEN",
        "SESSION_GAP_LOCKOUT_FLATTEN",
        "CONTROL_FRESH_START_FLATTEN",
        "CONTROL_GAP_FLATTEN",
        "CONTROL_MODEL_SNAPSHOT_MISMATCH_FLATTEN",
        "CONTROL_MODEL_UNAVAILABLE_FLATTEN",
    }:
        return before in {-1, 1} and desired == 0 and target == 0
    if action in {
        "KILL_SWITCH_BLOCK",
        "TOTAL_STOP_BLOCK",
        "DAILY_STOP_BLOCK",
        "BLOCK_SESSION_GAP_LOCKOUT",
        "CONTROL_FRESH_START_BLOCK",
        "CONTROL_GAP_BLOCK",
        "CONTROL_MODEL_SNAPSHOT_MISMATCH_BLOCK",
        "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP",
    }:
        return (before, desired, target) == (0, 0, 0)
    if action == "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL":
        return desired == before and target == before
    return False


def _expected_execution_trigger(action: str) -> str:
    if action in _CONTROL_TRIGGER_BY_ACTION:
        return _CONTROL_TRIGGER_BY_ACTION[action]
    if action.startswith("CONTROL_"):
        return action
    return "STRATEGY_DECISION"


def _validate_decision_evidence(
    decisions: pd.DataFrame,
    order_events: pd.DataFrame,
    runtime_events: pd.DataFrame,
    *,
    role: str,
    telemetry_run_ids: set[str],
    observation_start_utc: pd.Timestamp | None,
    observation_end_utc: pd.Timestamp | None,
) -> DecisionEvidenceValidation:
    """Validate persisted decision meaning without interrupting report output.

    Every problem is accumulated as an evidence issue.  The caller can fail the
    formal gate while still writing all daily and consolidated report artefacts.
    """

    reason_counts: dict[str, int] = {}
    invalid_rows: set[int] = set()

    def add(index: int, reason: str) -> None:
        invalid_rows.add(index)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    missing_columns = [
        column
        for column in _REQUIRED_DECISION_EVIDENCE_COLUMNS
        if column not in decisions.columns
    ]
    if missing_columns:
        affected = max(1, len(decisions))
        for column in missing_columns:
            reason_counts[f"missing_required_column:{column}"] = affected
        invalid_rows.update(range(len(decisions)))

    runtime_event_types = set(
        runtime_events.get(
            "event_type", pd.Series(dtype="object")
        ).fillna("").astype(str).str.strip().str.upper()
    )

    successful_orders = order_events.copy()
    if not successful_orders.empty:
        successful_orders = successful_orders.loc[
            _boolean(successful_orders, "order_send_passed")
            .reindex(successful_orders.index)
            .fillna(False)
            .astype(bool)
        ].copy()
        successful_orders["_role"] = successful_orders.get(
            "role", pd.Series("", index=successful_orders.index)
        ).fillna("").astype(str).str.strip().str.lower()
        successful_orders["_run_id"] = successful_orders.get(
            "run_id", pd.Series("", index=successful_orders.index)
        ).fillna("").astype(str).str.strip()
        successful_orders["_trigger"] = successful_orders.get(
            "trigger_type", pd.Series("", index=successful_orders.index)
        ).fillna("").astype(str).str.strip().str.upper()
        successful_orders["_decision_id"] = successful_orders.get(
            "decision_id", pd.Series("", index=successful_orders.index)
        ).fillna("").astype(str).str.strip()
        successful_orders["_event_time"] = pd.to_datetime(
            successful_orders.get(
                "event_time_utc",
                pd.Series(index=successful_orders.index, dtype="object"),
            ),
            utc=True,
            errors="coerce",
            format="mixed",
        )
        successful_orders["_completed_time"] = pd.to_datetime(
            successful_orders.get(
                "completed_utc",
                pd.Series(index=successful_orders.index, dtype="object"),
            ),
            utc=True,
            errors="coerce",
            format="mixed",
        )
        successful_orders["_position_after"] = pd.to_numeric(
            successful_orders.get(
                "position_after",
                pd.Series(index=successful_orders.index, dtype="float64"),
            ),
            errors="coerce",
        )

    for offset, (_, row) in enumerate(decisions.iterrows()):
        index = int(offset)
        schema = _audit_text(row.get("schema_version"))
        decision_id = _audit_text(row.get("decision_id"))
        row_role = _audit_text(row.get("role")).lower()
        run_id = _audit_text(row.get("run_id"))
        iteration = _audit_int(row.get("iteration"))
        mode = _audit_text(row.get("execution_mode")).lower()
        action = _audit_text(row.get("action")).upper()
        reason = _audit_text(row.get("reason"))
        disposition = _audit_text(row.get("broker_event_disposition")).upper()
        event_time = _audit_timestamp(row.get("event_time_utc"))
        decision_time = _audit_timestamp(row.get("decision_utc"))
        probability_up = _audit_float(row.get("probability_up"))
        prediction_available = _audit_bool(row.get("model_prediction_available"))
        prediction_event_time = _audit_timestamp(
            row.get("model_prediction_event_time_utc")
        )
        latest_completed_bar_time = _audit_timestamp(
            row.get("latest_completed_bar_time_utc")
        )
        event_is_latest_feature = _audit_bool(row.get("event_is_latest_feature"))
        event_is_latest_completed = _audit_bool(
            row.get("event_is_latest_completed_bar")
        )
        model_unavailable_reason = _audit_text(row.get("model_unavailable_reason"))
        before = _audit_int(row.get("position_before"))
        desired = _audit_int(row.get("desired_position"))
        target = _audit_int(row.get("target_position"))
        broker_before = _audit_int(row.get("broker_position_before"))
        broker_after = _audit_int(row.get("broker_position_after_inspection"))
        broker_after_execution = _audit_int(row.get("broker_position_after"))
        duplicate = _audit_bool(row.get("duplicate_event"))
        gap = _audit_bool(row.get("gap_from_previous_event"))
        stale = _audit_bool(row.get("stale_event_warning"))
        order_check_called = _audit_bool(row.get("order_check_called"))
        order_check_passed = _audit_bool(row.get("order_check_passed"))
        order_send_called = _audit_bool(row.get("order_send_called"))
        order_send_passed = _audit_bool(row.get("order_send_passed"))

        if schema != _CURRENT_DECISION_SCHEMA_VERSION:
            add(index, "invalid_schema_version")
        if not decision_id:
            add(index, "blank_decision_id")
        if row_role not in _VALID_DECISION_ROLES or row_role != role:
            add(index, "role_mismatch")
        if not run_id or run_id not in telemetry_run_ids:
            add(index, "run_id_not_in_telemetry")
        if iteration is None or iteration < 0:
            add(index, "invalid_iteration")
        if mode not in _VALID_EXECUTION_MODES:
            add(index, "invalid_execution_mode")
        if event_time is None:
            add(index, "invalid_event_time_utc")
        elif not _is_m15_grid_timestamp(event_time):
            add(index, "event_time_not_on_m15_grid")
        if decision_time is None:
            add(index, "invalid_decision_utc")
        elif event_time is not None and decision_time < event_time + _M15_DELTA:
            add(index, "decision_before_m15_completion")
        if (
            decision_time is not None
            and observation_start_utc is not None
            and decision_time < observation_start_utc
        ):
            add(index, "decision_before_observation_window")
        if (
            decision_time is not None
            and observation_end_utc is not None
            and decision_time > observation_end_utc
        ):
            add(index, "decision_after_observation_window")
        if (
            event_time is not None
            and run_id
            and iteration is not None
            and decision_id
        ):
            expected_decision_id = (
                historical_backfill_identifier(row_role, event_time.isoformat())
                if action == _HISTORICAL_BACKFILL_ACTION
                else decision_identifier(
                    row_role,
                    event_time.isoformat(),
                    run_id=run_id,
                    iteration=iteration,
                )
            )
            if decision_id != expected_decision_id:
                add(index, "noncanonical_decision_id")
        for column in _OPTIONAL_DECISION_TIMESTAMP_COLUMNS:
            raw = row.get(column)
            if _audit_text(raw) and _audit_timestamp(raw) is None:
                add(index, f"invalid_optional_timestamp:{column}")
        if action not in _KNOWN_DECISION_ACTIONS:
            add(index, "unknown_action")
        expected_reason = _CONTROL_REASON_BY_ACTION.get(action)
        if expected_reason is not None and reason != expected_reason:
            add(index, "invalid_action_reason")
        if action in {
            "SESSION_GAP_LOCKOUT_FLATTEN",
            "BLOCK_SESSION_GAP_LOCKOUT",
        }:
            if reason not in _SESSION_GAP_REASONS:
                add(index, "invalid_session_gap_reason")
            if "SESSION_GAP_LOCKOUT_STARTED" not in runtime_event_types:
                add(index, "session_gap_action_without_runtime_activation")
        if disposition != action:
            add(index, "broker_event_disposition_mismatch")
        if duplicate is not False:
            add(index, "persisted_duplicate_event_not_false")
        if gap is None:
            add(index, "invalid_gap_flag")
        if stale is None:
            add(index, "invalid_stale_flag")
        if any(value not in _VALID_POSITIONS for value in (before, desired, target)):
            add(index, "invalid_virtual_position_domain")
        elif action in _KNOWN_DECISION_ACTIONS and not _decision_action_contract_is_valid(
            action=action,
            role=row_role,
            before=before,
            desired=desired,
            target=target,
        ):
            add(index, "invalid_action_position_contract")
        if row_role == "model_b" and any(
            value == -1 for value in (before, desired, target)
        ):
            add(index, "model_b_short_exposure")
        if row_role == "model_b" and action in _MODEL_A_ONLY_ACTIONS:
            add(index, "model_b_model_a_only_action")
        if row_role == "model_a" and action in _MODEL_B_ONLY_ACTIONS:
            add(index, "model_a_model_b_only_action")
        if action == "BLOCK_SPREAD" and mode != "live":
            add(index, "block_spread_not_live")
        if action == "PARTIAL_REVERSAL_FLAT" and mode != "live":
            add(index, "partial_reversal_not_live")
        for controlled_action, flag_column in _RISK_CONTROL_FLAG_BY_ACTION.items():
            if action == controlled_action and _audit_bool(row.get(flag_column)) is not True:
                add(index, f"inactive_control_flag:{flag_column}")
        kill_switch_active = _audit_bool(row.get("kill_switch_active"))
        total_stop_active = _audit_bool(row.get("total_stop_active"))
        daily_stop_active = _audit_bool(row.get("daily_stop_active"))
        if kill_switch_active is True and action not in {
            "KILL_SWITCH_FLATTEN",
            "KILL_SWITCH_BLOCK",
        }:
            add(index, "active_kill_switch_without_kill_action")
        if (
            kill_switch_active is not True
            and total_stop_active is True
            and action not in {"TOTAL_STOP_FLATTEN", "TOTAL_STOP_BLOCK"}
        ):
            add(index, "active_total_stop_without_total_action")
        if (
            kill_switch_active is not True
            and total_stop_active is not True
            and daily_stop_active is True
            and action not in {"DAILY_STOP_FLATTEN", "DAILY_STOP_BLOCK"}
        ):
            add(index, "active_daily_stop_without_daily_action")
        if action.startswith("DAILY_STOP_") and "DAILY_STOP_TRIGGERED" not in runtime_event_types:
            add(index, "daily_stop_action_without_runtime_trigger")
        if action.startswith("TOTAL_STOP_") and "TOTAL_STOP_TRIGGERED" not in runtime_event_types:
            add(index, "total_stop_action_without_runtime_trigger")
        if action == "BLOCK_RECONCILIATION":
            status = _audit_text(row.get("reconciliation_status")).upper()
            if not status or status.startswith("PASS") or "FLAT_CONFIRMED" in status:
                add(index, "reconciliation_block_without_nonpass_status")
        if action in {"CONTROL_GAP_FLATTEN", "CONTROL_GAP_BLOCK"} and gap is not True:
            add(index, "gap_control_without_gap_flag")
        if action in {
            "CONTROL_FRESH_START_FLATTEN",
            "CONTROL_FRESH_START_BLOCK",
            "CONTROL_MODEL_SNAPSHOT_MISMATCH_FLATTEN",
            "CONTROL_MODEL_SNAPSHOT_MISMATCH_BLOCK",
            "CONTROL_MODEL_UNAVAILABLE_FLATTEN",
            "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP",
            "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL",
        } and stale is not True:
            add(index, "stale_control_without_stale_flag")
        if action == "CLOSE_ONLY_DAILY_POLICY_CAP":
            for column in (
                "policy_cap_reached",
                "entry_blocked_by_policy_cap",
                "exit_allowed_when_capped",
                "close_only_reversal",
            ):
                if _audit_bool(row.get(column)) is not True:
                    add(index, f"invalid_policy_flag:{column}")
        if action == "EXIT_POSITION_CAP_REACHED":
            for column in ("policy_cap_reached", "exit_allowed_when_capped"):
                if _audit_bool(row.get(column)) is not True:
                    add(index, f"invalid_policy_flag:{column}")
        if action == "BLOCK_DAILY_POLICY_CAP":
            for column in ("policy_cap_reached", "entry_blocked_by_policy_cap"):
                if _audit_bool(row.get(column)) is not True:
                    add(index, f"invalid_policy_flag:{column}")
        if action == "BLOCK_DAILY_ENTRY_CAP":
            if _audit_bool(row.get("entry_blocked_by_policy_cap")) is not True:
                add(index, "invalid_policy_flag:entry_blocked_by_policy_cap")

        if prediction_available is None:
            add(index, "invalid_model_prediction_available")
        elif prediction_available:
            if probability_up is None or not 0.0 <= probability_up <= 1.0:
                add(index, "invalid_available_probability")
            if prediction_event_time is None:
                add(index, "available_prediction_missing_endpoint")
            elif event_time is not None and prediction_event_time != event_time:
                add(index, "available_prediction_endpoint_mismatch")
            if latest_completed_bar_time is None:
                add(index, "available_prediction_missing_latest_completed_bar")
            elif event_time is not None and latest_completed_bar_time != event_time:
                add(index, "available_latest_completed_bar_mismatch")
            if event_is_latest_feature is not True:
                add(index, "available_prediction_not_latest_feature")
            if event_is_latest_completed is not True:
                add(index, "available_prediction_not_latest_completed_bar")
            if stale is not False:
                add(index, "available_prediction_marked_stale")
            if model_unavailable_reason:
                add(index, "available_prediction_has_unavailable_reason")
            if action in _MODEL_UNAVAILABLE_ACTIONS:
                add(index, "unavailable_action_has_available_prediction")
        else:
            if _audit_text(row.get("probability_up")):
                add(index, "unavailable_prediction_has_probability")
            if action in _NORMAL_MODEL_ACTIONS:
                add(index, "normal_model_action_without_prediction")
            if action in _MODEL_UNAVAILABLE_ACTIONS and not model_unavailable_reason:
                add(index, "unavailable_action_missing_reason")
            if action != _HISTORICAL_BACKFILL_ACTION:
                if latest_completed_bar_time is None:
                    add(index, "unavailable_prediction_missing_latest_completed_bar")
                elif event_time is not None and latest_completed_bar_time != event_time:
                    add(index, "unavailable_latest_completed_bar_mismatch")

        if mode == "shadow":
            if broker_before != 0 or broker_after != 0:
                add(index, "shadow_broker_not_flat")
            if (
                order_check_called is not False
                or order_check_passed not in {None, False}
                or order_send_called is not False
                or order_send_passed not in {None, False}
                or broker_after_execution is not None
            ):
                add(index, "shadow_order_evidence_present_or_missing")
        elif mode == "live" and before in _VALID_POSITIONS and target in _VALID_POSITIONS:
            reconciliation_exception = action == "BLOCK_RECONCILIATION"
            if not reconciliation_exception and broker_before != before:
                add(index, "live_broker_before_mismatch")
            if not reconciliation_exception and broker_after != target:
                add(index, "live_broker_after_mismatch")
            changes_position = target != before
            if changes_position:
                if any(
                    value is not True
                    for value in (
                        order_check_called,
                        order_check_passed,
                        order_send_called,
                        order_send_passed,
                    )
                ):
                    add(index, "live_transition_missing_successful_order_flags")
                if broker_after_execution != target:
                    add(index, "live_transition_broker_position_after_mismatch")
                if event_time is not None and not successful_orders.empty:
                    expected_trigger = _expected_execution_trigger(action)
                    candidates = successful_orders.loc[
                        (successful_orders["_role"] == row_role)
                        & (successful_orders["_run_id"] == run_id)
                        & (successful_orders["_trigger"] == expected_trigger)
                        & (successful_orders["_event_time"] == event_time)
                    ]
                    if expected_trigger == "STRATEGY_DECISION":
                        candidates = candidates.loc[
                            candidates["_decision_id"] == decision_id
                        ]
                    else:
                        # Mirror execution_audit_rows exactly. Control IDs keep
                        # the original event text and remove only colons.
                        event_token = _audit_text(
                            row.get("event_time_utc")
                        ).replace(":", "")
                        expected_control_id = (
                            f"{expected_trigger}:{row_role}:{run_id}:"
                            f"{iteration}:{event_token}"
                        )
                        candidates = candidates.loc[
                            candidates["_decision_id"] == expected_control_id
                        ]
                    if candidates.empty:
                        add(index, "live_transition_missing_successful_order_event")
                else:
                    add(index, "live_transition_missing_successful_order_event")
            else:
                if (
                    order_check_called is not False
                    or order_check_passed not in {None, False}
                    or order_send_called is not False
                    or order_send_passed not in {None, False}
                    or broker_after_execution is not None
                ):
                    add(index, "nontransition_order_send_evidence")

    continuity = decisions.copy()
    if not continuity.empty:
        continuity["_event_time"] = pd.to_datetime(
            continuity.get("event_time_utc"), utc=True, errors="coerce", format="mixed"
        )
        continuity["_decision_time"] = pd.to_datetime(
            continuity.get("decision_utc"), utc=True, errors="coerce", format="mixed"
        )
        continuity["_action"] = continuity.get(
            "action", pd.Series("", index=continuity.index)
        ).fillna("").astype(str).str.upper()
        continuity = continuity.loc[
            continuity["_action"] != _HISTORICAL_BACKFILL_ACTION
        ].sort_values(["_event_time", "_decision_time"], kind="stable")
        previous: pd.Series | None = None
        for offset, (_, row) in enumerate(continuity.iterrows()):
            if previous is not None:
                previous_target = _audit_int(previous.get("target_position"))
                previous_broker_after = _audit_int(
                    previous.get("broker_position_after_inspection")
                )
                previous_decision_time = _audit_timestamp(
                    previous.get("decision_utc")
                )
                current_decision_time = _audit_timestamp(row.get("decision_utc"))
                if (
                    previous_decision_time is not None
                    and current_decision_time is not None
                    and not successful_orders.empty
                ):
                    intervening_controls = successful_orders.loc[
                        (successful_orders["_role"] == role)
                        & successful_orders["_trigger"].str.startswith("CONTROL_")
                        & (
                            successful_orders["_completed_time"]
                            >= previous_decision_time
                        )
                        & (
                            successful_orders["_completed_time"]
                            < current_decision_time
                        )
                        & successful_orders["_position_after"].notna()
                    ].sort_values("_completed_time", kind="stable")
                    if not intervening_controls.empty:
                        controlled_position = _audit_int(
                            intervening_controls.iloc[-1]["_position_after"]
                        )
                        if controlled_position in _VALID_POSITIONS:
                            previous_target = controlled_position
                            previous_broker_after = controlled_position
                current_before = _audit_int(row.get("position_before"))
                if previous_target is not None and current_before != previous_target:
                    add(offset, "virtual_position_discontinuity")
                previous_mode = _audit_text(previous.get("execution_mode")).lower()
                current_mode = _audit_text(row.get("execution_mode")).lower()
                current_broker_before = _audit_int(row.get("broker_position_before"))
                if (
                    previous_mode == current_mode == "live"
                    and _audit_text(row.get("action")).upper() != "BLOCK_RECONCILIATION"
                    and previous_broker_after is not None
                    and current_broker_before != previous_broker_after
                ):
                    add(offset, "broker_position_discontinuity")
            previous = row

    ordered_reasons = tuple(
        f"{reason}={count}"
        for reason, count in sorted(reason_counts.items())
    )
    return DecisionEvidenceValidation(
        invalid_row_count=len(invalid_rows),
        issue_count=int(sum(reason_counts.values())),
        reason_counts=ordered_reasons,
    )


def _validate_evidence_ownership(
    *,
    role: str,
    telemetry_run_ids: set[str],
    telemetry: pd.DataFrame,
    completed_broker_events: pd.DataFrame,
    order_events: pd.DataFrame,
    deals: pd.DataFrame,
    orders: pd.DataFrame,
    runtime_events: pd.DataFrame,
    final_report: dict[str, Any],
) -> tuple[str, ...]:
    """Fail closed when raw files do not belong to the audited role/run set."""

    issues: list[str] = []

    def validate_frame(
        frame: pd.DataFrame,
        source: str,
        *,
        required: bool,
        required_columns: tuple[str, ...] = (),
    ) -> None:
        if frame.empty:
            if required:
                issues.append(f"missing_required_evidence:{source}")
            return
        needed = ("schema_version", "role", "run_id", *required_columns)
        missing = [column for column in needed if column not in frame.columns]
        for column in missing:
            issues.append(f"{source}:missing_column:{column}")
        if missing:
            return
        invalid_schema = int(
            (
                frame["schema_version"].fillna("").astype(str).str.strip()
                != _CURRENT_AUDIT_SCHEMA_VERSION
            ).sum()
        )
        if invalid_schema:
            issues.append(f"{source}:invalid_schema={invalid_schema}")
        role_mismatch = int(
            (
                frame["role"].fillna("").astype(str).str.strip().str.lower()
                != role
            ).sum()
        )
        if role_mismatch:
            issues.append(f"{source}:role_mismatch={role_mismatch}")
        run_values = frame["run_id"].fillna("").astype(str).str.strip()
        invalid_run = int(
            ((run_values == "") | ~run_values.isin(telemetry_run_ids)).sum()
        )
        if invalid_run:
            issues.append(f"{source}:run_id_not_in_telemetry={invalid_run}")

    validate_frame(
        telemetry,
        "telemetry.csv",
        required=True,
        required_columns=(
            "snapshot_id",
            "snapshot_utc",
            "latest_completed_event_time_utc",
            "latest_decision_event_time_utc",
        ),
    )
    validate_frame(
        completed_broker_events,
        "completed_broker_events.csv",
        required=True,
        required_columns=(
            "broker_event_key",
            "event_time_utc",
            "first_observed_utc",
        ),
    )
    validate_frame(order_events, "order_events.csv", required=False)
    validate_frame(deals, "broker_deals.csv", required=False)
    validate_frame(orders, "broker_orders.csv", required=False)
    validate_frame(
        runtime_events,
        "runtime_events.csv",
        required=True,
        required_columns=("runtime_event_id", "timestamp_utc", "event_type"),
    )

    if not final_report:
        issues.append("missing_required_evidence:final_report.json")
    else:
        if _audit_text(final_report.get("schema_version")) != _CURRENT_AUDIT_SCHEMA_VERSION:
            issues.append("final_report.json:invalid_schema")
        if _audit_text(final_report.get("role")).lower() != role:
            issues.append("final_report.json:role_mismatch")
        final_run_id = _audit_text(final_report.get("run_id"))
        if not final_run_id or final_run_id not in telemetry_run_ids:
            issues.append("final_report.json:run_id_not_in_telemetry")
        if _audit_timestamp(final_report.get("started_utc")) is None:
            issues.append("final_report.json:invalid_started_utc")
        if _audit_timestamp(final_report.get("completed_utc")) is None:
            issues.append("final_report.json:invalid_completed_utc")

    if not completed_broker_events.empty and {
        "broker_event_key",
        "role",
        "event_time_utc",
    }.issubset(completed_broker_events.columns):
        for _, row in completed_broker_events.iterrows():
            event_time = _audit_timestamp(row.get("event_time_utc"))
            if event_time is None:
                issues.append("completed_broker_events.csv:invalid_event_time")
                continue
            if not _is_m15_grid_timestamp(event_time):
                issues.append("completed_broker_events.csv:event_not_on_m15_grid")
            expected_key = completed_broker_event_identifier(
                _audit_text(row.get("role")).lower(), event_time.isoformat()
            )
            if _audit_text(row.get("broker_event_key")) != expected_key:
                issues.append("completed_broker_events.csv:invalid_broker_event_key")
            first_observed = _audit_timestamp(row.get("first_observed_utc"))
            if first_observed is None:
                issues.append("completed_broker_events.csv:invalid_first_observed_utc")
            elif first_observed < event_time + _M15_DELTA:
                issues.append(
                    "completed_broker_events.csv:observed_before_m15_completion"
                )

    return tuple(issues)


def _validate_telemetry_event_lag(telemetry: pd.DataFrame) -> tuple[str, ...]:
    issues: list[str] = []
    required = (
        "latest_completed_event_time_utc",
        "latest_decision_event_time_utc",
    )
    if any(column not in telemetry.columns for column in required):
        return ("telemetry_event_lag_columns_missing",)
    for _, row in telemetry.iterrows():
        completed_text = _audit_text(row.get(required[0]))
        decision_text = _audit_text(row.get(required[1]))
        if not completed_text and not decision_text:
            continue
        if not completed_text or not decision_text:
            if (
                _audit_text(row.get("snapshot_phase")).upper() == "STARTUP"
                and completed_text
                and not decision_text
            ):
                # A restarted worker restores the last completed event from
                # state before it has a current-process StrategyDecision.
                continue
            issues.append("telemetry_event_lag_endpoint_missing")
            continue
        completed = _audit_timestamp(completed_text)
        decision = _audit_timestamp(decision_text)
        if completed is None or decision is None:
            issues.append("telemetry_event_lag_endpoint_invalid")
            continue
        if not _is_m15_grid_timestamp(completed) or not _is_m15_grid_timestamp(
            decision
        ):
            issues.append("telemetry_event_lag_endpoint_not_on_m15_grid")
        if completed < decision:
            issues.append("telemetry_event_lag_negative")
    return tuple(issues)


def _validate_derived_broker_gaps(
    completed_broker_events: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[str, ...]:
    """Derive broker discontinuities independently of decision gap flags."""

    if completed_broker_events.empty or "event_time_utc" not in completed_broker_events:
        return ("broker_gap_validation_unavailable",)
    ledger_times = pd.to_datetime(
        completed_broker_events["event_time_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    ).dropna().drop_duplicates().sort_values()
    if len(ledger_times) < 2:
        return ()
    deltas = ledger_times.diff()
    post_gap_times = set(ledger_times.loc[deltas != _M15_DELTA].iloc[1:])
    if not post_gap_times:
        return ()
    decision_times = pd.to_datetime(
        decisions.get("event_time_utc", pd.Series(dtype="object")),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    actions = decisions.get(
        "action", pd.Series("", index=decisions.index)
    ).fillna("").astype(str).str.upper()
    issues: list[str] = []
    for post_gap_time in sorted(post_gap_times):
        matching_actions = set(actions.loc[decision_times == post_gap_time])
        if not matching_actions.intersection(
            {"CONTROL_GAP_FLATTEN", "CONTROL_GAP_BLOCK"}
        ):
            issues.append(
                "broker_gap_missing_control:"
                f"{pd.Timestamp(post_gap_time).isoformat()}"
            )
    return tuple(issues)


def _timestamp_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(
        frame[column], utc=True, errors="coerce", format="mixed"
    )


def _trade_ledger(deals: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "position_id",
        "open_time_utc",
        "close_time_utc",
        "side_type",
        "entry_volume",
        "exit_volume",
        "weighted_entry_price",
        "weighted_exit_price",
        "gross_profit",
        "commission",
        "swap",
        "fee",
        "net_pnl",
        "deal_count",
        "completed",
    )
    if deals.empty:
        return pd.DataFrame(columns=columns)
    required = ("position_id", "entry", "volume", "price", "profit", "commission", "swap", "fee")
    _require_columns(deals, required, source="broker_deals.csv")
    frame = deals.copy()
    frame["position_id"] = pd.to_numeric(frame["position_id"], errors="coerce")
    frame["entry"] = pd.to_numeric(frame["entry"], errors="coerce")
    for column in ("volume", "price", "profit", "commission", "swap", "fee", "type"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["time_utc"] = pd.to_datetime(
        frame.get("time_utc"), utc=True, errors="coerce", format="mixed"
    )
    frame = frame.dropna(subset=["position_id"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for position_id, group in frame.groupby("position_id", sort=True):
        group = group.sort_values("time_utc")
        entry_rows = group[group["entry"] == 0]
        exit_rows = group[group["entry"].isin([1, 2, 3])]

        def weighted_price(part: pd.DataFrame) -> float | None:
            valid = part.dropna(subset=["price", "volume"])
            total_volume = float(valid["volume"].sum())
            if valid.empty or total_volume <= 0.0:
                return None
            return float((valid["price"] * valid["volume"]).sum() / total_volume)

        first_type = group["type"].dropna()
        gross_profit = float(group["profit"].fillna(0.0).sum())
        commission = float(group["commission"].fillna(0.0).sum())
        swap = float(group["swap"].fillna(0.0).sum())
        fee = float(group["fee"].fillna(0.0).sum())
        rows.append(
            {
                "position_id": int(position_id),
                "open_time_utc": (
                    None
                    if entry_rows["time_utc"].dropna().empty
                    else entry_rows["time_utc"].dropna().min().isoformat()
                ),
                "close_time_utc": (
                    None
                    if exit_rows["time_utc"].dropna().empty
                    else exit_rows["time_utc"].dropna().max().isoformat()
                ),
                "side_type": None if first_type.empty else int(first_type.iloc[0]),
                "entry_volume": float(entry_rows["volume"].fillna(0.0).sum()),
                "exit_volume": float(exit_rows["volume"].fillna(0.0).sum()),
                "weighted_entry_price": weighted_price(entry_rows),
                "weighted_exit_price": weighted_price(exit_rows),
                "gross_profit": gross_profit,
                "commission": commission,
                "swap": swap,
                "fee": fee,
                "net_pnl": gross_profit + commission + swap + fee,
                "deal_count": int(len(group)),
                "completed": bool(not exit_rows.empty),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _execution_ledger(
    order_events: pd.DataFrame,
    deals: pd.DataFrame,
) -> pd.DataFrame:
    """Link successful order legs to authoritative broker fill prices."""

    output_columns = (
        "execution_id",
        "decision_id",
        "role",
        "run_id",
        "completed_utc",
        "purpose",
        "side",
        "requested_volume",
        "requested_price",
        "effective_fill_price",
        "fill_price_source",
        "linked_deal_tickets",
        "order_ticket",
        "deal_ticket",
        "symbol_point",
        "spread_points_before",
        "slippage_points_signed",
        "slippage_points_adverse",
    )
    if order_events.empty:
        return pd.DataFrame(columns=output_columns)

    successful = order_events.loc[
        _boolean(order_events, "order_send_passed").fillna(False)
    ].copy()
    if successful.empty:
        return pd.DataFrame(columns=output_columns)

    deal_frame = deals.copy()
    if not deal_frame.empty:
        for column in ("ticket", "order", "price", "volume"):
            deal_frame[column] = pd.to_numeric(
                deal_frame.get(column), errors="coerce"
            )
        deal_frame = deal_frame[
            (deal_frame["price"] > 0.0) & (deal_frame["volume"] > 0.0)
        ].copy()

    rows: list[dict[str, Any]] = []
    for _, event in successful.iterrows():
        order_ticket_value = pd.to_numeric(
            pd.Series([event.get("order_ticket")]), errors="coerce"
        ).iloc[0]
        deal_ticket_value = pd.to_numeric(
            pd.Series([event.get("deal_ticket")]), errors="coerce"
        ).iloc[0]
        order_ticket = (
            int(order_ticket_value)
            if pd.notna(order_ticket_value) and int(order_ticket_value) != 0
            else None
        )
        deal_ticket = (
            int(deal_ticket_value)
            if pd.notna(deal_ticket_value) and int(deal_ticket_value) != 0
            else None
        )
        linked = pd.DataFrame()
        if not deal_frame.empty and deal_ticket is not None:
            linked = deal_frame[deal_frame["ticket"] == deal_ticket]
        if linked.empty and not deal_frame.empty and order_ticket is not None:
            linked = deal_frame[deal_frame["order"] == order_ticket]

        fill_price = None
        fill_source = None
        linked_tickets: list[int] = []
        if not linked.empty:
            linked_tickets = [
                int(value) for value in linked["ticket"].dropna().astype(int)
            ]
            total_volume = float(linked["volume"].sum())
            if total_volume > 0.0:
                fill_price = float(
                    (linked["price"] * linked["volume"]).sum() / total_volume
                )
                fill_source = "broker_deal_history"
        if fill_price is None:
            broker_result = pd.to_numeric(
                pd.Series([event.get("broker_result_price")]), errors="coerce"
            ).iloc[0]
            if pd.notna(broker_result) and float(broker_result) > 0.0:
                fill_price = float(broker_result)
                fill_source = "order_send_result"

        requested = pd.to_numeric(
            pd.Series([event.get("requested_price")]), errors="coerce"
        ).iloc[0]
        point = pd.to_numeric(
            pd.Series([event.get("symbol_point")]), errors="coerce"
        ).iloc[0]
        side = str(event.get("side", "")).upper()
        signed_slippage = None
        adverse_slippage = None
        if (
            fill_price is not None
            and pd.notna(requested)
            and pd.notna(point)
            and float(point) > 0.0
        ):
            signed_slippage = (fill_price - float(requested)) / float(point)
            adverse_slippage = (
                signed_slippage if side == "BUY" else -signed_slippage
            )

        rows.append(
            {
                "execution_id": event.get("execution_id"),
                "decision_id": event.get("decision_id"),
                "role": event.get("role"),
                "run_id": event.get("run_id"),
                "completed_utc": event.get("completed_utc"),
                "purpose": event.get("purpose"),
                "side": side,
                "requested_volume": event.get("requested_volume"),
                "requested_price": (
                    None if pd.isna(requested) else float(requested)
                ),
                "effective_fill_price": fill_price,
                "fill_price_source": fill_source,
                "linked_deal_tickets": json.dumps(linked_tickets),
                "order_ticket": order_ticket,
                "deal_ticket": deal_ticket,
                "symbol_point": None if pd.isna(point) else float(point),
                "spread_points_before": event.get("spread_points_before"),
                "slippage_points_signed": signed_slippage,
                "slippage_points_adverse": adverse_slippage,
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def _write_daily_partitions(
    *,
    role: str,
    source_name: str,
    frame: pd.DataFrame,
    timestamp_column: str,
    output_root: Path,
) -> None:
    if frame.empty or timestamp_column not in frame.columns:
        return
    work = frame.copy()
    work["_timestamp_utc"] = pd.to_datetime(
        work[timestamp_column], utc=True, errors="coerce", format="mixed"
    )
    work = work.dropna(subset=["_timestamp_utc"])
    if work.empty:
        return
    work["_utc_date"] = work["_timestamp_utc"].dt.strftime("%Y-%m-%d")
    for utc_date, daily in work.groupby("_utc_date", sort=True):
        day_root = output_root / "daily" / str(utc_date) / role
        day_root.mkdir(parents=True, exist_ok=True)
        daily.drop(columns=["_timestamp_utc", "_utc_date"]).to_csv(
            day_root / source_name,
            index=False,
        )


def _daily_summary(
    telemetry: pd.DataFrame,
    decisions: pd.DataFrame,
    order_events: pd.DataFrame,
    deals: pd.DataFrame,
) -> pd.DataFrame:
    telemetry_work = telemetry.copy()
    telemetry_work["timestamp"] = pd.to_datetime(
        telemetry_work["snapshot_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    telemetry_work["utc_date"] = telemetry_work["timestamp"].dt.strftime("%Y-%m-%d")
    telemetry_work["equity_numeric"] = pd.to_numeric(
        telemetry_work["equity"], errors="coerce"
    )
    telemetry_work["broker_position_numeric"] = pd.to_numeric(
        telemetry_work.get("broker_position"), errors="coerce"
    )

    decision_dates = _timestamp_series(decisions, "event_time_utc")
    order_dates = _timestamp_series(order_events, "completed_utc")
    deal_dates = _timestamp_series(deals, "time_utc")
    deal_net = (
        _numeric(deals, "profit").fillna(0.0)
        + _numeric(deals, "commission").fillna(0.0)
        + _numeric(deals, "swap").fillna(0.0)
        + _numeric(deals, "fee").fillna(0.0)
    )

    rows: list[dict[str, Any]] = []
    for utc_date, day in telemetry_work.dropna(subset=["utc_date"]).groupby(
        "utc_date", sort=True
    ):
        day = day.sort_values("timestamp")
        equity = day["equity_numeric"].dropna()
        start_equity = None if equity.empty else float(equity.iloc[0])
        end_equity = None if equity.empty else float(equity.iloc[-1])
        day_return = None
        if start_equity not in (None, 0.0) and end_equity is not None:
            day_return = float(end_equity / start_equity - 1.0)
        date_mask_decisions = decision_dates.dt.strftime("%Y-%m-%d") == utc_date
        date_mask_orders = order_dates.dt.strftime("%Y-%m-%d") == utc_date
        date_mask_deals = deal_dates.dt.strftime("%Y-%m-%d") == utc_date
        positions = day["broker_position_numeric"].dropna()
        rows.append(
            {
                "utc_date": utc_date,
                "telemetry_rows": int(len(day)),
                "decision_rows": int(date_mask_decisions.fillna(False).sum()),
                "order_event_rows": int(date_mask_orders.fillna(False).sum()),
                "broker_deal_rows": int(date_mask_deals.fillna(False).sum()),
                "starting_equity": start_equity,
                "ending_equity": end_equity,
                "equity_return": day_return,
                "maximum_drawdown": _maximum_drawdown(equity),
                "realised_net_pnl": float(deal_net[date_mask_deals.fillna(False)].sum()),
                "exposure_snapshot_rate": (
                    None
                    if positions.empty
                    else float((positions != 0).mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def analyse_role(
    role_root: Path,
    *,
    role: str,
    output_root: Path,
    expected_poll_seconds: int = 30,
) -> RoleObservationSummary:
    telemetry = _read_optional(role_root / "telemetry.csv")
    if telemetry.empty:
        raise LiveAuditAnalysisError(f"Missing telemetry evidence for {role}: {role_root}")
    _require_columns(
        telemetry,
        (
            "snapshot_id", "snapshot_utc", "snapshot_phase", "equity",
            "broker_position", "pending_order_count",
        ),
        source=f"{role}/telemetry.csv",
    )
    decisions = _read_optional(role_root / "decisions.csv")
    completed_broker_events = _read_optional(
        role_root / "completed_broker_events.csv"
    )
    order_events = _read_optional(role_root / "order_events.csv")
    deals = _read_optional(role_root / "broker_deals.csv")
    orders = _read_optional(role_root / "broker_orders.csv")
    runtime_events = _read_optional(role_root / "runtime_events.csv")
    final_report = _read_json_optional(role_root / "final_report.json")

    telemetry = telemetry.copy()
    telemetry["snapshot_utc_parsed"] = pd.to_datetime(
        telemetry["snapshot_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    telemetry = telemetry.sort_values("snapshot_utc_parsed")
    valid_snapshot_times = telemetry["snapshot_utc_parsed"].dropna()
    observation_start_utc = (
        None if valid_snapshot_times.empty else pd.Timestamp(valid_snapshot_times.min())
    )
    observation_end_utc = (
        None if valid_snapshot_times.empty else pd.Timestamp(valid_snapshot_times.max())
    )
    telemetry_run_id_set = {
        str(value).strip()
        for value in telemetry.get(
            "run_id", pd.Series(dtype="object")
        ).dropna()
        if str(value).strip()
    }
    decision_evidence_validation = _validate_decision_evidence(
        decisions,
        order_events,
        runtime_events,
        role=role,
        telemetry_run_ids=telemetry_run_id_set,
        observation_start_utc=observation_start_utc,
        observation_end_utc=observation_end_utc,
    )
    evidence_ownership_issues = _validate_evidence_ownership(
        role=role,
        telemetry_run_ids=telemetry_run_id_set,
        telemetry=telemetry,
        completed_broker_events=completed_broker_events,
        order_events=order_events,
        deals=deals,
        orders=orders,
        runtime_events=runtime_events,
        final_report=final_report,
    )
    telemetry_event_lag_issues = _validate_telemetry_event_lag(telemetry)
    derived_broker_gap_issues = _validate_derived_broker_gaps(
        completed_broker_events, decisions
    )

    completed_broker_event_ledger_present = bool(
        not completed_broker_events.empty
    )
    if completed_broker_event_ledger_present:
        _require_columns(
            completed_broker_events,
            ("broker_event_key", "event_time_utc", "role"),
            source=f"{role}/completed_broker_events.csv",
        )
        ledger_event_times = pd.to_datetime(
            completed_broker_events["event_time_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        ).dropna()
        duplicate_broker_event_key_count = int(
            completed_broker_events["broker_event_key"]
            .fillna("")
            .astype(str)
            .duplicated()
            .sum()
        )
    else:
        ledger_event_times = pd.Series(dtype="datetime64[ns, UTC]")
        duplicate_broker_event_key_count = 0

    decision_event_times = pd.to_datetime(
        decisions.get("event_time_utc", pd.Series(dtype="object")),
        utc=True,
        errors="coerce",
        format="mixed",
    ).dropna()
    unique_completed_events = pd.DatetimeIndex(
        ledger_event_times.drop_duplicates().sort_values()
    )
    unique_decision_events = pd.DatetimeIndex(
        decision_event_times.drop_duplicates().sort_values()
    )
    (
        broker_event_with_multiple_dispositions_count,
        allowed_same_event_safety_override_count,
        unexpected_multiple_disposition_event_count,
        maximum_dispositions_per_broker_event,
    ) = _same_event_disposition_counts(decisions)
    missing_completed_events = unique_completed_events.difference(
        unique_decision_events
    )
    decision_events_without_broker_event = unique_decision_events.difference(
        unique_completed_events
    )
    broker_event_disposition_coverage_ratio = (
        None
        if len(unique_completed_events) == 0
        else float(
            (len(unique_completed_events) - len(missing_completed_events))
            / len(unique_completed_events)
        )
    )
    # Backward-compatible alias retained for existing reports.
    completed_event_coverage_ratio = broker_event_disposition_coverage_ratio

    decision_stale = (
        _boolean(decisions, "stale_event_warning")
        .reindex(decisions.index)
        .fillna(False)
        .astype(bool)
    )
    if "model_prediction_available" in decisions.columns:
        prediction_available = (
            _boolean(decisions, "model_prediction_available")
            .reindex(decisions.index)
            .fillna(False)
            .astype(bool)
        )
    elif "probability_up" in decisions.columns:
        prediction_available = (
            pd.to_numeric(decisions["probability_up"], errors="coerce").notna()
            & ~decision_stale
        )
    else:
        # Missing explicit provenance is invalid for the current audit schema.
        prediction_available = pd.Series(
            False, index=decisions.index, dtype="bool"
        )
    prediction_event_times = pd.to_datetime(
        decisions.loc[prediction_available, "event_time_utc"]
        if "event_time_utc" in decisions.columns
        else pd.Series(dtype="object"),
        utc=True,
        errors="coerce",
        format="mixed",
    ).dropna()
    unique_prediction_events = pd.DatetimeIndex(
        prediction_event_times.drop_duplicates().sort_values()
    )
    model_prediction_count = int(len(unique_prediction_events))
    model_unavailable_event_count = int(
        max(0, len(unique_completed_events) - model_prediction_count)
    )
    model_prediction_coverage_ratio = (
        None
        if len(unique_completed_events) == 0
        else float(model_prediction_count / len(unique_completed_events))
    )
    if model_prediction_coverage_ratio is None:
        model_availability_status = "NOT_OBSERVED"
    elif model_prediction_coverage_ratio >= 0.999999:
        model_availability_status = "FULL"
    elif model_prediction_coverage_ratio > 0.0:
        model_availability_status = "LIMITED"
    else:
        model_availability_status = "UNAVAILABLE"

    prediction_endpoint_mismatch_count = 0
    if "event_time_utc" in decisions.columns:
        disposition_event = pd.to_datetime(
            decisions["event_time_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        prediction_endpoint = pd.to_datetime(
            decisions.get(
                "model_prediction_event_time_utc",
                pd.Series(index=decisions.index, dtype="object"),
            ),
            utc=True,
            errors="coerce",
            format="mixed",
        )
        signal_latest_completed = pd.to_datetime(
            decisions.get(
                "latest_completed_bar_time_utc",
                pd.Series(index=decisions.index, dtype="object"),
            ),
            utc=True,
            errors="coerce",
            format="mixed",
        )
        mismatch_control_rows = (
            decisions.get(
                "action",
                pd.Series("", index=decisions.index, dtype="object"),
            )
            .fillna("")
            .astype(str)
            .str.startswith("CONTROL_MODEL_SNAPSHOT_MISMATCH")
        )
        endpoint_mismatch_must_be_counted = (
            (~decision_stale) | mismatch_control_rows
        )
        raw_endpoint_present = (
            prediction_endpoint.notna()
            | signal_latest_completed.notna()
        )
        unexpected_endpoint_mismatch = (
            endpoint_mismatch_must_be_counted
            & raw_endpoint_present
            & (
                disposition_event.isna()
                | prediction_endpoint.isna()
                | signal_latest_completed.isna()
                | (prediction_endpoint != disposition_event)
                | (signal_latest_completed != disposition_event)
            )
        )
        prediction_endpoint_mismatch_count = int(
            unexpected_endpoint_mismatch.sum()
        )

    action_series = decisions.get(
        "action", pd.Series("", index=decisions.index, dtype="object")
    ).fillna("").astype(str)
    contiguity_warmup_mask = action_series.eq(
        "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP"
    )
    contiguity_warmup_event_count = int(contiguity_warmup_mask.sum())
    historical_backfill_mask = action_series.eq(
        "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL"
    )
    historical_backfill_event_count = int(historical_backfill_mask.sum())
    unavailable_mask = ~prediction_available
    target_positions = _numeric(decisions, "target_position").fillna(0.0)
    broker_after_positions = _numeric(
        decisions, "broker_position_after_inspection"
    ).fillna(target_positions)
    historical_backfill_exposure_observed_count = int(
        (
            historical_backfill_mask
            & ((target_positions != 0.0) | (broker_after_positions != 0.0))
        ).sum()
    )
    model_unavailable_exposure_after_disposition_count = int(
        (
            unavailable_mask
            & ~historical_backfill_mask
            & ((target_positions != 0.0) | (broker_after_positions != 0.0))
        ).sum()
    )

    gap_control_mask = action_series.isin(
        {"CONTROL_GAP_FLATTEN", "CONTROL_GAP_BLOCK"}
    )
    gap_event_time = pd.to_datetime(
        decisions.get("event_time_utc", pd.Series(dtype="object")),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    gap_decision_time = pd.to_datetime(
        decisions.get("decision_utc", pd.Series(dtype="object")),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    gap_control_delay = (
        gap_decision_time
        - gap_event_time
        - pd.Timedelta(minutes=15)
    ).dt.total_seconds()
    gap_control_delay = gap_control_delay[
        gap_control_mask & gap_control_delay.notna()
    ].clip(lower=0.0)
    maximum_gap_control_processing_delay_seconds = (
        None
        if gap_control_delay.empty
        else float(gap_control_delay.max())
    )

    telemetry_latest_decision = pd.to_datetime(
        telemetry.get(
            "latest_decision_event_time_utc", pd.Series(dtype="object")
        ),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    completed_to_decision_lag = (
        (
            pd.to_datetime(
                telemetry.get(
                    "latest_completed_event_time_utc",
                    pd.Series(dtype="object"),
                ),
                utc=True,
                errors="coerce",
                format="mixed",
            )
            - telemetry_latest_decision
        )
        .dt.total_seconds()
        .div(60.0)
        .dropna()
    )
    completed_to_decision_lag = completed_to_decision_lag[
        completed_to_decision_lag >= 0.0
    ]
    maximum_completed_to_decision_lag = (
        None
        if completed_to_decision_lag.empty
        else float(completed_to_decision_lag.max())
    )
    broker_event_times_for_prediction_lag = pd.to_datetime(
        decisions.get("event_time_utc", pd.Series(dtype="object")),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    model_endpoint_times_for_lag = pd.to_datetime(
        decisions.get(
            "model_prediction_event_time_utc", pd.Series(dtype="object")
        ),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    # Current-event audit fields deliberately blank the stale model endpoint
    # while the frozen 48-bar sequence is unavailable.  Forward-fill only for
    # this availability metric so it reports how long the broker event clock
    # has advanced since the most recent actual model prediction.
    latest_available_model_endpoint = model_endpoint_times_for_lag.ffill()
    broker_event_to_model_prediction_lag = (
        broker_event_times_for_prediction_lag
        - latest_available_model_endpoint
    ).dt.total_seconds().div(60.0)
    broker_event_to_model_prediction_lag = (
        broker_event_to_model_prediction_lag[
            broker_event_to_model_prediction_lag.notna()
            & (broker_event_to_model_prediction_lag >= 0.0)
            & ~historical_backfill_mask
        ]
    )
    maximum_broker_event_to_model_prediction_lag = (
        None
        if broker_event_to_model_prediction_lag.empty
        else float(broker_event_to_model_prediction_lag.max())
    )
    timestamps = telemetry["snapshot_utc_parsed"].dropna()
    first_snapshot = None if timestamps.empty else timestamps.min().isoformat()
    last_snapshot = None if timestamps.empty else timestamps.max().isoformat()
    observed_hours = None
    interval_seconds = pd.Series(dtype="float64")
    if len(timestamps) >= 2:
        interval_seconds = timestamps.diff().dt.total_seconds().dropna()
        observed_hours = float(
            (timestamps.max() - timestamps.min()).total_seconds() / 3600.0
        )
    median_interval = (
        None if interval_seconds.empty else float(interval_seconds.median())
    )
    maximum_gap = None if interval_seconds.empty else float(interval_seconds.max())
    gap_threshold = max(90.0, float(expected_poll_seconds) * 3.0)
    gap_count = int((interval_seconds > gap_threshold).sum())
    expected_rows = None
    coverage_ratio = None
    if observed_hours is not None:
        expected_rows = observed_hours * 3600.0 / float(expected_poll_seconds) + 1.0
        coverage_ratio = float(len(timestamps) / expected_rows) if expected_rows > 0 else None

    starting_balance, ending_balance = _first_last(_numeric(telemetry, "balance"))
    starting_equity, ending_equity = _first_last(_numeric(telemetry, "equity"))
    balance_change = None
    if starting_balance is not None and ending_balance is not None:
        balance_change = float(ending_balance - starting_balance)
    net_return = None
    if starting_equity not in (None, 0.0) and ending_equity is not None:
        net_return = float(ending_equity / starting_equity - 1.0)
    initial_broker_position = _last_int(telemetry.head(1)["broker_position"])
    initial_pending_orders = _last_int(telemetry.head(1)["pending_order_count"])

    profit = _numeric(deals, "profit").fillna(0.0)
    commission = _numeric(deals, "commission").fillna(0.0)
    swap = _numeric(deals, "swap").fillna(0.0)
    fee = _numeric(deals, "fee").fillna(0.0)
    deal_net = profit + commission + swap + fee
    trade_ledger = _trade_ledger(deals)
    execution_ledger = _execution_ledger(order_events, deals)
    completed_trades = trade_ledger[trade_ledger.get("completed", False) == True].copy()  # noqa: E712
    trade_net = _numeric(completed_trades, "net_pnl")
    winners = trade_net[trade_net > 0.0]
    losers = trade_net[trade_net < 0.0]
    breakeven = trade_net[trade_net == 0.0]
    win_rate = None if trade_net.empty else float(len(winners) / len(trade_net))
    profit_factor = None
    if not losers.empty:
        profit_factor = (
            float(winners.sum() / abs(losers.sum())) if not winners.empty else 0.0
        )

    successful_orders_mask = _boolean(order_events, "order_send_passed").fillna(False)
    successful_orders = order_events.loc[successful_orders_mask].copy()
    trigger_types = (
        order_events["trigger_type"]
        if "trigger_type" in order_events.columns
        else pd.Series("", index=order_events.index, dtype="object")
    ).fillna("").astype(str).str.upper()
    decision_ids = set(
        decisions.get("decision_id", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
    )
    order_decision_ids = (
        order_events["decision_id"]
        if "decision_id" in order_events.columns
        else pd.Series("", index=order_events.index, dtype="object")
    ).fillna("").astype(str)
    strategy_trigger_mask = trigger_types == "STRATEGY_DECISION"
    valid_control_trigger_types = frozenset(
        {
            "CONTROL_CLEAN_STOP",
            "CONTROL_FLATTEN_ONLY",
            "CONTROL_SESSION_GAP_LOCKOUT",
            *_CONTROL_TRIGGER_BY_ACTION.values(),
            *(
                action
                for action in _KNOWN_DECISION_ACTIONS
                if action.startswith("CONTROL_")
            ),
        }
    )
    control_trigger_mask = trigger_types.isin(valid_control_trigger_types)
    unknown_trigger_mask = ~(strategy_trigger_mask | control_trigger_mask)
    strategy_execution_missing_decision_link_count = int(
        (
            strategy_trigger_mask
            & ~order_decision_ids.isin(decision_ids)
        ).sum()
    )
    invalid_control_execution_link_count = 0
    for row_index in order_events.index[control_trigger_mask]:
        trigger = trigger_types.loc[row_index]
        row_role = _audit_text(order_events.loc[row_index].get("role")).lower()
        row_run_id = _audit_text(order_events.loc[row_index].get("run_id"))
        control_id = order_decision_ids.loc[row_index]
        expected_prefix = f"{trigger}:{row_role}:{row_run_id}:"
        if not control_id.startswith(expected_prefix):
            invalid_control_execution_link_count += 1
            continue
        if trigger in {"CONTROL_CLEAN_STOP", "CONTROL_FLATTEN_ONLY"}:
            tail = control_id[len(expected_prefix) :]
            if not tail.isdigit():
                invalid_control_execution_link_count += 1
    unknown_execution_trigger_count = int(unknown_trigger_mask.sum())
    historical_backfill_decision_ids = set(
        decisions.loc[historical_backfill_mask, "decision_id"]
        .dropna()
        .astype(str)
        if "decision_id" in decisions.columns
        else []
    )
    historical_backfill_order_count = int(
        order_decision_ids.isin(historical_backfill_decision_ids).sum()
    )
    order_event_order_tickets = _ticket_set(successful_orders, "order_ticket")
    order_event_deal_tickets = _ticket_set(successful_orders, "deal_ticket")
    broker_order_tickets = _ticket_set(orders, "ticket")
    broker_deal_tickets = _ticket_set(deals, "ticket")
    broker_deal_order_tickets = _ticket_set(deals, "order")
    missing_order_ticket_count = int(
        (_numeric(successful_orders, "order_ticket").fillna(0.0) == 0.0).sum()
    )
    missing_deal_ticket_count = int(
        (_numeric(successful_orders, "deal_ticket").fillna(0.0) == 0.0).sum()
    )
    missing_broker_orders = order_event_order_tickets - broker_order_tickets
    missing_broker_deals: list[str] = []
    recovered_by_order = 0
    for _, event in successful_orders.iterrows():
        order_ticket = pd.to_numeric(
            pd.Series([event.get("order_ticket")]), errors="coerce"
        ).iloc[0]
        deal_ticket = pd.to_numeric(
            pd.Series([event.get("deal_ticket")]), errors="coerce"
        ).iloc[0]
        direct_link = bool(
            pd.notna(deal_ticket)
            and int(deal_ticket) != 0
            and int(deal_ticket) in broker_deal_tickets
        )
        order_link = bool(
            pd.notna(order_ticket)
            and int(order_ticket) != 0
            and int(order_ticket) in broker_deal_order_tickets
        )
        if not direct_link and order_link:
            recovered_by_order += 1
        if not direct_link and not order_link:
            order_label = (
                int(order_ticket)
                if pd.notna(order_ticket) and int(order_ticket) != 0
                else None
            )
            deal_label = (
                int(deal_ticket)
                if pd.notna(deal_ticket) and int(deal_ticket) != 0
                else None
            )
            missing_broker_deals.append(
                f"order_ticket={order_label},deal_ticket={deal_label}"
            )

    broker_orders_without_execution = broker_order_tickets - order_event_order_tickets
    broker_deals_without_execution = deals.copy()
    if not broker_deals_without_execution.empty:
        deal_ticket_values = _numeric(broker_deals_without_execution, "ticket").fillna(0).astype(int)
        deal_order_values = _numeric(broker_deals_without_execution, "order").fillna(0).astype(int)
        linked_mask = deal_ticket_values.isin(order_event_deal_tickets) | deal_order_values.isin(
            order_event_order_tickets
        )
        broker_deal_missing_execution_count = int((~linked_mask).sum())
    else:
        broker_deal_missing_execution_count = 0


    adverse_slippage = _numeric(
        execution_ledger, "slippage_points_adverse"
    ).dropna()
    order_spread = _numeric(execution_ledger, "spread_points_before").dropna()
    missing_fill_price_count = int(
        _numeric(execution_ledger, "effective_fill_price").isna().sum()
    )
    recovered_fill_count = int(
        (
            execution_ledger.get(
                "fill_price_source", pd.Series(dtype="object")
            ).fillna("").astype(str)
            == "broker_deal_history"
        ).sum()
    )
    maximum_broker_order_time_alignment_seconds = None
    if not successful_orders.empty and not orders.empty:
        event_times = successful_orders.loc[
            :, ["order_ticket", "completed_utc"]
        ].copy()
        broker_times = orders.loc[:, ["ticket", "time_done_utc"]].copy()
        event_times["ticket_key"] = pd.to_numeric(
            event_times["order_ticket"], errors="coerce"
        ).astype("Int64")
        broker_times["ticket_key"] = pd.to_numeric(
            broker_times["ticket"], errors="coerce"
        ).astype("Int64")
        event_times["completed_parsed"] = pd.to_datetime(
            event_times["completed_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        broker_times["done_parsed"] = pd.to_datetime(
            broker_times["time_done_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        aligned = event_times.merge(
            broker_times.loc[:, ["ticket_key", "done_parsed"]],
            on="ticket_key",
            how="inner",
        )
        alignment_seconds = (
            aligned["completed_parsed"] - aligned["done_parsed"]
        ).dt.total_seconds().abs().dropna()
        if not alignment_seconds.empty:
            maximum_broker_order_time_alignment_seconds = float(
                alignment_seconds.max()
            )
    telemetry_spread = _numeric(telemetry, "spread_points").dropna()
    actions = action_series
    blocked = actions.str.startswith("BLOCK_")
    stale_completed_event_count = int(decision_stale.sum())
    gap_decision_count = int(
        actions.isin({"CONTROL_GAP_FLATTEN", "CONTROL_GAP_BLOCK"}).sum()
    )
    event_types = runtime_events.get(
        "event_type", pd.Series(dtype="object")
    ).fillna("").astype(str)
    model_snapshot_mismatch_runtime_event_count = int(
        (event_types == "MODEL_SNAPSHOT_MISMATCH").sum()
    )
    non_startup_telemetry = telemetry[
        telemetry["snapshot_phase"].fillna("").astype(str).str.upper() != "STARTUP"
    ]
    reconciliations = non_startup_telemetry.get(
        "reconciliation_status", pd.Series(dtype="object")
    ).fillna("").astype(str)
    connected = _boolean(telemetry, "terminal_connected").dropna()
    broker_positions = _numeric(telemetry, "broker_position").dropna()
    daily_returns = _daily_equity_returns(telemetry)
    run_ids = telemetry.get("run_id", pd.Series(dtype="object")).dropna().astype(str)
    worker_pids = _numeric(telemetry, "worker_pid").dropna()

    duplicate_snapshot_ids = _duplicate_count(telemetry, "snapshot_id")
    duplicate_decision_ids = _duplicate_count(decisions, "decision_id")
    duplicate_execution_ids = _duplicate_count(order_events, "execution_id")
    duplicate_deal_keys = _duplicate_count(deals, "history_key")
    duplicate_order_keys = _duplicate_count(orders, "history_key")
    final_broker_position = _last_int(telemetry["broker_position"])
    final_pending_orders = _last_int(telemetry["pending_order_count"])
    final_status = str(final_report.get("status")) if final_report else None
    final_formal_gate = (
        bool(final_report.get("formal_gate")) if final_report else None
    )
    final_state = final_report.get("state", {}) if final_report else {}
    if not isinstance(final_state, dict):
        final_state = {}
    expected_decision_rows = (
        int(final_state["records_written"])
        if final_state.get("records_written") is not None
        else None
    )
    expected_order_send_calls = (
        int(final_state["order_send_calls"])
        if final_state.get("order_send_calls") is not None
        else None
    )
    expected_successful_sends = (
        int(final_state["successful_order_sends"])
        if final_state.get("successful_order_sends") is not None
        else None
    )
    decision_count_mismatch = (
        None
        if expected_decision_rows is None
        else int(len(decisions) - expected_decision_rows)
    )
    order_event_count_mismatch = (
        None
        if expected_order_send_calls is None
        else int(len(order_events) - expected_order_send_calls)
    )
    successful_event_count_mismatch = (
        None
        if expected_successful_sends is None
        else int(len(successful_orders) - expected_successful_sends)
    )
    realised_net_pnl = float(deal_net.sum())
    balance_pnl_difference = (
        None
        if balance_change is None
        else float(balance_change - realised_net_pnl)
    )
    final_equity_balance_difference = (
        None
        if ending_balance is None or ending_equity is None
        else float(ending_equity - ending_balance)
    )

    audit_failures: list[str] = []
    audit_failures.extend(evidence_ownership_issues)
    audit_failures.extend(telemetry_event_lag_issues)
    audit_failures.extend(derived_broker_gap_issues)
    for label, count in (
        ("duplicate_snapshot_ids", duplicate_snapshot_ids),
        (
            "decision_evidence_validation_issues",
            decision_evidence_validation.issue_count,
        ),
        ("duplicate_decision_ids", duplicate_decision_ids),
        ("duplicate_execution_ids", duplicate_execution_ids),
        ("duplicate_broker_deal_keys", duplicate_deal_keys),
        ("duplicate_broker_order_keys", duplicate_order_keys),
        (
            "duplicate_completed_broker_event_keys",
            duplicate_broker_event_key_count,
        ),
        (
            "decisions_without_broker_event_ledger_entry",
            len(decision_events_without_broker_event),
        ),
        ("successful_orders_missing_order_ticket", missing_order_ticket_count),
        ("missing_broker_order_links", len(missing_broker_orders)),
        ("missing_broker_deal_links", len(missing_broker_deals)),
        ("successful_orders_missing_fill_price", missing_fill_price_count),
        ("broker_orders_without_execution", len(broker_orders_without_execution)),
        ("broker_deals_without_execution", broker_deal_missing_execution_count),
        (
            "missing_completed_event_dispositions",
            len(missing_completed_events),
        ),
        (
            "model_prediction_endpoint_mismatches",
            prediction_endpoint_mismatch_count,
        ),
        (
            "unexpected_multiple_disposition_events",
            unexpected_multiple_disposition_event_count,
        ),
        (
            "model_unavailable_exposure_after_disposition",
            model_unavailable_exposure_after_disposition_count,
        ),
        (
            "historical_backfill_orders",
            historical_backfill_order_count,
        ),
        (
            "historical_backfill_exposure_observed",
            historical_backfill_exposure_observed_count,
        ),
        (
            "strategy_executions_missing_decision_link",
            strategy_execution_missing_decision_link_count,
        ),
        (
            "invalid_control_execution_links",
            invalid_control_execution_link_count,
        ),
        ("unknown_execution_triggers", unknown_execution_trigger_count),
    ):
        if count:
            audit_failures.append(f"{label}={count}")
    if not completed_broker_event_ledger_present:
        audit_failures.append("completed_broker_event_ledger_missing=1")
    if initial_broker_position != 0:
        audit_failures.append(f"initial_broker_position={initial_broker_position}")
    if initial_pending_orders != 0:
        audit_failures.append(
            f"initial_pending_order_count={initial_pending_orders}"
        )
    if coverage_ratio is not None and coverage_ratio < 0.99:
        audit_failures.append(
            f"telemetry_coverage_ratio={coverage_ratio:.6f}<0.99"
        )
    if (
        completed_event_coverage_ratio is not None
        and completed_event_coverage_ratio < 0.999999
    ):
        audit_failures.append(
            "broker_event_disposition_coverage_ratio="
            f"{completed_event_coverage_ratio:.6f}<1.0"
        )
    if (
        maximum_gap_control_processing_delay_seconds is not None
        and maximum_gap_control_processing_delay_seconds > 90.0
    ):
        audit_failures.append(
            "maximum_gap_control_processing_delay_seconds="
            f"{maximum_gap_control_processing_delay_seconds:.6f}>90.0"
        )
    if (
        maximum_completed_to_decision_lag is not None
        and maximum_completed_to_decision_lag > 15.1
    ):
        audit_failures.append(
            "maximum_completed_to_decision_lag_minutes="
            f"{maximum_completed_to_decision_lag:.6f}>15.1"
        )
    if (
        maximum_broker_order_time_alignment_seconds is not None
        and maximum_broker_order_time_alignment_seconds > 60.0
    ):
        audit_failures.append(
            "maximum_broker_order_time_alignment_seconds="
            f"{maximum_broker_order_time_alignment_seconds:.6f}>60.0"
        )
    for label, mismatch in (
        ("decision_record_count_mismatch", decision_count_mismatch),
        ("order_event_count_mismatch", order_event_count_mismatch),
        ("successful_order_event_count_mismatch", successful_event_count_mismatch),
    ):
        if mismatch not in (None, 0):
            audit_failures.append(f"{label}={mismatch}")
    if (
        balance_pnl_difference is not None
        and abs(balance_pnl_difference) > 0.05
    ):
        audit_failures.append(
            "balance_pnl_reconciliation_difference="
            f"{balance_pnl_difference:.6f}"
        )
    if (
        final_equity_balance_difference is not None
        and abs(final_equity_balance_difference) > 0.05
    ):
        audit_failures.append(
            "final_equity_balance_difference="
            f"{final_equity_balance_difference:.6f}"
        )
    if final_broker_position != 0:
        audit_failures.append(f"final_broker_position={final_broker_position}")
    if final_pending_orders != 0:
        audit_failures.append(f"final_pending_order_count={final_pending_orders}")
    worker_error_count = int((event_types == "WORKER_ERROR").sum())
    reconciliation_incident_count = int(
        (event_types == "RECONCILIATION_INCIDENT").sum()
    )
    reconciliation_nonpass_snapshot_count = int(
        (
            ~reconciliations.str.startswith("PASS")
            & ~reconciliations.str.contains("FLAT_CONFIRMED", regex=False)
        ).sum()
    )
    inferred_restart_count = max(
        0, int(max(run_ids.nunique(), worker_pids.nunique())) - 1
    )
    limited_recovery_reasons: list[str] = []
    if worker_error_count:
        limited_recovery_reasons.append(
            f"worker_error_count={worker_error_count}"
        )
    if inferred_restart_count:
        limited_recovery_reasons.append(
            f"inferred_worker_restart_count={inferred_restart_count}"
        )
    if model_snapshot_mismatch_runtime_event_count:
        limited_recovery_reasons.append(
            "model_snapshot_mismatch_runtime_event_count="
            f"{model_snapshot_mismatch_runtime_event_count}"
        )
    if reconciliation_incident_count:
        limited_recovery_reasons.append(
            f"reconciliation_incident_count={reconciliation_incident_count}"
        )
    if reconciliation_nonpass_snapshot_count:
        limited_recovery_reasons.append(
            "reconciliation_nonpass_snapshot_count="
            f"{reconciliation_nonpass_snapshot_count}"
        )
    if historical_backfill_event_count:
        limited_recovery_reasons.append(
            f"historical_backfill_event_count={historical_backfill_event_count}"
        )
    if gap_count:
        limited_recovery_reasons.append(
            f"material_telemetry_gap_count={gap_count}"
        )
    if coverage_ratio is not None and coverage_ratio < 0.99:
        # Coverage shortfall is recoverable only when the independent broker
        # event ledger and all dispositions remain complete.
        coverage_failure = (
            f"telemetry_coverage_ratio={coverage_ratio:.6f}<0.99"
        )
        if coverage_failure in audit_failures:
            audit_failures.remove(coverage_failure)
        limited_recovery_reasons.append(coverage_failure)
    if final_status != "PASS" or final_formal_gate is not True:
        audit_failures.append(
            f"final_worker_gate=status:{final_status},formal_gate:{final_formal_gate}"
        )

    if audit_failures:
        operational_acceptance_status = "FAIL"
    elif limited_recovery_reasons:
        operational_acceptance_status = "LIMITED_RECOVERED"
    else:
        operational_acceptance_status = "PASS"
    formal_audit_gate = bool(
        operational_acceptance_status == "PASS"
    )
    all_gate_reasons = tuple(
        [
            *audit_failures,
            *decision_evidence_validation.reason_counts,
            *limited_recovery_reasons,
        ]
    )

    summary = RoleObservationSummary(
        role=role,
        formal_audit_gate=formal_audit_gate,
        operational_acceptance_status=operational_acceptance_status,
        audit_gate_failures=all_gate_reasons,
        limited_recovery_reasons=tuple(limited_recovery_reasons),
        telemetry_rows=int(len(telemetry)),
        decision_rows=int(len(decisions)),
        invalid_decision_evidence_count=(
            decision_evidence_validation.invalid_row_count
        ),
        decision_evidence_issue_count=(
            decision_evidence_validation.issue_count
        ),
        decision_evidence_issue_reasons=(
            decision_evidence_validation.reason_counts
        ),
        unique_completed_event_count=int(len(unique_completed_events)),
        completed_broker_event_ledger_rows=int(
            len(completed_broker_events)
        ),
        completed_broker_event_ledger_present=(
            completed_broker_event_ledger_present
        ),
        duplicate_broker_event_key_count=(
            duplicate_broker_event_key_count
        ),
        decision_without_broker_event_count=int(
            len(decision_events_without_broker_event)
        ),
        broker_event_disposition_coverage_ratio=(
            broker_event_disposition_coverage_ratio
        ),
        completed_event_coverage_ratio=completed_event_coverage_ratio,
        missing_completed_event_decision_count=int(len(missing_completed_events)),
        model_prediction_count=model_prediction_count,
        model_unavailable_event_count=model_unavailable_event_count,
        model_prediction_coverage_ratio=model_prediction_coverage_ratio,
        model_availability_status=model_availability_status,
        model_prediction_endpoint_mismatch_count=(
            prediction_endpoint_mismatch_count
        ),
        model_snapshot_mismatch_runtime_event_count=(
            model_snapshot_mismatch_runtime_event_count
        ),
        broker_event_with_multiple_dispositions_count=(
            broker_event_with_multiple_dispositions_count
        ),
        allowed_same_event_safety_override_count=(
            allowed_same_event_safety_override_count
        ),
        unexpected_multiple_disposition_event_count=(
            unexpected_multiple_disposition_event_count
        ),
        maximum_dispositions_per_broker_event=(
            maximum_dispositions_per_broker_event
        ),
        contiguity_warmup_event_count=contiguity_warmup_event_count,
        historical_backfill_event_count=historical_backfill_event_count,
        historical_backfill_exposure_observed_count=(
            historical_backfill_exposure_observed_count
        ),
        historical_backfill_order_count=historical_backfill_order_count,
        model_unavailable_exposure_after_disposition_count=(
            model_unavailable_exposure_after_disposition_count
        ),
        maximum_gap_control_processing_delay_seconds=(
            maximum_gap_control_processing_delay_seconds
        ),
        maximum_completed_to_decision_lag_minutes=(
            maximum_completed_to_decision_lag
        ),
        maximum_current_broker_event_to_model_prediction_lag_minutes=(
            maximum_broker_event_to_model_prediction_lag
        ),
        maximum_broker_event_to_model_prediction_lag_minutes=(
            maximum_broker_event_to_model_prediction_lag
        ),
        stale_completed_event_count=stale_completed_event_count,
        gap_decision_count=gap_decision_count,
        order_event_rows=int(len(order_events)),
        control_execution_count=int(control_trigger_mask.sum()),
        strategy_execution_missing_decision_link_count=(
            strategy_execution_missing_decision_link_count
        ),
        invalid_control_execution_link_count=(
            invalid_control_execution_link_count
        ),
        unknown_execution_trigger_count=unknown_execution_trigger_count,
        maximum_broker_order_time_alignment_seconds=(
            maximum_broker_order_time_alignment_seconds
        ),
        broker_deal_rows=int(len(deals)),
        broker_order_rows=int(len(orders)),
        runtime_event_rows=int(len(runtime_events)),
        first_snapshot_utc=first_snapshot,
        last_snapshot_utc=last_snapshot,
        observed_hours=observed_hours,
        expected_poll_seconds=int(expected_poll_seconds),
        median_telemetry_interval_seconds=median_interval,
        maximum_telemetry_gap_seconds=maximum_gap,
        telemetry_gap_count_over_threshold=gap_count,
        telemetry_coverage_ratio=coverage_ratio,
        terminal_connected_snapshot_rate=(
            None if connected.empty else float(connected.mean())
        ),
        broker_exposure_snapshot_rate=(
            None if broker_positions.empty else float((broker_positions != 0).mean())
        ),
        worker_run_count=int(run_ids.nunique()),
        worker_pid_count=int(worker_pids.nunique()),
        inferred_worker_restart_count=inferred_restart_count,
        initial_broker_position=initial_broker_position,
        initial_pending_order_count=initial_pending_orders,
        starting_balance=starting_balance,
        ending_balance=ending_balance,
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        balance_change=balance_change,
        net_equity_return=net_return,
        maximum_equity_drawdown=_maximum_drawdown(_numeric(telemetry, "equity")),
        balance_pnl_reconciliation_difference=balance_pnl_difference,
        final_equity_balance_difference=final_equity_balance_difference,
        realised_profit=float(profit.sum()),
        commission=float(commission.sum()),
        swap=float(swap.sum()),
        fee=float(fee.sum()),
        realised_net_pnl=realised_net_pnl,
        completed_trade_count=int(len(trade_net)),
        winning_trade_count=int(len(winners)),
        losing_trade_count=int(len(losers)),
        breakeven_trade_count=int(len(breakeven)),
        win_rate=win_rate,
        average_winning_trade=(None if winners.empty else float(winners.mean())),
        average_losing_trade=(None if losers.empty else float(losers.mean())),
        profit_factor=profit_factor,
        total_order_volume_lots=float(
            _numeric(successful_orders, "requested_volume").fillna(0.0).sum()
        ),
        average_adverse_slippage_points=(
            None if adverse_slippage.empty else float(adverse_slippage.mean())
        ),
        maximum_adverse_slippage_points=(
            None if adverse_slippage.empty else float(adverse_slippage.max())
        ),
        average_order_spread_points=(
            None if order_spread.empty else float(order_spread.mean())
        ),
        maximum_order_spread_points=(
            None if order_spread.empty else float(order_spread.max())
        ),
        average_telemetry_spread_points=(
            None if telemetry_spread.empty else float(telemetry_spread.mean())
        ),
        maximum_telemetry_spread_points=(
            None if telemetry_spread.empty else float(telemetry_spread.max())
        ),
        distinct_decision_actions=int(actions[actions != ""].nunique()),
        blocked_decision_count=int(blocked.sum()),
        policy_cap_block_count=int((actions == "BLOCK_DAILY_POLICY_CAP").sum()),
        capped_exit_allowed_count=int((actions == "EXIT_POSITION_CAP_REACHED").sum()),
        close_only_reversal_count=int((actions == "CLOSE_ONLY_DAILY_POLICY_CAP").sum()),
        reconciliation_incident_count=reconciliation_incident_count,
        reconciliation_nonpass_snapshot_count=(
            reconciliation_nonpass_snapshot_count
        ),
        daily_stop_trigger_count=int((event_types == "DAILY_STOP_TRIGGERED").sum()),
        total_stop_trigger_count=int((event_types == "TOTAL_STOP_TRIGGERED").sum()),
        worker_error_count=worker_error_count,
        successful_order_event_count=int(len(successful_orders)),
        successful_order_event_missing_order_ticket_count=missing_order_ticket_count,
        successful_order_event_missing_deal_ticket_count=missing_deal_ticket_count,
        successful_order_missing_fill_price_count=missing_fill_price_count,
        broker_fill_price_recovered_count=recovered_fill_count,
        missing_broker_order_link_count=int(len(missing_broker_orders)),
        missing_broker_deal_link_count=int(len(missing_broker_deals)),
        broker_order_missing_execution_link_count=int(
            len(broker_orders_without_execution)
        ),
        broker_deal_missing_execution_link_count=broker_deal_missing_execution_count,
        recovered_deal_link_by_order_count=int(recovered_by_order),
        duplicate_snapshot_id_count=duplicate_snapshot_ids,
        duplicate_decision_id_count=duplicate_decision_ids,
        duplicate_execution_id_count=duplicate_execution_ids,
        duplicate_broker_deal_key_count=duplicate_deal_keys,
        duplicate_broker_order_key_count=duplicate_order_keys,
        expected_decision_rows_from_state=expected_decision_rows,
        decision_record_count_mismatch=decision_count_mismatch,
        expected_order_send_calls_from_state=expected_order_send_calls,
        order_event_count_mismatch=order_event_count_mismatch,
        expected_successful_order_sends_from_state=expected_successful_sends,
        successful_order_event_count_mismatch=successful_event_count_mismatch,
        final_broker_position=final_broker_position,
        final_pending_order_count=final_pending_orders,
        final_worker_status=final_status,
        final_worker_formal_gate=final_formal_gate,
        daily_return_observations=int(len(daily_returns)),
        descriptive_daily_sharpe_annualised_252=_descriptive_sharpe(daily_returns),
        sharpe_limitation=(
            "Descriptive only. A seven-day pilot provides too few daily returns "
            "for a stable or generalisable Sharpe estimate."
        ),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    daily_returns.rename("daily_equity_return").to_csv(
        output_root / f"{role}_daily_equity_returns.csv",
        index_label="utc_date",
    )
    actions.value_counts(dropna=False).rename_axis("action").reset_index(
        name="count"
    ).to_csv(output_root / f"{role}_decision_action_counts.csv", index=False)
    trade_ledger.to_csv(output_root / f"{role}_trade_ledger.csv", index=False)
    execution_ledger.to_csv(
        output_root / f"{role}_execution_ledger.csv", index=False
    )
    _daily_summary(telemetry, decisions, order_events, deals).to_csv(
        output_root / f"{role}_daily_summary.csv", index=False
    )
    (output_root / f"{role}_audit_gate.json").write_text(
        json.dumps(
            {
                "role": role,
                "formal_audit_gate": summary.formal_audit_gate,
                "operational_acceptance_status": (
                    summary.operational_acceptance_status
                ),
                "failures": list(summary.audit_gate_failures),
                "limited_recovery_reasons": list(
                    summary.limited_recovery_reasons
                ),
                "missing_broker_order_tickets": sorted(missing_broker_orders),
                "missing_broker_deal_tickets": sorted(missing_broker_deals),
                "broker_order_tickets_without_execution": sorted(
                    broker_orders_without_execution
                ),
                "broker_deal_rows_without_execution": (
                    broker_deal_missing_execution_count
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for name, frame, timestamp_column in (
        ("telemetry.csv", telemetry.drop(columns=["snapshot_utc_parsed"]), "snapshot_utc"),
        (
            "completed_broker_events.csv",
            completed_broker_events,
            "event_time_utc",
        ),
        ("decisions.csv", decisions, "event_time_utc"),
        ("order_events.csv", order_events, "completed_utc"),
        ("broker_deals.csv", deals, "time_utc"),
        ("broker_orders.csv", orders, "time_done_utc"),
        ("runtime_events.csv", runtime_events, "timestamp_utc"),
    ):
        _write_daily_partitions(
            role=role,
            source_name=name,
            frame=frame,
            timestamp_column=timestamp_column,
            output_root=output_root,
        )
    return summary


def build_observation_report(
    runtime_root: Path,
    output_root: Path,
    *,
    expected_poll_seconds: int = 30,
) -> dict[str, Any]:
    if expected_poll_seconds < 1:
        raise LiveAuditAnalysisError("expected_poll_seconds must be positive")
    summaries = [
        analyse_role(
            runtime_root / role,
            role=role,
            output_root=output_root,
            expected_poll_seconds=expected_poll_seconds,
        )
        for role in ("model_a", "model_b")
    ]
    summary_rows = [asdict(item) for item in summaries]
    pd.DataFrame(summary_rows).to_csv(
        output_root / "consolidated_model_summary.csv", index=False
    )
    formal_gate = all(item.formal_audit_gate for item in summaries)
    statuses = {item.operational_acceptance_status for item in summaries}
    if "FAIL" in statuses:
        operational_acceptance_status = "FAIL"
    elif "LIMITED_RECOVERED" in statuses:
        operational_acceptance_status = "LIMITED_RECOVERED"
    else:
        operational_acceptance_status = "PASS"
    report = {
        "schema_version": "1.0",
        "runtime_root": str(runtime_root),
        "output_root": str(output_root),
        "expected_poll_seconds": int(expected_poll_seconds),
        "formal_audit_gate": formal_gate,
        "operational_acceptance_status": operational_acceptance_status,
        "models": summary_rows,
        "interpretation": {
            "operational_scope": "MT5 demo operational pilot",
            "economic_scope": "descriptive only",
            "long_run_robustness_claim_allowed": False,
            "sharpe_is_descriptive_only": True,
            "raw_runtime_files_remain_authoritative": True,
        },
    }
    (output_root / "consolidated_observation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return report
