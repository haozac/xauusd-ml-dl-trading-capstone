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


class LiveAuditAnalysisError(RuntimeError):
    """Raised when raw observation evidence is missing or malformed."""


@dataclass(frozen=True)
class RoleObservationSummary:
    role: str
    formal_audit_gate: bool
    audit_gate_failures: tuple[str, ...]
    telemetry_rows: int
    decision_rows: int
    unique_completed_event_count: int
    broker_event_disposition_coverage_ratio: float | None
    completed_event_coverage_ratio: float | None
    missing_completed_event_decision_count: int
    model_prediction_count: int
    model_unavailable_event_count: int
    model_prediction_coverage_ratio: float | None
    model_availability_status: str
    model_prediction_endpoint_mismatch_count: int
    contiguity_warmup_event_count: int
    model_unavailable_exposure_after_disposition_count: int
    maximum_gap_control_processing_delay_seconds: float | None
    maximum_completed_to_decision_lag_minutes: float | None
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
        frame["snapshot_utc"], utc=True, errors="coerce"
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


def _timestamp_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame[column], utc=True, errors="coerce")


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
        frame.get("time_utc"), utc=True, errors="coerce"
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
        work[timestamp_column], utc=True, errors="coerce"
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
        telemetry_work["snapshot_utc"], utc=True, errors="coerce"
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
    order_events = _read_optional(role_root / "order_events.csv")
    deals = _read_optional(role_root / "broker_deals.csv")
    orders = _read_optional(role_root / "broker_orders.csv")
    runtime_events = _read_optional(role_root / "runtime_events.csv")
    final_report = _read_json_optional(role_root / "final_report.json")

    telemetry = telemetry.copy()
    telemetry["snapshot_utc_parsed"] = pd.to_datetime(
        telemetry["snapshot_utc"], utc=True, errors="coerce"
    )
    telemetry = telemetry.sort_values("snapshot_utc_parsed")

    completed_event_times = pd.to_datetime(
        telemetry.get(
            "latest_completed_event_time_utc", pd.Series(dtype="object")
        ),
        utc=True,
        errors="coerce",
        format="mixed",
    ).dropna()
    unique_completed_events = pd.DatetimeIndex(
        completed_event_times.drop_duplicates().sort_values()
    )
    decision_event_times = pd.to_datetime(
        decisions.get("event_time_utc", pd.Series(dtype="object")),
        utc=True,
        errors="coerce",
        format="mixed",
    ).dropna()
    unique_decision_events = pd.DatetimeIndex(
        decision_event_times.drop_duplicates().sort_values()
    )
    missing_completed_events = unique_completed_events.difference(
        unique_decision_events
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

    decision_stale = _boolean(decisions, "stale_event_warning").fillna(False)
    if "model_prediction_available" in decisions.columns:
        prediction_available = _boolean(
            decisions, "model_prediction_available"
        ).fillna(False)
    elif "probability_up" in decisions.columns:
        prediction_available = (
            pd.to_numeric(decisions["probability_up"], errors="coerce").notna()
            & ~decision_stale
        )
    else:
        # Legacy fixtures/reports predate the explicit availability field.
        prediction_available = pd.Series(
            True, index=decisions.index, dtype="bool"
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
    if {
        "model_prediction_event_time_utc",
        "event_time_utc",
    }.issubset(decisions.columns):
        prediction_endpoint = pd.to_datetime(
            decisions["model_prediction_event_time_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        disposition_event = pd.to_datetime(
            decisions["event_time_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        prediction_endpoint_mismatch_count = int(
            (
                prediction_available
                & (
                    prediction_endpoint.isna()
                    | disposition_event.isna()
                    | (prediction_endpoint != disposition_event)
                )
            ).sum()
        )

    action_series = decisions.get(
        "action", pd.Series("", index=decisions.index, dtype="object")
    ).fillna("").astype(str)
    contiguity_warmup_mask = action_series.eq(
        "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP"
    )
    contiguity_warmup_event_count = int(contiguity_warmup_mask.sum())
    unavailable_mask = ~prediction_available
    target_positions = _numeric(decisions, "target_position").fillna(0.0)
    broker_after_positions = _numeric(
        decisions, "broker_position_after_inspection"
    ).fillna(target_positions)
    model_unavailable_exposure_after_disposition_count = int(
        (
            unavailable_mask
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
    control_trigger_mask = trigger_types.str.startswith("CONTROL_")
    unknown_trigger_mask = ~(strategy_trigger_mask | control_trigger_mask)
    strategy_execution_missing_decision_link_count = int(
        (
            strategy_trigger_mask
            & ~order_decision_ids.isin(decision_ids)
        ).sum()
    )
    invalid_control_execution_link_count = int(
        (
            control_trigger_mask
            & ~order_decision_ids.str.startswith("CONTROL_")
        ).sum()
    )
    unknown_execution_trigger_count = int(unknown_trigger_mask.sum())
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
            event_times["completed_utc"], utc=True, errors="coerce"
        )
        broker_times["done_parsed"] = pd.to_datetime(
            broker_times["time_done_utc"], utc=True, errors="coerce"
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
    for label, count in (
        ("duplicate_snapshot_ids", duplicate_snapshot_ids),
        ("duplicate_decision_ids", duplicate_decision_ids),
        ("duplicate_execution_ids", duplicate_execution_ids),
        ("duplicate_broker_deal_keys", duplicate_deal_keys),
        ("duplicate_broker_order_keys", duplicate_order_keys),
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
            "model_unavailable_exposure_after_disposition",
            model_unavailable_exposure_after_disposition_count,
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
    if initial_broker_position != 0:
        audit_failures.append(f"initial_broker_position={initial_broker_position}")
    if initial_pending_orders != 0:
        audit_failures.append(
            f"initial_pending_order_count={initial_pending_orders}"
        )
    if coverage_ratio is not None and coverage_ratio < 0.90:
        audit_failures.append(
            f"telemetry_coverage_ratio={coverage_ratio:.6f}<0.90"
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
    if worker_error_count:
        audit_failures.append(f"worker_error_count={worker_error_count}")
    if final_status != "PASS" or final_formal_gate is not True:
        audit_failures.append(
            f"final_worker_gate=status:{final_status},formal_gate:{final_formal_gate}"
        )

    summary = RoleObservationSummary(
        role=role,
        formal_audit_gate=not audit_failures,
        audit_gate_failures=tuple(audit_failures),
        telemetry_rows=int(len(telemetry)),
        decision_rows=int(len(decisions)),
        unique_completed_event_count=int(len(unique_completed_events)),
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
        contiguity_warmup_event_count=contiguity_warmup_event_count,
        model_unavailable_exposure_after_disposition_count=(
            model_unavailable_exposure_after_disposition_count
        ),
        maximum_gap_control_processing_delay_seconds=(
            maximum_gap_control_processing_delay_seconds
        ),
        maximum_completed_to_decision_lag_minutes=(
            maximum_completed_to_decision_lag
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
        inferred_worker_restart_count=max(
            0, int(max(run_ids.nunique(), worker_pids.nunique())) - 1
        ),
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
        reconciliation_incident_count=int(
            (event_types == "RECONCILIATION_INCIDENT").sum()
        ),
        reconciliation_nonpass_snapshot_count=int(
            (~reconciliations.str.startswith("PASS")
             & ~reconciliations.str.contains("FLAT_CONFIRMED", regex=False)).sum()
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
                "failures": list(summary.audit_gate_failures),
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
    report = {
        "schema_version": "1.0",
        "runtime_root": str(runtime_root),
        "output_root": str(output_root),
        "expected_poll_seconds": int(expected_poll_seconds),
        "formal_audit_gate": formal_gate,
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
