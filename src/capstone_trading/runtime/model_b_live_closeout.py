"""Offline closeout audit for Stage 3 Step 3B Model B controlled live execution.

The controlled live process writes its event log and state incrementally.  When a
monitored run is stopped with Ctrl+C while flat, the normal final JSON report is
not produced.  This module audits the incremental artefacts for one explicit
run_id and produces a formal, reproducible closeout decision without connecting
to MT5 or calling any trading function.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import csv
import json
import math

DEFAULT_EVENTS_CSV_PATH = Path("runtime/execution_live/stage3_step3b_model_b_events.csv")
DEFAULT_STATE_PATH = Path("runtime/state/stage3_step3b_model_b_live_state.json")
DEFAULT_LATEST_DECISION_PATH = Path("runtime/reports/stage3_step3b_latest_decision.csv")
DEFAULT_LIVE_REPORT_PATH = Path("runtime/reports/stage3_step3b_model_b_controlled_live.json")
DEFAULT_AUDIT_REPORT_PATH = Path("runtime/reports/stage3_step3b_closeout_audit.json")
DEFAULT_SUMMARY_CSV_PATH = Path("runtime/reports/stage3_step3b_closeout_summary.csv")
DEFAULT_RUN_EVENTS_CSV_PATH = Path("runtime/reports/stage3_step3b_closeout_run_events.csv")
DEFAULT_FRESH_EVENTS_CSV_PATH = Path("runtime/reports/stage3_step3b_closeout_fresh_events.csv")

EXPECTED_ENTRY_THRESHOLD = 0.55
EXPECTED_SPREAD_GATE_POINTS = 800
DEFAULT_MIN_FRESH_EVENTS = 8


class Stage3Step3BCloseoutError(RuntimeError):
    """Raised when incremental evidence cannot support a formal closeout PASS."""


@dataclass(frozen=True)
class CloseoutInputs:
    events_csv: Path
    state_json: Path
    latest_decision_csv: Path
    live_report_json: Path | None = None


@dataclass(frozen=True)
class CloseoutOutputs:
    audit_report_json: Path
    summary_csv: Path
    run_events_csv: Path
    fresh_events_csv: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise Stage3Step3BCloseoutError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise Stage3Step3BCloseoutError(f"CSV has no header: {path}")
    return fieldnames, rows


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Stage3Step3BCloseoutError(f"Required JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Stage3Step3BCloseoutError(f"Unable to parse JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise Stage3Step3BCloseoutError(f"JSON must contain an object: {path}")
    return dict(payload)


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = "" if value is None else str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise Stage3Step3BCloseoutError(f"Invalid boolean for {field}: {value!r}")


def _int(value: Any, *, field: str, allow_blank: bool = False) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text and allow_blank:
        return None
    try:
        return int(float(text))
    except Exception as exc:
        raise Stage3Step3BCloseoutError(f"Invalid integer for {field}: {value!r}") from exc


def _float(value: Any, *, field: str, allow_blank: bool = False) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text and allow_blank:
        return None
    try:
        result = float(text)
    except Exception as exc:
        raise Stage3Step3BCloseoutError(f"Invalid float for {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise Stage3Step3BCloseoutError(f"Non-finite float for {field}: {value!r}")
    return result


def _utc_datetime(value: Any, *, field: str) -> datetime:
    text = "" if value is None else str(value).strip()
    if not text:
        raise Stage3Step3BCloseoutError(f"Missing timestamp for {field}")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception as exc:
        raise Stage3Step3BCloseoutError(f"Invalid timestamp for {field}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage3Step3BCloseoutError(f"Timestamp is not timezone-aware for {field}: {value!r}")
    return parsed.astimezone(timezone.utc)


def _required_columns() -> set[str]:
    return {
        "run_id",
        "iteration",
        "mode",
        "event_time_utc",
        "probability_up",
        "action",
        "reason",
        "live_position_before",
        "live_position_after",
        "duplicate_event",
        "stale_event_warning",
        "spread_points",
        "spread_gate_points",
        "actual_position_count",
        "pending_order_count",
        "order_check_called",
        "order_send_called",
        "decision_utc",
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    tmp.replace(path)


def _all_false(rows: Iterable[Mapping[str, Any]], field: str) -> bool:
    return all(not _bool(row.get(field), field=field) for row in rows)


def audit_model_b_live_closeout(
    *,
    run_id: str,
    inputs: CloseoutInputs,
    outputs: CloseoutOutputs,
    min_fresh_events: int = DEFAULT_MIN_FRESH_EVENTS,
    entry_threshold: float = EXPECTED_ENTRY_THRESHOLD,
    spread_gate_points: int = EXPECTED_SPREAD_GATE_POINTS,
    require_wide_spread_observation: bool = True,
    termination_reason: str = "keyboard_interrupt_after_manual_flat_check",
) -> dict[str, Any]:
    """Audit one interrupted Step 3B run and write formal closeout artefacts.

    This closeout is intentionally specific to a no-eligible-entry run.  It does
    not reinterpret a modified threshold and does not infer a trade that did not
    occur.
    """

    if not run_id.strip():
        raise Stage3Step3BCloseoutError("run_id must not be blank")
    if min_fresh_events < 1:
        raise Stage3Step3BCloseoutError("min_fresh_events must be at least 1")
    if abs(float(entry_threshold) - EXPECTED_ENTRY_THRESHOLD) > 1e-12:
        raise Stage3Step3BCloseoutError("Closeout requires the frozen Model B entry threshold 0.55")
    if int(spread_gate_points) != EXPECTED_SPREAD_GATE_POINTS:
        raise Stage3Step3BCloseoutError("Closeout requires the frozen 800-point spread gate")

    fieldnames, all_rows = _read_csv_rows(inputs.events_csv)
    missing = sorted(_required_columns().difference(fieldnames))
    if missing:
        raise Stage3Step3BCloseoutError(f"Events CSV is missing required columns: {missing}")

    run_rows = [row for row in all_rows if str(row.get("run_id", "")).strip() == run_id]
    if not run_rows:
        available = sorted({str(row.get("run_id", "")).strip() for row in all_rows if row.get("run_id")})
        raise Stage3Step3BCloseoutError(f"run_id {run_id!r} not found. Available run_ids: {available}")

    iterations = [_int(row.get("iteration"), field="iteration") for row in run_rows]
    assert all(value is not None for value in iterations)
    iteration_values = [int(value) for value in iterations if value is not None]
    if iteration_values[0] != 1:
        raise Stage3Step3BCloseoutError("Audited run must begin at iteration 1")
    if iteration_values != sorted(iteration_values):
        raise Stage3Step3BCloseoutError("Iterations are not monotonic")
    if len(set(iteration_values)) != len(iteration_values):
        raise Stage3Step3BCloseoutError("Duplicate iteration numbers found")

    fresh_rows = [row for row in run_rows if not _bool(row.get("duplicate_event"), field="duplicate_event")]
    duplicate_rows = [row for row in run_rows if _bool(row.get("duplicate_event"), field="duplicate_event")]
    if len(fresh_rows) < min_fresh_events:
        raise Stage3Step3BCloseoutError(
            f"Only {len(fresh_rows)} fresh completed M15 events found; require at least {min_fresh_events}"
        )

    fresh_times = [_utc_datetime(row.get("event_time_utc"), field="event_time_utc") for row in fresh_rows]
    if fresh_times != sorted(fresh_times):
        raise Stage3Step3BCloseoutError("Fresh completed M15 event times are not monotonic")
    if len(set(fresh_times)) != len(fresh_times):
        raise Stage3Step3BCloseoutError("A fresh completed M15 event was processed more than once")
    cadence_minutes = [
        (current - previous).total_seconds() / 60.0
        for previous, current in zip(fresh_times, fresh_times[1:])
    ]
    if any(abs(delta - 15.0) > 1e-9 for delta in cadence_minutes):
        raise Stage3Step3BCloseoutError(f"Fresh event cadence is not continuously M15: {cadence_minutes}")

    if any(str(row.get("mode", "")).strip() != "watch" for row in run_rows):
        raise Stage3Step3BCloseoutError("Audited run contains a non-watch decision")
    if not _all_false(run_rows, "stale_event_warning"):
        raise Stage3Step3BCloseoutError("Audited run contains a stale-event warning")

    fresh_probabilities = [
        float(_float(row.get("probability_up"), field="probability_up")) for row in fresh_rows
    ]
    max_probability = max(fresh_probabilities)
    min_probability = min(fresh_probabilities)
    eligible_rows = [
        row for row, probability in zip(fresh_rows, fresh_probabilities) if probability >= entry_threshold
    ]
    if eligible_rows:
        raise Stage3Step3BCloseoutError(
            "This closeout path is for a no-eligible-entry run, but at least one p_up reached 0.55"
        )

    fresh_actions = {str(row.get("action", "")).strip() for row in fresh_rows}
    if fresh_actions != {"HOLD_FLAT"}:
        raise Stage3Step3BCloseoutError(f"Unexpected fresh-event actions for no-entry closeout: {sorted(fresh_actions)}")
    if any(str(row.get("reason", "")).strip() != "p_up_below_model_b_entry_threshold" for row in fresh_rows):
        raise Stage3Step3BCloseoutError("A fresh HOLD_FLAT row has an unexpected reason")
    if any(str(row.get("action", "")).strip() != "DUPLICATE_SKIP" for row in duplicate_rows):
        raise Stage3Step3BCloseoutError("A duplicate poll was not recorded as DUPLICATE_SKIP")

    gate_values = {_int(row.get("spread_gate_points"), field="spread_gate_points") for row in run_rows}
    if gate_values != {spread_gate_points}:
        raise Stage3Step3BCloseoutError(f"Spread gate changed inside the run: {sorted(gate_values)}")

    spread_values = [
        int(_int(row.get("spread_points"), field="spread_points"))
        for row in fresh_rows
        if str(row.get("spread_points", "")).strip()
    ]
    if len(spread_values) != len(fresh_rows):
        raise Stage3Step3BCloseoutError("Every fresh event must record a live spread")
    wide_spread_rows = [
        row for row, spread in zip(fresh_rows, spread_values) if spread > spread_gate_points
    ]
    if require_wide_spread_observation and not wide_spread_rows:
        raise Stage3Step3BCloseoutError("No wide-spread event was observed, so the v1.1 continuation fix was not exercised")

    if not _all_false(run_rows, "order_check_called"):
        raise Stage3Step3BCloseoutError("order_check was called despite no eligible Model B entry")
    if not _all_false(run_rows, "order_send_called"):
        raise Stage3Step3BCloseoutError("order_send was called despite no eligible Model B entry")

    for row in run_rows:
        if _int(row.get("live_position_before"), field="live_position_before") != 0:
            raise Stage3Step3BCloseoutError("A decision began with a non-flat Model B state")
        if _int(row.get("live_position_after"), field="live_position_after") != 0:
            raise Stage3Step3BCloseoutError("A decision ended with a non-flat Model B state")
        if _int(row.get("actual_position_count"), field="actual_position_count") != 0:
            raise Stage3Step3BCloseoutError("Broker position count was not zero")
        if _int(row.get("pending_order_count"), field="pending_order_count") != 0:
            raise Stage3Step3BCloseoutError("Pending order count was not zero")

    state = _read_json_mapping(inputs.state_json)
    expected_last_event = fresh_rows[-1]["event_time_utc"]
    if str(state.get("last_event_time_utc", "")) != expected_last_event:
        raise Stage3Step3BCloseoutError("State last_event_time_utc does not match the final fresh event")
    if _int(state.get("records_written"), field="state.records_written") != len(fresh_rows):
        raise Stage3Step3BCloseoutError("State records_written does not match fresh-event count")
    if _int(state.get("live_position"), field="state.live_position") != 0:
        raise Stage3Step3BCloseoutError("Persisted state is not flat")
    if str(state.get("live_position_name", "")).strip().upper() != "FLAT":
        raise Stage3Step3BCloseoutError("Persisted state name is not FLAT")
    for key in ("open_event_time_utc", "open_order_ticket", "position_identifier", "position_ticket"):
        if state.get(key) not in {None, ""}:
            raise Stage3Step3BCloseoutError(f"Persisted state still contains {key}")
    if dict(state.get("successful_entry_dates", {}) or {}):
        raise Stage3Step3BCloseoutError("Persisted state records an entry despite no eligible signal")
    if _int(state.get("completed_entry_exit_cycles", 0), field="state.completed_entry_exit_cycles") != 0:
        raise Stage3Step3BCloseoutError("Persisted state records a completed cycle unexpectedly")

    latest_fields, latest_rows = _read_csv_rows(inputs.latest_decision_csv)
    if not set(_required_columns()).issubset(set(latest_fields)):
        raise Stage3Step3BCloseoutError("Latest-decision CSV does not contain the expected schema")
    if len(latest_rows) != 1:
        raise Stage3Step3BCloseoutError("Latest-decision CSV must contain exactly one decision")
    latest = latest_rows[0]
    if str(latest.get("run_id", "")).strip() != run_id:
        raise Stage3Step3BCloseoutError("Latest-decision run_id does not match the audited run")
    if _int(latest.get("iteration"), field="latest.iteration") != max(iteration_values):
        raise Stage3Step3BCloseoutError("Latest-decision iteration does not match the final logged iteration")
    if str(latest.get("event_time_utc", "")) != expected_last_event:
        raise Stage3Step3BCloseoutError("Latest-decision event time does not match the final fresh event")
    if str(latest.get("action", "")).strip() not in {"DUPLICATE_SKIP", "HOLD_FLAT"}:
        raise Stage3Step3BCloseoutError("Latest decision is not a safe flat action")
    if _int(latest.get("actual_position_count"), field="latest.actual_position_count") != 0:
        raise Stage3Step3BCloseoutError("Latest broker position count is not zero")
    if _int(latest.get("pending_order_count"), field="latest.pending_order_count") != 0:
        raise Stage3Step3BCloseoutError("Latest pending order count is not zero")
    if _bool(latest.get("order_send_called"), field="latest.order_send_called"):
        raise Stage3Step3BCloseoutError("Latest decision indicates order_send was called")

    stale_report_review: dict[str, Any] = {
        "path": None,
        "exists": False,
        "source_run_id": None,
        "matches_audited_run": False,
        "excluded_from_gate": True,
        "reason": "No normal final report was produced because the monitored run ended via Ctrl+C.",
    }
    if inputs.live_report_json is not None:
        stale_report_review["path"] = str(inputs.live_report_json)
        if inputs.live_report_json.exists():
            stale_report_review["exists"] = True
            live_report = _read_json_mapping(inputs.live_report_json)
            report_run_id = str(live_report.get("run_id", "")).strip() or None
            stale_report_review["source_run_id"] = report_run_id
            stale_report_review["matches_audited_run"] = report_run_id == run_id
            stale_report_review["status"] = live_report.get("status")
            if report_run_id == run_id:
                stale_report_review["reason"] = (
                    "The normal report belongs to the audited run, but incremental evidence remains the closeout source."
                )
            else:
                stale_report_review["reason"] = (
                    "The existing normal report belongs to an earlier run and is excluded from this audit."
                )

    source_paths = [inputs.events_csv, inputs.state_json, inputs.latest_decision_csv]
    if inputs.live_report_json is not None and inputs.live_report_json.exists():
        source_paths.append(inputs.live_report_json)
    source_fingerprints = {
        str(path): {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        for path in source_paths
    }

    summary_row: dict[str, Any] = {
        "run_id": run_id,
        "status": "PASS",
        "formal_gate": True,
        "outcome": "PASS_NO_ELIGIBLE_SIGNAL_MANUAL_STOP",
        "termination_reason": termination_reason,
        "total_poll_iterations": len(run_rows),
        "fresh_completed_m15_events": len(fresh_rows),
        "duplicate_polls_skipped": len(duplicate_rows),
        "first_event_time_utc": fresh_rows[0]["event_time_utc"],
        "last_event_time_utc": fresh_rows[-1]["event_time_utc"],
        "observation_minutes_between_first_and_last_event": int(
            (fresh_times[-1] - fresh_times[0]).total_seconds() / 60.0
        ),
        "minimum_probability_up": min_probability,
        "maximum_probability_up": max_probability,
        "entry_threshold": entry_threshold,
        "eligible_entry_events": len(eligible_rows),
        "spread_gate_points": spread_gate_points,
        "minimum_spread_points": min(spread_values),
        "maximum_spread_points": max(spread_values),
        "wide_spread_fresh_events": len(wide_spread_rows),
        "order_check_called_count": 0,
        "order_send_called_count": 0,
        "final_live_position": 0,
        "final_live_position_name": "FLAT",
        "pending_order_count_at_stop": 0,
    }

    validations = {
        "explicit_run_id_isolated": True,
        "minimum_fresh_events_observed": len(fresh_rows) >= min_fresh_events,
        "fresh_events_are_unique": len(set(fresh_times)) == len(fresh_times),
        "fresh_events_follow_15_minute_cadence": True,
        "duplicates_were_skipped": all(str(row.get("action", "")).strip() == "DUPLICATE_SKIP" for row in duplicate_rows),
        "frozen_entry_threshold_preserved": True,
        "frozen_spread_gate_preserved": True,
        "no_eligible_entry_signal_observed": len(eligible_rows) == 0,
        "wide_spread_continuation_observed": len(wide_spread_rows) >= 1,
        "no_order_check_called": True,
        "no_order_send_called": True,
        "all_logged_positions_flat": True,
        "all_logged_pending_orders_zero": True,
        "persisted_state_matches_incremental_log": True,
        "latest_decision_matches_incremental_log": True,
        "stale_normal_report_excluded": True,
    }

    report: dict[str, Any] = {
        "stage": 3,
        "step": "3B-CLOSEOUT",
        "patch_version": "stage3_step3b_closeout_v1_0",
        "status": "PASS",
        "formal_gate": True,
        "purpose": "offline_closeout_audit_for_manually_stopped_model_b_controlled_live_run",
        "audited_run_id": run_id,
        "completed_utc": utc_now_iso(),
        "termination": {
            "reason": termination_reason,
            "normal_finaliser_ran": False,
            "manual_stop_accepted": True,
            "basis": "incremental event log, latest broker-position snapshot fields, and persisted flat state",
        },
        "rules": {
            "variant": "MODEL_B_V2_CURRENT",
            "entry_threshold": entry_threshold,
            "exit_threshold": 0.50,
            "long_only": True,
            "max_successful_entries_per_utc_day": 1,
            "min_hold_bars": 0,
            "spread_gate_points": spread_gate_points,
        },
        "summary": summary_row,
        "validations": validations,
        "stale_normal_report_review": stale_report_review,
        "source_fingerprints": source_fingerprints,
        "outputs": {
            "audit_report_json": str(outputs.audit_report_json),
            "summary_csv": str(outputs.summary_csv),
            "run_events_csv": str(outputs.run_events_csv),
            "fresh_events_csv": str(outputs.fresh_events_csv),
        },
        "decision": {
            "stage3_step3b_closeout_passed": True,
            "another_stage3_step3b_rerun_required": False,
            "next_step": "Stage 3 Step 4A - dual-account and dual-terminal readiness design",
            "do_not_start_final_14_day_run_yet": True,
        },
    }

    _write_csv_atomic(outputs.run_events_csv, run_rows, fieldnames)
    _write_csv_atomic(outputs.fresh_events_csv, fresh_rows, fieldnames)
    _write_csv_atomic(outputs.summary_csv, [summary_row], list(summary_row.keys()))
    _write_json_atomic(outputs.audit_report_json, report)
    return report
