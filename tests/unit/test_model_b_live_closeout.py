from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from capstone_trading.runtime.model_b_live_closeout import (
    CloseoutInputs,
    CloseoutOutputs,
    Stage3Step3BCloseoutError,
    audit_model_b_live_closeout,
)

FIELDS = [
    "run_id", "iteration", "mode", "event_time_utc", "probability_up", "action", "reason",
    "live_position_before", "live_position_after", "duplicate_event", "stale_event_warning",
    "spread_points", "spread_gate_points", "actual_position_count", "pending_order_count",
    "order_check_called", "order_check_passed", "order_check_retcode", "order_check_comment",
    "order_check_margin_required", "order_send_called", "order_send_passed", "order_send_retcode",
    "order_send_comment", "broker_position_ticket", "order_ticket", "decision_utc",
]
RUN_ID = "stage3_step3b_20260713T114117Z"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def base_rows(count: int = 8) -> list[dict]:
    rows = []
    iteration = 1
    for index in range(count):
        hour = 11 + ((15 + index * 15) // 60)
        minute = (15 + index * 15) % 60
        event = f"2026-07-13T{hour:02d}:{minute:02d}:00+00:00"
        spread = 890 if index < 2 else 700
        fresh = {
            "run_id": RUN_ID,
            "iteration": iteration,
            "mode": "watch",
            "event_time_utc": event,
            "probability_up": 0.50 + index * 0.001,
            "action": "HOLD_FLAT",
            "reason": "p_up_below_model_b_entry_threshold",
            "live_position_before": 0,
            "live_position_after": 0,
            "duplicate_event": False,
            "stale_event_warning": False,
            "spread_points": spread,
            "spread_gate_points": 800,
            "actual_position_count": 0,
            "pending_order_count": 0,
            "order_check_called": False,
            "order_check_passed": "",
            "order_check_retcode": "",
            "order_check_comment": "",
            "order_check_margin_required": "",
            "order_send_called": False,
            "order_send_passed": "",
            "order_send_retcode": "",
            "order_send_comment": "",
            "broker_position_ticket": "",
            "order_ticket": "",
            "decision_utc": event,
        }
        rows.append(fresh)
        iteration += 1
        duplicate = dict(fresh)
        duplicate.update({
            "iteration": iteration,
            "action": "DUPLICATE_SKIP",
            "reason": "event_already_processed",
            "duplicate_event": True,
            "spread_points": "",
        })
        rows.append(duplicate)
        iteration += 1
    return rows


def setup_case(tmp_path: Path, rows: list[dict] | None = None):
    rows = rows or base_rows()
    events = tmp_path / "runtime/execution_live/events.csv"
    state = tmp_path / "runtime/state/state.json"
    latest = tmp_path / "runtime/reports/latest.csv"
    old_report = tmp_path / "runtime/reports/live.json"
    write_csv(events, rows)
    fresh = [row for row in rows if not bool(row["duplicate_event"])]
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({
        "completed_entry_exit_cycles": 0,
        "last_event_time_utc": fresh[-1]["event_time_utc"],
        "live_position": 0,
        "live_position_name": "FLAT",
        "open_event_time_utc": None,
        "open_order_ticket": None,
        "position_identifier": None,
        "position_ticket": None,
        "records_written": len(fresh),
        "schema_version": 1,
        "successful_entry_dates": {},
    }), encoding="utf-8")
    write_csv(latest, [rows[-1]])
    old_report.write_text(json.dumps({"run_id": "stage3_step3b_OLDER", "status": "FAIL"}), encoding="utf-8")
    inputs = CloseoutInputs(events, state, latest, old_report)
    outputs = CloseoutOutputs(
        tmp_path / "runtime/reports/audit.json",
        tmp_path / "runtime/reports/summary.csv",
        tmp_path / "runtime/reports/run_events.csv",
        tmp_path / "runtime/reports/fresh_events.csv",
    )
    return inputs, outputs, state, latest


def test_valid_interrupted_no_signal_run_passes(tmp_path: Path):
    inputs, outputs, _, _ = setup_case(tmp_path)
    report = audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)
    assert report["status"] == "PASS"
    assert report["formal_gate"] is True
    assert report["summary"]["fresh_completed_m15_events"] == 8
    assert report["summary"]["wide_spread_fresh_events"] == 2
    assert report["decision"]["another_stage3_step3b_rerun_required"] is False
    assert outputs.audit_report_json.exists()
    assert outputs.fresh_events_csv.exists()


def test_stale_normal_report_is_excluded_not_failed(tmp_path: Path):
    inputs, outputs, _, _ = setup_case(tmp_path)
    report = audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)
    review = report["stale_normal_report_review"]
    assert review["matches_audited_run"] is False
    assert review["excluded_from_gate"] is True


def test_fewer_than_minimum_fresh_events_fails(tmp_path: Path):
    inputs, outputs, _, _ = setup_case(tmp_path, base_rows(7))
    with pytest.raises(Stage3Step3BCloseoutError, match="require at least 8"):
        audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)


def test_eligible_entry_probability_fails_no_signal_closeout(tmp_path: Path):
    rows = base_rows()
    rows[0]["probability_up"] = 0.55
    inputs, outputs, _, _ = setup_case(tmp_path, rows)
    with pytest.raises(Stage3Step3BCloseoutError, match="reached 0.55"):
        audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)


def test_order_send_called_fails(tmp_path: Path):
    rows = base_rows()
    rows[0]["order_send_called"] = True
    inputs, outputs, _, _ = setup_case(tmp_path, rows)
    with pytest.raises(Stage3Step3BCloseoutError, match="order_send was called"):
        audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)


def test_nonflat_logged_position_fails(tmp_path: Path):
    rows = base_rows()
    rows[0]["actual_position_count"] = 1
    inputs, outputs, _, _ = setup_case(tmp_path, rows)
    with pytest.raises(Stage3Step3BCloseoutError, match="Broker position count was not zero"):
        audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)


def test_changed_spread_gate_fails(tmp_path: Path):
    rows = base_rows()
    rows[0]["spread_gate_points"] = 900
    inputs, outputs, _, _ = setup_case(tmp_path, rows)
    with pytest.raises(Stage3Step3BCloseoutError, match="Spread gate changed"):
        audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)


def test_no_wide_spread_observation_fails_by_default(tmp_path: Path):
    rows = base_rows()
    for row in rows:
        if not row["duplicate_event"]:
            row["spread_points"] = 700
    inputs, outputs, _, _ = setup_case(tmp_path, rows)
    with pytest.raises(Stage3Step3BCloseoutError, match="No wide-spread event"):
        audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)


def test_state_last_event_mismatch_fails(tmp_path: Path):
    inputs, outputs, state_path, _ = setup_case(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["last_event_time_utc"] = "2026-07-13T00:00:00+00:00"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(Stage3Step3BCloseoutError, match="State last_event_time_utc"):
        audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)


def test_latest_run_id_mismatch_fails(tmp_path: Path):
    inputs, outputs, _, latest_path = setup_case(tmp_path)
    _, latest_rows = read_csv(latest_path)
    latest_rows[0]["run_id"] = "wrong"
    write_csv(latest_path, latest_rows)
    with pytest.raises(Stage3Step3BCloseoutError, match="Latest-decision run_id"):
        audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)


def test_non_m15_cadence_fails(tmp_path: Path):
    rows = base_rows()
    fresh_indices = [index for index, row in enumerate(rows) if not row["duplicate_event"]]
    rows[fresh_indices[3]]["event_time_utc"] = "2026-07-13T12:01:00+00:00"
    inputs, outputs, _, _ = setup_case(tmp_path, rows)
    with pytest.raises(Stage3Step3BCloseoutError, match="cadence"):
        audit_model_b_live_closeout(run_id=RUN_ID, inputs=inputs, outputs=outputs)


def test_wrong_run_id_fails_with_available_ids(tmp_path: Path):
    inputs, outputs, _, _ = setup_case(tmp_path)
    with pytest.raises(Stage3Step3BCloseoutError, match="Available run_ids"):
        audit_model_b_live_closeout(run_id="missing", inputs=inputs, outputs=outputs)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)
