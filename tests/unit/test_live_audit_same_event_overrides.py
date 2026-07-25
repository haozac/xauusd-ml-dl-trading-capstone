from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from capstone_trading.runtime.live_audit_analysis import (
    analyse_role,
    build_observation_report,
)


def _write_role(role_root: Path, *, role: str = "model_a") -> None:
    role_root.mkdir(parents=True)
    times = pd.date_range("2026-07-24", periods=3, freq="1D", tz="UTC")
    pd.DataFrame(
        {
            "snapshot_id": [f"{role}:s0", f"{role}:s1", f"{role}:s2"],
            "snapshot_utc": times,
            "snapshot_phase": ["STARTUP", "POLL", "FINAL"],
            "run_id": [f"{role}-run-1"] * 3,
            "worker_pid": [100] * 3,
            "terminal_connected": [True] * 3,
            "balance": [10000.0] * 3,
            "equity": [10000.0] * 3,
            "spread_points": [10.0] * 3,
            "broker_position": [0] * 3,
            "pending_order_count": [0] * 3,
            "reconciliation_status": [
                "UNINITIALISED",
                "PASS_STATE_MATCHES_BROKER",
                "CLEAN_STOP_FLAT_CONFIRMED",
            ],
            "latest_completed_event_time_utc": times,
            "latest_decision_event_time_utc": times,
        }
    ).to_csv(role_root / "telemetry.csv", index=False)
    pd.DataFrame(
        {
            "decision_id": [f"{role}:d0", f"{role}:d1", f"{role}:d2"],
            "event_time_utc": times,
            "decision_utc": times + pd.Timedelta(minutes=15, seconds=1),
            "execution_mode": ["live", "live", "live"],
            "action": ["ENTER_LONG", "HOLD_LONG", "EXIT_POSITION"],
            "position_before": [0, 1, 1],
            "target_position": [1, 1, 0],
            "broker_position_before": [0, 1, 1],
            "broker_position_after_inspection": [1, 1, 0],
        }
    ).to_csv(role_root / "decisions.csv", index=False)
    pd.DataFrame(
        {
            "broker_event_key": [
                f"BROKER_EVENT:{role}:{item.isoformat()}" for item in times
            ],
            "role": [role] * 3,
            "event_time_utc": times,
        }
    ).to_csv(role_root / "completed_broker_events.csv", index=False)
    pd.DataFrame(
        {
            "runtime_event_id": [f"{role}:start", f"{role}:stop"],
            "timestamp_utc": [times[0], times[-1]],
            "event_type": ["WORKER_STARTED", "WORKER_STOPPED"],
        }
    ).to_csv(role_root / "runtime_events.csv", index=False)
    (role_root / "final_report.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "formal_gate": True,
                "state": {
                    "records_written": 3,
                    "order_send_calls": 0,
                    "successful_order_sends": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def _append_same_event_row(role_root: Path, **updates: object) -> None:
    path = role_root / "decisions.csv"
    decisions = pd.read_csv(path)
    row = decisions.iloc[0].copy()
    for key, value in updates.items():
        row[key] = value
    decisions = pd.concat([decisions, pd.DataFrame([row])], ignore_index=True)
    decisions.to_csv(path, index=False)
    report_path = role_root / "final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["state"]["records_written"] = len(decisions)
    report_path.write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize(
    "action",
    [
        "KILL_SWITCH_FLATTEN",
        "TOTAL_STOP_FLATTEN",
        "DAILY_STOP_FLATTEN",
        "SESSION_GAP_LOCKOUT_FLATTEN",
    ],
)
def test_recognised_same_event_safety_flatten_is_allowed(
    tmp_path: Path,
    action: str,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    _write_role(role_root)
    _append_same_event_row(
        role_root,
        decision_id=f"control:{action}",
        decision_utc="2026-07-24T00:16:00+00:00",
        execution_mode="live",
        action=action,
        position_before=1,
        broker_position_before=1,
        target_position=0,
        broker_position_after_inspection=0,
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is True
    assert summary.operational_acceptance_status == "PASS"
    assert summary.broker_event_with_multiple_dispositions_count == 1
    assert summary.allowed_same_event_safety_override_count == 1
    assert summary.unexpected_multiple_disposition_event_count == 0
    assert summary.maximum_dispositions_per_broker_event == 2


def test_two_ordinary_same_event_decisions_fail(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    _write_role(role_root)
    _append_same_event_row(
        role_root,
        decision_id="ordinary-duplicate",
        decision_utc="2026-07-24T00:16:00+00:00",
        action="ENTER_LONG",
        position_before=0,
        target_position=1,
        broker_position_after_inspection=1,
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "FAIL"
    assert summary.allowed_same_event_safety_override_count == 0
    assert summary.unexpected_multiple_disposition_event_count == 1
    assert "unexpected_multiple_disposition_events=1" in summary.audit_gate_failures


def test_backfill_and_current_same_event_fail(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    _write_role(role_root)
    _append_same_event_row(
        role_root,
        decision_id="backfill-collision",
        decision_utc="2026-07-24T00:16:00+00:00",
        action="MODEL_UNAVAILABLE_HISTORICAL_BACKFILL",
        position_before=0,
        target_position=0,
        broker_position_after_inspection=0,
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "FAIL"
    assert summary.unexpected_multiple_disposition_event_count == 1
    assert "unexpected_multiple_disposition_events=1" in summary.audit_gate_failures


def test_allowed_override_fields_are_exported_in_consolidated_report(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_role(runtime / "model_a", role="model_a")
    _write_role(runtime / "model_b", role="model_b")
    _append_same_event_row(
        runtime / "model_a",
        decision_id="control:DAILY_STOP_FLATTEN",
        decision_utc="2026-07-24T00:16:00+00:00",
        execution_mode="live",
        action="DAILY_STOP_FLATTEN",
        position_before=1,
        broker_position_before=1,
        target_position=0,
        broker_position_after_inspection=0,
    )

    output = tmp_path / "report"
    report = build_observation_report(
        runtime,
        output,
        expected_poll_seconds=86400,
    )

    assert report["formal_audit_gate"] is True
    model_a = next(item for item in report["models"] if item["role"] == "model_a")
    assert model_a["allowed_same_event_safety_override_count"] == 1
    assert model_a["unexpected_multiple_disposition_event_count"] == 0
    saved = json.loads(
        (output / "consolidated_observation_report.json").read_text(
            encoding="utf-8"
        )
    )
    saved_model_a = next(
        item for item in saved["models"] if item["role"] == "model_a"
    )
    assert saved_model_a["allowed_same_event_safety_override_count"] == 1

def _set_role_execution_mode(role_root: Path, mode: str) -> None:
    path = role_root / "decisions.csv"
    decisions = pd.read_csv(path)
    decisions["execution_mode"] = mode
    if mode == "shadow":
        decisions["broker_position_before"] = 0
        decisions["broker_position_after_inspection"] = 0
    decisions.to_csv(path, index=False)


def test_shadow_same_event_safety_flatten_is_allowed_by_analyse_role(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    _write_role(role_root)
    _set_role_execution_mode(role_root, "shadow")
    _append_same_event_row(
        role_root,
        decision_id="shadow-control:DAILY_STOP_FLATTEN",
        decision_utc="2026-07-24T00:16:00+00:00",
        execution_mode="shadow",
        action="DAILY_STOP_FLATTEN",
        position_before=1,
        broker_position_before=0,
        target_position=0,
        broker_position_after_inspection=0,
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is True
    assert summary.operational_acceptance_status == "PASS"
    assert summary.allowed_same_event_safety_override_count == 1
    assert summary.unexpected_multiple_disposition_event_count == 0


@pytest.mark.parametrize("later_broker_before", [None, 0, -1])
def test_live_mismatched_broker_before_fails_full_analyse_role(
    tmp_path: Path,
    later_broker_before: int | None,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    _write_role(role_root)
    _append_same_event_row(
        role_root,
        decision_id=f"invalid-live-broker-before:{later_broker_before}",
        decision_utc="2026-07-24T00:16:00+00:00",
        execution_mode="live",
        action="DAILY_STOP_FLATTEN",
        position_before=1,
        broker_position_before=later_broker_before,
        target_position=0,
        broker_position_after_inspection=0,
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "FAIL"
    assert summary.allowed_same_event_safety_override_count == 0
    assert summary.unexpected_multiple_disposition_event_count == 1
    assert "unexpected_multiple_disposition_events=1" in summary.audit_gate_failures


@pytest.mark.parametrize("first_action", ["BLOCK_SPREAD", "BLOCK_RECONCILIATION"])
def test_exposure_preserving_block_can_precede_live_safety_flatten(
    tmp_path: Path,
    first_action: str,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    _write_role(role_root)
    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[0, "action"] = first_action
    decisions.to_csv(decisions_path, index=False)
    _append_same_event_row(
        role_root,
        decision_id=f"control-after:{first_action}",
        decision_utc="2026-07-24T00:16:00+00:00",
        execution_mode="live",
        action="DAILY_STOP_FLATTEN",
        position_before=1,
        broker_position_before=1,
        target_position=0,
        broker_position_after_inspection=0,
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is True
    assert summary.allowed_same_event_safety_override_count == 1
    assert summary.unexpected_multiple_disposition_event_count == 0

