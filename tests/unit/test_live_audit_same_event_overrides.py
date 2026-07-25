from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from capstone_trading.runtime.live_audit import (
    completed_broker_event_identifier,
    decision_identifier,
    historical_backfill_identifier,
)
from capstone_trading.runtime.live_audit_analysis import (
    analyse_role,
    build_observation_report,
)


def _write_role(role_root: Path, *, role: str = "model_a") -> None:
    role_root.mkdir(parents=True)
    times = pd.date_range("2026-07-24", periods=3, freq="15min", tz="UTC")
    run_id = f"{role}-run-1"
    snapshot_times = [
        times[0],
        times[1] + pd.Timedelta(minutes=15, seconds=2),
        times[2] + pd.Timedelta(minutes=15, seconds=2),
    ]
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * 3,
            "snapshot_id": [f"{role}:s0", f"{role}:s1", f"{role}:s2"],
            "snapshot_utc": snapshot_times,
            "snapshot_phase": ["STARTUP", "POLL", "FINAL"],
            "role": [role] * 3,
            "run_id": [run_id] * 3,
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
    decision_actions = ["HOLD_LONG", "HOLD_LONG", "HOLD_LONG"]
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * 3,
            "decision_id": [
                decision_identifier(
                    role,
                    item.isoformat(),
                    run_id=run_id,
                    iteration=index,
                )
                for index, item in enumerate(times, start=1)
            ],
            "role": [role] * 3,
            "run_id": [run_id] * 3,
            "iteration": [1, 2, 3],
            "event_time_utc": times,
            "decision_utc": times + pd.Timedelta(minutes=15, seconds=1),
            "execution_mode": ["live", "live", "live"],
            "probability_up": [0.75] * 3,
            "model_prediction_available": [True] * 3,
            "model_prediction_event_time_utc": times,
            "model_unavailable_reason": [None] * 3,
            "latest_completed_bar_time_utc": times,
            "event_is_latest_feature": [True] * 3,
            "event_is_latest_completed_bar": [True] * 3,
            "broker_event_disposition": decision_actions,
            "action": decision_actions,
            "reason": ["probability_preserves_current_position"] * 3,
            "position_before": [1, 1, 1],
            "desired_position": [1, 1, 1],
            "target_position": [1, 1, 1],
            "duplicate_event": [False] * 3,
            "gap_from_previous_event": [False] * 3,
            "stale_event_warning": [False] * 3,
            "policy_cap_reached": [False] * 3,
            "entry_blocked_by_policy_cap": [False] * 3,
            "exit_allowed_when_capped": [False] * 3,
            "close_only_reversal": [False] * 3,
            "daily_stop_active": [False] * 3,
            "total_stop_active": [False] * 3,
            "kill_switch_active": [False] * 3,
            "reconciliation_status": ["PASS_STATE_MATCHES_BROKER"] * 3,
            "broker_position_before": [1, 1, 1],
            "broker_position_after_inspection": [1, 1, 1],
            "order_check_called": [False] * 3,
            "order_check_passed": [None] * 3,
            "order_send_called": [False] * 3,
            "order_send_passed": [None] * 3,
            "broker_position_after": [None] * 3,
        }
    ).to_csv(role_root / "decisions.csv", index=False)
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * 3,
            "broker_event_key": [
                completed_broker_event_identifier(role, item.isoformat())
                for item in times
            ],
            "role": [role] * 3,
            "event_time_utc": times,
            "first_observed_utc": times + pd.Timedelta(minutes=15, seconds=1),
            "run_id": [run_id] * 3,
        }
    ).to_csv(role_root / "completed_broker_events.csv", index=False)
    pd.DataFrame(
        {
            "schema_version": ["1.0", "1.0"],
            "runtime_event_id": [f"{role}:start", f"{role}:stop"],
            "timestamp_utc": [times[0], snapshot_times[-1]],
            "role": [role, role],
            "run_id": [run_id, run_id],
            "event_type": ["WORKER_STARTED", "WORKER_STOPPED"],
        }
    ).to_csv(role_root / "runtime_events.csv", index=False)
    (role_root / "final_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "role": role,
                "run_id": run_id,
                "status": "PASS",
                "formal_gate": True,
                "started_utc": times[0].isoformat(),
                "completed_utc": snapshot_times[-1].isoformat(),
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
    action = str(row.get("action", "")).upper()
    mode = str(row.get("execution_mode", "")).lower()
    row["iteration"] = int(pd.to_numeric(decisions["iteration"]).max()) + 1
    row["broker_event_disposition"] = action
    row["duplicate_event"] = False
    event_time = pd.Timestamp(row["event_time_utc"]).tz_convert("UTC")
    if action == "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL":
        row["decision_id"] = historical_backfill_identifier(
            str(row["role"]), event_time.isoformat()
        )
    else:
        row["decision_id"] = decision_identifier(
            str(row["role"]),
            event_time.isoformat(),
            run_id=str(row["run_id"]),
            iteration=int(row["iteration"]),
        )

    flatten_actions = {
        "KILL_SWITCH_FLATTEN": "CONTROL_KILL_SWITCH",
        "TOTAL_STOP_FLATTEN": "CONTROL_TOTAL_STOP",
        "DAILY_STOP_FLATTEN": "CONTROL_DAILY_STOP",
        "SESSION_GAP_LOCKOUT_FLATTEN": "CONTROL_SESSION_GAP_LOCKOUT",
    }
    if action in flatten_actions:
        row["desired_position"] = 0
        row["target_position"] = 0
        row["broker_position_after"] = 0 if mode == "live" else None
        row["kill_switch_active"] = action == "KILL_SWITCH_FLATTEN"
        row["total_stop_active"] = action == "TOTAL_STOP_FLATTEN"
        row["daily_stop_active"] = action == "DAILY_STOP_FLATTEN"
        row["reason"] = {
            "KILL_SWITCH_FLATTEN": "emergency_kill_switch_active",
            "TOTAL_STOP_FLATTEN": "total_drawdown_stop_active",
            "DAILY_STOP_FLATTEN": "daily_loss_stop_active_until_next_utc_day",
            "SESSION_GAP_LOCKOUT_FLATTEN": "daily_market_break_lockout",
        }[action]
        if mode == "live":
            row["order_check_called"] = True
            row["order_check_passed"] = True
            row["order_send_called"] = True
            row["order_send_passed"] = True
        else:
            row["order_check_called"] = False
            row["order_check_passed"] = False
            row["order_send_called"] = False
            row["order_send_passed"] = False
        future_mask = pd.to_datetime(
            decisions["event_time_utc"], utc=True
        ) > event_time
        decisions.loc[future_mask, "position_before"] = 0
        decisions.loc[future_mask, "desired_position"] = 0
        decisions.loc[future_mask, "target_position"] = 0
        decisions.loc[future_mask, "action"] = "HOLD_FLAT"
        decisions.loc[future_mask, "broker_event_disposition"] = "HOLD_FLAT"
        decisions.loc[future_mask, "reason"] = "probability_preserves_current_position"
        decisions.loc[future_mask, "broker_position_before"] = 0
        decisions.loc[future_mask, "broker_position_after_inspection"] = 0
    elif action == "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL":
        row["desired_position"] = row["position_before"]
        row["target_position"] = row["position_before"]
        row["stale_event_warning"] = True
        row["order_check_called"] = False
        row["order_check_passed"] = None
        row["order_send_called"] = False
        row["order_send_passed"] = None
        row["broker_position_after"] = None
        row["probability_up"] = None
        row["model_prediction_available"] = False
        row["model_prediction_event_time_utc"] = None
        row["model_unavailable_reason"] = (
            "completed_broker_event_discovered_after_worker_outage"
        )
        row["latest_completed_bar_time_utc"] = None
        row["event_is_latest_feature"] = None
        row["event_is_latest_completed_bar"] = None

    for column in decisions.columns:
        if decisions[column].isna().all():
            decisions[column] = decisions[column].astype("object")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA",
            category=FutureWarning,
        )
        decisions.loc[len(decisions)] = row
    decisions.to_csv(path, index=False)

    report_path = role_root / "final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["state"]["records_written"] = len(decisions)

    if action in flatten_actions and mode == "live":
        event_time = str(row["event_time_utc"])
        run_id = str(row["run_id"])
        role = str(row["role"])
        trigger = flatten_actions[action]
        event_token = event_time.replace(":", "")
        control_id = (
            f"{trigger}:{role}:{run_id}:{int(row['iteration'])}:{event_token}"
        )
        pd.DataFrame(
            {
                "schema_version": ["1.0"],
                "execution_id": [f"{control_id}:1:close"],
                "decision_id": [control_id],
                "role": [role],
                "run_id": [run_id],
                "trigger_type": [trigger],
                "event_time_utc": [event_time],
                "completed_utc": [row["decision_utc"]],
                "order_send_passed": [True],
                "order_ticket": [101],
                "deal_ticket": [201],
                "requested_volume": [0.01],
                "requested_price": [2400.0],
                "broker_result_price": [2400.0],
                "symbol_point": [0.01],
                "spread_points_before": [20.0],
            }
        ).to_csv(role_root / "order_events.csv", index=False)
        pd.DataFrame(
            {
                "schema_version": ["1.0"],
                "history_key": ["order:101"],
                "role": [role],
                "run_id": [run_id],
                "ticket": [101],
                "time_done_utc": [row["decision_utc"]],
            }
        ).to_csv(role_root / "broker_orders.csv", index=False)
        pd.DataFrame(
            {
                "schema_version": ["1.0"],
                "history_key": ["deal:201"],
                "role": [role],
                "run_id": [run_id],
                "ticket": [201],
                "order": [101],
                "position_id": [500],
                "time_utc": [row["decision_utc"]],
                "entry": [1],
                "type": [1],
                "volume": [0.01],
                "price": [2400.0],
                "profit": [0.0],
                "commission": [0.0],
                "swap": [0.0],
                "fee": [0.0],
            }
        ).to_csv(role_root / "broker_deals.csv", index=False)
        report["state"]["order_send_calls"] = 1
        report["state"]["successful_order_sends"] = 1

        required_event_type = {
            "TOTAL_STOP_FLATTEN": "TOTAL_STOP_TRIGGERED",
            "DAILY_STOP_FLATTEN": "DAILY_STOP_TRIGGERED",
            "SESSION_GAP_LOCKOUT_FLATTEN": "SESSION_GAP_LOCKOUT_STARTED",
        }.get(action)
        if required_event_type:
            runtime_path = role_root / "runtime_events.csv"
            runtime_events = pd.read_csv(runtime_path)
            runtime_events = pd.concat(
                [
                    runtime_events,
                    pd.DataFrame(
                        [
                            {
                                "schema_version": "1.0",
                                "runtime_event_id": (
                                    f"{role}:{run_id}:{row['iteration']}:"
                                    f"{required_event_type}"
                                ),
                                "timestamp_utc": row["decision_utc"],
                                "role": role,
                                "run_id": run_id,
                                "event_type": required_event_type,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            runtime_events.to_csv(runtime_path, index=False)

    if action in flatten_actions and mode != "live":
        required_event_type = {
            "TOTAL_STOP_FLATTEN": "TOTAL_STOP_TRIGGERED",
            "DAILY_STOP_FLATTEN": "DAILY_STOP_TRIGGERED",
            "SESSION_GAP_LOCKOUT_FLATTEN": "SESSION_GAP_LOCKOUT_STARTED",
        }.get(action)
        if required_event_type:
            runtime_path = role_root / "runtime_events.csv"
            runtime_events = pd.read_csv(runtime_path)
            runtime_events = pd.concat(
                [
                    runtime_events,
                    pd.DataFrame(
                        [
                            {
                                "schema_version": "1.0",
                                "runtime_event_id": (
                                    f"{row['role']}:{row['run_id']}:"
                                    f"{row['iteration']}:{required_event_type}"
                                ),
                                "timestamp_utc": row["decision_utc"],
                                "role": row["role"],
                                "run_id": row["run_id"],
                                "event_type": required_event_type,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            runtime_events.to_csv(runtime_path, index=False)

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
        decisions["broker_position_after"] = None
        for column in (
            "order_check_called",
            "order_check_passed",
            "order_send_called",
            "order_send_passed",
        ):
            decisions[column] = False
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
    decisions.loc[0, "broker_event_disposition"] = first_action
    if first_action == "BLOCK_SPREAD":
        decisions.loc[0, "desired_position"] = 0
    else:
        decisions.loc[0, "desired_position"] = 1
        decisions.loc[0, "reconciliation_status"] = "BLOCKED_BROKER_STATE_MISMATCH"
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


def test_control_execution_requires_exact_synthetic_control_id(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    _write_role(role_root)
    _append_same_event_row(
        role_root,
        decision_utc="2026-07-24T00:16:00+00:00",
        execution_mode="live",
        action="DAILY_STOP_FLATTEN",
        position_before=1,
        broker_position_before=1,
        target_position=0,
        broker_position_after_inspection=0,
    )
    events_path = role_root / "order_events.csv"
    events = pd.read_csv(events_path)
    events.loc[0, "decision_id"] = "CONTROL_WRONG:anything"
    events.to_csv(events_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert (
        "live_transition_missing_successful_order_event=1"
        in summary.decision_evidence_issue_reasons
    )
