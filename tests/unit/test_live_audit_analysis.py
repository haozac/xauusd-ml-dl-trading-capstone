from __future__ import annotations

import json
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


def write_broker_event_ledger(
    role_root: Path,
    role: str,
    events: list[object] | pd.DatetimeIndex,
    *,
    run_id: str | None = None,
) -> None:
    parsed = pd.to_datetime(list(events), utc=True)
    selected_run_id = run_id or f"dual_{role}_test"
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * len(parsed),
            "broker_event_key": [
                completed_broker_event_identifier(role, item.isoformat())
                for item in parsed
            ],
            "role": [role] * len(parsed),
            "event_time_utc": parsed,
            "first_observed_utc": parsed + pd.Timedelta(minutes=15, seconds=1),
            "observation_type": ["CURRENT_COMPLETED_EVENT"] * len(parsed),
            "run_id": [selected_run_id] * len(parsed),
            "iteration": list(range(1, len(parsed) + 1)),
            "worker_pid": [1234] * len(parsed),
            "is_latest_current_event": [True] * len(parsed),
            "is_historical_recovered_event": [False] * len(parsed),
            "source_fetch_count": [2] * len(parsed),
        }
    ).to_csv(role_root / "completed_broker_events.csv", index=False)


def write_role(root: Path, role: str, equity: list[float]) -> None:
    role_root = root / role
    role_root.mkdir(parents=True)
    times = pd.date_range("2026-07-24", periods=len(equity), freq="15min", tz="UTC")
    run_id = f"dual_{role}_test"
    snapshot_times = list(times)
    if snapshot_times:
        snapshot_times[-1] = times[-1] + pd.Timedelta(minutes=15, seconds=2)
    phases = ["STARTUP"] + ["POLL"] * max(0, len(equity) - 2) + ["FINAL"]
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * len(equity),
            "snapshot_id": [f"{role}:{index}" for index in range(len(equity))],
            "snapshot_utc": snapshot_times,
            "snapshot_phase": phases,
            "role": [role] * len(equity),
            "run_id": [run_id] * len(equity),
            "worker_pid": [1234] * len(equity),
            "terminal_connected": [True] * len(equity),
            "balance": [equity[0]] * len(equity),
            "equity": equity,
            "spread_points": [10] * len(equity),
            "broker_position": [0] * len(equity),
            "pending_order_count": [0] * len(equity),
            "reconciliation_status": [
                "UNINITIALISED",
                *["PASS_STATE_MATCHES_BROKER"] * (len(equity) - 1),
            ],
            "latest_completed_event_time_utc": times,
            "latest_decision_event_time_utc": times,
        }
    ).to_csv(role_root / "telemetry.csv", index=False)
    decision_actions = ["ENTER_LONG"] + ["HOLD_LONG"] * (len(equity) - 1)
    decision_positions_before = [0] + [1] * (len(equity) - 1)
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * len(equity),
            "decision_id": [
                decision_identifier(
                    role,
                    item.isoformat(),
                    run_id=run_id,
                    iteration=index,
                )
                for index, item in enumerate(times, start=1)
            ],
            "role": [role] * len(equity),
            "run_id": [run_id] * len(equity),
            "iteration": list(range(1, len(equity) + 1)),
            "event_time_utc": times,
            "decision_utc": times + pd.Timedelta(minutes=15, seconds=1),
            "execution_mode": ["shadow"] * len(equity),
            "probability_up": [0.75] * len(equity),
            "model_prediction_available": [True] * len(equity),
            "model_prediction_event_time_utc": times,
            "model_unavailable_reason": [None] * len(equity),
            "latest_completed_bar_time_utc": times,
            "event_is_latest_feature": [True] * len(equity),
            "event_is_latest_completed_bar": [True] * len(equity),
            "broker_event_disposition": decision_actions,
            "action": decision_actions,
            "reason": ["frozen_overlay_transition"] * len(equity),
            "position_before": decision_positions_before,
            "desired_position": [1] * len(equity),
            "target_position": [1] * len(equity),
            "duplicate_event": [False] * len(equity),
            "gap_from_previous_event": [False] * len(equity),
            "stale_event_warning": [False] * len(equity),
            "policy_cap_reached": [False] * len(equity),
            "entry_blocked_by_policy_cap": [False] * len(equity),
            "exit_allowed_when_capped": [False] * len(equity),
            "close_only_reversal": [False] * len(equity),
            "daily_stop_active": [False] * len(equity),
            "total_stop_active": [False] * len(equity),
            "kill_switch_active": [False] * len(equity),
            "reconciliation_status": ["PASS_SHADOW_BROKER_FLAT"] * len(equity),
            "broker_position_before": [0] * len(equity),
            "broker_position_after_inspection": [0] * len(equity),
            "order_check_called": [False] * len(equity),
            "order_check_passed": [None] * len(equity),
            "order_send_called": [False] * len(equity),
            "order_send_passed": [None] * len(equity),
            "broker_position_after": [None] * len(equity),
        }
    ).to_csv(role_root / "decisions.csv", index=False)
    write_broker_event_ledger(role_root, role, times, run_id=run_id)
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
                    "records_written": len(equity),
                    "order_send_calls": 0,
                    "successful_order_sends": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def write_completed_trade(role_root: Path, *, include_unlinked_order: bool = False) -> None:
    role_root.mkdir(parents=True)
    event_times = pd.to_datetime(
        [
            "2026-07-24T00:00:00Z",
            "2026-07-24T00:15:00Z",
        ],
        utc=True,
    )
    decision_times = event_times + pd.Timedelta(minutes=15, seconds=1)
    times = pd.date_range(
        "2026-07-24T00:00:00Z",
        "2026-07-24T00:30:00Z",
        freq="30s",
        tz="UTC",
    ).append(pd.DatetimeIndex(["2026-07-24T00:30:02Z"]))
    telemetry_count = len(times)
    exposed = (times >= decision_times[0]) & (times < decision_times[1])
    after_exit = times >= decision_times[1]
    run_id = "run-1"
    decision_ids = [
        decision_identifier(
            "model_a",
            event.isoformat(),
            run_id=run_id,
            iteration=iteration,
        )
        for iteration, event in enumerate(event_times, start=1)
    ]
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * telemetry_count,
            "snapshot_id": [
                f"model_a:snapshot:{index}" for index in range(telemetry_count)
            ],
            "snapshot_utc": times,
            "snapshot_phase": ["STARTUP"]
            + ["POLL"] * (telemetry_count - 2)
            + ["FINAL"],
            "role": ["model_a"] * telemetry_count,
            "run_id": [run_id] * telemetry_count,
            "worker_pid": [100] * telemetry_count,
            "terminal_connected": [True] * telemetry_count,
            "balance": [10009.0 if value else 10000.0 for value in after_exit],
            "equity": [
                10009.0 if exited else (10005.0 if open_ else 10000.0)
                for open_, exited in zip(exposed, after_exit, strict=True)
            ],
            "spread_points": [20.0] * telemetry_count,
            "broker_position": [1 if value else 0 for value in exposed],
            "pending_order_count": [0] * telemetry_count,
            "reconciliation_status": ["UNINITIALISED"]
            + ["PASS_STATE_MATCHES_BROKER"] * (telemetry_count - 2)
            + ["CLEAN_STOP_FLAT_CONFIRMED"],
            "latest_completed_event_time_utc": [
                (
                    event_times[-1]
                    if timestamp >= decision_times[-1]
                    else (
                        event_times[0]
                        if timestamp >= decision_times[0]
                        else None
                    )
                )
                for timestamp in times
            ],
            "latest_decision_event_time_utc": [
                (
                    event_times[-1]
                    if timestamp >= decision_times[-1]
                    else (
                        event_times[0]
                        if timestamp >= decision_times[0]
                        else None
                    )
                )
                for timestamp in times
            ],
        }
    ).to_csv(role_root / "telemetry.csv", index=False)
    decision_actions = ["ENTER_LONG", "EXIT_POSITION"]
    pd.DataFrame(
        {
            "schema_version": ["1.0", "1.0"],
            "decision_id": decision_ids,
            "role": ["model_a", "model_a"],
            "run_id": ["run-1", "run-1"],
            "iteration": [1, 2],
            "event_time_utc": event_times,
            "decision_utc": decision_times,
            "execution_mode": ["live", "live"],
            "probability_up": [0.75, 0.45],
            "model_prediction_available": [True, True],
            "model_prediction_event_time_utc": event_times,
            "model_unavailable_reason": [None, None],
            "latest_completed_bar_time_utc": event_times,
            "event_is_latest_feature": [True, True],
            "event_is_latest_completed_bar": [True, True],
            "broker_event_disposition": decision_actions,
            "action": decision_actions,
            "reason": ["frozen_overlay_transition", "frozen_overlay_transition"],
            "position_before": [0, 1],
            "desired_position": [1, 0],
            "target_position": [1, 0],
            "duplicate_event": [False, False],
            "gap_from_previous_event": [False, False],
            "stale_event_warning": [False, False],
            "policy_cap_reached": [False, False],
            "entry_blocked_by_policy_cap": [False, False],
            "exit_allowed_when_capped": [False, False],
            "close_only_reversal": [False, False],
            "daily_stop_active": [False, False],
            "total_stop_active": [False, False],
            "kill_switch_active": [False, False],
            "reconciliation_status": [
                "PASS_STATE_MATCHES_BROKER",
                "PASS_STATE_MATCHES_BROKER",
            ],
            "broker_position_before": [0, 1],
            "broker_position_after_inspection": [1, 0],
            "order_check_called": [True, True],
            "order_check_passed": [True, True],
            "order_send_called": [True, True],
            "order_send_passed": [True, True],
            "broker_position_after": [1, 0],
        }
    ).to_csv(role_root / "decisions.csv", index=False)
    write_broker_event_ledger(
        role_root, "model_a", event_times, run_id=run_id
    )
    pd.DataFrame(
        {
            "schema_version": ["1.0", "1.0"],
            "execution_id": ["e-entry", "e-exit"],
            "decision_id": decision_ids,
            "role": ["model_a", "model_a"],
            "run_id": ["run-1", "run-1"],
            "trigger_type": ["STRATEGY_DECISION", "STRATEGY_DECISION"],
            "event_time_utc": event_times,
            "completed_utc": decision_times,
            "position_before": [0, 1],
            "position_after": [1, 0],
            "order_send_passed": [True, True],
            "order_ticket": [101, 102],
            "deal_ticket": [None, None],
            "requested_volume": [0.01, 0.01],
            "slippage_points_adverse": [1.0, 2.0],
            "spread_points_before": [20.0, 22.0],
        }
    ).to_csv(role_root / "order_events.csv", index=False)
    pd.DataFrame(
        {
            "schema_version": ["1.0", "1.0"],
            "history_key": ["deal:201", "deal:202"],
            "role": ["model_a", "model_a"],
            "run_id": [run_id, run_id],
            "ticket": [201, 202],
            "order": [101, 102],
            "position_id": [500, 500],
            "time_utc": decision_times,
            "entry": [0, 1],
            "type": [0, 1],
            "volume": [0.01, 0.01],
            "price": [2400.0, 2401.0],
            "profit": [0.0, 10.0],
            "commission": [-0.5, -0.5],
            "swap": [0.0, 0.0],
            "fee": [0.0, 0.0],
        }
    ).to_csv(role_root / "broker_deals.csv", index=False)
    order_tickets = [101, 102, 999] if include_unlinked_order else [101, 102]
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * len(order_tickets),
            "history_key": [f"order:{ticket}" for ticket in order_tickets],
            "role": ["model_a"] * len(order_tickets),
            "run_id": [run_id] * len(order_tickets),
            "ticket": order_tickets,
            "time_done_utc": list(decision_times)
            + ([decision_times[-1]] if include_unlinked_order else []),
        }
    ).to_csv(role_root / "broker_orders.csv", index=False)
    pd.DataFrame(
        {
            "schema_version": ["1.0", "1.0"],
            "runtime_event_id": ["start", "stop"],
            "timestamp_utc": [times[0], times[-1]],
            "role": ["model_a", "model_a"],
            "run_id": [run_id, run_id],
            "event_type": ["WORKER_STARTED", "WORKER_STOPPED"],
        }
    ).to_csv(role_root / "runtime_events.csv", index=False)
    (role_root / "final_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "role": "model_a",
                "run_id": run_id,
                "status": "PASS",
                "formal_gate": True,
                "started_utc": times[0].isoformat(),
                "completed_utc": times[-1].isoformat(),
                "state": {
                    "records_written": 2,
                    "order_send_calls": 2,
                    "successful_order_sends": 2,
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_observation_report_is_offline_and_daily_partitioned(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    output = tmp_path / "report"
    write_role(runtime, "model_a", [10000.0, 10000.0, 10000.0])
    write_role(runtime, "model_b", [10000.0, 10000.0, 10000.0])

    report = build_observation_report(runtime, output, expected_poll_seconds=86400)

    assert len(report["models"]) == 2
    assert report["formal_audit_gate"] is True
    assert (output / "consolidated_model_summary.csv").exists()
    assert (output / "consolidated_observation_report.json").exists()
    assert (output / "model_a_daily_summary.csv").exists()
    assert (output / "model_a_trade_ledger.csv").exists()
    assert (
        output / "daily" / "2026-07-24" / "model_a" / "telemetry.csv"
    ).exists()


def test_broker_deals_can_be_recovered_by_order_ticket(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is True
    assert summary.completed_trade_count == 1
    assert summary.realised_net_pnl == 9.0
    assert summary.balance_pnl_reconciliation_difference == 0.0
    assert summary.recovered_deal_link_by_order_count == 2
    assert summary.broker_fill_price_recovered_count == 2
    assert summary.successful_order_missing_fill_price_count == 0
    assert summary.successful_order_event_missing_deal_ticket_count == 2
    assert summary.missing_broker_deal_link_count == 0


def test_unlinked_broker_order_fails_audit_gate(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root, include_unlinked_order=True)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.broker_order_missing_execution_link_count == 1
    assert "broker_orders_without_execution=1" in summary.audit_gate_failures


def test_successful_order_without_fill_price_fails_audit_gate(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)
    deals_path = role_root / "broker_deals.csv"
    deals = pd.read_csv(deals_path)
    deals["price"] = None
    deals.to_csv(deals_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.successful_order_missing_fill_price_count == 2
    assert "successful_orders_missing_fill_price=2" in summary.audit_gate_failures


def test_completed_broker_events_without_decisions_fail_gate(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    ledger_path = role_root / "completed_broker_events.csv"
    ledger = pd.read_csv(ledger_path)
    missing_event = {
        "broker_event_key": "BROKER_EVENT:model_a:2026-07-24T00:00:30Z",
        "role": "model_a",
        "event_time_utc": "2026-07-24T00:00:30+00:00",
        "first_observed_utc": "2026-07-24T00:00:31+00:00",
        "observation_type": "CURRENT_COMPLETED_EVENT",
        "run_id": "run-1",
        "iteration": 2,
        "worker_pid": 100,
        "is_latest_current_event": True,
        "is_historical_recovered_event": False,
        "source_fetch_count": 2,
    }
    ledger = pd.concat(
        [ledger, pd.DataFrame([missing_event])],
        ignore_index=True,
    )
    ledger.to_csv(ledger_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.missing_completed_event_decision_count == 1
    assert summary.completed_event_coverage_ratio == 2 / 3
    assert (
        "missing_completed_event_dispositions=1"
        in summary.audit_gate_failures
    )


def test_control_execution_does_not_require_strategy_decision_link(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[1, "action"] = "HOLD_LONG"
    decisions.loc[1, "broker_event_disposition"] = "HOLD_LONG"
    decisions.loc[1, "desired_position"] = 1
    decisions.loc[1, "target_position"] = 1
    decisions.loc[1, "broker_position_after_inspection"] = 1
    decisions.loc[1, "order_check_called"] = False
    decisions.loc[1, "order_check_passed"] = False
    decisions.loc[1, "order_send_called"] = False
    decisions.loc[1, "order_send_passed"] = False
    decisions.loc[1, "broker_position_after"] = None
    decisions.to_csv(decisions_path, index=False)

    events_path = role_root / "order_events.csv"
    events = pd.read_csv(events_path)
    events.loc[1, "trigger_type"] = "CONTROL_CLEAN_STOP"
    events.loc[1, "decision_id"] = "CONTROL_CLEAN_STOP:model_a:run-1:3"
    events.to_csv(events_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is True
    assert summary.control_execution_count == 1
    assert summary.strategy_execution_missing_decision_link_count == 0
    assert summary.invalid_control_execution_link_count == 0


def test_clean_stop_control_execution_rejects_unrelated_control_id(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    write_completed_trade(role_root)

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[1, "action"] = "HOLD_LONG"
    decisions.loc[1, "broker_event_disposition"] = "HOLD_LONG"
    decisions.loc[1, "desired_position"] = 1
    decisions.loc[1, "target_position"] = 1
    decisions.loc[1, "broker_position_after_inspection"] = 1
    decisions.loc[1, "order_check_called"] = False
    decisions.loc[1, "order_check_passed"] = False
    decisions.loc[1, "order_send_called"] = False
    decisions.loc[1, "order_send_passed"] = False
    decisions.loc[1, "broker_position_after"] = None
    decisions.to_csv(decisions_path, index=False)

    events_path = role_root / "order_events.csv"
    events = pd.read_csv(events_path)
    events.loc[1, "trigger_type"] = "CONTROL_CLEAN_STOP"
    events.loc[1, "decision_id"] = "CONTROL_WRONG:anything"
    events.to_csv(events_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.invalid_control_execution_link_count == 1
    assert "invalid_control_execution_links=1" in summary.audit_gate_failures


def test_clean_stop_control_bridges_position_continuity_across_restart(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    write_completed_trade(role_root)

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[1, "action"] = "HOLD_LONG"
    decisions.loc[1, "broker_event_disposition"] = "HOLD_LONG"
    decisions.loc[1, "desired_position"] = 1
    decisions.loc[1, "target_position"] = 1
    decisions.loc[1, "broker_position_after_inspection"] = 1
    decisions.loc[1, "order_check_called"] = False
    decisions.loc[1, "order_check_passed"] = False
    decisions.loc[1, "order_send_called"] = False
    decisions.loc[1, "order_send_passed"] = False
    decisions.loc[1, "broker_position_after"] = None

    restart_event = pd.Timestamp("2026-07-24T00:30:00Z")
    restart_decision_time = restart_event + pd.Timedelta(minutes=15, seconds=1)
    restarted = decisions.iloc[1].copy()
    restarted["run_id"] = "run-2"
    restarted["iteration"] = 1
    restarted["event_time_utc"] = restart_event.isoformat()
    restarted["decision_utc"] = restart_decision_time.isoformat()
    restarted["decision_id"] = decision_identifier(
        "model_a",
        restart_event.isoformat(),
        run_id="run-2",
        iteration=1,
    )
    restarted["action"] = "HOLD_FLAT"
    restarted["broker_event_disposition"] = "HOLD_FLAT"
    restarted["reason"] = "probability_preserves_current_position"
    restarted["position_before"] = 0
    restarted["desired_position"] = 0
    restarted["target_position"] = 0
    restarted["broker_position_before"] = 0
    restarted["broker_position_after_inspection"] = 0
    restarted["probability_up"] = 0.51
    restarted["model_prediction_available"] = True
    restarted["model_prediction_event_time_utc"] = restart_event.isoformat()
    restarted["model_unavailable_reason"] = None
    restarted["latest_completed_bar_time_utc"] = restart_event.isoformat()
    restarted["event_is_latest_feature"] = True
    restarted["event_is_latest_completed_bar"] = True
    restarted["stale_event_warning"] = False
    decisions = pd.concat(
        [decisions, pd.DataFrame([restarted])], ignore_index=True
    )
    decisions.to_csv(decisions_path, index=False)

    events_path = role_root / "order_events.csv"
    order_events = pd.read_csv(events_path)
    order_events.loc[1, "trigger_type"] = "CONTROL_CLEAN_STOP"
    order_events.loc[1, "decision_id"] = "CONTROL_CLEAN_STOP:model_a:run-1:3"
    order_events.to_csv(events_path, index=False)

    ledger_path = role_root / "completed_broker_events.csv"
    ledger = pd.read_csv(ledger_path)
    ledger_row = ledger.iloc[-1].copy()
    ledger_row["broker_event_key"] = completed_broker_event_identifier(
        "model_a", restart_event.isoformat()
    )
    ledger_row["event_time_utc"] = restart_event.isoformat()
    ledger_row["first_observed_utc"] = (
        restart_event + pd.Timedelta(minutes=15)
    ).isoformat()
    ledger_row["run_id"] = "run-2"
    ledger_row["iteration"] = 1
    ledger = pd.concat([ledger, pd.DataFrame([ledger_row])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    telemetry_path = role_root / "telemetry.csv"
    telemetry = pd.read_csv(telemetry_path)
    telemetry_row = telemetry.iloc[-1].copy()
    telemetry_row["snapshot_id"] = "model_a:run-2:final"
    telemetry_row["snapshot_utc"] = (
        restart_decision_time + pd.Timedelta(seconds=1)
    ).isoformat()
    telemetry_row["snapshot_phase"] = "FINAL"
    telemetry_row["run_id"] = "run-2"
    telemetry_row["latest_completed_event_time_utc"] = restart_event.isoformat()
    telemetry_row["latest_decision_event_time_utc"] = restart_event.isoformat()
    telemetry = pd.concat(
        [telemetry, pd.DataFrame([telemetry_row])], ignore_index=True
    )
    telemetry.to_csv(telemetry_path, index=False)

    runtime_path = role_root / "runtime_events.csv"
    runtime_events = pd.read_csv(runtime_path)
    runtime_events = pd.concat(
        [
            runtime_events,
            pd.DataFrame(
                [
                    {
                        "schema_version": "1.0",
                        "runtime_event_id": "run-2-start",
                        "timestamp_utc": (
                            restart_event + pd.Timedelta(seconds=5)
                        ).isoformat(),
                        "role": "model_a",
                        "run_id": "run-2",
                        "event_type": "WORKER_STARTED",
                    },
                    {
                        "schema_version": "1.0",
                        "runtime_event_id": "run-2-stop",
                        "timestamp_utc": (
                            restart_decision_time + pd.Timedelta(seconds=1)
                        ).isoformat(),
                        "role": "model_a",
                        "run_id": "run-2",
                        "event_type": "WORKER_STOPPED",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    runtime_events.to_csv(runtime_path, index=False)

    report_path = role_root / "final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["run_id"] = "run-2"
    report["started_utc"] = (
        restart_event + pd.Timedelta(seconds=5)
    ).isoformat()
    report["completed_utc"] = (
        restart_decision_time + pd.Timedelta(seconds=1)
    ).isoformat()
    report["state"]["records_written"] = 3
    report_path.write_text(json.dumps(report), encoding="utf-8")

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=900,
    )

    assert "virtual_position_discontinuity=1" not in (
        summary.decision_evidence_issue_reasons
    )
    assert "broker_position_discontinuity=1" not in (
        summary.decision_evidence_issue_reasons
    )
    assert summary.operational_acceptance_status == "LIMITED_RECOVERED"


def test_model_availability_is_reported_separately_from_disposition_coverage(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    role_root = runtime / "model_a"
    output = tmp_path / "report"
    write_role(runtime, "model_a", [10000.0, 10000.0])

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions["model_prediction_available"] = [True, False]
    decisions["probability_up"] = [0.51, None]
    decisions["model_prediction_event_time_utc"] = [
        decisions.loc[0, "event_time_utc"],
        decisions.loc[0, "event_time_utc"],
    ]
    decisions["latest_completed_bar_time_utc"] = [
        decisions.loc[0, "event_time_utc"],
        decisions.loc[1, "event_time_utc"],
    ]
    decisions["model_unavailable_reason"] = [
        None,
        "frozen_48_bar_contiguous_sequence_unavailable",
    ]
    decisions["event_is_latest_feature"] = [True, False]
    decisions["event_is_latest_completed_bar"] = [True, False]
    decisions["stale_event_warning"] = [False, True]
    decisions["action"] = [
        "HOLD_FLAT",
        "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP",
    ]
    decisions["broker_event_disposition"] = decisions["action"]
    decisions["reason"] = [
        "probability_preserves_current_position",
        "frozen_48_bar_contiguous_sequence_unavailable",
    ]
    decisions["position_before"] = [0, 0]
    decisions["desired_position"] = [0, 0]
    decisions["target_position"] = [0, 0]
    decisions["broker_position_before"] = [0, 0]
    decisions["broker_position_after_inspection"] = [0, 0]
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is True
    assert summary.broker_event_disposition_coverage_ratio == 1.0
    assert summary.model_prediction_count == 1
    assert summary.model_unavailable_event_count == 1
    assert summary.model_prediction_coverage_ratio == 0.5
    assert summary.model_availability_status == "LIMITED"
    assert summary.contiguity_warmup_event_count == 1
    assert summary.model_unavailable_exposure_after_disposition_count == 0


def test_acceptance_style_78_events_report_31_predictions_and_47_warmups(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    role_root.mkdir(parents=True)

    pre_gap_events = pd.date_range(
        "2026-07-23T16:00:00Z",
        periods=31,
        freq="15min",
        tz="UTC",
    )
    post_gap_events = pd.date_range(
        pre_gap_events[-1] + pd.Timedelta(minutes=75),
        periods=47,
        freq="15min",
        tz="UTC",
    )
    events = pre_gap_events.append(post_gap_events)
    snapshots = events + pd.Timedelta(minutes=15, seconds=1)
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * 78,
            "snapshot_id": [f"s:{index}" for index in range(78)],
            "snapshot_utc": snapshots,
            "snapshot_phase": ["STARTUP"] + ["POLL"] * 76 + ["FINAL"],
            "role": ["model_a"] * 78,
            "run_id": ["run-1"] * 78,
            "worker_pid": [100] * 78,
            "terminal_connected": [True] * 78,
            "balance": [10000.0] * 78,
            "equity": [10000.0] * 78,
            "spread_points": [20.0] * 78,
            "broker_position": [0] * 78,
            "pending_order_count": [0] * 78,
            "reconciliation_status": [
                "UNINITIALISED",
                *["PASS_STATE_MATCHES_BROKER"] * 76,
                "CLEAN_STOP_FLAT_CONFIRMED",
            ],
            "latest_completed_event_time_utc": events,
            "latest_decision_event_time_utc": events,
        }
    ).to_csv(role_root / "telemetry.csv", index=False)

    prediction_available = [True] * 31 + [False] * 47
    decision_actions = (
        ["HOLD_FLAT"] * 31
        + ["CONTROL_GAP_BLOCK"]
        + ["MODEL_UNAVAILABLE_CONTIGUITY_WARMUP"] * 46
    )
    pd.DataFrame(
        {
            "schema_version": ["1.0"] * 78,
            "decision_id": [
                decision_identifier(
                    "model_a",
                    event.isoformat(),
                    run_id="run-1",
                    iteration=index,
                )
                for index, event in enumerate(events, start=1)
            ],
            "role": ["model_a"] * 78,
            "run_id": ["run-1"] * 78,
            "iteration": list(range(1, 79)),
            "event_time_utc": events,
            "decision_utc": snapshots,
            "execution_mode": ["shadow"] * 78,
            "model_prediction_available": prediction_available,
            "model_prediction_event_time_utc": [
                event if available else None
                for event, available in zip(
                    events, prediction_available, strict=True
                )
            ],
            "latest_completed_bar_time_utc": events,
            "model_unavailable_reason": [None] * 31
            + ["non_contiguous_completed_m15_broker_event"]
            + ["frozen_48_bar_contiguous_sequence_unavailable"] * 46,
            "event_is_latest_feature": [True] * 31 + [False] * 47,
            "event_is_latest_completed_bar": [True] * 31 + [False] * 47,
            "probability_up": [0.51] * 31 + [None] * 47,
            "stale_event_warning": [False] * 31 + [True] * 47,
            "broker_event_disposition": decision_actions,
            "action": decision_actions,
            "reason": ["probability_preserves_current_position"] * 31
            + ["non_contiguous_completed_m15_broker_event"]
            + ["frozen_48_bar_contiguous_sequence_unavailable"] * 46,
            "position_before": [0] * 78,
            "desired_position": [0] * 78,
            "target_position": [0] * 78,
            "duplicate_event": [False] * 78,
            "gap_from_previous_event": [False] * 31 + [True] + [False] * 46,
            "policy_cap_reached": [False] * 78,
            "entry_blocked_by_policy_cap": [False] * 78,
            "exit_allowed_when_capped": [False] * 78,
            "close_only_reversal": [False] * 78,
            "daily_stop_active": [False] * 78,
            "total_stop_active": [False] * 78,
            "kill_switch_active": [False] * 78,
            "reconciliation_status": ["PASS_SHADOW_BROKER_FLAT"] * 78,
            "broker_position_before": [0] * 78,
            "broker_position_after_inspection": [0] * 78,
            "order_check_called": [False] * 78,
            "order_check_passed": [False] * 78,
            "order_send_called": [False] * 78,
            "order_send_passed": [False] * 78,
            "broker_position_after": [None] * 78,
        }
    ).to_csv(role_root / "decisions.csv", index=False)
    write_broker_event_ledger(role_root, "model_a", events, run_id="run-1")
    pd.DataFrame(
        {
            "schema_version": ["1.0", "1.0"],
            "runtime_event_id": ["start", "stop"],
            "timestamp_utc": [snapshots[0], snapshots[-1]],
            "role": ["model_a", "model_a"],
            "run_id": ["run-1", "run-1"],
            "event_type": ["WORKER_STARTED", "WORKER_STOPPED"],
        }
    ).to_csv(role_root / "runtime_events.csv", index=False)

    (role_root / "final_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "role": "model_a",
                "run_id": "run-1",
                "status": "PASS",
                "formal_gate": True,
                "started_utc": snapshots[0].isoformat(),
                "completed_utc": snapshots[-1].isoformat(),
                "state": {
                    "records_written": 78,
                    "order_send_calls": 0,
                    "successful_order_sends": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=3600,
    )

    assert summary.formal_audit_gate is True
    assert summary.unique_completed_event_count == 78
    assert summary.broker_event_disposition_coverage_ratio == 1.0
    assert summary.model_prediction_count == 31
    assert summary.model_unavailable_event_count == 47
    assert summary.model_prediction_coverage_ratio == 31 / 78
    assert summary.model_availability_status == "LIMITED"
    assert summary.gap_decision_count == 1
    assert summary.contiguity_warmup_event_count == 46
    assert summary.model_unavailable_exposure_after_disposition_count == 0
    assert summary.maximum_gap_control_processing_delay_seconds == 1.0
    assert (
        summary.maximum_broker_event_to_model_prediction_lag_minutes
        == 765.0
    )


def test_nonstale_endpoint_mismatch_fails_when_prediction_was_suppressed(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_role(role_root.parent, "model_a", [10000.0, 10000.0, 10000.0])

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions["model_prediction_available"] = [False, True, True]
    decisions["model_prediction_event_time_utc"] = [
        "2026-07-24T00:15:00+00:00",
        decisions.loc[1, "event_time_utc"],
        decisions.loc[2, "event_time_utc"],
    ]
    decisions["latest_completed_bar_time_utc"] = [
        "2026-07-24T00:15:00+00:00",
        decisions.loc[1, "event_time_utc"],
        decisions.loc[2, "event_time_utc"],
    ]
    decisions["stale_event_warning"] = [True, False, False]
    decisions.loc[0, "action"] = "CONTROL_MODEL_SNAPSHOT_MISMATCH_BLOCK"
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "FAIL"
    assert summary.model_prediction_endpoint_mismatch_count == 1
    assert "model_prediction_endpoint_mismatches=1" in summary.audit_gate_failures


def test_missing_completed_broker_event_ledger_fails_gate(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)
    (role_root / "completed_broker_events.csv").unlink()

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.completed_broker_event_ledger_present is False
    assert "completed_broker_event_ledger_missing=1" in summary.audit_gate_failures


def test_historical_backfill_exposure_is_hard_failure(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions["model_prediction_available"] = [True, False]
    decisions["stale_event_warning"] = [False, True]
    decisions.loc[1, "action"] = "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL"
    decisions["target_position"] = [1, 1]
    decisions["broker_position_after_inspection"] = [1, 1]
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "FAIL"
    assert summary.historical_backfill_exposure_observed_count == 1
    assert (
        "historical_backfill_exposure_observed=1"
        in summary.audit_gate_failures
    )


def test_safe_historical_backfill_is_limited_recovered(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    output = tmp_path / "report"
    write_role(runtime, "model_a", [10000.0, 10000.0, 10000.0])
    role_root = runtime / "model_a"

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions["model_prediction_available"] = [True, False, True]
    decisions["probability_up"] = [0.75, None, 0.75]
    decisions["stale_event_warning"] = [False, True, False]
    decisions["action"] = [
        "HOLD_FLAT",
        "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL",
        "HOLD_FLAT",
    ]
    decisions["broker_event_disposition"] = decisions["action"]
    decisions["reason"] = [
        "probability_preserves_current_position",
        "completed_broker_event_discovered_after_worker_outage",
        "probability_preserves_current_position",
    ]
    decisions["model_unavailable_reason"] = [
        None,
        "completed_broker_event_discovered_after_worker_outage",
        None,
    ]
    decisions["event_is_latest_feature"] = decisions[
        "event_is_latest_feature"
    ].astype("object")
    decisions["event_is_latest_completed_bar"] = decisions[
        "event_is_latest_completed_bar"
    ].astype("object")
    decisions.loc[1, "model_prediction_event_time_utc"] = None
    decisions.loc[1, "latest_completed_bar_time_utc"] = None
    decisions.loc[1, "event_is_latest_feature"] = None
    decisions.loc[1, "event_is_latest_completed_bar"] = None
    decisions.loc[1, "decision_id"] = historical_backfill_identifier(
        "model_a", pd.Timestamp(decisions.loc[1, "event_time_utc"]).isoformat()
    )
    decisions["position_before"] = [0, 0, 0]
    decisions["desired_position"] = [0, 0, 0]
    decisions["target_position"] = [0, 0, 0]
    decisions["broker_position_before"] = [0, 0, 0]
    decisions["broker_position_after_inspection"] = [0, 0, 0]
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "LIMITED_RECOVERED"
    assert summary.historical_backfill_event_count == 1
    assert summary.historical_backfill_exposure_observed_count == 0
    assert "historical_backfill_event_count=1" in summary.limited_recovery_reasons


def test_material_telemetry_gap_is_limited_recovered(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    telemetry_path = role_root / "telemetry.csv"
    telemetry = pd.read_csv(telemetry_path)
    telemetry = telemetry.drop(index=range(2, 7)).reset_index(drop=True)
    telemetry.to_csv(telemetry_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "LIMITED_RECOVERED"
    assert summary.telemetry_gap_count_over_threshold == 1
    assert any(
        reason.startswith("material_telemetry_gap_count=1")
        for reason in summary.limited_recovery_reasons
    )


def test_runtime_snapshot_mismatch_prevents_clean_pass(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    runtime_events = pd.read_csv(role_root / "runtime_events.csv")
    mismatch = runtime_events.iloc[0].copy()
    mismatch["event_id"] = "mismatch:1"
    mismatch["event_type"] = "MODEL_SNAPSHOT_MISMATCH"
    mismatch["severity"] = "ERROR"
    runtime_events = pd.concat(
        [runtime_events, pd.DataFrame([mismatch])], ignore_index=True
    )
    runtime_events.to_csv(role_root / "runtime_events.csv", index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "LIMITED_RECOVERED"
    assert summary.model_snapshot_mismatch_runtime_event_count == 1
    assert (
        "model_snapshot_mismatch_runtime_event_count=1"
        in summary.limited_recovery_reasons
    )


def test_multiple_dispositions_for_one_broker_event_fail_gate(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    decisions = pd.read_csv(role_root / "decisions.csv")
    duplicate = decisions.iloc[0].copy()
    duplicate["decision_id"] = "different-id-same-event"
    decisions = pd.concat(
        [decisions, pd.DataFrame([duplicate])], ignore_index=True
    )
    decisions.to_csv(role_root / "decisions.csv", index=False)

    final_report_path = role_root / "final_report.json"
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    final_report["state"]["records_written"] = len(decisions)
    final_report_path.write_text(
        json.dumps(final_report), encoding="utf-8"
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "FAIL"
    assert summary.broker_event_with_multiple_dispositions_count == 1
    assert summary.maximum_dispositions_per_broker_event == 2
    assert (
        "unexpected_multiple_disposition_events=1"
        in summary.audit_gate_failures
    )


def test_current_prediction_lag_has_explicit_scope_alias(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert (
        summary.maximum_current_broker_event_to_model_prediction_lag_minutes
        == summary.maximum_broker_event_to_model_prediction_lag_minutes
    )


@pytest.mark.parametrize(
    ("column", "value", "expected_reason"),
    [
        ("schema_version", "2.0", "invalid_schema_version=1"),
        ("decision_id", "", "blank_decision_id=1"),
        ("role", "model_b", "role_mismatch=1"),
        ("run_id", "unknown-run", "run_id_not_in_telemetry=1"),
        ("event_time_utc", "not-a-timestamp", "invalid_event_time_utc=1"),
        ("decision_utc", "not-a-timestamp", "invalid_decision_utc=1"),
        (
            "model_prediction_event_time_utc",
            "not-a-timestamp",
            "invalid_optional_timestamp:model_prediction_event_time_utc=1",
        ),
        ("duplicate_event", True, "persisted_duplicate_event_not_false=1"),
    ],
)
def test_systematic_decision_evidence_provenance_failures_are_gated(
    tmp_path: Path,
    column: str,
    value: object,
    expected_reason: str,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions[column] = decisions.get(
        column, pd.Series(index=decisions.index, dtype="object")
    ).astype("object")
    decisions.loc[0, column] = value
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "FAIL"
    assert summary.invalid_decision_evidence_count >= 1
    assert expected_reason in summary.decision_evidence_issue_reasons
    assert (output / "model_a_audit_gate.json").exists()
    assert (output / "model_a_daily_summary.csv").exists()


def test_enter_short_with_long_target_is_rejected(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[0, "action"] = "ENTER_SHORT"
    decisions.loc[0, "broker_event_disposition"] = "ENTER_SHORT"
    decisions.loc[0, "desired_position"] = -1
    decisions.loc[0, "target_position"] = 1
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert "invalid_action_position_contract=1" in summary.decision_evidence_issue_reasons


def test_enter_long_when_already_long_is_rejected(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[0, "position_before"] = 1
    decisions.loc[0, "broker_position_before"] = 1
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert "invalid_action_position_contract=1" in summary.decision_evidence_issue_reasons


def test_model_b_short_evidence_is_rejected(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    role_root = runtime / "model_b"
    output = tmp_path / "report"
    write_role(runtime, "model_b", [10000.0, 10000.0])

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[0, "action"] = "HOLD_SHORT"
    decisions.loc[0, "broker_event_disposition"] = "HOLD_SHORT"
    decisions.loc[0, "position_before"] = -1
    decisions.loc[0, "desired_position"] = -1
    decisions.loc[0, "target_position"] = -1
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_b",
        output_root=output,
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert "model_b_short_exposure=1" in summary.decision_evidence_issue_reasons


def test_shadow_block_spread_is_rejected(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    role_root = runtime / "model_a"
    output = tmp_path / "report"
    write_role(runtime, "model_a", [10000.0, 10000.0])

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[1, "action"] = "BLOCK_SPREAD"
    decisions.loc[1, "broker_event_disposition"] = "BLOCK_SPREAD"
    decisions.loc[1, "desired_position"] = -1
    decisions.loc[1, "target_position"] = 1
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert "block_spread_not_live=1" in summary.decision_evidence_issue_reasons


def _convert_exit_to_daily_stop(
    role_root: Path,
    *,
    active: bool,
    successful_execution: bool,
) -> None:
    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[1, "action"] = "DAILY_STOP_FLATTEN"
    decisions.loc[1, "broker_event_disposition"] = "DAILY_STOP_FLATTEN"
    decisions.loc[1, "daily_stop_active"] = active
    if not successful_execution:
        decisions.loc[1, "order_check_called"] = False
        decisions.loc[1, "order_check_passed"] = False
        decisions.loc[1, "order_send_called"] = False
        decisions.loc[1, "order_send_passed"] = False
        decisions.loc[1, "broker_position_after"] = None
    decisions.to_csv(decisions_path, index=False)

    events_path = role_root / "order_events.csv"
    events = pd.read_csv(events_path)
    events.loc[1, "trigger_type"] = "CONTROL_DAILY_STOP"
    events.loc[1, "decision_id"] = "CONTROL_DAILY_STOP:model_a:run-1:2"
    if not successful_execution:
        events = events.iloc[[0]].copy()
    events.to_csv(events_path, index=False)


def test_daily_stop_flatten_requires_active_control(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)
    _convert_exit_to_daily_stop(
        role_root,
        active=False,
        successful_execution=True,
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert "inactive_control_flag:daily_stop_active=1" in summary.decision_evidence_issue_reasons


def test_live_control_flatten_requires_successful_execution_evidence(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)
    _convert_exit_to_daily_stop(
        role_root,
        active=True,
        successful_execution=False,
    )

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert (
        "live_transition_missing_successful_order_flags=1"
        in summary.decision_evidence_issue_reasons
    )
    assert (
        "live_transition_missing_successful_order_event=1"
        in summary.decision_evidence_issue_reasons
    )


def test_reconciliation_incident_is_limited_recovered(tmp_path: Path) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    runtime_events_path = role_root / "runtime_events.csv"
    events = pd.read_csv(runtime_events_path)
    events = pd.concat(
        [
            events,
            pd.DataFrame(
                [
                    {
                        "schema_version": "1.0",
                        "runtime_event_id": "reconciliation-incident",
                        "timestamp_utc": "2026-07-24T00:00:45+00:00",
                        "role": "model_a",
                        "run_id": "run-1",
                        "event_type": "RECONCILIATION_INCIDENT",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    events.to_csv(runtime_events_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "LIMITED_RECOVERED"
    assert "reconciliation_incident_count=1" in summary.limited_recovery_reasons


def test_nonpass_reconciliation_snapshot_is_limited_recovered(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    telemetry_path = role_root / "telemetry.csv"
    telemetry = pd.read_csv(telemetry_path)
    telemetry.loc[1, "reconciliation_status"] = "BLOCK_BROKER_STATE_MISMATCH"
    telemetry.to_csv(telemetry_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
    )

    assert summary.formal_audit_gate is False
    assert summary.operational_acceptance_status == "LIMITED_RECOVERED"
    assert (
        "reconciliation_nonpass_snapshot_count=1"
        in summary.limited_recovery_reasons
    )



def test_consolidated_report_survives_invalid_decision_evidence(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    output = tmp_path / "report"
    write_role(runtime, "model_a", [10000.0, 10000.0])
    write_role(runtime, "model_b", [10000.0, 10000.0])

    decisions_path = runtime / "model_a" / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions["schema_version"] = decisions["schema_version"].astype("object")
    decisions.loc[0, "schema_version"] = "corrupt"
    decisions.to_csv(decisions_path, index=False)

    report = build_observation_report(
        runtime,
        output,
        expected_poll_seconds=86400,
    )

    assert report["formal_audit_gate"] is False
    statuses = {
        model["role"]: model["operational_acceptance_status"]
        for model in report["models"]
    }
    assert statuses == {"model_a": "FAIL", "model_b": "PASS"}
    assert (output / "consolidated_observation_report.json").exists()
    assert (output / "consolidated_model_summary.csv").exists()
    assert (output / "model_a_audit_gate.json").exists()
    assert (output / "model_b_audit_gate.json").exists()


def test_real_worker_noncalled_none_evidence_passes(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    role_root = runtime / "model_a"
    write_role(runtime, "model_a", [10000.0, 10000.0])

    decisions = pd.read_csv(role_root / "decisions.csv")
    assert decisions["order_check_passed"].isna().all()
    assert decisions["order_send_passed"].isna().all()

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is True


def test_restart_startup_may_restore_event_before_current_process_decision(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    role_root = runtime / "model_a"
    write_role(runtime, "model_a", [10000.0, 10000.0])
    telemetry_path = role_root / "telemetry.csv"
    telemetry = pd.read_csv(telemetry_path)
    telemetry.loc[0, "latest_decision_event_time_utc"] = None
    telemetry.to_csv(telemetry_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is True


def test_derived_broker_gap_requires_first_post_gap_control(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    role_root = runtime / "model_a"
    write_role(runtime, "model_a", [10000.0, 10000.0])
    post_gap = pd.Timestamp("2026-07-25T00:00:00Z")

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[1, "event_time_utc"] = post_gap.isoformat()
    decisions.loc[1, "model_prediction_event_time_utc"] = post_gap.isoformat()
    decisions.loc[1, "latest_completed_bar_time_utc"] = post_gap.isoformat()
    decisions.loc[1, "decision_utc"] = (
        post_gap + pd.Timedelta(minutes=15, seconds=1)
    ).isoformat()
    decisions.loc[1, "decision_id"] = decision_identifier(
        "model_a",
        post_gap.isoformat(),
        run_id="dual_model_a_test",
        iteration=2,
    )
    decisions.to_csv(decisions_path, index=False)

    ledger_path = role_root / "completed_broker_events.csv"
    ledger = pd.read_csv(ledger_path)
    ledger.loc[1, "event_time_utc"] = post_gap.isoformat()
    ledger.loc[1, "first_observed_utc"] = (
        post_gap + pd.Timedelta(minutes=15, seconds=1)
    ).isoformat()
    ledger.loc[1, "broker_event_key"] = completed_broker_event_identifier(
        "model_a", post_gap.isoformat()
    )
    ledger.to_csv(ledger_path, index=False)

    telemetry_path = role_root / "telemetry.csv"
    telemetry = pd.read_csv(telemetry_path)
    telemetry.loc[1, "snapshot_utc"] = (
        post_gap + pd.Timedelta(minutes=15, seconds=2)
    ).isoformat()
    telemetry.loc[1, "latest_completed_event_time_utc"] = post_gap.isoformat()
    telemetry.loc[1, "latest_decision_event_time_utc"] = post_gap.isoformat()
    telemetry.to_csv(telemetry_path, index=False)

    report_path = role_root / "final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["completed_utc"] = (
        post_gap + pd.Timedelta(minutes=15, seconds=2)
    ).isoformat()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert any(
        reason.startswith("broker_gap_missing_control:")
        for reason in summary.audit_gate_failures
    )


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing_prediction_column", "missing_required_column:probability_up=2"),
        ("probability_out_of_range", "invalid_available_probability=1"),
        ("decision_before_bar_close", "decision_before_m15_completion=1"),
        ("virtual_state_jump", "virtual_position_discontinuity=1"),
        ("ordinary_action_with_kill", "active_kill_switch_without_kill_action=1"),
        (
            "policy_block_without_flags",
            "invalid_policy_flag:policy_cap_reached=1",
        ),
        ("shadow_execution_position", "shadow_order_evidence_present_or_missing=1"),
    ],
)
def test_remaining_decision_fail_open_paths_are_gated(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
) -> None:
    runtime = tmp_path / "runtime"
    role_root = runtime / "model_a"
    write_role(runtime, "model_a", [10000.0, 10000.0])
    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)

    if mutation == "missing_prediction_column":
        decisions = decisions.drop(columns=["probability_up"])
    elif mutation == "probability_out_of_range":
        decisions.loc[0, "probability_up"] = 1.5
    elif mutation == "decision_before_bar_close":
        event = pd.Timestamp(decisions.loc[0, "event_time_utc"])
        decisions.loc[0, "decision_utc"] = (
            event + pd.Timedelta(minutes=5)
        ).isoformat()
    elif mutation == "virtual_state_jump":
        decisions.loc[1, "action"] = "HOLD_SHORT"
        decisions.loc[1, "broker_event_disposition"] = "HOLD_SHORT"
        decisions.loc[1, "position_before"] = -1
        decisions.loc[1, "desired_position"] = -1
        decisions.loc[1, "target_position"] = -1
    elif mutation == "ordinary_action_with_kill":
        decisions.loc[0, "kill_switch_active"] = True
    elif mutation == "policy_block_without_flags":
        decisions.loc[1, "action"] = "BLOCK_DAILY_POLICY_CAP"
        decisions.loc[1, "broker_event_disposition"] = "BLOCK_DAILY_POLICY_CAP"
        decisions.loc[1, "position_before"] = 1
        decisions.loc[1, "desired_position"] = -1
        decisions.loc[1, "target_position"] = 1
        decisions.loc[1, "policy_cap_reached"] = False
        decisions.loc[1, "entry_blocked_by_policy_cap"] = False
    elif mutation == "shadow_execution_position":
        decisions.loc[0, "broker_position_after"] = 1
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert expected_reason in summary.decision_evidence_issue_reasons


@pytest.mark.parametrize(
    ("source", "expected_failure"),
    [
        ("telemetry_lag", "telemetry_event_lag_columns_missing"),
        ("ledger_role", "completed_broker_events.csv:role_mismatch=2"),
        ("final_role", "final_report.json:role_mismatch"),
    ],
)
def test_cross_file_identity_and_lag_fail_closed(
    tmp_path: Path,
    source: str,
    expected_failure: str,
) -> None:
    runtime = tmp_path / "runtime"
    role_root = runtime / "model_a"
    write_role(runtime, "model_a", [10000.0, 10000.0])

    if source == "telemetry_lag":
        path = role_root / "telemetry.csv"
        telemetry = pd.read_csv(path).drop(
            columns=[
                "latest_completed_event_time_utc",
                "latest_decision_event_time_utc",
            ]
        )
        telemetry.to_csv(path, index=False)
    elif source == "ledger_role":
        path = role_root / "completed_broker_events.csv"
        ledger = pd.read_csv(path)
        ledger["role"] = "model_b"
        ledger.to_csv(path, index=False)
    else:
        path = role_root / "final_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["role"] = "model_b"
        path.write_text(json.dumps(report), encoding="utf-8")

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=tmp_path / "report",
        expected_poll_seconds=86400,
    )

    assert summary.formal_audit_gate is False
    assert expected_failure in summary.audit_gate_failures
