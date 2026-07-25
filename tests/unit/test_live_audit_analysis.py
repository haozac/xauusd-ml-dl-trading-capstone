from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from capstone_trading.runtime.live_audit_analysis import (
    analyse_role,
    build_observation_report,
)


def write_broker_event_ledger(
    role_root: Path,
    role: str,
    events: list[object] | pd.DatetimeIndex,
) -> None:
    parsed = pd.to_datetime(list(events), utc=True)
    pd.DataFrame(
        {
            "broker_event_key": [
                f"BROKER_EVENT:{role}:{item.isoformat()}"
                for item in parsed
            ],
            "role": [role] * len(parsed),
            "event_time_utc": parsed,
            "first_observed_utc": parsed,
            "observation_type": ["CURRENT_COMPLETED_EVENT"] * len(parsed),
            "run_id": [f"dual_{role}_test"] * len(parsed),
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
    times = pd.date_range("2026-07-24", periods=len(equity), freq="1D", tz="UTC")
    phases = ["STARTUP"] + ["POLL"] * max(0, len(equity) - 2) + ["FINAL"]
    pd.DataFrame(
        {
            "snapshot_id": [f"{role}:{index}" for index in range(len(equity))],
            "snapshot_utc": times,
            "snapshot_phase": phases,
            "run_id": [f"dual_{role}_test"] * len(equity),
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
    pd.DataFrame(
        {
            "decision_id": [f"{role}:decision:{index}" for index in range(len(equity))],
            "event_time_utc": times,
            "action": ["ENTER_LONG"] + ["HOLD_LONG"] * (len(equity) - 1),
        }
    ).to_csv(role_root / "decisions.csv", index=False)
    write_broker_event_ledger(role_root, role, times)
    (role_root / "final_report.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "formal_gate": True,
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
    times = pd.to_datetime(
        [
            "2026-07-24T00:00:00Z",
            "2026-07-24T00:00:30Z",
            "2026-07-24T00:01:00Z",
        ],
        utc=True,
    )
    pd.DataFrame(
        {
            "snapshot_id": [
                "model_a:startup",
                "model_a:poll",
                "model_a:final",
            ],
            "snapshot_utc": times,
            "snapshot_phase": ["STARTUP", "POLL", "FINAL"],
            "run_id": ["run-1", "run-1", "run-1"],
            "worker_pid": [100, 100, 100],
            "terminal_connected": [True, True, True],
            "balance": [10000.0, 10000.0, 10009.0],
            "equity": [10000.0, 10005.0, 10009.0],
            "spread_points": [20.0, 21.0, 22.0],
            "broker_position": [0, 1, 0],
            "pending_order_count": [0, 0, 0],
            "reconciliation_status": [
                "UNINITIALISED",
                "PASS_STATE_MATCHES_BROKER",
                "CLEAN_STOP_FLAT_CONFIRMED",
            ],
            "latest_completed_event_time_utc": [times[0], times[0], times[-1]],
            "latest_decision_event_time_utc": [times[0], times[0], times[-1]],
        }
    ).to_csv(role_root / "telemetry.csv", index=False)
    pd.DataFrame(
        {
            "decision_id": ["d-entry", "d-exit"],
            "event_time_utc": [times[0], times[-1]],
            "action": ["ENTER_LONG", "EXIT_POSITION"],
        }
    ).to_csv(role_root / "decisions.csv", index=False)
    write_broker_event_ledger(
        role_root, "model_a", [times[0], times[-1]]
    )
    pd.DataFrame(
        {
            "execution_id": ["e-entry", "e-exit"],
            "decision_id": ["d-entry", "d-exit"],
            "trigger_type": ["STRATEGY_DECISION", "STRATEGY_DECISION"],
            "completed_utc": [times[0], times[-1]],
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
            "history_key": ["deal:201", "deal:202"],
            "ticket": [201, 202],
            "order": [101, 102],
            "position_id": [500, 500],
            "time_utc": [times[0], times[-1]],
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
            "history_key": [f"order:{ticket}" for ticket in order_tickets],
            "ticket": order_tickets,
            "time_done_utc": [times[0], times[-1]]
            + ([times[-1]] if include_unlinked_order else []),
        }
    ).to_csv(role_root / "broker_orders.csv", index=False)
    pd.DataFrame(
        {
            "runtime_event_id": ["start", "stop"],
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


def test_model_availability_is_reported_separately_from_disposition_coverage(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "runtime" / "model_a"
    output = tmp_path / "report"
    write_completed_trade(role_root)

    decisions_path = role_root / "decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions["model_prediction_available"] = [True, False]
    decisions["model_prediction_event_time_utc"] = [
        decisions.loc[0, "event_time_utc"],
        decisions.loc[0, "event_time_utc"],
    ]
    decisions["latest_completed_bar_time_utc"] = [
        decisions.loc[0, "event_time_utc"],
        decisions.loc[1, "event_time_utc"],
    ]
    decisions["stale_event_warning"] = [False, True]
    decisions["action"] = [
        "ENTER_LONG",
        "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP",
    ]
    decisions["target_position"] = [1, 0]
    decisions["broker_position_after_inspection"] = [1, 0]
    decisions["decision_utc"] = [
        "2026-07-24T00:15:01+00:00",
        "2026-07-24T00:16:01+00:00",
    ]
    decisions.to_csv(decisions_path, index=False)

    summary = analyse_role(
        role_root,
        role="model_a",
        output_root=output,
        expected_poll_seconds=30,
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

    events = pd.date_range(
        "2026-07-23T16:00:00Z",
        periods=78,
        freq="15min",
        tz="UTC",
    )
    snapshots = events + pd.Timedelta(minutes=15, seconds=1)
    pd.DataFrame(
        {
            "snapshot_id": [f"s:{index}" for index in range(78)],
            "snapshot_utc": snapshots,
            "snapshot_phase": ["STARTUP"] + ["POLL"] * 76 + ["FINAL"],
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
    pd.DataFrame(
        {
            "decision_id": [f"d:{index}" for index in range(78)],
            "event_time_utc": events,
            "decision_utc": snapshots,
            "model_prediction_available": prediction_available,
            "model_prediction_event_time_utc": [
                event if available else None
                for event, available in zip(
                    events, prediction_available, strict=True
                )
            ],
            "latest_completed_bar_time_utc": events,
            "probability_up": [0.51] * 31 + [None] * 47,
            "stale_event_warning": [False] * 31 + [True] * 47,
            "action": (
                ["HOLD_FLAT"] * 31
                + ["CONTROL_GAP_BLOCK"]
                + ["MODEL_UNAVAILABLE_CONTIGUITY_WARMUP"] * 46
            ),
            "target_position": [0] * 78,
            "broker_position_after_inspection": [0] * 78,
        }
    ).to_csv(role_root / "decisions.csv", index=False)
    write_broker_event_ledger(role_root, "model_a", events)

    (role_root / "final_report.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "formal_gate": True,
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
        expected_poll_seconds=900,
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
        == 47 * 15
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
    decisions["stale_event_warning"] = [False, True, False]
    decisions.loc[1, "action"] = "MODEL_UNAVAILABLE_HISTORICAL_BACKFILL"
    decisions["target_position"] = [0, 0, 0]
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
    telemetry["snapshot_utc"] = [
        "2026-07-24T00:00:00+00:00",
        "2026-07-24T00:00:30+00:00",
        "2026-07-24T00:03:00+00:00",
    ]
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
        "broker_events_with_multiple_dispositions=1"
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
