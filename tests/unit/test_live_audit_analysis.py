from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from capstone_trading.runtime.live_audit_analysis import (
    analyse_role,
    build_observation_report,
)


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
        }
    ).to_csv(role_root / "telemetry.csv", index=False)
    pd.DataFrame(
        {
            "decision_id": [f"{role}:decision:{index}" for index in range(len(equity))],
            "event_time_utc": times,
            "action": ["ENTER_LONG"] + ["HOLD_LONG"] * (len(equity) - 1),
        }
    ).to_csv(role_root / "decisions.csv", index=False)
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
        }
    ).to_csv(role_root / "telemetry.csv", index=False)
    pd.DataFrame(
        {
            "decision_id": ["d-entry", "d-exit"],
            "event_time_utc": [times[0], times[-1]],
            "action": ["ENTER_LONG", "EXIT_POSITION"],
        }
    ).to_csv(role_root / "decisions.csv", index=False)
    pd.DataFrame(
        {
            "execution_id": ["e-entry", "e-exit"],
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
            "time_done_utc": [times[-1]] * len(order_tickets),
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
