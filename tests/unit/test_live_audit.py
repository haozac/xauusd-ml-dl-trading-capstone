from __future__ import annotations

from pathlib import Path

import pytest

from capstone_trading.runtime.live_audit import (
    LiveAuditError,
    append_csv_row,
    append_unique_rows,
)


def test_append_csv_row_writes_header_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "audit.csv"
    fields = ("id", "value")
    append_csv_row(path, {"id": "a", "value": 1}, fieldnames=fields)
    append_csv_row(path, {"id": "b", "value": 2}, fieldnames=fields)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["id,value", "a,1", "b,2"]


def test_append_csv_row_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "audit.csv"
    append_csv_row(path, {"id": "a"}, fieldnames=("id",))
    with pytest.raises(LiveAuditError, match="schema mismatch"):
        append_csv_row(path, {"id": "b", "value": 2}, fieldnames=("id", "value"))


def test_append_unique_rows_is_restart_safe(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    fields = ("history_key", "value")
    rows = [
        {"history_key": "deal:1", "value": 10},
        {"history_key": "deal:2", "value": 20},
    ]
    assert append_unique_rows(
        path,
        rows,
        fieldnames=fields,
        key_field="history_key",
    ) == 2
    assert append_unique_rows(
        path,
        rows,
        fieldnames=fields,
        key_field="history_key",
    ) == 0


def test_telemetry_row_uses_broker_position_and_observed_spread() -> None:
    from types import SimpleNamespace

    from capstone_trading.runtime.dual_live_state import (
        BrokerPositionSnapshot,
        BrokerSnapshot,
    )
    from capstone_trading.runtime.live_audit import telemetry_audit_row

    snapshot = BrokerSnapshot(
        account_login_masked="****4309",
        account_equity=9999.0,
        account_balance=10000.0,
        symbol="XAUUSD",
        positions=(
            BrokerPositionSnapshot(
                position=1,
                ticket=50,
                identifier=60,
                order_ticket=40,
                volume=0.01,
                magic=26070101,
                symbol="XAUUSD",
            ),
        ),
        pending_order_count=0,
        connected=True,
        terminal_trade_allowed=True,
        account_trade_allowed=True,
        account_expert_allowed=True,
        trade_api_disabled=False,
    )
    inspection = SimpleNamespace(
        snapshot=snapshot,
        package={"version": "test"},
        terminal={"trade_allowed": True},
        account={
            "currency": "SGD",
            "leverage": 100,
            "company": "Broker",
            "server": "Demo",
            "balance": 10000.0,
            "equity": 9999.0,
            "profit": -1.0,
            "margin": 10.0,
            "margin_free": 9989.0,
            "margin_level": 99990.0,
        },
        symbol_info={"spread": 999, "point": 0.01},
        tick={
            "time": 1_700_000_000,
            "time_msc": 1_700_000_000_000,
            "bid": 2400.0,
            "ask": 2400.2,
            "last": 2400.1,
        },
        positions_raw=(
            {
                "ticket": 50,
                "identifier": 60,
                "order": 40,
                "type": 0,
                "volume": 0.01,
                "time": 1_700_000_000,
                "time_msc": 1_700_000_000_000,
                "price_open": 2399.0,
                "price_current": 2400.0,
                "profit": 1.0,
                "swap": 0.0,
                "magic": 26070101,
                "comment": "CP_DUAL_A",
            },
        ),
        pending_orders_raw=(),
        capital_review={"capstone_10x_leverage_cap_passed": True},
        mt5_calls=("account_info",),
        forbidden_attempts=(),
        shutdown_called=True,
    )
    state = SimpleNamespace(
        broker_position=0,
        virtual_position=1,
        hold_bars=2,
        flat_bars_since_exit=0,
        policy_changes_today=1,
        successful_entries_today=1,
        daily_return=-0.0001,
        total_drawdown=-0.0002,
        daily_stop_active=False,
        total_stop_active=False,
        kill_switch_active=False,
        reconciliation_status="PASS_STATE_MATCHES_BROKER",
        reconciliation_incidents=0,
    )
    decision = SimpleNamespace(
        event_time_utc="2026-07-24T00:00:00+00:00",
        action="ENTER_LONG",
        reason="frozen_overlay_transition",
    )

    row = telemetry_audit_row(
        role="model_a",
        run_id="run-1",
        iteration=1,
        worker_pid=123,
        snapshot_phase="STARTUP",
        execution_mode="live",
        orders_enabled=True,
        latest_completed_event_time_utc="2026-07-24T00:00:00+00:00",
        latest_decision=decision,
        state=state,
        broker_inspection=inspection,
        stop_file_exists=False,
        kill_switch_file_exists=False,
    )

    assert row["snapshot_phase"] == "STARTUP"
    assert row["full_context_captured"] is True
    assert row["broker_position"] == 1
    assert row["spread_points"] == pytest.approx(20.0)
    assert row["symbol_reported_spread_points"] == 999
    assert row["position_open_price"] == 2399.0
    assert row["account_json"]["currency"] == "SGD"
    assert row["positions_json"][0]["ticket"] == 50

    compact = telemetry_audit_row(
        role="model_a",
        run_id="run-1",
        iteration=2,
        worker_pid=123,
        snapshot_phase="POLL",
        execution_mode="live",
        orders_enabled=True,
        latest_completed_event_time_utc="2026-07-24T00:00:00+00:00",
        latest_decision=decision,
        state=state,
        broker_inspection=inspection,
        stop_file_exists=False,
        kill_switch_file_exists=False,
    )
    assert compact["full_context_captured"] is False
    assert compact["account_json"] is None
    assert compact["symbol_info_json"] is None
    assert compact["tick_json"]["bid"] == 2400.0
    assert compact["positions_json"][0]["ticket"] == 50


def test_decision_row_preserves_risk_price_and_rule_context() -> None:
    from types import SimpleNamespace

    from capstone_trading.runtime.dual_live_state import BrokerSnapshot
    from capstone_trading.runtime.live_audit import (
        DECISION_FIELDS,
        decision_audit_row,
    )

    snapshot = BrokerSnapshot(
        account_login_masked="****4309",
        account_equity=10000.0,
        account_balance=10000.0,
        symbol="XAUUSD",
        positions=(),
        pending_order_count=0,
        connected=True,
        terminal_trade_allowed=True,
        account_trade_allowed=True,
        account_expert_allowed=True,
        trade_api_disabled=False,
    )
    inspection = SimpleNamespace(
        snapshot=snapshot,
        package={},
        terminal={},
        account={"balance": 10000.0, "equity": 10000.0, "profit": 0.0},
        symbol_info={"point": 0.01, "spread": 30},
        tick={"bid": 2400.0, "ask": 2400.2},
        positions_raw=(),
        pending_orders_raw=(),
        capital_review={},
        mt5_calls=(),
        forbidden_attempts=(),
        shutdown_called=True,
    )
    state_before = SimpleNamespace(
        policy_changes_today=3,
        successful_entries_today=2,
        hold_bars=4,
        flat_bars_since_exit=0,
        current_utc_date="2026-07-24",
        day_start_equity=10000.0,
        peak_equity=10020.0,
        daily_return=-0.001,
        total_drawdown=-0.002,
        daily_stop_active=False,
        total_stop_active=False,
        kill_switch_active=False,
        reconciliation_status="PASS_STATE_MATCHES_BROKER",
        reconciliation_incidents=0,
    )
    state_after = SimpleNamespace(
        **{
            **state_before.__dict__,
            "policy_changes_today": 4,
            "hold_bars": 0,
            "daily_return": -0.0015,
            "total_drawdown": -0.0025,
        }
    )
    decision = SimpleNamespace(
        role="model_a",
        run_id="run-1",
        iteration=5,
        event_time_utc="2026-07-24T01:00:00+00:00",
        decision_utc="2026-07-24T01:00:02+00:00",
        execution_mode="live",
        probability_up=0.50,
        position_before=1,
        desired_position=0,
        target_position=0,
        action="EXIT_POSITION_CAP_REACHED",
        reason="daily_policy_cap_cannot_block_risk_reducing_exit",
        duplicate_event=False,
        gap_from_previous_event=False,
        stale_event_warning=False,
        requested_policy_event_units=1,
        policy_event_units=1,
        policy_changes_today_before=3,
        successful_entries_today_before=2,
        policy_cap_reached=True,
        entry_blocked_by_policy_cap=False,
        exit_allowed_when_capped=True,
        close_only_reversal=False,
        daily_stop_active=False,
        total_stop_active=False,
        kill_switch_active=False,
        reconciliation_status="PASS_STATE_MATCHES_BROKER",
        order_check_called=True,
        order_check_passed=True,
        order_send_called=True,
        order_send_passed=True,
        broker_position_after=0,
        broker_position_ticket_after=None,
    )

    row = decision_audit_row(
        decision=decision,
        state_before=state_before,
        state_after=state_after,
        bar_context={
            "m15_open": 2399.0,
            "m15_high": 2401.0,
            "m15_low": 2398.0,
            "m15_close": 2400.0,
            "m15_tick_volume": 100,
        },
        signal_context={"sequence_length": 96, "feature_count": 51},
        snapshot_context={
            "strategy_rules": {"max_policy_changes_per_utc_day": 3},
            "risk_rules": {"daily_loss_stop_simple_return": -0.02},
        },
        broker_before=inspection,
        broker_after=inspection,
    )

    assert set(row) == set(DECISION_FIELDS)
    assert row["m15_close"] == 2400.0
    assert row["spread_points_at_decision"] == pytest.approx(20.0)
    assert row["policy_changes_today_after"] == 4
    assert row["daily_return_before"] == -0.001
    assert row["daily_return_after"] == -0.0015
    assert row["strategy_rules_json"]["max_policy_changes_per_utc_day"] == 3
    assert row["broker_before_json"]["snapshot"]["symbol"] == "XAUUSD"

def test_runtime_event_row_uses_persisted_broker_position() -> None:
    from types import SimpleNamespace

    from capstone_trading.runtime.live_audit import (
        RUNTIME_EVENT_FIELDS,
        runtime_event_audit_row,
    )

    state = SimpleNamespace(
        virtual_position=1,
        broker_position=1,
        reconciliation_status="PASS_STATE_MATCHES_BROKER",
        restart_count=0,
    )
    row = runtime_event_audit_row(
        role="model_a",
        run_id="run-1",
        iteration=3,
        worker_pid=123,
        event_type="ORDER_EXECUTION",
        event_reason="ENTER_LONG",
        severity="INFO",
        state=state,
    )

    assert set(row) == set(RUNTIME_EVENT_FIELDS)
    assert row["virtual_position"] == 1
    assert row["broker_position"] == 1

