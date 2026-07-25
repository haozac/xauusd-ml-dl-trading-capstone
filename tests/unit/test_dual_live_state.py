from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


from capstone_trading.runtime.dual_live_state import (
    BrokerPositionSnapshot,
    BrokerSnapshot,
    RiskRules,
    StrategyRules,
    apply_transition,
    decide_strategy_transition,
    initial_state,
    reconcile_state,
    session_gap_lockout_status,
    update_risk_state,
    write_state,
    load_state,
)


def model_a_rules() -> StrategyRules:
    return StrategyRules(
        role="model_a",
        long_threshold=0.53,
        short_threshold=0.47,
        exit_threshold=None,
        minimum_hold_bars=3,
        max_policy_changes_per_utc_day=3,
        max_successful_entries_per_utc_day=None,
        long_only=False,
        reversal_policy_event_units=1,
    )


def model_b_rules() -> StrategyRules:
    return StrategyRules(
        role="model_b",
        long_threshold=0.55,
        short_threshold=None,
        exit_threshold=0.50,
        minimum_hold_bars=0,
        max_policy_changes_per_utc_day=None,
        max_successful_entries_per_utc_day=1,
        long_only=True,
    )


def broker(*, positions=(), pending=0, equity=10_000.0) -> BrokerSnapshot:
    return BrokerSnapshot(
        account_login_masked="*****0001",
        account_equity=equity,
        account_balance=equity,
        symbol="XAUUSD",
        positions=tuple(positions),
        pending_order_count=pending,
        connected=True,
        terminal_trade_allowed=True,
        account_trade_allowed=True,
        account_expert_allowed=True,
        trade_api_disabled=False,
    )


def test_model_b_one_successful_entry_per_utc_day() -> None:
    state = replace(
        initial_state("model_b", execution_mode="shadow"),
        current_utc_date="2026-07-21",
        successful_entries_today=1,
    )
    decision = decide_strategy_transition(
        state,
        rules=model_b_rules(),
        run_id="test",
        iteration=1,
        event_time_utc="2026-07-21T15:45:00+00:00",
        probability_up=0.60,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    assert decision.action == "BLOCK_DAILY_ENTRY_CAP"
    assert decision.target_position == 0


def test_model_b_exit_uses_hysteresis() -> None:
    state = replace(
        initial_state("model_b", execution_mode="shadow"),
        virtual_position=1,
        last_event_time_utc="2026-07-21T15:30:00+00:00",
    )
    hold = decide_strategy_transition(
        state,
        rules=model_b_rules(),
        run_id="test",
        iteration=1,
        event_time_utc="2026-07-21T15:45:00+00:00",
        probability_up=0.50,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    exit_decision = decide_strategy_transition(
        state,
        rules=model_b_rules(),
        run_id="test",
        iteration=2,
        event_time_utc="2026-07-21T15:45:00+00:00",
        probability_up=0.499,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    assert hold.target_position == 1
    assert exit_decision.target_position == 0
    assert exit_decision.action == "EXIT_POSITION"


def test_model_a_minimum_hold_matches_frozen_replay_semantics() -> None:
    rules = model_a_rules()
    state = replace(
        initial_state("model_a", execution_mode="shadow"),
        virtual_position=1,
        hold_bars=3,
        last_event_time_utc="2026-07-21T15:30:00+00:00",
    )
    blocked = decide_strategy_transition(
        state,
        rules=rules,
        run_id="test",
        iteration=1,
        event_time_utc="2026-07-21T15:45:00+00:00",
        probability_up=0.40,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    eligible = decide_strategy_transition(
        replace(state, hold_bars=4),
        rules=rules,
        run_id="test",
        iteration=2,
        event_time_utc="2026-07-21T15:45:00+00:00",
        probability_up=0.40,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    assert blocked.action == "BLOCK_MINIMUM_HOLD"
    assert blocked.target_position == 1
    assert eligible.action == "REVERSE_LONG_TO_SHORT"
    assert eligible.target_position == -1


def test_gap_forces_flat_and_gap_exit_does_not_start_cooldown() -> None:
    state = replace(
        initial_state("model_a", execution_mode="shadow"),
        virtual_position=1,
        hold_bars=8,
        last_event_time_utc="2026-07-21T15:00:00+00:00",
    )
    decision = decide_strategy_transition(
        state,
        rules=model_a_rules(),
        run_id="test",
        iteration=1,
        event_time_utc="2026-07-21T15:30:00+00:00",
        probability_up=0.70,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    after = apply_transition(
        state,
        decision,
        confirmed_position=0,
        broker_ticket=None,
    )
    assert decision.action == "CONTROL_GAP_FLATTEN"
    assert after.virtual_position == 0
    assert after.flat_bars_since_exit == 1_000_000_000


def test_duplicate_event_is_never_reapplied() -> None:
    state = replace(
        initial_state("model_b", execution_mode="shadow"),
        last_event_time_utc="2026-07-21T15:45:00+00:00",
        records_written=7,
    )
    decision = decide_strategy_transition(
        state,
        rules=model_b_rules(),
        run_id="test",
        iteration=2,
        event_time_utc="2026-07-21T15:45:00+00:00",
        probability_up=0.80,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    after = apply_transition(
        state,
        decision,
        confirmed_position=state.virtual_position,
        broker_ticket=None,
    )
    assert decision.action == "DUPLICATE_SKIP"
    assert after.records_written == 7


def test_shadow_reconciliation_blocks_real_exposure() -> None:
    state = initial_state("model_a", execution_mode="shadow")
    position = BrokerPositionSnapshot(
        position=1,
        ticket=10,
        identifier=20,
        order_ticket=30,
        volume=0.01,
        magic=26070101,
        symbol="XAUUSD",
    )
    result = reconcile_state(
        state,
        broker(positions=(position,)),
        expected_magic=26070101,
        execution_mode="shadow",
    )
    assert result.blocked is True
    assert result.status == "BLOCK_EXPOSURE_IN_SHADOW"


def test_live_reconciliation_adopts_broker_position_after_restart() -> None:
    state = replace(
        initial_state("model_a", execution_mode="live"),
        virtual_position=0,
    )
    position = BrokerPositionSnapshot(
        position=-1,
        ticket=10,
        identifier=20,
        order_ticket=30,
        volume=0.01,
        magic=26070101,
        symbol="XAUUSD",
    )
    result = reconcile_state(
        state,
        broker(positions=(position,)),
        expected_magic=26070101,
        execution_mode="live",
    )
    assert result.blocked is False
    assert result.incident is True
    assert result.state.virtual_position == -1
    assert result.state.reconciliation_status == "PASS_BROKER_STATE_ADOPTED"


def test_live_reconciliation_external_flat_starts_cooldown() -> None:
    state = replace(
        initial_state("model_a", execution_mode="live"),
        virtual_position=1,
        hold_bars=10,
        flat_bars_since_exit=1_000_000_000,
    )
    result = reconcile_state(
        state,
        broker(),
        expected_magic=26070101,
        execution_mode="live",
    )
    assert result.state.virtual_position == 0
    assert result.state.flat_bars_since_exit == 0


def test_risk_stops_use_live_account_equity() -> None:
    state = initial_state("model_a", execution_mode="live")
    first = update_risk_state(
        state,
        equity=10_000,
        event_time_utc="2026-07-21T00:15:00+00:00",
        rules=RiskRules(),
        kill_switch_active=False,
    ).state
    loss = update_risk_state(
        first,
        equity=9_790,
        event_time_utc="2026-07-21T00:30:00+00:00",
        rules=RiskRules(),
        kill_switch_active=False,
    )
    assert loss.state.daily_stop_active is True
    assert loss.daily_stop_triggered_now is True


def test_state_round_trip_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = replace(
        initial_state("model_b", execution_mode="shadow", worker_pid=123),
        records_written=4,
    )
    write_state(path, state)
    loaded = load_state(
        path,
        role="model_b",
        execution_mode="shadow",
        worker_pid=123,
    )
    assert loaded.records_written == 4
    assert not path.with_suffix(".json.tmp").exists()


def test_kill_switch_can_flatten_on_already_processed_event() -> None:
    state = replace(
        initial_state("model_a", execution_mode="live"),
        virtual_position=1,
        last_event_time_utc="2026-07-21T15:45:00+00:00",
        kill_switch_active=True,
    )
    decision = decide_strategy_transition(
        state,
        rules=model_a_rules(),
        run_id="test",
        iteration=3,
        event_time_utc="2026-07-21T15:45:00+00:00",
        probability_up=0.80,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    assert decision.action == "KILL_SWITCH_FLATTEN"
    assert decision.duplicate_event is False
    assert decision.target_position == 0


def test_live_reconciliation_conservatively_counts_adopted_entry() -> None:
    state = replace(
        initial_state("model_b", execution_mode="live"),
        current_utc_date="2026-07-21",
        successful_entries_today=0,
        policy_changes_today=0,
    )
    position = BrokerPositionSnapshot(
        position=1,
        ticket=10,
        identifier=20,
        order_ticket=30,
        volume=0.01,
        magic=26070102,
        symbol="XAUUSD",
    )
    result = reconcile_state(
        state,
        broker(positions=(position,)),
        expected_magic=26070102,
        execution_mode="live",
    )
    assert result.incident is True
    assert result.state.successful_entries_today == 1
    assert result.state.policy_changes_today == 1


def test_model_a_daily_cap_allows_normal_exit() -> None:
    state = replace(
        initial_state("model_a", execution_mode="shadow"),
        virtual_position=1,
        hold_bars=4,
        policy_changes_today=3,
        last_event_time_utc="2026-07-21T15:30:00+00:00",
    )
    decision = decide_strategy_transition(
        state,
        rules=model_a_rules(),
        run_id="test",
        iteration=1,
        event_time_utc="2026-07-21T15:45:00+00:00",
        probability_up=0.50,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    after = apply_transition(
        state,
        decision,
        confirmed_position=0,
        broker_ticket=None,
    )
    assert decision.action == "EXIT_POSITION_CAP_REACHED"
    assert decision.target_position == 0
    assert decision.exit_allowed_when_capped is True
    assert after.policy_changes_today == 4


def test_model_a_daily_cap_converts_reversal_to_close_only() -> None:
    state = replace(
        initial_state("model_a", execution_mode="shadow"),
        virtual_position=1,
        hold_bars=4,
        policy_changes_today=3,
        last_event_time_utc="2026-07-21T15:30:00+00:00",
    )
    decision = decide_strategy_transition(
        state,
        rules=model_a_rules(),
        run_id="test",
        iteration=1,
        event_time_utc="2026-07-21T15:45:00+00:00",
        probability_up=0.40,
        stale_event_warning=False,
        reconciliation_blocked=False,
    )
    assert decision.action == "CLOSE_ONLY_DAILY_POLICY_CAP"
    assert decision.desired_position == -1
    assert decision.target_position == 0
    assert decision.close_only_reversal is True
    assert decision.entry_blocked_by_policy_cap is True


def test_gap_warmup_uses_market_event_clock_and_stays_flat() -> None:
    """A stale model sequence must not hide new completed broker bars."""

    rules = model_a_rules()
    state = replace(
        initial_state("model_a", execution_mode="live"),
        virtual_position=-1,
        broker_position=-1,
        hold_bars=8,
        last_event_time_utc="2026-07-23T20:45:00+00:00",
    )

    first_post_gap = decide_strategy_transition(
        state,
        rules=rules,
        run_id="weekend-test",
        iteration=1,
        event_time_utc="2026-07-23T22:00:00+00:00",
        probability_up=0.40,
        stale_event_warning=True,
        reconciliation_blocked=False,
    )
    assert first_post_gap.action == "CONTROL_GAP_FLATTEN"
    assert first_post_gap.target_position == 0
    assert first_post_gap.gap_from_previous_event is True

    after_flatten = apply_transition(
        state,
        first_post_gap,
        confirmed_position=0,
        broker_ticket=None,
    )
    assert after_flatten.last_event_time_utc == "2026-07-23T22:00:00+00:00"
    assert after_flatten.virtual_position == 0

    next_warmup_bar = decide_strategy_transition(
        after_flatten,
        rules=rules,
        run_id="weekend-test",
        iteration=2,
        event_time_utc="2026-07-23T22:15:00+00:00",
        probability_up=0.40,
        stale_event_warning=True,
        reconciliation_blocked=False,
    )
    assert (
        next_warmup_bar.action
        == "MODEL_UNAVAILABLE_CONTIGUITY_WARMUP"
    )
    assert next_warmup_bar.probability_up is None
    assert next_warmup_bar.target_position == 0

    after_warmup = apply_transition(
        after_flatten,
        next_warmup_bar,
        confirmed_position=0,
        broker_ticket=None,
    )
    assert after_warmup.last_event_time_utc == "2026-07-23T22:15:00+00:00"
    assert after_warmup.virtual_position == 0


def test_model_unavailable_warmup_never_preserves_exposure() -> None:
    state = replace(
        initial_state("model_a", execution_mode="live"),
        virtual_position=1,
        broker_position=1,
        last_event_time_utc="2026-07-24T00:00:00+00:00",
    )
    decision = decide_strategy_transition(
        state,
        rules=model_a_rules(),
        run_id="warmup-restart",
        iteration=1,
        event_time_utc="2026-07-24T00:15:00+00:00",
        probability_up=None,
        stale_event_warning=True,
        reconciliation_blocked=False,
    )

    assert decision.action == "CONTROL_MODEL_UNAVAILABLE_FLATTEN"
    assert decision.probability_up is None
    assert decision.target_position == 0


def test_session_gap_lockout_flattens_before_duplicate_suppression() -> None:
    state = replace(
        initial_state("model_a", execution_mode="live"),
        virtual_position=1,
        broker_position=1,
        last_event_time_utc="2026-07-24T20:15:00+00:00",
    )
    decision = decide_strategy_transition(
        state,
        rules=model_a_rules(),
        run_id="lockout-test",
        iteration=1,
        event_time_utc="2026-07-24T20:15:00+00:00",
        probability_up=0.60,
        stale_event_warning=False,
        reconciliation_blocked=False,
        session_gap_lockout_active=True,
        session_gap_lockout_reason="weekend_market_lockout_friday_preclose",
    )

    assert decision.action == "SESSION_GAP_LOCKOUT_FLATTEN"
    assert decision.target_position == 0
    assert decision.duplicate_event is False


def test_session_gap_lockout_schedule_covers_daily_break_and_weekend() -> None:
    start = 20 * 60 + 30
    daily_reopen = 22 * 60
    weekend_reopen = 22 * 60

    friday_preclose = session_gap_lockout_status(
        datetime(2026, 7, 24, 20, 30, tzinfo=timezone.utc),
        enabled=True,
        start_utc_minutes=start,
        daily_end_utc_minutes=daily_reopen,
        weekend_end_utc_minutes=weekend_reopen,
    )
    saturday = session_gap_lockout_status(
        datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
        enabled=True,
        start_utc_minutes=start,
        daily_end_utc_minutes=daily_reopen,
        weekend_end_utc_minutes=weekend_reopen,
    )
    sunday_before_open = session_gap_lockout_status(
        datetime(2026, 7, 26, 21, 59, tzinfo=timezone.utc),
        enabled=True,
        start_utc_minutes=start,
        daily_end_utc_minutes=daily_reopen,
        weekend_end_utc_minutes=weekend_reopen,
    )
    sunday_open = session_gap_lockout_status(
        datetime(2026, 7, 26, 22, 0, tzinfo=timezone.utc),
        enabled=True,
        start_utc_minutes=start,
        daily_end_utc_minutes=daily_reopen,
        weekend_end_utc_minutes=weekend_reopen,
    )
    monday_break = session_gap_lockout_status(
        datetime(2026, 7, 27, 20, 45, tzinfo=timezone.utc),
        enabled=True,
        start_utc_minutes=start,
        daily_end_utc_minutes=daily_reopen,
        weekend_end_utc_minutes=weekend_reopen,
    )
    monday_open = session_gap_lockout_status(
        datetime(2026, 7, 27, 10, 0, tzinfo=timezone.utc),
        enabled=True,
        start_utc_minutes=start,
        daily_end_utc_minutes=daily_reopen,
        weekend_end_utc_minutes=weekend_reopen,
    )

    assert friday_preclose[0] is True
    assert saturday[0] is True
    assert sunday_before_open[0] is True
    assert sunday_open == (False, "market_session_open")
    assert monday_break[0] is True
    assert monday_open == (False, "market_session_open")


def test_worker_transition_uses_latest_completed_event_clock() -> None:
    import ast
    from pathlib import Path

    worker_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "capstone_trading"
        / "runtime"
        / "dual_live_worker.py"
    )
    tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decide_strategy_transition"
    ]
    assert len(calls) == 1
    event_keyword = next(
        keyword for keyword in calls[0].keywords if keyword.arg == "event_time_utc"
    )
    assert isinstance(event_keyword.value, ast.Name)
    assert event_keyword.value.id == "latest_completed_event_time"


def test_fresh_runtime_first_event_is_adopt_only_when_flat() -> None:
    state = initial_state("model_a", execution_mode="live")
    decision = decide_strategy_transition(
        state,
        rules=model_a_rules(),
        run_id="fresh",
        iteration=1,
        event_time_utc="2026-07-26T20:45:00+00:00",
        probability_up=0.80,
        stale_event_warning=False,
        reconciliation_blocked=False,
        fresh_start_adopt_only=True,
    )
    assert decision.action == "CONTROL_FRESH_START_BLOCK"
    assert decision.target_position == 0
    assert decision.probability_up is None
    after = apply_transition(
        state,
        decision,
        confirmed_position=0,
        broker_ticket=None,
    )
    assert after.last_event_time_utc == "2026-07-26T20:45:00+00:00"


def test_fresh_runtime_first_event_flattens_adopted_exposure() -> None:
    state = replace(
        initial_state("model_a", execution_mode="live"),
        virtual_position=-1,
        broker_position=-1,
    )
    decision = decide_strategy_transition(
        state,
        rules=model_a_rules(),
        run_id="fresh",
        iteration=1,
        event_time_utc="2026-07-26T20:45:00+00:00",
        probability_up=0.20,
        stale_event_warning=False,
        reconciliation_blocked=False,
        fresh_start_adopt_only=True,
    )
    assert decision.action == "CONTROL_FRESH_START_FLATTEN"
    assert decision.target_position == 0


def test_atomic_json_replace_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import capstone_trading.runtime.dual_live_state as state_module

    target = tmp_path / "heartbeat.json"
    real_replace = state_module.os.replace
    calls = {"count": 0}

    def flaky_replace(source, destination):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("transient Windows reader lock")
        return real_replace(source, destination)

    monkeypatch.setattr(state_module.os, "replace", flaky_replace)
    monkeypatch.setattr(state_module, "ATOMIC_REPLACE_RETRY_SECONDS", 0.0)
    state_module.write_json_atomic(target, {"status": "RUNNING"})

    assert calls["count"] == 2
    assert '"status": "RUNNING"' in target.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_historical_unseen_event_times_excludes_latest_current_event() -> None:
    from capstone_trading.runtime.dual_live_state import (
        historical_unseen_event_times,
    )

    result = historical_unseen_event_times(
        [
            "2026-07-24T10:00:00+00:00",
            "2026-07-24T10:15:00+00:00",
            "2026-07-24T10:30:00+00:00",
            "2026-07-24T10:45:00+00:00",
        ],
        previous_event_time_utc="2026-07-24T10:00:00+00:00",
    )

    assert result == (
        "2026-07-24T10:15:00+00:00",
        "2026-07-24T10:30:00+00:00",
    )
