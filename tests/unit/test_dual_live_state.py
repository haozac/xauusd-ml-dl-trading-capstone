from __future__ import annotations

from dataclasses import replace
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
    assert decision.action == "GAP_FLATTEN"
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
