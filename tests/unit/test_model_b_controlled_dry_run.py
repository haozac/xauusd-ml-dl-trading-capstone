from __future__ import annotations

from pathlib import Path

import pytest

from capstone_trading.runtime.model_b_controlled_dry_run import (
    DryRunDecision,
    GuardedMt5DryRunProxy,
    ModelBDryRunRules,
    ModelBDryRunState,
    Stage3Step3ADryRunError,
    apply_decision_to_state,
    decide_model_b_action,
    load_required_stage3_step2_report,
    position_name,
)


RULES = ModelBDryRunRules()


def signal(prob: float, event: str = "2026-07-08T13:45:00+00:00") -> dict:
    return {
        "event_time_utc": event,
        "probability_up": prob,
        "stale_event_warning": False,
    }


def decide(prob: float, state: ModelBDryRunState | None = None, **kwargs) -> DryRunDecision:
    return decide_model_b_action(
        signal=signal(prob, kwargs.pop("event", "2026-07-08T13:45:00+00:00")),
        state=state or ModelBDryRunState(),
        rules=RULES,
        spread_points=kwargs.pop("spread_points", 500),
        spread_gate_points=800,
        actual_position_count=kwargs.pop("actual_position_count", 0),
        pending_order_count=kwargs.pop("pending_order_count", 0),
        run_id="unit",
        iteration=1,
        mode="unit",
    )


def test_model_b_enters_long_when_flat_and_probability_crosses_entry_threshold():
    decision = decide(0.56)
    assert decision.action == "ENTER_LONG"
    assert decision.would_send_order_in_stage3b is True
    assert decision.virtual_position_after == 1


def test_model_b_holds_flat_below_entry_threshold():
    decision = decide(0.549)
    assert decision.action == "HOLD_FLAT"
    assert decision.would_send_order_in_stage3b is False
    assert decision.virtual_position_after == 0


def test_model_b_exits_long_below_exit_threshold():
    state = ModelBDryRunState(virtual_position=1, virtual_position_name="LONG")
    decision = decide(0.49, state=state)
    assert decision.action == "EXIT_LONG"
    assert decision.would_send_order_in_stage3b is True
    assert decision.virtual_position_after == 0


def test_model_b_holds_long_at_exit_threshold_or_above():
    state = ModelBDryRunState(virtual_position=1, virtual_position_name="LONG")
    decision = decide(0.50, state=state)
    assert decision.action == "HOLD_LONG"
    assert decision.virtual_position_after == 1


def test_daily_entry_cap_blocks_second_entry_same_utc_day():
    state = ModelBDryRunState(successful_entry_dates={"2026-07-08": 1})
    decision = decide(0.60, state=state)
    assert decision.action == "BLOCK_DAILY_ENTRY_CAP"
    assert decision.would_send_order_in_stage3b is False


def test_duplicate_event_is_skipped_without_state_change():
    state = ModelBDryRunState(last_event_time_utc="2026-07-08T13:45:00+00:00")
    decision = decide(0.60, state=state)
    assert decision.action == "DUPLICATE_SKIP"
    next_state = apply_decision_to_state(state, decision)
    assert next_state is state


def test_spread_gate_blocks_entry():
    decision = decide(0.60, spread_points=801)
    assert decision.action == "BLOCK_SPREAD"
    assert decision.would_send_order_in_stage3b is False


def test_actual_position_blocks_dry_run_decision():
    decision = decide(0.60, actual_position_count=1)
    assert decision.action == "BLOCK_ACTUAL_POSITION_EXISTS"


def test_apply_decision_records_successful_entry_date():
    state = ModelBDryRunState()
    decision = decide(0.60, state=state)
    next_state = apply_decision_to_state(state, decision)
    assert next_state.virtual_position == 1
    assert next_state.successful_entry_dates == {"2026-07-08": 1}
    assert next_state.records_written == 1


class FakeMt5:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0

    def order_send(self, request):
        return None


def test_dry_run_proxy_blocks_order_send():
    proxy = GuardedMt5DryRunProxy(FakeMt5())
    with pytest.raises(Stage3Step3ADryRunError):
        proxy.order_send({})
    assert proxy.forbidden_attempts == ["order_send"]


def test_load_required_stage3_step2_report_requires_clean_pass(tmp_path: Path):
    report_path = tmp_path / "runtime" / "reports" / "stage3_step2_v1_1_tiny_order_test.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        '{"status":"PASS","formal_gate":true,"open_close_completed":true,"orders_executed":true,'
        '"validations":{"history_records_recovered":true,"no_position_after_close":true}}',
        encoding="utf-8",
    )
    report = load_required_stage3_step2_report(tmp_path)
    assert report["status"] == "PASS"


def test_position_name():
    assert position_name(0) == "FLAT"
    assert position_name(1) == "LONG"
