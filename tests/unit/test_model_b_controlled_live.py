from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from capstone_trading.runtime.model_b_controlled_dry_run import ModelBDryRunRules
from capstone_trading.runtime.model_b_controlled_live import (
    CONFIRM_SEND_TOKEN,
    ModelBLiveState,
    Stage3Step3BLiveError,
    apply_live_decision_to_state,
    compact_comment,
    decide_model_b_live_action,
    inspect_symbol_for_live_runtime,
    load_live_state,
    load_required_stage3_step2_report,
    load_required_stage3_step3a_report,
    summarise_live_decisions,
    write_live_state,
)


def rules() -> ModelBDryRunRules:
    return ModelBDryRunRules(
        variant="MODEL_B_V2_CURRENT",
        entry_threshold=0.55,
        exit_threshold=0.50,
        long_only=True,
        max_successful_entries_per_utc_day=1,
        min_hold_bars=0,
    )


def signal(p: float, event: str = "2026-07-09T12:00:00+00:00") -> dict:
    return {"event_time_utc": event, "probability_up": p, "stale_event_warning": False}


def test_flat_below_entry_holds_flat():
    decision = decide_model_b_live_action(
        signal=signal(0.53),
        state=ModelBLiveState(),
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=0,
        pending_order_count=0,
    )
    assert decision.action == "HOLD_FLAT"
    assert decision.live_position_after == 0
    assert decision.order_send_called is False


def test_flat_at_entry_enters_long():
    decision = decide_model_b_live_action(
        signal=signal(0.55),
        state=ModelBLiveState(),
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=0,
        pending_order_count=0,
    )
    assert decision.action == "ENTER_LONG"
    assert decision.live_position_before == 0
    assert decision.live_position_after == 1


def test_daily_entry_cap_blocks_second_entry():
    state = ModelBLiveState(successful_entry_dates={"2026-07-09": 1})
    decision = decide_model_b_live_action(
        signal=signal(0.60),
        state=state,
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=0,
        pending_order_count=0,
    )
    assert decision.action == "BLOCK_DAILY_ENTRY_CAP"
    assert decision.live_position_after == 0


def test_long_holds_above_exit_threshold():
    state = ModelBLiveState(live_position=1, live_position_name="LONG", position_ticket=123)
    decision = decide_model_b_live_action(
        signal=signal(0.51),
        state=state,
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=1,
        pending_order_count=0,
    )
    assert decision.action == "HOLD_LONG"
    assert decision.live_position_after == 1


def test_long_exits_below_exit_threshold():
    state = ModelBLiveState(live_position=1, live_position_name="LONG", position_ticket=123)
    decision = decide_model_b_live_action(
        signal=signal(0.499),
        state=state,
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=1,
        pending_order_count=0,
    )
    assert decision.action == "EXIT_LONG"
    assert decision.live_position_after == 0


def test_duplicate_event_is_skipped():
    state = ModelBLiveState(last_event_time_utc="2026-07-09T12:00:00+00:00")
    decision = decide_model_b_live_action(
        signal=signal(0.60),
        state=state,
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=0,
        pending_order_count=0,
    )
    assert decision.action == "DUPLICATE_SKIP"
    assert decision.duplicate_event is True


def test_spread_gate_blocks_entry():
    decision = decide_model_b_live_action(
        signal=signal(0.60),
        state=ModelBLiveState(),
        rules=rules(),
        spread_points=801,
        spread_gate_points=800,
        actual_position_count=0,
        pending_order_count=0,
    )
    assert decision.action == "BLOCK_SPREAD"


def test_high_spread_does_not_override_flat_below_entry_threshold():
    decision = decide_model_b_live_action(
        signal=signal(0.54),
        state=ModelBLiveState(),
        rules=rules(),
        spread_points=890,
        spread_gate_points=800,
        actual_position_count=0,
        pending_order_count=0,
    )
    assert decision.action == "HOLD_FLAT"
    assert decision.reason == "p_up_below_model_b_entry_threshold"


def test_high_spread_does_not_block_risk_reducing_exit():
    state = ModelBLiveState(live_position=1, live_position_name="LONG", position_ticket=123)
    decision = decide_model_b_live_action(
        signal=signal(0.49),
        state=state,
        rules=rules(),
        spread_points=1200,
        spread_gate_points=800,
        actual_position_count=1,
        pending_order_count=0,
    )
    assert decision.action == "EXIT_LONG"
    assert decision.live_position_after == 0


def test_continuous_symbol_inspection_records_wide_spread_without_raising():
    symbol = SimpleNamespace(
        name="XAUUSD",
        visible=True,
        trade_mode=4,
        volume_min=0.01,
        volume_step=0.01,
        spread=890,
    )

    class FakeSymbolMt5:
        def symbol_info(self, name):
            assert name == "XAUUSD"
            return symbol

    controls = SimpleNamespace(
        symbol="XAUUSD",
        stage3_first_order_test_volume_lots=0.01,
        max_spread_points_for_entry=800,
    )
    snapshot = inspect_symbol_for_live_runtime(FakeSymbolMt5(), controls)
    assert snapshot["spread"] == 890
    assert snapshot["live_runtime_checks"]["spread_within_entry_gate"] is False


def test_state_actual_position_mismatch_blocks():
    decision = decide_model_b_live_action(
        signal=signal(0.60),
        state=ModelBLiveState(),
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=1,
        pending_order_count=0,
    )
    assert decision.action == "BLOCK_ACTUAL_POSITION_STATE_MISMATCH"


def test_apply_entry_updates_state_and_daily_count():
    decision = decide_model_b_live_action(
        signal=signal(0.60),
        state=ModelBLiveState(),
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=0,
        pending_order_count=0,
    )
    # Simulate successful send by replacing the frozen dataclass through dict.
    payload = decision.__dict__.copy()
    payload.update({"order_send_called": True, "order_send_passed": True, "order_ticket": 999, "broker_position_ticket": 123})
    decision = type(decision)(**payload)
    state = apply_live_decision_to_state(
        ModelBLiveState(),
        decision,
        opened_position={"ticket": 123, "identifier": 123, "volume": 0.01},
    )
    assert state.live_position == 1
    assert state.position_ticket == 123
    assert state.open_order_ticket == 999
    assert state.successful_entry_dates == {"2026-07-09": 1}


def test_apply_exit_clears_position_and_counts_cycle():
    state = ModelBLiveState(live_position=1, live_position_name="LONG", position_ticket=123, completed_entry_exit_cycles=0)
    decision = decide_model_b_live_action(
        signal=signal(0.49),
        state=state,
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=1,
        pending_order_count=0,
    )
    payload = decision.__dict__.copy()
    payload.update({"order_send_called": True, "order_send_passed": True})
    decision = type(decision)(**payload)
    state = apply_live_decision_to_state(state, decision, closed_position=True)
    assert state.live_position == 0
    assert state.position_ticket is None
    assert state.completed_entry_exit_cycles == 1


def test_compact_comment_is_mt5_safe_length():
    comment = compact_comment("CP_S3P3B_B", "stage3_step3b_20260709T123456Z", "2026-07-09T12:00:00+00:00", "_O")
    assert comment.startswith("CP_S3P3B_B")
    assert len(comment) <= 31


def test_write_and_load_live_state(tmp_path: Path):
    path = tmp_path / "state.json"
    write_live_state(path, ModelBLiveState(live_position=1, live_position_name="LONG", position_ticket=321))
    loaded = load_live_state(path)
    assert loaded.live_position == 1
    assert loaded.position_ticket == 321


def test_load_required_step2_report_requires_history(tmp_path: Path):
    path = tmp_path / "step2.json"
    path.write_text(json.dumps({
        "status": "PASS",
        "formal_gate": True,
        "open_close_completed": True,
        "orders_executed": True,
        "validations": {"history_records_recovered": False, "no_position_after_close": True},
    }), encoding="utf-8")
    with pytest.raises(Stage3Step3BLiveError):
        load_required_stage3_step2_report(tmp_path, Path("step2.json"))


def test_load_required_step3a_report_requires_permission(tmp_path: Path):
    path = tmp_path / "step3a.json"
    path.write_text(json.dumps({
        "status": "PASS",
        "formal_gate": True,
        "order_send_called": False,
        "safety": {"stage3_step3b_single_model_execution_allowed": False},
        "summary": {"unique_completed_m15_events": 8},
    }), encoding="utf-8")
    with pytest.raises(Stage3Step3BLiveError):
        load_required_stage3_step3a_report(tmp_path, Path("step3a.json"))


def test_summarise_live_decisions_counts_actions():
    d1 = decide_model_b_live_action(
        signal=signal(0.53, "2026-07-09T12:00:00+00:00"),
        state=ModelBLiveState(),
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=0,
        pending_order_count=0,
    )
    d2 = decide_model_b_live_action(
        signal=signal(0.60, "2026-07-09T12:15:00+00:00"),
        state=ModelBLiveState(),
        rules=rules(),
        spread_points=600,
        spread_gate_points=800,
        actual_position_count=0,
        pending_order_count=0,
    )
    summary = summarise_live_decisions([d1, d2])
    assert summary["unique_completed_m15_events"] == 2
    assert summary["action_counts"]["HOLD_FLAT"] == 1
    assert summary["action_counts"]["ENTER_LONG"] == 1


def test_confirmation_token_constant_is_explicit():
    assert CONFIRM_SEND_TOKEN == "I_UNDERSTAND_STAGE3_STEP3B_MODEL_B_ORDER_SEND"
