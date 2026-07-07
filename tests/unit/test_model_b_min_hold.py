from __future__ import annotations

import pandas as pd
import pytest

from capstone_trading.evaluation.model_b_min_hold import (
    create_min_hold_rules,
    replay_model_b_min_hold,
)
from capstone_trading.evaluation.model_b_replay import ModelBOverlayRules, replay_model_b


def _rules() -> ModelBOverlayRules:
    return ModelBOverlayRules(
        entry_threshold=0.55,
        exit_threshold=0.50,
        max_successful_entries_per_day=5,
        daily_loss_log_threshold=-99.0,
        total_drawdown_stop=-0.99,
    )


def _predictions(probabilities: list[float]) -> pd.DataFrame:
    times = pd.date_range("2024-01-01 00:00:00+00:00", periods=len(probabilities), freq="15min")
    return pd.DataFrame(
        {
            "time": times,
            "p_up": probabilities,
            "target_dir": [1] * len(probabilities),
            "target_ret_fwd": [0.001] * len(probabilities),
        }
    )


def test_min_hold_blocks_normal_exit_until_three_completed_bars() -> None:
    rules = create_min_hold_rules(_rules(), minimum_hold_bars=3)
    predictions = _predictions([0.56, 0.49, 0.48, 0.47, 0.47])

    log, _metrics, diagnostics = replay_model_b_min_hold(predictions, rules, cost_bps=1.0)

    assert log["position"].tolist() == [1, 1, 1, 0, 0]
    assert log["change_reason"].tolist()[1:4] == [
        "minimum_hold_exit_block",
        "minimum_hold_exit_block",
        "policy_exit",
    ]
    assert diagnostics.successful_entry_count == 1
    assert diagnostics.normal_exit_count == 1
    assert diagnostics.min_hold_blocked_exit_count == 2
    assert diagnostics.short_position_count == 0
    assert diagnostics.invariant_passed is True


def test_zero_min_hold_matches_current_model_b_positions() -> None:
    base_rules = _rules()
    min_hold_rules = create_min_hold_rules(base_rules, minimum_hold_bars=0)
    predictions = _predictions([0.56, 0.52, 0.49, 0.60, 0.48])

    current_log, current_metrics, _current_diag = replay_model_b(predictions, base_rules, cost_bps=0.5)
    candidate_log, candidate_metrics, candidate_diag = replay_model_b_min_hold(
        predictions,
        min_hold_rules,
        cost_bps=0.5,
    )

    assert candidate_log["position"].tolist() == current_log["position"].tolist()
    assert candidate_metrics.net_total_return == pytest.approx(current_metrics.net_total_return)
    assert candidate_metrics.turnover_units == pytest.approx(current_metrics.turnover_units)
    assert candidate_diag.min_hold_blocked_exit_count == 0
    assert candidate_diag.invariant_passed is True


def test_gap_exit_overrides_minimum_hold() -> None:
    rules = create_min_hold_rules(_rules(), minimum_hold_bars=3)
    times = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:15:00Z",
            "2024-01-01T01:00:00Z",
        ],
        utc=True,
    )
    predictions = pd.DataFrame(
        {
            "time": times,
            "p_up": [0.56, 0.49, 0.49],
            "target_dir": [1, 1, 1],
            "target_ret_fwd": [0.001, 0.001, 0.001],
        }
    )

    log, _metrics, diagnostics = replay_model_b_min_hold(predictions, rules, cost_bps=1.0)

    assert log["position"].tolist() == [1, 1, 0]
    assert log["change_reason"].tolist()[2] == "gap_exit"
    assert diagnostics.gap_exit_events == 1
    assert diagnostics.min_hold_blocked_exit_count == 1
    assert diagnostics.invariant_passed is True


def test_daily_entry_cap_still_applies() -> None:
    base = ModelBOverlayRules(
        entry_threshold=0.55,
        exit_threshold=0.50,
        max_successful_entries_per_day=1,
        daily_loss_log_threshold=-99.0,
        total_drawdown_stop=-0.99,
    )
    rules = create_min_hold_rules(base, minimum_hold_bars=0)
    predictions = _predictions([0.56, 0.49, 0.60, 0.49])

    log, _metrics, diagnostics = replay_model_b_min_hold(predictions, rules, cost_bps=1.0)

    assert log["position"].tolist() == [1, 0, 0, 0]
    assert log["change_reason"].tolist()[2] == "daily_entry_cap_active"
    assert diagnostics.successful_entry_count == 1
    assert diagnostics.daily_entry_cap_block_count == 1
    assert diagnostics.max_successful_entries_in_utc_day == 1
    assert diagnostics.invariant_passed is True
