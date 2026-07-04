from __future__ import annotations

import math

import pandas as pd
import pytest

from capstone_trading.errors import TradingReplayError
from capstone_trading.evaluation.trading_replay import (
    ModelAOverlayRules,
    compare_replayed_bar_log,
    overlay_rules_from_config,
    replay_model_a,
)


def rules() -> ModelAOverlayRules:
    return ModelAOverlayRules(
        long_threshold=0.53,
        short_threshold=0.47,
        minimum_hold_bars=3,
        max_policy_changes_per_day=3,
        daily_loss_log_threshold=math.log(0.98),
        total_drawdown_stop=-0.15,
    )


def frame(probabilities, returns=None, *, start="2025-01-01 00:00:00+00:00"):
    if returns is None:
        returns = [0.001] * len(probabilities)
    times = pd.date_range(start, periods=len(probabilities), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "p_up": probabilities,
            "target_ret_fwd": returns,
            "target_dir": [1 if value > 0 else 0 for value in returns],
        }
    )


def test_replay_applies_entry_hold_exit_and_costs() -> None:
    predictions = frame([0.56, 0.50, 0.50, 0.50, 0.50])
    log, metrics = replay_model_a(predictions, rules(), cost_bps=1.0)

    assert log["position"].tolist() == [1, 1, 1, 1, 0]
    assert log["turnover"].tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert metrics.turnover_units == 2.0
    assert metrics.policy_change_events == 2
    assert metrics.round_turn_equivalent_trades == 1.0
    assert metrics.final_net_equity == pytest.approx(math.exp(0.004 - 0.0002))


def test_reversal_counts_one_policy_event_but_two_turnover_units() -> None:
    predictions = frame([0.56, 0.56, 0.56, 0.56, 0.44])
    log, metrics = replay_model_a(predictions, rules(), cost_bps=1.0)

    assert log["position"].tolist() == [1, 1, 1, 1, -1]
    assert log["policy_event_units"].tolist() == [1, 0, 0, 0, 1]
    assert log["turnover"].tolist() == [1.0, 0.0, 0.0, 0.0, 2.0]
    assert metrics.policy_change_events == 2
    assert metrics.turnover_units == 3.0


def test_gap_exit_overrides_minimum_hold_and_does_not_count_policy_cap() -> None:
    predictions = frame([0.56, 0.56, 0.56, 0.56])
    predictions.loc[3, "time"] = pd.Timestamp("2025-01-02 00:00:00+00:00")
    log, metrics = replay_model_a(predictions, rules(), cost_bps=0.0)

    assert log.loc[3, "position"] == 0
    assert log.loc[3, "change_reason"] == "gap_exit"
    assert metrics.gap_exit_events == 1
    assert metrics.policy_change_events == 1


def test_gap_bar_blocks_new_entry_even_when_already_flat() -> None:
    predictions = frame([0.50, 0.56], start="2025-01-01 00:00:00+00:00")
    predictions.loc[1, "time"] = pd.Timestamp("2025-01-02 00:00:00+00:00")
    log, metrics = replay_model_a(predictions, rules(), cost_bps=0.0)

    assert log["position"].tolist() == [0, 0]
    assert log.loc[1, "change_reason"] == "gap_block"
    assert metrics.gap_exit_events == 0
    assert metrics.policy_change_events == 0


def test_policy_exit_requires_flat_cooldown_before_reentry() -> None:
    predictions = frame([0.56, 0.50, 0.50, 0.50, 0.50, 0.56, 0.56, 0.56, 0.56])
    log, metrics = replay_model_a(predictions, rules(), cost_bps=0.0)

    assert log["position"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 1]
    assert log.loc[5, "change_reason"] == "minimum_hold_active"
    assert log.loc[8, "change_reason"] == "policy_entry"
    assert metrics.policy_change_events == 3


def test_daily_stop_takes_effect_from_next_eligible_bar_until_next_day() -> None:
    predictions = frame([0.56, 0.56, 0.56, 0.56, 0.56], returns=[-0.03, 0.01, 0.01, 0.01, 0.01])
    log, metrics = replay_model_a(predictions, rules(), cost_bps=0.0)

    assert log.loc[0, "daily_stop_triggered"] is True or bool(log.loc[0, "daily_stop_triggered"])
    assert log.loc[1, "position"] == 0
    assert log.loc[1, "change_reason"] == "daily_loss_stop"
    assert metrics.daily_stop_trigger_count == 1
    assert metrics.daily_stop_exit_events == 1


def test_total_stop_blocks_new_entries_after_trigger() -> None:
    predictions = frame([0.56, 0.56, 0.56, 0.56, 0.56], returns=[-0.20, 0.03, 0.03, 0.03, 0.03])
    log, metrics = replay_model_a(predictions, rules(), cost_bps=0.0)

    assert bool(log.loc[0, "total_stop_triggered"])
    assert log.loc[1, "position"] == 0
    assert log.loc[1, "change_reason"] == "total_drawdown_stop"
    assert metrics.total_stop_triggered is True
    assert metrics.total_stop_exit_events == 1


def test_compare_replayed_bar_log_detects_exact_match_and_mismatch() -> None:
    predictions = frame([0.56, 0.56, 0.56, 0.50])
    generated, _metrics = replay_model_a(predictions, rules(), cost_bps=1.0)
    reference = generated[["time", "position", "turnover", "gross_log_return", "net_log_return", "net_equity"]].copy()

    comparisons = compare_replayed_bar_log(generated, reference, tolerance=1e-12)
    assert {item.column for item in comparisons} >= {"position", "turnover", "net_equity"}

    reference.loc[0, "position"] = 0
    with pytest.raises(TradingReplayError):
        compare_replayed_bar_log(generated, reference, tolerance=1e-12)


def test_overlay_rules_from_config_accepts_frozen_daily_loss_key() -> None:
    config = {
        "overlay": {
            "long_when_p_up_gte": 0.53,
            "short_when_p_up_lte": 0.47,
            "minimum_hold_eligible_bars": 3,
            "maximum_overlay_position_change_events_per_utc_day": 3,
        },
        "historical_risk_semantics": {
            "daily_loss_stop_log_threshold": math.log(0.98),
            "total_drawdown_stop": -0.15,
        },
    }

    parsed = overlay_rules_from_config(config)

    assert parsed.daily_loss_log_threshold == pytest.approx(math.log(0.98))


def test_compare_overlay_selection_metrics_accepts_selected_overlay_keys() -> None:
    from capstone_trading.evaluation.trading_replay import (
        ReplayMetrics,
        compare_overlay_selection_metrics,
    )

    metric = ReplayMetrics(
        cost_bps=1.0,
        row_count=10,
        active_bar_count=5,
        active_bar_rate=0.5,
        turnover_units=8.0,
        round_turn_equivalent_trades=4.0,
        policy_change_events=4,
        gap_exit_events=0,
        daily_stop_exit_events=0,
        daily_stop_trigger_count=0,
        total_stop_exit_events=0,
        total_stop_triggered=False,
        first_total_stop_trigger_utc=None,
        final_gross_equity=1.2,
        final_net_equity=1.1,
        gross_total_return=0.2,
        net_total_return=0.1,
        max_drawdown=-0.05,
        gross_log_return_sum=0.18,
        net_log_return_sum=0.095,
        daily_stop_dates=(),
    )
    selected = {
        "validation_selected_net_return": 0.1,
        "validation_selected_max_drawdown": -0.05,
        "validation_selected_trade_count": 4.0,
        "validation_selected_active_rate": 0.5,
    }

    comparisons = compare_overlay_selection_metrics(metric, selected, tolerance=1e-12)

    assert len(comparisons) == 4
    assert all(item.passed for item in comparisons)
