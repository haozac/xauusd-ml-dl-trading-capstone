from __future__ import annotations

import math

import pandas as pd
import pytest

from capstone_trading.errors import TradingReplayError
from capstone_trading.evaluation.model_b_replay import (
    ModelBOverlayRules,
    overlay_rules_from_model_b_config,
    replay_model_b,
)


def rules() -> ModelBOverlayRules:
    return ModelBOverlayRules(
        entry_threshold=0.55,
        exit_threshold=0.50,
        max_successful_entries_per_day=1,
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


def test_model_b_enters_holds_and_exits_long_only() -> None:
    predictions = frame([0.56, 0.51, 0.50, 0.499, 0.60])
    log, metrics, diagnostics = replay_model_b(predictions, rules(), cost_bps=1.0)

    assert log["position"].tolist() == [1, 1, 1, 0, 0]
    assert log["turnover"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert metrics.turnover_units == 2.0
    assert diagnostics.successful_entry_count == 1
    assert diagnostics.short_position_count == 0
    assert diagnostics.invariant_passed


def test_one_successful_entry_per_day_blocks_reentry() -> None:
    predictions = frame([0.56, 0.49, 0.60, 0.60])
    log, _metrics, diagnostics = replay_model_b(predictions, rules(), cost_bps=0.0)

    assert log["position"].tolist() == [1, 0, 0, 0]
    assert log.loc[2, "change_reason"] == "daily_entry_cap_active"
    assert diagnostics.successful_entry_count == 1
    assert diagnostics.daily_entry_cap_block_count >= 1
    assert diagnostics.max_successful_entries_in_utc_day == 1


def test_new_utc_day_resets_entry_cap() -> None:
    predictions = frame([0.56, 0.49, 0.60])
    predictions.loc[2, "time"] = pd.Timestamp("2025-01-02 00:00:00+00:00")
    log, _metrics, diagnostics = replay_model_b(predictions, rules(), cost_bps=0.0)

    assert log["position"].tolist() == [1, 0, 0]
    assert log.loc[2, "change_reason"] == "gap_block"
    assert diagnostics.successful_entry_count == 1


def test_gap_exit_and_gap_block_are_non_tradable() -> None:
    predictions = frame([0.56, 0.56, 0.56])
    predictions.loc[1, "time"] = pd.Timestamp("2025-01-02 00:00:00+00:00")
    predictions.loc[2, "time"] = pd.Timestamp("2025-01-03 00:00:00+00:00")
    log, metrics, diagnostics = replay_model_b(predictions, rules(), cost_bps=0.0)

    assert log["position"].tolist() == [1, 0, 0]
    assert log.loc[1, "change_reason"] == "gap_exit"
    assert log.loc[2, "change_reason"] == "gap_block"
    assert metrics.gap_exit_events == 1
    assert diagnostics.gap_block_count == 1


def test_daily_and_total_stop_flatten_from_next_bar() -> None:
    predictions = frame([0.56, 0.56, 0.56], returns=[-0.03, 0.01, 0.01])
    log, metrics, diagnostics = replay_model_b(predictions, rules(), cost_bps=0.0)

    assert bool(log.loc[0, "daily_stop_triggered"])
    assert log.loc[1, "position"] == 0
    assert log.loc[1, "change_reason"] == "daily_loss_stop"
    assert metrics.daily_stop_trigger_count == 1
    assert diagnostics.invariant_passed


def test_overlay_rules_from_model_b_config_parses_frozen_shape() -> None:
    config = {
        "strategy_id": "MODEL_B_V2",
        "status": "FROZEN_STAGE_0",
        "overlay": {
            "short_positions_allowed": False,
            "entry": {"threshold": 0.55},
            "normal_exit": {"threshold": 0.50},
            "minimum_hold_eligible_bars": 0,
            "maximum_successful_new_entries_per_utc_day": 1,
        },
        "risk_governance": {
            "daily_loss_stop_simple_return": -0.02,
            "total_drawdown_stop": -0.15,
        },
    }

    parsed = overlay_rules_from_model_b_config(config)

    assert parsed.entry_threshold == pytest.approx(0.55)
    assert parsed.exit_threshold == pytest.approx(0.50)
    assert parsed.daily_loss_log_threshold == pytest.approx(math.log(0.98))


def test_model_b_config_rejects_shorts() -> None:
    config = {
        "strategy_id": "MODEL_B_V2",
        "status": "FROZEN_STAGE_0",
        "overlay": {
            "short_positions_allowed": True,
            "entry": {"threshold": 0.55},
            "normal_exit": {"threshold": 0.50},
            "minimum_hold_eligible_bars": 0,
            "maximum_successful_new_entries_per_utc_day": 1,
        },
        "risk_governance": {
            "daily_loss_stop_simple_return": -0.02,
            "total_drawdown_stop": -0.15,
        },
    }

    with pytest.raises(TradingReplayError):
        overlay_rules_from_model_b_config(config)
