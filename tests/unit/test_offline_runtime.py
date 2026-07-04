from __future__ import annotations

import math

import pandas as pd
import pytest

from capstone_trading.evaluation.model_b_replay import ModelBOverlayRules, replay_model_b
from capstone_trading.evaluation.trading_replay import ModelAOverlayRules, replay_model_a
from capstone_trading.runtime.offline_simulation import (
    compare_full_vs_resumed,
    prediction_events_from_partition,
    run_model_a_runtime,
    run_model_a_runtime_with_resume,
    run_model_b_runtime,
    run_model_b_runtime_with_resume,
)


def model_a_rules() -> ModelAOverlayRules:
    return ModelAOverlayRules(
        long_threshold=0.53,
        short_threshold=0.47,
        minimum_hold_bars=3,
        max_policy_changes_per_day=3,
        daily_loss_log_threshold=math.log(0.98),
        total_drawdown_stop=-0.15,
    )


def model_b_rules() -> ModelBOverlayRules:
    return ModelBOverlayRules(
        entry_threshold=0.55,
        exit_threshold=0.50,
        max_successful_entries_per_day=1,
        daily_loss_log_threshold=math.log(0.98),
        total_drawdown_stop=-0.15,
    )


def events(probabilities, returns=None, *, start="2025-01-01 00:00:00+00:00") -> pd.DataFrame:
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


def assert_same_log(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    columns = [
        "position",
        "turnover",
        "gross_log_return",
        "cost_log_return",
        "net_log_return",
        "gross_equity",
        "net_equity",
        "net_drawdown",
    ]
    assert actual["time"].tolist() == expected["time"].tolist()
    for column in columns:
        assert actual[column].tolist() == pytest.approx(expected[column].tolist())


def test_model_a_runtime_matches_batch_replay() -> None:
    frame = events([0.56, 0.56, 0.56, 0.56, 0.50, 0.44, 0.44, 0.44, 0.44])
    runtime_log, runtime_metrics = run_model_a_runtime(frame, model_a_rules(), cost_bps=1.0)
    batch_log, batch_metrics = replay_model_a(frame, model_a_rules(), cost_bps=1.0)

    assert_same_log(runtime_log, batch_log)
    assert runtime_metrics.net_total_return == pytest.approx(batch_metrics.net_total_return)
    assert runtime_metrics.turnover_units == pytest.approx(batch_metrics.turnover_units)


def test_model_b_runtime_matches_batch_replay() -> None:
    frame = events([0.56, 0.54, 0.51, 0.49, 0.56, 0.56, 0.49])
    runtime_log, runtime_metrics, runtime_diagnostics = run_model_b_runtime(
        frame,
        model_b_rules(),
        cost_bps=1.0,
    )
    batch_log, batch_metrics, batch_diagnostics = replay_model_b(
        frame,
        model_b_rules(),
        cost_bps=1.0,
    )

    assert_same_log(runtime_log, batch_log)
    assert runtime_metrics.net_total_return == pytest.approx(batch_metrics.net_total_return)
    assert runtime_diagnostics.short_position_count == batch_diagnostics.short_position_count == 0


def test_model_a_resume_reproduces_uninterrupted_runtime() -> None:
    frame = events([0.56, 0.56, 0.56, 0.56, 0.50, 0.50, 0.44, 0.44, 0.44, 0.44])
    full_log, _metrics = run_model_a_runtime(frame, model_a_rules(), cost_bps=0.5)
    resumed_log, _resumed_metrics = run_model_a_runtime_with_resume(
        frame,
        model_a_rules(),
        cost_bps=0.5,
        split_at=5,
    )
    comparison = compare_full_vs_resumed(
        model_id="MODEL_A",
        partition="unit",
        cost_bps=0.5,
        full_log=full_log,
        resumed_log=resumed_log,
        columns=("position", "turnover", "net_equity"),
        tolerance=1e-12,
    )

    assert comparison.passed
    assert comparison.mismatch_count == 0


def test_model_b_resume_reproduces_uninterrupted_runtime() -> None:
    frame = events([0.56, 0.54, 0.51, 0.49, 0.56, 0.56, 0.49, 0.56, 0.49])
    full_log, _metrics, _diag = run_model_b_runtime(frame, model_b_rules(), cost_bps=0.5)
    resumed_log, _resumed_metrics, resumed_diag = run_model_b_runtime_with_resume(
        frame,
        model_b_rules(),
        cost_bps=0.5,
        split_at=4,
    )
    comparison = compare_full_vs_resumed(
        model_id="MODEL_B_V2",
        partition="unit",
        cost_bps=0.5,
        full_log=full_log,
        resumed_log=resumed_log,
        columns=("position", "turnover", "net_equity"),
        tolerance=1e-12,
    )

    assert comparison.passed
    assert resumed_diag.short_position_count == 0


def test_prediction_events_from_partition_uses_endpoint_rows() -> None:
    index = pd.date_range("2025-01-01 00:00:00+00:00", periods=5, freq="15min", tz="UTC")
    partition = pd.DataFrame(
        {
            "target_ret_fwd": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target_dir": [1, 1, 1, 1, 1],
        },
        index=index,
    )
    out = prediction_events_from_partition(
        partition=partition,
        endpoint_positions=[2, 4],
        probabilities=[0.6, 0.4],
    )

    assert out["time"].tolist() == [index[2], index[4]]
    assert out["p_up"].tolist() == [0.6, 0.4]
    assert out["target_ret_fwd"].tolist() == [0.3, 0.5]


def test_streaming_event_materialisation_matches_batch_plan() -> None:
    from capstone_trading.runtime.offline_simulation import verify_streaming_event_materialisation

    index = pd.date_range("2025-01-01 00:00:00+00:00", periods=6, freq="15min", tz="UTC")
    partition = pd.DataFrame(
        {
            "target_ret_fwd": [0.01, 0.02, 0.03, -0.01, 0.04, 0.05],
            "target_dir": [1, 1, 1, 0, 1, 1],
        },
        index=index,
    )
    batch_events = prediction_events_from_partition(
        partition=partition,
        endpoint_positions=[2, 3, 4, 5],
        probabilities=[0.51, 0.52, 0.53, 0.54],
    )

    report = verify_streaming_event_materialisation(
        partition=partition,
        endpoint_positions=[2, 3, 4, 5],
        batch_events=batch_events,
        sequence_length=3,
        partition_name="unit",
    )

    assert report.passed
    assert report.streaming_event_count == 4
    assert report.endpoint_position_mismatch_count == 0
    assert report.timestamp_mismatch_count == 0
    assert report.maximum_target_return_difference == pytest.approx(0.0)


def test_streaming_event_materialisation_respects_gaps() -> None:
    from capstone_trading.runtime.offline_simulation import streaming_endpoint_positions_from_features

    index = pd.DatetimeIndex(
        [
            "2025-01-01 00:00:00+00:00",
            "2025-01-01 00:15:00+00:00",
            "2025-01-01 00:30:00+00:00",
            "2025-01-02 00:00:00+00:00",
            "2025-01-02 00:15:00+00:00",
            "2025-01-02 00:30:00+00:00",
        ]
    )

    endpoints = streaming_endpoint_positions_from_features(index, sequence_length=3)

    assert endpoints.tolist() == [2, 5]
