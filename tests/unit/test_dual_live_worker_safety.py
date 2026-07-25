from __future__ import annotations

import pandas as pd

from capstone_trading.runtime.dual_live_worker import (
    historical_unseen_event_rows,
)


def test_historical_unseen_events_exclude_latest_executable_event() -> None:
    rows = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-07-24T10:00:00Z",
                    "2026-07-24T10:15:00Z",
                    "2026-07-24T10:30:00Z",
                    "2026-07-24T10:45:00Z",
                ],
                utc=True,
            ),
            "open": [1.0] * 4,
            "high": [1.0] * 4,
            "low": [1.0] * 4,
            "close": [1.0] * 4,
            "tick_volume": [1] * 4,
            "spread": [1] * 4,
            "real_volume": [0] * 4,
        }
    )

    backfills = historical_unseen_event_rows(
        rows,
        previous_event_time_utc="2026-07-24T10:00:00+00:00",
    )

    assert list(backfills["time"].dt.strftime("%H:%M")) == ["10:15", "10:30"]
    assert "10:45" not in set(backfills["time"].dt.strftime("%H:%M"))


def test_model_snapshot_coherence_rejects_two_fetch_bar_boundary_race() -> None:
    from types import SimpleNamespace

    from capstone_trading.runtime.dual_live_worker import (
        model_snapshot_coherence,
    )

    signal = SimpleNamespace(
        event_time_utc="2026-07-24T10:15:00+00:00",
        latest_completed_bar_time_utc="2026-07-24T10:15:00+00:00",
    )

    coherent, details = model_snapshot_coherence(
        latest_completed_event_time_utc="2026-07-24T10:00:00+00:00",
        signal=signal,
    )

    assert coherent is False
    assert details["broker_event_time_utc"] == "2026-07-24T10:00:00+00:00"
    assert details["signal_event_time_utc"] == "2026-07-24T10:15:00+00:00"
    assert (
        details["signal_latest_completed_bar_time_utc"]
        == "2026-07-24T10:15:00+00:00"
    )


def test_model_snapshot_coherence_accepts_one_completed_event() -> None:
    from types import SimpleNamespace

    from capstone_trading.runtime.dual_live_worker import (
        model_snapshot_coherence,
    )

    signal = SimpleNamespace(
        event_time_utc="2026-07-24T10:00:00Z",
        latest_completed_bar_time_utc="2026-07-24T10:00:00+00:00",
    )

    coherent, _ = model_snapshot_coherence(
        latest_completed_event_time_utc="2026-07-24T10:00:00+00:00",
        signal=signal,
    )

    assert coherent is True


def test_unseen_completed_event_rows_adopts_only_latest_on_fresh_root() -> None:
    from capstone_trading.runtime.dual_live_worker import (
        unseen_completed_event_rows,
    )

    rows = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-07-24T20:30:00Z",
                    "2026-07-24T20:45:00Z",
                ],
                utc=True,
            )
        }
    )

    unseen = unseen_completed_event_rows(
        rows,
        previous_event_time_utc=None,
    )

    assert list(unseen["time"].dt.strftime("%H:%M")) == ["20:45"]
