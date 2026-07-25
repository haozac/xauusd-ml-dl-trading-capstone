from __future__ import annotations

import pandas as pd
import pytest

from capstone_trading.runtime.live_audit_analysis import (
    _same_event_disposition_counts,
)

EVENT_TIME = "2026-07-27T00:15:00+00:00"
FIRST_TIME = "2026-07-27T00:15:01+00:00"
LATER_TIME = "2026-07-27T00:18:00+00:00"


def _decision_pair(
    *,
    execution_mode: str = "live",
    first_action: str = "ENTER_LONG",
    first_target: int | float | None = 1,
    first_broker_before: int | float | None = 0,
    first_broker_after: int | float | None = 1,
    later_action: str = "DAILY_STOP_FLATTEN",
    later_position_before: int | float | None = 1,
    later_broker_before: int | float | None = 1,
    later_target: int | float | None = 0,
    later_broker_after: int | float | None = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "ordinary-decision",
                "event_time_utc": EVENT_TIME,
                "decision_utc": FIRST_TIME,
                "execution_mode": execution_mode,
                "action": first_action,
                "position_before": 0,
                "target_position": first_target,
                "broker_position_before": first_broker_before,
                "broker_position_after_inspection": first_broker_after,
            },
            {
                "decision_id": "safety-flatten",
                "event_time_utc": EVENT_TIME,
                "decision_utc": LATER_TIME,
                "execution_mode": execution_mode,
                "action": later_action,
                "position_before": later_position_before,
                "target_position": later_target,
                "broker_position_before": later_broker_before,
                "broker_position_after_inspection": later_broker_after,
            },
        ]
    )


def _assert_allowed(frame: pd.DataFrame) -> None:
    assert _same_event_disposition_counts(frame) == (1, 1, 0, 2)


def _assert_rejected(frame: pd.DataFrame) -> None:
    assert _same_event_disposition_counts(frame) == (1, 0, 1, 2)


@pytest.mark.parametrize(
    ("first_action", "position", "later_action"),
    [
        ("ENTER_LONG", 1, "DAILY_STOP_FLATTEN"),
        ("HOLD_SHORT", -1, "KILL_SWITCH_FLATTEN"),
        ("BLOCK_MINIMUM_HOLD", 1, "TOTAL_STOP_FLATTEN"),
        ("BLOCK_DAILY_POLICY_CAP", -1, "SESSION_GAP_LOCKOUT_FLATTEN"),
    ],
)
def test_consistent_known_exposed_decision_then_safety_flatten_is_allowed(
    first_action: str,
    position: int,
    later_action: str,
) -> None:
    _assert_allowed(
        _decision_pair(
            first_action=first_action,
            first_target=position,
            first_broker_after=position,
            later_action=later_action,
            later_position_before=position,
            later_broker_before=position,
        )
    )


def test_first_row_ending_flat_cannot_support_later_exposure_claim() -> None:
    _assert_rejected(
        _decision_pair(
            first_action="HOLD_FLAT",
            first_target=0,
            first_broker_after=0,
        )
    )


def test_missing_later_broker_after_evidence_is_rejected() -> None:
    _assert_rejected(_decision_pair(later_broker_after=None))


def test_invalid_later_position_before_is_rejected() -> None:
    _assert_rejected(
        _decision_pair(
            first_target=2,
            first_broker_after=2,
            later_position_before=2,
            later_broker_before=2,
        )
    )


def test_unknown_first_action_is_rejected() -> None:
    _assert_rejected(_decision_pair(first_action="UNKNOWN_ACTION"))


def test_control_action_cannot_be_the_preceding_ordinary_disposition() -> None:
    _assert_rejected(
        _decision_pair(first_action="CONTROL_MODEL_SNAPSHOT_MISMATCH_BLOCK")
    )


def test_first_ending_position_must_match_later_position_before() -> None:
    _assert_rejected(
        _decision_pair(
            first_action="ENTER_LONG",
            first_target=1,
            first_broker_after=1,
            later_position_before=-1,
            later_broker_before=-1,
        )
    )



def test_missing_first_broker_after_evidence_is_rejected() -> None:
    _assert_rejected(_decision_pair(first_broker_after=None))


def test_first_target_must_match_first_broker_ending_position() -> None:
    _assert_rejected(_decision_pair(first_target=-1, first_broker_after=1))

@pytest.mark.parametrize("first_action", ["BLOCK_SPREAD", "BLOCK_RECONCILIATION"])
def test_additional_exposure_preserving_actions_are_allowed(
    first_action: str,
) -> None:
    _assert_allowed(_decision_pair(first_action=first_action))


def test_change_position_is_not_a_recognised_preceding_action() -> None:
    _assert_rejected(_decision_pair(first_action="CHANGE_POSITION"))


@pytest.mark.parametrize("later_broker_before", [None, 0, -1])
def test_live_later_broker_before_must_match_exposure(
    later_broker_before: int | None,
) -> None:
    _assert_rejected(
        _decision_pair(
            execution_mode="live",
            later_broker_before=later_broker_before,
        )
    )


def test_shadow_virtual_exposure_with_flat_broker_is_allowed() -> None:
    _assert_allowed(
        _decision_pair(
            execution_mode="shadow",
            first_broker_before=0,
            first_broker_after=0,
            later_broker_before=0,
            later_broker_after=0,
        )
    )


@pytest.mark.parametrize(
    ("first_broker_before", "first_broker_after", "later_broker_before", "later_broker_after"),
    [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ],
)
def test_shadow_requires_explicitly_flat_broker_evidence(
    first_broker_before: int,
    first_broker_after: int,
    later_broker_before: int,
    later_broker_after: int,
) -> None:
    _assert_rejected(
        _decision_pair(
            execution_mode="shadow",
            first_broker_before=first_broker_before,
            first_broker_after=first_broker_after,
            later_broker_before=later_broker_before,
            later_broker_after=later_broker_after,
        )
    )


@pytest.mark.parametrize("first_mode,later_mode", [("live", "shadow"), ("", "live")])
def test_execution_modes_must_be_present_valid_and_equal(
    first_mode: str,
    later_mode: str,
) -> None:
    frame = _decision_pair(execution_mode=first_mode)
    frame.loc[1, "execution_mode"] = later_mode
    _assert_rejected(frame)

