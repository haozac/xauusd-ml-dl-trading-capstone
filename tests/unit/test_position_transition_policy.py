from __future__ import annotations

from capstone_trading.policy.position_transition import resolve_position_transition


def resolve(current: int, desired: int, used: int, *, safe: bool = True):
    return resolve_position_transition(
        current_position=current,
        desired_position=desired,
        policy_changes_today=used,
        max_policy_changes_per_day=3,
        reversal_policy_event_units=1,
        allow_risk_reducing_exit_when_capped=safe,
    )


def test_flat_entry_is_blocked_when_cap_reached() -> None:
    result = resolve(0, 1, 3)
    assert result.effective_target_position == 0
    assert result.entry_blocked is True
    assert result.consumed_policy_units == 0


def test_exit_is_allowed_when_cap_reached() -> None:
    result = resolve(1, 0, 3)
    assert result.effective_target_position == 0
    assert result.exit_allowed is True
    assert result.entry_blocked is False
    assert result.consumed_policy_units == 1


def test_reversal_becomes_close_only_when_cap_reached() -> None:
    result = resolve(1, -1, 3)
    assert result.effective_target_position == 0
    assert result.close_only_reversal is True
    assert result.entry_blocked is True
    assert result.exit_allowed is True


def test_legacy_mode_reproduces_original_blocking() -> None:
    result = resolve(1, 0, 3, safe=False)
    assert result.effective_target_position == 1
    assert result.transition_allowed is False
    assert result.action == "BLOCK_DAILY_POLICY_CAP"
