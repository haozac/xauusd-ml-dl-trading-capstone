"""Pure position-transition policy shared by replay, simulation, and live runtime.

The helper deliberately separates a model's *desired* position from the safe
position that may actually be executed after the daily policy-change cap is
considered.  The exit-safe mode preserves the turnover cap for new exposure
while guaranteeing that the cap cannot trap an already-open position.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_POSITIONS = frozenset({-1, 0, 1})


class PositionTransitionPolicyError(ValueError):
    """Raised when a transition-policy input is malformed."""


@dataclass(frozen=True)
class PositionTransitionResolution:
    """Resolved target and audit facts for one requested position change."""

    current_position: int
    desired_position: int
    effective_target_position: int
    requested_policy_units: int
    consumed_policy_units: int
    cap_reached: bool
    entry_blocked: bool
    exit_allowed: bool
    close_only_reversal: bool
    action: str
    reason: str

    @property
    def transition_allowed(self) -> bool:
        return self.effective_target_position != self.current_position


def _validate_position(value: int, *, name: str) -> int:
    parsed = int(value)
    if parsed not in VALID_POSITIONS:
        raise PositionTransitionPolicyError(
            f"{name} must be -1, 0, or 1; found {value!r}"
        )
    return parsed


def policy_event_units(
    current_position: int,
    desired_position: int,
    *,
    reversal_policy_event_units: int = 1,
) -> int:
    """Return the configured policy units for a requested transition."""

    current = _validate_position(current_position, name="current_position")
    desired = _validate_position(desired_position, name="desired_position")
    reversal_units = int(reversal_policy_event_units)
    if reversal_units < 1:
        raise PositionTransitionPolicyError(
            "reversal_policy_event_units must be positive"
        )
    if current == desired:
        return 0
    if current != 0 and desired != 0:
        return reversal_units
    return 1


def resolve_position_transition(
    *,
    current_position: int,
    desired_position: int,
    policy_changes_today: int,
    max_policy_changes_per_day: int,
    reversal_policy_event_units: int = 1,
    allow_risk_reducing_exit_when_capped: bool,
) -> PositionTransitionResolution:
    """Resolve a requested Model A transition under the daily policy cap.

    In legacy mode, any transition that would exceed the cap is blocked.  This
    reproduces the original Notebook 7 overlay for parity checks.

    In exit-safe mode, a normal exit is always allowed.  A capped reversal is
    converted into a close-only transition: the existing position is closed,
    while the opposite-side entry remains blocked.  This keeps the cap useful
    for limiting new exposure without allowing it to trap an open position.
    """

    current = _validate_position(current_position, name="current_position")
    desired = _validate_position(desired_position, name="desired_position")
    used = int(policy_changes_today)
    maximum = int(max_policy_changes_per_day)
    if used < 0:
        raise PositionTransitionPolicyError(
            "policy_changes_today must be non-negative"
        )
    if maximum < 0:
        raise PositionTransitionPolicyError(
            "max_policy_changes_per_day must be non-negative"
        )

    requested_units = policy_event_units(
        current,
        desired,
        reversal_policy_event_units=reversal_policy_event_units,
    )
    if requested_units == 0:
        return PositionTransitionResolution(
            current_position=current,
            desired_position=desired,
            effective_target_position=current,
            requested_policy_units=0,
            consumed_policy_units=0,
            cap_reached=False,
            entry_blocked=False,
            exit_allowed=False,
            close_only_reversal=False,
            action="HOLD",
            reason="requested_position_matches_current_position",
        )

    exceeds_cap = used + requested_units > maximum
    if not exceeds_cap:
        return PositionTransitionResolution(
            current_position=current,
            desired_position=desired,
            effective_target_position=desired,
            requested_policy_units=requested_units,
            consumed_policy_units=requested_units,
            cap_reached=False,
            entry_blocked=False,
            exit_allowed=current != 0 and desired == 0,
            close_only_reversal=False,
            action="ALLOW_TRANSITION",
            reason="daily_policy_cap_not_exceeded",
        )

    if allow_risk_reducing_exit_when_capped and current != 0:
        if desired == 0:
            return PositionTransitionResolution(
                current_position=current,
                desired_position=desired,
                effective_target_position=0,
                requested_policy_units=requested_units,
                consumed_policy_units=1,
                cap_reached=True,
                entry_blocked=False,
                exit_allowed=True,
                close_only_reversal=False,
                action="ALLOW_EXIT_CAP_REACHED",
                reason="daily_policy_cap_cannot_block_risk_reducing_exit",
            )
        if desired == -current:
            return PositionTransitionResolution(
                current_position=current,
                desired_position=desired,
                effective_target_position=0,
                requested_policy_units=requested_units,
                consumed_policy_units=1,
                cap_reached=True,
                entry_blocked=True,
                exit_allowed=True,
                close_only_reversal=True,
                action="CLOSE_ONLY_REVERSAL_CAP_REACHED",
                reason=(
                    "daily_policy_cap_blocks_opposite_entry_but_allows_close"
                ),
            )

    return PositionTransitionResolution(
        current_position=current,
        desired_position=desired,
        effective_target_position=current,
        requested_policy_units=requested_units,
        consumed_policy_units=0,
        cap_reached=True,
        entry_blocked=desired != 0,
        exit_allowed=False,
        close_only_reversal=False,
        action="BLOCK_DAILY_POLICY_CAP",
        reason="maximum_overlay_position_changes_per_utc_day_reached",
    )
