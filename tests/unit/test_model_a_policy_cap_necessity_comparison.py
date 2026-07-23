from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "deployment"
    / "run_model_a_policy_cap_necessity_comparison.py"
)
HELPER_NAMES = {
    "_build_exit_safe_rules",
    "_reason_column",
    "_diagnostic_reason_at",
    "_diagnostic_count",
    "_numeric_deltas",
    "_safe_ratio",
    "_daily_activity",
    "_necessity_indicators",
}


@dataclass(frozen=True)
class FakeRules:
    long_threshold: float = 0.53
    short_threshold: float = 0.47
    minimum_hold_bars: int = 3
    max_policy_changes_per_day: int = 3
    daily_loss_log_threshold: float = -0.02
    total_drawdown_stop: float = -0.15
    count_gap_exits_against_cap: bool = False
    count_risk_exits_against_cap: bool = False
    reversal_policy_event_units: int = 1
    allow_risk_reducing_exit_when_capped: bool = False


class FakeComparisonError(RuntimeError):
    pass


def _load_helpers() -> SimpleNamespace:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    constant_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id
            in {
                "NO_CAP_REPLAY_LIMIT",
                "POLICY_CAP_ACTIVE_REASON",
                "POLICY_CAP_EXIT_ALLOWED_REASON",
                "POLICY_CAP_CLOSE_ONLY_REASON",
            }
            for target in node.targets
        )
    ]
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in HELPER_NAMES
    ]
    assert {node.name for node in helper_nodes} == HELPER_NAMES
    module = ast.Module(body=constant_nodes + helper_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "ModelAOverlayRules": FakeRules,
        "PolicyCapNecessityComparisonError": FakeComparisonError,
        "pd": pd,
        "replace": replace,
    }
    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)
    return SimpleNamespace(
        build_rules=namespace["_build_exit_safe_rules"],
        reason_column=namespace["_reason_column"],
        diagnostic_reason_at=namespace["_diagnostic_reason_at"],
        diagnostic_count=namespace["_diagnostic_count"],
        numeric_deltas=namespace["_numeric_deltas"],
        safe_ratio=namespace["_safe_ratio"],
        daily_activity=namespace["_daily_activity"],
        necessity_indicators=namespace["_necessity_indicators"],
        no_cap_limit=namespace["NO_CAP_REPLAY_LIMIT"],
    )


def _sample_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:15:00+00:00",
                "2026-01-01T00:30:00+00:00",
                "2026-01-02T00:00:00+00:00",
            ],
            "position_before": [0, 1, 0, -1],
            "position_after": [1, 0, -1, 1],
            "policy_event_units": [1, 1, 1, 1],
            "turnover_units": [1.0, 1.0, 1.0, 2.0],
            "change_reason": [
                "normal_policy_change",
                "daily_change_cap_exit_allowed",
                "normal_policy_change",
                "daily_change_cap_close_only_reversal",
            ],
        }
    )


def test_build_exit_safe_rules_preserves_cap_and_builds_no_cap_variant() -> None:
    helpers = _load_helpers()
    frozen = FakeRules()

    cap_rules, no_cap_rules = helpers.build_rules(frozen)

    assert cap_rules.max_policy_changes_per_day == 3
    assert cap_rules.allow_risk_reducing_exit_when_capped is True
    assert no_cap_rules.max_policy_changes_per_day == helpers.no_cap_limit
    assert no_cap_rules.allow_risk_reducing_exit_when_capped is True
    assert frozen.allow_risk_reducing_exit_when_capped is False


def test_daily_activity_counts_entries_exits_reversals_and_cap_excess() -> None:
    helpers = _load_helpers()

    rows, summary = helpers.daily_activity(
        _sample_log(),
        partition="final_holdout",
        cost_bps=0.5,
        policy="exit_safe_no_cap",
        frozen_cap=2,
    )

    assert len(rows) == 2
    assert rows[0]["policy_change_events"] == 3
    assert rows[0]["entry_events"] == 2
    assert rows[0]["exit_events"] == 1
    assert rows[0]["reversal_events"] == 0
    assert rows[0]["exceeds_frozen_cap"] is True
    assert rows[1]["entry_events"] == 1
    assert rows[1]["exit_events"] == 1
    assert rows[1]["reversal_events"] == 1
    assert rows[1]["turnover_units"] == 2.0
    assert summary["observation_days"] == 2
    assert summary["days_exceeding_frozen_cap"] == 1
    assert summary["max_policy_change_events_in_utc_day"] == 3
    assert summary["max_entry_events_in_utc_day"] == 2
    assert summary["total_reversal_events"] == 1


def test_daily_activity_rejects_incomplete_replay_schema() -> None:
    helpers = _load_helpers()
    incomplete = pd.DataFrame({"time": ["2026-01-01T00:00:00+00:00"]})

    try:
        helpers.daily_activity(
            incomplete,
            partition="overlay_validation",
            cost_bps=0.0,
            policy="exit_safe_cap",
            frozen_cap=3,
        )
    except FakeComparisonError as exc:
        assert "missing daily-activity columns" in str(exc)
    else:
        raise AssertionError("Expected incomplete replay schema to fail")


def test_diagnostic_helpers_support_both_reason_schemas() -> None:
    helpers = _load_helpers()
    current = pd.DataFrame(
        {"change_reason": ["daily_change_cap_active", None]}
    )
    legacy = pd.DataFrame(
        {"blocked_reason": ["daily_change_cap_exit_allowed"]}
    )

    assert helpers.reason_column(current) == "change_reason"
    assert helpers.diagnostic_count(current, "daily_change_cap_active") == 1
    assert helpers.diagnostic_reason_at(current, 1) == ""
    assert helpers.reason_column(legacy) == "blocked_reason"
    assert helpers.diagnostic_reason_at(legacy, 0) == (
        "daily_change_cap_exit_allowed"
    )


def test_numeric_helpers_handle_boolean_and_zero_denominator() -> None:
    helpers = _load_helpers()

    deltas = helpers.numeric_deltas(
        {"return": 1.0, "triggered": False, "label": "a"},
        {"return": 1.25, "triggered": True, "label": "b"},
    )

    assert deltas["return"] == 0.25
    assert deltas["triggered"] is None
    assert deltas["label"] is None
    assert helpers.safe_ratio(2.0, 0.0) is None
    assert helpers.safe_ratio(6.0, 3.0) == 2.0


def test_necessity_indicators_report_turnover_and_daily_cap_pressure() -> None:
    helpers = _load_helpers()

    indicators = helpers.necessity_indicators(
        cap_metrics={
            "net_total_return": -0.1,
            "max_drawdown": -0.15,
            "active_bar_rate": 0.2,
            "turnover_units": 100.0,
            "round_turn_equivalent_trades": 50.0,
        },
        no_cap_metrics={
            "net_total_return": -0.08,
            "max_drawdown": -0.12,
            "active_bar_rate": 0.4,
            "turnover_units": 250.0,
            "round_turn_equivalent_trades": 125.0,
        },
        cap_activity={
            "max_policy_change_events_in_utc_day": 3,
            "max_entry_events_in_utc_day": 2,
        },
        no_cap_activity={
            "days_exceeding_frozen_cap": 10,
            "share_of_days_exceeding_frozen_cap": 0.25,
            "max_policy_change_events_in_utc_day": 8,
            "max_entry_events_in_utc_day": 4,
        },
    )

    assert indicators["no_cap_turnover_multiplier"] == 2.5
    assert indicators["no_cap_round_turn_multiplier"] == 2.5
    assert indicators["no_cap_days_exceeding_frozen_cap"] == 10
    assert indicators["no_cap_max_policy_changes_in_day"] == 8
    assert indicators["no_cap_max_entries_in_day"] == 4
