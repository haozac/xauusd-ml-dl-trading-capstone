from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "deployment"
    / "run_model_a_exit_safe_policy_comparison.py"
)
HELPER_NAMES = {
    "_reason_column",
    "_diagnostic_reason_at",
    "_diagnostic_count",
}


def _load_helpers() -> SimpleNamespace:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in HELPER_NAMES
    ]
    assert {node.name for node in helper_nodes} == HELPER_NAMES
    module = ast.Module(body=helper_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)
    return SimpleNamespace(
        reason_column=namespace["_reason_column"],
        diagnostic_reason_at=namespace["_diagnostic_reason_at"],
        diagnostic_count=namespace["_diagnostic_count"],
    )


def test_diagnostic_count_uses_current_change_reason_schema() -> None:
    helpers = _load_helpers()
    log = pd.DataFrame(
        {
            "change_reason": [
                "daily_change_cap_active",
                "hold",
                "daily_change_cap_active",
            ]
        }
    )

    assert helpers.reason_column(log) == "change_reason"
    assert helpers.diagnostic_count(log, "daily_change_cap_active") == 2
    assert helpers.diagnostic_reason_at(log, 0) == "daily_change_cap_active"


def test_diagnostic_helpers_support_legacy_blocked_reason_schema() -> None:
    helpers = _load_helpers()
    log = pd.DataFrame(
        {
            "blocked_reason": [
                "daily_change_cap_exit_allowed",
                None,
            ]
        }
    )

    assert helpers.reason_column(log) == "blocked_reason"
    assert helpers.diagnostic_count(
        log,
        "daily_change_cap_exit_allowed",
    ) == 1
    assert helpers.diagnostic_reason_at(log, 0) == (
        "daily_change_cap_exit_allowed"
    )
    assert helpers.diagnostic_reason_at(log, 1) == ""


def test_diagnostic_helpers_are_safe_when_no_reason_column_exists() -> None:
    helpers = _load_helpers()
    log = pd.DataFrame({"position": [0, 1]})

    assert helpers.reason_column(log) is None
    assert helpers.diagnostic_count(log, "daily_change_cap_active") == 0
    assert helpers.diagnostic_reason_at(log, 0) == ""
