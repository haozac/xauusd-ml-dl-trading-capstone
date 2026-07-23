#!/usr/bin/env python
"""Compare legacy and exit-safe Model A overlays without retraining.

The script consumes the already-frozen prediction reference files.  It does not
load TensorFlow, refit a scaler, retrain a model, change thresholds, or select a
new policy based on profitability.  Its purpose is to quantify the consequence
of the engineering correction that prevents a daily turnover cap from trapping
an open position.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from capstone_trading.artifacts import (
    sha256_file,
    verify_notebook7_artifact_bundle,
    verify_stage0_freeze_manifest,
)
from capstone_trading.config import load_model_a_config, safe_repository_path
from capstone_trading.data.historical_adapter import (
    load_prediction_reference,
    load_step2_reference_manifest,
)
from capstone_trading.evaluation.trading_replay import (
    DEFAULT_COSTS_BPS,
    metrics_to_dict,
    overlay_rules_from_config,
    replay_model_a,
)


class ExitSafePolicyComparisonError(RuntimeError):
    """Raised when the frozen-prediction comparison cannot be completed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen legacy Model A overlay with the exit-safe "
            "daily-cap correction using existing prediction references only."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument(
        "--freeze-manifest",
        default="config/stage0_freeze_manifest.json",
    )
    parser.add_argument(
        "--step2-manifest",
        default="config/stage1_step2_reference_manifest.json",
    )
    parser.add_argument(
        "--report",
        default=(
            "runtime/reports/"
            "model_a_exit_safe_policy_comparison.json"
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        default=(
            "runtime/reports/"
            "model_a_exit_safe_policy_comparison_metrics.csv"
        ),
    )
    parser.add_argument(
        "--path-difference-csv",
        default=(
            "runtime/reports/"
            "model_a_exit_safe_policy_path_differences.csv"
        ),
    )
    return parser.parse_args()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    rows: list[Mapping[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_prediction_reference(
    *,
    repository_root: Path,
    partition: str,
    detail: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    path = safe_repository_path(
        repository_root,
        str(detail["path"]),
        description=f"{partition} frozen prediction reference",
    )
    actual_hash = sha256_file(path)
    expected_hash = str(detail["sha256"]).lower()
    if actual_hash != expected_hash:
        raise ExitSafePolicyComparisonError(
            f"Prediction hash mismatch for {partition}: expected "
            f"{expected_hash}, found {actual_hash}"
        )
    predictions = load_prediction_reference(path)
    expected_rows = int(detail.get("row_count", len(predictions)))
    if len(predictions) != expected_rows:
        raise ExitSafePolicyComparisonError(
            f"Prediction row mismatch for {partition}: expected "
            f"{expected_rows}, found {len(predictions)}"
        )
    return predictions, {
        "path": str(path.relative_to(repository_root)),
        "sha256": actual_hash,
        "rows": int(len(predictions)),
    }


def _reason_column(log: Any) -> str | None:
    """Return the action-reason column exposed by a replay log.

    Current replay logs expose the normalised reason as ``change_reason``.
    Older diagnostic frames may expose the pre-normalised name
    ``blocked_reason``. Supporting both schemas keeps this comparison script
    compatible with frozen references and avoids hiding cap diagnostics.
    """

    columns = getattr(log, "columns", ())
    if "change_reason" in columns:
        return "change_reason"
    if "blocked_reason" in columns:
        return "blocked_reason"
    return None


def _diagnostic_reason_at(log: Any, row_index: Any) -> str:
    column = _reason_column(log)
    if column is None:
        return ""
    value = log.at[row_index, column]
    if value is None:
        return ""
    try:
        if bool(value != value):  # NaN without importing another dependency.
            return ""
    except Exception:
        pass
    return str(value)


def _diagnostic_count(log: Any, reason: str) -> int:
    column = _reason_column(log)
    if column is None:
        return 0
    return int((log[column].fillna("").astype(str) == reason).sum())


def main() -> int:
    args = parse_args()
    repository_root = args.repo_root.expanduser().resolve()
    report_path = safe_repository_path(
        repository_root,
        args.report,
        description="exit-safe policy comparison report",
        must_exist=False,
    )
    metrics_path = safe_repository_path(
        repository_root,
        args.metrics_csv,
        description="exit-safe policy comparison metrics CSV",
        must_exist=False,
    )
    path_difference_path = safe_repository_path(
        repository_root,
        args.path_difference_csv,
        description="exit-safe policy path-difference CSV",
        must_exist=False,
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "RUNNING",
        "formal_gate": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model_retraining": False,
            "scaler_refitting": False,
            "threshold_tuning": False,
            "daily_cap_value_changed": False,
            "prediction_sources": "existing frozen references only",
            "comparison_purpose": (
                "engineering consistency and consequence analysis; not "
                "post-hoc strategy selection"
            ),
        },
    }
    try:
        config_path = safe_repository_path(
            repository_root,
            args.model_a_config,
            description="frozen Model A configuration",
        )
        config = load_model_a_config(config_path)
        verify_stage0_freeze_manifest(repository_root, args.freeze_manifest)
        verify_notebook7_artifact_bundle(repository_root, config)
        legacy_rules = overlay_rules_from_config(config.raw)
        exit_safe_rules = replace(
            legacy_rules,
            allow_risk_reducing_exit_when_capped=True,
        )
        reference_manifest = load_step2_reference_manifest(
            repository_root,
            args.step2_manifest,
        )

        prediction_integrity: dict[str, Any] = {}
        metrics_rows: list[dict[str, Any]] = []
        path_difference_rows: list[dict[str, Any]] = []
        comparison: dict[str, Any] = {}

        for partition in ("overlay_validation", "final_holdout"):
            detail = reference_manifest.prediction_references.get(partition)
            if not isinstance(detail, Mapping):
                raise ExitSafePolicyComparisonError(
                    f"Missing prediction reference for {partition}"
                )
            predictions, integrity = _load_prediction_reference(
                repository_root=repository_root,
                partition=partition,
                detail=detail,
            )
            prediction_integrity[partition] = integrity
            comparison[partition] = {}

            for cost_bps in DEFAULT_COSTS_BPS:
                legacy_log, legacy_metrics = replay_model_a(
                    predictions,
                    legacy_rules,
                    cost_bps=cost_bps,
                )
                safe_log, safe_metrics = replay_model_a(
                    predictions,
                    exit_safe_rules,
                    cost_bps=cost_bps,
                )
                legacy_map = metrics_to_dict(legacy_metrics)
                safe_map = metrics_to_dict(safe_metrics)
                deltas = {
                    key: (
                        float(safe_map[key]) - float(legacy_map[key])
                        if isinstance(legacy_map.get(key), (int, float))
                        and isinstance(safe_map.get(key), (int, float))
                        and not isinstance(legacy_map.get(key), bool)
                        and not isinstance(safe_map.get(key), bool)
                        else None
                    )
                    for key in sorted(set(legacy_map) | set(safe_map))
                }
                path_diff_mask = (
                    legacy_log["position"].astype(int)
                    != safe_log["position"].astype(int)
                )
                path_diff_count = int(path_diff_mask.sum())
                diagnostics = {
                    "position_path_difference_rows": path_diff_count,
                    "legacy_cap_block_rows": _diagnostic_count(
                        legacy_log, "daily_change_cap_active"
                    ),
                    "exit_safe_cap_block_rows": _diagnostic_count(
                        safe_log, "daily_change_cap_active"
                    ),
                    "exit_safe_capped_exit_rows": _diagnostic_count(
                        safe_log, "daily_change_cap_exit_allowed"
                    ),
                    "exit_safe_close_only_reversal_rows": _diagnostic_count(
                        safe_log, "daily_change_cap_close_only_reversal"
                    ),
                }
                comparison[partition][str(cost_bps)] = {
                    "legacy": legacy_map,
                    "exit_safe": safe_map,
                    "exit_safe_minus_legacy": deltas,
                    "diagnostics": diagnostics,
                }
                for policy_name, metrics in (
                    ("legacy", legacy_map),
                    ("exit_safe", safe_map),
                ):
                    metrics_rows.append(
                        {
                            "partition": partition,
                            "cost_bps": cost_bps,
                            "policy": policy_name,
                            **metrics,
                        }
                    )
                for row_index in legacy_log.index[path_diff_mask]:
                    path_difference_rows.append(
                        {
                            "partition": partition,
                            "cost_bps": cost_bps,
                            "time": legacy_log.at[row_index, "time"],
                            "probability_up": legacy_log.at[row_index, "p_up"],
                            "legacy_position": int(
                                legacy_log.at[row_index, "position"]
                            ),
                            "exit_safe_position": int(
                                safe_log.at[row_index, "position"]
                            ),
                            "legacy_change_reason": _diagnostic_reason_at(
                                legacy_log, row_index
                            ),
                            "exit_safe_change_reason": _diagnostic_reason_at(
                                safe_log, row_index
                            ),
                        }
                    )

        metric_fields = sorted(
            set().union(*(row.keys() for row in metrics_rows))
        )
        _write_csv_atomic(metrics_path, metrics_rows, metric_fields)
        _write_csv_atomic(
            path_difference_path,
            path_difference_rows,
            [
                "partition",
                "cost_bps",
                "time",
                "probability_up",
                "legacy_position",
                "exit_safe_position",
                "legacy_change_reason",
                "exit_safe_change_reason",
            ],
        )
        report.update(
            {
                "status": "PASS",
                "formal_gate": True,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "configuration_id": config.configuration_id,
                "prediction_integrity": prediction_integrity,
                "rules": {
                    "legacy": {
                        **legacy_rules.__dict__,
                    },
                    "exit_safe": {
                        **exit_safe_rules.__dict__,
                    },
                },
                "comparison": comparison,
                "outputs": {
                    "metrics_csv": str(metrics_path.relative_to(repository_root)),
                    "path_difference_csv": str(
                        path_difference_path.relative_to(repository_root)
                    ),
                },
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "FAIL",
                "formal_gate": False,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json_atomic(report_path, report)
        raise

    _write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
