#!/usr/bin/env python
"""Compare exit-safe Model A overlays with and without a daily change cap.

This script consumes only the already-frozen Notebook 7 prediction reference
files. It does not retrain a model, refit a scaler, alter thresholds, or change
live configuration. Its purpose is to test whether the existing daily policy-
change cap adds meaningful turnover control beyond the frozen M15 thresholds,
minimum hold, gap handling, and risk stops.

The uncapped replay uses a deliberately unreachable integer limit internally.
That keeps the frozen replay engine unchanged while making the comparison
explicit and deterministic.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

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
    ModelAOverlayRules,
    metrics_to_dict,
    overlay_rules_from_config,
    replay_model_a,
)


NO_CAP_REPLAY_LIMIT = 2_147_483_647
POLICY_CAP_ACTIVE_REASON = "daily_change_cap_active"
POLICY_CAP_EXIT_ALLOWED_REASON = "daily_change_cap_exit_allowed"
POLICY_CAP_CLOSE_ONLY_REASON = "daily_change_cap_close_only_reversal"


class PolicyCapNecessityComparisonError(RuntimeError):
    """Raised when the frozen-prediction comparison cannot be completed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exit-safe Model A with the frozen daily change cap "
            "against an exit-safe no-cap replay using frozen predictions only."
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
            "model_a_policy_cap_necessity_comparison.json"
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        default=(
            "runtime/reports/"
            "model_a_policy_cap_necessity_comparison_metrics.csv"
        ),
    )
    parser.add_argument(
        "--daily-activity-csv",
        default=(
            "runtime/reports/"
            "model_a_policy_cap_necessity_daily_activity.csv"
        ),
    )
    parser.add_argument(
        "--path-difference-csv",
        default=(
            "runtime/reports/"
            "model_a_policy_cap_necessity_path_differences.csv"
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = safe_repository_path(
        repository_root,
        str(detail["path"]),
        description=f"{partition} frozen prediction reference",
    )
    actual_hash = sha256_file(path)
    expected_hash = str(detail["sha256"]).lower()
    if actual_hash != expected_hash:
        raise PolicyCapNecessityComparisonError(
            f"Prediction hash mismatch for {partition}: expected "
            f"{expected_hash}, found {actual_hash}"
        )
    predictions = load_prediction_reference(path)
    expected_rows = int(detail.get("row_count", len(predictions)))
    if len(predictions) != expected_rows:
        raise PolicyCapNecessityComparisonError(
            f"Prediction row mismatch for {partition}: expected "
            f"{expected_rows}, found {len(predictions)}"
        )
    return predictions, {
        "path": str(path.relative_to(repository_root)),
        "sha256": actual_hash,
        "rows": int(len(predictions)),
    }


def _build_exit_safe_rules(
    frozen_rules: ModelAOverlayRules,
) -> tuple[ModelAOverlayRules, ModelAOverlayRules]:
    """Return cap-enabled and no-cap exit-safe rules.

    The frozen cap value is retained in the first variant. The no-cap variant
    uses an unreachable internal limit rather than changing the replay engine.
    """

    cap_rules = replace(
        frozen_rules,
        allow_risk_reducing_exit_when_capped=True,
    )
    no_cap_rules = replace(
        cap_rules,
        max_policy_changes_per_day=NO_CAP_REPLAY_LIMIT,
    )
    return cap_rules, no_cap_rules


def _reason_column(log: pd.DataFrame) -> str | None:
    if "change_reason" in log.columns:
        return "change_reason"
    if "blocked_reason" in log.columns:
        return "blocked_reason"
    return None


def _diagnostic_reason_at(log: pd.DataFrame, row_index: Any) -> str:
    column = _reason_column(log)
    if column is None:
        return ""
    value = log.at[row_index, column]
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _diagnostic_count(log: pd.DataFrame, reason: str) -> int:
    column = _reason_column(log)
    if column is None:
        return 0
    return int((log[column].fillna("").astype(str) == reason).sum())


def _numeric_deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for key in sorted(set(baseline) | set(candidate)):
        left = baseline.get(key)
        right = candidate.get(key)
        if (
            isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and not isinstance(left, bool)
            and not isinstance(right, bool)
        ):
            deltas[key] = float(right) - float(left)
        else:
            deltas[key] = None
    return deltas


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator) / float(denominator)


def _daily_activity(
    log: pd.DataFrame,
    *,
    partition: str,
    cost_bps: float,
    policy: str,
    frozen_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build per-UTC-day turnover diagnostics from a replay log."""

    required = {
        "time",
        "position_before",
        "position_after",
        "policy_event_units",
    }
    missing = sorted(required - set(log.columns))
    if missing:
        raise PolicyCapNecessityComparisonError(
            f"Replay log is missing daily-activity columns: {missing}"
        )

    frame = log.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    frame["utc_date"] = frame["time"].dt.strftime("%Y-%m-%d")
    before = frame["position_before"].astype(int)
    after = frame["position_after"].astype(int)
    frame["position_change_events"] = (before != after).astype(int)
    frame["entry_events"] = (
        (after != 0) & ((before == 0) | (before != after))
    ).astype(int)
    frame["exit_events"] = (
        (before != 0) & ((after == 0) | (before != after))
    ).astype(int)
    frame["reversal_events"] = (
        (before != 0) & (after != 0) & (before != after)
    ).astype(int)
    frame["active_bars"] = (after != 0).astype(int)
    turnover_column = (
        "turnover_units" if "turnover_units" in frame.columns else "turnover"
    )
    if turnover_column not in frame.columns:
        raise PolicyCapNecessityComparisonError(
            "Replay log is missing turnover_units/turnover"
        )

    grouped = frame.groupby("utc_date", sort=True, observed=True)
    rows: list[dict[str, Any]] = []
    for utc_date, daily in grouped:
        policy_units = int(daily["policy_event_units"].sum())
        turnover_units = float(daily[turnover_column].sum())
        rows.append(
            {
                "partition": partition,
                "cost_bps": float(cost_bps),
                "policy": policy,
                "utc_date": str(utc_date),
                "rows": int(len(daily)),
                "active_bars": int(daily["active_bars"].sum()),
                "position_change_events": int(
                    daily["position_change_events"].sum()
                ),
                "policy_change_events": policy_units,
                "turnover_units": turnover_units,
                "entry_events": int(daily["entry_events"].sum()),
                "exit_events": int(daily["exit_events"].sum()),
                "reversal_events": int(daily["reversal_events"].sum()),
                "exceeds_frozen_cap": bool(policy_units > frozen_cap),
            }
        )

    daily_frame = pd.DataFrame(rows)
    policy_series = daily_frame["policy_change_events"].astype(float)
    turnover_series = daily_frame["turnover_units"].astype(float)
    entry_series = daily_frame["entry_events"].astype(float)
    active_day_mask = policy_series > 0
    summary = {
        "observation_days": int(len(daily_frame)),
        "days_with_policy_changes": int(active_day_mask.sum()),
        "days_exceeding_frozen_cap": int(
            (policy_series > float(frozen_cap)).sum()
        ),
        "share_of_days_exceeding_frozen_cap": float(
            (policy_series > float(frozen_cap)).mean()
        ),
        "max_policy_change_events_in_utc_day": int(policy_series.max()),
        "mean_policy_change_events_per_day": float(policy_series.mean()),
        "median_policy_change_events_per_day": float(policy_series.median()),
        "p95_policy_change_events_per_day": float(
            policy_series.quantile(0.95)
        ),
        "max_turnover_units_in_utc_day": float(turnover_series.max()),
        "mean_turnover_units_per_day": float(turnover_series.mean()),
        "p95_turnover_units_per_day": float(
            turnover_series.quantile(0.95)
        ),
        "max_entry_events_in_utc_day": int(entry_series.max()),
        "mean_entry_events_per_day": float(entry_series.mean()),
        "total_entry_events": int(entry_series.sum()),
        "total_exit_events": int(daily_frame["exit_events"].sum()),
        "total_reversal_events": int(daily_frame["reversal_events"].sum()),
    }
    return rows, summary


def _necessity_indicators(
    *,
    cap_metrics: Mapping[str, Any],
    no_cap_metrics: Mapping[str, Any],
    cap_activity: Mapping[str, Any],
    no_cap_activity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "no_cap_minus_cap_net_total_return": (
            float(no_cap_metrics["net_total_return"])
            - float(cap_metrics["net_total_return"])
        ),
        "no_cap_minus_cap_max_drawdown": (
            float(no_cap_metrics["max_drawdown"])
            - float(cap_metrics["max_drawdown"])
        ),
        "no_cap_minus_cap_active_bar_rate": (
            float(no_cap_metrics["active_bar_rate"])
            - float(cap_metrics["active_bar_rate"])
        ),
        "no_cap_turnover_multiplier": _safe_ratio(
            float(no_cap_metrics["turnover_units"]),
            float(cap_metrics["turnover_units"]),
        ),
        "no_cap_round_turn_multiplier": _safe_ratio(
            float(no_cap_metrics["round_turn_equivalent_trades"]),
            float(cap_metrics["round_turn_equivalent_trades"]),
        ),
        "no_cap_days_exceeding_frozen_cap": int(
            no_cap_activity["days_exceeding_frozen_cap"]
        ),
        "no_cap_share_of_days_exceeding_frozen_cap": float(
            no_cap_activity["share_of_days_exceeding_frozen_cap"]
        ),
        "cap_max_policy_changes_in_day": int(
            cap_activity["max_policy_change_events_in_utc_day"]
        ),
        "no_cap_max_policy_changes_in_day": int(
            no_cap_activity["max_policy_change_events_in_utc_day"]
        ),
        "cap_max_entries_in_day": int(
            cap_activity["max_entry_events_in_utc_day"]
        ),
        "no_cap_max_entries_in_day": int(
            no_cap_activity["max_entry_events_in_utc_day"]
        ),
    }


def main() -> int:
    args = parse_args()
    repository_root = args.repo_root.expanduser().resolve()
    report_path = safe_repository_path(
        repository_root,
        args.report,
        description="policy-cap necessity comparison report",
        must_exist=False,
    )
    metrics_path = safe_repository_path(
        repository_root,
        args.metrics_csv,
        description="policy-cap necessity metrics CSV",
        must_exist=False,
    )
    daily_activity_path = safe_repository_path(
        repository_root,
        args.daily_activity_csv,
        description="policy-cap necessity daily activity CSV",
        must_exist=False,
    )
    path_difference_path = safe_repository_path(
        repository_root,
        args.path_difference_csv,
        description="policy-cap necessity path-difference CSV",
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
            "minimum_hold_changed": False,
            "risk_limits_changed": False,
            "live_configuration_changed": False,
            "prediction_sources": "existing frozen references only",
            "comparison_purpose": (
                "daily policy-cap necessity and turnover-control analysis; "
                "not post-hoc return optimisation"
            ),
        },
        "interpretation_guardrails": [
            "Do not choose a policy solely because it has the highest return.",
            "Assess turnover, entry frequency, cost sensitivity, exposure, and drawdown together.",
            "Normal and safety-driven exits remain allowed in every candidate policy.",
            "This script does not modify the live runtime configuration.",
        ],
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
        frozen_rules = overlay_rules_from_config(config.raw)
        cap_rules, no_cap_rules = _build_exit_safe_rules(frozen_rules)
        reference_manifest = load_step2_reference_manifest(
            repository_root,
            args.step2_manifest,
        )

        prediction_integrity: dict[str, Any] = {}
        metrics_rows: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        path_difference_rows: list[dict[str, Any]] = []
        comparison: dict[str, Any] = {}
        formal_checks: dict[str, bool] = {
            "frozen_cap_preserved_in_cap_variant": (
                cap_rules.max_policy_changes_per_day
                == frozen_rules.max_policy_changes_per_day
            ),
            "both_variants_allow_risk_reducing_exit_when_capped": (
                cap_rules.allow_risk_reducing_exit_when_capped
                and no_cap_rules.allow_risk_reducing_exit_when_capped
            ),
            "no_cap_internal_limit_is_above_frozen_cap": (
                no_cap_rules.max_policy_changes_per_day
                > frozen_rules.max_policy_changes_per_day
            ),
        }

        for partition in ("overlay_validation", "final_holdout"):
            detail = reference_manifest.prediction_references.get(partition)
            if not isinstance(detail, Mapping):
                raise PolicyCapNecessityComparisonError(
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
                cap_log, cap_metrics_obj = replay_model_a(
                    predictions,
                    cap_rules,
                    cost_bps=cost_bps,
                )
                no_cap_log, no_cap_metrics_obj = replay_model_a(
                    predictions,
                    no_cap_rules,
                    cost_bps=cost_bps,
                )
                cap_metrics = metrics_to_dict(cap_metrics_obj)
                no_cap_metrics = metrics_to_dict(no_cap_metrics_obj)

                cap_daily_rows, cap_activity = _daily_activity(
                    cap_log,
                    partition=partition,
                    cost_bps=cost_bps,
                    policy="exit_safe_cap",
                    frozen_cap=frozen_rules.max_policy_changes_per_day,
                )
                no_cap_daily_rows, no_cap_activity = _daily_activity(
                    no_cap_log,
                    partition=partition,
                    cost_bps=cost_bps,
                    policy="exit_safe_no_cap",
                    frozen_cap=frozen_rules.max_policy_changes_per_day,
                )
                daily_rows.extend(cap_daily_rows)
                daily_rows.extend(no_cap_daily_rows)

                no_cap_block_rows = _diagnostic_count(
                    no_cap_log,
                    POLICY_CAP_ACTIVE_REASON,
                )
                no_cap_capped_exit_rows = _diagnostic_count(
                    no_cap_log,
                    POLICY_CAP_EXIT_ALLOWED_REASON,
                )
                no_cap_close_only_rows = _diagnostic_count(
                    no_cap_log,
                    POLICY_CAP_CLOSE_ONLY_REASON,
                )
                check_prefix = f"{partition}_{cost_bps}"
                formal_checks[f"{check_prefix}_row_count_equal"] = bool(
                    len(cap_log) == len(no_cap_log) == len(predictions)
                )
                formal_checks[f"{check_prefix}_no_cap_has_no_cap_blocks"] = bool(
                    no_cap_block_rows == 0
                    and no_cap_capped_exit_rows == 0
                    and no_cap_close_only_rows == 0
                )

                path_diff_mask = (
                    cap_log["position"].astype(int)
                    != no_cap_log["position"].astype(int)
                )
                path_diff_count = int(path_diff_mask.sum())
                diagnostics = {
                    "position_path_difference_rows": path_diff_count,
                    "cap_block_rows": _diagnostic_count(
                        cap_log,
                        POLICY_CAP_ACTIVE_REASON,
                    ),
                    "cap_exit_allowed_rows": _diagnostic_count(
                        cap_log,
                        POLICY_CAP_EXIT_ALLOWED_REASON,
                    ),
                    "cap_close_only_reversal_rows": _diagnostic_count(
                        cap_log,
                        POLICY_CAP_CLOSE_ONLY_REASON,
                    ),
                    "no_cap_block_rows": no_cap_block_rows,
                    "no_cap_exit_allowed_rows": no_cap_capped_exit_rows,
                    "no_cap_close_only_reversal_rows": no_cap_close_only_rows,
                }
                indicators = _necessity_indicators(
                    cap_metrics=cap_metrics,
                    no_cap_metrics=no_cap_metrics,
                    cap_activity=cap_activity,
                    no_cap_activity=no_cap_activity,
                )
                comparison[partition][str(cost_bps)] = {
                    "exit_safe_cap": cap_metrics,
                    "exit_safe_no_cap": no_cap_metrics,
                    "no_cap_minus_cap": _numeric_deltas(
                        cap_metrics,
                        no_cap_metrics,
                    ),
                    "daily_activity": {
                        "exit_safe_cap": cap_activity,
                        "exit_safe_no_cap": no_cap_activity,
                    },
                    "necessity_indicators": indicators,
                    "diagnostics": diagnostics,
                }

                for policy_name, metrics in (
                    ("exit_safe_cap", cap_metrics),
                    ("exit_safe_no_cap", no_cap_metrics),
                ):
                    metrics_rows.append(
                        {
                            "partition": partition,
                            "cost_bps": float(cost_bps),
                            "policy": policy_name,
                            **metrics,
                        }
                    )

                for row_index in cap_log.index[path_diff_mask]:
                    path_difference_rows.append(
                        {
                            "partition": partition,
                            "cost_bps": float(cost_bps),
                            "time": cap_log.at[row_index, "time"],
                            "probability_up": cap_log.at[row_index, "p_up"],
                            "cap_position": int(
                                cap_log.at[row_index, "position"]
                            ),
                            "no_cap_position": int(
                                no_cap_log.at[row_index, "position"]
                            ),
                            "cap_change_reason": _diagnostic_reason_at(
                                cap_log,
                                row_index,
                            ),
                            "no_cap_change_reason": _diagnostic_reason_at(
                                no_cap_log,
                                row_index,
                            ),
                        }
                    )

        metric_fields = sorted(set().union(*(row.keys() for row in metrics_rows)))
        _write_csv_atomic(metrics_path, metrics_rows, metric_fields)
        _write_csv_atomic(
            daily_activity_path,
            daily_rows,
            [
                "partition",
                "cost_bps",
                "policy",
                "utc_date",
                "rows",
                "active_bars",
                "position_change_events",
                "policy_change_events",
                "turnover_units",
                "entry_events",
                "exit_events",
                "reversal_events",
                "exceeds_frozen_cap",
            ],
        )
        _write_csv_atomic(
            path_difference_path,
            path_difference_rows,
            [
                "partition",
                "cost_bps",
                "time",
                "probability_up",
                "cap_position",
                "no_cap_position",
                "cap_change_reason",
                "no_cap_change_reason",
            ],
        )

        formal_gate = all(formal_checks.values())
        report.update(
            {
                "status": "PASS" if formal_gate else "FAIL",
                "formal_gate": formal_gate,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "configuration_id": config.configuration_id,
                "prediction_integrity": prediction_integrity,
                "frozen_cap_value": int(
                    frozen_rules.max_policy_changes_per_day
                ),
                "rules": {
                    "exit_safe_cap": {
                        **cap_rules.__dict__,
                        "cap_mode": "frozen_limit_enabled",
                    },
                    "exit_safe_no_cap": {
                        **no_cap_rules.__dict__,
                        "max_policy_changes_per_day": None,
                        "cap_mode": "disabled_for_comparison",
                        "replay_internal_limit": NO_CAP_REPLAY_LIMIT,
                    },
                },
                "formal_checks": formal_checks,
                "comparison": comparison,
                "outputs": {
                    "metrics_csv": str(metrics_path.relative_to(repository_root)),
                    "daily_activity_csv": str(
                        daily_activity_path.relative_to(repository_root)
                    ),
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
    return 0 if report["formal_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
