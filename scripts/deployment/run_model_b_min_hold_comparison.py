#!/usr/bin/env python
"""Stage 2 Step 2C diagnostic comparison for Model B minimum hold.

This script compares the already frozen Model B V2 long-only overlay against one
pre-declared candidate execution refinement: the same overlay plus a three
completed-M15-bar minimum hold before normal probability-based exits.

It is diagnostic only.  It does not retrain, tune thresholds, touch MT5, place
orders, or create a new untouched holdout claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from capstone_trading.artifacts import (
    sha256_file,
    verify_notebook7_artifact_bundle,
    verify_stage0_freeze_manifest,
)
from capstone_trading.config import load_model_a_config, load_yaml_mapping, safe_repository_path
from capstone_trading.data.historical_adapter import (
    load_prediction_reference,
    load_step2_reference_manifest,
)
from capstone_trading.errors import (
    IntegrityError,
    Step1VerificationError,
    Step2ParityError,
    Step4TradingReplayError,
    TradingReplayError,
)
from capstone_trading.evaluation.model_b_min_hold import (
    create_min_hold_rules,
    diagnostics_to_dict as min_hold_diagnostics_to_dict,
    model_b_min_hold_metrics_row,
    replay_model_b_min_hold,
)
from capstone_trading.evaluation.model_b_replay import (
    diagnostics_to_dict as current_diagnostics_to_dict,
    overlay_rules_from_model_b_config,
    replay_model_b,
)
from capstone_trading.evaluation.trading_replay import ReplayMetrics, metrics_to_dict

LOGGER = logging.getLogger("stage2_step2c")
DEFAULT_COSTS_BPS: tuple[float, ...] = (0.0, 0.5, 1.0)
CANDIDATE_MINIMUM_HOLD_BARS = 3
PARTITIONS: tuple[str, ...] = ("overlay_validation", "final_holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current Model B V2 against Model B-MH, a diagnostic candidate "
            "that adds a fixed three-bar minimum hold before normal probability exits."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument("--model-b-config", default="config/model_b_v2_frozen.yaml")
    parser.add_argument("--freeze-manifest", default="config/stage0_freeze_manifest.json")
    parser.add_argument("--step2-manifest", default="config/stage1_step2_reference_manifest.json")
    parser.add_argument(
        "--stage1-step5-report",
        default="runtime/reports/stage1_step5_model_b_diagnostic_replay.json",
        help="Precondition report proving the current Model B diagnostic replay already passed.",
    )
    parser.add_argument(
        "--stage2-step2a-report",
        default="runtime/reports/stage2_step2a_v1_1_mt5_shadow_logger.json",
        help="Precondition report proving timestamp-corrected live shadow mode already passed.",
    )
    parser.add_argument(
        "--report",
        default="runtime/reports/stage2_step2c_model_b_min_hold_comparison.json",
    )
    parser.add_argument(
        "--metrics-csv",
        default="runtime/reports/stage2_step2c_model_b_variant_metrics_by_cost.csv",
    )
    parser.add_argument(
        "--comparison-csv",
        default="runtime/reports/stage2_step2c_model_b_min_hold_comparison.csv",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_json_file(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TradingReplayError(f"Unable to read JSON file {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TradingReplayError(f"JSON file must contain an object: {path}")
    return payload


def verify_stage1_step5_report(repository_root: Path, raw_path: str) -> dict[str, Any]:
    path = safe_repository_path(
        repository_root,
        raw_path,
        description="Stage 1 Step 5 current Model B diagnostic report",
    )
    payload = load_json_file(path)
    if payload.get("status") != "PASS" or payload.get("formal_gate") is not True:
        raise TradingReplayError("Stage 2 Step 2C requires the Stage 1 Step 5 Model B diagnostic report to pass")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise TradingReplayError("Stage 1 Step 5 report is missing checks")
    frozen_contract = checks.get("frozen_contract")
    invariants = checks.get("model_b_invariants")
    if not isinstance(frozen_contract, Mapping) or frozen_contract.get("passed") is not True:
        raise TradingReplayError("Stage 1 Step 5 report does not prove the frozen Model B contract")
    if not isinstance(invariants, Mapping) or invariants.get("passed") is not True:
        raise TradingReplayError("Stage 1 Step 5 report does not prove current Model B invariants")
    return {
        "path": str(path.relative_to(repository_root)),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "formal_gate": payload.get("formal_gate"),
    }


def verify_stage2_step2a_report(repository_root: Path, raw_path: str) -> dict[str, Any]:
    path = safe_repository_path(
        repository_root,
        raw_path,
        description="Stage 2 Step 2A v1.1 live shadow report",
    )
    payload = load_json_file(path)
    if payload.get("status") != "PASS" or payload.get("formal_gate") is not True:
        raise TradingReplayError("Stage 2 Step 2C requires the Stage 2 Step 2A v1.1 shadow report to pass")
    if payload.get("shadow_only") is not True or payload.get("orders_enabled") is not False:
        raise TradingReplayError("Stage 2 Step 2A report must prove shadow-only no-order execution")
    safety = payload.get("safety")
    if not isinstance(safety, Mapping) or safety.get("order_send_called") is not False:
        raise TradingReplayError("Stage 2 Step 2A report does not prove order_send was blocked")
    time_norm = payload.get("time_normalisation")
    if not isinstance(time_norm, Mapping):
        raise TradingReplayError("Stage 2 Step 2A report is missing time_normalisation")
    if time_norm.get("conversion_applied") is not True:
        raise TradingReplayError("Stage 2 Step 2A v1.1 must apply broker-server-time to UTC conversion")
    future_minutes = float(time_norm.get("latest_bar_future_minutes_after_conversion", 999.0))
    if future_minutes > 0.0:
        raise TradingReplayError(
            "Stage 2 Step 2A latest canonical bar is still in the future after conversion: "
            f"{future_minutes} minutes"
        )
    return {
        "path": str(path.relative_to(repository_root)),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "formal_gate": payload.get("formal_gate"),
        "latest_event_utc": payload.get("latest_signal", {}).get("event_time_utc"),
        "mt5_server_time_offset_hours": time_norm.get("mt5_server_time_offset_hours"),
        "latest_bar_age_minutes_after_conversion": time_norm.get("latest_bar_age_minutes_after_conversion"),
    }


def verify_prediction_reference(
    *,
    repository_root: Path,
    name: str,
    detail: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = safe_repository_path(
        repository_root,
        str(detail["path"]),
        description=f"{name} prediction reference",
    )
    actual_hash = sha256_file(path)
    expected_hash = str(detail["sha256"]).lower()
    if actual_hash != expected_hash:
        raise IntegrityError(
            f"Prediction reference hash mismatch for {name}: expected {expected_hash}, found {actual_hash}"
        )
    predictions = load_prediction_reference(path)
    expected_rows = int(detail.get("row_count", len(predictions)))
    if len(predictions) != expected_rows:
        raise TradingReplayError(
            f"{name} prediction row count mismatch: expected {expected_rows}, found {len(predictions)}"
        )
    return predictions, {
        "path": str(path.relative_to(repository_root)),
        "sha256": actual_hash,
        "rows": int(len(predictions)),
    }


def cost_key(cost_bps: float) -> str:
    return f"{float(cost_bps):.1f}"


def transaction_cost_burden(metrics: ReplayMetrics) -> float:
    return float(metrics.final_gross_equity - metrics.final_net_equity)


def current_variant_metrics_row(
    *,
    partition: str,
    cost_bps: float,
    metrics: ReplayMetrics,
    diagnostics: Any,
) -> dict[str, Any]:
    return {
        "variant": "MODEL_B_V2_CURRENT",
        "minimum_hold_bars": 0,
        "partition": partition,
        "cost_bps": float(cost_bps),
        **metrics_to_dict(metrics),
        **current_diagnostics_to_dict(diagnostics),
        "min_hold_blocked_exit_count": 0,
        "active_below_exit_threshold_after_eligible_count": diagnostics.active_below_exit_threshold_count,
        "max_hold_bars_completed": "",
    }


def comparison_row(
    *,
    partition: str,
    cost_bps: float,
    current_metrics: ReplayMetrics,
    current_diagnostics: Any,
    min_hold_metrics: ReplayMetrics,
    min_hold_diagnostics: Any,
) -> dict[str, Any]:
    current_cost_burden = transaction_cost_burden(current_metrics)
    min_hold_cost_burden = transaction_cost_burden(min_hold_metrics)
    return {
        "partition": partition,
        "cost_bps": float(cost_bps),
        "current_net_total_return": float(current_metrics.net_total_return),
        "min_hold_net_total_return": float(min_hold_metrics.net_total_return),
        "delta_net_total_return_min_hold_minus_current": float(
            min_hold_metrics.net_total_return - current_metrics.net_total_return
        ),
        "current_gross_total_return": float(current_metrics.gross_total_return),
        "min_hold_gross_total_return": float(min_hold_metrics.gross_total_return),
        "delta_gross_total_return_min_hold_minus_current": float(
            min_hold_metrics.gross_total_return - current_metrics.gross_total_return
        ),
        "current_max_drawdown": float(current_metrics.max_drawdown),
        "min_hold_max_drawdown": float(min_hold_metrics.max_drawdown),
        "drawdown_delta_positive_means_less_severe": float(
            min_hold_metrics.max_drawdown - current_metrics.max_drawdown
        ),
        "current_turnover_units": float(current_metrics.turnover_units),
        "min_hold_turnover_units": float(min_hold_metrics.turnover_units),
        "turnover_reduction_units": float(current_metrics.turnover_units - min_hold_metrics.turnover_units),
        "current_round_turn_equivalent_trades": float(current_metrics.round_turn_equivalent_trades),
        "min_hold_round_turn_equivalent_trades": float(min_hold_metrics.round_turn_equivalent_trades),
        "trade_reduction_round_turn_equivalent": float(
            current_metrics.round_turn_equivalent_trades - min_hold_metrics.round_turn_equivalent_trades
        ),
        "current_active_bar_rate": float(current_metrics.active_bar_rate),
        "min_hold_active_bar_rate": float(min_hold_metrics.active_bar_rate),
        "active_bar_rate_delta": float(min_hold_metrics.active_bar_rate - current_metrics.active_bar_rate),
        "current_transaction_cost_burden": current_cost_burden,
        "min_hold_transaction_cost_burden": min_hold_cost_burden,
        "transaction_cost_burden_reduction": float(current_cost_burden - min_hold_cost_burden),
        "current_successful_entry_count": int(current_diagnostics.successful_entry_count),
        "min_hold_successful_entry_count": int(min_hold_diagnostics.successful_entry_count),
        "current_normal_exit_count": int(current_diagnostics.normal_exit_count),
        "min_hold_normal_exit_count": int(min_hold_diagnostics.normal_exit_count),
        "min_hold_blocked_exit_count": int(min_hold_diagnostics.min_hold_blocked_exit_count),
        "current_invariant_passed": bool(current_diagnostics.invariant_passed),
        "min_hold_invariant_passed": bool(min_hold_diagnostics.invariant_passed),
    }


def decision_guidance(comparison_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    final_1bp = next(
        row for row in comparison_rows
        if row["partition"] == "final_holdout" and float(row["cost_bps"]) == 1.0
    )
    overlay_1bp = next(
        row for row in comparison_rows
        if row["partition"] == "overlay_validation" and float(row["cost_bps"]) == 1.0
    )
    net_tolerance = -0.005
    drawdown_tolerance = -0.005
    favourable = (
        float(final_1bp["delta_net_total_return_min_hold_minus_current"]) >= net_tolerance
        and float(final_1bp["drawdown_delta_positive_means_less_severe"]) >= drawdown_tolerance
        and float(final_1bp["turnover_reduction_units"]) >= 0.0
        and float(overlay_1bp["delta_net_total_return_min_hold_minus_current"]) >= net_tolerance
        and bool(final_1bp["min_hold_invariant_passed"])
        and bool(overlay_1bp["min_hold_invariant_passed"])
    )
    return {
        "automatic_verdict": "FAVOUR_MODEL_B_MIN_HOLD_FOR_REVIEW" if favourable else "KEEP_MODEL_B_CURRENT_UNLESS_USER_ACCEPTS_TRADEOFF",
        "not_a_final_freeze_decision": True,
        "requires_user_and_methodology_review": True,
        "criteria_used": {
            "final_holdout_1bp_net_delta_minimum": net_tolerance,
            "overlay_validation_1bp_net_delta_minimum": net_tolerance,
            "final_holdout_1bp_drawdown_delta_minimum": drawdown_tolerance,
            "final_holdout_turnover_reduction_required": True,
            "invariants_required": True,
        },
        "final_holdout_1bp": dict(final_1bp),
        "overlay_validation_1bp": dict(overlay_1bp),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    repository_root = args.repo_root.expanduser().resolve()
    report_path = safe_repository_path(repository_root, args.report, description="Step 2C JSON report path", must_exist=False)
    metrics_csv_path = safe_repository_path(repository_root, args.metrics_csv, description="Step 2C metrics CSV path", must_exist=False)
    comparison_csv_path = safe_repository_path(repository_root, args.comparison_csv, description="Step 2C comparison CSV path", must_exist=False)

    report: dict[str, Any] = {
        "stage": 2,
        "step": "2C",
        "status": "RUNNING",
        "formal_gate": True,
        "diagnostic_only": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "candidate": {
            "variant": "MODEL_B_MIN_HOLD_3",
            "minimum_hold_bars": CANDIDATE_MINIMUM_HOLD_BARS,
            "changes_thresholds": False,
            "retraining": False,
            "uses_untouched_holdout_claim": False,
        },
        "checks": {},
    }

    try:
        model_a_config_path = safe_repository_path(repository_root, args.model_a_config, description="Frozen Model A configuration")
        model_b_config_path = safe_repository_path(repository_root, args.model_b_config, description="Frozen Model B configuration")
        model_a_config = load_model_a_config(model_a_config_path)
        model_b_config_raw = load_yaml_mapping(model_b_config_path)
        verify_stage0_freeze_manifest(repository_root, args.freeze_manifest)
        verify_notebook7_artifact_bundle(repository_root, model_a_config)
        base_rules = overlay_rules_from_model_b_config(model_b_config_raw)
        min_hold_rules = create_min_hold_rules(base_rules, minimum_hold_bars=CANDIDATE_MINIMUM_HOLD_BARS)

        stage1_step5 = verify_stage1_step5_report(repository_root, args.stage1_step5_report)
        stage2_step2a = verify_stage2_step2a_report(repository_root, args.stage2_step2a_report)
        report["checks"]["preconditions"] = {
            "passed": True,
            "stage1_step5_current_model_b": stage1_step5,
            "stage2_step2a_live_shadow_v1_1": stage2_step2a,
        }
        report["checks"]["rule_contract"] = {
            "passed": True,
            "current_variant": {
                "variant": "MODEL_B_V2_CURRENT",
                "entry_threshold": base_rules.entry_threshold,
                "exit_threshold": base_rules.exit_threshold,
                "minimum_hold_bars": 0,
                "long_only": True,
                "max_successful_entries_per_day": base_rules.max_successful_entries_per_day,
            },
            "candidate_variant": {
                "variant": "MODEL_B_MIN_HOLD_3",
                "entry_threshold": min_hold_rules.entry_threshold,
                "exit_threshold": min_hold_rules.exit_threshold,
                "minimum_hold_bars": min_hold_rules.minimum_hold_bars,
                "long_only": True,
                "max_successful_entries_per_day": min_hold_rules.max_successful_entries_per_day,
                "normal_exit_minimum_hold_only": True,
                "gap_and_risk_stops_override_minimum_hold": True,
            },
        }

        reference_manifest = load_step2_reference_manifest(repository_root, args.step2_manifest)
        prediction_frames: dict[str, pd.DataFrame] = {}
        prediction_sources: dict[str, Any] = {}
        for partition in PARTITIONS:
            if partition not in reference_manifest.prediction_references:
                raise Step2ParityError(f"Missing prediction reference: {partition}")
            predictions, integrity = verify_prediction_reference(
                repository_root=repository_root,
                name=partition,
                detail=reference_manifest.prediction_references[partition],
            )
            prediction_frames[partition] = predictions
            prediction_sources[partition] = integrity
        report["checks"]["prediction_references"] = {
            "passed": True,
            "references": prediction_sources,
        }

        variant_metric_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []
        payload_by_partition: dict[str, dict[str, Any]] = {}
        invariant_failures: list[str] = []

        for partition in PARTITIONS:
            payload_by_partition[partition] = {}
            for cost_bps in DEFAULT_COSTS_BPS:
                _current_log, current_metrics, current_diag = replay_model_b(
                    prediction_frames[partition],
                    base_rules,
                    cost_bps=cost_bps,
                )
                _mh_log, mh_metrics, mh_diag = replay_model_b_min_hold(
                    prediction_frames[partition],
                    min_hold_rules,
                    cost_bps=cost_bps,
                )
                variant_metric_rows.append(
                    current_variant_metrics_row(
                        partition=partition,
                        cost_bps=cost_bps,
                        metrics=current_metrics,
                        diagnostics=current_diag,
                    )
                )
                variant_metric_rows.append(
                    model_b_min_hold_metrics_row(
                        variant="MODEL_B_MIN_HOLD_3",
                        partition=partition,
                        cost_bps=cost_bps,
                        metrics=mh_metrics,
                        diagnostics=mh_diag,
                    )
                )
                row = comparison_row(
                    partition=partition,
                    cost_bps=cost_bps,
                    current_metrics=current_metrics,
                    current_diagnostics=current_diag,
                    min_hold_metrics=mh_metrics,
                    min_hold_diagnostics=mh_diag,
                )
                comparison_rows.append(row)
                payload_by_partition[partition][cost_key(cost_bps)] = {
                    "current": {
                        "metrics": metrics_to_dict(current_metrics),
                        "diagnostics": current_diagnostics_to_dict(current_diag),
                    },
                    "min_hold_3": {
                        "metrics": metrics_to_dict(mh_metrics),
                        "diagnostics": min_hold_diagnostics_to_dict(mh_diag),
                    },
                    "comparison": row,
                }
                if not current_diag.invariant_passed:
                    invariant_failures.extend(
                        f"current {partition} cost={cost_bps}: {failure}"
                        for failure in current_diag.invariant_failures
                    )
                if not mh_diag.invariant_passed:
                    invariant_failures.extend(
                        f"min_hold {partition} cost={cost_bps}: {failure}"
                        for failure in mh_diag.invariant_failures
                    )

        if invariant_failures:
            raise TradingReplayError("Model B Step 2C invariants failed: " + "; ".join(invariant_failures))

        guidance = decision_guidance(comparison_rows)
        report["checks"]["variant_comparison"] = {
            "passed": True,
            "costs_bps": list(DEFAULT_COSTS_BPS),
            "partitions": payload_by_partition,
            "comparison_rows": comparison_rows,
        }
        report["checks"]["decision_guidance"] = guidance
        report["status"] = "PASS"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report["metrics_csv"] = str(metrics_csv_path.relative_to(repository_root))
        report["comparison_csv"] = str(comparison_csv_path.relative_to(repository_root))

        metric_fieldnames = [
            "variant", "minimum_hold_bars", "partition", "cost_bps", "row_count", "active_bar_count",
            "active_bar_rate", "turnover_units", "round_turn_equivalent_trades", "policy_change_events",
            "gap_exit_events", "daily_stop_exit_events", "daily_stop_trigger_count", "total_stop_exit_events",
            "total_stop_triggered", "first_total_stop_trigger_utc", "final_gross_equity", "final_net_equity",
            "gross_total_return", "net_total_return", "max_drawdown", "gross_log_return_sum",
            "net_log_return_sum", "daily_stop_dates", "successful_entry_count", "normal_exit_count",
            "min_hold_blocked_exit_count", "daily_entry_cap_block_count", "gap_block_count",
            "short_position_count", "entry_below_threshold_count", "active_below_exit_threshold_count",
            "active_below_exit_threshold_after_eligible_count", "max_successful_entries_in_utc_day",
            "max_hold_bars_completed", "worst_daily_net_return", "transaction_cost_log_sum",
            "transaction_cost_burden", "invariant_passed", "invariant_failures",
        ]
        comparison_fieldnames = [
            "partition", "cost_bps", "current_net_total_return", "min_hold_net_total_return",
            "delta_net_total_return_min_hold_minus_current", "current_gross_total_return",
            "min_hold_gross_total_return", "delta_gross_total_return_min_hold_minus_current",
            "current_max_drawdown", "min_hold_max_drawdown", "drawdown_delta_positive_means_less_severe",
            "current_turnover_units", "min_hold_turnover_units", "turnover_reduction_units",
            "current_round_turn_equivalent_trades", "min_hold_round_turn_equivalent_trades",
            "trade_reduction_round_turn_equivalent", "current_active_bar_rate", "min_hold_active_bar_rate",
            "active_bar_rate_delta", "current_transaction_cost_burden", "min_hold_transaction_cost_burden",
            "transaction_cost_burden_reduction", "current_successful_entry_count", "min_hold_successful_entry_count",
            "current_normal_exit_count", "min_hold_normal_exit_count", "min_hold_blocked_exit_count",
            "current_invariant_passed", "min_hold_invariant_passed",
        ]
        write_csv_atomic(metrics_csv_path, variant_metric_rows, metric_fieldnames)
        write_csv_atomic(comparison_csv_path, comparison_rows, comparison_fieldnames)
        write_json_atomic(report_path, report)
        LOGGER.info("Stage 2 Step 2C status: PASS")
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Variant metrics CSV: %s", metrics_csv_path)
        LOGGER.info("Comparison CSV: %s", comparison_csv_path)
        return 0

    except (
        IntegrityError,
        Step1VerificationError,
        Step2ParityError,
        Step4TradingReplayError,
        TradingReplayError,
        ValueError,
        KeyError,
    ) as exc:
        report["status"] = "FAIL"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write Step 2C failure report")
        if args.debug:
            LOGGER.exception("Stage 2 Step 2C failed")
        else:
            LOGGER.error("Stage 2 Step 2C failed: %s", exc)
        return 2
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected Step 2C failure report")
        LOGGER.exception("Unexpected Stage 2 Step 2C failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
