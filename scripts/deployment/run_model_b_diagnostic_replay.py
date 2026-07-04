#!/usr/bin/env python
"""Stage 1 Step 5 frozen Model B historical diagnostic replay."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
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
from capstone_trading.evaluation.model_b_replay import (
    compare_model_b_to_model_a,
    diagnostics_to_dict,
    model_b_metrics_row,
    overlay_rules_from_model_b_config,
    replay_model_b,
)
from capstone_trading.evaluation.trading_replay import metrics_to_dict

LOGGER = logging.getLogger("stage1_step5")
DEFAULT_COSTS_BPS: tuple[float, ...] = (0.0, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the frozen post-holdout Model B long-only diagnostic overlay "
            "from frozen Notebook 7 prediction probabilities and compare diagnostic "
            "outcomes against the already-passed Model A replay. This is not an "
            "optimisation or replacement holdout."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument("--model-b-config", default="config/model_b_v2_frozen.yaml")
    parser.add_argument("--freeze-manifest", default="config/stage0_freeze_manifest.json")
    parser.add_argument("--step2-manifest", default="config/stage1_step2_reference_manifest.json")
    parser.add_argument(
        "--step4-report",
        default="runtime/reports/stage1_step4_model_a_replay_parity.json",
        help="Required formal Model A replay PASS report used as the diagnostic baseline.",
    )
    parser.add_argument(
        "--report",
        default="runtime/reports/stage1_step5_model_b_diagnostic_replay.json",
    )
    parser.add_argument(
        "--metrics-csv",
        default="runtime/reports/stage1_step5_model_b_metrics_by_cost.csv",
    )
    parser.add_argument(
        "--comparison-csv",
        default="runtime/reports/stage1_step5_model_b_vs_model_a_comparison.csv",
    )
    parser.add_argument(
        "--diagnostics-csv",
        default="runtime/reports/stage1_step5_model_b_diagnostics.csv",
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


def verify_prior_step4_report(repository_root: Path, raw_path: str) -> dict[str, Any]:
    path = safe_repository_path(
        repository_root,
        raw_path,
        description="Stage 1 Step 4 Model A replay parity report",
    )
    payload = load_json_file(path)
    if payload.get("status") != "PASS" or payload.get("formal_gate") is not True:
        raise TradingReplayError(
            "Stage 1 Step 5 requires a formal Stage 1 Step 4 PASS report before Model B diagnostics"
        )
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise TradingReplayError("Stage 1 Step 4 report is missing checks")
    final_metrics = checks.get("model_a_replay_metrics")
    if not isinstance(final_metrics, Mapping):
        raise TradingReplayError("Stage 1 Step 4 report is missing Model A replay metrics")
    if checks.get("final_holdout_bar_log_1bps_parity", {}).get("passed") is not True:
        raise TradingReplayError("Stage 1 Step 4 report does not prove final holdout bar-log parity")
    if checks.get("final_holdout_metrics_parity", {}).get("passed") is not True:
        raise TradingReplayError("Stage 1 Step 4 report does not prove final holdout metrics parity")
    return {
        "path": str(path.relative_to(repository_root)),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "formal_gate": payload.get("formal_gate"),
        "model_a_replay_metrics": final_metrics,
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


def model_a_metrics_for(
    step4_metrics: Mapping[str, Any],
    *,
    partition: str,
    cost_bps: float,
) -> Mapping[str, Any]:
    partition_metrics = step4_metrics.get(partition)
    if not isinstance(partition_metrics, Mapping):
        raise TradingReplayError(f"Step 4 report is missing Model A metrics for partition {partition}")
    key = cost_key(cost_bps)
    value = partition_metrics.get(key)
    if not isinstance(value, Mapping):
        # JSON dumps sometimes preserves plain float-string keys such as "1.0";
        # fail explicitly rather than silently comparing with the wrong cost.
        raise TradingReplayError(f"Step 4 report is missing Model A metrics for {partition} cost {key}")
    return value


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    repository_root = args.repo_root.expanduser().resolve()
    report_path = safe_repository_path(repository_root, args.report, description="Step 5 JSON report path", must_exist=False)
    metrics_csv_path = safe_repository_path(repository_root, args.metrics_csv, description="Step 5 metrics CSV path", must_exist=False)
    comparison_csv_path = safe_repository_path(repository_root, args.comparison_csv, description="Step 5 comparison CSV path", must_exist=False)
    diagnostics_csv_path = safe_repository_path(repository_root, args.diagnostics_csv, description="Step 5 diagnostics CSV path", must_exist=False)

    report: dict[str, Any] = {
        "stage": 1,
        "step": 5,
        "status": "RUNNING",
        "formal_gate": True,
        "diagnostic_only": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "checks": {},
    }

    try:
        model_a_config_path = safe_repository_path(repository_root, args.model_a_config, description="Frozen Model A configuration")
        model_b_config_path = safe_repository_path(repository_root, args.model_b_config, description="Frozen Model B configuration")
        model_a_config = load_model_a_config(model_a_config_path)
        model_b_config_raw = load_yaml_mapping(model_b_config_path)
        verify_stage0_freeze_manifest(repository_root, args.freeze_manifest)
        verify_notebook7_artifact_bundle(repository_root, model_a_config)
        rules = overlay_rules_from_model_b_config(model_b_config_raw)

        step4 = verify_prior_step4_report(repository_root, args.step4_report)
        step4_model_a_metrics = step4["model_a_replay_metrics"]

        report["checks"]["frozen_contract"] = {
            "passed": True,
            "model_b_configuration_id": model_b_config_raw.get("configuration_id"),
            "model_b_strategy_id": model_b_config_raw.get("strategy_id"),
            "diagnostic_label": model_b_config_raw.get("historical_replay_label"),
            "entry_threshold": rules.entry_threshold,
            "exit_threshold": rules.exit_threshold,
            "max_successful_entries_per_day": rules.max_successful_entries_per_day,
            "daily_loss_log_threshold": rules.daily_loss_log_threshold,
            "total_drawdown_stop": rules.total_drawdown_stop,
            "no_short_positions": True,
            "not_an_optimisation": True,
        }
        report["checks"]["stage1_step4_precondition"] = {
            "passed": True,
            "path": step4["path"],
            "sha256": step4["sha256"],
            "status": step4["status"],
            "formal_gate": step4["formal_gate"],
        }

        reference_manifest = load_step2_reference_manifest(repository_root, args.step2_manifest)
        prediction_frames: dict[str, pd.DataFrame] = {}
        prediction_sources: dict[str, Any] = {}
        for partition in ("overlay_validation", "final_holdout"):
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

        all_metrics_rows: list[dict[str, Any]] = []
        all_comparison_rows: list[dict[str, Any]] = []
        all_diagnostic_rows: list[dict[str, Any]] = []
        model_b_metrics_payload: dict[str, dict[str, Any]] = {}
        invariant_failures: list[str] = []

        for partition in ("overlay_validation", "final_holdout"):
            model_b_metrics_payload[partition] = {}
            for cost_bps in DEFAULT_COSTS_BPS:
                _log, metrics, diagnostics = replay_model_b(
                    prediction_frames[partition],
                    rules,
                    cost_bps=cost_bps,
                )
                row = model_b_metrics_row(
                    partition=partition,
                    cost_bps=cost_bps,
                    metrics=metrics,
                    diagnostics=diagnostics,
                )
                all_metrics_rows.append(row)
                all_diagnostic_rows.append(
                    {
                        "partition": partition,
                        "cost_bps": cost_bps,
                        **diagnostics_to_dict(diagnostics),
                    }
                )
                model_b_metrics_payload[partition][cost_key(cost_bps)] = {
                    "metrics": metrics_to_dict(metrics),
                    "diagnostics": diagnostics_to_dict(diagnostics),
                }
                comparison = compare_model_b_to_model_a(
                    partition=partition,
                    cost_bps=cost_bps,
                    model_a_metrics=model_a_metrics_for(
                        step4_model_a_metrics,
                        partition=partition,
                        cost_bps=cost_bps,
                    ),
                    model_b_metrics=metrics,
                    model_b_diagnostics=diagnostics,
                )
                all_comparison_rows.append(comparison)
                if not diagnostics.invariant_passed:
                    invariant_failures.extend(
                        f"{partition} cost={cost_bps}: {failure}"
                        for failure in diagnostics.invariant_failures
                    )

        if invariant_failures:
            raise TradingReplayError("Model B diagnostic invariants failed: " + "; ".join(invariant_failures))

        report["checks"]["model_b_diagnostic_replay"] = {
            "passed": True,
            "diagnostic_only": True,
            "optimised": False,
            "costs_bps": list(DEFAULT_COSTS_BPS),
            "partitions": model_b_metrics_payload,
        }
        report["checks"]["model_b_vs_model_a_diagnostic_comparison"] = {
            "passed": True,
            "pass_fail_profitability_test": False,
            "rows": all_comparison_rows,
        }
        report["checks"]["model_b_invariants"] = {
            "passed": True,
            "short_position_count_all_partitions": 0,
            "entry_below_threshold_count_all_partitions": 0,
            "active_below_exit_threshold_count_all_partitions": 0,
            "max_successful_entries_per_day_limit": rules.max_successful_entries_per_day,
        }

        metric_fieldnames = [
            "partition", "cost_bps", "row_count", "active_bar_count", "active_bar_rate",
            "turnover_units", "round_turn_equivalent_trades", "policy_change_events",
            "gap_exit_events", "daily_stop_exit_events", "daily_stop_trigger_count",
            "total_stop_exit_events", "total_stop_triggered", "first_total_stop_trigger_utc",
            "final_gross_equity", "final_net_equity", "gross_total_return", "net_total_return",
            "max_drawdown", "gross_log_return_sum", "net_log_return_sum", "daily_stop_dates",
            "successful_entry_count", "normal_exit_count", "daily_entry_cap_block_count",
            "gap_block_count", "short_position_count", "entry_below_threshold_count",
            "active_below_exit_threshold_count", "max_successful_entries_in_utc_day",
            "worst_daily_net_return", "transaction_cost_log_sum", "transaction_cost_burden",
            "invariant_passed", "invariant_failures",
        ]
        comparison_fieldnames = [
            "partition", "cost_bps", "model_a_net_total_return", "model_b_net_total_return",
            "delta_net_total_return_b_minus_a", "model_a_gross_total_return",
            "model_b_gross_total_return", "delta_gross_total_return_b_minus_a",
            "model_a_max_drawdown", "model_b_max_drawdown",
            "drawdown_reduction_positive_means_less_negative", "model_a_turnover_units",
            "model_b_turnover_units", "turnover_reduction_units", "model_a_active_bar_rate",
            "model_b_active_bar_rate", "active_rate_reduction", "model_a_transaction_cost_burden",
            "model_b_transaction_cost_burden", "transaction_cost_burden_reduction",
            "model_b_successful_entry_count", "model_b_short_position_count", "model_b_invariant_passed",
        ]
        diagnostic_fieldnames = [
            "partition", "cost_bps", "successful_entry_count", "normal_exit_count",
            "daily_entry_cap_block_count", "gap_block_count", "gap_exit_events",
            "daily_stop_exit_events", "total_stop_exit_events", "short_position_count",
            "entry_below_threshold_count", "active_below_exit_threshold_count",
            "max_successful_entries_in_utc_day", "worst_daily_net_return",
            "transaction_cost_log_sum", "transaction_cost_burden", "invariant_passed",
            "invariant_failures",
        ]
        write_csv_atomic(metrics_csv_path, all_metrics_rows, metric_fieldnames)
        write_csv_atomic(comparison_csv_path, all_comparison_rows, comparison_fieldnames)
        write_csv_atomic(diagnostics_csv_path, all_diagnostic_rows, diagnostic_fieldnames)

        report["status"] = "PASS"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report["metrics_csv"] = str(metrics_csv_path.relative_to(repository_root))
        report["comparison_csv"] = str(comparison_csv_path.relative_to(repository_root))
        report["diagnostics_csv"] = str(diagnostics_csv_path.relative_to(repository_root))
        write_json_atomic(report_path, report)
        LOGGER.info("Stage 1 Step 5 status: PASS")
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Metrics CSV: %s", metrics_csv_path)
        LOGGER.info("Comparison CSV: %s", comparison_csv_path)
        LOGGER.info("Diagnostics CSV: %s", diagnostics_csv_path)
        return 0

    except (
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
            LOGGER.exception("Unable to write Step 5 failure report")
        if args.debug:
            LOGGER.exception("Stage 1 Step 5 failed")
        else:
            LOGGER.error("Stage 1 Step 5 failed: %s", exc)
        return 2
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected Step 5 failure report")
        LOGGER.exception("Unexpected Stage 1 Step 5 failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
