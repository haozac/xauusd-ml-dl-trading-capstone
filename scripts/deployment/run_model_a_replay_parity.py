#!/usr/bin/env python
"""Stage 1 Step 4 frozen Model A historical trading replay verifier."""

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
from capstone_trading.config import load_model_a_config, safe_repository_path
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
from capstone_trading.evaluation.trading_replay import (
    DEFAULT_COSTS_BPS,
    compare_metrics_to_reference,
    compare_overlay_selection_metrics,
    compare_replayed_bar_log,
    comparison_to_dict,
    load_strategy_bar_log,
    metrics_to_dict,
    overlay_rules_from_config,
    replay_model_a,
)

LOGGER = logging.getLogger("stage1_step4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the frozen Notebook 7 Model A trading overlay from saved "
            "prediction probabilities and compare the resulting trading path and "
            "metrics with frozen Notebook 7 artefacts."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument("--freeze-manifest", default="config/stage0_freeze_manifest.json")
    parser.add_argument(
        "--step2-manifest",
        default="config/stage1_step2_reference_manifest.json",
    )
    parser.add_argument(
        "--step3-report",
        default="runtime/reports/stage1_step3_inference_parity.json",
        help="Used as a precondition check only; Step 4 still consumes frozen prediction CSVs.",
    )
    parser.add_argument(
        "--report",
        default="runtime/reports/stage1_step4_model_a_replay_parity.json",
    )
    parser.add_argument(
        "--metrics-csv",
        default="runtime/reports/stage1_step4_model_a_metrics_by_cost.csv",
    )
    parser.add_argument(
        "--bar-comparison-csv",
        default="runtime/reports/stage1_step4_bar_log_comparison.csv",
    )
    parser.add_argument(
        "--bar-log-tolerance",
        type=float,
        default=1e-10,
        help="Tolerance for generated-vs-Notebook 7 per-row trading log columns.",
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-10,
        help="Tolerance for generated-vs-Notebook 7 aggregate trading metrics.",
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def verify_prior_step3_report(repository_root: Path, raw_path: str) -> dict[str, Any]:
    path = safe_repository_path(
        repository_root,
        raw_path,
        description="Stage 1 Step 3 inference parity report",
    )
    payload = load_json_file(path)
    if payload.get("status") != "PASS" or payload.get("formal_gate") is not True:
        raise TradingReplayError(
            "Stage 1 Step 4 requires a formal Stage 1 Step 3 PASS report before trading replay"
        )
    aggregate = payload.get("checks", {}).get("aggregate_inference_parity", {})
    if not isinstance(aggregate, Mapping) or aggregate.get("total_threshold_flips") != 0:
        raise TradingReplayError(
            "Stage 1 Step 3 report does not prove zero threshold flips"
        )
    return {
        "path": str(path.relative_to(repository_root)),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "formal_gate": payload.get("formal_gate"),
        "total_sequences": aggregate.get("total_sequences"),
        "total_threshold_flips": aggregate.get("total_threshold_flips"),
    }


def verify_prediction_reference(
    *,
    repository_root: Path,
    name: str,
    detail: Mapping[str, Any],
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
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
    return path, predictions, {
        "path": str(path.relative_to(repository_root)),
        "sha256": actual_hash,
        "rows": int(len(predictions)),
    }


def resolve_evidence_path(repository_root: Path, config_raw: Mapping[str, Any], key: str) -> Path:
    evidence = config_raw.get("evidence_sources")
    if not isinstance(evidence, Mapping) or key not in evidence:
        raise TradingReplayError(f"Frozen Model A configuration is missing evidence source: {key}")
    return safe_repository_path(repository_root, str(evidence[key]), description=key)


def metrics_rows(metrics_by_label: Mapping[str, Mapping[float, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition, by_cost in metrics_by_label.items():
        for cost_bps, metrics in by_cost.items():
            row = {"partition": partition, **metrics_to_dict(metrics)}
            row["cost_bps"] = cost_bps
            rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    repository_root = args.repo_root.expanduser().resolve()
    report_path = safe_repository_path(
        repository_root,
        args.report,
        description="Step 4 JSON report path",
        must_exist=False,
    )
    metrics_csv_path = safe_repository_path(
        repository_root,
        args.metrics_csv,
        description="Step 4 metrics CSV report path",
        must_exist=False,
    )
    bar_comparison_csv_path = safe_repository_path(
        repository_root,
        args.bar_comparison_csv,
        description="Step 4 bar-log comparison CSV report path",
        must_exist=False,
    )

    report: dict[str, Any] = {
        "stage": 1,
        "step": 4,
        "status": "RUNNING",
        "formal_gate": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "checks": {},
    }

    try:
        if args.bar_log_tolerance < 0:
            raise ValueError("--bar-log-tolerance must be non-negative")
        if args.metric_tolerance < 0:
            raise ValueError("--metric-tolerance must be non-negative")

        config_path = safe_repository_path(
            repository_root,
            args.model_a_config,
            description="Frozen Model A configuration",
        )
        config = load_model_a_config(config_path)
        verify_stage0_freeze_manifest(repository_root, args.freeze_manifest)
        bundle = verify_notebook7_artifact_bundle(repository_root, config)
        rules = overlay_rules_from_config(config.raw)
        report["checks"]["frozen_contract"] = {
            "passed": True,
            "configuration_id": config.configuration_id,
            "strategy_id": config.raw.get("strategy_id"),
            "sequence_length": config.sequence_length,
            "overlay_rules": {
                "long_threshold": rules.long_threshold,
                "short_threshold": rules.short_threshold,
                "minimum_hold_bars": rules.minimum_hold_bars,
                "max_policy_changes_per_day": rules.max_policy_changes_per_day,
                "daily_loss_log_threshold": rules.daily_loss_log_threshold,
                "total_drawdown_stop": rules.total_drawdown_stop,
            },
        }

        step3_precondition = verify_prior_step3_report(repository_root, args.step3_report)
        report["checks"]["stage1_step3_precondition"] = {
            "passed": True,
            **step3_precondition,
        }

        reference_manifest = load_step2_reference_manifest(
            repository_root,
            args.step2_manifest,
        )
        prediction_sources: dict[str, Any] = {}
        prediction_frames: dict[str, pd.DataFrame] = {}
        for partition in ("overlay_validation", "final_holdout"):
            if partition not in reference_manifest.prediction_references:
                raise Step2ParityError(f"Missing prediction reference: {partition}")
            path, predictions, integrity = verify_prediction_reference(
                repository_root=repository_root,
                name=partition,
                detail=reference_manifest.prediction_references[partition],
            )
            prediction_sources[partition] = integrity
            prediction_frames[partition] = predictions
        report["checks"]["prediction_references"] = {
            "passed": True,
            "references": prediction_sources,
        }

        cost_bps_values = DEFAULT_COSTS_BPS
        final_holdout_metrics: dict[float, Any] = {}
        final_holdout_logs: dict[float, pd.DataFrame] = {}
        overlay_validation_metrics: dict[float, Any] = {}

        for cost_bps in cost_bps_values:
            holdout_log, holdout_metric = replay_model_a(
                prediction_frames["final_holdout"],
                rules,
                cost_bps=cost_bps,
            )
            final_holdout_logs[cost_bps] = holdout_log
            final_holdout_metrics[cost_bps] = holdout_metric

            _validation_log, validation_metric = replay_model_a(
                prediction_frames["overlay_validation"],
                rules,
                cost_bps=cost_bps,
            )
            overlay_validation_metrics[cost_bps] = validation_metric

        report["checks"]["model_a_replay_metrics"] = {
            "passed": True,
            "costs_bps": list(cost_bps_values),
            "final_holdout": {
                str(cost): metrics_to_dict(metrics)
                for cost, metrics in final_holdout_metrics.items()
            },
            "overlay_validation": {
                str(cost): metrics_to_dict(metrics)
                for cost, metrics in overlay_validation_metrics.items()
            },
        }

        trading_metrics_path = resolve_evidence_path(
            repository_root,
            config.raw,
            "trading_metrics_by_cost",
        )
        reference_metrics = pd.read_csv(trading_metrics_path)
        metric_comparisons = compare_metrics_to_reference(
            final_holdout_metrics,
            reference_metrics,
            tolerance=args.metric_tolerance,
        )
        report["checks"]["final_holdout_metrics_parity"] = {
            "passed": True,
            "reference_path": str(trading_metrics_path.relative_to(repository_root)),
            "comparison_count": len(metric_comparisons),
            "comparisons": [comparison_to_dict(item) for item in metric_comparisons],
        }

        selected_overlay = load_json_file(bundle.selected_overlay_path)
        overlay_comparisons = compare_overlay_selection_metrics(
            overlay_validation_metrics[1.0],
            selected_overlay,
            tolerance=args.metric_tolerance,
        )
        report["checks"]["overlay_validation_selection_metric_parity"] = {
            "passed": True,
            "comparison_count": len(overlay_comparisons),
            "note": (
                "Zero comparisons means selected_overlay.json did not expose comparable validation "
                "aggregate metric keys; frozen overlay parameters were verified in Step 1."
            ),
            "comparisons": [comparison_to_dict(item) for item in overlay_comparisons],
        }

        strategy_bar_log_path = resolve_evidence_path(
            repository_root,
            config.raw,
            "strategy_bar_log_1bps",
        )
        reference_bar_log = load_strategy_bar_log(strategy_bar_log_path)
        bar_comparisons = compare_replayed_bar_log(
            final_holdout_logs[1.0],
            reference_bar_log,
            tolerance=args.bar_log_tolerance,
        )
        report["checks"]["final_holdout_bar_log_1bps_parity"] = {
            "passed": True,
            "reference_path": str(strategy_bar_log_path.relative_to(repository_root)),
            "comparison_count": len(bar_comparisons),
            "comparisons": [comparison_to_dict(item) for item in bar_comparisons],
        }

        write_csv_atomic(
            metrics_csv_path,
            metrics_rows(
                {
                    "overlay_validation": overlay_validation_metrics,
                    "final_holdout": final_holdout_metrics,
                }
            ),
            [
                "partition",
                "cost_bps",
                "row_count",
                "active_bar_count",
                "active_bar_rate",
                "turnover_units",
                "round_turn_equivalent_trades",
                "policy_change_events",
                "gap_exit_events",
                "daily_stop_exit_events",
                "daily_stop_trigger_count",
                "total_stop_exit_events",
                "total_stop_triggered",
                "first_total_stop_trigger_utc",
                "final_gross_equity",
                "final_net_equity",
                "gross_total_return",
                "net_total_return",
                "max_drawdown",
                "gross_log_return_sum",
                "net_log_return_sum",
                "daily_stop_dates",
            ],
        )
        write_csv_atomic(
            bar_comparison_csv_path,
            [comparison_to_dict(item) for item in bar_comparisons],
            [
                "column",
                "compared_rows",
                "maximum_absolute_difference",
                "mismatch_count",
                "tolerance",
                "passed",
            ],
        )

        report["status"] = "PASS"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report["metrics_csv"] = str(metrics_csv_path.relative_to(repository_root))
        report["bar_comparison_csv"] = str(bar_comparison_csv_path.relative_to(repository_root))
        write_json_atomic(report_path, report)
        LOGGER.info("Stage 1 Step 4 status: PASS")
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Metrics CSV: %s", metrics_csv_path)
        LOGGER.info("Bar comparison CSV: %s", bar_comparison_csv_path)
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
            LOGGER.exception("Unable to write Step 4 failure report")
        if args.debug:
            LOGGER.exception("Stage 1 Step 4 failed")
        else:
            LOGGER.error("Stage 1 Step 4 failed: %s", exc)
        return 2
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected Step 4 failure report")
        LOGGER.exception("Unexpected Stage 1 Step 4 failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
