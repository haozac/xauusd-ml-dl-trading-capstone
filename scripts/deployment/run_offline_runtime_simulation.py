#!/usr/bin/env python
"""Stage 1 Step 6 offline dual-system runtime simulation verifier."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from capstone_trading.artifacts import (
    sha256_file,
    verify_notebook7_artifact_bundle,
    verify_stage0_freeze_manifest,
)
from capstone_trading.config import (
    load_model_a_config,
    load_yaml_mapping,
    safe_repository_path,
)
from capstone_trading.data.canonical_bars import validate_m15_bars
from capstone_trading.data.features_m15 import TARGET_COLUMNS, build_volume_assisted_dataset
from capstone_trading.data.historical_adapter import (
    load_parquet_reference,
    load_prediction_reference,
    load_step2_reference_manifest,
    resolve_and_verify_file,
)
from capstone_trading.data.sequences import (
    scale_feature_frame,
    valid_sequence_positions,
    validate_plan_contiguity,
)
from capstone_trading.errors import (
    InferenceParityError,
    Step1VerificationError,
    Step2ParityError,
    Step3InferenceError,
    Step4TradingReplayError,
    TradingReplayError,
)
from capstone_trading.evaluation.feature_parity import compare_model_ready_datasets
from capstone_trading.evaluation.inference_parity import (
    DEFAULT_THRESHOLDS,
    compare_probability_vectors,
    predict_probabilities_in_batches,
    probability_report_to_dict,
)
from capstone_trading.evaluation.model_b_replay import (
    overlay_rules_from_model_b_config,
    replay_model_b,
)
from capstone_trading.evaluation.trading_replay import (
    DEFAULT_COSTS_BPS,
    overlay_rules_from_config,
    replay_model_a,
)
from capstone_trading.model_loader import (
    check_runtime_environment,
    load_and_validate_model,
    load_and_validate_scaler,
    report_to_dict,
)
from capstone_trading.runtime.offline_simulation import (
    compact_audit_row,
    compare_full_vs_resumed,
    compare_runtime_log,
    comparison_dict,
    diagnostic_dict,
    metric_dict,
    prediction_events_from_partition,
    streaming_report_dict,
    verify_streaming_event_materialisation,
    run_model_a_runtime,
    run_model_a_runtime_with_resume,
    run_model_b_runtime,
    run_model_b_runtime_with_resume,
)

LOGGER = logging.getLogger("stage1_step6")
RUNTIME_LOG_COLUMNS = (
    "position",
    "turnover",
    "gross_log_return",
    "cost_log_return",
    "net_log_return",
    "gross_equity",
    "net_equity",
    "net_drawdown",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an offline event-driven simulation of the future deployment loop. "
            "The script rebuilds M15 features, materialises contiguous windows, "
            "runs the frozen CNN-LSTM, feeds Model A and Model B stateful ledgers "
            "one prediction event at a time, and verifies full-run vs resumed-run determinism."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument("--model-b-config", default="config/model_b_v2_frozen.yaml")
    parser.add_argument("--freeze-manifest", default="config/stage0_freeze_manifest.json")
    parser.add_argument("--step2-manifest", default="config/stage1_step2_reference_manifest.json")
    parser.add_argument(
        "--step5-report",
        default="runtime/reports/stage1_step5_model_b_diagnostic_replay.json",
        help="Formal Step 5 PASS report used as the precondition for Step 6.",
    )
    parser.add_argument("--report", default="runtime/reports/stage1_step6_offline_runtime_simulation.json")
    parser.add_argument("--summary-csv", default="runtime/reports/stage1_step6_runtime_summary.csv")
    parser.add_argument("--audit-csv", default="runtime/reports/stage1_step6_runtime_event_audit_1bps.csv")
    parser.add_argument("--resume-csv", default="runtime/reports/stage1_step6_resume_determinism.csv")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--probability-tolerance", type=float, default=1e-5)
    parser.add_argument("--runtime-tolerance", type=float, default=1e-10)
    parser.add_argument("--resume-fraction", type=float, default=0.5)
    parser.add_argument(
        "--audit-cost-bps",
        type=float,
        default=1.0,
        help="Cost level used for the compact event audit CSV. All formal costs are still simulated.",
    )
    parser.add_argument(
        "--allow-onednn",
        action="store_true",
        help="Do not force TF_ENABLE_ONEDNN_OPTS=0. Formal Step 6 should leave this unset.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def prepare_formal_cpu_environment(*, allow_onednn: bool) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    if not allow_onednn:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


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


def verify_prior_step5_report(repository_root: Path, raw_path: str) -> dict[str, Any]:
    path = safe_repository_path(repository_root, raw_path, description="Stage 1 Step 5 report")
    payload = load_json_file(path)
    if payload.get("status") != "PASS" or payload.get("formal_gate") is not True:
        raise TradingReplayError("Stage 1 Step 6 requires a formal Stage 1 Step 5 PASS report")
    if payload.get("diagnostic_only") is not True:
        raise TradingReplayError("Stage 1 Step 5 report must be diagnostic_only=true")
    invariants = payload.get("checks", {}).get("model_b_invariants", {})
    if not isinstance(invariants, Mapping) or invariants.get("passed") is not True:
        raise TradingReplayError("Stage 1 Step 5 report does not prove Model B invariants")
    return {
        "path": str(path.relative_to(repository_root)),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "formal_gate": payload.get("formal_gate"),
        "diagnostic_only": payload.get("diagnostic_only"),
    }


def select_partition(frame: pd.DataFrame, detail: Mapping[str, Any]) -> pd.DataFrame:
    selected = frame
    start = detail.get("start_inclusive_utc")
    end = detail.get("end_exclusive_utc")
    if start:
        selected = selected.loc[selected.index >= pd.Timestamp(str(start), tz="UTC")]
    if end:
        selected = selected.loc[selected.index < pd.Timestamp(str(end), tz="UTC")]
    return selected


def validate_reference_dataset(frame: pd.DataFrame, feature_order: tuple[str, ...]) -> None:
    expected_columns = (*feature_order, *TARGET_COLUMNS)
    if tuple(frame.columns) != expected_columns:
        raise Step2ParityError("Official model-ready dataset column order is invalid")
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise Step2ParityError("Official model-ready dataset index is duplicated or unordered")
    values = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise Step2ParityError("Official model-ready dataset contains non-finite values")


def verify_prediction_reference(
    *,
    repository_root: Path,
    name: str,
    detail: Mapping[str, Any],
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    prediction_path = safe_repository_path(
        repository_root,
        str(detail["path"]),
        description=f"{name} prediction reference",
    )
    actual_hash = sha256_file(prediction_path)
    expected_hash = str(detail["sha256"]).lower()
    if actual_hash != expected_hash:
        from capstone_trading.errors import IntegrityError

        raise IntegrityError(
            f"Prediction reference hash mismatch for {name}: expected {expected_hash}, found {actual_hash}"
        )
    predictions = load_prediction_reference(prediction_path)
    return prediction_path, predictions, {
        "path": str(prediction_path.relative_to(repository_root)),
        "sha256": actual_hash,
        "rows": int(len(predictions)),
    }


def summary_row(
    *,
    partition: str,
    model_id: str,
    cost_bps: float,
    metrics: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {"partition": partition, "model_id": model_id, "cost_bps": float(cost_bps), **metrics}
    if extra:
        row.update(extra)
    return row


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    repository_root = args.repo_root.expanduser().resolve()
    report_path = safe_repository_path(repository_root, args.report, description="Step 6 report path", must_exist=False)
    summary_csv_path = safe_repository_path(repository_root, args.summary_csv, description="Step 6 summary CSV path", must_exist=False)
    audit_csv_path = safe_repository_path(repository_root, args.audit_csv, description="Step 6 audit CSV path", must_exist=False)
    resume_csv_path = safe_repository_path(repository_root, args.resume_csv, description="Step 6 resume CSV path", must_exist=False)

    report: dict[str, Any] = {
        "stage": 1,
        "step": 6,
        "status": "RUNNING",
        "formal_gate": True,
        "offline_only": True,
        "mt5_used": False,
        "orders_enabled": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "checks": {},
    }

    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if args.probability_tolerance <= 0:
            raise ValueError("--probability-tolerance must be positive")
        if args.runtime_tolerance < 0:
            raise ValueError("--runtime-tolerance must be non-negative")
        if not 0.05 <= args.resume_fraction <= 0.95:
            raise ValueError("--resume-fraction must be between 0.05 and 0.95")
        if float(args.audit_cost_bps) not in {float(value) for value in DEFAULT_COSTS_BPS}:
            raise ValueError(f"--audit-cost-bps must be one of {DEFAULT_COSTS_BPS}")

        prepare_formal_cpu_environment(allow_onednn=args.allow_onednn)
        report["checks"]["formal_cpu_environment"] = {
            "passed": True,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
            "TF_ENABLE_ONEDNN_OPTS": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
            "allow_onednn": bool(args.allow_onednn),
        }

        config_a_path = safe_repository_path(repository_root, args.model_a_config, description="Frozen Model A configuration")
        config_b_path = safe_repository_path(repository_root, args.model_b_config, description="Frozen Model B configuration")
        config_a = load_model_a_config(config_a_path)
        config_b_raw = load_yaml_mapping(config_b_path)
        verify_stage0_freeze_manifest(repository_root, args.freeze_manifest)
        bundle = verify_notebook7_artifact_bundle(repository_root, config_a)
        environment = check_runtime_environment(config_a, strict=True)
        rules_a = overlay_rules_from_config(config_a.raw)
        rules_b = overlay_rules_from_model_b_config(config_b_raw)
        report["checks"]["frozen_contracts"] = {
            "passed": True,
            "model_a_configuration_id": config_a.configuration_id,
            "model_b_configuration_id": config_b_raw.get("configuration_id"),
            "feature_count": len(bundle.feature_order),
            "sequence_length": config_a.sequence_length,
            "runtime_environment": report_to_dict(environment),
        }

        step5_precondition = verify_prior_step5_report(repository_root, args.step5_report)
        report["checks"]["stage1_step5_precondition"] = {"passed": True, **step5_precondition}

        reference_manifest = load_step2_reference_manifest(repository_root, args.step2_manifest)
        if reference_manifest.bar_minutes != 15 or reference_manifest.sequence_length != config_a.sequence_length:
            raise Step2ParityError("Step 6 requires the audited M15 Step 2 manifest and 48-bar sequence length")

        bars_spec = reference_manifest.files["m15_bars"]
        dataset_spec = reference_manifest.files["model_ready_dataset"]
        bars_path = resolve_and_verify_file(repository_root, bars_spec)
        dataset_path = resolve_and_verify_file(repository_root, dataset_spec)
        bars_raw = load_parquet_reference(bars_path, bars_spec)
        official_dataset = load_parquet_reference(dataset_path, dataset_spec)
        bars, bars_report = validate_m15_bars(bars_raw)
        validate_reference_dataset(official_dataset, bundle.feature_order)
        rebuilt, feature_report = build_volume_assisted_dataset(bars, bundle.feature_order)
        feature_parity = compare_model_ready_datasets(rebuilt, official_dataset, tolerance=reference_manifest.feature_tolerance)
        report["checks"]["historical_runtime_input"] = {
            "passed": True,
            "input_source": "audited_historical_m15_bars_rebuilt_to_features_then_streaming_verified_event_stream",
            "uses_saved_probabilities_as_runtime_input": False,
            "streaming_feature_event_materialisation_required": True,
            "m15_bar_rows": int(len(bars)),
            "rebuilt_feature_rows": int(len(rebuilt)),
            "feature_mismatch_count": feature_parity.total_mismatch_count,
            "maximum_feature_difference": feature_parity.maximum_absolute_difference,
            "bars_report": bars_report.__dict__,
            "feature_report": feature_report.__dict__,
        }

        scaler, scaler_report = load_and_validate_scaler(bundle.scaler_path, config_a, bundle.feature_order)
        model, model_report = load_and_validate_model(bundle.model_path, config_a)
        report["checks"]["model_and_scaler"] = {
            "passed": True,
            "scaler": report_to_dict(scaler_report),
            "model": report_to_dict(model_report),
        }

        partition_reports: dict[str, Any] = {}
        summary_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        resume_rows: list[dict[str, Any]] = []
        total_events = 0
        total_probability_flips = 0
        max_probability_difference = 0.0
        runtime_log_comparisons: list[dict[str, Any]] = []
        resume_comparisons: list[dict[str, Any]] = []
        streaming_materialisation_reports: list[dict[str, Any]] = []

        for partition_name in ("overlay_validation", "final_holdout"):
            partition_detail = reference_manifest.partitions[partition_name]
            partition = select_partition(rebuilt, partition_detail)
            plan = valid_sequence_positions(partition.index, config_a.sequence_length)
            validate_plan_contiguity(partition.index, plan)
            prediction_path, predictions, prediction_integrity = verify_prediction_reference(
                repository_root=repository_root,
                name=partition_name,
                detail=reference_manifest.prediction_references[partition_name],
            )
            scaled = scale_feature_frame(scaler, partition, bundle.feature_order)
            actual_probabilities = predict_probabilities_in_batches(model, scaled, plan, batch_size=args.batch_size)
            expected_probabilities = predictions["p_up"].to_numpy(dtype=np.float64)
            probability_report = compare_probability_vectors(
                name=partition_name,
                expected_probabilities=expected_probabilities,
                actual_probabilities=actual_probabilities,
                tolerance=args.probability_tolerance,
                thresholds=DEFAULT_THRESHOLDS,
            )
            events = prediction_events_from_partition(
                partition=partition,
                endpoint_positions=plan.ends,
                probabilities=actual_probabilities,
            )
            streaming_report = verify_streaming_event_materialisation(
                partition=partition,
                endpoint_positions=plan.ends,
                batch_events=events,
                sequence_length=config_a.sequence_length,
                partition_name=partition_name,
            )
            streaming_materialisation_reports.append(streaming_report_dict(streaming_report))
            # Strict timestamp and target alignment with the frozen prediction reference.
            if not np.array_equal(pd.DatetimeIndex(events["time"]).asi8, pd.DatetimeIndex(predictions["time"]).asi8):
                raise InferenceParityError(f"{partition_name} runtime event timestamps do not match Notebook 7")
            if not np.array_equal(events["target_dir"].to_numpy(dtype=np.int8), predictions["target_dir"].to_numpy(dtype=np.int8)):
                raise InferenceParityError(f"{partition_name} runtime target directions do not match Notebook 7")

            partition_result: dict[str, Any] = {
                "passed": True,
                "row_count": int(len(partition)),
                "runtime_event_count": int(len(events)),
                "prediction_reference": prediction_integrity,
                "streaming_event_materialisation": streaming_report_dict(streaming_report),
                "probability_recheck": probability_report_to_dict(probability_report),
                "runtime_costs": {},
            }
            total_events += int(len(events))
            total_probability_flips += probability_report.total_threshold_flips
            max_probability_difference = max(max_probability_difference, probability_report.maximum_absolute_difference)
            split_at = int(round(len(events) * float(args.resume_fraction)))
            split_at = max(1, min(len(events) - 1, split_at))

            for cost_bps in DEFAULT_COSTS_BPS:
                runtime_a_log, runtime_a_metrics = run_model_a_runtime(events, rules_a, cost_bps=cost_bps)
                batch_a_log, batch_a_metrics = replay_model_a(events, rules_a, cost_bps=cost_bps)
                a_comparisons = compare_runtime_log(
                    model_id="MODEL_A",
                    partition=partition_name,
                    cost_bps=cost_bps,
                    runtime_log=runtime_a_log,
                    reference_log=batch_a_log,
                    columns=RUNTIME_LOG_COLUMNS,
                    tolerance=args.runtime_tolerance,
                )
                runtime_b_log, runtime_b_metrics, runtime_b_diag = run_model_b_runtime(events, rules_b, cost_bps=cost_bps)
                batch_b_log, _batch_b_metrics, _batch_b_diag = replay_model_b(events, rules_b, cost_bps=cost_bps)
                b_comparisons = compare_runtime_log(
                    model_id="MODEL_B_V2",
                    partition=partition_name,
                    cost_bps=cost_bps,
                    runtime_log=runtime_b_log,
                    reference_log=batch_b_log,
                    columns=RUNTIME_LOG_COLUMNS,
                    tolerance=args.runtime_tolerance,
                )

                resumed_a_log, _resumed_a_metrics = run_model_a_runtime_with_resume(
                    events,
                    rules_a,
                    cost_bps=cost_bps,
                    split_at=split_at,
                )
                resumed_b_log, _resumed_b_metrics, _resumed_b_diag = run_model_b_runtime_with_resume(
                    events,
                    rules_b,
                    cost_bps=cost_bps,
                    split_at=split_at,
                )
                resume_a = compare_full_vs_resumed(
                    model_id="MODEL_A",
                    partition=partition_name,
                    cost_bps=cost_bps,
                    full_log=runtime_a_log,
                    resumed_log=resumed_a_log,
                    columns=RUNTIME_LOG_COLUMNS,
                    tolerance=args.runtime_tolerance,
                )
                resume_b = compare_full_vs_resumed(
                    model_id="MODEL_B_V2",
                    partition=partition_name,
                    cost_bps=cost_bps,
                    full_log=runtime_b_log,
                    resumed_log=resumed_b_log,
                    columns=RUNTIME_LOG_COLUMNS,
                    tolerance=args.runtime_tolerance,
                )
                runtime_log_comparisons.extend(comparison_dict(item) for item in (*a_comparisons, *b_comparisons))
                resume_comparisons.extend([comparison_dict(resume_a), comparison_dict(resume_b)])
                resume_rows.extend([comparison_dict(resume_a), comparison_dict(resume_b)])

                if cost_bps == args.audit_cost_bps:
                    event_records = events.to_dict("records")
                    a_records = runtime_a_log.to_dict("records")
                    b_records = runtime_b_log.to_dict("records")
                    audit_rows.extend(
                        compact_audit_row(
                            partition=partition_name,
                            row_position=idx,
                            event=event,
                            model_a_row=a_records[idx],
                            model_b_row=b_records[idx],
                        )
                        for idx, event in enumerate(event_records)
                    )

                model_a_extra = {"batch_net_total_return": batch_a_metrics.net_total_return}
                summary_rows.append(
                    summary_row(
                        partition=partition_name,
                        model_id="MODEL_A",
                        cost_bps=cost_bps,
                        metrics=metric_dict(runtime_a_metrics),
                        extra=model_a_extra,
                    )
                )
                summary_rows.append(
                    summary_row(
                        partition=partition_name,
                        model_id="MODEL_B_V2",
                        cost_bps=cost_bps,
                        metrics=metric_dict(runtime_b_metrics),
                        extra=diagnostic_dict(runtime_b_diag),
                    )
                )
                partition_result["runtime_costs"][str(cost_bps)] = {
                    "model_a": metric_dict(runtime_a_metrics),
                    "model_b": {
                        "metrics": metric_dict(runtime_b_metrics),
                        "diagnostics": diagnostic_dict(runtime_b_diag),
                    },
                    "resume_split_at_event": split_at,
                }

            partition_reports[partition_name] = partition_result

        if total_probability_flips != 0:
            raise InferenceParityError(f"Step 6 runtime inference created {total_probability_flips} threshold flips")
        if any(not item["passed"] for item in resume_comparisons):
            raise TradingReplayError("Step 6 resume determinism comparison failed")

        report["checks"]["runtime_partitions"] = {
            "passed": True,
            "partitions": partition_reports,
        }
        report["checks"]["streaming_event_materialisation"] = {
            "passed": True,
            "note": (
                "Feature rows were scanned chronologically through a rolling 48-row buffer; "
                "events were emitted only when the buffer was contiguous and then compared "
                "against the audited batch sequence/event plan."
            ),
            "reports": streaming_materialisation_reports,
        }
        report["checks"]["aggregate_runtime_simulation"] = {
            "passed": True,
            "total_runtime_events": total_events,
            "audit_cost_bps": float(args.audit_cost_bps),
            "audit_rows": len(audit_rows),
            "probability_tolerance": args.probability_tolerance,
            "maximum_probability_difference_vs_notebook7": max_probability_difference,
            "total_probability_threshold_flips": total_probability_flips,
            "runtime_batch_log_comparison_count": len(runtime_log_comparisons),
            "resume_comparison_count": len(resume_comparisons),
            "streaming_materialisation_check_count": len(streaming_materialisation_reports),
            "mt5_used": False,
            "orders_enabled": False,
        }
        report["checks"]["runtime_vs_batch_parity"] = {
            "passed": True,
            "comparisons": runtime_log_comparisons,
        }
        report["checks"]["restart_resume_determinism"] = {
            "passed": True,
            "resume_fraction": float(args.resume_fraction),
            "comparisons": resume_comparisons,
        }

        write_csv_atomic(
            summary_csv_path,
            summary_rows,
            sorted({key for row in summary_rows for key in row.keys()}),
        )
        write_csv_atomic(
            audit_csv_path,
            audit_rows,
            [
                "partition",
                "row_position",
                "time",
                "p_up",
                "target_dir",
                "target_ret_fwd",
                "model_a_position",
                "model_a_turnover",
                "model_a_net_equity",
                "model_a_change_reason",
                "model_b_position",
                "model_b_turnover",
                "model_b_net_equity",
                "model_b_change_reason",
            ],
        )
        write_csv_atomic(
            resume_csv_path,
            resume_rows,
            [
                "model_id",
                "partition",
                "cost_bps",
                "row_count",
                "comparable_columns",
                "maximum_absolute_difference",
                "mismatch_count",
                "tolerance",
                "passed",
            ],
        )

        report["status"] = "PASS"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report["summary_csv"] = str(summary_csv_path.relative_to(repository_root))
        report["audit_csv"] = str(audit_csv_path.relative_to(repository_root))
        report["resume_csv"] = str(resume_csv_path.relative_to(repository_root))
        write_json_atomic(report_path, report)
        LOGGER.info("Stage 1 Step 6 status: PASS")
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Summary CSV: %s", summary_csv_path)
        LOGGER.info("Audit CSV: %s", audit_csv_path)
        LOGGER.info("Resume CSV: %s", resume_csv_path)
        return 0

    except (
        Step1VerificationError,
        Step2ParityError,
        Step3InferenceError,
        Step4TradingReplayError,
        InferenceParityError,
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
            LOGGER.exception("Unable to write Step 6 failure report")
        if args.debug:
            LOGGER.exception("Stage 1 Step 6 failed")
        else:
            LOGGER.error("Stage 1 Step 6 failed: %s", exc)
        return 2
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected Step 6 failure report")
        LOGGER.exception("Unexpected Stage 1 Step 6 failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
