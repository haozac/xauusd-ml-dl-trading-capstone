#!/usr/bin/env python
"""Stage 1 Step 3 full CNN-LSTM inference parity verifier."""

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
from capstone_trading.config import load_model_a_config, safe_repository_path
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
)
from capstone_trading.evaluation.feature_parity import compare_model_ready_datasets
from capstone_trading.evaluation.inference_parity import (
    DEFAULT_THRESHOLDS,
    build_diagnostic_rows,
    compare_probability_vectors,
    predict_probabilities_in_batches,
    probability_report_to_dict,
    report_to_dict as inference_report_to_dict,
    validate_prediction_alignment,
)
from capstone_trading.model_loader import (
    check_runtime_environment,
    load_and_validate_model,
    load_and_validate_scaler,
    report_to_dict,
)

LOGGER = logging.getLogger("stage1_step3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the verified M15 feature sequences, run the frozen CNN-LSTM "
            "over all Notebook 7 2024 and holdout sequences, and compare every "
            "generated probability with the saved Notebook 7 p_up values."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument(
        "--freeze-manifest", default="config/stage0_freeze_manifest.json"
    )
    parser.add_argument(
        "--step2-manifest",
        default="config/stage1_step2_reference_manifest.json",
    )
    parser.add_argument(
        "--report",
        default="runtime/reports/stage1_step3_inference_parity.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="runtime/reports/stage1_step3_probability_parity_summary.csv",
    )
    parser.add_argument(
        "--diagnostic-csv",
        default="runtime/reports/stage1_step3_probability_parity_diagnostics.csv",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--probability-tolerance", type=float, default=1e-5)
    parser.add_argument("--return-tolerance", type=float, default=1e-15)
    parser.add_argument("--top-diagnostics", type=int, default=10)
    parser.add_argument(
        "--allow-onednn",
        action="store_true",
        help=(
            "Do not force TF_ENABLE_ONEDNN_OPTS=0. Formal Step 3 should leave this unset."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def prepare_formal_cpu_environment(*, allow_onednn: bool) -> None:
    """Set CPU deterministic environment variables before TensorFlow imports."""

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    if not allow_onednn:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
        raise Step2ParityError(
            "Official model-ready dataset index is duplicated or unordered"
        )
    if frame.isna().any().any():
        raise Step2ParityError("Official model-ready dataset contains missing values")
    values = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise Step2ParityError("Official model-ready dataset contains infinite values")


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
        description="Step 3 JSON report path",
        must_exist=False,
    )
    summary_csv_path = safe_repository_path(
        repository_root,
        args.summary_csv,
        description="Step 3 summary CSV report path",
        must_exist=False,
    )
    diagnostic_csv_path = safe_repository_path(
        repository_root,
        args.diagnostic_csv,
        description="Step 3 diagnostic CSV report path",
        must_exist=False,
    )

    report: dict[str, Any] = {
        "stage": 1,
        "step": 3,
        "status": "RUNNING",
        "formal_gate": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "checks": {},
    }

    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if args.probability_tolerance <= 0:
            raise ValueError("--probability-tolerance must be positive")
        if args.return_tolerance < 0:
            raise ValueError("--return-tolerance must be non-negative")
        if args.top_diagnostics < 1:
            raise ValueError("--top-diagnostics must be positive")

        prepare_formal_cpu_environment(allow_onednn=args.allow_onednn)
        report["checks"]["formal_cpu_environment"] = {
            "passed": True,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
            "TF_ENABLE_ONEDNN_OPTS": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
            "allow_onednn": bool(args.allow_onednn),
        }

        config_path = safe_repository_path(
            repository_root,
            args.model_a_config,
            description="Frozen Model A configuration",
        )
        config = load_model_a_config(config_path)
        verify_stage0_freeze_manifest(repository_root, args.freeze_manifest)
        bundle = verify_notebook7_artifact_bundle(repository_root, config)
        environment = check_runtime_environment(config, strict=True)
        report["checks"]["frozen_contract"] = {
            "passed": True,
            "configuration_id": config.configuration_id,
            "feature_count": len(bundle.feature_order),
            "sequence_length": config.sequence_length,
            "runtime_environment": report_to_dict(environment),
        }

        reference_manifest = load_step2_reference_manifest(
            repository_root,
            args.step2_manifest,
        )
        if reference_manifest.bar_minutes != 15:
            raise Step2ParityError("Step 3 requires the audited M15 Step 2 manifest")
        if reference_manifest.sequence_length != config.sequence_length:
            raise Step2ParityError(
                "Step 3 reference sequence length differs from frozen Model A"
            )

        bars_spec = reference_manifest.files["m15_bars"]
        dataset_spec = reference_manifest.files["model_ready_dataset"]
        bars_path = resolve_and_verify_file(repository_root, bars_spec)
        dataset_path = resolve_and_verify_file(repository_root, dataset_spec)
        bars_raw = load_parquet_reference(bars_path, bars_spec)
        official_dataset = load_parquet_reference(dataset_path, dataset_spec)
        report["checks"]["historical_file_integrity"] = {
            "passed": True,
            "m15_bars_sha256": bars_spec.sha256,
            "model_ready_dataset_sha256": dataset_spec.sha256,
        }

        bars, _bars_report = validate_m15_bars(bars_raw)
        validate_reference_dataset(official_dataset, bundle.feature_order)
        rebuilt, _feature_report = build_volume_assisted_dataset(
            bars,
            bundle.feature_order,
        )
        feature_parity = compare_model_ready_datasets(
            rebuilt,
            official_dataset,
            tolerance=reference_manifest.feature_tolerance,
        )
        report["checks"]["feature_sequence_precondition"] = {
            "passed": True,
            "rebuilt_rows": int(len(rebuilt)),
            "feature_count": len(bundle.feature_order),
            "maximum_feature_difference": feature_parity.maximum_absolute_difference,
            "feature_mismatch_count": feature_parity.total_mismatch_count,
        }

        scaler, scaler_report = load_and_validate_scaler(
            bundle.scaler_path,
            config,
            bundle.feature_order,
        )
        model, model_report = load_and_validate_model(bundle.model_path, config)
        report["checks"]["model_and_scaler"] = {
            "passed": True,
            "scaler": report_to_dict(scaler_report),
            "model": report_to_dict(model_report),
        }

        partition_results: dict[str, Any] = {}
        summary_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        total_sequences = 0
        total_threshold_flips = 0
        maximum_difference_all = 0.0

        for name in ("overlay_validation", "final_holdout"):
            if name not in reference_manifest.partitions:
                raise Step2ParityError(f"Missing Step 2 partition reference: {name}")
            if name not in reference_manifest.prediction_references:
                raise Step2ParityError(f"Missing prediction reference for Step 3: {name}")

            partition = select_partition(rebuilt, reference_manifest.partitions[name])
            plan = valid_sequence_positions(partition.index, config.sequence_length)
            validate_plan_contiguity(partition.index, plan)
            prediction_path, predictions, prediction_integrity = verify_prediction_reference(
                repository_root=repository_root,
                name=name,
                detail=reference_manifest.prediction_references[name],
            )
            alignment = validate_prediction_alignment(
                name=name,
                partition=partition,
                plan=plan,
                predictions=predictions,
                return_tolerance=args.return_tolerance,
            )
            scaled = scale_feature_frame(scaler, partition, bundle.feature_order)
            actual_probabilities = predict_probabilities_in_batches(
                model,
                scaled,
                plan,
                batch_size=args.batch_size,
            )
            expected_probabilities = predictions["p_up"].to_numpy(dtype=np.float64)
            probability_report = compare_probability_vectors(
                name=name,
                expected_probabilities=expected_probabilities,
                actual_probabilities=actual_probabilities,
                tolerance=args.probability_tolerance,
                thresholds=DEFAULT_THRESHOLDS,
            )

            probability_dict = probability_report_to_dict(probability_report)
            partition_results[name] = {
                "passed": True,
                "prediction_reference": prediction_integrity,
                "prediction_reference_path": str(prediction_path.relative_to(repository_root)),
                "alignment": inference_report_to_dict(alignment),
                "probability_parity": probability_dict,
            }
            total_sequences += probability_report.row_count
            total_threshold_flips += probability_report.total_threshold_flips
            maximum_difference_all = max(
                maximum_difference_all,
                probability_report.maximum_absolute_difference,
            )
            summary_rows.append(
                {
                    "partition": name,
                    "row_count": probability_report.row_count,
                    "tolerance": probability_report.tolerance,
                    "passed": probability_report.passed,
                    "maximum_absolute_difference": probability_report.maximum_absolute_difference,
                    "mean_absolute_difference": probability_report.mean_absolute_difference,
                    "median_absolute_difference": probability_report.median_absolute_difference,
                    "p95_absolute_difference": probability_report.p95_absolute_difference,
                    "p99_absolute_difference": probability_report.p99_absolute_difference,
                    "p999_absolute_difference": probability_report.p999_absolute_difference,
                    "mismatches_above_tolerance": probability_report.mismatches_above_tolerance,
                    "total_threshold_flips": probability_report.total_threshold_flips,
                    "expected_probability_mean": probability_report.expected_probability_mean,
                    "actual_probability_mean": probability_report.actual_probability_mean,
                    "prediction_reference": str(prediction_path.relative_to(repository_root)),
                }
            )
            diagnostic_rows.extend(
                build_diagnostic_rows(
                    partition_name=name,
                    predictions=predictions,
                    expected_probabilities=expected_probabilities,
                    actual_probabilities=actual_probabilities,
                    thresholds=DEFAULT_THRESHOLDS,
                    top_n=args.top_diagnostics,
                )
            )

        report["checks"]["inference_partitions"] = {
            "passed": True,
            "partitions": partition_results,
        }
        report["checks"]["aggregate_inference_parity"] = {
            "passed": True,
            "total_sequences": total_sequences,
            "maximum_absolute_difference": maximum_difference_all,
            "total_threshold_flips": total_threshold_flips,
            "probability_tolerance": args.probability_tolerance,
            "thresholds_checked": list(DEFAULT_THRESHOLDS),
        }

        write_csv_atomic(
            summary_csv_path,
            summary_rows,
            [
                "partition",
                "row_count",
                "tolerance",
                "passed",
                "maximum_absolute_difference",
                "mean_absolute_difference",
                "median_absolute_difference",
                "p95_absolute_difference",
                "p99_absolute_difference",
                "p999_absolute_difference",
                "mismatches_above_tolerance",
                "total_threshold_flips",
                "expected_probability_mean",
                "actual_probability_mean",
                "prediction_reference",
            ],
        )
        write_csv_atomic(
            diagnostic_csv_path,
            diagnostic_rows,
            [
                "partition",
                "category",
                "row_position",
                "time_utc",
                "target_dir",
                "threshold",
                "expected_p_up",
                "actual_p_up",
                "absolute_difference",
                "decision_flip",
            ],
        )

        report["status"] = "PASS"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report["summary_csv"] = str(summary_csv_path.relative_to(repository_root))
        report["diagnostic_csv"] = str(diagnostic_csv_path.relative_to(repository_root))
        write_json_atomic(report_path, report)
        LOGGER.info("Stage 1 Step 3 status: PASS")
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Summary CSV: %s", summary_csv_path)
        LOGGER.info("Diagnostic CSV: %s", diagnostic_csv_path)
        return 0

    except (
        Step1VerificationError,
        Step2ParityError,
        Step3InferenceError,
        InferenceParityError,
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
            LOGGER.exception("Unable to write Step 3 failure report")
        if args.debug:
            LOGGER.exception("Stage 1 Step 3 failed")
        else:
            LOGGER.error("Stage 1 Step 3 failed: %s", exc)
        return 2
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected Step 3 failure report")
        LOGGER.exception("Unexpected Stage 1 Step 3 failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
