#!/usr/bin/env python
"""Stage 1 Step 2 historical M15 feature and sequence parity verifier."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from capstone_trading.artifacts import (
    verify_notebook7_artifact_bundle,
    verify_stage0_freeze_manifest,
)
from capstone_trading.config import load_model_a_config, safe_repository_path
from capstone_trading.data.canonical_bars import report_to_dict as bar_report_to_dict
from capstone_trading.data.canonical_bars import validate_m15_bars
from capstone_trading.data.features_m15 import (
    TARGET_COLUMNS,
    build_volume_assisted_dataset,
    report_to_dict as feature_report_to_dict,
)
from capstone_trading.data.historical_adapter import (
    load_parquet_reference,
    load_prediction_reference,
    load_step2_reference_manifest,
    resolve_and_verify_file,
)
from capstone_trading.data.sequences import (
    materialize_sequences,
    partition_report,
    report_to_dict as partition_report_to_dict,
    scale_feature_frame,
    valid_sequence_positions,
    validate_plan_contiguity,
)
from capstone_trading.errors import Step1VerificationError, Step2ParityError
from capstone_trading.evaluation.feature_parity import (
    compare_model_ready_datasets,
    compare_sequence_endpoints_to_predictions,
    dataset_report_to_dict,
    report_to_dict as parity_report_to_dict,
)
from capstone_trading.model_loader import load_and_validate_scaler, report_to_dict

LOGGER = logging.getLogger("stage1_step2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the exact M15 volume-assisted feature dataset and verify "
            "Notebook 7 contiguous 48-bar sequence parity."
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
        default="runtime/reports/stage1_step2_feature_sequence_parity.json",
    )
    parser.add_argument(
        "--column-report",
        default="runtime/reports/stage1_step2_feature_column_parity.csv",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_column_report(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [
        "column",
        "dtype_reference",
        "dtype_rebuilt",
        "maximum_absolute_difference",
        "mismatch_count",
        "passed",
    ]
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


def validate_partition_reference(
    *,
    name: str,
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    fields = (
        "row_count",
        "sequence_count",
        "first_row_utc",
        "last_row_utc",
        "first_sequence_end_utc",
        "last_sequence_end_utc",
    )
    mismatches = {
        field: {"expected": expected.get(field), "actual": report.get(field)}
        for field in fields
        if report.get(field) != expected.get(field)
    }
    if mismatches:
        from capstone_trading.errors import SequenceParityError

        raise SequenceParityError(f"Partition {name} reference mismatch: {mismatches}")


def validate_reference_dataset(
    frame: pd.DataFrame, feature_order: tuple[str, ...]
) -> None:
    expected_columns = (*feature_order, *TARGET_COLUMNS)
    if tuple(frame.columns) != expected_columns:
        raise Step2ParityError("Official model-ready dataset column order is invalid")
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise Step2ParityError(
            "Official model-ready dataset index is duplicated or unordered"
        )
    if frame.isna().any().any():
        raise Step2ParityError("Official model-ready dataset contains missing values")
    numeric = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise Step2ParityError("Official model-ready dataset contains infinite values")
    target_dir = frame["target_dir"].to_numpy()
    target_class = frame["target_class_3"].to_numpy()
    if not np.isin(target_dir, [0, 1]).all():
        raise Step2ParityError("target_dir contains values outside {0, 1}")
    if not np.isin(target_class, [-1, 0, 1]).all():
        raise Step2ParityError("target_class_3 contains invalid values")


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
        description="Step 2 JSON report path",
        must_exist=False,
    )
    column_report_path = safe_repository_path(
        repository_root,
        args.column_report,
        description="Step 2 column report path",
        must_exist=False,
    )
    report: dict[str, Any] = {
        "stage": 1,
        "step": 2,
        "status": "RUNNING",
        "formal_gate": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "checks": {},
    }

    try:
        config_path = safe_repository_path(
            repository_root,
            args.model_a_config,
            description="Frozen Model A configuration",
        )
        config = load_model_a_config(config_path)
        verify_stage0_freeze_manifest(repository_root, args.freeze_manifest)
        bundle = verify_notebook7_artifact_bundle(repository_root, config)
        report["checks"]["frozen_contract"] = {
            "passed": True,
            "configuration_id": config.configuration_id,
            "feature_count": len(bundle.feature_order),
            "sequence_length": config.sequence_length,
        }

        reference_manifest = load_step2_reference_manifest(
            repository_root,
            args.step2_manifest,
        )
        if reference_manifest.bar_minutes != 15:
            raise Step2ParityError("Step 2 reference must use M15 bars")
        if reference_manifest.sequence_length != config.sequence_length:
            raise Step2ParityError(
                "Step 2 reference sequence length differs from frozen Model A"
            )

        bars_spec = reference_manifest.files["m15_bars"]
        dataset_spec = reference_manifest.files["model_ready_dataset"]
        bars_path = resolve_and_verify_file(repository_root, bars_spec)
        dataset_path = resolve_and_verify_file(repository_root, dataset_spec)
        bars_raw = load_parquet_reference(bars_path, bars_spec)
        official_dataset = load_parquet_reference(dataset_path, dataset_spec)
        report["checks"]["historical_file_integrity"] = {
            "passed": True,
            "m15_bars": {
                "path": bars_spec.relative_path.as_posix(),
                "sha256": bars_spec.sha256,
                "size_bytes": bars_spec.size_bytes,
            },
            "model_ready_dataset": {
                "path": dataset_spec.relative_path.as_posix(),
                "sha256": dataset_spec.sha256,
                "size_bytes": dataset_spec.size_bytes,
            },
        }

        bars, bars_report = validate_m15_bars(bars_raw)
        report["checks"]["m15_bar_validation"] = {
            "passed": True,
            **bar_report_to_dict(bars_report),
        }

        validate_reference_dataset(official_dataset, bundle.feature_order)
        rebuilt, feature_build_report = build_volume_assisted_dataset(
            bars,
            bundle.feature_order,
        )
        report["checks"]["feature_build"] = {
            "passed": True,
            **feature_report_to_dict(feature_build_report),
        }

        parity = compare_model_ready_datasets(
            rebuilt,
            official_dataset,
            tolerance=reference_manifest.feature_tolerance,
        )
        report["checks"]["feature_parity"] = {
            "passed": True,
            **dataset_report_to_dict(parity),
        }
        write_column_report(
            column_report_path,
            [
                {
                    "column": item.column,
                    "dtype_reference": item.dtype_reference,
                    "dtype_rebuilt": item.dtype_rebuilt,
                    "maximum_absolute_difference": item.maximum_absolute_difference,
                    "mismatch_count": item.mismatch_count,
                    "passed": item.passed,
                }
                for item in parity.column_results
            ],
        )

        scaler, scaler_report = load_and_validate_scaler(
            bundle.scaler_path,
            config,
            bundle.feature_order,
        )
        scaled = scale_feature_frame(scaler, rebuilt, bundle.feature_order)
        sample_positions = [0, len(rebuilt) // 2, len(rebuilt) - 1]
        report["checks"]["scaling_contract"] = {
            "passed": True,
            "scaler": report_to_dict(scaler_report),
            "input_cast_before_scaling": "float32",
            "output_dtype": str(scaled.dtype),
            "shape": list(scaled.shape),
            "finite": bool(np.isfinite(scaled).all()),
            "sample_row_positions": sample_positions,
            "sample_scaled_row_l2_norms": [
                float(np.linalg.norm(scaled[position])) for position in sample_positions
            ],
        }

        partition_reports: dict[str, Any] = {}
        prediction_reports: dict[str, Any] = {}
        for name, expected in reference_manifest.partitions.items():
            partition = select_partition(rebuilt, expected)
            plan = valid_sequence_positions(partition.index, config.sequence_length)
            validate_plan_contiguity(partition.index, plan)
            current_report = partition_report(name, partition, plan)
            current_dict = partition_report_to_dict(current_report)
            validate_partition_reference(
                name=name, report=current_dict, expected=expected
            )
            partition_reports[name] = {"passed": True, **current_dict}

            if name in reference_manifest.prediction_references:
                prediction_detail = reference_manifest.prediction_references[name]
                prediction_path = safe_repository_path(
                    repository_root,
                    str(prediction_detail["path"]),
                    description=f"{name} prediction reference",
                )
                from capstone_trading.artifacts import sha256_file
                from capstone_trading.errors import IntegrityError

                actual_prediction_hash = sha256_file(prediction_path)
                if actual_prediction_hash != prediction_detail["sha256"]:
                    raise IntegrityError(
                        f"Prediction reference hash mismatch for {name}: "
                        f"expected {prediction_detail['sha256']}, found {actual_prediction_hash}"
                    )
                predictions = load_prediction_reference(prediction_path)
                alignment = compare_sequence_endpoints_to_predictions(
                    name=name,
                    partition=partition,
                    plan=plan,
                    predictions=predictions,
                )
                prediction_reports[name] = {
                    "passed": True,
                    **parity_report_to_dict(alignment),
                }

        report["checks"]["sequence_partitions"] = {
            "passed": True,
            "partitions": partition_reports,
        }
        report["checks"]["prediction_alignment"] = {
            "passed": True,
            "references": prediction_reports,
        }

        full_plan = valid_sequence_positions(rebuilt.index, config.sequence_length)
        validate_plan_contiguity(rebuilt.index, full_plan)
        diagnostic_sequences = materialize_sequences(
            scaled,
            full_plan,
            [0, full_plan.sequence_count // 2, full_plan.sequence_count - 1],
        )
        report["checks"]["sequence_materialisation"] = {
            "passed": True,
            "full_dataset_sequence_count": full_plan.sequence_count,
            "diagnostic_shape": list(diagnostic_sequences.shape),
            "diagnostic_dtype": str(diagnostic_sequences.dtype),
            "finite": bool(np.isfinite(diagnostic_sequences).all()),
        }

        report["status"] = "PASS"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report["column_report"] = str(column_report_path.relative_to(repository_root))
        write_json_atomic(report_path, report)
        LOGGER.info("Stage 1 Step 2 status: PASS")
        LOGGER.info("JSON report: %s", report_path)
        LOGGER.info("Column report: %s", column_report_path)
        return 0

    except (Step1VerificationError, Step2ParityError, ValueError, KeyError) as exc:
        report["status"] = "FAIL"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write Step 2 failure report")
        if args.debug:
            LOGGER.exception("Stage 1 Step 2 failed")
        else:
            LOGGER.error("Stage 1 Step 2 failed: %s", exc)
        return 2
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected Step 2 failure report")
        LOGGER.exception("Unexpected Stage 1 Step 2 failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
