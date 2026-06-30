#!/usr/bin/env python
"""Stage 1 Step 1 verifier for frozen Notebook 7 deployment artefacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capstone_trading.artifacts import (
    integrity_results_to_dict,
    verify_model_b_shared_artifacts,
    verify_notebook7_artifact_bundle,
    verify_stage0_freeze_manifest,
)
from capstone_trading.config import load_model_a_config, safe_repository_path
from capstone_trading.errors import Step1VerificationError
from capstone_trading.model_loader import (
    check_runtime_environment,
    load_and_validate_model,
    load_and_validate_scaler,
    load_reference_fixture,
    report_to_dict,
    run_reference_inference,
)

LOGGER = logging.getLogger("stage1_step1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify frozen Stage 0 files and Notebook 7 model artefacts."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-a-config", default="config/model_a_frozen.yaml")
    parser.add_argument("--model-b-config", default="config/model_b_v2_frozen.yaml")
    parser.add_argument("--freeze-manifest", default="config/stage0_freeze_manifest.json")
    parser.add_argument(
        "--reference-fixture",
        default="tests/fixtures/notebook7_reference_sequence.csv",
    )
    parser.add_argument(
        "--reference-metadata",
        default="tests/fixtures/notebook7_reference_sequence.json",
    )
    parser.add_argument(
        "--report",
        default="runtime/reports/stage1_step1_verification.json",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Verify hashes and JSON/YAML contracts without loading pickle or Keras files.",
    )
    parser.add_argument(
        "--non-strict-environment",
        action="store_true",
        help="Report version mismatches instead of failing. Not valid for the formal Step 1 gate.",
    )
    parser.add_argument("--inference-tolerance", type=float, default=1e-5)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


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
        description="Step 1 report path",
        must_exist=False,
    )
    report: dict[str, Any] = {
        "stage": 1,
        "step": 1,
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "formal_gate": not args.metadata_only and not args.non_strict_environment,
        "checks": {},
    }

    try:
        model_a_config_path = safe_repository_path(
            repository_root,
            args.model_a_config,
            description="Model A configuration",
        )
        model_b_config_path = safe_repository_path(
            repository_root,
            args.model_b_config,
            description="Model B configuration",
        )
        config = load_model_a_config(model_a_config_path)
        report["checks"]["model_a_configuration"] = {
            "passed": True,
            "configuration_id": config.configuration_id,
            "status": config.status,
        }

        frozen_results = verify_stage0_freeze_manifest(repository_root, args.freeze_manifest)
        report["checks"]["stage0_freeze_manifest"] = {
            "passed": True,
            "files": integrity_results_to_dict(frozen_results),
        }

        verify_model_b_shared_artifacts(model_b_config_path, config)
        report["checks"]["model_b_shared_artifacts"] = {"passed": True}

        bundle = verify_notebook7_artifact_bundle(repository_root, config)
        report["checks"]["notebook7_artifact_integrity"] = {
            "passed": True,
            "files": integrity_results_to_dict(bundle.integrity_results),
            "evaluation_manifest_files": integrity_results_to_dict(
                bundle.manifest_integrity_results
            ),
            "completion_lock": str(bundle.completion_lock_path.relative_to(repository_root)),
            "evaluation_manifest": str(
                bundle.evaluation_manifest_path.relative_to(repository_root)
            ),
            "feature_count": len(bundle.feature_order),
        }

        environment = check_runtime_environment(
            config,
            strict=not args.non_strict_environment and not args.metadata_only,
        )
        report["checks"]["runtime_environment"] = report_to_dict(environment)

        if args.metadata_only:
            report["checks"]["scaler_and_model_load"] = {
                "passed": False,
                "skipped": True,
                "reason": "--metadata-only was requested",
            }
            report["status"] = "PARTIAL_PASS_METADATA_ONLY"
        else:
            scaler, scaler_report = load_and_validate_scaler(
                bundle.scaler_path,
                config,
                bundle.feature_order,
            )
            model, model_report = load_and_validate_model(bundle.model_path, config)
            fixture_path = safe_repository_path(
                repository_root,
                args.reference_fixture,
                description="Reference sequence fixture",
            )
            metadata_path = safe_repository_path(
                repository_root,
                args.reference_metadata,
                description="Reference sequence metadata",
            )
            raw_sequence, metadata = load_reference_fixture(
                fixture_path, metadata_path, bundle.feature_order
            )
            inference_report = run_reference_inference(
                model,
                scaler,
                raw_sequence,
                metadata,
                bundle.feature_order,
                tolerance=args.inference_tolerance,
            )
            report["checks"]["scaler"] = report_to_dict(scaler_report)
            report["checks"]["model"] = report_to_dict(model_report)
            report["checks"]["reference_inference"] = report_to_dict(inference_report)
            report["status"] = "PASS" if environment.passed else "NON_FORMAL_PASS"

        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(report_path, report)
        LOGGER.info("Stage 1 Step 1 status: %s", report["status"])
        LOGGER.info("Report: %s", report_path)
        return 0

    except Step1VerificationError as exc:
        report["status"] = "FAIL"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_report(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write failure report")
        if args.debug:
            LOGGER.exception("Stage 1 Step 1 failed")
        else:
            LOGGER.error("Stage 1 Step 1 failed: %s", exc)
        return 2
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_report(report_path, report)
        except Exception:
            LOGGER.exception("Unable to write unexpected failure report")
        LOGGER.exception("Unexpected Stage 1 Step 1 failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
