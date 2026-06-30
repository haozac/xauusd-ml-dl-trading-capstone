"""Integrity and contract validation for frozen Stage 0 and Notebook 7 files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ModelAConfig, load_yaml_mapping, safe_repository_path
from .errors import ArtifactValidationError, ConfigurationError, IntegrityError


@dataclass(frozen=True)
class FileIntegrityResult:
    logical_name: str
    relative_path: str
    expected_sha256: str
    actual_sha256: str
    size_bytes: int
    passed: bool


@dataclass(frozen=True)
class ArtifactBundle:
    model_path: Path
    scaler_path: Path
    feature_order_path: Path
    model_parameters_path: Path
    selected_epoch_path: Path
    selected_overlay_path: Path
    completion_lock_path: Path
    evaluation_manifest_path: Path
    feature_order: tuple[str, ...]
    model_parameters: Mapping[str, Any]
    integrity_results: tuple[FileIntegrityResult, ...]
    manifest_integrity_results: tuple[FileIntegrityResult, ...]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise IntegrityError(f"Unable to hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Unable to read JSON artefact {path}: {exc}") from exc


def verify_stage0_freeze_manifest(
    repository_root: Path,
    manifest_relative_path: str | Path = "config/stage0_freeze_manifest.json",
) -> tuple[FileIntegrityResult, ...]:
    manifest_path = safe_repository_path(
        repository_root,
        manifest_relative_path,
        description="Stage 0 freeze manifest",
    )
    raw = _load_json(manifest_path)
    if not isinstance(raw, Mapping):
        raise IntegrityError("Stage 0 freeze manifest root must be an object")
    if raw.get("status") != "FROZEN_STAGE_0":
        raise IntegrityError("Stage 0 freeze manifest is not marked FROZEN_STAGE_0")
    if raw.get("hash_algorithm") != "SHA-256":
        raise IntegrityError("Stage 0 freeze manifest must use SHA-256")
    files = raw.get("files")
    if not isinstance(files, Mapping) or not files:
        raise IntegrityError("Stage 0 freeze manifest contains no files")

    results: list[FileIntegrityResult] = []
    for relative_path, detail in files.items():
        if not isinstance(relative_path, str) or not isinstance(detail, Mapping):
            raise IntegrityError("Invalid file entry in Stage 0 freeze manifest")
        expected = detail.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise IntegrityError(f"Invalid SHA-256 for frozen file: {relative_path}")
        path = safe_repository_path(
            repository_root,
            relative_path,
            description=f"Frozen Stage 0 file {relative_path}",
        )
        actual = sha256_file(path)
        result = FileIntegrityResult(
            logical_name="stage0_frozen_file",
            relative_path=relative_path,
            expected_sha256=expected.lower(),
            actual_sha256=actual,
            size_bytes=path.stat().st_size,
            passed=actual == expected.lower(),
        )
        results.append(result)
        if not result.passed:
            raise IntegrityError(
                f"Stage 0 freeze hash mismatch for {relative_path}: "
                f"expected {expected.lower()}, found {actual}"
            )
    return tuple(results)


def _validate_feature_order(value: Any, expected_count: int, source: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ArtifactValidationError(
            f"Feature order in {source} must contain exactly {expected_count} names"
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ArtifactValidationError(f"Feature order contains an invalid feature name: {source}")
    feature_order = tuple(item.strip() for item in value)
    if len(set(feature_order)) != len(feature_order):
        raise ArtifactValidationError(f"Feature order contains duplicate names: {source}")
    return feature_order



def _verify_completion_lock_and_evaluation_manifest(
    repository_root: Path,
    config: ModelAConfig,
) -> tuple[Path, Path, tuple[FileIntegrityResult, ...]]:
    evidence = config.raw.get("evidence_sources")
    if not isinstance(evidence, Mapping):
        raise ArtifactValidationError("Model A evidence_sources must be a mapping")
    completion_lock_path = safe_repository_path(
        repository_root,
        str(evidence.get("completion_lock", "")),
        description="Notebook 7 completion lock",
    )
    manifest_path = safe_repository_path(
        repository_root,
        str(evidence.get("artefact_manifest", "")),
        description="Notebook 7 evaluation artefact manifest",
    )
    completion = _load_json(completion_lock_path)
    manifest = _load_json(manifest_path)
    if not isinstance(completion, Mapping) or not isinstance(manifest, Mapping):
        raise ArtifactValidationError("Completion lock and evaluation manifest must be JSON objects")
    research_source = config.raw.get("research_source")
    if not isinstance(research_source, Mapping):
        raise ArtifactValidationError("Model A research_source must be a mapping")
    expected_fingerprint = research_source.get("config_fingerprint")
    checks = {
        "completion status": (completion.get("status"), "COMPLETE_AND_LOCKED"),
        "completion holdout status": (completion.get("holdout_status"), "NEGATIVE"),
        "completion fingerprint": (completion.get("config_fingerprint"), expected_fingerprint),
        "manifest fingerprint": (manifest.get("config_fingerprint"), expected_fingerprint),
        "manifest holdout status": (manifest.get("holdout_status"), "NEGATIVE"),
        "manifest selected epoch": (
            manifest.get("selected_epoch"),
            config.raw["model_contract"]["selected_epoch"],
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ArtifactValidationError(
                f"Notebook 7 {label} mismatch: expected {expected}, found {actual}"
            )

    actual_manifest_hash = sha256_file(manifest_path)
    expected_manifest_hash = completion.get("evaluation_manifest_sha256")
    if actual_manifest_hash != expected_manifest_hash:
        raise IntegrityError(
            "Evaluation manifest hash does not match the Notebook 7 completion lock"
        )

    artefacts = manifest.get("artefacts")
    if not isinstance(artefacts, Mapping) or not artefacts:
        raise ArtifactValidationError("Evaluation manifest contains no artefact map")
    base_directory = config.artefact_base_directory
    results: list[FileIntegrityResult] = []
    for relative_inside_bundle, expected_hash in artefacts.items():
        if not isinstance(relative_inside_bundle, str) or not isinstance(expected_hash, str):
            raise ArtifactValidationError("Invalid artefact entry in evaluation manifest")
        repository_relative = base_directory / relative_inside_bundle
        path = safe_repository_path(
            repository_root,
            repository_relative,
            description=f"Evaluation manifest artefact {relative_inside_bundle}",
        )
        if not path.is_file():
            raise ArtifactValidationError(f"Evaluation manifest artefact is not a file: {path}")
        actual_hash = sha256_file(path)
        result = FileIntegrityResult(
            logical_name="evaluation_manifest_artefact",
            relative_path=repository_relative.as_posix(),
            expected_sha256=expected_hash.lower(),
            actual_sha256=actual_hash,
            size_bytes=path.stat().st_size,
            passed=actual_hash == expected_hash.lower(),
        )
        results.append(result)
        if not result.passed:
            raise IntegrityError(
                f"Evaluation manifest hash mismatch for {relative_inside_bundle}: "
                f"expected {expected_hash.lower()}, found {actual_hash}"
            )

    prediction_hash = completion.get("holdout_predictions_sha256")
    manifest_prediction_hash = artefacts.get("tables/final_holdout_predictions.csv")
    if prediction_hash != manifest_prediction_hash:
        raise IntegrityError(
            "Holdout prediction hash differs between completion lock and evaluation manifest"
        )
    return completion_lock_path, manifest_path, tuple(results)

def verify_notebook7_artifact_bundle(
    repository_root: Path,
    config: ModelAConfig,
) -> ArtifactBundle:
    resolved: dict[str, Path] = {}
    results: list[FileIntegrityResult] = []
    completion_lock_path, evaluation_manifest_path, manifest_results = (
        _verify_completion_lock_and_evaluation_manifest(repository_root, config)
    )

    for spec in config.artefacts:
        path = safe_repository_path(
            repository_root,
            spec.relative_path,
            description=f"Notebook 7 artefact {spec.name}",
        )
        if not path.is_file():
            raise ArtifactValidationError(f"Notebook 7 artefact is not a file: {path}")
        actual = sha256_file(path)
        result = FileIntegrityResult(
            logical_name=spec.name,
            relative_path=spec.relative_path.as_posix(),
            expected_sha256=spec.sha256,
            actual_sha256=actual,
            size_bytes=path.stat().st_size,
            passed=actual == spec.sha256,
        )
        results.append(result)
        if not result.passed:
            raise IntegrityError(
                f"Notebook 7 artefact hash mismatch for {spec.name}: "
                f"expected {spec.sha256}, found {actual}"
            )
        resolved[spec.name] = path

    feature_order = _validate_feature_order(
        _load_json(resolved["feature_order"]),
        config.feature_count,
        resolved["feature_order"],
    )
    model_parameters = _load_json(resolved["model_parameters"])
    if not isinstance(model_parameters, Mapping):
        raise ArtifactValidationError("Model parameter artefact must contain a JSON object")
    frozen_params = model_parameters.get("frozen_params")
    if not isinstance(frozen_params, Mapping):
        raise ArtifactValidationError("Model parameter artefact is missing frozen_params")
    expected_model_values = {
        "model_family": config.raw["model_contract"]["family"].lower().replace("-", "_"),
        "track": config.raw["model_contract"]["track"],
        "selected_epoch": config.raw["model_contract"]["selected_epoch"],
        "model_parameter_count": config.parameter_count,
    }
    for field, expected in expected_model_values.items():
        actual = model_parameters.get(field)
        if actual != expected:
            raise ArtifactValidationError(
                f"Model parameter artefact mismatch for {field}: expected {expected}, found {actual}"
            )
    expected_frozen_params = {
        "sequence_length": config.sequence_length,
        "conv_filters_1": config.raw["model_contract"]["architecture"]["conv_filters_1"],
        "conv_filters_2": config.raw["model_contract"]["architecture"]["conv_filters_2"],
        "kernel_size": config.raw["model_contract"]["architecture"]["kernel_size"],
        "pool_size": config.raw["model_contract"]["architecture"]["pool_size"],
        "lstm_units": config.raw["model_contract"]["architecture"]["lstm_units"],
        "dropout": config.raw["model_contract"]["architecture"]["external_dropout"],
        "l2": config.raw["model_contract"]["architecture"]["l2"],
    }
    for field, expected in expected_frozen_params.items():
        actual = frozen_params.get(field)
        if actual != expected:
            raise ArtifactValidationError(
                f"Frozen model parameter mismatch for {field}: expected {expected}, found {actual}"
            )

    selected_epoch = _load_json(resolved["selected_epoch"])
    if not isinstance(selected_epoch, Mapping):
        raise ArtifactValidationError("Selected epoch artefact must contain a JSON object")
    configured_epoch = config.raw["artifact_bundle"]["selected_epoch"]["value"]
    observed_epoch = selected_epoch.get("selected_epoch", selected_epoch.get("epoch"))
    if observed_epoch != configured_epoch:
        raise ArtifactValidationError(
            f"Selected epoch mismatch: configuration={configured_epoch}, artefact={observed_epoch}"
        )

    selected_overlay = _load_json(resolved["selected_overlay"])
    if not isinstance(selected_overlay, Mapping):
        raise ArtifactValidationError("Selected overlay artefact must contain a JSON object")
    overlay_config = config.raw["overlay"]
    expected_overlay_values = {
        "upper_threshold": overlay_config["long_when_p_up_gte"],
        "lower_threshold": overlay_config["short_when_p_up_lte"],
        "min_hold_bars": overlay_config["minimum_hold_eligible_bars"],
        "max_position_change_events_per_day": overlay_config[
            "maximum_overlay_position_change_events_per_utc_day"
        ],
        "selection_cost_bps": config.raw["historical_cost_semantics"][
            "main_one_way_cost_bps_per_turnover_unit"
        ],
    }
    for field, expected in expected_overlay_values.items():
        actual = selected_overlay.get(field)
        if actual != expected:
            raise ArtifactValidationError(
                f"Selected overlay mismatch for {field}: expected {expected}, found {actual}"
            )

    return ArtifactBundle(
        model_path=resolved["model"],
        scaler_path=resolved["scaler"],
        feature_order_path=resolved["feature_order"],
        model_parameters_path=resolved["model_parameters"],
        selected_epoch_path=resolved["selected_epoch"],
        selected_overlay_path=resolved["selected_overlay"],
        completion_lock_path=completion_lock_path,
        evaluation_manifest_path=evaluation_manifest_path,
        feature_order=feature_order,
        model_parameters=model_parameters,
        integrity_results=tuple(results),
        manifest_integrity_results=manifest_results,
    )


def verify_model_b_shared_artifacts(
    model_b_config_path: Path,
    model_a_config: ModelAConfig,
) -> None:
    raw = load_yaml_mapping(model_b_config_path)
    if raw.get("status") != "FROZEN_STAGE_0":
        raise ConfigurationError("Model B configuration is not frozen")
    if raw.get("strategy_id") != "MODEL_B_V2":
        raise ConfigurationError("Unexpected Model B strategy_id")
    shared = raw.get("shared_frozen_artifacts")
    if not isinstance(shared, Mapping):
        raise ConfigurationError("Model B shared_frozen_artifacts must be a mapping")

    expected = {item.name: item for item in model_a_config.artefacts}
    for name in ("model", "scaler", "feature_order", "model_parameters"):
        detail = shared.get(name)
        if not isinstance(detail, Mapping):
            raise ConfigurationError(f"Model B shared artefact is missing: {name}")
        configured_path = Path(str(shared.get("base_directory"))) / str(detail.get("path"))
        configured_hash = str(detail.get("sha256", "")).lower()
        source = expected[name]
        if configured_path != source.relative_path or configured_hash != source.sha256:
            raise ConfigurationError(
                f"Model B {name} does not match the frozen Model A artefact contract"
            )


def integrity_results_to_dict(
    results: Sequence[FileIntegrityResult],
) -> list[dict[str, Any]]:
    return [asdict(item) for item in results]
