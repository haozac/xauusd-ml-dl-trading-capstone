"""Loading and validation for immutable Stage 0 configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigurationError


@dataclass(frozen=True)
class ArtifactSpec:
    """A repository-relative artefact and its frozen SHA-256 digest."""

    name: str
    relative_path: Path
    sha256: str


@dataclass(frozen=True)
class RuntimeEnvironmentContract:
    """Pinned runtime versions required for Notebook 7 compatibility."""

    python_supported: tuple[str, ...]
    tensorflow: str
    keras: str
    scikit_learn: str
    parity_reference_device: str


@dataclass(frozen=True)
class ModelAConfig:
    """Validated subset of the frozen Model A configuration used by Step 1."""

    source_path: Path
    raw: Mapping[str, Any]
    configuration_id: str
    strategy_id: str
    status: str
    deployment_scope: str
    artefact_base_directory: Path
    artefacts: tuple[ArtifactSpec, ...]
    sequence_length: int
    feature_count: int
    input_dtype: str
    parameter_count: int
    scaler_class: str
    scaler_fitted_rows: int
    environment: RuntimeEnvironmentContract

    def artefact(self, name: str) -> ArtifactSpec:
        for item in self.artefacts:
            if item.name == name:
                return item
        raise ConfigurationError(f"Frozen artefact is not configured: {name}")


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to read YAML configuration {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"YAML root must be a mapping: {path}")
    return value


def _mapping(parent: Mapping[str, Any], key: str, source: Path) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Expected mapping '{key}' in {source}")
    return value


def _non_empty_string(parent: Mapping[str, Any], key: str, source: Path) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Expected non-empty string '{key}' in {source}")
    return value.strip()


def _positive_int(parent: Mapping[str, Any], key: str, source: Path) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"Expected positive integer '{key}' in {source}")
    return value


def _sha256(value: Any, field: str, source: Path) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ConfigurationError(f"Expected 64-character SHA-256 '{field}' in {source}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ConfigurationError(f"SHA-256 '{field}' is not hexadecimal in {source}") from exc
    return value.lower()


def safe_repository_path(
    repository_root: Path,
    relative_path: str | Path,
    *,
    description: str,
    must_exist: bool = True,
) -> Path:
    """Resolve a repository-relative path while blocking path traversal and symlink escape."""

    root = repository_root.expanduser().resolve()
    candidate_relative = Path(relative_path)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ConfigurationError(
            f"{description} must be repository-relative without '..': {relative_path}"
        )
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError(f"{description} escapes the repository root: {relative_path}") from exc
    if must_exist and not candidate.exists():
        raise ConfigurationError(f"{description} does not exist: {candidate}")
    return candidate


def load_model_a_config(path: str | Path) -> ModelAConfig:
    source = Path(path).expanduser().resolve()
    raw = _read_yaml_mapping(source)

    configuration_id = _non_empty_string(raw, "configuration_id", source)
    strategy_id = _non_empty_string(raw, "strategy_id", source)
    status = _non_empty_string(raw, "status", source)
    scope = _non_empty_string(raw, "deployment_scope", source)

    if strategy_id != "MODEL_A":
        raise ConfigurationError(f"Expected strategy_id MODEL_A in {source}, found {strategy_id}")
    if status != "FROZEN_STAGE_0":
        raise ConfigurationError(f"Model A configuration is not frozen: {status}")
    if scope != "MT5_DEMO_ONLY":
        raise ConfigurationError(f"Unexpected deployment scope for Model A: {scope}")

    bundle = _mapping(raw, "artifact_bundle", source)
    base_directory = Path(_non_empty_string(bundle, "base_directory", source))

    artefact_keys = (
        ("model", "model"),
        ("scaler", "scaler"),
        ("feature_order", "feature_order"),
        ("model_parameters", "model_parameters"),
        ("selected_epoch", "selected_epoch"),
        ("selected_overlay", "selected_overlay"),
    )
    artefacts: list[ArtifactSpec] = []
    for name, key in artefact_keys:
        section = _mapping(bundle, key, source)
        relative = Path(_non_empty_string(section, "path", source))
        digest = _sha256(section.get("sha256"), f"artifact_bundle.{key}.sha256", source)
        artefacts.append(ArtifactSpec(name, base_directory / relative, digest))

    model_contract = _mapping(raw, "model_contract", source)
    sequence_length = _positive_int(model_contract, "sequence_length", source)
    feature_count = _positive_int(model_contract, "feature_count", source)
    parameter_count = _positive_int(model_contract, "parameter_count", source)
    input_dtype = _non_empty_string(model_contract, "input_dtype", source)
    expected_shape = model_contract.get("input_shape")
    if expected_shape != [sequence_length, feature_count]:
        raise ConfigurationError(
            f"input_shape must equal [{sequence_length}, {feature_count}] in {source}"
        )

    scaler = _mapping(bundle, "scaler", source)
    scaler_class = _non_empty_string(scaler, "class", source)
    scaler_feature_count = _positive_int(scaler, "fitted_feature_count", source)
    scaler_fitted_rows = _positive_int(scaler, "fitted_rows", source)
    if scaler_feature_count != feature_count:
        raise ConfigurationError(
            f"Scaler feature count {scaler_feature_count} differs from model feature count {feature_count}"
        )

    environment_raw = _mapping(raw, "runtime_environment_contract", source)
    python_supported_raw = environment_raw.get("python_supported")
    if not isinstance(python_supported_raw, list) or not python_supported_raw:
        raise ConfigurationError(f"python_supported must be a non-empty list in {source}")
    python_supported = tuple(str(item) for item in python_supported_raw)
    environment = RuntimeEnvironmentContract(
        python_supported=python_supported,
        tensorflow=_non_empty_string(environment_raw, "tensorflow", source),
        keras=_non_empty_string(environment_raw, "keras", source),
        scikit_learn=_non_empty_string(environment_raw, "scikit_learn", source),
        parity_reference_device=_non_empty_string(
            environment_raw, "parity_reference_device", source
        ),
    )

    return ModelAConfig(
        source_path=source,
        raw=raw,
        configuration_id=configuration_id,
        strategy_id=strategy_id,
        status=status,
        deployment_scope=scope,
        artefact_base_directory=base_directory,
        artefacts=tuple(artefacts),
        sequence_length=sequence_length,
        feature_count=feature_count,
        input_dtype=input_dtype,
        parameter_count=parameter_count,
        scaler_class=scaler_class,
        scaler_fitted_rows=scaler_fitted_rows,
        environment=environment,
    )


def load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    """Public safe YAML reader for consistency checks."""

    return _read_yaml_mapping(Path(path).expanduser().resolve())
