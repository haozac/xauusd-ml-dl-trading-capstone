"""Secure loading and validation of the frozen Notebook 7 scaler and Keras model."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np

from .config import ModelAConfig
from .errors import (
    ArtifactValidationError,
    EnvironmentCompatibilityError,
    ModelLoadError,
)


@dataclass(frozen=True)
class PackageVersionCheck:
    package: str
    expected: str
    actual: str | None
    passed: bool


@dataclass(frozen=True)
class EnvironmentReport:
    python_expected: tuple[str, ...]
    python_actual: str
    python_passed: bool
    package_checks: tuple[PackageVersionCheck, ...]
    strict: bool

    @property
    def passed(self) -> bool:
        return self.python_passed and all(item.passed for item in self.package_checks)


@dataclass(frozen=True)
class ScalerReport:
    class_name: str
    feature_count: int
    fitted_rows: int
    has_feature_names: bool
    feature_names_match: bool


@dataclass(frozen=True)
class ModelReport:
    class_name: str
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    parameter_count: int
    layer_classes: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceInferenceReport:
    fixture_end_utc: str
    expected_probability: float
    actual_probability: float
    absolute_difference: float
    tolerance: float
    threshold_flip_count: int
    passed: bool


def _installed_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _same_version(actual: str | None, expected: str) -> bool:
    return actual == expected


def check_runtime_environment(config: ModelAConfig, *, strict: bool = True) -> EnvironmentReport:
    python_actual = f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"
    package_expectations = (
        ("tensorflow", config.environment.tensorflow),
        ("keras", config.environment.keras),
        ("scikit-learn", config.environment.scikit_learn),
    )
    package_checks = tuple(
        PackageVersionCheck(
            package=name,
            expected=expected,
            actual=(actual := _installed_version(name)),
            passed=_same_version(actual, expected),
        )
        for name, expected in package_expectations
    )
    report = EnvironmentReport(
        python_expected=config.environment.python_supported,
        python_actual=python_actual,
        python_passed=python_actual in config.environment.python_supported,
        package_checks=package_checks,
        strict=strict,
    )
    if strict and not report.passed:
        detail = "; ".join(
            [
                f"Python expected {config.environment.python_supported}, found {python_actual}",
                *[
                    f"{item.package} expected {item.expected}, found {item.actual or 'NOT INSTALLED'}"
                    for item in package_checks
                    if not item.passed
                ],
            ]
        )
        raise EnvironmentCompatibilityError(f"Pinned runtime environment check failed: {detail}")
    return report


def configure_deterministic_cpu() -> None:
    """Set deterministic CPU controls before TensorFlow and Keras are imported."""

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")


def load_and_validate_scaler(
    path: Path,
    config: ModelAConfig,
    feature_order: tuple[str, ...],
) -> tuple[Any, ScalerReport]:
    # Security control: this function must only be called after the file hash has passed.
    try:
        scaler = joblib.load(path)
    except Exception as exc:  # joblib can raise several pickle/version exceptions
        raise ArtifactValidationError(f"Unable to load frozen scaler {path}: {exc}") from exc

    class_name = f"{type(scaler).__module__}.{type(scaler).__name__}"
    if type(scaler).__name__ != "StandardScaler":
        raise ArtifactValidationError(f"Expected StandardScaler, found {class_name}")

    feature_count = int(getattr(scaler, "n_features_in_", -1))
    if feature_count != config.feature_count:
        raise ArtifactValidationError(
            f"Scaler feature count mismatch: expected {config.feature_count}, found {feature_count}"
        )

    raw_fitted_rows = getattr(scaler, "n_samples_seen_", -1)
    fitted_rows = int(np.asarray(raw_fitted_rows).reshape(-1)[0])
    if fitted_rows != config.scaler_fitted_rows:
        raise ArtifactValidationError(
            f"Scaler fitted-row mismatch: expected {config.scaler_fitted_rows}, found {fitted_rows}"
        )

    feature_names = getattr(scaler, "feature_names_in_", None)
    has_feature_names = feature_names is not None
    feature_names_match = True
    if has_feature_names:
        feature_names_match = tuple(str(item) for item in feature_names) == feature_order
        if not feature_names_match:
            raise ArtifactValidationError("Scaler feature_names_in_ differs from frozen feature order")

    for attribute in ("mean_", "var_", "scale_"):
        values = np.asarray(getattr(scaler, attribute, None))
        if values.shape != (config.feature_count,) or not np.isfinite(values).all():
            raise ArtifactValidationError(f"Scaler attribute {attribute} is invalid")
    if np.any(np.asarray(scaler.scale_) <= 0):
        raise ArtifactValidationError("Scaler contains a non-positive scale value")

    return scaler, ScalerReport(
        class_name=class_name,
        feature_count=feature_count,
        fitted_rows=fitted_rows,
        has_feature_names=has_feature_names,
        feature_names_match=feature_names_match,
    )


def _normalise_single_shape(shape: Any, *, label: str) -> tuple[int | None, ...]:
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ModelLoadError(f"Frozen model must have one {label}, found {len(shape)}")
        shape = shape[0]
    try:
        return tuple(None if value is None else int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ModelLoadError(f"Unable to interpret model {label}: {shape}") from exc


def _layer_class_names(model: Any) -> tuple[str, ...]:
    return tuple(type(layer).__name__ for layer in getattr(model, "layers", ()))


def _find_single_layer(model: Any, class_name: str) -> Any:
    matches = [layer for layer in getattr(model, "layers", ()) if type(layer).__name__ == class_name]
    if len(matches) != 1:
        raise ModelLoadError(f"Expected one {class_name} layer, found {len(matches)}")
    return matches[0]


def validate_model_contract(model: Any, config: ModelAConfig) -> ModelReport:
    input_shape = _normalise_single_shape(getattr(model, "input_shape", None), label="input")
    output_shape = _normalise_single_shape(getattr(model, "output_shape", None), label="output")
    expected_input = (None, config.sequence_length, config.feature_count)
    if input_shape != expected_input:
        raise ModelLoadError(f"Model input shape mismatch: expected {expected_input}, found {input_shape}")
    if output_shape != (None, 1):
        raise ModelLoadError(f"Model output shape mismatch: expected (None, 1), found {output_shape}")

    try:
        parameter_count = int(model.count_params())
    except Exception as exc:
        raise ModelLoadError(f"Unable to count model parameters: {exc}") from exc
    if parameter_count != config.parameter_count:
        raise ModelLoadError(
            f"Model parameter count mismatch: expected {config.parameter_count}, found {parameter_count}"
        )

    architecture = config.raw["model_contract"]["architecture"]
    conv = _find_single_layer(model, "Conv1D")
    lstm = _find_single_layer(model, "LSTM")
    pool = _find_single_layer(model, "MaxPooling1D")
    spatial_dropout = _find_single_layer(model, "SpatialDropout1D")
    dropout = _find_single_layer(model, "Dropout")
    dense = _find_single_layer(model, "Dense")

    checks = (
        (int(conv.filters), int(architecture["conv_filters_1"]), "Conv1D filters"),
        (tuple(int(v) for v in conv.kernel_size), (int(architecture["kernel_size"]),), "Conv1D kernel"),
        (str(conv.padding), "causal", "Conv1D padding"),
        (tuple(int(v) for v in pool.pool_size), (int(architecture["pool_size"]),), "pool size"),
        (int(lstm.units), int(architecture["lstm_units"]), "LSTM units"),
        (float(lstm.dropout), float(architecture["lstm_dropout"]), "LSTM dropout"),
        (
            float(lstm.recurrent_dropout),
            float(architecture["lstm_recurrent_dropout"]),
            "LSTM recurrent dropout",
        ),
        (float(spatial_dropout.rate), float(architecture["external_dropout"]), "spatial dropout"),
        (float(dropout.rate), float(architecture["external_dropout"]), "external dropout"),
        (int(dense.units), 1, "Dense units"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ModelLoadError(f"{label} mismatch: expected {expected}, found {actual}")
    if getattr(conv.activation, "__name__", "") != "relu":
        raise ModelLoadError("Conv1D activation must be relu")
    if getattr(dense.activation, "__name__", "") != "sigmoid":
        raise ModelLoadError("Final Dense activation must be sigmoid")

    expected_l2 = float(architecture["l2"])
    for layer, label in ((conv, "Conv1D"), (lstm, "LSTM")):
        regularizer = getattr(layer, "kernel_regularizer", None)
        actual_l2 = float(getattr(regularizer, "l2", float("nan")))
        if not np.isclose(actual_l2, expected_l2, rtol=0.0, atol=1e-12):
            raise ModelLoadError(
                f"{label} L2 mismatch: expected {expected_l2}, found {actual_l2}"
            )

    return ModelReport(
        class_name=f"{type(model).__module__}.{type(model).__name__}",
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_count=parameter_count,
        layer_classes=_layer_class_names(model),
    )


def load_and_validate_model(path: Path, config: ModelAConfig) -> tuple[Any, ModelReport]:
    configure_deterministic_cpu()
    try:
        import tensorflow as tf
        import keras
    except Exception as exc:
        raise EnvironmentCompatibilityError(
            "TensorFlow and Keras could not be imported. Install the pinned deployment environment."
        ) from exc

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:
        raise EnvironmentCompatibilityError(f"Unable to enable TensorFlow determinism: {exc}") from exc

    for setter in (
        tf.config.threading.set_inter_op_parallelism_threads,
        tf.config.threading.set_intra_op_parallelism_threads,
    ):
        try:
            setter(1)
        except RuntimeError:
            # TensorFlow may already be initialised by the caller. Deterministic ops remain enabled.
            pass

    try:
        model = keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise ModelLoadError(f"Unable to load frozen Keras model {path}: {exc}") from exc
    return model, validate_model_contract(model, config)


def load_reference_fixture(
    csv_path: Path,
    metadata_path: Path,
    feature_order: tuple[str, ...],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    import hashlib
    import json
    import pandas as pd

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactValidationError(f"Unable to read reference fixture metadata: {exc}") from exc
    if not isinstance(metadata, Mapping):
        raise ArtifactValidationError("Reference fixture metadata root must be an object")
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise ArtifactValidationError(f"Unable to read reference fixture CSV: {exc}") from exc
    if tuple(frame.columns) != feature_order:
        raise ArtifactValidationError(
            "Reference fixture columns do not match the frozen feature order"
        )
    raw = frame.astype(np.float32).to_numpy(dtype=np.float32)
    expected_shape = tuple(int(item) for item in metadata.get("shape", ()))
    if raw.shape != expected_shape:
        raise ArtifactValidationError(
            f"Reference fixture shape mismatch: metadata={expected_shape}, array={raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise ArtifactValidationError("Reference fixture must contain only finite values")

    expected_raw_hash = metadata.get("raw_array_sha256")
    actual_raw_hash = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
    if expected_raw_hash != actual_raw_hash:
        raise ArtifactValidationError(
            f"Reference fixture raw-array hash mismatch: expected {expected_raw_hash}, found {actual_raw_hash}"
        )
    expected_feature_hash = metadata.get("feature_order_sha256")
    actual_feature_hash = hashlib.sha256(
        json.dumps(list(feature_order), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if expected_feature_hash != actual_feature_hash:
        raise ArtifactValidationError(
            "Reference fixture feature-order hash does not match the frozen feature order"
        )
    return raw, metadata


def run_reference_inference(
    model: Any,
    scaler: Any,
    raw_sequence: np.ndarray,
    metadata: Mapping[str, Any],
    feature_order: tuple[str, ...],
    *,
    tolerance: float = 1e-5,
    thresholds: Iterable[float] = (0.47, 0.50, 0.53, 0.55),
) -> ReferenceInferenceReport:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    expected_probability = float(metadata["expected_probability"])
    if len(feature_order) != raw_sequence.shape[1]:
        raise ArtifactValidationError(
            "Reference inference feature-order length differs from the fixture width"
        )
    # Notebook 7 fitted and transformed a DataFrame after casting it to float32.
    import pandas as pd

    frame = pd.DataFrame(raw_sequence.astype(np.float32), columns=list(feature_order))
    transformed = np.asarray(scaler.transform(frame), dtype=np.float32)
    if transformed.shape != raw_sequence.shape or not np.isfinite(transformed).all():
        raise ArtifactValidationError(
            f"Scaled reference sequence is invalid: expected {raw_sequence.shape}, found {transformed.shape}"
        )
    batch = transformed[np.newaxis, ...]
    try:
        output = model(batch, training=False)
        if hasattr(output, "detach"):
            output = output.detach()
        if hasattr(output, "cpu"):
            output = output.cpu()
        if hasattr(output, "numpy"):
            output = output.numpy()
        actual_probability = float(np.asarray(output).reshape(-1)[0])
    except Exception as exc:
        raise ModelLoadError(f"Frozen model inference failed: {exc}") from exc
    if not np.isfinite(actual_probability) or not 0.0 <= actual_probability <= 1.0:
        raise ModelLoadError(f"Model produced invalid probability: {actual_probability}")
    difference = abs(actual_probability - expected_probability)
    threshold_flip_count = sum(
        (actual_probability >= threshold) != (expected_probability >= threshold)
        for threshold in thresholds
    )
    passed = difference <= tolerance and threshold_flip_count == 0
    if not passed:
        raise ModelLoadError(
            "Reference inference mismatch: "
            f"expected={expected_probability:.10f}, actual={actual_probability:.10f}, "
            f"difference={difference:.3e}, threshold_flips={threshold_flip_count}"
        )
    return ReferenceInferenceReport(
        fixture_end_utc=str(metadata["sequence_end_utc"]),
        expected_probability=expected_probability,
        actual_probability=actual_probability,
        absolute_difference=difference,
        tolerance=tolerance,
        threshold_flip_count=threshold_flip_count,
        passed=passed,
    )


def report_to_dict(report: Any) -> dict[str, Any]:
    return asdict(report)
