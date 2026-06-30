import numpy as np
import pytest

from capstone_trading.config import load_model_a_config
from capstone_trading.errors import ModelLoadError
from capstone_trading.model_loader import (
    check_runtime_environment,
    run_reference_inference,
    validate_model_contract,
)


class L2:
    l2 = 1e-5


class Conv1D:
    filters = 32
    kernel_size = (3,)
    padding = "causal"
    activation = staticmethod(lambda value: value)
    kernel_regularizer = L2()


Conv1D.activation.__name__ = "relu"


class SpatialDropout1D:
    rate = 0.2


class MaxPooling1D:
    pool_size = (2,)


class LSTM:
    units = 64
    dropout = 0.0
    recurrent_dropout = 0.0
    kernel_regularizer = L2()


class Dropout:
    rate = 0.2


class Dense:
    units = 1
    activation = staticmethod(lambda value: value)


Dense.activation.__name__ = "sigmoid"


class FakeModel:
    input_shape = (None, 48, 51)
    output_shape = (None, 1)
    layers = [Conv1D(), SpatialDropout1D(), MaxPooling1D(), LSTM(), Dropout(), Dense()]

    def count_params(self) -> int:
        return 29825


class PredictingModel(FakeModel):
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, batch, training=False):
        assert batch.shape == (1, 48, 51)
        assert training is False
        return np.array([[self.probability]], dtype=np.float32)


class IdentityScaler:
    def transform(self, values):
        return values


def test_validate_model_contract_accepts_expected_architecture() -> None:
    config = load_model_a_config("config/model_a_frozen.yaml")
    report = validate_model_contract(FakeModel(), config)
    assert report.input_shape == (None, 48, 51)
    assert report.output_shape == (None, 1)
    assert report.parameter_count == 29825


def test_validate_model_contract_rejects_wrong_input_shape() -> None:
    config = load_model_a_config("config/model_a_frozen.yaml")
    model = FakeModel()
    model.input_shape = (None, 47, 51)
    with pytest.raises(ModelLoadError, match="input shape mismatch"):
        validate_model_contract(model, config)


def test_reference_inference_requires_numeric_and_decision_parity() -> None:
    raw = np.zeros((48, 51), dtype=np.float32)
    metadata = {
        "sequence_end_utc": "2025-01-01T00:00:00+00:00",
        "expected_probability": 0.585,
    }
    report = run_reference_inference(
        PredictingModel(0.58500004),
        IdentityScaler(),
        raw,
        metadata,
        tuple(f"f{i}" for i in range(51)),
        tolerance=1e-5,
    )
    assert report.passed
    assert report.threshold_flip_count == 0


def test_reference_inference_rejects_threshold_flip_even_inside_loose_tolerance() -> None:
    raw = np.zeros((48, 51), dtype=np.float32)
    metadata = {
        "sequence_end_utc": "2025-01-01T00:00:00+00:00",
        "expected_probability": 0.5299,
    }
    with pytest.raises(ModelLoadError, match="threshold_flips=1"):
        run_reference_inference(
            PredictingModel(0.5301),
            IdentityScaler(),
            raw,
            metadata,
            tuple(f"f{i}" for i in range(51)),
            tolerance=1e-3,
        )


def test_environment_report_is_non_strict_when_requested() -> None:
    config = load_model_a_config("config/model_a_frozen.yaml")
    report = check_runtime_environment(config, strict=False)
    assert report.strict is False
    assert report.python_actual
