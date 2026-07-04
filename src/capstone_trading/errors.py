"""Typed exceptions used by the deployment verification pipeline."""


class Step1VerificationError(RuntimeError):
    """Base class for an expected Stage 1 Step 1 verification failure."""


class ConfigurationError(Step1VerificationError):
    """Raised when a frozen configuration is missing or internally invalid."""


class IntegrityError(Step1VerificationError):
    """Raised when a frozen file hash or immutable manifest check fails."""


class EnvironmentCompatibilityError(Step1VerificationError):
    """Raised when the local Python or package environment is incompatible."""


class ArtifactValidationError(Step1VerificationError):
    """Raised when a frozen model, scaler, feature or parameter artefact is invalid."""


class ModelLoadError(Step1VerificationError):
    """Raised when the frozen Keras model cannot be loaded or validated."""


class Step2ParityError(RuntimeError):
    """Base class for an expected Stage 1 Step 2 parity failure."""


class HistoricalDataError(Step2ParityError):
    """Raised when historical bars or model-ready reference data are invalid."""


class FeatureParityError(Step2ParityError):
    """Raised when reconstructed features differ from the official dataset."""


class SequenceParityError(Step2ParityError):
    """Raised when sequence continuity or alignment differs from Notebook 7."""
