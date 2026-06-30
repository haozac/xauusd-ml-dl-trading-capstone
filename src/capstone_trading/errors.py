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
