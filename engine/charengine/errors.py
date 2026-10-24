class CharEngineError(Exception):
    """Base class for errors that are safe to show to API users."""


class InputImageError(CharEngineError):
    """The uploaded image is invalid or unusable (format, size, no subject found)."""


class ReconstructionError(CharEngineError):
    """The 3D model produced no usable geometry."""


class ExportValidationError(CharEngineError):
    """The exported GLB failed validation."""
