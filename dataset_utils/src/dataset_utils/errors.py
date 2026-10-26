"""Errors shared by dataset readers and their application adapters."""


class DatasetError(RuntimeError):
    """A dataset cannot be read or has inconsistent metadata."""


class DatasetConfigurationError(DatasetError, ValueError):
    """Invalid dataset configuration or an incompatible contract version."""


class UnsupportedCapabilityError(DatasetError):
    """The configured loader does not offer this operation."""


class UnknownIdentifierError(DatasetError):
    """A scene or frame identifier is unknown."""


class DataUnavailableError(DatasetError):
    """A known observation does not contain the requested data."""


class DatasetIOError(DatasetError):
    """A declared dataset source cannot be accessed or read."""


class DatasetValidationError(DatasetError, ValueError):
    """Source data or a normalized record is corrupt or inconsistent."""


class DatasetClosedError(DatasetError):
    """An operation was requested after a loader was closed."""
