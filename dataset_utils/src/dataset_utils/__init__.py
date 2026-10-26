"""GUI-independent dataset readers and the version-1 extension contract."""

from .contracts import *
from .contracts import __all__ as _contract_exports
from .errors import (
    DataUnavailableError, DatasetClosedError, DatasetConfigurationError,
    DatasetError, DatasetIOError, DatasetValidationError,
    UnknownIdentifierError, UnsupportedCapabilityError,
)

__version__ = "0.2.0"
__all__ = list(_contract_exports) + [
    "DataUnavailableError", "DatasetClosedError", "DatasetConfigurationError",
    "DatasetError", "DatasetIOError", "DatasetValidationError",
    "UnknownIdentifierError", "UnsupportedCapabilityError",
]
