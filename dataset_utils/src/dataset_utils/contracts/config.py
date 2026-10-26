"""Immutable local dataset configuration shared by every provider."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType

from ..errors import DatasetConfigurationError


def _freeze_option(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetConfigurationError("Dataset options cannot contain non-finite numbers.")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise DatasetConfigurationError("Dataset option keys must be nonempty strings.")
        return MappingProxyType({key: _freeze_option(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_option(item) for item in value)
    raise DatasetConfigurationError(f"Unsupported dataset option value type: {type(value).__name__}.")


def _path(value, base):
    try:
        if "://" in str(value):
            raise DatasetConfigurationError("Version 1 dataset sources must be local filesystem paths.")
        result = Path(value).expanduser()
        return (base / result).resolve() if not result.is_absolute() else result.resolve()
    except (TypeError, ValueError, OSError) as error:
        raise DatasetConfigurationError(f"Invalid dataset path {value!r}: {error}") from error


@dataclass(frozen=True)
class DatasetConfig:
    root: Path | None = None
    release: str | None = None
    storage_format: str = "auto"
    source_paths: Mapping[str, Path] = field(default_factory=dict)
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        base = Path.cwd()
        root = None if self.root is None else _path(self.root, base)
        if self.release is not None and (not isinstance(self.release, str) or not self.release.strip()):
            raise DatasetConfigurationError("release must be a nonempty string or None.")
        if not isinstance(self.storage_format, str) or not self.storage_format.strip():
            raise DatasetConfigurationError("storage_format must be a nonempty string.")
        if not isinstance(self.source_paths, Mapping) or not isinstance(self.options, Mapping):
            raise DatasetConfigurationError("source_paths and options must be mappings.")
        if any(not isinstance(key, str) or not key for key in self.source_paths):
            raise DatasetConfigurationError("Source names must be nonempty strings.")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "release", None if self.release is None else self.release.strip())
        object.__setattr__(self, "storage_format", self.storage_format.strip().lower())
        object.__setattr__(self, "source_paths", MappingProxyType({
            key: _path(value, root or base) for key, value in self.source_paths.items()
        }))
        object.__setattr__(self, "options", _freeze_option(self.options))
