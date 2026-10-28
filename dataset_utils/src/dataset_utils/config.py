"""Immutable local dataset configuration shared by every provider."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import MappingProxyType

from .errors import DatasetConfigurationError


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
            raise DatasetConfigurationError("Dataset sources must be local filesystem paths.")
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


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetConfigurationError(f"Duplicate configuration key {key!r}.")
        result[key] = value
    return result


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Read a strict JSON configuration, resolving paths relative to its file.

    The object accepts exactly the fields of :class:`DatasetConfig`. A relative
    root is anchored to the configuration directory; relative named sources
    are anchored to that root, or to the configuration directory without one.
    Reading configuration does not import or construct a dataset provider.
    """
    config_path = _path(path, Path.cwd())
    try:
        values = json.loads(config_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, ValueError) as error:
        raise DatasetConfigurationError(f"Cannot read dataset configuration {config_path}: {error}") from error
    if not isinstance(values, dict):
        raise DatasetConfigurationError(f"Dataset configuration {config_path} must be a JSON object.")
    unknown = set(values).difference({"root", "release", "storage_format", "source_paths", "options"})
    if unknown:
        raise DatasetConfigurationError(f"Unknown dataset configuration fields: {', '.join(sorted(unknown))}.")
    root = values.get("root")
    if root is not None:
        root = _path(root, config_path.parent)
        values["root"] = root
    sources = values.get("source_paths", {})
    if not isinstance(sources, Mapping):
        raise DatasetConfigurationError("source_paths must be a JSON object.")
    values["source_paths"] = {
        key: _path(value, root or config_path.parent) for key, value in sources.items()
    }
    return DatasetConfig(**values)
