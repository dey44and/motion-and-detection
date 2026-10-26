"""One lazy provider registry shared by every application."""

from __future__ import annotations

from collections.abc import Callable

from .config import DatasetConfig
from .loader import DatasetLoader
from .models import DatasetMetadata
from ..errors import DatasetConfigurationError, DatasetError, DatasetIOError


def _nuscenes(config):
    from ..nuscenes.loader import NuScenesLoader
    return NuScenesLoader(config)


def _waymo(config):
    from ..waymo.loader import WaymoLoader
    return WaymoLoader(config)


_REGISTRY: dict[str, Callable[[DatasetConfig], DatasetLoader]] = {
    "nuscenes": _nuscenes,
    "waymo": _waymo,
}


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigurationError("Dataset registry name must be a nonempty string.")
    return value.strip().lower()


def register_loader(name: str, constructor: Callable[[DatasetConfig], DatasetLoader]) -> None:
    key = _name(name)
    if not callable(constructor):
        raise DatasetConfigurationError("Dataset constructor must be callable.")
    if key in _REGISTRY:
        raise DatasetConfigurationError(f"Dataset {key!r} is already registered.")
    _REGISTRY[key] = constructor


def available_loaders() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create_loader(name: str, config: DatasetConfig) -> DatasetLoader:
    key = _name(name)
    if not isinstance(config, DatasetConfig):
        raise DatasetConfigurationError("create_loader requires DatasetConfig.")
    try:
        constructor = _REGISTRY[key]
    except KeyError:
        raise DatasetConfigurationError(
            f"Unknown dataset {name!r}. Available loaders: {', '.join(available_loaders())}."
        ) from None
    try:
        loader = constructor(config)
    except DatasetError:
        raise
    except OSError as error:
        raise DatasetIOError(f"Cannot open dataset {key!r}: {error}") from error
    except (TypeError, ValueError) as error:
        raise DatasetConfigurationError(f"Cannot construct dataset {key!r}: {error}") from error
    if not isinstance(loader, DatasetLoader):
        raise DatasetConfigurationError(f"Constructor for {key!r} must return DatasetLoader.")
    try:
        metadata = loader.metadata
        if not isinstance(metadata, DatasetMetadata) or metadata.contract_version != 1:
            raise DatasetConfigurationError(f"Dataset {key!r} does not implement contract version 1.")
        for capability, method in (("point_cloud", "load_point_cloud"), ("cameras", "load_cameras")):
            if capability in metadata.capabilities and getattr(type(loader), method) is getattr(DatasetLoader, method):
                raise DatasetConfigurationError(f"Dataset {key!r} advertises {capability} without implementing {method}.")
        loader._ensure_open()
    except Exception as error:
        try:
            loader.close()
        except Exception as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(f"Provider cleanup also failed: {cleanup_error}")
        raise
    return loader
