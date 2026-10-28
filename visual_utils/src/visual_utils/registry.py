"""One lazy provider registry shared by every application."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from importlib import metadata as package_metadata

from .config import DatasetConfig
from .dataset import DatasetLoader
from .models import DatasetMetadata
from .errors import DatasetConfigurationError, DatasetError, DatasetIOError


def _nuscenes(config):
    from .nuscenes.loader import NuScenesLoader
    return NuScenesLoader(config)


def _waymo(config):
    from .waymo.loader import WaymoLoader
    return WaymoLoader(config)


_REGISTRY: dict[str, Callable[[DatasetConfig], DatasetLoader]] = {
    "nuscenes": _nuscenes,
    "waymo": _waymo,
}

ENTRY_POINT_GROUP = "visual_utils.loaders"


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigurationError("Dataset registry name must be a nonempty string.")
    return value.strip().lower()


@lru_cache(maxsize=1)
def _installed_loaders() -> dict[str, package_metadata.EntryPoint]:
    """Read installed names once per process without importing provider code."""
    providers = {}
    for entry in package_metadata.entry_points(group=ENTRY_POINT_GROUP):
        key = _name(entry.name)
        if key in providers:
            raise DatasetConfigurationError(
                f"Multiple installed loaders declare {key!r}: "
                f"{providers[key].value!r} and {entry.value!r}."
            )
        providers[key] = entry
    return providers


def _check_conflicts() -> dict[str, package_metadata.EntryPoint]:
    installed = _installed_loaders()
    conflicts = set(_REGISTRY).intersection(installed)
    if conflicts:
        raise DatasetConfigurationError(
            f"Installed dataset loaders conflict with registered names: {', '.join(sorted(conflicts))}."
        )
    return installed


def register_loader(name: str, constructor: Callable[[DatasetConfig], DatasetLoader]) -> None:
    """Register a provider in this process; installed names cannot be replaced."""
    key = _name(name)
    if not callable(constructor):
        raise DatasetConfigurationError("Dataset constructor must be callable.")
    if key in _REGISTRY or key in _installed_loaders():
        raise DatasetConfigurationError(f"Dataset {key!r} is already registered.")
    _REGISTRY[key] = constructor


def available_loaders() -> tuple[str, ...]:
    """List built-in, explicit, and installed provider names without loading them."""
    return tuple(sorted(set(_REGISTRY).union(_check_conflicts())))


def create_loader(name: str, config: DatasetConfig) -> DatasetLoader:
    """Construct and validate one provider, importing only the selected plugin."""
    key = _name(name)
    if not isinstance(config, DatasetConfig):
        raise DatasetConfigurationError("create_loader requires DatasetConfig.")
    installed = _check_conflicts()
    if key in _REGISTRY:
        constructor = _REGISTRY[key]
    elif key in installed:
        try:
            constructor = installed[key].load()
        except Exception as error:
            raise DatasetConfigurationError(
                f"Cannot import dataset plugin {key!r} ({installed[key].value}): {error}"
            ) from error
        if not callable(constructor):
            raise DatasetConfigurationError(f"Dataset plugin {key!r} must export a callable constructor.")
    else:
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
        if not isinstance(metadata, DatasetMetadata):
            raise DatasetConfigurationError(f"Dataset {key!r} must provide DatasetMetadata.")
        for capability, method in (("point_cloud", "_load_point_cloud"), ("cameras", "_load_cameras")):
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
