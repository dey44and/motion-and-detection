"""A registry factory for adding datasets without changing viewer code."""

from collections.abc import Callable
from pathlib import Path

from dataset_utils import available_loaders

from .base import DatasetAdapter

_REGISTRY: dict[str, Callable[..., DatasetAdapter]] = {}


def _dataset_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Dataset name must be a nonempty string.")
    return name.strip().lower()


def register_dataset(name: str, constructor: Callable[..., DatasetAdapter]) -> None:
    """Register an adapter constructor accepting ``root=`` and ``version=``."""
    key = _dataset_name(name)
    if not callable(constructor):
        raise TypeError("Dataset constructor must be callable.")
    if key in _REGISTRY or key in available_loaders():
        raise ValueError(f"Dataset {key!r} is already registered.")
    _REGISTRY[key] = constructor


def available_datasets() -> tuple[str, ...]:
    return tuple(sorted(set(_REGISTRY).union(available_loaders())))


def create_dataset(name: str, root: str | Path | None = None, version: str = "v1.0-mini", *, config=None) -> DatasetAdapter:
    key = _dataset_name(name)
    if key not in _REGISTRY and key in available_loaders():
        from .shared import open_shared
        return open_shared(key, root=root, version=version, config=config)
    if config is not None:
        raise ValueError("DatasetConfig is supported by shared loaders; legacy adapters use root/version.")
    try:
        constructor = _REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {', '.join(available_datasets())}.") from None
    adapter = constructor(root=root, version=version)
    if not isinstance(adapter, DatasetAdapter):
        raise TypeError(f"Constructor for {key!r} must return a DatasetAdapter.")
    return adapter
