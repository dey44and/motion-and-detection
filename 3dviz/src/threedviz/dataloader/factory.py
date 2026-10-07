"""A registry factory that keeps dataset selection out of the UI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .base import DatasetAdapter

_REGISTRY: dict[str, Callable[..., DatasetAdapter]] = {}


def _dataset_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Dataset name must be a nonempty string.")
    return name.strip().lower()


def register_dataset(name: str, constructor: Callable[..., DatasetAdapter]) -> None:
    """Register a constructor accepting ``root=`` and ``version=`` keywords.

    Existing names cannot be overwritten accidentally. Import a module which
    registers its adapter before creating it; the viewer needs no new imports.
    """

    key = _dataset_name(name)
    if not callable(constructor):
        raise TypeError("Dataset constructor must be callable.")
    if key in _REGISTRY:
        raise ValueError(f"Dataset {key!r} is already registered.")
    _REGISTRY[key] = constructor


def available_datasets() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create_dataset(
    name: str,
    root: str | Path | None = None,
    version: str = "v1.0-mini",
) -> DatasetAdapter:
    key = _dataset_name(name)
    try:
        constructor = _REGISTRY[key]
    except KeyError:
        choices = ", ".join(available_datasets())
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {choices}.") from None
    adapter = constructor(root=root, version=version)
    if not isinstance(adapter, DatasetAdapter):
        raise TypeError(f"Constructor for {key!r} must return a DatasetAdapter.")
    return adapter
