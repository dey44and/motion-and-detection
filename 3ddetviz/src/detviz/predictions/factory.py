"""Register prediction formats separately from dataset providers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from .base import PredictionSource
from .models import PredictionError


class PredictionOpener(Protocol):
    """Open one file; cancellation must leave no published partial cache."""

    def __call__(
        self, path: Path, *, cache_dir: str | Path | None = None,
        progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> PredictionSource: ...


_REGISTRY: dict[str, PredictionOpener] = {}
_ALIASES = {"json": "nuscenes_json", "nuscenes": "nuscenes_json", "pickle": "pcdet_pickle", "pcdet": "pcdet_pickle"}


def register_prediction_format(name: str, opener: PredictionOpener) -> None:
    """Register a canonical name in this process before creating the viewer.

    Names are case insensitive. Built-in aliases and ``auto`` are reserved so
    every successfully registered opener remains addressable by its own name.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Prediction format name must be nonempty.")
    name = name.strip().lower()
    if not callable(opener):
        raise TypeError("Prediction opener must be callable.")
    if name in _REGISTRY or name == "auto" or name in _ALIASES:
        raise ValueError(f"Prediction format {name!r} is already registered or reserved.")
    _REGISTRY[name] = opener


def available_prediction_formats() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def open_predictions(path, format: str = "auto", *, cache_dir=None, progress=None, cancelled=None) -> PredictionSource:
    """Open a registered source; the caller owns and must close the result.

    Automatic detection recognizes the built-in JSON/pickle suffixes only.
    Custom formats use an explicit registered name, regardless of suffix.
    """
    if not isinstance(format, str):
        raise PredictionError("Prediction format must be a string.")
    format = format.strip().lower()
    path = Path(path).expanduser()
    if format == "auto":
        format = {".json": "nuscenes_json", ".pkl": "pcdet_pickle", ".pickle": "pcdet_pickle"}.get(path.suffix.lower())
        if format is None:
            raise PredictionError("Cannot infer prediction format. Select nuscenes_json or pcdet_pickle explicitly.")
    format = _ALIASES.get(format, format)
    try:
        opener = _REGISTRY[format]
    except KeyError:
        raise PredictionError(f"Unknown prediction format {format!r}. Available: {', '.join(available_prediction_formats())}.") from None
    if not path.is_file():
        raise PredictionError(f"Prediction file does not exist: {path}")
    source = opener(path, cache_dir=cache_dir, progress=progress, cancelled=cancelled)
    if not isinstance(source, PredictionSource):
        raise TypeError(f"Prediction opener for {format!r} must return a PredictionSource.")
    return source
