"""Register prediction formats separately from dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import PredictionSource
from .models import PredictionError

_REGISTRY: dict[str, Callable] = {}


def register_prediction_format(name: str, opener: Callable) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Prediction format name must be nonempty.")
    name = name.strip().lower()
    if not callable(opener):
        raise TypeError("Prediction opener must be callable.")
    if name in _REGISTRY or name == "auto":
        raise ValueError(f"Prediction format {name!r} is already registered or reserved.")
    _REGISTRY[name] = opener


def available_prediction_formats() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def open_predictions(path, format: str = "auto", *, cache_dir=None, progress=None, cancelled=None) -> PredictionSource:
    if not isinstance(format, str):
        raise PredictionError("Prediction format must be a string.")
    format = format.strip().lower()
    path = Path(path).expanduser()
    if format == "auto":
        format = {".json": "nuscenes_json", ".pkl": "pcdet_pickle", ".pickle": "pcdet_pickle"}.get(path.suffix.lower())
        if format is None:
            raise PredictionError("Cannot infer prediction format. Select nuscenes_json or pcdet_pickle explicitly.")
    format = {"json": "nuscenes_json", "nuscenes": "nuscenes_json", "pickle": "pcdet_pickle", "pcdet": "pcdet_pickle"}.get(format, format)
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
