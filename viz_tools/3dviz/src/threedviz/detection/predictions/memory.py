"""A reference source for predictions produced directly by Python programs."""

from __future__ import annotations

from collections.abc import Iterable

from .base import PredictionSource
from .models import PredictionError, PredictionFrame, PredictionInfo


class InMemoryPredictionSource(PredictionSource):
    """Retain canonical frames, including empty coverage, in input order.

    This source eagerly consumes its iterable and is intended for small exports
    and interactive analysis. Use an indexed file source for large results.
    Prediction records are already immutable, so reads share those records.
    """

    def __init__(self, frames: Iterable[PredictionFrame], *, info: PredictionInfo):
        if not isinstance(info, PredictionInfo):
            raise TypeError("info must be a PredictionInfo.")
        self._info = info
        self._frames: dict[str, PredictionFrame] = {}
        classes: dict[str, None] = {}
        for frame in frames:
            if not isinstance(frame, PredictionFrame):
                raise TypeError("frames must contain PredictionFrame instances.")
            if frame.sample_token in self._frames:
                raise PredictionError(f"Duplicate prediction frame {frame.sample_token!r}.")
            self._frames[frame.sample_token] = frame
            classes.update(dict.fromkeys(item.box.class_id for item in frame.detections))
        self._class_names = tuple(classes)
        self._frame_tokens = frozenset(self._frames)
        self._closed = False

    @property
    def info(self) -> PredictionInfo:
        return self._info

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    @property
    def frame_tokens(self) -> frozenset[str]:
        return self._frame_tokens

    def get_frame(self, token: str) -> PredictionFrame | None:
        if self._closed:
            raise PredictionError("Prediction source is closed.")
        return self._frames.get(token)

    def close(self) -> None:
        self._closed = True
        self._frames.clear()
