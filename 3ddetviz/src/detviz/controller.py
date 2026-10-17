"""Bind dataset frames to predictions without coupling either file format."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, replace

from threedviz.dataloader import DatasetAdapter, FrameData

from .predictions import normalize_frame
from .predictions.demo import make_demo_predictions
from .predictions.models import PredictionFrame
from .processing.filters import FilterConfig, FilterResult, filter_predictions


@dataclass(frozen=True)
class ReviewFrame(FrameData):
    predictions: PredictionFrame | None = None
    filtered: FilterResult | None = None


def refilter(frame: ReviewFrame, config: FilterConfig) -> ReviewFrame:
    return replace(frame, filtered=None if frame.predictions is None else filter_predictions(frame.predictions, config))


class ReviewDatasetAdapter(DatasetAdapter):
    """A composed adapter; clouds/GT retain their original dataset semantics."""

    def __init__(self, base, *, split="val", source=None, config=None, demo=False):
        self.base = base
        self.split = split
        self.source = source
        self.config = FilterConfig() if config is None else config
        self.demo = demo
        self._frame_cache = OrderedDict()

    @property
    def class_names(self):
        additional = () if self.source is None else self.source.class_names
        return tuple(dict.fromkeys((*self.base.class_names, *additional)))

    @property
    def supports_camera_rgb(self):
        return self.base.supports_camera_rgb

    @property
    def supports_point_timing(self):
        return self.base.supports_point_timing

    def list_scenes(self):
        return self.base.list_scenes_for_split(self.split)

    def list_scenes_for_split(self, split="all"):
        return self.base.list_scenes_for_split(split)

    def list_frames(self, scene_token):
        return self.base.list_frames(scene_token)

    def compose(self, frame):
        predictions = None
        if self.demo:
            predictions = make_demo_predictions(frame)
        elif self.source is not None:
            raw = self.source.get_frame(frame.info.token)
            if raw is not None:
                predictions = normalize_frame(raw, frame, coordinate_frame=self.source.info.coordinate_frame)
        attributes = {field.name: getattr(frame, field.name) for field in fields(FrameData)}
        result = ReviewFrame(**attributes, predictions=predictions)
        return refilter(result, self.config)

    def load_frame(self, token):
        if token not in self._frame_cache:
            self._frame_cache[token] = self.base.load_frame(token)
            if len(self._frame_cache) > 2:
                self._frame_cache.popitem(last=False)
        self._frame_cache.move_to_end(token)
        return self.compose(self._frame_cache[token])

    def prepare_frame(self, frame, *, camera_rgb=False, point_timing=False):
        prepared = self.base.prepare_frame(frame, camera_rgb=camera_rgb, point_timing=point_timing)
        # dataclasses.replace in existing adapters retains ReviewFrame fields;
        # custom adapters may return a plain FrameData instead.
        if not isinstance(prepared, ReviewFrame):
            prepared = self.compose(prepared)
        return prepared
