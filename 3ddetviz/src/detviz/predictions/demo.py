"""Deterministic overlapping predictions make demo filters observable."""

from __future__ import annotations

import threading

import numpy as np

from threedviz.dataloader.models import BoundingBox3D, FrameData

from .base import PredictionSource
from .models import Detection, PredictionError, PredictionFrame, PredictionInfo


def make_demo_predictions(frame: FrameData) -> PredictionFrame:
    detections = []
    for index, box in enumerate(frame.boxes):
        score = (0.92, 0.74, 0.51, 0.28)[index % 4]
        center = box.center + np.array([0.18, -0.12, 0.05])
        identifier = f"{frame.info.token}:{len(detections)}"
        predicted_box = BoundingBox3D(center, box.size * 1.02, box.rotation, box.class_name, identifier)
        detections.append(Detection(predicted_box, score, identifier))
        if index % 3 == 0:
            identifier = f"{frame.info.token}:{len(detections)}"
            duplicate = BoundingBox3D(center + [0.12, 0.08, 0], box.size * 1.01, box.rotation, box.class_name, identifier)
            detections.append(Detection(duplicate, score * 0.72, identifier))
    return PredictionFrame(frame.info.token, tuple(detections))


class DemoPredictionSource(PredictionSource):
    """Generate predictions against any adapter exposing ordinary demo frames."""

    def __init__(self, dataset):
        self.dataset = dataset
        self._tokens = frozenset(frame.token for scene in dataset.list_scenes() for frame in dataset.list_frames(scene.token))
        self._info = PredictionInfo("Demo predictions", "demo", "lidar", "pre_nms", {"synthetic": True})
        self._closed = False
        self._lock = threading.RLock()

    @property
    def info(self):
        return self._info

    @property
    def frame_tokens(self):
        return self._tokens

    @property
    def class_names(self):
        return self.dataset.class_names

    def get_frame(self, token):
        with self._lock:
            if self._closed:
                raise PredictionError("Prediction source is closed.")
            return make_demo_predictions(self.dataset.load_frame(token)) if token in self._tokens else None

    def close(self):
        with self._lock:
            self._closed = True
