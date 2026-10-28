"""Deterministic overlapping predictions make demo filters observable."""

from __future__ import annotations

import numpy as np

from dataset_utils import Box3D, Modality, PointCloudFrame

from .base import PredictionSource
from .models import Detection, PredictionError, PredictionFrame, PredictionInfo


def make_demo_predictions(frame: PointCloudFrame) -> PredictionFrame:
    detections = []
    for index, box in enumerate(frame.annotations.boxes):
        score = (0.92, 0.74, 0.51, 0.28)[index % 4]
        center = box.center + np.array([0.18, -0.12, 0.05])
        identifier = f"{frame.info.frame_id}:{len(detections)}"
        predicted = Box3D(identifier, box.class_id, center, box.size * 1.02, box.rotation)
        detections.append(Detection(predicted, score, identifier))
        if index % 3 == 0:
            identifier = f"{frame.info.frame_id}:{len(detections)}"
            duplicate = Box3D(identifier, box.class_id, center + [0.12, 0.08, 0], box.size * 1.01, box.rotation)
            detections.append(Detection(duplicate, score * 0.72, identifier))
    return PredictionFrame(frame.info.frame_id, tuple(detections))


class DemoPredictionSource(PredictionSource):
    """Generate predictions from a borrowed canonical dataset loader."""

    def __init__(self, dataset):
        self.dataset = dataset
        self._tokens = frozenset(frame.frame_id for scene in dataset.list_scenes()
                                 for frame in dataset.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD))
        self._info = PredictionInfo("Demo predictions", "demo", "lidar", "pre_nms", {"synthetic": True})
        self._closed = False

    @property
    def info(self):
        return self._info

    @property
    def frame_tokens(self):
        return self._tokens

    @property
    def class_names(self):
        return tuple(item.class_id for item in self.dataset.metadata.classes)

    def get_frame(self, token):
        if self._closed:
            raise PredictionError("Prediction source is closed.")
        return make_demo_predictions(self.dataset.load_point_cloud(token)) if token in self._tokens else None

    def close(self):
        self._closed = True
