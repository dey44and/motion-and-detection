"""Convert world predictions into the dataset frame through public pose data."""

from __future__ import annotations

import numpy as np

from threedviz.dataloader.models import BoundingBox3D, FrameData

from .models import Detection, PredictionError, PredictionFrame, rigid_matrix


def normalize_frame(
    prediction_frame: PredictionFrame,
    frame: FrameData,
    *,
    coordinate_frame: str = "lidar",
) -> PredictionFrame:
    if prediction_frame.sample_token != frame.info.token:
        raise PredictionError("Prediction and dataset sample tokens do not match.")
    if coordinate_frame not in {"world", "lidar"}:
        raise PredictionError("Prediction coordinate frame must be 'world' or 'lidar'.")
    transform = None
    if coordinate_frame == "world":
        value = getattr(frame, "reference_sensor_to_world", None)
        if value is None:
            raise PredictionError("This dataset frame has no reference_sensor_to_world pose for world predictions.")
        transform = rigid_matrix(value, "reference_sensor_to_world")
    detections = []
    for detection in prediction_frame.detections:
        box = detection.box
        if transform is None:
            center, rotation, velocity = box.center, box.rotation, detection.velocity
        else:
            inverse_rotation = transform[:3, :3].T
            center = inverse_rotation @ (box.center - transform[:3, 3])
            rotation = inverse_rotation @ box.rotation
            velocity = None if detection.velocity is None else inverse_rotation @ detection.velocity
        detections.append(Detection(
            BoundingBox3D(center, box.size, rotation, box.class_name, box.token),
            detection.score, detection.identifier, velocity,
        ))
    return PredictionFrame(prediction_frame.sample_token, tuple(detections))
