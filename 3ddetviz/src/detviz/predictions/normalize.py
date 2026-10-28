"""Convert world predictions into a cloud's declared reference coordinates."""

from __future__ import annotations

from dataset_utils import Box3D, PointCloudFrame

from .models import Detection, PredictionError, PredictionFrame


def normalize_frame(
    prediction_frame: PredictionFrame,
    observation: PointCloudFrame,
    *,
    coordinate_frame: str = "lidar",
) -> PredictionFrame:
    """Preserve canonical reference predictions; transform world boxes once."""
    if prediction_frame.sample_token != observation.info.frame_id:
        raise PredictionError("Prediction and dataset frame IDs do not match.")
    if coordinate_frame not in {"world", "lidar"}:
        raise PredictionError("Prediction coordinate frame must be 'world' or 'lidar'.")
    if coordinate_frame == "lidar":
        return prediction_frame
    transform = observation.T_world_from_reference
    if transform is None:
        raise PredictionError("This dataset frame has no T_world_from_reference pose for world predictions.")
    inverse_rotation = transform[:3, :3].T
    detections = []
    for detection in prediction_frame.detections:
        box = detection.box
        normalized = Box3D(
            box.annotation_id, box.class_id,
            inverse_rotation @ (box.center - transform[:3, 3]), box.size,
            inverse_rotation @ box.rotation, box.instance_id, box.timing_override,
        )
        velocity = None if detection.velocity is None else inverse_rotation @ detection.velocity
        detections.append(Detection(normalized, detection.score, detection.identifier, velocity))
    return PredictionFrame(prediction_frame.sample_token, tuple(detections))
