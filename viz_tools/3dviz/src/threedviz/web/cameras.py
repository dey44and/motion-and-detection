"""Camera images and overlays for one committed browser frame revision."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from types import MappingProxyType

import numpy as np
from PIL import Image

from threedviz.detection.predictions import Detection
from visual_utils import (
    Box3D, CameraFrameSet, CameraObservation, Capability, DatasetLoader, DataUnavailableError,
    DatasetValidationError, Modality, PointCloudFrame, ProjectionResult, UnsupportedCapabilityError,
)
from visual_utils.box_projection import DEFAULT_NEAR_PLANE, project_box


_POINT_ROW = np.dtype([("pixel", "<f4", (2,)), ("index", "<u4")])
_IDENTITY = np.eye(4)
_THUMBNAIL_SIZE = 256


@dataclass(frozen=True)
class _ImageBox:
    identifier: str
    class_id: str
    group: str
    segments: tuple[tuple[float, float, float, float], ...]
    score: float | None = None

    def public(self):
        item = {
            "id": self.identifier, "class_id": self.class_id, "group": self.group,
            "segments": [list(segment) for segment in self.segments],
        }
        if self.score is not None:
            item["score"] = self.score
        return item


@dataclass(frozen=True)
class _CameraView:
    observation: CameraObservation
    name: str
    role: str | None
    jpeg: bytes
    projection: bool
    ground_truth: tuple[_ImageBox, ...]
    predictions: tuple[_ImageBox, ...] = ()
    point_rows: bytes | None = None

    def public(self, include_points):
        height, width = self.observation.image.shape[:2]
        return {
            "id": self.observation.sensor_id, "name": self.name, "role": self.role,
            "width": width, "height": height, "projection": self.projection,
            "ground_truth": [box.public() for box in self.ground_truth],
            "predictions": [box.public() for box in self.predictions],
            "point_count": (len(self.point_rows) // _POINT_ROW.itemsize
                            if include_points and self.point_rows is not None else None),
        }


@dataclass(frozen=True, eq=False)
class CameraReview:
    revision: int
    _frame: PointCloudFrame
    _cameras: CameraFrameSet
    _views: tuple[_CameraView, ...]
    _detections: tuple[Detection, ...]
    _include_points: bool

    def public(self):
        return {"revision": self.revision, "views": [view.public(self._include_points) for view in self._views]}

    def _view(self, camera_id: str) -> _CameraView:
        for view in self._views:
            if view.observation.sensor_id == camera_id:
                return view
        raise KeyError(camera_id)

    def image(self, camera_id: str) -> bytes:
        return self._view(camera_id).jpeg

    def points(self, camera_id: str) -> bytes:
        view = self._view(camera_id)
        if not self._include_points or view.point_rows is None:
            raise DataUnavailableError(f"Camera {camera_id!r} has no point overlay for this frame.")
        return view.point_rows


@dataclass(frozen=True, eq=False)
class CameraThumbnails:
    revision: int
    _images: Mapping[str, bytes]

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self._images)

    def image(self, camera_id: str) -> bytes:
        return self._images[camera_id]


def _camera_from_lidar(frame: PointCloudFrame, cameras: CameraFrameSet) -> np.ndarray | None:
    """Map into the camera set reference, before each camera model applies its pose."""

    if frame.reference_frame_id == cameras.reference_frame_id:
        return _IDENTITY
    if frame.world_frame_id == cameras.reference_frame_id:
        return frame.T_world_from_reference
    if cameras.world_frame_id == frame.reference_frame_id:
        return np.linalg.inv(cameras.T_world_from_reference)
    if frame.world_frame_id is not None and frame.world_frame_id == cameras.world_frame_id:
        return np.linalg.solve(cameras.T_world_from_reference, frame.T_world_from_reference)
    return None


def _box_in_camera_reference(box: Box3D, transform: np.ndarray) -> Box3D:
    if transform is _IDENTITY:
        return box
    rotation = transform[:3, :3]
    return replace(box, center=rotation @ box.center + transform[:3, 3], rotation=rotation @ box.rotation)


def _image_boxes(camera, boxes, groups) -> tuple[_ImageBox, ...]:
    output = []
    for box, identifier, score in boxes:
        projected = project_box(box, camera)
        if projected is not None:
            segments = tuple(tuple(segment) for segment in projected.segments.reshape(-1, 4).tolist())
            output.append(_ImageBox(identifier, box.class_id, groups.get(box.class_id, box.class_id), segments, score))
    return tuple(output)


def _jpeg(camera: CameraObservation, *, thumbnail: bool = False) -> bytes:
    image = Image.fromarray(camera.image)
    quality = 90
    if thumbnail:
        image.thumbnail((_THUMBNAIL_SIZE, _THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
        quality = 80
    with BytesIO() as buffer:
        image.save(buffer, "JPEG", quality=quality)
        return buffer.getvalue()


def _point_rows(camera: CameraObservation, points: np.ndarray) -> bytes:
    if not len(points):
        return b""
    result = camera.projection_model.project_reference(points)
    if not isinstance(result, ProjectionResult) or len(result.uv) != len(points):
        raise DatasetValidationError(f"Projection for camera {camera.sensor_id!r} did not preserve LiDAR point rows.")
    u, v = result.uv.T
    height, width = camera.image.shape[:2]
    visible = (result.valid_mask & (result.depth >= DEFAULT_NEAR_PLANE)
               & (0 <= u) & (u <= width - 1) & (0 <= v) & (v <= height - 1))
    indices = np.flatnonzero(visible)
    indices = indices[np.argsort(-result.depth[indices], kind="stable")]
    packed = np.empty(len(indices), dtype=_POINT_ROW)
    packed["pixel"] = result.uv[indices]
    packed["index"] = indices
    return packed.tobytes()


def _load_camera_frame(loader: DatasetLoader, frame: PointCloudFrame, progress) -> CameraFrameSet:
    if Capability.CAMERAS not in loader.metadata.capabilities:
        raise UnsupportedCapabilityError("This dataset does not provide camera observations.")
    if progress is not None:
        progress("Reading camera images")
    cameras = loader.load_cameras(frame.info.frame_id)
    if (cameras.info.frame_id, cameras.info.scene_id) != (frame.info.frame_id, frame.info.scene_id):
        raise DatasetValidationError("The camera observation does not match the selected scene and frame.")
    return cameras


def build_camera_thumbnails(
    loader: DatasetLoader, frame: PointCloudFrame, *, revision: int, review: CameraReview | None = None,
    progress: Callable[[str], None] | None = None,
) -> CameraThumbnails:
    """Encode small previews of the selected frame without projecting overlays."""

    if review is not None and review.revision == revision and review._frame is frame:
        cameras = review._cameras
    else:
        cameras = _load_camera_frame(loader, frame, progress)
    if progress is not None:
        progress("Preparing camera thumbnails")
    images = {camera.sensor_id: _jpeg(camera, thumbnail=True) for camera in cameras.cameras}
    return CameraThumbnails(revision, MappingProxyType(images))


def _start_review(loader: DatasetLoader, frame: PointCloudFrame, groups, progress):
    cameras = _load_camera_frame(loader, frame, progress)
    sensors = {sensor.sensor_id: sensor for sensor in loader.metadata.sensors if sensor.modality == Modality.CAMERAS}
    views = []
    for camera in cameras.cameras:
        sensor = sensors.get(camera.sensor_id)
        name = camera.sensor_id if sensor is None else sensor.display_name
        role = None if sensor is None else sensor.view_role
        if progress is not None:
            progress(f"Preparing camera view: {name}")
        ground_truth = ()
        supported = camera.projection_model is not None
        if supported:
            annotations = camera.annotations if camera.annotations is not None else cameras.annotations
            try:
                ground_truth = _image_boxes(camera, ((box, box.annotation_id, None) for box in annotations.boxes), groups)
            except (UnsupportedCapabilityError, NotImplementedError):
                supported = False
        views.append(_CameraView(camera, name, role, _jpeg(camera), supported, ground_truth))
    return cameras, tuple(views)


def build_camera_review(
    loader: DatasetLoader, frame: PointCloudFrame, filtered_detections: Sequence[Detection] | None, *,
    revision: int, include_points: bool, previous: CameraReview | None = None,
    progress: Callable[[str], None] | None = None,
) -> CameraReview:
    """Stage images and overlays while keeping the cloud's native row indices."""
    detections = tuple(filtered_detections or ())
    groups = {item.class_id: item.display_group or item.display_name for item in loader.metadata.classes}
    reuse = previous is not None and previous.revision == revision and previous._frame is frame
    if reuse:
        cameras, views = previous._cameras, previous._views
        update_predictions = len(detections) != len(previous._detections) or any(
            current is not prior for current, prior in zip(detections, previous._detections)
        )
    else:
        cameras, views = _start_review(loader, frame, groups, progress)
        update_predictions = bool(detections)
    transform = _camera_from_lidar(frame, cameras)
    prediction_boxes = ()
    if update_predictions and transform is not None:
        prediction_boxes = tuple(
            (_box_in_camera_reference(item.box, transform), item.identifier, item.score) for item in detections
        )
    coordinates = None
    if include_points and transform is not None and any(view.projection and view.point_rows is None for view in views):
        if len(frame.points) > np.iinfo(np.uint32).max + 1:
            raise DatasetValidationError("The LiDAR cloud exceeds the camera overlay row index range.")
        coordinates = (frame.points if transform is _IDENTITY
                       else frame.points @ transform[:3, :3].T + transform[:3, 3])

    output = []
    for view in views:
        if view.projection and update_predictions:
            try:
                predictions = _image_boxes(view.observation, prediction_boxes, groups)
            except (UnsupportedCapabilityError, NotImplementedError):
                predictions = ()
            view = replace(view, predictions=predictions)
        if view.projection and coordinates is not None and view.point_rows is None:
            if progress is not None:
                progress(f"Projecting LiDAR points: {view.name}")
            try:
                view = replace(view, point_rows=_point_rows(view.observation, coordinates))
            except (UnsupportedCapabilityError, NotImplementedError):
                pass
        output.append(view)
    return CameraReview(revision, frame, cameras, tuple(output), detections, include_points)
