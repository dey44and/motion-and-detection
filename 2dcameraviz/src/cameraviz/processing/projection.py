"""Project world-space cuboids into undistorted pinhole camera images.

Camera optical coordinates have x right, y down and z forward. Clipping takes
place before perspective division at the near plane and afterwards in image
coordinates. The rectangle includes visible faces even when the object's
original edges all fall outside the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cameraviz.dataloader.models import BoundingBox3D, CameraFrame, FrameData


_CORNER_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)
_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_FACES = (
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
)


@dataclass(frozen=True)
class ProjectedBox:
    """Visible original cuboid edges and the bounds of its visible projection.

    ``segments`` has shape ``(N, 2, 2)``: N image-space line segments, each
    containing two (x, y) endpoints. It can be empty when a face covers the
    image but all original edges are outside it. ``rectangle`` is
    ``(xmin, ymin, xmax, ymax)`` in the same native image pixel coordinates.
    """

    class_name: str
    token: str
    segments: np.ndarray
    rectangle: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        segments = np.array(self.segments, dtype=np.float64, copy=True)
        if segments.ndim != 3 or segments.shape[1:] != (2, 2) or not np.isfinite(segments).all():
            raise ValueError("segments must be finite with shape (N, 2, 2).")
        rectangle = np.asarray(self.rectangle, dtype=np.float64)
        if rectangle.shape != (4,) or not np.isfinite(rectangle).all():
            raise ValueError("rectangle must contain four finite coordinates.")
        if rectangle[0] > rectangle[2] or rectangle[1] > rectangle[3]:
            raise ValueError("rectangle coordinates must be ordered xmin, ymin, xmax, ymax.")
        segments.setflags(write=False)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "rectangle", tuple(float(value) for value in rectangle))


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite with shape {shape}.") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return array


def _rotation(value: object, name: str) -> np.ndarray:
    rotation = _array(value, (3, 3), name)
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6):
        raise ValueError(f"{name} must be an orthonormal rotation.")
    if not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-6):
        raise ValueError(f"{name} must be a proper rotation.")
    return rotation


def _clip_polygon(polygon: np.ndarray, axis: int, boundary: float, *, keep_greater: bool) -> np.ndarray:
    """Sutherland-Hodgman clipping against one axis-aligned half-plane."""

    if not len(polygon):
        return polygon
    result = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
    for current in polygon:
        current_inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
        if current_inside != previous_inside:
            fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
            intersection = previous + fraction * (current - previous)
            intersection[axis] = boundary
            result.append(intersection)
        if current_inside:
            result.append(current)
        previous, previous_inside = current, current_inside
    return np.asarray(result, dtype=np.float64).reshape(-1, polygon.shape[1])


def _project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = points @ intrinsics.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2, None]
    if not np.isfinite(pixels).all():
        raise ValueError("Camera projection produced nonfinite image coordinates.")
    return pixels


def _clip_segment_near(start: np.ndarray, end: np.ndarray, near: float) -> np.ndarray | None:
    start_inside, end_inside = start[2] >= near, end[2] >= near
    if not start_inside and not end_inside:
        return None
    if start_inside and end_inside:
        return np.array([start, end])
    intersection = start + (near - start[2]) / (end[2] - start[2]) * (end - start)
    intersection[2] = near
    return np.array([start if start_inside else intersection, end if end_inside else intersection])


def _clip_segment_image(segment: np.ndarray, xmax: float, ymax: float) -> np.ndarray | None:
    """Clip a line parametrically, retaining only its part inside the image."""

    start, end = segment
    delta = end - start
    lower, upper = 0.0, 1.0
    for axis, maximum in ((0, xmax), (1, ymax)):
        if delta[axis] == 0:
            if start[axis] < 0 or start[axis] > maximum:
                return None
            continue
        limits = sorted(((0 - start[axis]) / delta[axis], (maximum - start[axis]) / delta[axis]))
        lower, upper = max(lower, limits[0]), min(upper, limits[1])
        if lower > upper:
            return None
    clipped = np.array([start + lower * delta, start + upper * delta])
    # Intersection arithmetic can drift a few ulps past the image boundary.
    return np.clip(clipped, [0.0, 0.0], [xmax, ymax])


def project_box(box: BoundingBox3D, camera: CameraFrame, *, near_plane: float = 0.1) -> ProjectedBox | None:
    """Return a clipped cuboid projection, or None if it misses the camera view.

    Box dimensions are (length, width, height); its rotation maps box-local
    coordinates into world coordinates. ``camera.sensor_to_world`` maps
    optical coordinates into world coordinates and is inverted as a full rigid
    transform, including pitch and roll. No assumption about vehicle yaw is
    made. Image bounds use pixel centers from zero through width/height minus
    one. The model assumes an undistorted pinhole image.
    """

    try:
        near = float(near_plane)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("near_plane must be positive and finite.") from exc
    if isinstance(near_plane, bool) or not np.isfinite(near) or near <= 0:
        raise ValueError("near_plane must be positive and finite.")
    center = _array(box.center, (3,), "Box center")
    size = _array(box.size, (3,), "Box size")
    if np.any(size <= 0):
        raise ValueError("Box size must be positive.")
    box_rotation = _rotation(box.rotation, "Box rotation")
    pose = _array(camera.sensor_to_world, (4, 4), "Camera sensor_to_world")
    if not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-9):
        raise ValueError("Camera sensor_to_world must have last row [0, 0, 0, 1].")
    camera_rotation = _rotation(pose[:3, :3], "Camera sensor_to_world rotation")
    intrinsics = _array(camera.intrinsics, (3, 3), "Camera intrinsics")
    if (intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1], rtol=0, atol=1e-9)):
        raise ValueError("Camera intrinsics must be a pinhole matrix with positive focal lengths.")
    image = np.asarray(camera.image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 or image.dtype != np.uint8:
        raise ValueError("Camera image must be uint8 RGB with shape (H, W, 3).")
    ymax, xmax = float(image.shape[0] - 1), float(image.shape[1] - 1)

    world = (_CORNER_SIGNS * (size / 2)) @ box_rotation.T + center
    optical = (world - pose[:3, 3]) @ camera_rotation
    if not np.isfinite(optical).all():
        raise ValueError("Box transform produced nonfinite camera coordinates.")
    if np.all(optical[:, 2] < near):
        return None

    visible_faces = []
    for face in _FACES:
        polygon = _clip_polygon(optical[list(face)], 2, near, keep_greater=True)
        if not len(polygon):
            continue
        polygon = _project(polygon, intrinsics)
        for axis, boundary, keep_greater in ((0, 0.0, True), (0, xmax, False), (1, 0.0, True), (1, ymax, False)):
            polygon = _clip_polygon(polygon, axis, boundary, keep_greater=keep_greater)
        if len(polygon):
            visible_faces.append(polygon)
    if not visible_faces:
        return None
    pixels = np.concatenate(visible_faces)
    minimum, maximum = pixels.min(axis=0), pixels.max(axis=0)
    rectangle = (minimum[0], minimum[1], maximum[0], maximum[1])

    segments = []
    for first, second in _EDGES:
        segment = _clip_segment_near(optical[first], optical[second], near)
        if segment is None:
            continue
        segment = _clip_segment_image(_project(segment, intrinsics), xmax, ymax)
        if segment is not None:
            segments.append(segment)
    return ProjectedBox(
        class_name=box.class_name,
        token=box.token,
        segments=np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2),
        rectangle=rectangle,
    )


def project_boxes(frame: FrameData, *, near_plane: float = 0.1) -> dict[str, tuple[ProjectedBox, ...]]:
    """Project a frame's boxes independently into every available camera."""

    result = {}
    for camera in frame.cameras:
        channel = camera.info.channel
        if channel in result:
            raise ValueError(f"Duplicate camera channel {channel!r}.")
        boxes = []
        for box in frame.boxes:
            projected = project_box(box, camera, near_plane=near_plane)
            if projected is not None:
                boxes.append(projected)
        result[channel] = tuple(boxes)
    return result
