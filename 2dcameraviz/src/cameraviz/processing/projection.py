"""Project reference-frame cuboids through pinhole or native camera models.

Camera optical coordinates have x right, y down and z forward. Clipping takes
place before perspective division at the near plane and afterwards in image
coordinates. The rectangle includes visible faces even when the object's
original edges all fall outside the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset_utils import Box3D, CameraFrameSet, CameraObservation, PinholeProjection


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

    class_id: str
    annotation_id: str
    segments: np.ndarray
    rectangle: tuple[float, float, float, float]


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


def _model_project(model, points):
    result = model.project_reference(np.asarray(points, dtype=np.float64))
    pixels = np.where(result.valid_mask[:, None], result.uv, np.nan)
    return pixels, result.depth


def _nonlinear_near_intersections(model, first, second, near):
    count = len(first)
    pixels, depth = _model_project(model, np.concatenate((first, second)))
    inside = np.isfinite(depth) & (depth >= near)
    front = np.where(inside[:count, None], first, second).copy()
    back = np.where(inside[:count, None], second, first).copy()
    front_pixel = np.where(inside[:count, None], pixels[:count], pixels[count:]).copy()
    usable = (np.isfinite(depth[:count]) & np.isfinite(depth[count:])
              & (inside[:count] != inside[count:]) & np.isfinite(front_pixel).all(axis=1))
    for _ in range(28):
        middle = (front + back) / 2
        uv, z = _model_project(model, middle)
        valid = np.isfinite(z) & (z >= near)
        usable &= np.isfinite(z) & (~valid | np.isfinite(uv).all(axis=1))
        valid &= usable
        front[valid], front_pixel[valid] = middle[valid], uv[valid]
        back[usable & ~valid] = middle[usable & ~valid]
    front_pixel[~usable] = np.nan
    return front_pixel


def _project_nonlinear_box(box, camera, reference, near, xmax, ymax):
    """Sample curved edges and faces before viewport clipping.

    Sixteen pieces per cuboid edge retain distortion/shutter curvature. Face
    boundaries also supply rectangle visibility when every original edge lies
    outside the image. Near intersections are refined in reference coordinates
    against the camera model's returned forward depth.
    """

    model = camera.projection_model
    fractions = np.linspace(0., 1., 17)
    samples_by_edge = np.asarray([
        reference[first] + fractions[:, None] * (reference[second] - reference[first])
        for first, second in _EDGES
    ])
    all_pixels, all_depth = _model_project(model, samples_by_edge.reshape(-1, 3))
    pixels_by_edge = all_pixels.reshape(12, 17, 2)
    depth_by_edge = all_depth.reshape(12, 17)
    if not np.any(np.isfinite(all_pixels).all(axis=1) & np.isfinite(all_depth) & (all_depth >= near)):
        return None
    edge_lookup = {edge: index for index, edge in enumerate(_EDGES)}
    valid_by_edge = (np.isfinite(pixels_by_edge).all(axis=2)
                     & np.isfinite(depth_by_edge) & (depth_by_edge >= near))
    depth_inside = depth_by_edge >= near
    crossings = np.argwhere(
        np.isfinite(depth_by_edge[:, :-1]) & np.isfinite(depth_by_edge[:, 1:])
        & (depth_inside[:, :-1] != depth_inside[:, 1:])
        & (valid_by_edge[:, :-1] | valid_by_edge[:, 1:]))
    near_pixels, near_by_points = {}, {}
    if len(crossings):
        starts = np.asarray([samples_by_edge[edge, index] for edge, index in crossings])
        ends = np.asarray([samples_by_edge[edge, index + 1] for edge, index in crossings])
        intersections = _nonlinear_near_intersections(model, starts, ends, near)
        for (edge, index), first, second, pixel in zip(crossings, starts, ends, intersections):
            if not np.isfinite(pixel).all():
                continue
            near_pixels[(edge, index)] = pixel
            near_by_points[tuple(sorted((tuple(first), tuple(second))))] = pixel
    segments = []
    for edge_index in range(len(_EDGES)):
        pixels = pixels_by_edge[edge_index]
        valid = valid_by_edge[edge_index]
        for index in range(16):
            if not valid[index] and not valid[index + 1]:
                continue
            ends = pixels[index:index + 2].copy()
            if valid[index] != valid[index + 1]:
                intersection = near_pixels.get((edge_index, index))
                if intersection is None:
                    continue
                ends[0 if not valid[index] else 1] = intersection
            clipped = _clip_segment_image(ends, xmax, ymax)
            if clipped is not None:
                segments.append(clipped)
    visible = []
    for face in _FACES:
        face_samples, face_pixels, face_depth = [], [], []
        for first, second in zip(face, (*face[1:], face[0])):
            direct = (first, second) in edge_lookup
            index = edge_lookup[(first, second) if direct else (second, first)]
            order = slice(None) if direct else slice(None, None, -1)
            face_samples.append(samples_by_edge[index][order][:-1])
            face_pixels.append(pixels_by_edge[index][order][:-1])
            face_depth.append(depth_by_edge[index][order][:-1])
        samples = np.concatenate(face_samples)
        pixels, depth = np.concatenate(face_pixels), np.concatenate(face_depth)
        # Undefined projection intervals are not a geometric clipping plane.
        # A face boundary crossing one cannot safely define a closed polygon;
        # retain its independently valid wireframe segments instead.
        undefined = ~np.isfinite(depth) | ((depth >= near) & ~np.isfinite(pixels).all(axis=1))
        if np.any(undefined):
            continue
        valid = np.isfinite(pixels).all(axis=1) & np.isfinite(depth) & (depth >= near)
        polygon = []
        complete = True
        previous = len(samples) - 1
        for index in range(len(samples)):
            if valid[index] != valid[previous]:
                key = tuple(sorted((tuple(samples[previous]), tuple(samples[index]))))
                intersection = near_by_points.get(key)
                if intersection is None:
                    complete = False
                    break
                polygon.append(intersection)
            if valid[index]:
                polygon.append(pixels[index])
            previous = index
        if not complete:
            continue
        polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        for axis, boundary, keep_greater in ((0, 0., True), (0, xmax, False), (1, 0., True), (1, ymax, False)):
            polygon = _clip_polygon(polygon, axis, boundary, keep_greater=keep_greater)
        if len(polygon):
            visible.append(polygon)
    if not visible and not segments:
        return None
    pixels = np.concatenate(visible) if visible else np.asarray(segments).reshape(-1, 2)
    minimum, maximum = pixels.min(axis=0), pixels.max(axis=0)
    return ProjectedBox(
        box.class_id, box.annotation_id, np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2),
        (minimum[0], minimum[1], maximum[0], maximum[1]),
    )


def project_box(box: Box3D, camera: CameraObservation, *, near_plane: float = 0.1) -> ProjectedBox | None:
    """Project one box in the camera model's declared reference coordinates.

    Explicit pinhole models use exact near-plane and image clipping. Other
    models retain sampled curved edges and native distortion/shutter behavior.
    Images without a projection model remain viewable without an overlay.
    """
    try:
        near = float(near_plane)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("near_plane must be positive and finite.") from exc
    if isinstance(near_plane, bool) or not np.isfinite(near) or near <= 0:
        raise ValueError("near_plane must be positive and finite.")
    model = camera.projection_model
    if model is None:
        return None
    ymax, xmax = float(camera.image.shape[0] - 1), float(camera.image.shape[1] - 1)
    reference = (_CORNER_SIGNS * (box.size / 2)) @ box.rotation.T + box.center
    if not isinstance(model, PinholeProjection):
        return _project_nonlinear_box(box, camera, reference, near, xmax, ymax)
    pose = model.T_reference_from_camera
    intrinsics = model.intrinsics
    optical = (reference - pose[:3, 3]) @ pose[:3, :3]
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
        class_id=box.class_id,
        annotation_id=box.annotation_id,
        segments=np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2),
        rectangle=rectangle,
    )


def project_boxes(frame: CameraFrameSet, *, near_plane: float = 0.1) -> dict[str, tuple[ProjectedBox, ...]]:
    """Project default or camera-specific annotations without changing records."""
    result = {}
    for camera in frame.cameras:
        annotations = frame.annotations if camera.annotations is None else camera.annotations
        projected = (project_box(box, camera, near_plane=near_plane) for box in annotations.boxes)
        result[camera.sensor_id] = tuple(box for box in projected if box is not None)
    return result
