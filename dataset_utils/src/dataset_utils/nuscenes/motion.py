"""Pose interpolation, nominal nuScenes acquisition times, and LiDAR decompensation.

The timing convention follows NVIDIA NCore v19.8.0, commit ``dde366e``:
https://github.com/NVIDIA/ncore/blob/dde366e7e8e9e7dbb9b1278488936005a7dee4e2/tools/data_converter/structured_lidar_model.py

nuScenes XYZ already refers to the LiDAR frame at the sweep's end. Its ordered
32-beam firing columns provide approximate acquisition times; compensated XYZ
azimuths alone cannot provide a reliable acquisition clock. These routines use a
nominal clockwise HDL-32E model, never an empirical fit or model optimization.
"""

from __future__ import annotations

import numpy as np


HDL32E_BEAMS = 32
HDL32E_COLUMNS = 1085
MAX_POSE_EXTRAPOLATION_US = 50_000


def _timestamps(value: object, name: str) -> np.ndarray:
    result = np.asarray(value)
    if not result.size:
        return result.astype(np.int64)
    if result.dtype.kind not in "iu" or (
        result.dtype.kind == "u" and np.any(result > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} must contain integer microsecond timestamps")
    return result.astype(np.int64, copy=False)


def _points(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (3,) or not np.isfinite(result).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    return result


def _rigid_matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (4, 4)
        or not np.isfinite(result).all()
        or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(result[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError(f"{name} must be a rigid 4 x 4 transform")
    return result


def _quaternion_matrices(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternions.T
    result = np.empty((len(quaternions), 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1 - 2 * (y * y + z * z)
    result[:, 0, 1] = 2 * (x * y - z * w)
    result[:, 0, 2] = 2 * (x * z + y * w)
    result[:, 1, 0] = 2 * (x * y + z * w)
    result[:, 1, 1] = 1 - 2 * (x * x + z * z)
    result[:, 1, 2] = 2 * (y * z - x * w)
    result[:, 2, 0] = 2 * (x * z - y * w)
    result[:, 2, 1] = 2 * (y * z + x * w)
    result[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return result


class PoseTrajectory:
    """Ego-to-world poses with linear translation and shortest-arc quaternion SLERP.

    Microseconds remain integers until *differences* are converted to floating
    point, preserving precision for Unix timestamps. Queries outside the pose
    timeline extrapolate its first/last velocity for at most 50 ms (one nominal
    LiDAR revolution). Longer extrapolation raises ``ValueError``. A one-pose
    trajectory is stationary within the same 50 ms limit.
    """

    def __init__(
        self,
        timestamps_us: object,
        translations: object,
        quaternions_wxyz: object,
    ) -> None:
        timestamps = _timestamps(timestamps_us, "timestamps_us")
        translation = np.asarray(translations, dtype=np.float64)
        quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
        if (
            timestamps.ndim != 1
            or not len(timestamps)
            or np.any(timestamps[1:] <= timestamps[:-1])
        ):
            raise ValueError("timestamps_us must be a nonempty, strictly increasing vector")
        if translation.shape != (len(timestamps), 3) or not np.isfinite(translation).all():
            raise ValueError("translations must be finite with shape (N, 3)")
        if quaternions.shape != (len(timestamps), 4) or not np.isfinite(quaternions).all():
            raise ValueError("quaternions_wxyz must be finite with shape (N, 4)")
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError("quaternions_wxyz must have nonzero norm")
        self.timestamps_us = timestamps.copy()
        self.translations = translation.copy()
        self.quaternions_wxyz = quaternions / norms[:, None]
        for array in (self.timestamps_us, self.translations, self.quaternions_wxyz):
            array.setflags(write=False)

    def at(self, timestamps_us: object) -> np.ndarray:
        """Return ego-to-world transforms with shape ``timestamps.shape + (4, 4)``."""
        query = _timestamps(timestamps_us, "timestamps_us")
        flat = query.reshape(-1)
        result = np.broadcast_to(np.eye(4), (flat.size, 4, 4)).copy()
        if not flat.size:
            return result.reshape(query.shape + (4, 4))
        if (
            int(flat.min()) < int(self.timestamps_us[0]) - MAX_POSE_EXTRAPOLATION_US
            or int(flat.max()) > int(self.timestamps_us[-1]) + MAX_POSE_EXTRAPOLATION_US
        ):
            raise ValueError("pose query requires more than 50000 us of extrapolation")
        if len(self.timestamps_us) == 1:
            translations = np.broadcast_to(self.translations[0], (flat.size, 3))
            rotations = np.broadcast_to(self.quaternions_wxyz[0], (flat.size, 4))
        else:
            lower = np.clip(
                np.searchsorted(self.timestamps_us, flat, side="right") - 1,
                0,
                len(self.timestamps_us) - 2,
            )
            durations = self.timestamps_us[lower + 1] - self.timestamps_us[lower]
            fraction = (flat - self.timestamps_us[lower]).astype(np.float64) / durations
            translations = (
                self.translations[lower]
                + fraction[:, None] * (self.translations[lower + 1] - self.translations[lower])
            )
            q0 = self.quaternions_wxyz[lower]
            q1 = self.quaternions_wxyz[lower + 1].copy()
            dot = np.einsum("ij,ij->i", q0, q1)
            q1[dot < 0] *= -1
            angle = np.arccos(np.clip(np.abs(dot), 0, 1))
            # sinc avoids a divide-by-zero special case when the rotations coincide.
            denominator = np.sinc(angle / np.pi)
            weight0 = (1 - fraction) * np.sinc((1 - fraction) * angle / np.pi) / denominator
            weight1 = fraction * np.sinc(fraction * angle / np.pi) / denominator
            rotations = weight0[:, None] * q0 + weight1[:, None] * q1
            rotations /= np.linalg.norm(rotations, axis=1)[:, None]
        result[:, :3, :3] = _quaternion_matrices(rotations)
        result[:, :3, 3] = translations
        return result.reshape(query.shape + (4, 4))


def compensated_points_to_world(
    points: object,
    reference_timestamp_us: int,
    trajectory: PoseTrajectory,
    sensor_to_ego: object,
    *,
    reference_sensor_to_world: object | None = None,
) -> np.ndarray:
    """Map a compensated cloud through its single reference-time sensor pose."""
    points_array = _points(points)
    sensor_pose = _rigid_matrix(sensor_to_ego, "sensor_to_ego")
    reference = (
        trajectory.at(reference_timestamp_us) @ sensor_pose
        if reference_sensor_to_world is None
        else _rigid_matrix(reference_sensor_to_world, "reference_sensor_to_world")
    )
    return points_array @ reference[:3, :3].T + reference[:3, 3]


def decompensate_points(
    points: object,
    point_timestamps_us: object,
    reference_timestamp_us: int,
    trajectory: PoseTrajectory,
    sensor_to_ego: object,
    *,
    reference_sensor_to_world: object | None = None,
) -> np.ndarray:
    """Undo ego compensation into each point's own acquisition-time LiDAR frame.

    The transform is ``inverse(T_world_sensor(t_i)) @ T_world_sensor(t_ref)``.
    Restoring the raw points through the same acquisition poses returns the
    original world points; per-point timing alone does not correct object motion.
    """
    points_array = _points(points)
    timestamps = _timestamps(point_timestamps_us, "point_timestamps_us")
    if timestamps.shape != (len(points_array),):
        raise ValueError("point_timestamps_us must have shape (N,)")
    sensor_pose = _rigid_matrix(sensor_to_ego, "sensor_to_ego")
    if not len(points_array):
        return points_array.copy()
    world = compensated_points_to_world(
        points_array,
        reference_timestamp_us,
        trajectory,
        sensor_pose,
        reference_sensor_to_world=reference_sensor_to_world,
    )
    unique_timestamps, inverse = np.unique(timestamps, return_inverse=True)
    poses = trajectory.at(unique_timestamps) @ sensor_pose
    return np.einsum(
        "nji,nj->ni", poses[inverse, :3, :3], world - poses[inverse, :3, 3]
    )


def _column_azimuths(points: np.ndarray, columns: np.ndarray, n_columns: int, min_range: float) -> np.ndarray:
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    valid = np.linalg.norm(points, axis=1) > min_range
    medians = np.full(n_columns, np.nan)
    grouped = np.full(n_columns * HDL32E_BEAMS, np.nan)
    grouped[:len(points)][valid] = azimuth[valid]
    grouped = grouped.reshape(n_columns, HDL32E_BEAMS)
    enough = np.isfinite(grouped).sum(axis=1) >= 3
    # All beams in a column differ by a fraction of a degree. Unwrap them about
    # one valid beam before taking a median so a column crossing +/-pi remains
    # at the back of the sensor instead of erroneously jumping to zero.
    selected = grouped[enough]
    anchors = selected[np.arange(len(selected)), np.argmax(np.isfinite(selected), axis=1)]
    relative = selected - anchors[:, None]
    medians[enough] = anchors + np.nanmedian(np.arctan2(np.sin(relative), np.cos(relative)), axis=1)
    return medians


def _interpolate_azimuths(azimuths: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(azimuths)
    if not valid.any():
        return None
    return np.interp(np.arange(len(azimuths)), np.flatnonzero(valid), np.unwrap(azimuths[valid]))


def _nominal_model(phase: float, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform clockwise angles and analytical HDL-32E firing geometry.

    Row offsets express geometry, not additional timestamp offsets, matching
    NCore. Model rows run from highest to lowest elevation: row = 31 - ring.
    """
    angles = phase - 2 * np.pi * np.arange(HDL32E_COLUMNS * resolution) / (HDL32E_COLUMNS * resolution)
    ring_time_us = (np.arange(HDL32E_BEAMS) % 16) * 2 * 1.152
    row_offsets = (-2 * np.pi / 50_000 * ring_time_us)[::-1]
    row_offsets -= row_offsets.mean()
    return angles, row_offsets


def _column_times(columns: np.ndarray, count: int, start: int, end: int) -> np.ndarray:
    # Round the small relative offset before adding the integer Unix epoch.
    # Truncation matches NCore without converting the absolute timestamp to float.
    offset = np.floor(columns.astype(np.float64) / count * (end - start)).astype(np.int64)
    return np.int64(start) + offset


def estimate_nominal_phase(points_compensated: object) -> float:
    """Initialize a scene's nominal model phase as NCore does.

    Use a complete, unfiltered, ordered sweep where possible. Far points in the
    first ten firing columns reduce ego-translation bias in the compensated
    azimuths. This initial phase is approximate; subsequent column alignment
    resolves its integer-column offset. Returns zero if no such points exist.
    """
    points = _points(points_compensated)
    first = points[:10 * HDL32E_BEAMS]
    far = np.linalg.norm(first, axis=1) > 20
    return float(np.median(np.unwrap(np.arctan2(first[far, 1], first[far, 0])))) if far.any() else 0.0


def nominal_point_timestamps(
    points_compensated: object,
    ring_indices: object,
    sweep_start_us: int,
    sweep_end_us: int,
    trajectory: PoseTrajectory,
    sensor_to_ego: object,
    resolution: int = 4,
    *,
    azimuth_phase_rad: float | None = None,
    reference_sensor_to_world: object | None = None,
) -> np.ndarray:
    """Approximate acquisition times using NCore's nominal two-pass alignment.

    Points must retain nuScenes file order: one occurrence of each ring per
    complete 32-point column. Times are shared by the beams in a column and use
    ``column / model_columns`` (the next revolution starts at sweep end).

    ``azimuth_phase_rad`` can retain one nominal model phase across a scene.
    Otherwise it is initialized as NCore does from far-range points in the
    first ten columns. Under 20 columns, or a cloud without usable directions,
    falls back to ordered columns on the fixed 1085-column nominal time grid.
    This fallback supports partial scans without inventing an azimuth clock.

    NCore drops columns outside model overlap. The viewer retains their order
    and clamps their times to the nearest model boundary instead. Beam offsets
    affect nominal geometry only, never time. No empirical model is derived.
    """
    points = _points(points_compensated)
    rings = np.asarray(ring_indices)
    if (
        rings.shape != (len(points),)
        or (rings.size and rings.dtype.kind not in "iu")
        or np.any((rings < 0) | (rings >= HDL32E_BEAMS))
    ):
        raise ValueError("ring_indices must contain integer ring IDs in [0, 31], with shape (N,)")
    start_array = _timestamps(sweep_start_us, "sweep_start_us")
    end_array = _timestamps(sweep_end_us, "sweep_end_us")
    if start_array.ndim or end_array.ndim:
        raise ValueError("sweep start and end must be scalar timestamps")
    start, end = int(start_array), int(end_array)
    if end <= start:
        raise ValueError("sweep_end_us must be later than sweep_start_us")
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)) or resolution not in (1, 2, 4):
        raise ValueError("resolution must be one of 1, 2, or 4")
    _rigid_matrix(sensor_to_ego, "sensor_to_ego")
    if azimuth_phase_rad is not None and not np.isfinite(azimuth_phase_rad):
        raise ValueError("azimuth_phase_rad must be finite")
    if not len(points):
        return np.empty(0, dtype=np.int64)
    full_count = len(points) // HDL32E_BEAMS
    complete = rings[:full_count * HDL32E_BEAMS].reshape(full_count, HDL32E_BEAMS)
    if complete.size and not np.all(np.sort(complete, axis=1) == np.arange(HDL32E_BEAMS)):
        raise ValueError("points must preserve complete 32-beam nuScenes firing columns")
    remainder = rings[full_count * HDL32E_BEAMS:]
    if len(np.unique(remainder)) != len(remainder):
        raise ValueError("a partial firing column must contain distinct ring IDs")
    columns = np.arange(len(points)) // HDL32E_BEAMS
    n_physical = int(columns[-1]) + 1
    if n_physical > HDL32E_COLUMNS + 40:
        raise ValueError("point cloud contains more than one nominal HDL-32E revolution")
    ordered = _column_times(np.minimum(columns, HDL32E_COLUMNS - 1), HDL32E_COLUMNS, start, end)
    if n_physical < 20:
        return ordered
    if azimuth_phase_rad is None:
        phase = estimate_nominal_phase(points)
    else:
        phase = float(azimuth_phase_rad)
    model, _row_geometry = _nominal_model(phase, resolution)
    native_model = model[::resolution]
    current_points = points
    overlap = (0, n_physical)
    times = ordered
    for iteration in range(2):
        measured = _column_azimuths(current_points, columns, n_physical, 20 if iteration == 0 else 0.5)
        if iteration == 0 and np.isfinite(measured).sum() < 0.3 * n_physical:
            measured = _column_azimuths(current_points, columns, n_physical, 5)
        if iteration:
            measured[:overlap[0]] = np.nan
            measured[overlap[1]:] = np.nan
        measured = _interpolate_azimuths(measured)
        if measured is None:
            return ordered
        best_error = np.inf
        best_shift = 0
        for shift in range(-20, 21):
            first, stop = max(0, -shift), min(n_physical, HDL32E_COLUMNS - shift)
            if stop <= first:
                continue
            errors = np.abs(np.arctan2(
                np.sin(measured[first:stop] - native_model[first + shift:stop + shift]),
                np.cos(measured[first:stop] - native_model[first + shift:stop + shift]),
            ))
            mean_error = float(errors.mean())
            if mean_error < best_error:
                best_error, best_shift = mean_error, shift
                overlap = (first, stop)
        physical_model = np.clip(
            (np.arange(n_physical) + best_shift) * resolution,
            0,
            len(model) - 1,
        )
        if iteration == 1 and resolution > 1:
            for column in range(overlap[0], overlap[1]):
                coarse = int(physical_model[column])
                lower, upper = max(0, coarse - resolution), min(len(model), coarse + resolution + 1)
                candidates = model[lower:upper]
                errors = np.abs(np.arctan2(np.sin(candidates - measured[column]), np.cos(candidates - measured[column])))
                physical_model[column] = lower + int(errors.argmin())
        times = _column_times(physical_model[columns], len(model), start, end)
        current_points = decompensate_points(
            points,
            times,
            end,
            trajectory,
            sensor_to_ego,
            reference_sensor_to_world=reference_sensor_to_world,
        )
    return times
