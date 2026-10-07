"""Check independent pose geometry and the nominal HDL-32E acquisition clock."""

from __future__ import annotations

import numpy as np
import pytest

from threedviz.processing.motion import (
    HDL32E_COLUMNS,
    PoseTrajectory,
    compensated_points_to_world,
    decompensate_points,
    estimate_nominal_phase,
    nominal_point_timestamps,
)


EPOCH_US = 1_531_234_567_890_123


def quaternion_yaw(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])


def pose_yaw(angle: float, translation: object = (0, 0, 0)) -> np.ndarray:
    result = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = translation
    return result


def stationary_trajectory(start: int, end: int) -> PoseTrajectory:
    return PoseTrajectory([start, end], np.zeros((2, 3)), [[1, 0, 0, 0]] * 2)


def nominal_scan(phase: float = 0, columns: int = HDL32E_COLUMNS) -> tuple[np.ndarray, np.ndarray]:
    # Independent analytical beam directions: two 16-beam firing banks, CW spin.
    rings = np.tile(np.arange(32), columns)
    col = np.repeat(np.arange(columns), 32)
    delays = (np.arange(32) % 16) * 2 * 1.152
    delays -= delays.mean()
    azimuth = phase - 2 * np.pi * col / HDL32E_COLUMNS - 2 * np.pi / 50_000 * delays[rings]
    elevation = np.deg2rad(-30.67 + rings * 41.34 / 31)
    points = 40 * np.column_stack([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    return points, rings


def test_pose_translation_and_yaw_slerp_preserve_integer_microseconds() -> None:
    trajectory = PoseTrajectory(
        [EPOCH_US, EPOCH_US + 50_000],
        [[0, 0, 0], [50_000, 2, 4]],
        [quaternion_yaw(0), quaternion_yaw(np.pi / 2)],
    )
    one_us = trajectory.at(EPOCH_US + 1)
    np.testing.assert_allclose(one_us[:3, 3], [1, 0.00004, 0.00008], atol=1e-12)
    halfway = trajectory.at(np.array([[EPOCH_US, EPOCH_US + 25_000]]))
    assert halfway.shape == (1, 2, 4, 4)
    np.testing.assert_allclose(halfway[0, 1], pose_yaw(np.pi / 4, [25_000, 1, 2]), atol=1e-12)
    assert trajectory.at(np.array([], dtype=np.int64)).shape == (0, 4, 4)


def test_quaternion_interpolation_follows_shortest_arc_and_accepts_sign_changes() -> None:
    trajectory = PoseTrajectory(
        [0, 100], np.zeros((2, 3)), [quaternion_yaw(np.deg2rad(170)), quaternion_yaw(np.deg2rad(-170))]
    )
    np.testing.assert_allclose(trajectory.at(50), pose_yaw(np.pi), atol=1e-12)
    same_rotation = PoseTrajectory([0, 100], np.zeros((2, 3)), [[1, 0, 0, 0], [-2, 0, 0, 0]])
    np.testing.assert_allclose(same_rotation.at([0, 50, 100]), np.broadcast_to(np.eye(4), (3, 4, 4)))


def test_endpoint_extrapolation_is_limited_and_uses_pose_velocity() -> None:
    trajectory = PoseTrajectory(
        [EPOCH_US, EPOCH_US + 100_000],
        [[0, 0, 0], [10, 0, 0]],
        [quaternion_yaw(0), quaternion_yaw(np.pi / 2)],
    )
    np.testing.assert_allclose(trajectory.at(EPOCH_US - 50_000), pose_yaw(-np.pi / 4, [-5, 0, 0]), atol=1e-12)
    np.testing.assert_allclose(trajectory.at(EPOCH_US + 150_000), pose_yaw(3 * np.pi / 4, [15, 0, 0]), atol=1e-12)
    for timestamp in [EPOCH_US - 50_001, EPOCH_US + 150_001]:
        with pytest.raises(ValueError, match="extrapolation"):
            trajectory.at(timestamp)
    single = PoseTrajectory([EPOCH_US], [[1, 2, 3]], [[1, 0, 0, 0]])
    np.testing.assert_allclose(single.at(EPOCH_US - 50_000), pose_yaw(0, [1, 2, 3]))


@pytest.mark.parametrize("times,translation,quaternion", [
    ([], np.empty((0, 3)), np.empty((0, 4))),
    ([10, 10], np.zeros((2, 3)), [[1, 0, 0, 0]] * 2),
    ([20, 10], np.zeros((2, 3)), [[1, 0, 0, 0]] * 2),
    ([10.5], np.zeros((1, 3)), [[1, 0, 0, 0]]),
    ([10], [[np.nan, 0, 0]], [[1, 0, 0, 0]]),
    ([10], np.zeros((1, 3)), [[0, 0, 0, 0]]),
    ([10], np.zeros((1, 3)), [[np.nan, 0, 0, 0]]),
])
def test_invalid_trajectories_are_rejected(times: object, translation: object, quaternion: object) -> None:
    with pytest.raises(ValueError):
        PoseTrajectory(times, translation, quaternion)


def test_decompensation_undoes_known_translation_yaw_and_sensor_extrinsics() -> None:
    start, end = EPOCH_US, EPOCH_US + 50_000
    timestamps = np.array([start, start + 25_000, end])
    trajectory = PoseTrajectory(
        [start, end], [[0, 0, 0], [2, -1, 0.2]],
        [quaternion_yaw(0), quaternion_yaw(np.deg2rad(30))],
    )
    sensor_to_ego = pose_yaw(np.deg2rad(20), [0.5, -0.3, 2])
    points = np.array([[8, 2, -1], [15, 0, 0], [3, -1, 2]], dtype=float)
    reference = pose_yaw(np.deg2rad(30), [2, -1, 0.2]) @ sensor_to_ego
    world = points @ reference[:3, :3].T + reference[:3, 3]
    expected = []
    for fraction, point in zip([0, 0.5, 1], world):
        pose = pose_yaw(np.deg2rad(30) * fraction, np.array([2, -1, 0.2]) * fraction) @ sensor_to_ego
        expected.append(pose[:3, :3].T @ (point - pose[:3, 3]))
    raw = decompensate_points(points, timestamps, end, trajectory, sensor_to_ego)
    np.testing.assert_allclose(raw, expected, atol=1e-12)
    np.testing.assert_allclose(raw[-1], points[-1], atol=1e-12)
    acquired_poses = trajectory.at(timestamps) @ sensor_to_ego
    restored = np.einsum("nij,nj->ni", acquired_poses[:, :3, :3], raw) + acquired_poses[:, :3, 3]
    np.testing.assert_allclose(restored, world, atol=1e-12)
    np.testing.assert_allclose(compensated_points_to_world(points, end, trajectory, sensor_to_ego), world, atol=1e-12)


def test_reference_pose_override_and_empty_cloud() -> None:
    trajectory = stationary_trajectory(0, 50_000)
    points = np.array([[1, 0, 0]], dtype=float)
    override = pose_yaw(np.pi / 2, [3, 2, 1])
    expected = [[3, 3, 1]]
    np.testing.assert_allclose(decompensate_points(points, [0], 50_000, trajectory, np.eye(4), reference_sensor_to_world=override), expected)
    np.testing.assert_allclose(compensated_points_to_world(points, 50_000, trajectory, np.eye(4), reference_sensor_to_world=override), expected)
    assert decompensate_points(np.empty((0, 3)), [], 50_000, trajectory, np.eye(4)).shape == (0, 3)
    with pytest.raises(ValueError, match="point_timestamps_us"):
        decompensate_points(points, [0, 1], 50_000, trajectory, np.eye(4))


@pytest.mark.parametrize("resolution", [1, 2, 4])
def test_full_nominal_scan_uses_native_columns_and_fencepost_convention(resolution: int) -> None:
    points, rings = nominal_scan()
    start, end = EPOCH_US, EPOCH_US + 50_000
    trajectory = stationary_trajectory(start, end)
    timestamps = nominal_point_timestamps(points, rings, start, end, trajectory, np.eye(4), resolution, azimuth_phase_rad=0)
    physical_columns = np.arange(len(points)) // 32
    expected = start + np.floor(physical_columns / HDL32E_COLUMNS * 50_000).astype(np.int64)
    np.testing.assert_array_equal(timestamps, expected)
    assert timestamps[0] == start
    assert timestamps[-1] == end - 47
    assert timestamps.dtype == np.int64
    # Ring geometry must not introduce per-ring timestamp offsets.
    assert np.all(timestamps.reshape(-1, 32) == timestamps.reshape(-1, 32)[:, :1])


def test_nominal_alignment_refines_subcolumn_phase_at_four_times_resolution() -> None:
    phase = -2 * np.pi / HDL32E_COLUMNS / 4
    points, rings = nominal_scan(phase=phase)
    start, end = EPOCH_US, EPOCH_US + 50_000
    timestamps = nominal_point_timestamps(
        points, rings, start, end, stationary_trajectory(start, end), np.eye(4), azimuth_phase_rad=0
    )
    model_columns = np.minimum((np.arange(len(points)) // 32) * 4 + 1, HDL32E_COLUMNS * 4 - 1)
    expected = start + np.floor(model_columns / (HDL32E_COLUMNS * 4) * 50_000).astype(np.int64)
    np.testing.assert_array_equal(timestamps, expected)


def test_two_pass_nominal_alignment_recovers_moving_scan_acquisition_times() -> None:
    start, end = EPOCH_US, EPOCH_US + 50_000
    phase = 0.7
    raw_points, rings = nominal_scan(phase=phase)
    columns = np.arange(len(raw_points)) // 32
    expected = start + np.floor(columns / HDL32E_COLUMNS * 50_000).astype(np.int64)
    trajectory = PoseTrajectory(
        [start, end], [[0, 0, 0], [1, -0.2, 0]],
        [quaternion_yaw(0), quaternion_yaw(0.02)],
    )
    sensor_to_ego = pose_yaw(0.15, [0.5, 0.1, 2])
    # Build the source cloud independently from analytical poses at known times.
    fractions = (expected - start) / 50_000
    acquired = np.stack([
        pose_yaw(0.02 * fraction, np.array([1, -0.2, 0]) * fraction) @ sensor_to_ego
        for fraction in fractions
    ])
    world = np.einsum("nij,nj->ni", acquired[:, :3, :3], raw_points) + acquired[:, :3, 3]
    reference = pose_yaw(0.02, [1, -0.2, 0]) @ sensor_to_ego
    compensated = (world - reference[:3, 3]) @ reference[:3, :3]
    actual = nominal_point_timestamps(
        compensated, rings, start, end, trajectory, sensor_to_ego,
        azimuth_phase_rad=phase,
    )
    # Alignment near the scan seam can retain an uncertain boundary column; all
    # interior columns must recover the original clock, not compensated azimuth.
    np.testing.assert_array_equal(actual[2 * 32:-2 * 32], expected[2 * 32:-2 * 32])
    assert np.abs(actual - expected).max() <= 47
    recovered = decompensate_points(compensated, actual, end, trajectory, sensor_to_ego)
    np.testing.assert_allclose(recovered[2 * 32:-2 * 32], raw_points[2 * 32:-2 * 32], atol=1e-11)


def test_partial_scan_preserves_points_and_uses_nominal_order_clock() -> None:
    points, rings = nominal_scan(columns=2)
    points, rings = points[:37], rings[:37]
    start, end = EPOCH_US, EPOCH_US + 50_000
    timestamps = nominal_point_timestamps(points, rings, start, end, stationary_trajectory(start, end), np.eye(4))
    np.testing.assert_array_equal(timestamps, [start] * 32 + [start + 46] * 5)
    # One column is never stretched across a complete revolution.
    one = nominal_point_timestamps(points[:1], rings[:1], start, end, stationary_trajectory(start, end), np.eye(4))
    np.testing.assert_array_equal(one, [start])


def test_extra_edge_columns_are_preserved_and_clamped_to_sweep_bounds() -> None:
    points, rings = nominal_scan(columns=HDL32E_COLUMNS + 1)
    start, end = EPOCH_US, EPOCH_US + 50_000
    timestamps = nominal_point_timestamps(points, rings, start, end, stationary_trajectory(start, end), np.eye(4), azimuth_phase_rad=0)
    assert timestamps.shape == (len(points),)
    assert np.all((timestamps >= start) & (timestamps < end))
    assert np.all(timestamps[-32:] == end - 12)


def test_phase_initialization_uses_far_points_in_first_ten_columns() -> None:
    points, _ = nominal_scan(phase=0.7)
    expected = 0.7 - 4.5 * 2 * np.pi / HDL32E_COLUMNS
    assert estimate_nominal_phase(points) == pytest.approx(expected)
    assert estimate_nominal_phase(np.zeros((1, 3))) == 0
    assert estimate_nominal_phase(np.empty((0, 3))) == 0


@pytest.mark.parametrize("rings,resolution,start,end", [
    ([32], 4, 0, 50_000),
    ([-1], 4, 0, 50_000),
    ([0.5], 4, 0, 50_000),
    ([0], 3, 0, 50_000),
    ([0], 4, 50_000, 0),
    ([0], 4, 1.5, 50_000),
])
def test_invalid_nominal_timing_arguments_are_rejected(rings: object, resolution: int, start: int, end: int) -> None:
    with pytest.raises(ValueError):
        nominal_point_timestamps([[30, 0, 0]], rings, start, end, stationary_trajectory(0, 50_000), np.eye(4), resolution)


def test_arbitrary_reordered_cloud_is_not_silently_assigned_a_lidar_clock() -> None:
    trajectory = stationary_trajectory(0, 50_000)
    with pytest.raises(ValueError, match="firing columns"):
        nominal_point_timestamps(np.ones((32, 3)), np.zeros(32, dtype=int), 0, 50_000, trajectory, np.eye(4))
    with pytest.raises(ValueError, match="distinct"):
        nominal_point_timestamps(np.ones((2, 3)), [0, 0], 0, 50_000, trajectory, np.eye(4))
