"""Exercise camera exposure selection, geometry, sampling, and visibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from threedviz.processing.projection import CameraObservation, project_camera_rgb


def camera(
    channel: str = "CAM_FRONT",
    timestamp: int = 10_000,
    *,
    pose: np.ndarray | None = None,
    path: Path | str | None = None,
) -> CameraObservation:
    return CameraObservation(
        channel=channel,
        timestamp_us=timestamp,
        image_path=Path(path or f"{channel}-{timestamp}.png"),
        intrinsics=np.array([[2, 0, 2], [0, 2, 2], [0, 0, 1]], dtype=float),
        sensor_to_world=np.eye(4) if pose is None else pose,
        width=5,
        height=5,
    )


def solid(red: int, green: int, blue: int) -> np.ndarray:
    return np.tile(np.array([red, green, blue], dtype=np.uint8), (5, 5, 1))


def test_bilinear_rgb_samples_gradient_and_all_four_image_edges() -> None:
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(5)[None, :] * 50
    image[:, :, 1] = np.arange(5)[:, None] * 40
    image[:, :, 2] = 30
    # For z=2, pixel coordinates are x+2,y+2. Sampling is analytic here.
    points = np.array([[-0.75, -0.5, 2], [-2, -2, 2], [2, -2, 2], [-2, 2, 2], [2, 2, 2]])
    rgb, valid = project_camera_rgb(points, np.full(5, 10_000), [camera()], image_loader=lambda _: image)
    assert valid.all()
    np.testing.assert_allclose(rgb * 255, [[62.5, 60, 30], [0, 0, 30], [200, 0, 30], [0, 160, 30], [200, 160, 30]])


def test_world_points_use_actual_camera_rotation_and_translation() -> None:
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    pose[:3, 3] = [10, 0, 2]
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(5)[None, :] * 50
    rgb, valid = project_camera_rgb(
        np.array([[15, 0, 2], [15, 0, 3]]),
        np.array([10_000, 10_000]),
        [camera(pose=pose)],
        image_loader=lambda _: image,
    )
    assert valid.all()
    np.testing.assert_allclose(rgb[:, 0] * 255, [100, 80])


def test_nearest_actual_exposure_is_selected_for_each_point_with_earlier_ties() -> None:
    observations = [camera(timestamp=t) for t in [0, 40_000, 80_000]]
    images = {observations[0].image_path: solid(255, 0, 0),
              observations[1].image_path: solid(0, 255, 0),
              observations[2].image_path: solid(0, 0, 255)}
    rgb, valid = project_camera_rgb(
        np.tile([0, 0, 5], (4, 1)), np.array([5_000, 39_000, 60_000, 79_000]),
        list(reversed(observations)), image_loader=images.__getitem__,
    )
    assert valid.all()
    np.testing.assert_array_equal(rgb, [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])


def test_each_selected_exposure_has_its_own_pose() -> None:
    translated = np.eye(4)
    translated[0, 3] = 10
    observations = [camera(timestamp=0), camera(timestamp=50_000, pose=translated)]
    rgb, valid = project_camera_rgb(
        np.array([[0, 0, 5], [10, 0, 5]]), np.array([0, 50_000]), observations,
        image_loader=lambda path: solid(255, 0, 0) if "-0.png" in str(path) else solid(0, 0, 255),
    )
    assert valid.all()
    np.testing.assert_array_equal(rgb, [[1, 0, 0], [0, 0, 1]])


def test_behind_camera_outside_image_and_stale_points_keep_invalid_gray() -> None:
    calls = []
    points = np.array([[0, 0, -1], [0, 0, 0], [100, 0, 1], [0, 100, 1], [0, 0, 5]])
    rgb, valid = project_camera_rgb(
        points, np.array([10_000, 10_000, 10_000, 10_000, 200_001]), [camera()],
        image_loader=lambda path: calls.append(path),
    )
    assert not valid.any()
    np.testing.assert_array_equal(rgb, np.full((5, 3), 0.5))
    assert calls == []


def test_depth_buffer_excludes_rear_returns_with_configurable_absolute_tolerance() -> None:
    points = np.array([[0, 0, 5], [0, 0, 5.15], [0, 0, 5.3], [0, 0, 10], [5, 0, 10]])
    rgb, valid = project_camera_rgb(
        points, np.full(5, 10_000), [camera()], image_loader=lambda _: solid(255, 255, 0),
        occlusion_tolerance_m=0.2,
    )
    np.testing.assert_array_equal(valid, [True, True, False, False, True])
    np.testing.assert_array_equal(rgb[~valid], [[0.5] * 3] * 2)
    _, exact = project_camera_rgb(
        points, np.full(5, 10_000), [camera()], image_loader=lambda _: solid(255, 255, 0),
        occlusion_tolerance_m=0,
    )
    np.testing.assert_array_equal(exact, [True, False, False, False, True])


def test_front_return_still_occludes_when_its_nearest_image_is_another_exposure() -> None:
    observations = [camera(timestamp=0), camera(timestamp=40_000)]
    rgb, valid = project_camera_rgb(
        np.array([[0, 0, 10], [0, 0, 5]]), np.array([0, 30_000]), observations,
        image_loader=lambda _: solid(0, 0, 255),
    )
    np.testing.assert_array_equal(valid, [False, True])
    np.testing.assert_array_equal(rgb, [[0.5] * 3, [0, 0, 1]])


def test_stale_returns_do_not_occlude_current_observation() -> None:
    observations = [camera(timestamp=0), camera(timestamp=100_000)]
    _, valid = project_camera_rgb(
        np.array([[0, 0, 10], [0, 0, 5]]), np.array([0, 100_000]), observations,
        image_loader=lambda _: solid(0, 0, 255), max_time_delta_us=10_000,
    )
    assert valid.all()


def test_overlap_chooses_central_camera_then_time_then_channel_deterministically() -> None:
    off_center = np.eye(4)
    off_center[0, 3] = 2
    central_late = camera("B", timestamp=15_000)
    central_close = camera("C", timestamp=10_000)
    tie_first = camera("A", timestamp=10_000)
    off = camera("0", timestamp=10_000, pose=off_center)
    images = {off.image_path: solid(255, 0, 0), central_late.image_path: solid(0, 255, 0),
              central_close.image_path: solid(0, 0, 255), tie_first.image_path: solid(255, 255, 0)}
    point = np.array([[0, 0, 5]])
    time = np.array([10_000])
    for observations, expected in [
        ([off, central_late], [0, 1, 0]),
        ([central_late, central_close], [0, 0, 1]),
        ([tie_first, central_close], [1, 1, 0]),
    ]:
        forward, valid = project_camera_rgb(point, time, observations, image_loader=images.__getitem__)
        reverse, _ = project_camera_rgb(point, time, observations[::-1], image_loader=images.__getitem__)
        assert valid.all()
        np.testing.assert_array_equal(forward[0], expected)
        np.testing.assert_array_equal(reverse, forward)


def test_default_loader_reads_rgb_file(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "camera.png"
    Image.fromarray(solid(20, 40, 80)).save(path)
    rgb, valid = project_camera_rgb(np.array([[0, 0, 5]]), np.array([10_000]), [camera(path=path)])
    assert valid.all()
    np.testing.assert_allclose(rgb[0], np.array([20, 40, 80]) / 255)


def test_missing_matched_image_has_actionable_channel_and_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError, match="CAM_FRONT.*missing.png.*dataset root"):
        project_camera_rgb(np.array([[0, 0, 5]]), np.array([10_000]), [camera(path=path)])


@pytest.mark.parametrize("image", [np.zeros((3, 5, 3), dtype=np.uint8), np.zeros((5, 5)), np.zeros((5, 5, 3))])
def test_loader_must_return_rgb_without_silent_resizing(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uint8 RGB.*shape"):
        project_camera_rgb(np.array([[0, 0, 5]]), np.array([10_000]), [camera()], image_loader=lambda _: image)


def test_empty_cloud_and_no_cameras_need_no_images() -> None:
    def unexpected(_: Path) -> np.ndarray:
        pytest.fail("No image should be loaded")

    colors, valid = project_camera_rgb(np.empty((0, 3)), np.empty(0), [camera()], image_loader=unexpected)
    assert colors.shape == (0, 3) and valid.shape == (0,)
    colors, valid = project_camera_rgb(np.array([[0, 0, 5]]), np.array([10_000.5]), [], image_loader=unexpected)
    np.testing.assert_array_equal(colors, [[0.5] * 3])
    assert not valid.any()


@pytest.mark.parametrize("points,times", [
    ([], []), ([[0, 0]], [0]), ([[np.nan, 0, 1]], [0]), ([[np.inf, 0, 1]], [0]),
    ([[0, 0, 1]], []), ([[0, 0, 1]], [[0]]), ([[0, 0, 1]], [np.nan]),
    ([[0, 0, 1]], [np.inf]), ([[0, 0, 1]], [-1]),
])
def test_malformed_points_and_times_are_rejected(points: object, times: object) -> None:
    with pytest.raises(ValueError):
        project_camera_rgb(points, times, [])


@pytest.mark.parametrize("field,value", [
    ("channel", ""), ("channel", None), ("timestamp_us", -1), ("timestamp_us", 0.5),
    ("width", 0), ("height", 2.5), ("intrinsics", np.eye(4)),
    ("intrinsics", [[-1, 0, 2], [0, 2, 2], [0, 0, 1]]),
    ("intrinsics", [[2, 0, 2], [0, 2, 2], [0, 0, 2]]),
    ("intrinsics", [[np.nan, 0, 2], [0, 2, 2], [0, 0, 1]]),
    ("sensor_to_world", np.eye(3)), ("sensor_to_world", 2 * np.eye(4)),
    ("sensor_to_world", np.diag([-1, 1, 1, 1])),
])
def test_malformed_camera_metadata_is_rejected(field: str, value: object) -> None:
    values = dict(channel="CAM_FRONT", timestamp_us=10_000, image_path=Path("image.png"),
                  intrinsics=np.eye(3), sensor_to_world=np.eye(4), width=5, height=5)
    values[field] = value
    with pytest.raises(ValueError):
        CameraObservation(**values)


@pytest.mark.parametrize("options", [
    {"max_time_delta_us": -1}, {"max_time_delta_us": np.inf}, {"max_time_delta_us": 1.5},
    {"occlusion_tolerance_m": -1}, {"occlusion_tolerance_m": np.inf}, {"occlusion_tolerance_m": np.nan},
])
def test_invalid_projection_options_are_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        project_camera_rgb(np.empty((0, 3)), np.empty(0), [], **options)


def test_duplicate_or_untyped_camera_metadata_is_rejected() -> None:
    for observations in [[camera(), camera()], [None]]:
        with pytest.raises(ValueError):
            project_camera_rgb(np.empty((0, 3)), np.empty(0), observations)


def test_camera_matrices_are_copied_and_read_only() -> None:
    pose = np.eye(4)
    observation = camera(pose=pose)
    pose[0, 3] = 100
    assert observation.sensor_to_world[0, 3] == 0
    assert not observation.sensor_to_world.flags.writeable
    assert not observation.intrinsics.flags.writeable
