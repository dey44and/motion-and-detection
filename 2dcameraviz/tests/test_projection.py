"""Geometric checks for camera-space clipping, transforms and visibility."""

from types import SimpleNamespace

import numpy as np
import pytest

from cameraviz.processing import ProjectedBox, class_color, project_box, project_boxes


def camera(*, pose=None, intrinsics=None, channel="CAM_FRONT", width=101, height=101):
    return SimpleNamespace(
        info=SimpleNamespace(channel=channel),
        image=np.zeros((height, width, 3), dtype=np.uint8),
        intrinsics=np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
        if intrinsics is None else intrinsics,
        sensor_to_world=np.eye(4) if pose is None else pose,
        timestamp_us=0,
    )


def box(*, center=(0, 0, 10), size=(2, 2, 2), rotation=None, class_name="car", token="box"):
    return SimpleNamespace(
        center=np.asarray(center, dtype=float),
        size=np.asarray(size, dtype=float),
        rotation=np.eye(3) if rotation is None else rotation,
        class_name=class_name,
        token=token,
    )


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_front_box_has_twelve_edges_and_perspective_bounds():
    projected = project_box(box(), camera())
    assert isinstance(projected, ProjectedBox)
    assert projected.class_name == "car"
    assert projected.token == "box"
    assert projected.segments.shape == (12, 2, 2)
    np.testing.assert_allclose(projected.rectangle, [50 - 100 / 9, 50 - 100 / 9, 50 + 100 / 9, 50 + 100 / 9])
    assert not projected.segments.flags.writeable


def test_length_width_height_are_distinct_local_axes():
    projected = project_box(box(size=(4, 2, 2)), camera())
    np.testing.assert_allclose(projected.rectangle, [50 - 200 / 9, 50 - 100 / 9, 50 + 200 / 9, 50 + 100 / 9])
    rotated = project_box(box(size=(4, 2, 2), rotation=rotation_z(np.pi / 2)), camera())
    np.testing.assert_allclose(rotated.rectangle, [50 - 100 / 9, 50 - 200 / 9, 50 + 100 / 9, 50 + 200 / 9])


def test_camera_optical_forward_can_be_world_x_with_translation():
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    pose[:3, 3] = [4, 2, 3]
    projected = project_box(box(center=(14, 2, 3), rotation=pose[:3, :3]), camera(pose=pose))
    np.testing.assert_allclose(projected.rectangle, [50 - 100 / 9, 50 - 100 / 9, 50 + 100 / 9, 50 + 100 / 9])


def test_camera_and_box_pitch_roll_and_translation_use_full_rigid_transforms():
    local_rotation = rotation_z(0.5) @ rotation_y(-0.3)
    optical_center = np.array([0.3, -0.2, 7.0])
    original = project_box(box(center=optical_center, size=(2, 1, 3), rotation=local_rotation), camera())
    pose = np.eye(4)
    pose[:3, :3] = rotation_y(0.7) @ rotation_x(-0.4) @ rotation_z(0.3)
    pose[:3, 3] = [4, -2, 7]
    world_box = box(
        center=pose[:3, :3] @ optical_center + pose[:3, 3],
        size=(2, 1, 3),
        rotation=pose[:3, :3] @ local_rotation,
    )
    transformed = project_box(world_box, camera(pose=pose))
    np.testing.assert_allclose(transformed.rectangle, original.rectangle, atol=1e-12)
    np.testing.assert_allclose(transformed.segments, original.segments, atol=1e-12)


def test_intrinsic_skew_and_noncentral_principal_point_are_applied():
    intrinsics = np.array([[100, 30, 40], [0, 80, 45], [0, 0, 1]], dtype=float)
    projected = project_box(box(), camera(intrinsics=intrinsics))
    np.testing.assert_allclose(projected.rectangle, [40 - 130 / 9, 45 - 80 / 9, 40 + 130 / 9, 45 + 80 / 9])


@pytest.mark.parametrize("center", [(0, 0, -10), (100, 0, 10), (-100, 0, 10), (0, 100, 10), (0, -100, 10)])
def test_boxes_wholly_behind_or_outside_image_are_omitted(center):
    assert project_box(box(center=center), camera()) is None


def test_screen_bounding_rectangle_overlap_does_not_invent_visibility():
    # Its corner bounds overlap the image's lower-left corner, but this thin
    # diagonal cuboid stays beyond the frustum: y - x is about 16 metres.
    outside = box(center=(-8, 8, 10), size=(20, 0.2, 1), rotation=rotation_z(np.pi / 4))
    assert project_box(outside, camera()) is None


def test_partially_visible_box_is_clipped_to_native_image_bounds():
    projected = project_box(box(center=(5, 0, 10), size=(4, 4, 2)), camera())
    np.testing.assert_allclose(projected.rectangle, [50 + 300 / 11, 50 - 200 / 9, 100, 50 + 200 / 9])
    assert len(projected.segments) > 0
    assert np.all(projected.segments >= 0)
    assert np.all(projected.segments <= 100)
    assert np.any(projected.segments[:, :, 0] == 100)


def test_box_crossing_near_plane_is_clipped_before_perspective_division():
    projected = project_box(box(center=(0, 0, 0.15), size=(0.1, 0.1, 0.2)), camera(), near_plane=0.1)
    assert projected.segments.shape == (8, 2, 2)
    assert np.isfinite(projected.segments).all()
    np.testing.assert_allclose(projected.rectangle, [0, 0, 100, 100], atol=1e-12)
    assert np.all(projected.segments >= 0)
    assert np.all(projected.segments <= 100)
    assert project_box(box(center=(0, 0, 0.04), size=(0.05, 0.05, 0.1)), camera(), near_plane=0.1) is None


def test_visible_face_can_fill_image_with_all_original_edges_offscreen():
    projected = project_box(box(size=(100, 100, 2)), camera())
    assert projected is not None
    assert projected.segments.shape == (0, 2, 2)
    np.testing.assert_allclose(projected.rectangle, [0, 0, 100, 100])


def test_camera_inside_box_still_has_a_visible_projection():
    projected = project_box(box(center=(0, 0, 0), size=(100, 100, 2)), camera())
    assert projected is not None
    np.testing.assert_allclose(projected.rectangle, [0, 0, 100, 100])
    assert projected.segments.shape == (0, 2, 2)


def test_each_camera_uses_its_own_pose_and_preserves_visible_box_order():
    backwards = np.eye(4)
    backwards[:3, :3] = rotation_y(np.pi)
    frame = SimpleNamespace(
        cameras=(camera(), camera(channel="CAM_BACK", pose=backwards)),
        boxes=(box(token="front"), box(center=(0, 0, -10), token="back"), box(center=(0, 0, 8), token="front-two")),
    )
    projected = project_boxes(frame)
    assert tuple(projected) == ("CAM_FRONT", "CAM_BACK")
    assert tuple(item.token for item in projected["CAM_FRONT"]) == ("front", "front-two")
    assert tuple(item.token for item in projected["CAM_BACK"]) == ("back",)


def test_duplicate_camera_channel_is_rejected():
    with pytest.raises(ValueError, match="Duplicate camera channel"):
        project_boxes(SimpleNamespace(cameras=(camera(), camera()), boxes=()))


@pytest.mark.parametrize("near", [0, -0.1, np.inf, np.nan, True, "bad"])
def test_invalid_near_plane_is_rejected(near):
    with pytest.raises(ValueError, match="near_plane"):
        project_box(box(), camera(), near_plane=near)


def test_nonrigid_calibration_invalid_intrinsics_and_box_dimensions_are_rejected():
    pose = np.eye(4)
    pose[0, 0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        project_box(box(), camera(pose=pose))
    reflected = np.diag([-1.0, 1, 1])
    with pytest.raises(ValueError, match="proper rotation"):
        project_box(box(rotation=reflected), camera())
    intrinsics = np.eye(3)
    intrinsics[2, 0] = 0.1
    with pytest.raises(ValueError, match="pinhole"):
        project_box(box(), camera(intrinsics=intrinsics))
    with pytest.raises(ValueError, match="size must be positive"):
        project_box(box(size=(0, 1, 1)), camera())


def test_projected_box_copies_and_protects_its_segment_array():
    segments = np.zeros((1, 2, 2))
    projected = ProjectedBox("car", "one", segments, (0, 0, 0, 0))
    segments[0, 0, 0] = 2
    assert projected.segments[0, 0, 0] == 0
    with pytest.raises(ValueError):
        projected.segments[0, 0, 0] = 2


@pytest.mark.parametrize("name,color", [
    ("car", "#0d59bf"), ("truck", "#bf400d"), ("bus", "#8c33b2"),
    ("trailer", "#8c5926"), ("construction_vehicle", "#a67300"),
    ("pedestrian", "#d91a33"), ("motorcycle", "#008066"),
    ("bicycle", "#1a8c1a"), ("traffic_cone", "#e65900"), ("barrier", "#4c5966"),
])
def test_known_class_palette_matches_3d_viewer(name, color):
    assert class_color(name) == color


def test_unknown_class_color_is_deterministic_and_valid_hex():
    color = class_color("vehicle.emergency.ambulance")
    assert color == class_color("vehicle.emergency.ambulance")
    assert len(color) == 7 and color[0] == "#"
    assert 0 <= int(color[1:], 16) <= 0xFFFFFF
    assert color != class_color("vehicle.emergency.police")
    with pytest.raises(ValueError, match="nonempty string"):
        class_color("")
