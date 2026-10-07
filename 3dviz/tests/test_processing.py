"""Verify geometry conventions and the viewer's color behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from threedviz.processing import (
    BOX_EDGES,
    COLOR_SCHEMES,
    box_corners,
    class_color,
    colorize_points,
)


def test_distance_colors_use_euclidean_range_and_clip() -> None:
    points = np.array([[0, 0, 0], [3, 4, 0], [0, 0, -5], [0, 10, 0], [100, 0, 0]])
    rgb = colorize_points(points, max_distance=10)
    np.testing.assert_allclose(rgb[:, 0], [0.12, 0.42, 0.42, 0.72, 0.72])
    np.testing.assert_allclose(rgb[:, 0], rgb[:, 1])
    np.testing.assert_allclose(rgb[:, 1], rgb[:, 2])


@pytest.mark.parametrize("scheme", COLOR_SCHEMES)
def test_colors_do_not_depend_on_cloud_extent_or_orientation(scheme: str) -> None:
    points = np.array([[1, 2, 2], [6, 8, 0]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    expected = colorize_points(points, scheme, max_distance=15)
    np.testing.assert_allclose(colorize_points(points @ rotation.T, scheme, 15), expected)
    extra_point = np.vstack([points, [300, 300, 300]])
    np.testing.assert_allclose(colorize_points(extra_point, scheme, 15)[:2], expected)
    assert expected.shape == points.shape
    assert np.all((expected >= 0) & (expected <= 1))


@pytest.mark.parametrize("scheme", COLOR_SCHEMES)
def test_empty_point_cloud_and_large_finite_coordinates(scheme: str) -> None:
    assert colorize_points(np.empty((0, 3)), scheme).shape == (0, 3)
    with np.errstate(over="raise", invalid="raise"):
        far = colorize_points(np.array([[1e308, -1e308, 1e308]]), scheme)
    np.testing.assert_allclose(far, colorize_points(np.array([[80, 0, 0]]), scheme))


@pytest.mark.parametrize("max_distance", [0, -1, np.inf, np.nan, "bad", None])
def test_invalid_distance_limit_is_rejected(max_distance: float) -> None:
    with pytest.raises(ValueError, match="max_distance"):
        colorize_points(np.zeros((1, 3)), max_distance=max_distance)


@pytest.mark.parametrize("points", [[], np.zeros(3), np.zeros((2, 4)), [[np.nan, 0, 0]], [[np.inf, 0, 0]]])
def test_invalid_point_coordinates_are_rejected(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        colorize_points(points)


def test_unknown_color_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown color scheme"):
        colorize_points(np.zeros((1, 3)), scheme="rainbow")


def test_rotated_box_corners_follow_length_width_height_and_center() -> None:
    box = SimpleNamespace(
        center=np.array([10, -3, 2]),
        size=np.array([4, 2, 6]),
        rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    )
    expected = np.array([
        [11, -5, -1], [11, -1, -1], [9, -1, -1], [9, -5, -1],
        [11, -5, 5], [11, -1, 5], [9, -1, 5], [9, -5, 5],
    ])
    corners = box_corners(box)
    np.testing.assert_allclose(corners, expected)
    np.testing.assert_allclose(corners.mean(axis=0), box.center)


def test_box_edges_form_twelve_unique_edges_with_correct_lengths() -> None:
    box = SimpleNamespace(center=np.zeros(3), size=np.array([4, 2, 6]), rotation=np.eye(3))
    corners = box_corners(box)
    assert BOX_EDGES.shape == (12, 2)
    assert len({tuple(sorted(edge)) for edge in BOX_EDGES}) == 12
    np.testing.assert_array_equal(np.bincount(BOX_EDGES.ravel()), np.full(8, 3))
    lengths = np.linalg.norm(corners[BOX_EDGES[:, 0]] - corners[BOX_EDGES[:, 1]], axis=1)
    np.testing.assert_allclose(np.sort(lengths), [2] * 4 + [4] * 4 + [6] * 4)


@pytest.mark.parametrize("attribute,value", [
    ("center", [0, 0]),
    ("center", [0, 0, np.nan]),
    ("size", [1, 2]),
    ("size", [0, 2, 3]),
    ("size", [-1, 2, 3]),
    ("rotation", np.eye(2)),
    ("rotation", 2 * np.eye(3)),
    ("rotation", np.diag([-1, 1, 1])),
])
def test_invalid_box_geometry_is_rejected(attribute: str, value: np.ndarray) -> None:
    box = SimpleNamespace(center=np.zeros(3), size=np.ones(3), rotation=np.eye(3))
    setattr(box, attribute, value)
    with pytest.raises(ValueError):
        box_corners(box)


def test_known_and_unknown_class_colors_are_visible_and_stable() -> None:
    known = ["car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
             "motorcycle", "bicycle", "traffic_cone", "barrier"]
    colors = [class_color(name) for name in known]
    assert len(set(colors)) == len(known)
    for name in [*known, "animal.deer", "custom_sensor_class"]:
        color = class_color(name)
        assert color == class_color(name)
        assert len(color) == 3
        assert all(0 <= component <= 1 for component in color)
        assert min(color) <= 0.35


def test_fallback_class_color_is_independent_of_python_hash_seed() -> None:
    source = str(Path(__file__).resolve().parents[1] / "src")
    command = [sys.executable, "-c", "import json; from threedviz.processing import class_color; print(json.dumps(class_color('unknown.vehicle')))"]
    outputs = []
    for seed in ["1", "9999"]:
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": source}
        outputs.append(json.loads(subprocess.check_output(command, env=env, text=True)))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("class_name", ["", None, 123])
def test_invalid_class_name_is_rejected(class_name: str) -> None:
    with pytest.raises(ValueError, match="class_name"):
        class_color(class_name)
