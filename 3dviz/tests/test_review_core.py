"""Canonical observations retain rendering, layer and camera behavior."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_utils import AnnotationSet, AnnotationStatus, Box3D, FrameInfo, Modality, PointCloudFrame


class _FakeScene:
    def __init__(self):
        self.geometries = {}
        self.operations = []
        self.visibility = {}
        self.builtin_axes_calls = []
        self.camera_calls = []
        self.view = SimpleNamespace(set_post_processing=lambda enabled: None)
        self.camera = SimpleNamespace(look_at=lambda *args: self.camera_calls.append(args))

    def set_background(self, color):
        pass

    def show_axes(self, visible):
        self.builtin_axes_calls.append(visible)

    def show_geometry(self, name, visible):
        assert name in self.geometries
        self.visibility[name] = visible

    def geometry_is_visible(self, name):
        return self.visibility.get(name, True)

    def has_geometry(self, name):
        return name in self.geometries

    def add_geometry(self, name, geometry, material, add_downsampled_copy=True):
        self.geometries[name] = (geometry, material)
        self.operations.append(("add", name))

    def remove_geometry(self, name):
        self.geometries.pop(name)
        self.visibility.pop(name, None)
        self.operations.append(("remove", name))


class _FakeWidget:
    def __init__(self):
        self.scene = _FakeScene()
        self.camera_setups = []

    def set_view_controls(self, controls):
        pass

    def force_redraw(self):
        pass

    def setup_camera(self, *args):
        self.camera_setups.append(args)


@pytest.fixture
def renderer():
    pytest.importorskip("open3d")
    from threedviz.visualization.renderer import SceneRenderer

    result = SceneRenderer(_FakeWidget(), scaling=2.0)
    result.settings.selected_classes = {"car", "pedestrian"}
    return result


def _box(class_name="car", x=0):
    return Box3D(f"{class_name}:{x}", class_name, np.array([x, 0., 0.]), np.ones(3), np.eye(3))


def _frame(index=0, boxes=None):
    reference = f"lidar:{index}"
    return PointCloudFrame(
        FrameInfo(f"frame-{index}", "scene", index, index, {Modality.POINT_CLOUD}),
        "synthetic", reference, np.array([[index, 0, 0.0]]),
        AnnotationSet(reference, AnnotationStatus.AVAILABLE,
                      (_box(), _box("pedestrian", 3)) if boxes is None else boxes),
    )


def test_layer_colors_widths_class_filters_and_toggles_preserve_points_and_camera(renderer):
    from threedviz.visualization.renderer import BoxLayer

    frame = _frame()
    renderer.set_frame(frame)
    assert set(renderer._box_names) == {"boxes:car", "boxes:pedestrian"}
    cloud = renderer._cloud
    point_geometry = renderer.widget.scene.geometries["points"][0]
    renderer.widget.scene.operations.clear()
    layers = (BoxLayer("ground-truth", frame.annotations.boxes, (0, 1, 0)),
              BoxLayer("detections", (_box(x=2),), (1, 0, 0), line_width=4.5))
    renderer.set_box_layers(layers)
    assert renderer.box_layers is layers
    assert set(renderer._box_names) == {"layer:ground-truth:car", "layer:ground-truth:pedestrian", "layer:detections:car"}
    scene = renderer.widget.scene
    for name, (geometry, material) in scene.geometries.items():
        if name in {"points", "coordinate_axes"}:
            continue
        np.testing.assert_allclose(np.asarray(geometry.colors), np.tile([1, 0, 0] if "detections" in name else [0, 1, 0], (12, 1)))
        assert material.line_width == pytest.approx(9 if "detections" in name else 4)
    renderer.set_box_layers((replace(layers[0], visible=False), layers[1]))
    assert renderer._box_names == ["layer:detections:car"]
    renderer.settings.selected_classes = {"pedestrian"}
    renderer.update_boxes()
    assert renderer._box_names == []
    renderer.set_box_layers(layers)
    assert renderer._box_names == ["layer:ground-truth:pedestrian"]
    assert renderer._cloud is cloud
    assert scene.geometries["points"][0] is point_geometry
    assert not any(name == "points" for _, name in scene.operations)
    assert renderer.widget.camera_setups == [] and scene.camera_calls == []
    renderer.set_box_layers(None)
    assert renderer.box_layers is None
    assert renderer._box_names == ["boxes:pedestrian"]


@pytest.mark.parametrize("invalid", ["color", "width", "duplicate", "geometry", "hidden-geometry"])
def test_invalid_layer_update_is_transactional(renderer, invalid):
    from threedviz.visualization.renderer import BoxLayer

    frame = _frame()
    good = (BoxLayer("detections", (_box(),), (1, 0, 0)),)
    renderer.set_frame(frame, box_layers=good)
    if invalid == "color":
        candidate = (replace(good[0], color=(2, 0, 0)),)
    elif invalid == "width":
        candidate = (replace(good[0], line_width="invalid"),)
    elif invalid == "duplicate":
        candidate = good + good
    else:
        candidate = (replace(good[0], boxes=(object(),), visible=invalid != "hidden-geometry"),)
    geometries = renderer.widget.scene.geometries.copy()
    renderer.widget.scene.operations.clear()
    with pytest.raises(ValueError):
        renderer.set_box_layers(candidate)
    assert renderer.box_layers is good
    assert renderer.frame is frame
    assert renderer.widget.scene.geometries == geometries
    assert renderer.widget.scene.operations == []
    with pytest.raises(ValueError):
        renderer.set_frame(_frame(1), box_layers=candidate)
    assert renderer.frame is frame and renderer.box_layers is good
    assert renderer.widget.scene.operations == []


def test_atomic_frame_layer_replacement_and_default_layer_restore(renderer):
    from threedviz.visualization.renderer import BoxLayer

    renderer.set_frame(_frame(), box_layers=(BoxLayer("old", (_box(),), (1, 0, 0)),))
    next_frame = _frame(1)
    next_layers = (BoxLayer("new", (_box(x=100),), (0, 0, 1)),)
    renderer.set_frame(next_frame, box_layers=next_layers)
    assert renderer.frame is next_frame and renderer.box_layers is next_layers
    assert renderer._box_names == ["layer:new:car"]
    np.testing.assert_allclose(np.asarray(renderer._cloud.points), next_frame.points)
    renderer.set_frame(_frame(2))
    assert renderer._box_names == ["layer:new:car"]
    renderer.set_frame(_frame(3), box_layers=None)
    assert renderer.box_layers is None
    assert set(renderer._box_names) == {"boxes:car", "boxes:pedestrian"}
    renderer.set_box_layers(())
    assert renderer._box_names == []


def test_detection_layers_contribute_to_empty_cloud_camera_bounds(renderer):
    from threedviz.visualization.renderer import BoxLayer

    frame = replace(_frame(boxes=()), points=np.empty((0, 3)))
    renderer.set_frame(frame, box_layers=(BoxLayer("detections", (_box(x=100),), (1, 0, 0)),), reset_camera=True)
    assert renderer.widget.camera_setups
    _, bounds, center = renderer.widget.camera_setups[-1]
    np.testing.assert_allclose(center, [100, 0, 0])
    assert bounds.get_min_bound()[0] <= 99.5 and bounds.get_max_bound()[0] >= 100.5


def test_axes_have_fixed_sensor_scale_and_toggle_without_scene_or_camera_changes(renderer):
    frame = replace(_frame(), points=np.array([[-500., -300., -100.], [500., 300., 100.]]))
    renderer.set_frame(frame, reset_camera=True)
    scene = renderer.widget.scene
    axes = scene.geometries["coordinate_axes"][0]
    vertices = np.asarray(axes.vertices).copy()
    np.testing.assert_allclose(vertices.max(axis=0), [3., 3., 3.])
    assert np.all(vertices.min(axis=0) >= -.2)
    cloud, boxes = renderer._cloud, tuple(renderer._box_names)
    camera_calls = list(scene.camera_calls)
    camera_setups = list(renderer.widget.camera_setups)
    scene.operations.clear()
    for _ in range(5):
        renderer.show_axes(False)
        assert not renderer.axes_visible
        assert not scene.geometry_is_visible("coordinate_axes")
        renderer.show_axes(True)
        assert renderer.axes_visible
        assert scene.geometry_is_visible("coordinate_axes")
    assert scene.operations == []
    assert scene.builtin_axes_calls == [False]
    assert scene.geometries["coordinate_axes"][0] is axes
    np.testing.assert_array_equal(np.asarray(axes.vertices), vertices)
    assert renderer._cloud is cloud and renderer.frame is frame
    assert tuple(renderer._box_names) == boxes
    assert scene.camera_calls == camera_calls
    assert renderer.widget.camera_setups == camera_setups


def test_hidden_axes_persist_across_initial_load_frames_recolor_and_box_layers(renderer):
    from threedviz.visualization.renderer import BoxLayer

    renderer.show_axes(False)
    scene = renderer.widget.scene
    assert "coordinate_axes" not in scene.geometries
    renderer.set_frame(_frame())
    axes = scene.geometries["coordinate_axes"][0]
    assert not scene.geometry_is_visible("coordinate_axes")
    renderer.update_points(recolor=True)
    renderer.set_frame(_frame(1))
    renderer.set_box_layers((BoxLayer("far", (_box(x=10000),), (1, 0, 0)),))
    assert not renderer.axes_visible and not scene.geometry_is_visible("coordinate_axes")
    assert scene.geometries["coordinate_axes"][0] is axes
    renderer.show_axes(True)
    np.testing.assert_allclose(np.asarray(axes.vertices).max(axis=0), [3., 3., 3.])
    assert scene.geometry_is_visible("coordinate_axes")


def test_axes_are_excluded_from_camera_bounds_and_visibility_requires_boolean(renderer):
    frame = replace(_frame(boxes=()), points=np.empty((0, 3)))
    renderer.set_frame(frame, reset_camera=True)
    _, bounds, center = renderer.widget.camera_setups[-1]
    np.testing.assert_allclose(center, [0, 0, 0])
    np.testing.assert_allclose(bounds.get_min_bound(), [-10, -10, -5])
    np.testing.assert_allclose(bounds.get_max_bound(), [10, 10, 5])
    with pytest.raises(TypeError, match="boolean"):
        renderer.show_axes("false")
    assert renderer.axes_visible


def test_native_class_ids_are_retained_while_display_groups_control_styles(renderer):
    native_box = Box3D("native-box", "vehicle.private_car", np.zeros(3), np.ones(3), np.eye(3))
    frame = _frame(boxes=(native_box,))
    renderer.set_class_labels({"vehicle.private_car": "car"})
    renderer.set_frame(frame)
    assert renderer.frame is frame
    assert frame.annotations.boxes[0].class_id == "vehicle.private_car"
    assert renderer._box_names == ["boxes:car"]
    renderer.settings.selected_classes.clear()
    renderer.update_boxes()
    assert renderer._box_names == []


def test_camera_colors_use_canonical_mask_and_distance_fallback(renderer):
    from dataset_utils import PointColors, ProcessingProvenance
    from threedviz.processing import colorize_points

    points = np.array([[1., 0, 0], [2., 0, 0]])
    colors = PointColors(np.array([[.2, .4, .8], [.8, .4, .2]]), np.array([True, False]),
                         ProcessingProvenance("synthetic-rgb"))
    frame = replace(_frame(), points=points, rgb=colors)
    renderer.settings.color_scheme = "camera_rgb"
    renderer.set_frame(frame)
    expected = colorize_points(points, "gray", renderer.settings.max_distance)
    expected[0] = colors.values[0]
    np.testing.assert_allclose(renderer._cloud.colors, expected)
    assert renderer.frame.rgb is colors


def test_partial_acquisition_geometry_preserves_previous_display(renderer):
    from dataset_utils import AcquisitionGeometry, ProcessingProvenance

    frame = _frame()
    renderer.set_frame(frame)
    cloud = renderer._cloud
    partial = AcquisitionGeometry(frame.points + [1., 0, 0], np.array([False]),
                                  "lidar@acquisition", ProcessingProvenance("partial-poses"))
    renderer.settings.use_raw_points = True
    with pytest.raises(ValueError, match="every point"):
        renderer.set_frame(replace(frame, acquisition=partial))
    assert renderer.frame is frame and renderer._cloud is cloud
