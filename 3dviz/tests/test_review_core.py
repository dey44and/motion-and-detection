"""Shared review-viewer capabilities without creating a native GUI window."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from threedviz.dataloader import BoundingBox3D, DatasetAdapter, DatasetError, FrameData, FrameInfo, create_dataset
from threedviz.dataloader.splits import MINI_TRAIN, MINI_VAL, TEST, TRAIN, VAL, scene_names_for_split


def _nuscenes(tmp_path, version="v1.0-mini", names=("scene-0061", "scene-0103", "scene-0916")):
    root = tmp_path / version
    root.mkdir(parents=True)
    s = np.sqrt(.5)
    tables = {
        "scene": [{"token": f"scene-{index}", "name": name} for index, name in enumerate(names)],
        "sample": [{"token": f"frame-{index}", "scene_token": f"scene-{index}", "timestamp": 1_000_000 + index}
                   for index in range(len(names))],
        "sensor": [{"token": "lidar", "channel": "LIDAR_TOP", "modality": "lidar"}],
        "calibrated_sensor": [{"token": "calibration", "sensor_token": "lidar", "translation": [1, 2, 3], "rotation": [s, s, 0, 0]}],
        "ego_pose": [{"token": "pose", "translation": [10, 20, 30], "rotation": [s, 0, 0, s]}],
        "sample_data": [{"token": f"lidar-{index}", "sample_token": f"frame-{index}", "is_key_frame": True,
                         "calibrated_sensor_token": "calibration", "ego_pose_token": "pose", "filename": "scan.bin"}
                        for index in range(len(names))],
        "sample_annotation": [{"token": "car", "sample_token": "frame-0", "category_name": "vehicle.car",
                               "translation": [14, 25, 38], "size": [2, 4, 1.5], "rotation": [s, 0, s, 0]}],
    }
    for name, records in tables.items():
        (root / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")
    np.array([[1, 2, 3, 5, 0]], dtype="<f4").tofile(root / "scan.bin")
    return create_dataset("nuscenes", root=tmp_path, version=version)


def test_reference_transform_preserves_full_ego_and_sensor_orientation(tmp_path):
    adapter = _nuscenes(tmp_path)
    frame = adapter.load_frame("frame-0")
    expected = np.array([[0, 0, 1, 8], [1, 0, 0, 21], [0, 1, 0, 33], [0, 0, 0, 1]])
    np.testing.assert_allclose(frame.reference_sensor_to_world, expected, atol=1e-12)
    box = frame.boxes[0]
    world_center = frame.reference_sensor_to_world[:3, :3] @ box.center + frame.reference_sensor_to_world[:3, 3]
    world_rotation = frame.reference_sensor_to_world[:3, :3] @ box.rotation
    np.testing.assert_allclose(world_center, [14, 25, 38], atol=1e-12)
    np.testing.assert_allclose(world_rotation, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)


def test_reference_transform_optional_for_existing_adapters_and_identity_in_demo():
    frame = FrameData(FrameInfo("frame", "scene", 0, 0), np.empty((0, 3)), ())
    assert frame.reference_sensor_to_world is None
    demo = create_dataset("demo")
    info = demo.list_frames(demo.list_scenes()[0].token)[0]
    np.testing.assert_array_equal(demo.load_frame(info.token).reference_sensor_to_world, np.eye(4))


def test_official_split_memberships_and_release_compatibility():
    assert tuple(map(len, (TRAIN, VAL, TEST, MINI_TRAIN, MINI_VAL))) == (700, 150, 150, 8, 2)
    assert len(TRAIN | VAL | TEST) == 1000
    assert MINI_TRAIN | MINI_VAL <= TRAIN | VAL
    assert not MINI_TRAIN & MINI_VAL
    assert MINI_VAL == {"scene-0103", "scene-0916"}
    assert "scene-0757" in MINI_TRAIN
    assert scene_names_for_split("train", "v1.0-trainval") is TRAIN
    assert scene_names_for_split("val", "v1.0-trainval") is VAL
    assert scene_names_for_split("test", "v1.0-test") is TEST
    assert scene_names_for_split(" all ", "custom-version") is None


def test_nuscenes_split_filters_keep_scene_and_frame_identities(tmp_path):
    adapter = _nuscenes(tmp_path)
    all_scenes = adapter.list_scenes()
    assert adapter.list_scenes_for_split() is all_scenes
    validation = adapter.list_scenes_for_split("val")
    assert [scene.name for scene in validation] == ["scene-0103", "scene-0916"]
    assert adapter.list_scenes_for_split("mini_val") == validation
    assert adapter.list_scenes_for_split("train") == adapter.list_scenes_for_split("mini_train")
    assert [scene.name for scene in adapter.list_scenes_for_split("train")] == ["scene-0061"]
    assert all(scene is next(item for item in all_scenes if item.token == scene.token) for scene in validation)
    assert adapter.list_frames(validation[0].token)[0].token == "frame-1"


@pytest.mark.parametrize("split,version", [
    ("test", "v1.0-mini"), ("mini_val", "v1.0-trainval"),
    ("val", "v1.0-test"), ("train", "v1.0-test"), ("val", "custom"),
    ("missing", "v1.0-mini"), ("", "v1.0-mini"), (None, "v1.0-mini"),
])
def test_incompatible_and_unknown_splits_fail_explicitly(split, version):
    with pytest.raises(DatasetError):
        scene_names_for_split(split, version)


def test_existing_adapter_split_default_and_demo_validation_aliases():
    demo = create_dataset("demo")
    assert DatasetAdapter.list_scenes_for_split(demo, "all") is demo.list_scenes()
    with pytest.raises(DatasetError, match="does not support split"):
        DatasetAdapter.list_scenes_for_split(demo, "val")
    assert demo.list_scenes_for_split("val") is demo.list_scenes()
    assert demo.list_scenes_for_split("mini_val") is demo.list_scenes()
    with pytest.raises(DatasetError, match="Demo data supports"):
        demo.list_scenes_for_split("test")


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
    return BoundingBox3D(np.array([x, 0, 0]), np.ones(3), np.eye(3), class_name)


def _frame(index=0, boxes=None):
    return FrameData(FrameInfo(f"frame-{index}", "scene", index, index), np.array([[index, 0, 0.0]]),
                     (_box(), _box("pedestrian", 3)) if boxes is None else boxes)


def test_layer_colors_widths_class_filters_and_toggles_preserve_points_and_camera(renderer):
    from threedviz.visualization.renderer import BoxLayer

    frame = _frame()
    renderer.set_frame(frame)
    assert set(renderer._box_names) == {"boxes:car", "boxes:pedestrian"}
    cloud = renderer._cloud
    point_geometry = renderer.widget.scene.geometries["points"][0]
    renderer.widget.scene.operations.clear()
    layers = (BoxLayer("ground-truth", frame.boxes, (0, 1, 0)),
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
        bad_box = replace(_box(), center=np.array([np.nan, 0, 0]))
        candidate = (replace(good[0], boxes=(bad_box,), visible=invalid != "hidden-geometry"),)
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


def test_atomic_frame_layer_replacement_and_legacy_restore(renderer):
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

    frame = replace(_frame(), points=np.empty((0, 3)), boxes=())
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
    frame = replace(_frame(), points=np.empty((0, 3)), boxes=())
    renderer.set_frame(frame, reset_camera=True)
    _, bounds, center = renderer.widget.camera_setups[-1]
    np.testing.assert_allclose(center, [0, 0, 0])
    np.testing.assert_allclose(bounds.get_min_bound(), [-10, -10, -5])
    np.testing.assert_allclose(bounds.get_max_bound(), [10, 10, 5])
    with pytest.raises(TypeError, match="boolean"):
        renderer.show_axes("false")
    assert renderer.axes_visible
