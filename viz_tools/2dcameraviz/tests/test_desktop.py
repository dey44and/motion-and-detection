"""Exercise actual Tk widgets, image overlays, navigation and load recovery."""

from dataclasses import replace
import gc
import os
import time

import pytest


pytestmark = [
    pytest.mark.desktop,
    pytest.mark.skipif(os.environ.get("CAMERA_VIZ_DESKTOP_TESTS") != "1", reason="Set CAMERA_VIZ_DESKTOP_TESTS=1 with a desktop display."),
]


@pytest.fixture(autouse=True)
def _collect_destroyed_tk_widgets():
    yield
    # Tk objects from a closed test window must be finalized on this thread,
    # before a later viewer's loader can trigger cyclic garbage collection.
    gc.collect()


def _wait(root, condition, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for the viewer.")


def test_native_viewer_navigation_overlays_and_failure_recovery(monkeypatch):
    import tkinter as tk
    from cameraviz.ui.app import CameraViewer

    root = tk.Tk()
    viewer = CameraViewer(root, dataset="demo", autoload=True)
    try:
        _wait(root, lambda: viewer.frame_data is not None and not viewer.busy and all(panel._photo is not None for panel in viewer.panels.values()))
        assert viewer.last_error is None
        expected = [
            ("CAM_FRONT_LEFT", 0, 0), ("CAM_FRONT", 0, 1), ("CAM_FRONT_RIGHT", 0, 2),
            ("CAM_BACK_RIGHT", 1, 0), ("CAM_BACK", 1, 1), ("CAM_BACK_LEFT", 1, 2),
        ]
        assert [(name, panel.info.row, panel.info.column) for name, panel in viewer.panels.items()] == expected
        assert all(panel._photo is not None for panel in viewer.panels.values())
        assert all(panel.canvas.find_withtag("camera_image") for panel in viewer.panels.values())
        assert any(panel.canvas.find_withtag("bounding_box") for panel in viewer.panels.values())
        assert "Frame 1 / 24" in viewer.frame_label.cget("text")

        viewer.show_classes(False)
        assert not any(panel.canvas.find_withtag("bounding_box") for panel in viewer.panels.values())
        viewer.show_classes(True)
        viewer.mode_var.set("2D rectangles")
        viewer.width_var.set(4)
        viewer._width_changed("4")
        rectangles = [
            (panel, item) for panel in viewer.panels.values()
            for item in panel.canvas.find_withtag("bounding_box")
        ]
        assert rectangles
        assert all(panel.canvas.type(item) == "rectangle" for panel, item in rectangles)
        assert all(float(panel.canvas.itemcget(item, "width")) == 4 for panel, item in rectangles)
        panel, item = rectangles[0]
        box_class = next(viewer._class_key(box.class_id) for box in panel.boxes if viewer._class_key(box.class_id) in panel.visible_classes)
        monkeypatch.setattr("cameraviz.ui.app.colorchooser.askcolor", lambda *args, **kwargs: ((255, 0, 255), "#ff00ff"))
        viewer.choose_color(box_class)
        assert viewer.class_colors[box_class] == "#ff00ff"
        assert panel.canvas.itemcget(panel.canvas.find_withtag("bounding_box")[0], "outline") == "#ff00ff"

        old_token = viewer.frame_data.info.frame_id
        viewer.step_frame(1)
        _wait(root, lambda: not viewer.busy and viewer.frame_index == 1)
        assert viewer.frame_data.info.frame_id != old_token
        assert viewer.mode_var.get() == "2D rectangles"
        assert viewer.width_var.get() == 4
        assert viewer.class_colors[box_class] == "#ff00ff"

        # A pending slider debounce must not override a navigation request.
        viewer._slider_changed("8")
        viewer.step_frame(1)
        _wait(root, lambda: not viewer.busy and viewer.frame_index == 2)
        elapsed = time.monotonic()
        _wait(root, lambda: time.monotonic() - elapsed > .2)
        assert viewer.frame_index == 2
        viewer.step_frame(-1)
        _wait(root, lambda: not viewer.busy and viewer.frame_index == 1)

        # A missing frame must retain all existing images and navigation.
        old_frame = viewer.frame_data
        old_images = {name: panel.camera for name, panel in viewer.panels.items()}
        load_frame = viewer.loader.load_cameras
        def fail(_token):
            raise FileNotFoundError("Missing camera image: recovery-test.jpg")
        viewer.loader.load_cameras = fail
        viewer.step_frame(1)
        _wait(root, lambda: not viewer.busy)
        assert "recovery-test.jpg" in viewer.last_error
        assert viewer.frame_data is old_frame
        assert viewer.frame_index == 1
        assert all(panel.camera is old_images[name] for name, panel in viewer.panels.items())
        assert str(viewer.next_button.cget("state")) == "normal"
        viewer.loader.load_cameras = load_frame
        viewer.step_frame(1)
        _wait(root, lambda: not viewer.busy and viewer.frame_index == 2)
        assert viewer.last_error is None

        viewer.scene_combo.current(1)
        viewer._select_scene()
        _wait(root, lambda: not viewer.busy and viewer.scene_index == 1)
        assert viewer.frame_index == 0
        assert viewer.frame_data.info.scene_id == "demo-roundabout"
        root.geometry("1100x650")
        _wait(root, lambda: root.winfo_width() == 1100)
        # Wait for the debounced resize and verify original aspect ratio.
        _wait(root, lambda: all(panel._photo_size is not None and panel._photo_size[0] <= panel.canvas.winfo_width() for panel in viewer.panels.values()))
        for panel in viewer.panels.values():
            width, height = panel._photo_size
            assert abs(width / height - 16 / 9) < .02
            for item in panel.canvas.find_withtag("bounding_box"):
                assert float(panel.canvas.itemcget(item, "width")) == 4

        viewer.dataset_var.set("nuscenes")
        viewer.root_var.set("/nonexistent/camera-viewer-test-dataset")
        viewer.load_dataset()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error
        assert viewer.frame_data.info.scene_id == "demo-roundabout"
        assert str(viewer.scene_combo.cget("state")) == "readonly"
    finally:
        viewer.close()


def test_custom_layout_missing_camera_and_annotation_free_dataset():
    import tkinter as tk
    from visual_utils import AnnotationSet, Modality, SensorInfo, register_loader
    from cameraviz.dataloader.demo import DemoDataset
    from cameraviz.ui.app import CameraViewer

    class CustomDataset(DemoDataset):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self._metadata = replace(self.metadata, classes=(), splits=("all", "val"), sensors=(
                SensorInfo("SINGLE_CAMERA", Modality.CAMERAS, "Custom view"),
                SensorInfo("MISSING_CAMERA", Modality.CAMERAS, "Optional view"),
            ))

        def _list_scenes(self, *, split="all"):
            return self._scenes[1:] if split == "val" else self._scenes

        def _load_cameras(self, frame_id):
            frame = super()._load_cameras(frame_id)
            return replace(frame, cameras=(replace(frame.cameras[1], sensor_id="SINGLE_CAMERA"),),
                           annotations=AnnotationSet(frame.reference_frame_id))

    register_loader("desktop-custom-camera-test", CustomDataset)
    root = tk.Tk()
    viewer = CameraViewer(root, dataset="desktop-custom-camera-test", autoload=True)
    try:
        _wait(root, lambda: viewer.frame_data is not None and not viewer.busy and viewer.panels.get("SINGLE_CAMERA") is not None and viewer.panels["SINGLE_CAMERA"]._photo is not None)
        assert viewer.last_error is None
        assert tuple(viewer.panels) == ("SINGLE_CAMERA", "MISSING_CAMERA")
        assert viewer.grid_frame.grid_size() == (2, 1)
        assert sum(panel.winfo_width() for panel in viewer.panels.values()) > .9 * viewer.grid_frame.winfo_width()
        assert not viewer.class_vars
        assert viewer.panels["SINGLE_CAMERA"].canvas.find_withtag("camera_image")
        assert viewer.panels["MISSING_CAMERA"].canvas.find_withtag("placeholder")
        assert not any(panel.canvas.find_withtag("bounding_box") for panel in viewer.panels.values())

        viewer.split_var.set("val")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert viewer._active_split == "val"
        assert len(viewer.scenes) == 1
        assert viewer.frame_data.info.scene_id == "demo-roundabout"
        old_frame = viewer.frame_data
        viewer.split_var.set("missing")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error
        assert viewer.split_var.get() == "val"
        assert viewer.frame_data is old_frame

        viewer.dataset_var.set("demo")
        viewer._dataset_changed()
        assert viewer.split_var.get() == "all"
        viewer.load_dataset()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert len(viewer.panels) == 6
        assert len(viewer.scenes) == 2
        assert viewer.frame_data.info.scene_id == "demo-intersection"
    finally:
        viewer.close()


def test_split_changes_load_pending_source_and_preserve_active_source_on_failure(tmp_path):
    import tkinter as tk
    from visual_utils import DatasetError, register_loader
    from cameraviz.dataloader.demo import DemoDataset
    from cameraviz.ui.app import CameraViewer

    class SourceDataset(DemoDataset):
        def __init__(self, config):
            if config.root is not None and config.root.name == "missing":
                raise DatasetError("The selected dataset root is missing.")
            super().__init__()
            self.config = config
            self.loaded_root = config.root
            self.loaded_version = config.release
            self._metadata = replace(self.metadata, splits=("all", "val"))
            self.split_calls = []

        def _list_scenes(self, *, split="all"):
            if split == "val":
                self.split_calls.append(split)
                return self._scenes[1:]
            return self._scenes

    class ReplacementDataset(SourceDataset):
        pass

    register_loader("desktop-source-first", SourceDataset)
    register_loader("desktop-source-second", ReplacementDataset)
    root = tk.Tk()
    viewer = CameraViewer(root, dataset="desktop-source-first", root_path=tmp_path / "missing", version="release-a", autoload=True)
    try:
        viewer.split_var.set("val")
        _wait(root, lambda: viewer.last_error is not None and not viewer.busy)
        assert viewer.loader is None
        assert viewer.split_var.get() == "val"
        viewer.root_var.set(str(tmp_path / "first"))
        viewer.split_var.set("all")
        viewer.load_dataset()
        _wait(root, lambda: viewer.frame_data is not None and not viewer.busy)
        original = viewer.loader
        assert viewer.last_error is None

        # Changing a split after selecting another dataset loads that source.
        viewer.dataset_var.set("desktop-source-second")
        viewer._dataset_changed()
        viewer.split_var.set("val")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert isinstance(viewer.loader, ReplacementDataset)
        assert viewer.loader is not original
        assert original.split_calls == []
        assert viewer.loader.split_calls == ["val"]
        assert viewer._active_split == "val"
        assert viewer.frame_data.info.scene_id == "demo-roundabout"

        # Root/version changes also reload, even if the split is unchanged.
        replacement = viewer.loader
        viewer.root_var.set(str(tmp_path / "second"))
        viewer.version_var.set("release-b")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert viewer.loader is not replacement
        assert viewer.loader.loaded_root == tmp_path / "second"
        assert viewer.loader.loaded_version == "release-b"
        assert viewer._active_split == "val"

        # Equivalent paths and surrounding whitespace keep the same adapter.
        active = viewer.loader
        viewer.root_var.set(str(tmp_path / "second") + "/.")
        viewer.version_var.set(" release-b ")
        viewer.split_var.set("all")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert viewer.loader is active
        assert len(viewer.scenes) == 2
        active_source, active_frame = viewer._active_source, viewer.frame_data

        # A failed replacement keeps both the committed source and its frame.
        viewer.root_var.set(str(tmp_path / "missing"))
        viewer.split_var.set("val")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert "root is missing" in viewer.last_error
        assert viewer.loader is active
        assert viewer._active_source == active_source
        assert viewer.frame_data is active_frame
        assert viewer._active_split == viewer.split_var.get() == "all"

        viewer.root_var.set(str(tmp_path / "second"))
        viewer.split_var.set("val")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert viewer.loader is active
        assert viewer._active_split == "val"
    finally:
        viewer.close()


def test_shared_camera_only_provider_uncalibrated_images_and_source_cleanup():
    import threading
    import tkinter as tk
    import numpy as np
    from visual_utils import DatasetLoader, register_loader
    from visual_utils import (
        AnnotationSet, AnnotationStatus, CameraFrameSet, CameraObservation, Capability, DatasetMetadata, FrameInfo,
        Modality, SceneInfo, SensorInfo,
    )
    from cameraviz.ui.app import CameraViewer

    class CameraOnlyLoader(DatasetLoader):
        instances = []

        def __init__(self, config):
            super().__init__(config)
            self.released = False
            self.read_threads = []
            self.instances.append(self)

        @property
        def metadata(self):
            return DatasetMetadata("image-only-ui", "Image only", frozenset((Capability.CAMERAS,)),
                sensors=(SensorInfo("new-camera", Modality.CAMERAS, "A new camera"),))

        def _list_scenes(self, *, split="all"):
            return (SceneInfo("scene", "Camera scene", frame_counts={Modality.CAMERAS: 1}),)

        def _list_frames(self, scene_id, *, modality):
            return (FrameInfo("frame", scene_id, 0, None, frozenset((Modality.CAMERAS,))),)

        def _load_cameras(self, frame_id):
            self.read_threads.append(threading.get_ident())
            return CameraFrameSet(self.list_frames("scene", modality=Modality.CAMERAS)[0],
                "images", "local", (CameraObservation("new-camera", np.full((30, 40, 3), 120, dtype=np.uint8)),))

        def _close(self):
            self.released = True

    class WithheldLoader(CameraOnlyLoader):
        def _load_cameras(self, frame_id):
            frame = super()._load_cameras(frame_id)
            return replace(frame, annotations=AnnotationSet("local", AnnotationStatus.WITHHELD))

    class AvailableEmptyLoader(CameraOnlyLoader):
        @property
        def metadata(self):
            return replace(super().metadata, capabilities=frozenset((Capability.CAMERAS, Capability.BOXES_3D)))

        def _load_cameras(self, frame_id):
            frame = super()._load_cameras(frame_id)
            return replace(frame, annotations=AnnotationSet("local", AnnotationStatus.AVAILABLE))

    register_loader("native-image-only-ui", CameraOnlyLoader)
    register_loader("native-withheld-ui", WithheldLoader)
    register_loader("native-available-empty-ui", AvailableEmptyLoader)
    root = tk.Tk()
    viewer = CameraViewer(root, dataset="native-image-only-ui", autoload=True)
    active = None
    try:
        _wait(root, lambda: viewer.frame_data is not None and not viewer.busy and viewer.panels["new-camera"]._photo is not None)
        assert viewer.last_error is None
        active = viewer.loader
        assert active.read_threads and all(thread != threading.get_ident() for thread in active.read_threads)
        camera = viewer.frame_data.cameras[0]
        assert camera.intrinsics is camera.T_reference_from_camera is camera.projection_model is None
        assert viewer.panels["new-camera"].canvas.find_withtag("camera_image")
        assert not viewer.panels["new-camera"].canvas.find_withtag("bounding_box")
        assert "time unknown" in viewer.frame_label.cget("text")
        assert "Annotations not provided" in viewer.status_label.cget("text")
        assert "Annotations not provided" in viewer.panels["new-camera"].header.cget("text")

        viewer.dataset_var.set("demo")
        viewer._dataset_changed()
        viewer.load_dataset()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None and active.released

        viewer.dataset_var.set("native-image-only-ui")
        viewer._dataset_changed()
        viewer.load_dataset()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        active = viewer.loader
        for name, status, description in (
            ("native-withheld-ui", "withheld", "Ground truth withheld"),
            ("native-available-empty-ui", "available", "0 annotated objects"),
        ):
            previous = active
            viewer.dataset_var.set(name)
            viewer._dataset_changed()
            viewer.load_dataset()
            _wait(root, lambda: not viewer.busy)
            assert viewer.last_error is None and previous.released
            assert viewer.frame_data.annotations.status == status
            assert description in viewer.status_label.cget("text")
            active = viewer.loader
    finally:
        viewer.close()
    deadline = time.monotonic() + 5
    while active is not None and not active.released and time.monotonic() < deadline:
        time.sleep(.01)
    assert active is not None and active.released


def test_explicit_configuration_reaches_provider_and_keeps_sources_locked(tmp_path):
    import tkinter as tk
    import numpy as np
    from visual_utils import DatasetConfig, DatasetLoader, register_loader
    from visual_utils import (
        AnnotationSet, AnnotationStatus, Box3D, CameraFrameSet, CameraObservation,
        Capability, ClassInfo, DatasetMetadata, FrameInfo, Modality, PinholeProjection, SceneInfo, SensorInfo,
    )
    from cameraviz.ui.app import CameraViewer

    class ConfiguredLoader(DatasetLoader):
        @property
        def metadata(self):
            return DatasetMetadata("configured-ui", "Configured", frozenset((Capability.CAMERAS, Capability.BOXES_3D)),
                classes=(ClassInfo("native.car", "native.car", "Car", "Vehicles"),),
                sensors=(SensorInfo("camera", Modality.CAMERAS, "Camera"),), splits=("all", "val"))

        def _list_scenes(self, *, split="all"):
            return (SceneInfo("scene", split, frame_counts={Modality.CAMERAS: 1}),)

        def _list_frames(self, scene_id, *, modality):
            return (FrameInfo("frame", scene_id, 0, modalities=frozenset((Modality.CAMERAS,))),)

        def _load_cameras(self, frame_id):
            model = PinholeProjection(np.array([[5., 0, 5], [0, 5., 5], [0, 0, 1]]), np.eye(4))
            camera = CameraObservation("camera", np.zeros((10, 10, 3), dtype=np.uint8), projection_model=model)
            box = Box3D("car", "native.car", np.array([0., 0, 10]), np.ones(3) * 2, np.eye(3))
            return CameraFrameSet(self.list_frames("scene", modality=Modality.CAMERAS)[0],
                "images", "reference", (camera,), AnnotationSet("reference", AnnotationStatus.AVAILABLE, (box,)))

    register_loader("configured-camera-desktop", ConfiguredLoader)
    config = DatasetConfig(root=tmp_path, release="custom-release", options={"custom_option": 17})
    root = tk.Tk()
    viewer = CameraViewer(root, dataset="configured-camera-desktop", config=config, autoload=True)
    try:
        _wait(root, lambda: viewer.frame_data is not None and not viewer.busy)
        assert viewer.last_error is None
        assert viewer.loader.config is config
        assert tuple(viewer.class_vars) == ("Vehicles",)
        assert viewer.frame_data.annotations.boxes[0].class_id == "native.car"
        _wait(root, lambda: bool(viewer.panels["camera"].canvas.find_withtag("bounding_box")))
        viewer.show_classes(False)
        assert not viewer.panels["camera"].canvas.find_withtag("bounding_box")
        viewer.show_classes(True)
        assert viewer.panels["camera"].canvas.find_withtag("bounding_box")
        assert viewer.root_var.get() == str(tmp_path)
        assert viewer.version_var.get() == "custom-release"
        for control in (viewer.dataset_combo, viewer.root_entry, viewer.version_entry, viewer.browse_button):
            assert str(control.cget("state")) == "disabled"
        assert str(viewer.load_button.cget("state")) == "normal"
        assert str(viewer.split_combo.cget("state")) == "readonly"
        active = viewer.loader
        viewer.split_var.set("val")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert viewer.loader is active
        assert viewer.scenes[0].display_name == "val"
        assert str(viewer.root_entry.cget("state")) == "disabled"
    finally:
        viewer.close()
