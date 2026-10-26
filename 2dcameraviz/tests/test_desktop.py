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
        box_class = next(box.class_name for box in panel.boxes if box.class_name in panel.visible_classes)
        monkeypatch.setattr("cameraviz.ui.app.colorchooser.askcolor", lambda *args, **kwargs: ((255, 0, 255), "#ff00ff"))
        viewer.choose_color(box_class)
        assert viewer.class_colors[box_class] == "#ff00ff"
        assert panel.canvas.itemcget(panel.canvas.find_withtag("bounding_box")[0], "outline") == "#ff00ff"

        old_token = viewer.frame_data.info.token
        viewer.step_frame(1)
        _wait(root, lambda: not viewer.busy and viewer.frame_index == 1)
        assert viewer.frame_data.info.token != old_token
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
        load_frame = viewer.adapter.load_frame
        def fail(_token):
            raise FileNotFoundError("Missing camera image: recovery-test.jpg")
        viewer.adapter.load_frame = fail
        viewer.step_frame(1)
        _wait(root, lambda: not viewer.busy)
        assert "recovery-test.jpg" in viewer.last_error
        assert viewer.frame_data is old_frame
        assert viewer.frame_index == 1
        assert all(panel.camera is old_images[name] for name, panel in viewer.panels.items())
        assert str(viewer.next_button.cget("state")) == "normal"
        viewer.adapter.load_frame = load_frame
        viewer.step_frame(1)
        _wait(root, lambda: not viewer.busy and viewer.frame_index == 2)
        assert viewer.last_error is None

        viewer.scene_combo.current(1)
        viewer._select_scene()
        _wait(root, lambda: not viewer.busy and viewer.scene_index == 1)
        assert viewer.frame_index == 0
        assert viewer.frame_data.info.scene_token == "demo-roundabout"
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
        assert viewer.frame_data.info.scene_token == "demo-roundabout"
        assert str(viewer.scene_combo.cget("state")) == "readonly"
    finally:
        viewer.close()


def test_custom_layout_missing_camera_and_annotation_free_dataset():
    import tkinter as tk
    from cameraviz.dataloader import CameraInfo, DemoAdapter, register_dataset
    from cameraviz.ui.app import CameraViewer

    present = CameraInfo("SINGLE_CAMERA", "Custom view", 0, 0)
    missing = CameraInfo("MISSING_CAMERA", "Optional view", 0, 1)
    class Adapter(DemoAdapter):
        @property
        def available_splits(self):
            return ("all", "val")
        def list_scenes_for_split(self, split="all"):
            if split == "val":
                return self.list_scenes()[1:]
            return super().list_scenes_for_split(split)
        @property
        def camera_layout(self):
            return (present, missing)
        @property
        def class_names(self):
            return ()
        def load_frame(self, token):
            frame = super().load_frame(token)
            return replace(frame, cameras=(replace(frame.cameras[1], info=present),), boxes=())

    register_dataset("desktop-custom-camera-test", Adapter)
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
        assert viewer.frame_data.info.scene_token == "demo-roundabout"
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
        assert viewer.frame_data.info.scene_token == "demo-intersection"
    finally:
        viewer.close()


def test_split_changes_load_pending_source_and_preserve_active_source_on_failure(tmp_path):
    import tkinter as tk
    from cameraviz.dataloader import DatasetError, DemoAdapter, register_dataset
    from cameraviz.ui.app import CameraViewer

    class Adapter(DemoAdapter):
        def __init__(self, *, root=None, version="v1.0-mini"):
            if root is not None and root.name == "missing":
                raise DatasetError("The selected dataset root is missing.")
            super().__init__()
            self.loaded_root = root
            self.loaded_version = version
            self.split_calls = []

        @property
        def available_splits(self):
            return ("all", "val")

        def list_scenes_for_split(self, split="all"):
            self.split_calls.append(split)
            if split == "val":
                return self.list_scenes()[1:]
            return super().list_scenes_for_split(split)

    class ReplacementAdapter(Adapter):
        pass

    register_dataset("desktop-source-first", Adapter)
    register_dataset("desktop-source-second", ReplacementAdapter)
    root = tk.Tk()
    viewer = CameraViewer(root, dataset="desktop-source-first", root_path=tmp_path / "missing", version="release-a", autoload=True)
    try:
        viewer.split_var.set("val")
        _wait(root, lambda: viewer.last_error is not None and not viewer.busy)
        assert viewer.adapter is None
        assert viewer.split_var.get() == "val"
        viewer.root_var.set(str(tmp_path / "first"))
        viewer.split_var.set("all")
        viewer.load_dataset()
        _wait(root, lambda: viewer.frame_data is not None and not viewer.busy)
        original = viewer.adapter
        assert viewer.last_error is None

        # Changing a split after selecting another dataset loads that source.
        viewer.dataset_var.set("desktop-source-second")
        viewer._dataset_changed()
        viewer.split_var.set("val")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert isinstance(viewer.adapter, ReplacementAdapter)
        assert viewer.adapter is not original
        assert original.split_calls == []
        assert viewer.adapter.split_calls == ["val"]
        assert viewer._active_split == "val"
        assert viewer.frame_data.info.scene_token == "demo-roundabout"

        # Root/version changes also reload, even if the split is unchanged.
        replacement = viewer.adapter
        viewer.root_var.set(str(tmp_path / "second"))
        viewer.version_var.set("release-b")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert viewer.adapter is not replacement
        assert viewer.adapter.loaded_root == tmp_path / "second"
        assert viewer.adapter.loaded_version == "release-b"
        assert viewer._active_split == "val"

        # Equivalent paths and surrounding whitespace keep the same adapter.
        active = viewer.adapter
        viewer.root_var.set(str(tmp_path / "second") + "/.")
        viewer.version_var.set(" release-b ")
        viewer.split_var.set("all")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert viewer.adapter is active
        assert len(viewer.scenes) == 2
        active_source, active_frame = viewer._active_source, viewer.frame_data

        # A failed replacement keeps both the committed source and its frame.
        viewer.root_var.set(str(tmp_path / "missing"))
        viewer.split_var.set("val")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert "root is missing" in viewer.last_error
        assert viewer.adapter is active
        assert viewer._active_source == active_source
        assert viewer.frame_data is active_frame
        assert viewer._active_split == viewer.split_var.get() == "all"

        viewer.root_var.set(str(tmp_path / "second"))
        viewer.split_var.set("val")
        viewer._split_changed()
        _wait(root, lambda: not viewer.busy)
        assert viewer.last_error is None
        assert viewer.adapter is active
        assert viewer._active_split == "val"
    finally:
        viewer.close()


def test_shared_camera_only_provider_uncalibrated_images_and_source_cleanup():
    import threading
    import tkinter as tk
    import numpy as np
    from dataset_utils import DatasetLoader, register_loader
    from dataset_utils.contracts.models import (
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

        def list_scenes(self, *, split="all"):
            return (SceneInfo("scene", "Camera scene", frame_counts={Modality.CAMERAS: 1}),)

        def list_frames(self, scene_id, *, modality):
            return (FrameInfo("frame", scene_id, 0, None, frozenset((Modality.CAMERAS,))),)

        def load_cameras(self, frame_id):
            self.read_threads.append(threading.get_ident())
            return CameraFrameSet(self.list_frames("scene", modality=Modality.CAMERAS)[0],
                "images", "local", (CameraObservation("new-camera", np.full((30, 40, 3), 120, dtype=np.uint8)),))

        def _close(self):
            self.released = True

    class WithheldLoader(CameraOnlyLoader):
        def load_cameras(self, frame_id):
            frame = super().load_cameras(frame_id)
            return replace(frame, annotations=AnnotationSet("local", AnnotationStatus.WITHHELD))

    class AvailableEmptyLoader(CameraOnlyLoader):
        @property
        def metadata(self):
            return replace(super().metadata, capabilities=frozenset((Capability.CAMERAS, Capability.BOXES_3D)))

        def load_cameras(self, frame_id):
            frame = super().load_cameras(frame_id)
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
        active = viewer.adapter.loader
        assert active.read_threads and all(thread != threading.get_ident() for thread in active.read_threads)
        camera = viewer.frame_data.cameras[0]
        assert camera.intrinsics is camera.sensor_to_world is camera.projection_model is None
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
        active = viewer.adapter.loader
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
            assert viewer.frame_data.annotation_status == status
            assert description in viewer.status_label.cget("text")
            active = viewer.adapter.loader
    finally:
        viewer.close()
    deadline = time.monotonic() + 5
    while active is not None and not active.released and time.monotonic() < deadline:
        time.sleep(.01)
    assert active is not None and active.released
