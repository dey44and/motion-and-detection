"""Exercise actual Tk widgets, image overlays, navigation and load recovery."""

from dataclasses import replace
import os
import time

import pytest


pytestmark = [
    pytest.mark.desktop,
    pytest.mark.skipif(os.environ.get("CAMERA_VIZ_DESKTOP_TESTS") != "1", reason="Set CAMERA_VIZ_DESKTOP_TESTS=1 with a desktop display."),
]


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
            ("CAM_BACK_LEFT", 1, 0), ("CAM_BACK", 1, 1), ("CAM_BACK_RIGHT", 1, 2),
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
    finally:
        viewer.close()
