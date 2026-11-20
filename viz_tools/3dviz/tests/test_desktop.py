"""Opt-in real-window integration covering canonical dataset and display flows."""

import json
import os
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("THREEDVIZ_DESKTOP_TESTS") != "1",
    reason="Set THREEDVIZ_DESKTOP_TESTS=1 with a display and OpenGL",
)


def test_native_loading_navigation_styles_and_error_recovery(tmp_path, monkeypatch):
    from open3d.visualization import gui, rendering
    from visual_utils import (
        AcquisitionGeometry, Capability, DatasetConfig, DatasetIOError, PointColors,
        ProcessingProvenance, register_loader, registry,
    )
    from threedviz.dataloader import DemoLoader
    from threedviz.ui.app import ViewerWindow
    from threedviz.ui import camera as camera_ui

    class OptionalLoader(DemoLoader):
        def __init__(self, config):
            super().__init__(config)
            self._metadata = replace(
                self.metadata,
                capabilities=self.metadata.capabilities | {Capability.RGB_COLORING, Capability.ACQUISITION_GEOMETRY},
            )
            self.requests = []
            self.fail_rgb = False

        def _prepare_point_cloud(self, frame, *, request):
            self.requests.append((request, threading.get_ident()))
            if request.rgb and self.fail_rgb:
                raise DatasetIOError("Fixture camera images are missing")
            values = {}
            if request.rgb:
                colors = np.tile([.2, .4, .8], (len(frame.points), 1))
                values["rgb"] = PointColors(colors, np.arange(len(frame.points)) % 2 == 0,
                                            ProcessingProvenance("synthetic-colors"))
            if request.acquisition_geometry:
                values["acquisition"] = AcquisitionGeometry(
                    frame.points + [1.5, -.75, .25], np.ones(len(frame.points), dtype=bool),
                    "lidar@acquisition", ProcessingProvenance("synthetic-motion"))
            return replace(frame, **values)

    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    register_loader("rgb-ui", OptionalLoader)
    app = gui.Application.instance
    app.initialize()
    viewer = ViewerWindow(dataset="demo", autoload=True)
    main_thread = threading.get_ident()

    def idle():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            assert app.run_one_tick(), "Window closed unexpectedly"
            if not viewer._busy:
                return
            time.sleep(.01)
        pytest.fail("Dataset loading timed out")

    def tick_until(predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            assert app.run_one_tick(), "Window closed unexpectedly"
            if predicate():
                return
            time.sleep(.01)
        pytest.fail("Camera overlay did not follow the live view")

    def snapshot():
        camera = viewer.viewport.scene.camera
        return (camera.get_view_matrix().copy(), camera.get_projection_matrix().copy(),
                np.asarray(viewer.viewport.center_of_rotation).copy())

    def assert_snapshot(expected):
        for actual, saved in zip(snapshot(), expected):
            np.testing.assert_allclose(actual, saved, rtol=2e-6, atol=2e-5)

    def expected_view(position, angles):
        # Independent axis-matrix composition checks the documented convention.
        x, y, z = np.radians(angles)
        rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        rotation = rz @ ry @ rx
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.T
        matrix[:3, 3] = -rotation.T @ np.asarray(position)
        return matrix, rotation

    try:
        idle()
        panel = viewer.camera_panel
        tick_until(lambda: panel.widget.enabled and bool(panel.view_edit.text_value))
        assert viewer.viewport.frame.width > 1 and viewer.viewport.frame.height > 1
        position, angles, pivot = [18., -26., 11.], [23., -17., 41.], [2.5, -1.25, 3.]
        expected, rotation = expected_view(position, angles)
        eye = np.asarray(position)
        viewer.viewport.look_at(eye - rotation[:, 2], eye, rotation[:, 1])
        viewer.viewport.center_of_rotation = np.asarray(pivot)
        viewer.viewport.scene.camera.set_projection(
            43., viewer.viewport.frame.width / viewer.viewport.frame.height,
            .35, 650., rendering.Camera.FovType.Vertical,
        )
        tick_until(lambda: np.allclose(json.loads(panel.view_edit.text_value)["position"], position))
        np.testing.assert_allclose(viewer.viewport.scene.camera.get_view_matrix(), expected,
                                   rtol=2e-6, atol=2e-5)
        saved_json = panel.view_edit.text_value
        saved_view = json.loads(saved_json)
        np.testing.assert_allclose(saved_view["rotation"], angles, atol=2e-4)
        np.testing.assert_allclose(saved_view["pivot"], pivot, atol=2e-5)
        assert saved_view["fov"] == pytest.approx(43., abs=2e-4)
        assert saved_view["near"] == pytest.approx(.35)
        assert saved_view["far"] == pytest.approx(650.)
        copied_camera = snapshot()

        # Polling must preserve a pending full-view draft while the camera moves.
        draft = "{unfinished view"
        panel.view_edit.text_value = draft
        panel._on_edit(draft)
        assert panel._dirty and panel.apply_button.enabled
        viewer.viewport.look_at([0., 0., 0.], [3., 4., 9.], [0., 0., 1.])
        for _ in range(4):
            assert app.run_one_tick()
        assert panel.view_edit.text_value == draft and panel._dirty

        # Failed Copy preserves the draft. Successful Copy refreshes it from
        # the live camera without applying that unfinished text.
        unchanged = snapshot()
        with monkeypatch.context() as patch:
            def unavailable_clipboard():
                raise RuntimeError("Clipboard unavailable for test")
            patch.setattr(camera_ui, "_clipboard", unavailable_clipboard)
            panel.copy_view()
        assert panel.view_edit.text_value == draft and panel._dirty
        assert panel.apply_button.enabled
        assert_snapshot(unchanged)
        panel.copy_view()
        assert not panel._dirty and not panel.apply_button.enabled
        np.testing.assert_allclose(json.loads(panel.view_edit.text_value)["position"], [3., 4., 9.], atol=2e-5)

        # Reject malformed JSON and invalid view values before touching pose or lens.
        invalid_views = [
            "{", "[]", json.dumps({**saved_view, "position": [1, 2]}),
            json.dumps({**saved_view, "rotation": [0, float("nan"), 0]}),
            json.dumps({**saved_view, "pivot": [0, 0, float("inf")]}),
            json.dumps({**saved_view, "fov": 0}), json.dumps({**saved_view, "fov": 180}),
            json.dumps({**saved_view, "near": 2, "far": 1}),
            json.dumps({key: value for key, value in saved_view.items() if key != "near"}),
        ]
        for text in invalid_views:
            unchanged = snapshot()
            panel.view_edit.text_value = text
            panel._on_edit(text)
            panel.apply()
            assert_snapshot(unchanged)
            assert panel._dirty and panel.apply_button.enabled and panel.message.text

        # A valid full-view edit updates pose, pivot and lens as one operation.
        edited = {**saved_view, "position": [8.5, -14, 6.25], "rotation": [11, -29, 67],
                  "pivot": [1, 2, 3], "fov": 55, "near": .2, "far": 900}
        panel.view_edit.text_value = json.dumps(edited)
        panel._on_edit(panel.view_edit.text_value)
        panel.apply()
        assert not panel._dirty and not panel.apply_button.enabled
        expected, _ = expected_view([8.5, -14, 6.25], [11, -29, 67])
        np.testing.assert_allclose(viewer.viewport.scene.camera.get_view_matrix(), expected,
                                   rtol=2e-6, atol=2e-5)
        applied = json.loads(panel.view_edit.text_value)
        assert applied["fov"] == pytest.approx(55., abs=2e-4)
        np.testing.assert_allclose(applied["pivot"], [1, 2, 3], atol=2e-5)
        panel.view_edit.text_value = saved_json
        panel._on_edit(saved_json)
        panel.apply()
        assert_snapshot(copied_camera)
        assert json.loads(panel.view_edit.text_value)["fov"] == pytest.approx(43., abs=2e-4)

        # Copy/Paste use the system clipboard, with Paste waiting for Apply.
        panel.copy_view()
        viewer.viewport.look_at([0., 0., 0.], [-7., 12., 5.], [0., 0., 1.])
        before_paste = snapshot()
        panel.paste_view()
        assert panel._dirty and panel.apply_button.enabled
        assert_snapshot(before_paste)
        panel.apply()
        assert_snapshot(copied_camera)

        # Reset also reaches the overlay through the shared live callback.
        viewer.renderer.reset_camera()
        tick_until(lambda: abs(json.loads(panel.view_edit.text_value)["fov"] - 60.) < 2e-4)
        np.testing.assert_allclose(json.loads(panel.view_edit.text_value)["position"],
                                   viewer.viewport.scene.camera.get_model_matrix()[:3, 3], atol=2e-5)
        assert viewer.renderer.frame.info.scene_id == "demo-urban"
        assert viewer.viewport.scene.has_geometry("points")
        assert viewer._class_count_labels["car"].text == "(1)"
        previous = viewer.renderer.frame
        viewer._request_frame(1)
        idle()
        assert viewer.renderer.frame.info.index == 1
        viewer._on_scene_choice("Demo: highway", 1)
        idle()
        assert viewer.renderer.frame.info.scene_id == "demo-highway"
        assert previous.info.scene_id == "demo-urban"
        viewer._on_dataset_split("val", 0)
        idle()
        assert viewer._dataset_split == "val"
        camera = viewer.viewport.scene.camera.get_view_matrix().copy()
        projection = viewer.viewport.scene.camera.get_projection_matrix().copy()
        displayed_points = np.asarray(viewer.renderer._cloud.points).copy()
        colors_by_scheme = {}
        for scheme in ("gray", "viridis", "concerto"):
            viewer._on_color_scheme(scheme, 0)
            colors_by_scheme[scheme] = np.asarray(viewer.renderer._cloud.colors).copy()
            assert viewer.renderer.settings.color_scheme == scheme
            assert viewer.color_choice.selected_text == scheme
            assert colors_by_scheme[scheme].shape == viewer.renderer.frame.points.shape
            np.testing.assert_array_equal(viewer.renderer._cloud.points, displayed_points)
            np.testing.assert_allclose(viewer.viewport.scene.camera.get_view_matrix(), camera)
            np.testing.assert_allclose(viewer.viewport.scene.camera.get_projection_matrix(), projection)
        assert not np.allclose(colors_by_scheme["gray"], colors_by_scheme["viridis"])
        assert not np.allclose(colors_by_scheme["viridis"], colors_by_scheme["concerto"])
        viewer._on_distance(40)
        assert not np.allclose(viewer.renderer._cloud.colors, colors_by_scheme["concerto"])
        np.testing.assert_array_equal(viewer.renderer._cloud.points, displayed_points)
        np.testing.assert_allclose(viewer.viewport.scene.camera.get_view_matrix(), camera)
        np.testing.assert_allclose(viewer.viewport.scene.camera.get_projection_matrix(), projection)
        viewer._on_distance(80)
        np.testing.assert_allclose(viewer.renderer._cloud.colors, colors_by_scheme["concerto"])
        viewer._on_point_size(7)
        viewer._on_line_width(5)
        assert viewer.renderer.settings.point_size == 7 and viewer.renderer.settings.line_width == 5
        camera = viewer.viewport.scene.camera.get_view_matrix().copy()
        projection = viewer.viewport.scene.camera.get_projection_matrix().copy()
        axes = viewer.renderer._axes_geometry
        for _ in range(3):
            viewer.renderer.show_axes(False)
            viewer.renderer.show_axes(True)
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())
        np.testing.assert_array_equal(projection, viewer.viewport.scene.camera.get_projection_matrix())
        np.testing.assert_allclose(np.asarray(axes.vertices).max(axis=0), [3, 3, 3])
        viewer._on_class_checked("car", False)
        assert "boxes:car" not in viewer.renderer._box_names
        viewer._on_class_checked("car", True)
        assert "boxes:car" in viewer.renderer._box_names

        previous_loader = viewer._adapter
        viewer.dataset_choice.selected_text = "rgb-ui"
        viewer._on_dataset_choice("rgb-ui", 0)
        viewer.load_dataset()
        idle()
        loader = viewer._adapter
        assert previous_loader._closed
        assert viewer.raw_checkbox.enabled
        viewer._on_color_scheme("Camera RGB", 0)
        idle()
        colored = viewer.renderer.frame
        assert colored.rgb is not None and colored.acquisition is None
        assert "Camera RGB:" in viewer.color_status.text
        assert all(thread != main_thread for _, thread in loader.requests)
        viewer._on_raw_points(True)
        idle()
        prepared = viewer.renderer.frame
        assert prepared.rgb is colored.rgb and prepared.acquisition is not None
        assert prepared.timing is None
        np.testing.assert_allclose(viewer.renderer._cloud.points, prepared.acquisition.points)
        viewer._on_raw_points(False)
        np.testing.assert_allclose(viewer.renderer._cloud.points, prepared.points)
        viewer._on_color_scheme("gray", 0)
        viewer._request_frame(1)
        idle()
        loader.fail_rgb = True
        prior = viewer.renderer.frame
        viewer._on_color_scheme("Camera RGB", 0)
        idle()
        assert viewer.renderer.frame is prior
        assert viewer.renderer.settings.color_scheme == "gray"
        assert "Loading failed" in viewer.status.text
        viewer.window.close_dialog()

        config = DatasetConfig(root=tmp_path)
        viewer._dataset_config = config
        viewer._on_dataset_choice("rgb-ui", 0)
        viewer._set_busy(False)
        assert not viewer.dataset_choice.enabled and not viewer.root_edit.enabled
        assert not viewer.browse_button.enabled and not viewer.version_edit.enabled
        assert viewer.dataset_split_choice.enabled
        with viewer._create_dataset_adapter("rgb-ui", None, "v1.0-mini") as configured:
            assert configured.config is config
    finally:
        viewer.window.close()
        while app.run_one_tick():
            pass
    assert loader._closed
