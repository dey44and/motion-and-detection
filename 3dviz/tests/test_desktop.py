"""Opt-in integration check using the real native window and renderer.

Run with THREEDVIZ_DESKTOP_TESTS=1 on a desktop or under xvfb-run.
"""

import os
import threading
import time
from dataclasses import replace

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("THREEDVIZ_DESKTOP_TESTS") != "1",
    reason="Set THREEDVIZ_DESKTOP_TESTS=1 with a display and OpenGL to test the native UI",
)


def test_native_loading_navigation_styles_and_error_recovery(tmp_path, monkeypatch):
    from open3d.visualization import gui

    from threedviz.dataloader import DatasetError
    from threedviz.dataloader import factory
    from threedviz.dataloader.demo import DemoAdapter
    from threedviz.processing import colorize_points
    from threedviz.ui.app import ViewerWindow

    class RGBUIFixture(DemoAdapter):
        """Stage optional frame data to test the native UI, not projection math."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepare_calls = []
            self.fail_rgb = False

        @property
        def supports_camera_rgb(self):
            return True

        @property
        def supports_point_timing(self):
            return True

        def prepare_frame(self, frame, *, camera_rgb=False, point_timing=False):
            self.prepare_calls.append((frame.info.token, camera_rgb, point_timing, threading.get_ident()))
            if camera_rgb and self.fail_rgb:
                raise DatasetError("Fixture camera images are missing")
            attributes = {}
            if camera_rgb:
                attributes["rgb_colors"] = np.tile(
                    np.array([51, 102, 204], dtype=np.uint8) / 255.0, (len(frame.points), 1)
                )
                attributes["rgb_valid_mask"] = np.arange(len(frame.points)) % 2 == 0
            if camera_rgb or point_timing:
                attributes["point_timestamps_us"] = frame.info.timestamp_us + np.linspace(
                    0, 50_000, len(frame.points)
                )
            if point_timing:
                attributes["raw_points"] = frame.points + np.array([1.5, -0.75, 0.25])
            return replace(frame, **attributes)

    monkeypatch.setattr(factory, "_REGISTRY", factory._REGISTRY.copy())
    factory.register_dataset("rgb-ui-fixture", RGBUIFixture)
    application = gui.Application.instance
    application.initialize()
    viewer = ViewerWindow(dataset="demo", autoload=True)
    main_thread = threading.get_ident()

    def wait_for_idle():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            assert application.run_one_tick(), "The window closed unexpectedly"
            if not viewer._busy:
                return
            time.sleep(0.01)
        pytest.fail("The dataset worker did not finish within 20 seconds")

    try:
        wait_for_idle()
        assert "points ·" in viewer.status.text
        assert viewer.renderer.frame.info.scene_token == "demo-urban"
        assert viewer.viewport.scene.has_geometry("points")
        assert viewer.renderer._box_names
        assert viewer._class_count_labels["car"].text == "(1)"
        old_frame = viewer.renderer.frame
        with pytest.raises(ValueError, match="finite"):
            viewer.renderer.set_frame(replace(old_frame, points=np.array([[np.nan, 0, 0]])))
        assert viewer.renderer.frame is old_frame
        assert viewer.viewport.scene.has_geometry("points")
        camera = viewer.viewport.scene.camera.get_view_matrix().copy()

        for scheme in ("gray", "jet", "turbo"):
            viewer._on_color_scheme(scheme, 0)
            colors = np.asarray(viewer.renderer._cloud.colors)
            assert colors.shape == viewer.renderer.frame.points.shape
            assert np.isfinite(colors).all()
        viewer._on_point_size(6.0)
        viewer._on_distance(40.0)
        viewer._on_line_width(5.0)
        viewer._pick_class_color("car")
        viewer._on_class_color("car", gui.Color(0.9, 0.1, 0.2))
        viewer.window.close_dialog()
        assert viewer.renderer.settings.class_colors["car"] == pytest.approx((0.9, 0.1, 0.2))
        viewer._on_class_checked("car", False)
        assert "boxes:car" not in viewer.renderer._box_names
        viewer._select_classes(False)
        assert viewer.renderer._box_names == []
        viewer._select_classes(True)
        assert "boxes:car" in viewer.renderer._box_names
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())

        viewer._request_frame(1)
        wait_for_idle()
        assert viewer.renderer.frame.info.index == 1
        assert "Frame 2 / 36" in viewer.frame_label.text
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())
        viewer._on_scene_choice("Demo: highway", 1)
        wait_for_idle()
        assert viewer.renderer.frame.info.scene_token == "demo-highway"
        assert viewer.renderer.frame.info.index == 0
        assert "boxes:construction_vehicle" in viewer.renderer._box_names

        old_frame = viewer.renderer.frame
        viewer.dataset_choice.selected_text = "nuscenes"
        viewer.root_edit.text_value = str(tmp_path / "missing-dataset")
        viewer.load_dataset()
        wait_for_idle()
        assert "Loading failed" in viewer.status.text
        assert viewer.renderer.frame is old_frame
        assert viewer.load_button.enabled
        assert viewer.scene_choice.selected_index == 1
        viewer.window.close_dialog()

        viewer.dataset_choice.selected_text = "demo"
        viewer.load_dataset()
        wait_for_idle()
        assert "points ·" in viewer.status.text
        assert viewer.renderer.frame.info.scene_token == "demo-urban"
        assert viewer.previous_button.enabled is False
        assert viewer.next_button.enabled is True

        viewer.dataset_choice.selected_text = "rgb-ui-fixture"
        viewer.load_dataset()
        wait_for_idle()
        fixture = viewer._adapter
        assert viewer.point_color_label.text == "Point colors"
        assert viewer.raw_checkbox.enabled
        assert viewer.renderer.frame.rgb_colors is None
        assert viewer.renderer.frame.raw_points is None
        assert all(not rgb and not raw for _, rgb, raw, _ in fixture.prepare_calls)
        camera = viewer.viewport.scene.camera.get_view_matrix().copy()
        before_prepare = len(fixture.prepare_calls)

        viewer.color_choice.selected_text = "Camera RGB"
        viewer._on_color_scheme("Camera RGB", 3)
        assert viewer._busy, "Camera colors must be prepared on the dataset worker"
        wait_for_idle()
        assert len(fixture.prepare_calls) == before_prepare + 1
        assert fixture.prepare_calls[-1][1:3] == (True, False)
        assert fixture.prepare_calls[-1][3] != main_thread
        assert viewer.renderer.settings.color_scheme == "camera_rgb"
        assert viewer.color_choice.selected_text == "Camera RGB"
        assert not viewer.distance_label.visible
        assert not viewer.distance_edit.visible
        assert not viewer.distance_hint.visible
        assert viewer.color_status.visible
        frame = viewer.renderer.frame
        valid = frame.rgb_valid_mask
        count = int(valid.sum())
        assert f"{count:,} / {len(frame.points):,} points" in viewer.color_status.text
        rendered_colors = np.asarray(viewer.renderer._cloud.colors)
        np.testing.assert_allclose(rendered_colors[valid], frame.rgb_colors[valid])
        np.testing.assert_allclose(
            rendered_colors[~valid],
            colorize_points(frame.points, "gray", viewer.renderer.settings.max_distance)[~valid],
        )
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())

        viewer._on_raw_points(True)
        wait_for_idle()
        frame = viewer.renderer.frame
        assert viewer.renderer.settings.use_raw_points
        assert viewer.raw_checkbox.checked
        assert viewer.raw_hint.visible
        assert frame.point_timestamps_us.shape == (len(frame.points),)
        np.testing.assert_allclose(frame.raw_points, frame.points + [1.5, -0.75, 0.25])
        np.testing.assert_allclose(np.asarray(viewer.renderer._cloud.points), frame.raw_points)
        np.testing.assert_allclose(np.asarray(viewer.renderer._cloud.colors)[valid], frame.rgb_colors[valid])
        np.testing.assert_allclose(
            np.asarray(viewer.renderer._cloud.colors)[~valid],
            colorize_points(frame.raw_points, "gray", viewer.renderer.settings.max_distance)[~valid],
        )
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())

        viewer._request_frame(1)
        wait_for_idle()
        assert viewer.renderer.frame.info.index == 1
        assert viewer.renderer.settings.color_scheme == "camera_rgb"
        assert viewer.renderer.settings.use_raw_points
        assert fixture.prepare_calls[-1][1:3] == (True, True)
        np.testing.assert_allclose(
            np.asarray(viewer.renderer._cloud.points), viewer.renderer.frame.raw_points
        )
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())

        prepared_frame = viewer.renderer.frame
        viewer._on_color_scheme("jet", 1)
        assert viewer.renderer.frame is prepared_frame
        assert viewer.renderer.frame.rgb_colors is prepared_frame.rgb_colors
        assert viewer.renderer.frame.raw_points is prepared_frame.raw_points
        assert viewer.distance_label.visible and viewer.distance_edit.visible and viewer.distance_hint.visible
        assert not viewer.color_status.visible
        np.testing.assert_allclose(
            np.asarray(viewer.renderer._cloud.colors),
            colorize_points(prepared_frame.raw_points, "jet", viewer.renderer.settings.max_distance),
        )
        viewer._on_raw_points(False)
        assert viewer.renderer.frame is prepared_frame
        assert not viewer.renderer.settings.use_raw_points
        assert not viewer.raw_hint.visible
        np.testing.assert_allclose(np.asarray(viewer.renderer._cloud.points), prepared_frame.points)

        # A fresh frame has no optional arrays, so this tests asynchronous failure
        # rollback instead of reusing the successful color preparation above.
        viewer._request_frame(2)
        wait_for_idle()
        old_frame = viewer.renderer.frame
        assert old_frame.rgb_colors is None and old_frame.raw_points is None
        old_colors = np.asarray(viewer.renderer._cloud.colors).copy()
        fixture.fail_rgb = True
        viewer.color_choice.selected_text = "Camera RGB"
        viewer._on_color_scheme("Camera RGB", 3)
        wait_for_idle()
        assert "Loading failed" in viewer.status.text
        assert viewer.renderer.frame is old_frame
        assert viewer.renderer.settings.color_scheme == "jet"
        assert viewer.color_choice.selected_text == "jet"
        assert not viewer.renderer.settings.use_raw_points
        assert viewer.distance_edit.visible
        assert viewer.load_button.enabled and viewer.color_choice.enabled
        np.testing.assert_array_equal(np.asarray(viewer.renderer._cloud.points), old_frame.points)
        np.testing.assert_array_equal(np.asarray(viewer.renderer._cloud.colors), old_colors)
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())
        viewer.window.close_dialog()

        fixture.fail_rgb = False
        viewer._on_color_scheme("Camera RGB", 3)
        wait_for_idle()
        assert "points ·" in viewer.status.text
        assert viewer.renderer.settings.color_scheme == "camera_rgb"
        assert viewer.renderer.frame.rgb_colors is not None
        viewer._on_raw_points(True)
        wait_for_idle()
        assert viewer.renderer.settings.use_raw_points

        viewer.dataset_choice.selected_text = "demo"
        viewer.load_dataset()
        wait_for_idle()
        assert viewer.renderer.settings.color_scheme == "gray"
        assert viewer.color_choice.selected_text == "gray"
        assert not viewer.renderer.settings.use_raw_points
        assert not viewer.raw_checkbox.checked and not viewer.raw_checkbox.enabled
        assert viewer.distance_label.visible and viewer.distance_edit.visible and viewer.distance_hint.visible
        assert viewer.renderer.frame.rgb_colors is None
        assert viewer.renderer.frame.raw_points is None
        viewer.renderer.reset_camera()
        images = []
        viewer.viewport.scene.scene.render_to_image(lambda image: images.append(np.asarray(image).copy()))
        deadline = time.monotonic() + 20
        while not images and time.monotonic() < deadline:
            application.run_one_tick()
            time.sleep(0.01)
        assert images, "The renderer did not produce an image"
        assert np.all(images[0][5, 5, :3] >= 254), "The viewport background must be white"
        assert (images[0][:, :, :3] < 200).any(), "The viewport must contain visible geometry"
        # Allow native layout and drawing to settle before optional inspection.
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            application.run_one_tick()
            time.sleep(0.01)
        screenshot = os.environ.get("THREEDVIZ_SCREENSHOT")
        if screenshot:
            import subprocess

            subprocess.run(["import", "-window", "root", screenshot], check=True)
    finally:
        viewer.window.close()
        while application.run_one_tick():
            pass
