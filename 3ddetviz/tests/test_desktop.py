"""Native rendering and controller checks; run separately from 3dviz's GUI test."""

from dataclasses import replace
import json
import os
import time

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(os.environ.get("DETVIZ_DESKTOP_TESTS") != "1", reason="Set DETVIZ_DESKTOP_TESTS=1 with a desktop display.")


def test_native_review_filters_layers_navigation_and_recovery(tmp_path, monkeypatch):
    from open3d.visualization import gui
    from threedviz.dataloader import factory
    from threedviz.dataloader.demo import DemoAdapter
    from detviz.processing import FilterConfig
    from detviz.ui.app import DetectionWindow

    class FixtureAdapter(DemoAdapter):
        @property
        def supports_camera_rgb(self):
            return True
        @property
        def supports_point_timing(self):
            return True
        def prepare_frame(self, frame, *, camera_rgb=False, point_timing=False):
            attributes = {}
            if camera_rgb:
                attributes.update(rgb_colors=np.full(frame.points.shape, .7), rgb_valid_mask=np.ones(len(frame.points), dtype=bool))
            if point_timing:
                attributes.update(raw_points=frame.points + [0.1, 0.2, 0], point_timestamps_us=np.full(len(frame.points), frame.info.timestamp_us))
            return replace(frame, **attributes)

    monkeypatch.setattr(factory, "_REGISTRY", factory._REGISTRY.copy())
    factory.register_dataset("review-ui-fixture", FixtureAdapter)
    application = gui.Application.instance
    application.initialize()
    viewer = DetectionWindow(dataset="demo", autoload=True, cache_dir=tmp_path / "cache")
    def wait_idle():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            assert application.run_one_tick()
            if not viewer._busy and viewer._filter_deadline is None:
                return
            time.sleep(.01)
        pytest.fail("Review window worker did not finish.")
    try:
        wait_idle()
        assert viewer.renderer.frame is not None
        assert len(viewer.renderer.box_layers) == 2
        assert not viewer._class_color_buttons
        frame = viewer.renderer.frame
        gt = frame.boxes
        cloud = viewer.renderer._cloud
        camera = viewer.viewport.scene.camera.get_view_matrix().copy()
        axes = viewer.renderer._axes_geometry
        axes_vertices = np.asarray(axes.vertices).copy()
        def assert_axes_unchanged():
            assert viewer.renderer._axes_geometry is axes
            np.testing.assert_array_equal(np.asarray(axes.vertices), axes_vertices)
            # Each positive axis ends three metres from the sensor origin.
            # The origin sphere/arrow radii also extend slightly below zero.
            np.testing.assert_allclose(axes_vertices.max(axis=0), [3, 3, 3])
            assert np.max(np.abs(axes_vertices)) == pytest.approx(3)
        def toggle_axes_repeatedly():
            current_frame = viewer.renderer.frame
            current_cloud = viewer.renderer._cloud
            current_camera = viewer.viewport.scene.camera.get_view_matrix().copy()
            for visible in (False, True, False, True):
                viewer.renderer.show_axes(visible)
                assert viewer.renderer.axes_visible is visible
                assert viewer.viewport.scene.geometry_is_visible("coordinate_axes") is visible
                assert_axes_unchanged()
                assert viewer.renderer.frame is current_frame
                assert viewer.renderer._cloud is current_cloud
                np.testing.assert_array_equal(current_camera, viewer.viewport.scene.camera.get_view_matrix())
        toggle_axes_repeatedly()
        original_count = frame.filtered.output_count
        assert original_count > len(gt)
        # Every class shares its layer color, with no class palette applied.
        for name, geometry, _ in viewer.renderer._box_geometries(frame, viewer.renderer.box_layers):
            expected = viewer._gt_color if name.startswith("layer:gt:") else viewer._detection_color
            np.testing.assert_allclose(np.asarray(geometry.colors), np.tile(expected, (len(geometry.lines), 1)))

        viewer.apply_filters(FilterConfig(score_threshold=0, nms_mode="class"))
        wait_idle()
        assert viewer.renderer.frame.filtered.output_count < original_count
        assert viewer.renderer.frame.boxes is gt
        assert viewer.renderer._cloud is cloud
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())
        toggle_axes_repeatedly()
        viewer.apply_filters(FilterConfig(score_threshold=.85))
        wait_idle()
        assert all(item.score >= .85 for item in viewer.renderer.frame.filtered.detections)
        viewer.reset_filters()
        wait_idle()
        assert viewer.renderer.frame.filtered.output_count == original_count
        assert viewer.renderer._cloud is cloud

        viewer._set_layer_visible("gt", False)
        assert not any(name.startswith("layer:gt:") for name in viewer.renderer._box_names)
        assert any(name.startswith("layer:detections:") for name in viewer.renderer._box_names)
        viewer._set_layer_visible("gt", True)
        viewer._select_classes(False)
        assert not viewer.renderer._box_names
        viewer._select_classes(True)
        viewer._on_layer_color("gt", gui.Color(.2, .4, .6))
        assert viewer.renderer.box_layers[0].color == pytest.approx((.2, .4, .6))
        viewer._on_line_width(4)
        assert viewer.renderer.settings.line_width == 4
        viewer.score_edit.double_value = .62
        viewer._score_edit_changed(.62)
        wait_idle()
        preset = tmp_path / "filters.json"
        viewer.write_preset(preset)
        assert json.loads(preset.read_text())["filters"]["score_threshold"] == pytest.approx(.62)
        viewer.reset_filters()
        wait_idle()
        viewer.read_preset(preset)
        wait_idle()
        assert viewer._filter_config.score_threshold == pytest.approx(.62)
        invalid = tmp_path / "invalid.json"
        invalid.write_text('{"schema_version":1,"filters":{"score_threshold":2}}')
        old_config = viewer._filter_config
        with pytest.raises(ValueError):
            viewer.read_preset(invalid)
        assert viewer._filter_config is old_config
        # One-sided bounds, class overrides and zero limits must survive the
        # UI/preset round trip without silently adding bounds or uncapping.
        special = tmp_path / "special.json"
        special.write_text(json.dumps({"schema_version": 1, "filters": {
            "score_threshold": 0, "min_distance": 2, "max_height": 3,
            "max_detections": 0, "per_class_scores": {"pedestrian": .4}}}))
        viewer.read_preset(special)
        wait_idle()
        assert viewer._filter_config.min_height is None
        assert viewer._filter_config.max_height == 3
        assert viewer._filter_config.max_distance is None
        assert viewer._filter_config.max_detections == 0
        assert viewer._filter_config.per_class_scores["pedestrian"] == .4
        assert viewer.renderer.frame.filtered.output_count == 0
        viewer.reset_filters()
        wait_idle()

        viewer.renderer.show_axes(False)
        viewer._request_frame(1)
        wait_idle()
        assert viewer.renderer.frame.info.index == 1
        assert not viewer.renderer.axes_visible
        assert not viewer.viewport.scene.geometry_is_visible("coordinate_axes")
        assert_axes_unchanged()
        toggle_axes_repeatedly()
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())
        # Load real-format JSON against a custom dataset, including an empty
        # frame followed by an absent frame. No list-order joins are allowed.
        fixture = FixtureAdapter()
        frames = fixture.list_frames(fixture.list_scenes()[0].token)
        predictions = tmp_path / "results.json"
        predictions.write_text(json.dumps({"results": {
            frames[0].token: [{"sample_token": frames[0].token, "translation": [10, 0, 0],
                "size": [2, 4, 2], "rotation": [1, 0, 0, 0], "velocity": [0, 0],
                "detection_name": "car", "detection_score": .9}], frames[1].token: []}}))
        viewer.dataset_choice.selected_text = "review-ui-fixture"
        viewer.prediction_path.text_value = str(predictions)
        viewer.load_dataset()
        wait_idle()
        assert viewer.renderer.frame.filtered.loaded_count == 1
        assert_axes_unchanged()
        assert "2 / 36" in viewer.coverage_label.text
        viewer._on_color_scheme("Camera RGB", 3)
        wait_idle()
        assert viewer.renderer.settings.color_scheme == "camera_rgb"
        assert viewer.renderer.frame.filtered.loaded_count == 1
        toggle_axes_repeatedly()
        viewer.renderer.show_axes(False)
        viewer._on_raw_points(True)
        wait_idle()
        assert viewer.renderer.settings.use_raw_points
        np.testing.assert_allclose(np.asarray(viewer.renderer._cloud.points), viewer.renderer.frame.raw_points)
        assert len(viewer.renderer.box_layers) == 2
        assert not viewer.renderer.axes_visible
        assert not viewer.viewport.scene.geometry_is_visible("coordinate_axes")
        assert_axes_unchanged()
        toggle_axes_repeatedly()
        viewer._request_frame(1)
        wait_idle()
        assert viewer.renderer.frame.predictions is not None
        assert viewer.renderer.frame.filtered.loaded_count == 0
        viewer._request_frame(2)
        wait_idle()
        assert viewer.renderer.frame.predictions is None
        assert "unavailable" in viewer.filter_counts.text
        assert not any(name.startswith("layer:detections:") for name in viewer.renderer._box_names)

        viewer._request_frame(0)
        wait_idle()
        old_frame = viewer.renderer.frame
        old_cloud = viewer.renderer._cloud
        viewer.score_edit.double_value = .95
        viewer._score_edit_changed(.95)
        viewer.prediction_path.text_value = str(tmp_path / "missing.json")
        viewer.load_predictions()
        wait_idle()
        assert viewer.renderer.frame.info.token == old_frame.info.token
        assert viewer.renderer._cloud is old_cloud
        assert "missing.json" in viewer.last_error
        assert viewer.renderer.frame.filtered.output_count == 0
        assert viewer._filter_config.score_threshold == pytest.approx(.95)
        assert viewer.prediction_load.enabled
        viewer.window.close_dialog()
        viewer.prediction_path.text_value = str(predictions)
        viewer.load_predictions()
        wait_idle()
        assert viewer.renderer.frame.info.token == old_frame.info.token
        assert viewer.renderer.frame.predictions is not None
        # A newly opened adapter owns resources even if coordinate binding
        # fails before it becomes the active source.
        from types import SimpleNamespace
        from detviz.predictions import PredictionFrame
        class BadSource:
            info = SimpleNamespace(coordinate_frame="world")
            closed = False
            def get_frame(self, _token):
                return PredictionFrame("different-sample", ())
            def close(self):
                self.closed = True
        source = BadSource()
        with monkeypatch.context() as patch:
            patch.setattr(viewer, "_open_source", lambda _path: source)
            viewer.load_predictions()
            wait_idle()
        assert source.closed
        assert "tokens do not match" in viewer.last_error
        assert viewer.renderer.frame.info.token == old_frame.info.token
        viewer.window.close_dialog()
        # Unknown-frame error recovery leaves navigation and split usable.
        viewer._on_split("test", 0)
        wait_idle()
        assert viewer.split_choice.selected_text == "val"
        viewer.window.close_dialog()
    finally:
        viewer._on_close()
        viewer.window.close()
        for _ in range(4):
            application.run_one_tick()
