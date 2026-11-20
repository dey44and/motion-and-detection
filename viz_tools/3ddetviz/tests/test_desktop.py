"""Native detection controls compose review state beside canonical cloud data."""

from dataclasses import replace
import json
import pickle
import os
import time
import threading

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("DETVIZ_DESKTOP_TESTS") != "1",
                                reason="Set DETVIZ_DESKTOP_TESTS=1 with a desktop display.")


def test_native_canonical_review_filters_navigation_sources_and_presets(tmp_path, monkeypatch):
    from open3d.visualization import gui
    from visual_utils import DatasetConfig
    from threedviz.dataloader import DemoLoader
    from detviz.processing import FilterConfig
    from detviz.ui import app

    application = gui.Application.instance
    application.initialize()
    viewer = app.DetectionWindow(dataset="demo", autoload=True, cache_dir=tmp_path / "cache")
    def idle():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            application.run_one_tick()
            if not viewer._busy and viewer._filter_deadline is None:
                return
            time.sleep(.01)
        pytest.fail("Review loading did not finish.")
    def tick_until(predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            assert application.run_one_tick(), "Window closed unexpectedly"
            if predicate():
                return
            time.sleep(.01)
        pytest.fail("Inherited camera overlay did not follow the live view")

    def snapshot():
        camera = viewer.viewport.scene.camera
        return (camera.get_view_matrix().copy(), camera.get_projection_matrix().copy(),
                np.asarray(viewer.viewport.center_of_rotation).copy())

    def assert_snapshot(expected):
        for actual, saved in zip(snapshot(), expected):
            np.testing.assert_allclose(actual, saved, rtol=2e-6, atol=2e-5)

    try:
        idle()
        panel = viewer.camera_panel
        tick_until(lambda: panel.widget.enabled and bool(panel.view_edit.text_value))
        viewer.viewport.look_at([.5, 1., 2.], [12., -18., 9.], [.3, .1, 1.])
        tick_until(lambda: np.allclose(json.loads(panel.view_edit.text_value)["position"], [12., -18., 9.]))
        full_view = json.loads(panel.view_edit.text_value)
        full_view.update(fov=47., near=.4, far=700., pivot=[1., 2., 3.])
        panel.view_edit.text_value = json.dumps(full_view)
        panel._on_edit(panel.view_edit.text_value)
        panel.apply()
        copied_json, copied_camera = panel.view_edit.text_value, snapshot()
        assert json.loads(copied_json)["fov"] == pytest.approx(47., abs=2e-4)

        # The inherited tick must keep live sync and review debounce working
        # together, without replacing a pending full-view draft.
        draft = "{unfinished view"
        panel.view_edit.text_value = draft
        panel._on_edit(draft)
        viewer.viewport.look_at([1., 0., 0.], [4., -13., 7.], [0., 0., 1.])
        viewer.score_edit.double_value = .99
        viewer._score_edit_changed(.99)
        idle()
        assert viewer._review_frame.filtered.output_count == 0
        assert panel.view_edit.text_value == draft and panel._dirty
        assert panel.apply_button.enabled
        panel.copy_view()
        assert not panel._dirty and not panel.apply_button.enabled
        np.testing.assert_allclose(json.loads(panel.view_edit.text_value)["position"],
                                   [4., -13., 7.], atol=2e-5)
        unchanged = snapshot()
        panel.view_edit.text_value = "{"
        panel._on_edit("{")
        panel.apply()
        assert_snapshot(unchanged)
        assert panel._dirty and panel.apply_button.enabled and panel.message.text

        # Full-view transfer restores orientation, pivot and projection, not
        # only the camera position, in the inherited detection viewport.
        panel.view_edit.text_value = copied_json
        panel._on_edit(copied_json)
        panel.apply()
        assert_snapshot(copied_camera)
        assert json.loads(panel.view_edit.text_value)["fov"] == pytest.approx(47., abs=2e-4)
        viewer.reset_filters()
        idle()
        assert_snapshot(copied_camera)
        observation, review = viewer.renderer.frame, viewer._review_frame
        assert review.observation is observation
        assert review.predictions is not None and len(viewer.renderer.box_layers) == 2
        # Failed rendering must not commit a new review policy/state.
        pending = replace(review, filtered=None)
        viewer._pending_review = pending
        with monkeypatch.context() as patch:
            def fail_render(*args, **kwargs):
                raise ValueError("renderer rejected geometry")
            patch.setattr(viewer.renderer, "set_frame", fail_render)
            with pytest.raises(ValueError, match="rejected geometry"):
                viewer._render_loaded_frame(observation)
        assert viewer._review_frame is review and viewer._pending_review is pending
        viewer._pending_review = None
        camera = viewer.viewport.scene.camera.get_view_matrix().copy()
        projection = viewer.viewport.scene.camera.get_projection_matrix().copy()
        displayed_points = np.asarray(viewer.renderer._cloud.points).copy()
        layers = viewer.renderer.box_layers
        colors_by_scheme = {}
        for scheme in ("gray", "viridis", "concerto"):
            viewer._on_color_scheme(scheme, 0)
            colors_by_scheme[scheme] = np.asarray(viewer.renderer._cloud.colors).copy()
            assert viewer.renderer.settings.color_scheme == scheme
            assert viewer.color_choice.selected_text == scheme
            assert viewer.renderer.frame is observation
            assert viewer._review_frame.predictions is review.predictions
            assert viewer._review_frame.filtered is review.filtered
            np.testing.assert_array_equal(viewer.renderer._cloud.points, displayed_points)
            np.testing.assert_allclose(viewer.viewport.scene.camera.get_view_matrix(), camera)
            np.testing.assert_allclose(viewer.viewport.scene.camera.get_projection_matrix(), projection)
            assert len(viewer.renderer.box_layers) == len(layers)
            for current, original in zip(viewer.renderer.box_layers, layers):
                assert (current.name, current.color, current.visible) == (original.name, original.color, original.visible)
                assert len(current.boxes) == len(original.boxes)
                assert all(left is right for left, right in zip(current.boxes, original.boxes))
        assert not np.allclose(colors_by_scheme["gray"], colors_by_scheme["viridis"])
        assert not np.allclose(colors_by_scheme["viridis"], colors_by_scheme["concerto"])
        cloud, camera = viewer.renderer._cloud, viewer.viewport.scene.camera.get_view_matrix().copy()
        projection = viewer.viewport.scene.camera.get_projection_matrix().copy()
        viewer.apply_filters(FilterConfig(score_threshold=.99))
        idle()
        assert viewer._review_frame.filtered.output_count == 0
        assert viewer.renderer.frame is observation and viewer.renderer._cloud is cloud
        np.testing.assert_allclose(viewer.viewport.scene.camera.get_view_matrix(), camera)
        np.testing.assert_allclose(viewer.viewport.scene.camera.get_projection_matrix(), projection)
        viewer.apply_filters(FilterConfig(score_threshold=0, nms_mode="class"))
        idle()
        assert 0 < viewer._review_frame.filtered.output_count < len(review.predictions.detections)
        viewer._set_layer_visible("gt", False)
        assert not viewer._show_gt
        viewer._request_frame(1)
        idle()
        assert viewer.renderer.frame.info.frame_id != observation.info.frame_id
        assert viewer._review_frame.observation is viewer.renderer.frame
        preset = tmp_path / "preset.json"
        viewer.write_preset(preset)
        viewer.reset_filters()
        idle()
        viewer.read_preset(preset)
        idle()
        assert viewer._filter_config.nms_mode == "class"
        # A distinct configured loader uses the same canonical observation API.
        config = DatasetConfig(root=tmp_path / "configured")
        calls = []
        with monkeypatch.context() as patch:
            def create(_window, name, root, version):
                calls.append((name, _window._dataset_config))
                return DemoLoader()
            patch.setattr(app.ViewerWindow, "_create_dataset_adapter", create)
            viewer._dataset_config = config
            viewer.dataset_choice.selected_text = "nuscenes"
            viewer.prediction_path.text_value = ""
            viewer.load_dataset()
            idle()
        assert calls == [("nuscenes", config)]
        assert viewer._review_frame.predictions is None
        assert not viewer.root_edit.enabled and not viewer.dataset_choice.enabled
        # Sparse frame IDs are joined exactly; missing source reads retain the current cloud.
        current = viewer.renderer.frame
        results = tmp_path / "results.pkl"
        with results.open("wb") as stream:
            pickle.dump([{"metadata": {"token": current.info.frame_id},
                          "boxes_lidar": np.empty((0, 7)), "score": np.empty(0)}], stream)
        viewer.prediction_path.text_value = str(results)
        viewer.load_predictions()
        idle()
        assert viewer._review_frame.predictions.detections == ()
        assert viewer.renderer.frame is current
        source = viewer._source
        viewer.prediction_path.text_value = str(tmp_path / "missing.json")
        viewer.load_predictions()
        idle()
        assert viewer._source is source and viewer.renderer.frame is current
        assert "missing.json" in viewer.last_error
        viewer.window.close_dialog()
        # A source opened by an uncommitted job remains owned during shutdown.
        from detviz.predictions import InMemoryPredictionSource, PredictionError, PredictionInfo
        entered, release = threading.Event(), threading.Event()
        pending = InMemoryPredictionSource([], info=PredictionInfo("pending", "memory", "lidar"))
        def open_pending(_path):
            entered.set()
            assert release.wait(5)
            return pending
        with monkeypatch.context() as patch:
            patch.setattr(viewer, "_open_source", open_pending)
            viewer.load_predictions()
            assert entered.wait(5)
            viewer._on_close()
            release.set()
            viewer._executor.shutdown(wait=True)
        with pytest.raises(PredictionError, match="closed"):
            pending.get_frame(current.info.frame_id)
    finally:
        viewer._on_close()
        viewer.window.close()
        for _ in range(4):
            application.run_one_tick()
