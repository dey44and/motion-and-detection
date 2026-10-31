"""Opt-in real-window integration covering canonical dataset and display flows."""

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
    from open3d.visualization import gui
    from visual_utils import (
        AcquisitionGeometry, Capability, DatasetConfig, DatasetIOError, PointColors,
        ProcessingProvenance, register_loader, registry,
    )
    from threedviz.dataloader import DemoLoader
    from threedviz.ui.app import ViewerWindow

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

    try:
        idle()
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
        assert not np.allclose(colors_by_scheme["gray"], colors_by_scheme["viridis"])
        assert not np.allclose(colors_by_scheme["viridis"], colors_by_scheme["concerto"])
        viewer._on_distance(40)
        assert not np.allclose(viewer.renderer._cloud.colors, colors_by_scheme["concerto"])
        np.testing.assert_array_equal(viewer.renderer._cloud.points, displayed_points)
        np.testing.assert_allclose(viewer.viewport.scene.camera.get_view_matrix(), camera)
        viewer._on_distance(80)
        np.testing.assert_allclose(viewer.renderer._cloud.colors, colors_by_scheme["concerto"])
        viewer._on_point_size(7)
        viewer._on_line_width(5)
        assert viewer.renderer.settings.point_size == 7 and viewer.renderer.settings.line_width == 5
        camera = viewer.viewport.scene.camera.get_view_matrix().copy()
        axes = viewer.renderer._axes_geometry
        for _ in range(3):
            viewer.renderer.show_axes(False)
            viewer.renderer.show_axes(True)
        np.testing.assert_array_equal(camera, viewer.viewport.scene.camera.get_view_matrix())
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
