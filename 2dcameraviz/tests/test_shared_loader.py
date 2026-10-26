"""A single backend registration feeds both generic frontend bridges."""

from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pytest

from dataset_utils import DatasetLoader, register_loader
from dataset_utils.contracts.models import (
    AcquisitionGeometry, AnnotationSet, AnnotationStatus, Box3D, CameraFrameSet,
    CameraObservation, Capability, ClassInfo, DatasetMetadata, FrameInfo,
    Modality, PointCloudFrame, PointColors, ProcessingProvenance, SceneInfo, SensorInfo,
)
from dataset_utils.contracts.projection import PinholeProjection
from dataset_utils.errors import UnsupportedCapabilityError


class PluginLoader(DatasetLoader):
    """No viewer imports, native file types, or application registrations."""

    def __init__(self, config):
        super().__init__(config)
        self.requests = []
        self.released = False
        self._metadata = DatasetMetadata(
            dataset_id="independent-plugin:v1", display_name="Independent plugin",
            capabilities=frozenset((Capability.POINT_CLOUD, Capability.CAMERAS,
                                    Capability.BOXES_3D, Capability.RGB_COLORING,
                                    Capability.ACQUISITION_GEOMETRY)),
            classes=(ClassInfo("native.vehicle", "native.vehicle", "Vehicle", "vehicles"),),
            sensors=(SensorInfo("unusual/front", Modality.CAMERAS, "Forward", "front"),
                     SensorInfo("unusual/unknown", Modality.CAMERAS, "Other")),
            feature_coverage={Capability.ACQUISITION_GEOMETRY: "full"},
        )

    @property
    def metadata(self):
        return self._metadata

    def list_scenes(self, *, split="all"):
        return (SceneInfo("scene", "A plugin scene", frame_counts={Modality.POINT_CLOUD: 1, Modality.CAMERAS: 1}),)

    def list_frames(self, scene_id, *, modality):
        return (FrameInfo("opaque/frame", scene_id, 0, None, frozenset((Modality.POINT_CLOUD, Modality.CAMERAS))),)

    def _annotations(self):
        box = Box3D("object", "native.vehicle", np.array([0., 0, 10]), np.array([2., 2, 2]), np.eye(3))
        return AnnotationSet("local", AnnotationStatus.AVAILABLE, (box,))

    def load_point_cloud(self, frame_id):
        return PointCloudFrame(self.list_frames("scene", modality=Modality.POINT_CLOUD)[0],
                               "source", "local", np.array([[0., 0, 10], [1., 0, 10]]), self._annotations())

    def load_cameras(self, frame_id):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        intrinsics = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1.]])
        camera = CameraObservation("unusual/front", image,
            projection_model=PinholeProjection(intrinsics, np.eye(4)))
        # Calibration fields without a projection model must not imply pinhole.
        uncalibrated = CameraObservation("unusual/unknown", image, intrinsics=intrinsics,
                                        T_reference_from_camera=np.eye(4))
        return CameraFrameSet(self.list_frames("scene", modality=Modality.CAMERAS)[0],
                              "source", "local", (camera, uncalibrated), self._annotations())

    def prepare_point_cloud(self, frame, *, request):
        self.requests.append(request)
        provenance = ProcessingProvenance("plugin.native")
        values = {}
        if request.rgb:
            values["rgb"] = PointColors(np.zeros((2, 3)), np.array([True, False]), provenance)
        if request.acquisition_geometry:
            values["acquisition"] = AcquisitionGeometry(frame.points - [1., 0, 0], np.ones(2, dtype=bool), "sensor@acquisition", provenance)
        return replace(frame, **values)

    def _close(self):
        self.released = True


def test_one_shared_registration_opens_generic_camera_and_point_views():
    from cameraviz.dataloader import available_datasets, create_dataset
    from cameraviz.dataloader.shared import SharedCameraAdapter
    from cameraviz.processing import project_boxes

    name = "shared-camera-and-point-plugin"
    register_loader(name, PluginLoader)
    assert name in available_datasets()
    camera = create_dataset(name)
    try:
        assert isinstance(camera, SharedCameraAdapter)
        assert camera.class_names == ("vehicles",)
        assert [info.channel for info in camera.camera_layout] == ["unusual/front", "unusual/unknown"]
        frame = camera.load_frame(camera.list_frames("scene")[0].token)
        assert frame.info.timestamp_us is None
        assert frame.cameras[0].sensor_to_world is None
        assert frame.cameras[0].intrinsics is None
        assert len(project_boxes(frame)["unusual/front"]) == 1
        assert project_boxes(frame)["unusual/unknown"] == ()
        assert frame.boxes[0].class_name == "vehicles"
        # The same registration is discovered by the separately installed 3D app.
        if find_spec("threedviz") is not None:
            from threedviz.dataloader import available_datasets as point_datasets, create_dataset as create_points
            assert name in point_datasets()
            point = create_points(name)
            try:
                assert point.supports_acquisition_geometry and not point.supports_point_timing
                baseline = point.load_frame(point.list_frames("scene")[0].token)
                prepared = point.prepare_frame(baseline, camera_rgb=True, point_timing=True)
                assert prepared.point_timestamps_us is None
                np.testing.assert_array_equal(prepared.raw_points, baseline.points - [1., 0, 0])
                np.testing.assert_array_equal(prepared.rgb_valid_mask, [True, False])
                np.testing.assert_array_equal(prepared.rgb_colors[0], [0., 0, 0])
                assert point.loader.requests[-1].acquisition_geometry
                assert not point.loader.requests[-1].timestamps
                assert prepared.points is baseline.points
            finally:
                point.close()
            assert point.loader.released
    finally:
        camera.close()
    assert camera.loader.released


def test_camera_only_loader_is_discoverable_but_rejected_by_point_bridge():
    class CameraOnly(PluginLoader):
        def __init__(self, config):
            super().__init__(config)
            self._metadata = replace(self._metadata, capabilities=frozenset((Capability.CAMERAS, Capability.BOXES_3D)), feature_coverage={})

    register_loader("shared-camera-only-plugin", CameraOnly)
    from cameraviz.dataloader import create_dataset as camera_dataset
    camera = camera_dataset("shared-camera-only-plugin")
    try:
        assert len(camera.load_frame("opaque/frame").cameras) == 2
    finally:
        camera.close()
    if find_spec("threedviz") is not None:
        from threedviz.dataloader import create_dataset
        with pytest.raises(UnsupportedCapabilityError, match="point-cloud observations"):
            create_dataset("shared-camera-only-plugin")


def test_point_only_loader_is_rejected_by_camera_bridge():
    from cameraviz.dataloader import create_dataset

    class PointOnly(PluginLoader):
        def __init__(self, config):
            super().__init__(config)
            self._metadata = replace(self._metadata,
                capabilities=self._metadata.capabilities.difference((Capability.CAMERAS,)), sensors=())

    register_loader("shared-point-only-plugin", PointOnly)
    with pytest.raises(UnsupportedCapabilityError, match="camera observations"):
        create_dataset("shared-point-only-plugin")


def test_mixed_catalog_hides_only_scenes_known_empty_for_the_active_modality():
    from cameraviz.dataloader import create_dataset

    class MixedCatalog(PluginLoader):
        def list_scenes(self, *, split="all"):
            return (
                SceneInfo("camera", "Camera", frame_counts={Modality.POINT_CLOUD: 0, Modality.CAMERAS: 1}),
                SceneInfo("point", "Point", frame_counts={Modality.POINT_CLOUD: 1, Modality.CAMERAS: 0}),
                SceneInfo("both", "Both", frame_counts={Modality.POINT_CLOUD: 1, Modality.CAMERAS: 1}),
                SceneInfo("unknown", "Unknown", frame_counts={Modality.POINT_CLOUD: None, Modality.CAMERAS: None}),
            )

    register_loader("shared-mixed-catalog-plugin", MixedCatalog)
    camera = create_dataset("shared-mixed-catalog-plugin")
    try:
        assert [scene.token for scene in camera.list_scenes()] == ["camera", "both", "unknown"]
        if find_spec("threedviz") is not None:
            from threedviz.dataloader import create_dataset as point_dataset
            point = point_dataset("shared-mixed-catalog-plugin")
            try:
                assert [scene.token for scene in point.list_scenes()] == ["point", "both", "unknown"]
            finally:
                point.close()
    finally:
        camera.close()
