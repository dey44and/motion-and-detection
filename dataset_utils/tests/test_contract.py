"""Conformance invariants exercised independently of either built-in dataset."""

from dataclasses import replace

import numpy as np
import pytest

from dataset_utils import (
    AcquisitionGeometry, AnnotationAlignment, AnnotationSet, AnnotationStatus,
    AnnotationTiming, Box3D, CameraFrameSet, CameraObservation, Capability,
    ClassInfo, DataUnavailableError, DatasetClosedError, DatasetConfig,
    DatasetConfigurationError, DatasetIOError, DatasetLoader, DatasetMetadata,
    DatasetValidationError, FrameInfo, Modality, PinholeProjection,
    PointCloudFrame, PointCloudPreparation, PointColors, ProcessingProvenance,
    SceneInfo, SensorInfo, UnknownIdentifierError, UnsupportedCapabilityError,
    available_loaders, create_loader, register_loader,
)
from dataset_utils.contracts import registry


class PointProvider(DatasetLoader):
    def __init__(self, config):
        super().__init__(config)
        self.reads = []
        self.closed_count = 0
        self.break_preparation = None
        self._metadata = DatasetMetadata(
            "test:point", "Point fixture",
            {Capability.POINT_CLOUD, Capability.RGB_COLORING, Capability.ACQUISITION_GEOMETRY},
            sensors=(SensorInfo("lidar", Modality.POINT_CLOUD, "LiDAR"),),
        )

    @property
    def metadata(self):
        return self._metadata

    def list_scenes(self, *, split="all"):
        return (SceneInfo("scene", "Scene", frame_counts={Modality.POINT_CLOUD: None}),)

    def list_frames(self, scene_id, *, modality):
        if scene_id != "scene":
            raise UnknownIdentifierError(scene_id)
        return tuple(FrameInfo(token, "scene", index, modalities={Modality.POINT_CLOUD})
                     for index, token in enumerate(("native-15", "native-39")))

    def load_point_cloud(self, frame_id):
        frames = self.list_frames("scene", modality=Modality.POINT_CLOUD)
        try:
            info = next(frame for frame in frames if frame.frame_id == frame_id)
        except StopIteration:
            raise UnknownIdentifierError(frame_id) from None
        self.reads.append(frame_id)
        return PointCloudFrame(
            info, "native:all-returns", f"local:{frame_id}",
            np.array([[1, 0, 2], [0, 1, 3], [2, 0, 4]], dtype=np.float32),
            attributes={"intensity": np.array([255., 1., 2.])},
        )

    def prepare_point_cloud(self, frame, *, request):
        result = frame
        if request.rgb:
            result = replace(result, rgb=PointColors(
                np.array([[0., 0., 0.], [.2, .3, .4], [1., 1., 1.]]),
                np.array([True, False, True]),
                ProcessingProvenance("test.native_colors", (frame.source_id,)),
            ))
        if request.acquisition_geometry:
            result = replace(result, acquisition=AcquisitionGeometry(
                frame.points + [1., 0., 0.], np.ones(len(frame.points), dtype=bool),
                f"acquisition:{frame.info.frame_id}",
                ProcessingProvenance("test.recorded_poses", (frame.source_id,)),
            ))
        if self.break_preparation == "rows":
            result = replace(result, points=result.points[::-1])
        elif self.break_preparation == "source":
            result = replace(result, source_id="another-representation")
        elif self.break_preparation == "missing":
            result = replace(result, rgb=None)
        elif self.break_preparation == "io":
            raise FileNotFoundError("declared-camera.jpg")
        return result

    def _close(self):
        self.closed_count += 1


class CameraProvider(DatasetLoader):
    def __init__(self, config):
        super().__init__(config)
        self.image = np.full((4, 6, 3), 23, dtype=np.uint8)
        self._metadata = DatasetMetadata(
            "test:camera", "Camera fixture", {Capability.CAMERAS},
            sensors=(SensorInfo("unusual-view", Modality.CAMERAS, "Custom camera", "unknown"),),
        )

    @property
    def metadata(self):
        return self._metadata

    def list_scenes(self, *, split="all"):
        return (SceneInfo("scene", "Camera scene", frame_counts={Modality.CAMERAS: 1}),)

    def list_frames(self, scene_id, *, modality):
        if scene_id != "scene":
            raise UnknownIdentifierError(scene_id)
        return (FrameInfo("image-8", "scene", 0, modalities={Modality.CAMERAS}),)

    def load_cameras(self, frame_id):
        if frame_id != "image-8":
            raise UnknownIdentifierError(frame_id)
        return CameraFrameSet(
            self.list_frames("scene", modality=Modality.CAMERAS)[0],
            "original-images", "local:image-8",
            (CameraObservation("unusual-view", self.image),),
        )


def test_configuration_snapshots_paths_and_nested_mutable_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    options = {"nested": {"channels": ["a", "b"]}}
    sources = {"native": "raw"}
    config = DatasetConfig(root="dataset", source_paths=sources, options=options)
    options["nested"]["channels"].append("c")
    sources["native"] = "elsewhere"
    assert config.root == tmp_path / "dataset"
    assert config.source_paths["native"] == tmp_path / "dataset/raw"
    assert config.options["nested"]["channels"] == ("a", "b")
    with pytest.raises(TypeError):
        config.options["nested"]["new"] = 1


@pytest.mark.parametrize("options", [{"x": float("nan")}, {"x": object()}, {1: "invalid"}])
def test_configuration_rejects_nonserializable_or_ambiguous_values(options):
    with pytest.raises(DatasetConfigurationError):
        DatasetConfig(options=options)


def test_registry_rejects_duplicate_wrong_type_and_unknown_versions(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    register_loader(" Point-Fixture ", PointProvider)
    assert available_loaders() == ("point-fixture",)
    with pytest.raises(DatasetConfigurationError, match="registered"):
        register_loader("POINT-FIXTURE", PointProvider)
    loader = create_loader("POINT-FIXTURE", DatasetConfig())
    assert isinstance(loader, DatasetLoader)
    register_loader("wrong", lambda config: object())
    with pytest.raises(DatasetConfigurationError, match="DatasetLoader"):
        create_loader("wrong", DatasetConfig())
    instance = PointProvider(DatasetConfig())
    instance._metadata = replace(instance.metadata, contract_version=2)
    register_loader("future", lambda config: instance)
    with pytest.raises(DatasetConfigurationError, match="version"):
        create_loader("future", DatasetConfig())
    assert instance.closed_count == 1


def test_sparse_native_ids_unknown_counts_and_no_fabricated_times():
    loader = PointProvider(DatasetConfig())
    assert loader.list_scenes()[0].frame_counts[Modality.POINT_CLOUD] is None
    infos = loader.list_frames("scene", modality=Modality.POINT_CLOUD)
    assert loader.reads == []
    assert [(info.index, info.frame_id, info.timestamp_us) for info in infos] == [
        (0, "native-15", None), (1, "native-39", None),
    ]
    frame = loader.load_point_cloud(infos[1].frame_id)
    assert loader.reads == ["native-39"]
    assert frame.info.index == 1 and frame.T_world_from_reference is None
    with pytest.raises(UnsupportedCapabilityError):
        loader.list_frames("scene", modality=Modality.CAMERAS)
    with pytest.raises(UnsupportedCapabilityError):
        loader.load_cameras("native-39")


def test_camera_only_uncalibrated_data_is_valid_and_retained_after_close():
    loader = CameraProvider(DatasetConfig())
    frame = loader.load_cameras("image-8")
    camera = frame.cameras[0]
    assert camera.projection_model is None and camera.intrinsics is None
    assert camera.T_reference_from_camera is None
    assert frame.annotations.status == AnnotationStatus.NOT_PROVIDED
    loader.image[:] = 0
    loader.close()
    assert np.all(camera.image == 23)
    assert not camera.image.flags.writeable
    with pytest.raises(UnsupportedCapabilityError):
        CameraProvider(DatasetConfig()).load_point_cloud("image-8")


def test_geometry_can_be_prepared_without_scalar_point_timestamps():
    loader = PointProvider(DatasetConfig())
    frame = loader.load_point_cloud("native-15")
    prepared = loader.prepare_point_cloud(frame, request=PointCloudPreparation(acquisition_geometry=True))
    assert prepared.timing is None
    assert prepared.acquisition.valid_mask.all()
    np.testing.assert_array_equal(prepared.points, frame.points)
    assert prepared.info is frame.info
    with pytest.raises(UnsupportedCapabilityError):
        loader.prepare_point_cloud(frame, request=PointCloudPreparation(timestamps=True))


def test_rgb_black_pixels_are_valid_and_previous_data_is_preserved():
    loader = PointProvider(DatasetConfig())
    frame = loader.load_point_cloud("native-15")
    colored = loader.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True))
    assert colored.rgb.valid_mask[0] and not colored.rgb.values[0].any()
    assert not colored.rgb.valid_mask[1]
    assert frame.rgb is None
    full = loader.prepare_point_cloud(colored, request=PointCloudPreparation(acquisition_geometry=True))
    assert full.rgb is colored.rgb
    assert loader.prepare_point_cloud(full, request=PointCloudPreparation(rgb=True)) is full
    assert not full.points.flags.writeable and not full.rgb.values.flags.writeable


@pytest.mark.parametrize("problem", ["rows", "source", "missing"])
def test_failed_preparation_is_atomic_and_does_not_publish_wrong_correspondence(problem):
    loader = PointProvider(DatasetConfig())
    frame = loader.load_point_cloud("native-15")
    loader.break_preparation = problem
    with pytest.raises(DatasetValidationError):
        loader.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True))
    assert frame.rgb is None
    np.testing.assert_array_equal(frame.points[0], [1, 0, 2])


def test_foreign_reordered_or_injected_derived_frames_cannot_be_prepared():
    loader = PointProvider(DatasetConfig())
    frame = loader.load_point_cloud("native-15")
    other = PointProvider(DatasetConfig())
    with pytest.raises(DatasetValidationError, match="foreign"):
        other.prepare_point_cloud(frame, request=PointCloudPreparation())
    with pytest.raises(DatasetValidationError, match="foreign"):
        loader.prepare_point_cloud(replace(frame, points=frame.points[::-1]), request=PointCloudPreparation(rgb=True))
    forged = replace(frame, rgb=PointColors(
        np.zeros((3, 3)), np.ones(3, dtype=bool), ProcessingProvenance("fake"),
    ))
    with pytest.raises(DatasetValidationError, match="foreign"):
        loader.prepare_point_cloud(forged, request=PointCloudPreparation(rgb=True))


def test_declared_missing_inputs_are_errors_not_all_false_masks():
    loader = PointProvider(DatasetConfig())
    frame = loader.load_point_cloud("native-15")
    loader.break_preparation = "io"
    with pytest.raises(DatasetIOError, match="declared-camera.jpg") as failure:
        loader.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True))
    assert isinstance(failure.value.__cause__, FileNotFoundError)
    assert frame.rgb is None


def test_annotations_distinguish_zero_withheld_and_mixed_time():
    zero = AnnotationSet("local", AnnotationStatus.AVAILABLE)
    withheld = AnnotationSet("local", AnnotationStatus.WITHHELD)
    missing = AnnotationSet("local")
    assert zero.boxes == withheld.boxes == missing.boxes == ()
    assert len({zero.status, withheld.status, missing.status}) == 3
    box = Box3D("object", "native", np.zeros(3), np.ones(3), np.eye(3))
    with pytest.raises(DatasetValidationError, match="timing override"):
        AnnotationSet("local", AnnotationStatus.AVAILABLE, (box,), AnnotationTiming(None, AnnotationAlignment.MIXED))
    synced = replace(box, timing_override=AnnotationTiming(None, AnnotationAlignment.CAMERA_SYNCHRONIZED))
    mixed = AnnotationSet("local", AnnotationStatus.AVAILABLE, (synced,), AnnotationTiming(None, AnnotationAlignment.MIXED))
    assert mixed.boxes[0].timing_override.timestamp_us is None


def test_pinhole_projection_retains_signed_depth_and_outside_image_coordinates():
    projection = PinholeProjection(np.array([[10., 0, 5], [0, 10, 5], [0, 0, 1]]), np.eye(4))
    result = projection.project_reference(np.array([[0., 0, 1], [0, 0, -1], [10, 0, 1]]))
    np.testing.assert_array_equal(result.valid_mask, [True, False, True])
    assert result.depth[1] == -1
    assert result.uv[2, 0] > 100
    assert np.isnan(result.uv[1]).all()


def test_close_is_idempotent_and_guards_all_public_operations():
    loader = PointProvider(DatasetConfig())
    frame = loader.load_point_cloud("native-15")
    metadata = loader.metadata
    loader.close()
    loader.close()
    assert loader.closed_count == 1 and loader.metadata is metadata
    for operation in (
        lambda: loader.list_scenes(),
        lambda: loader.list_frames("scene", modality=Modality.POINT_CLOUD),
        lambda: loader.load_point_cloud("native-15"),
        lambda: loader.load_cameras("native-15"),
        lambda: loader.prepare_point_cloud(frame, request=PointCloudPreparation()),
    ):
        with pytest.raises(DatasetClosedError):
            operation()
    np.testing.assert_array_equal(frame.points[0], [1, 0, 2])


def test_keyword_preparation_is_the_same_public_api():
    loader = PointProvider(DatasetConfig())
    baseline = loader.load_point_cloud(frame_id="native-15")
    prepared = loader.prepare_point_cloud(frame=baseline, request=PointCloudPreparation(rgb=True))
    assert prepared.rgb.valid_mask.tolist() == [True, False, True]


def test_readonly_views_of_reused_buffers_are_frozen_snapshots():
    backing = np.array([[1., 2., 3.]])
    view = backing.view()
    view.setflags(write=False)
    frame = PointCloudFrame(FrameInfo("f", "s", 0, modalities={Modality.POINT_CLOUD}), "native", "local:f", view)
    backing[:] = 99
    np.testing.assert_array_equal(frame.points, [[1., 2., 3.]])
    with pytest.raises(ValueError):
        frame.points.setflags(write=True)


def test_preparation_cannot_publish_undeclared_additional_features():
    from dataset_utils import PointTiming, TimingQuality

    class UndeclaredTiming(PointProvider):
        def prepare_point_cloud(self, frame, *, request):
            colored = super().prepare_point_cloud(frame, request=request)
            return replace(colored, timing=PointTiming(
                np.zeros(len(frame.points), dtype=np.int64), np.ones(len(frame.points), dtype=bool),
                TimingQuality.RECORDED, ProcessingProvenance("injected.undeclared"),
            ))

    loader = UndeclaredTiming(DatasetConfig())
    with pytest.raises(UnsupportedCapabilityError, match="point_timestamps"):
        loader.prepare_point_cloud(loader.load_point_cloud("native-15"), request=PointCloudPreparation(rgb=True))


def test_bad_contract_error_survives_provider_cleanup_failure(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})

    class BrokenCleanup(PointProvider):
        def _close(self):
            raise RuntimeError("cleanup error")

    loader = BrokenCleanup(DatasetConfig())
    loader._metadata = replace(loader.metadata, contract_version=2)
    register_loader("bad-version", lambda config: loader)
    with pytest.raises(DatasetConfigurationError, match="version"):
        create_loader("bad-version", DatasetConfig())
    with pytest.raises(DatasetClosedError):
        loader.list_scenes()


def test_projection_protocol_member_must_be_callable():
    class InvalidProjection:
        project_reference = 42

    with pytest.raises(DatasetValidationError, match="Projection model"):
        CameraObservation("camera", np.zeros((2, 2, 3), dtype=np.uint8), projection_model=InvalidProjection())
