"""Camera sources return the same canonical records consumed by projection."""

from dataclasses import replace

import numpy as np
import pytest

from visual_utils import (
    AnnotationSet, AnnotationStatus, Box3D, CameraFrameSet, CameraObservation,
    Capability, ClassInfo, DatasetConfig, DatasetConfigurationError, DatasetLoader,
    DatasetMetadata, FrameInfo, Modality, PinholeProjection, SceneInfo, SensorInfo,
    UnsupportedCapabilityError, register_loader,
)
from cameraviz.dataloader import available_datasets, create_dataset
from cameraviz.dataloader.demo import DemoDataset
from cameraviz.processing import project_boxes
from cameraviz.visualization.layout import camera_layout


class CameraDataset(DatasetLoader):
    def __init__(self, config):
        super().__init__(config)
        self.released = False
        self._metadata = DatasetMetadata(
            "research", "Research cameras", frozenset((Capability.CAMERAS, Capability.BOXES_3D)),
            classes=(ClassInfo("native.car", "native.car", "Car", "vehicles"),),
            sensors=(SensorInfo("front", Modality.CAMERAS, "Front", "front"),
                     SensorInfo("other", Modality.CAMERAS, "Other")),
        )
        info = FrameInfo("opaque-15", "scene", 0, -100, frozenset((Modality.CAMERAS,)))
        boxes = (Box3D("box", "native.car", np.array([0., 0, 10]), np.ones(3) * 2, np.eye(3)),)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        model = PinholeProjection(np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]]), np.eye(4))
        cameras = (CameraObservation("front", image, -110, "relative exposure", projection_model=model),
                   CameraObservation("other", image))
        world = np.eye(4)
        world[:3, 3] = [500., 200., 10.]
        self.frame = CameraFrameSet(info, "recording", "local", cameras,
            AnnotationSet("local", AnnotationStatus.AVAILABLE, boxes),
            world_frame_id="world", T_world_from_reference=world)

    @property
    def metadata(self):
        return self._metadata

    def _list_scenes(self, *, split="all"):
        return (SceneInfo("scene", "Scene", frame_counts={Modality.CAMERAS: 1}),)

    def _list_frames(self, scene_id, *, modality):
        return (self.frame.info,)

    def _load_cameras(self, frame_id):
        return self.frame

    def _close(self):
        self.released = True


def test_registered_provider_preserves_records_native_ids_reference_and_time():
    register_loader("camera-canonical-test", CameraDataset)
    config = DatasetConfig()
    with create_dataset("camera-canonical-test", config=config) as dataset:
        assert dataset.config is config
        assert isinstance(dataset, CameraDataset)
        frame = dataset.load_cameras("opaque-15")
        assert frame is dataset.frame
        assert frame.annotations.boxes[0].class_id == "native.car"
        assert frame.info.timestamp_us == -100
        assert frame.cameras[0].timestamp_us == -110
        # Projection uses the declared local reference; no world round trip.
        overlays = project_boxes(frame)
        assert overlays["front"][0].class_id == "native.car"
        np.testing.assert_allclose(overlays["front"][0].rectangle, [50 - 100 / 9, 50 - 100 / 9, 50 + 100 / 9, 50 + 100 / 9])
        assert overlays["other"] == ()  # No projection model means no overlay.
    assert dataset.released


def test_demo_records_are_deterministic_and_scene_navigation_uses_stable_ids():
    with create_dataset("demo") as dataset:
        assert isinstance(dataset, DemoDataset)
        scenes = dataset.list_scenes()
        assert len(scenes) == 2
        infos = dataset.list_frames(scenes[0].scene_id, modality=Modality.CAMERAS)
        first = dataset.load_cameras(infos[0].frame_id)
        repeated = dataset.load_cameras(infos[0].frame_id)
        following = dataset.load_cameras(infos[1].frame_id)
        assert first.info.index == 0 and following.info.index == 1
        assert len(first.cameras) == 6 and len(first.annotations.boxes) == 10
        assert all(np.array_equal(a.image, b.image) for a, b in zip(first.cameras, repeated.cameras))
        assert not np.array_equal(first.annotations.boxes[0].center, following.annotations.boxes[0].center)
        assert any(project_boxes(first).values())


def test_layout_uses_roles_and_preserves_unknown_or_missing_camera_slots():
    sensors = (SensorInfo("front", Modality.CAMERAS, "Front", "front"),
               SensorInfo("unknown", Modality.CAMERAS, "Unknown"),
               SensorInfo("another-front", Modality.CAMERAS, "Second", "front"),
               SensorInfo("side", Modality.CAMERAS, "Side", "side_left"))
    layout = camera_layout(sensors)
    assert [slot.sensor_id for slot in layout] == [sensor.sensor_id for sensor in sensors]
    assert [(slot.row, slot.column) for slot in layout] == [(0, 1), (0, 0), (0, 2), (1, 0)]


@pytest.mark.parametrize("source_args", [{"root": "/data"}, {"root": ""}, {"version": "custom"}])
def test_configuration_and_convenience_source_arguments_are_exclusive(source_args):
    with pytest.raises(DatasetConfigurationError, match="either DatasetConfig or root/version"):
        create_dataset("nuscenes", config=DatasetConfig(), **source_args)


def test_point_only_source_is_closed_when_opened_in_camera_viewer(monkeypatch):
    from cameraviz.dataloader import factory

    dataset = CameraDataset(DatasetConfig())
    dataset._metadata = replace(dataset.metadata, capabilities=frozenset((Capability.POINT_CLOUD,)), sensors=())
    monkeypatch.setattr(factory, "create_loader", lambda *args: dataset)
    with pytest.raises(UnsupportedCapabilityError, match="camera observations"):
        create_dataset("nuscenes")
    assert dataset.released


def test_demo_name_collision_is_reported_before_source_selection(monkeypatch):
    from cameraviz.dataloader import factory

    monkeypatch.setattr(factory, "available_loaders", lambda: ("demo", "research"))
    with pytest.raises(DatasetConfigurationError, match="reserves 'demo'"):
        available_datasets()
