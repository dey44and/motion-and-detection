"""Camera review joins calibrated images and overlays by frame and point row."""

from dataclasses import replace
from io import BytesIO
import struct

import numpy as np
from PIL import Image
import pytest

from threedviz.detection.predictions import Detection
from threedviz.web import cameras as review
from visual_utils import (
    AcquisitionGeometry, AnnotationSet, AnnotationStatus, Box3D, CameraFrameSet, CameraObservation,
    Capability, ClassInfo, DatasetMetadata, DataUnavailableError, DatasetValidationError, FrameInfo,
    Modality, PinholeProjection, PointCloudFrame, ProcessingProvenance, SensorInfo, UnsupportedCapabilityError,
)
from visual_utils.box_projection import project_box


class Loader:
    def __init__(self, cameras):
        self.cameras = cameras
        self.camera_reads = 0
        self.metadata = DatasetMetadata(
            "camera-test", "Test cameras", {Capability.POINT_CLOUD, Capability.CAMERAS},
            classes=(ClassInfo("vehicle.car", "Car", "Car", "car"),),
            sensors=(SensorInfo("front", Modality.CAMERAS, "Front camera", "front"),),
        )

    def load_cameras(self, frame_id):
        assert frame_id == "frame-id"
        self.camera_reads += 1
        return self.cameras


@pytest.fixture
def sensors():
    info = FrameInfo("frame-id", "scene-id", 0, None, {Modality.POINT_CLOUD, Modality.CAMERAS})
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    world_from_cameras = np.eye(4)
    world_from_cameras[:3, :3] = rotation
    world_from_cameras[:3, 3] = [8, 4, 3]
    camera_from_lidar = np.eye(4)
    camera_from_lidar[:3, :3] = rotation
    camera_from_lidar[:3, 3] = [1, 0, 0]
    rows = np.array([[0., 0., 2.], [0., 0., 8.], [1., 0., 8.], [100., 0., 2.], [0., 0., -1.], [0., 0., 0.05]])
    cloud = PointCloudFrame(info, "lidar", "lidar-reference", rows, world_frame_id="world",
                           T_world_from_reference=world_from_cameras @ camera_from_lidar,
                           acquisition=AcquisitionGeometry(np.full_like(rows, 30.), np.ones(len(rows), dtype=bool),
                                                           "raw-reference", ProcessingProvenance("fixture")))
    model = PinholeProjection(np.array([[4., 0., 4.], [0., 4., 3.], [0., 0., 1.]]), np.eye(4))
    image = np.full((7, 9, 3), 140, dtype=np.uint8)
    default = Box3D("shared", "vehicle.car", [0, 0, 4], [1, 1, 1], np.eye(3))
    override = replace(default, annotation_id="camera-specific", center=[1, 0, 4])
    cameras = CameraFrameSet(
        info, "images", "camera-reference",
        (CameraObservation("front", image, projection_model=model,
                           annotations=AnnotationSet("camera-reference", AnnotationStatus.AVAILABLE, (override,))),
         CameraObservation("uncalibrated", image),
         CameraObservation("other", image, projection_model=model)),
        AnnotationSet("camera-reference", AnnotationStatus.AVAILABLE, (default,)),
        world_frame_id="world", T_world_from_reference=world_from_cameras,
    )
    prediction = Detection(Box3D("original", "car", [0, 0, 4], [.5, 1, 1], np.eye(3)), .875, "result:1")
    return Loader(cameras), cloud, prediction, rotation


def test_review_projects_native_annotations_predictions_and_exact_point_rows(sensors):
    loader, cloud, prediction, rotation = sensors
    snapshot = review.build_camera_review(loader, cloud, (prediction,), revision=3, include_points=True)
    state = snapshot.public()
    assert state["revision"] == 3
    front, uncalibrated, other = state["views"]
    assert (front["id"], front["name"], front["role"]) == ("front", "Front camera", "front")
    assert (front["width"], front["height"], front["projection"]) == (9, 7, True)
    assert front["point_count"] == 3 and other["point_count"] == 3
    assert [item["id"] for item in front["ground_truth"]] == ["camera-specific"]
    assert front["ground_truth"][0]["group"] == "car"
    assert [item["id"] for item in other["ground_truth"]] == ["shared"]
    expected_box = replace(prediction.box, center=[1, 0, 4], rotation=rotation)
    expected_coordinates = project_box(expected_box, loader.cameras.cameras[0]).segments.reshape(-1, 4)
    assert front["predictions"][0]["id"] == "result:1"
    assert front["predictions"][0]["score"] == .875
    np.testing.assert_allclose(front["predictions"][0]["segments"], expected_coordinates)
    assert uncalibrated["projection"] is False and uncalibrated["point_count"] is None
    assert uncalibrated["ground_truth"] == uncalibrated["predictions"] == []
    assert list(struct.iter_unpack("<ffI", snapshot.points("front"))) == [(4.5, 3., 1), (4.5, 3.5, 2), (6., 3., 0)]
    with Image.open(BytesIO(snapshot.image("uncalibrated"))) as jpeg:
        assert jpeg.format == "JPEG" and jpeg.size == (9, 7)
    with pytest.raises(DataUnavailableError):
        snapshot.points("uncalibrated")
    with pytest.raises(KeyError):
        snapshot.image("unknown")


def test_previous_review_reuses_images_and_unchanged_overlays(sensors, monkeypatch):
    loader, cloud, prediction, _ = sensors
    first = review.build_camera_review(loader, cloud, (prediction,), revision=3, include_points=False)
    assert first.public()["views"][0]["point_count"] is None

    def forbid(*args):
        raise AssertionError("Existing images and box geometry must be reused.")

    with monkeypatch.context() as patch:
        patch.setattr(review, "_jpeg", forbid)
        patch.setattr(review, "project_box", forbid)
        enabled = review.build_camera_review(loader, cloud, (prediction,), revision=3, include_points=True, previous=first)
        assert enabled.public()["views"][0]["predictions"] == first.public()["views"][0]["predictions"]
        assert enabled.points("front")
    with monkeypatch.context() as patch:
        patch.setattr(review, "_jpeg", forbid)
        patch.setattr(review, "_point_rows", forbid)
        filtered = review.build_camera_review(loader, cloud, (), revision=3, include_points=True, previous=enabled)
        assert filtered.points("front") is enabled.points("front")
        assert filtered.image("front") is enabled.image("front")
        assert filtered.public()["views"][0]["predictions"] == []
        assert filtered.public()["views"][0]["ground_truth"] == first.public()["views"][0]["ground_truth"]
        disabled = review.build_camera_review(loader, cloud, (), revision=3, include_points=False, previous=filtered)
        with pytest.raises(DataUnavailableError):
            disabled.points("front")
        reenabled = review.build_camera_review(loader, cloud, (), revision=3, include_points=True, previous=disabled)
        assert reenabled.points("front") is enabled.points("front")
    assert loader.camera_reads == 1


def test_independent_camera_world_preserves_its_native_annotations(sensors):
    loader, cloud, prediction, _ = sensors
    loader.cameras = replace(loader.cameras, world_frame_id="unrelated-world")
    state = review.build_camera_review(loader, cloud, (prediction,), revision=1, include_points=True).public()
    assert [item["id"] for item in state["views"][0]["ground_truth"]] == ["camera-specific"]
    assert state["views"][0]["predictions"] == []
    assert state["views"][0]["point_count"] is None


def test_shared_reference_needs_no_world_pose_and_unsupported_model_keeps_image(sensors):
    loader, cloud, prediction, _ = sensors
    cloud = replace(cloud, reference_frame_id="camera-reference", annotations=None,
                    world_frame_id=None, T_world_from_reference=None)

    class UnsupportedModel:
        def project_reference(self, points):
            raise NotImplementedError()

    front, uncalibrated, other = loader.cameras.cameras
    unsupported = replace(other, projection_model=UnsupportedModel())
    loader.cameras = replace(loader.cameras, cameras=(front, uncalibrated, unsupported))
    state = review.build_camera_review(loader, cloud, (prediction,), revision=1, include_points=True)
    assert list(struct.iter_unpack("<ffI", state.points("front"))) == [(4., 3., 1), (4.5, 3., 2), (4., 3., 0)]
    assert state.public()["views"][2]["projection"] is False
    assert state.image("other").startswith(b"\xff\xd8")


def test_capability_and_camera_identity_are_validated(sensors):
    loader, cloud, prediction, _ = sensors
    loader.cameras = replace(loader.cameras, info=replace(loader.cameras.info, scene_id="wrong-scene"))
    with pytest.raises(DatasetValidationError, match="selected scene and frame"):
        review.build_camera_review(loader, cloud, (prediction,), revision=1, include_points=False)
    loader.metadata = replace(loader.metadata, capabilities={Capability.POINT_CLOUD})
    with pytest.raises(UnsupportedCapabilityError, match="does not provide camera"):
        review.build_camera_review(loader, cloud, (prediction,), revision=1, include_points=False)
    assert loader.camera_reads == 1


def test_thumbnails_keep_orientation_and_avoid_overlay_work(sensors, monkeypatch):
    loader, cloud, _, _ = sensors
    front, *other = loader.cameras.cameras
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    colors = ((240, 20, 20), (20, 240, 20), (20, 20, 240), (240, 240, 20))
    image[:160, :320], image[:160, 320:], image[160:, :320], image[160:, 320:] = colors
    loader.cameras = replace(loader.cameras, cameras=(replace(front, image=image), *other))

    def forbid(*args):
        pytest.fail("Thumbnail extraction must skip overlay computation.")

    monkeypatch.setattr(review, "project_box", forbid)
    monkeypatch.setattr(review, "_point_rows", forbid)
    thumbnails = review.build_camera_thumbnails(loader, cloud, revision=4)
    assert thumbnails.revision == 4
    assert thumbnails.camera_ids == ("front", "uncalibrated", "other")
    with Image.open(BytesIO(thumbnails.image("front"))) as jpeg:
        assert jpeg.size == (256, 128) and jpeg.format == "JPEG"
        pixels = tuple(jpeg.getpixel(pos) for pos in ((16, 16), (240, 16), (16, 112), (240, 112)))
        np.testing.assert_allclose(pixels, colors, atol=4)
    with Image.open(BytesIO(thumbnails.image("uncalibrated"))) as jpeg:
        assert jpeg.size == (9, 7)
    with pytest.raises(KeyError):
        thumbnails.image("unknown")
    assert loader.camera_reads == 1


def test_thumbnails_reuse_only_a_current_camera_review(sensors):
    loader, cloud, _, _ = sensors
    previous = review.build_camera_review(loader, cloud, (), revision=4, include_points=False)
    thumbnails = review.build_camera_thumbnails(loader, cloud, revision=4, review=previous)
    assert loader.camera_reads == 1

    front, *other = loader.cameras.cameras
    image = np.full_like(front.image, 255)
    loader.cameras = replace(loader.cameras, cameras=(replace(front, image=image), *other))
    fresh = review.build_camera_thumbnails(loader, cloud, revision=5, review=previous)
    assert loader.camera_reads == 2
    assert fresh.image("front") != thumbnails.image("front")
