"""Native Waymo camera calibration, shutter geometry and adapter contracts."""

from dataclasses import replace
import numpy as np
import pytest

from visual_utils.waymo.camera import OPTICAL_TO_WAYMO, WaymoCameraProjection
from visual_utils.waymo.models import NativeBox, NativeCamera, NativeFrame, NativeFrameHeader, NativeSequence
from visual_utils import (
    AnnotationSet, AnnotationStatus, Box3D, CameraFrameSet, CameraObservation,
    DatasetError, FrameInfo, Modality, ProjectionResult,
)
from cameraviz.dataloader import create_dataset
from cameraviz.visualization.layout import camera_layout
from cameraviz.processing import project_box, project_boxes


def native_camera(*, name=1, width=100, height=100, intrinsic=None, direction=5, velocity=None, pose=None, extrinsic=None, timestamp=.05):
    return NativeCamera(
        name=name, image_rgb=np.zeros((height, width, 3), dtype=np.uint8),
        intrinsic=np.array([100., 100., 50., 50., 0, 0, 0, 0, 0]) if intrinsic is None else np.asarray(intrinsic, dtype=float),
        extrinsic=np.eye(4) if extrinsic is None else extrinsic,
        vehicle_to_world=np.eye(4) if pose is None else pose,
        pose_timestamp_s=timestamp, velocity=np.zeros(6) if velocity is None else np.asarray(velocity, dtype=float),
        shutter_s=0., trigger_time_s=0., readout_done_time_s=.1,
        rolling_shutter_direction=direction,
    )


def camera_frame(native, *, boxes=None):
    class NativeProjection:
        def project_reference(self, points):
            uv, depth = WaymoCameraProjection(native).project_world(points)
            return ProjectionResult(uv, depth, np.isfinite(uv).all(axis=1) & (depth > 0))

    annotations = None if boxes is None else AnnotationSet("world", AnnotationStatus.AVAILABLE, boxes)
    return CameraObservation("CAM_FRONT", native.image_rgb, projection_model=NativeProjection(), annotations=annotations)


def world_box(center=(10, 0, 0), size=(4, 2, 2)):
    return Box3D("box", "Vehicle", np.asarray(center, dtype=float), np.asarray(size, dtype=float), np.eye(3))


def test_native_forward_x_is_mapped_to_optical_forward_z_and_behind_is_rejected():
    model = WaymoCameraProjection(native_camera())
    pixels, depth = model.project_world(np.array([[10., -2, -1], [-1., 0, 0]]))
    np.testing.assert_allclose(pixels[0], [70, 60])
    assert depth[0] == 10
    assert np.isnan(pixels[1]).all()
    assert depth[1] < 0
    pixels, depth = model.project_world(np.empty((0, 3)))
    assert pixels.shape == (0, 2) and depth.shape == (0,)


def test_native_world_pose_and_camera_extrinsic_both_apply():
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [20, 30, 4]
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [1, 2, 3]
    native = native_camera(pose=pose, extrinsic=extrinsic)
    expected_cam_point = np.array([10., -2, -1])
    world = pose @ extrinsic @ np.r_[expected_cam_point, 1.]
    pixels, depth = WaymoCameraProjection(native).project_world(world[None, :3])
    np.testing.assert_allclose(pixels, [[70, 60]])
    np.testing.assert_allclose(depth, [10])


def test_native_brown_radial_and_tangential_distortion_matches_hand_formula():
    native = native_camera(intrinsic=[100, 120, 50, 40, .01, -.005, .002, -.003, .001])
    pixels, _ = WaymoCameraProjection(native).project_world(np.array([[10., -2, -1]]))
    u, v, r2 = .2, .1, .05
    radial = 1 + .01 * r2 - .005 * r2**2 + .001 * r2**3
    expected_u = 100 * (u * radial + 2 * .002 * u * v - .003 * (r2 + 2 * u * u)) + 50
    expected_v = 120 * (v * radial + .002 * (r2 + 2 * v * v) - 2 * .003 * u * v) + 40
    np.testing.assert_allclose(pixels, [[expected_u, expected_v]], rtol=0, atol=1e-12)


@pytest.mark.parametrize("direction,point,velocity,expected", [
    (2, [10, -2, 0], [0, 10, 0, 0, 0, 0], [50 + 100 * 2 / 9, 50]),
    (4, [10, -2, 0], [0, 10, 0, 0, 0, 0], [50 + 100 * 2 / 11, 50]),
    (1, [10, 0, -2], [0, 0, 10, 0, 0, 0], [50, 50 + 100 * 2 / 9]),
    (3, [10, 0, -2], [0, 0, 10, 0, 0, 0], [50, 50 + 100 * 2 / 11]),
])
def test_all_four_rolling_shutter_directions_solve_capture_time(direction, point, velocity, expected):
    # At the anchor pose, normalized coordinate is .2. Motion makes it .2+t;
    # the .1-second readout implies t=+/-.1*(.2+t), with the two exact roots.
    model = WaymoCameraProjection(native_camera(direction=direction, velocity=velocity))
    pixels, depth = model.project_world(np.array([point], dtype=float))
    np.testing.assert_allclose(pixels[0], expected, rtol=0, atol=1e-8)
    assert depth[0] > 0


def test_camera_pose_timestamp_offset_is_not_substituted_by_image_center_time():
    model = WaymoCameraProjection(native_camera(direction=2, velocity=[0, 10, 0, 0, 0, 0], timestamp=.03))
    pixels, _ = model.project_world(np.array([[10., -2, 0]]))
    # t=.1*(.2+t)+.02, so normalized u=.2+.04/.9.
    np.testing.assert_allclose(pixels[0], [50 + 100 * (.2 + .04 / .9), 50], atol=1e-8)


def test_angular_motion_includes_camera_lever_arm_velocity():
    extrinsic = np.eye(4)
    extrinsic[0, 3] = 1
    model = WaymoCameraProjection(native_camera(direction=2, velocity=[0, 0, 0, 0, 0, 2], extrinsic=extrinsic))
    pixels, _ = model.project_world(np.array([[11., -2, 0]]))
    # C(t)=(I-t*[omega]x), camera translation=(1,2t,0), giving
    # u=(2+22t)/(10-4t-4t^2) and t=.1u.
    roots = np.roots([-4., -4., 7.8, -.2])
    elapsed = min((float(root.real) for root in roots if abs(root.imag) < 1e-12), key=abs)
    normalized = (2 + 22 * elapsed) / (10 - 4 * elapsed - 4 * elapsed**2)
    np.testing.assert_allclose(pixels[0], [50 + 100 * normalized, 50], atol=1e-7)


def test_global_shutter_does_not_apply_rolling_motion():
    native = native_camera(direction=5, velocity=[0, 100, 0, 0, 0, 5])
    pixels, _ = WaymoCameraProjection(native).project_world(np.array([[10., -2, 0]]))
    np.testing.assert_allclose(pixels, [[70, 50]])


def test_native_nonlinear_wireframes_are_sampled_and_keep_expected_pinhole_bounds():
    camera = camera_frame(native_camera(width=101, height=101))
    projected = project_box(world_box(), camera)
    assert projected.segments.shape == (192, 2, 2)
    np.testing.assert_allclose(projected.rectangle, [37.5, 37.5, 62.5, 62.5])
    distorted = native_camera(width=101, height=101, intrinsic=[100, 100, 50, 50, .05, 0, 0, 0, 0])
    pixels, _ = WaymoCameraProjection(distorted).project_world(np.array([[8., -1, 1], [8., 0, 1], [8., 1, 1]]))
    assert pixels[0, 1] != pixels[1, 1]  # Straight 3D edge becomes curved.
    assert project_box(world_box(), camera_frame(distorted)).segments.shape == (192, 2, 2)


def test_nonlinear_clipping_handles_near_plane_outside_and_viewport_filling_faces():
    camera = camera_frame(native_camera(width=101, height=101))
    assert project_box(world_box(center=(-10, 0, 0)), camera) is None
    assert project_box(world_box(center=(10, 100, 0)), camera) is None
    covering = project_box(world_box(size=(2, 100, 100)), camera)
    assert covering.segments.shape == (0, 2, 2)
    np.testing.assert_allclose(covering.rectangle, [0, 0, 100, 100])
    crossing = project_box(world_box(center=(.15, 0, 0), size=(.2, .1, .1)), camera, near_plane=.1)
    np.testing.assert_allclose(crossing.rectangle, [0, 0, 100, 100], atol=1e-6)
    assert np.isfinite(crossing.segments).all()
    assert np.all(crossing.segments >= 0) and np.all(crossing.segments <= 100)


def test_camera_world_box_overrides_are_independent_of_reference_frame_boxes():
    native = native_camera()
    camera = camera_frame(native, boxes=(world_box(),))
    frame = CameraFrameSet(FrameInfo("sample", "context", 0, 1, frozenset((Modality.CAMERAS,))),
        "source", "world", (camera,), AnnotationSet("world", AnnotationStatus.AVAILABLE, (world_box(center=(-10, 0, 0)),)))
    assert len(project_boxes(frame)["CAM_FRONT"]) == 1
    empty_camera = replace(camera, annotations=AnnotationSet("world", AnnotationStatus.AVAILABLE))
    assert project_boxes(replace(frame, cameras=(empty_camera,)))["CAM_FRONT"] == ()
    assert project_boxes(replace(frame, cameras=(replace(camera, annotations=None),)))["CAM_FRONT"] == ()


@pytest.fixture
def native_reader(monkeypatch, tmp_path):
    import visual_utils.waymo.loader as backend
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [100, 200, 3]
    camera_pose = pose.copy()
    camera_pose[0, 3] += .5
    synced = NativeBox(np.array([11., 0, 1]), np.array([4., 2, 2]), .2, "Vehicle", "vehicle")
    boxes = (
        NativeBox(np.array([10., 0, 1]), np.array([4., 2, 2]), .2, "Vehicle", "vehicle", "FRONT", synced),
        NativeBox(np.array([5., 1, 1]), np.array([1., 1, 2]), -.3, "Sign", "sign"),
    )
    cameras = tuple(native_camera(name=number, pose=camera_pose) for number in (5, 3, 1, 4, 2))
    native = NativeFrame("validation-context", 1_000_003, pose, cameras, boxes)
    class Reader:
        native_root = tmp_path
        header_calls = []
        frame_calls = []
        def list_sequences(self):
            return (NativeSequence("train-context", tmp_path / "train.tfrecord", "training"),
                    NativeSequence("validation-context", tmp_path / "val.tfrecord", "validation"))
        def list_frame_headers(self, context):
            self.header_calls.append(context)
            return (NativeFrameHeader(0, 1_000_003), NativeFrameHeader(1, 1_104_525))
        def read_frame(self, context, index, timestamp_us=None, **_modalities):
            self.frame_calls.append((context, index, timestamp_us))
            return replace(native, context_name=context, timestamp_us=timestamp_us)
    reader = Reader()
    monkeypatch.setattr(backend, "NativeWaymoReader", lambda root: reader)
    return reader, native, tmp_path


def test_native_adapter_indexes_only_selected_sequence_and_preserves_true_timestamps(native_reader):
    reader, native, root = native_reader
    adapter = create_dataset("waymo", root=root)
    assert adapter.metadata.splits == ("all", "train", "val")
    assert len(adapter.list_scenes()) == 2
    assert reader.header_calls == reader.frame_calls == []
    assert [scene.scene_id for scene in adapter.list_scenes(split="val")] == ["validation-context"]
    frames = adapter.list_frames("validation-context", modality=Modality.CAMERAS)
    assert [frame.timestamp_us for frame in frames] == [1_000_003, 1_104_525]
    assert reader.header_calls == ["validation-context"]
    assert not reader.frame_calls
    assert adapter.list_frames("validation-context", modality=Modality.CAMERAS) is frames
    loaded = adapter.load_cameras(frames[0].frame_id)
    assert reader.frame_calls == [("validation-context", 0, 1_000_003)]
    assert [camera.sensor_id for camera in loaded.cameras] == [slot.sensor_id for slot in camera_layout(adapter.metadata.sensors)]
    assert "Sign" in [item.class_id for item in adapter.metadata.classes]
    assert [(slot.row, slot.column) for slot in camera_layout(adapter.metadata.sensors)] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)]
    np.testing.assert_allclose(loaded.annotations.boxes[0].center, native.boxes[0].center)
    front = next(camera for camera in loaded.cameras if camera.sensor_id == "CAM_FRONT")
    left = next(camera for camera in loaded.cameras if camera.sensor_id == "CAM_FRONT_LEFT")
    native_front = next(camera for camera in native.cameras if camera.name == 1)
    np.testing.assert_allclose(front.T_reference_from_camera, np.linalg.inv(native.vehicle_to_world) @ native_front.vehicle_to_world @ native_front.extrinsic @ OPTICAL_TO_WAYMO)
    np.testing.assert_allclose(front.annotations.boxes[0].center, native.boxes[0].camera_synced_box.center)
    assert left.annotations is None  # No override: use the frame's reference-time boxes.
    assert front.image.dtype == np.uint8 and not front.image.flags.writeable
    # Native model consumes exactly the camera set reference used by boxes.
    world_points = np.array([[101., 210, 4.]])
    expected_uv, expected_depth = WaymoCameraProjection(native_front).project_world(world_points)
    reference_points = (world_points - native.vehicle_to_world[:3, 3]) @ native.vehicle_to_world[:3, :3]
    actual = front.projection_model.project_reference(reference_points)
    np.testing.assert_allclose(actual.uv, expected_uv, equal_nan=True)
    np.testing.assert_allclose(actual.depth, expected_depth)


def test_waymo_adapter_errors_are_contextual(native_reader):
    reader, _, root = native_reader
    adapter = create_dataset("waymo", root=root)
    with pytest.raises(DatasetError, match="needs root"):
        create_dataset("waymo", root=None)
    with pytest.raises(DatasetError, match="split"):
        adapter.list_scenes(split="mini_val")
    with pytest.raises(DatasetError, match="Unknown Waymo scene"):
        adapter.list_frames("missing", modality=Modality.CAMERAS)
    with pytest.raises(DatasetError, match="Unknown Waymo frame"):
        adapter.load_cameras("missing")
    reader.list_frame_headers = lambda context: (NativeFrameHeader(0, 5), NativeFrameHeader(1, 4))
    with pytest.raises(DatasetError, match="increase"):
        adapter.list_frames("validation-context", modality=Modality.CAMERAS)


def test_sparse_reader_indices_are_distinct_from_display_frame_positions(native_reader):
    reader, _, root = native_reader
    reader.list_frame_headers = lambda context: (NativeFrameHeader(2, 1_000_003), NativeFrameHeader(5, 1_104_525))
    adapter = create_dataset("waymo", root=root)
    frames = adapter.list_frames("validation-context", modality=Modality.CAMERAS)
    assert [frame.index for frame in frames] == [0, 1]
    assert frames[1].frame_id == "waymo:validation-context:1104525"
    adapter.load_cameras(frames[1].frame_id)
    assert reader.frame_calls == [("validation-context", 5, 1_104_525)]


@pytest.mark.parametrize("kwargs", [
    {"direction": 0}, {"direction": 6}, {"intrinsic": np.zeros(9)},
    {"velocity": [0, 0, np.nan, 0, 0, 0]},
])
def test_invalid_native_camera_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        WaymoCameraProjection(native_camera(**kwargs))
