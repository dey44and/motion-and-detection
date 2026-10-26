"""OpenPCDet Waymo fixtures require no SDK, TensorFlow, or dataset conversion."""

import pickle
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from threedviz.dataloader import DatasetError, SceneInfo, create_dataset
from threedviz.dataloader.waymo import DEFAULT_PROCESSED_TAG, WAYMO_CLASSES, NativeWaymoAdapter, WaymoAdapter
from threedviz.dataloader.shared import SharedPointCloudAdapter
from dataset_utils.waymo.loader import WaymoLoader


def _write_sequence(processed, context, *, indices=(0, 5), annotated=True):
    sequence = f"segment-{context}_with_camera_labels"
    directory = processed / sequence
    directory.mkdir(parents=True)
    records = []
    for index in indices:
        record = {
            "point_cloud": {"num_features": 5, "lidar_sequence": sequence, "sample_idx": index},
            "frame_id": f"{sequence}_{index:03d}",
            "metadata": {"context_name": context, "timestamp_micros": 1_000_000 + index * 100_000},
            "pose": np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]], dtype=np.float32),
        }
        if annotated:
            record["annos"] = {
                "name": np.array(["Vehicle", "Sign", "Custom"]),
                "obj_ids": np.array(["vehicle-id", "sign-id", "custom-id"]),
                "num_points_in_gt": np.array([2, 0, 1]),
                "gt_boxes_lidar": np.array([[3, 4, 5, 4, 2, 1.5, np.pi / 2, 12, 14],
                                            [8, 9, 10, .1, 1, 1, -.25, 0, 0],
                                            [1, 2, 3, 1, 1, 1, 0, 0, 0]]),
            }
        records.append(record)
        cloud = np.array([[index, 2, 3, 29.5, .25, -1], [4, 5, 6, 100, .5, 1]], dtype=np.float32)
        np.save(directory / f"{index:04d}.npy", cloud)
    metadata = directory / f"{sequence}.pkl"
    # Deliberately unsorted records ensure original indices drive point pairing.
    metadata.write_bytes(pickle.dumps(records[::-1], protocol=4))
    return sequence


@pytest.fixture
def dataset(tmp_path):
    parent = tmp_path / "waymo_1.4.2"
    workspace = parent / "waymo_pcdet_workspace"
    processed = workspace / DEFAULT_PROCESSED_TAG
    processed.mkdir(parents=True)
    alpha = _write_sequence(processed, "alpha")
    beta = _write_sequence(processed, "beta")
    gamma = _write_sequence(processed, "gamma", annotated=False)
    (processed / "_write_test").write_text("artifact")
    splits = workspace / "ImageSets"
    splits.mkdir()
    for name, members in {"train": [alpha], "train_1of16": [alpha], "train_full": [alpha, beta],
                          "val": [gamma], "test": ["segment-missing-test_with_camera_labels"]}.items():
        (splits / f"{name}.txt").write_text("".join(member + ".tfrecord\n" for member in members))
    return SimpleNamespace(parent=parent, workspace=workspace, processed=processed,
                           alpha=alpha, beta=beta, gamma=gamma)


def _records(dataset, sequence):
    path = dataset.processed / sequence / f"{sequence}.pkl"
    # Fixtures were generated locally above; the production adapter still uses
    # its restricted NumPy-only reader.
    return path, pickle.loads(path.read_bytes())


@pytest.mark.parametrize("root_name", ["parent", "workspace", "processed"])
def test_root_discovery_and_factory_registration(dataset, root_name):
    adapter = create_dataset("waymo", root=getattr(dataset, root_name))
    assert isinstance(adapter, SharedPointCloudAdapter)
    assert isinstance(adapter.loader, WaymoLoader)
    assert isinstance(adapter.loader.processed_adapter, WaymoAdapter)
    assert adapter.processed_root == dataset.processed
    assert [scene.name for scene in adapter.list_scenes()] == ["alpha", "beta", "gamma"]
    assert adapter.class_names == WAYMO_CLASSES
    assert not adapter.supports_point_timing and not adapter.supports_camera_rgb
    assert adapter.available_splits == ("all", "train", "val", "test", "train_full", "train_1of16")


def test_explicit_processed_tag_and_directory_override_default(dataset):
    alternate = dataset.workspace / "waymo_processed_data_v0_5_0_1of16"
    _write_sequence(alternate, "reduced")
    assert WaymoAdapter(root=dataset.parent).processed_root == dataset.processed
    assert WaymoAdapter(root=dataset.workspace, version=alternate.name).processed_root == alternate
    assert WaymoAdapter(root=alternate).list_scenes()[0].name == "reduced"


def test_train_prefers_full_membership_and_reduced_split_is_explicit(dataset):
    adapter = WaymoAdapter(root=dataset.workspace)
    assert [scene.token for scene in adapter.list_scenes_for_split("train")] == [dataset.alpha, dataset.beta]
    assert adapter.list_scenes_for_split("train_full") == adapter.list_scenes_for_split("train")
    assert [scene.token for scene in adapter.list_scenes_for_split("train_1of16")] == [dataset.alpha]
    assert [scene.token for scene in adapter.list_scenes_for_split("val")] == [dataset.gamma]
    with pytest.raises(DatasetError, match="No complete processed.*test"):
        adapter.list_scenes_for_split("test")
    (dataset.workspace / "ImageSets/train_full.txt").unlink()
    with pytest.raises(DatasetError, match="reduced train_1of16"):
        WaymoAdapter(root=dataset.workspace).list_scenes_for_split("train")


@pytest.mark.parametrize("prefix,suffix", [("validation_", "_with_camera_labels"), ("", ""), ("validation_", "")])
def test_split_membership_resolves_openpcdet_sequence_filename_variants(dataset, prefix, suffix):
    original = dataset.processed / dataset.gamma
    metadata, records = _records(dataset, dataset.gamma)
    new_name = f"{prefix}segment-gamma{suffix}"
    replacement = dataset.processed / new_name
    original.rename(replacement)
    (replacement / metadata.name).unlink()
    for record in records:
        index = record["point_cloud"]["sample_idx"]
        record["point_cloud"]["lidar_sequence"] = new_name
        record["frame_id"] = f"{new_name}_{index:03d}"
    (replacement / f"{new_name}.pkl").write_bytes(pickle.dumps(records))
    # ImageSets keeps the original unprefixed, camera-label filename.
    adapter = WaymoAdapter(root=dataset.parent)
    scenes = adapter.list_scenes_for_split("val")
    assert [scene.token for scene in scenes] == [new_name]
    assert adapter.load_frame(adapter.list_frames(new_name)[0].token).info.token == "waymo:gamma:1000000"


def test_reduced_training_comparison_normalizes_filename_variants_and_order(dataset):
    splits = dataset.workspace / "ImageSets"
    (splits / "train_full.txt").unlink()
    (splits / "train.txt").write_text(f"{dataset.alpha}.tfrecord\n{dataset.beta}.tfrecord\n")
    (splits / "train_1of16.txt").write_text("training_segment-beta.tfrecord\ntraining_segment-alpha.tfrecord\n")
    with pytest.raises(DatasetError, match="reduced train_1of16"):
        WaymoAdapter(root=dataset.workspace).list_scenes_for_split("train")


def test_original_frame_indices_timestamps_coordinates_and_gt_heading(dataset):
    adapter = WaymoAdapter(root=dataset.parent)
    frames = adapter.list_frames(dataset.alpha)
    assert [(item.index, item.timestamp_us) for item in frames] == [(0, 1_000_000), (1, 1_500_000)]
    assert frames[1].token == "waymo:alpha:1500000"
    frame = adapter.load_frame(frames[1].token)
    np.testing.assert_array_equal(frame.points, [[5, 2, 3], [4, 5, 6]])
    # Preserve raw intensity and both NLZ states; no training-time tanh/filter.
    np.testing.assert_array_equal(frame.intensity, [29.5, 100])
    assert frame.lidar_timestamp_us == 1_500_000
    np.testing.assert_array_equal(frame.reference_sensor_to_world,
                                  [[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]])
    assert [box.class_name for box in frame.boxes] == ["Vehicle", "Sign", "Custom"]
    assert frame.boxes[1].token == "sign-id"  # Zero-point GT is retained.
    np.testing.assert_allclose(frame.boxes[0].rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert "Custom" in adapter.class_names
    assert not frame.points.flags.writeable and not frame.reference_sensor_to_world.flags.writeable
    assert adapter.list_scenes()[0].frame_count == 2


def test_test_style_frame_without_ground_truth_and_direct_token_lookup(dataset):
    adapter = WaymoAdapter(root=dataset.workspace)
    frame = adapter.load_frame("waymo:gamma:1000000")
    assert frame.boxes == ()
    assert frame.info.index == 0 and len(frame.points) == 2


def test_seven_column_and_empty_annotations_are_supported(dataset):
    path, records = _records(dataset, dataset.alpha)
    records[1]["annos"]["gt_boxes_lidar"] = records[1]["annos"]["gt_boxes_lidar"][:, :7]
    records[0]["annos"] = {"name": np.array([], dtype=str), "gt_boxes_lidar": np.empty((0, 7))}
    path.write_bytes(pickle.dumps(records))
    adapter = WaymoAdapter(root=dataset.parent)
    first = adapter.load_frame("waymo:alpha:1000000")
    assert len(first.boxes) == 3
    np.testing.assert_allclose(first.boxes[0].rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert adapter.load_frame("waymo:alpha:1500000").boxes == ()


def test_camera_rgb_without_paired_records_reports_required_layout(dataset):
    adapter = WaymoAdapter(root=dataset.parent)
    frame = adapter.load_frame("waymo:alpha:1000000")
    with pytest.raises(DatasetError, match="original TFRecord"):
        adapter.prepare_frame(frame, camera_rgb=True)


def test_metadata_and_point_reads_are_lazy_and_sequence_cache_bounded(dataset, monkeypatch):
    from threedviz.dataloader import waymo

    reads = []
    original = waymo._NumpyMetadataUnpickler.load
    monkeypatch.setattr(waymo._NumpyMetadataUnpickler, "load", lambda self: (reads.append(1), original(self))[1])
    loads = []
    original_load = np.load
    monkeypatch.setattr(np, "load", lambda path, **kwargs: (loads.append((path, kwargs)), original_load(path, **kwargs))[1])
    adapter = WaymoAdapter(root=dataset.parent)
    adapter.list_scenes()
    adapter.list_scenes_for_split("val")
    assert reads == [] and loads == []
    adapter.list_frames(dataset.alpha)
    assert len(reads) == 1 and loads == []
    adapter.load_frame("waymo:alpha:1500000")
    assert len(reads) == 1 and len(loads) == 1
    assert loads[0][0].name == "0005.npy" and loads[0][1] == {"mmap_mode": "r", "allow_pickle": False}
    adapter.list_frames(dataset.beta)
    adapter.list_frames(dataset.gamma)
    assert len(adapter._sequence_cache) == 2
    adapter.load_frame("waymo:alpha:1000000")
    assert len(reads) == 4


def test_partial_sequence_is_reported_without_hiding_complete_sequences(dataset):
    partial = dataset.processed / "segment-partial_with_camera_labels"
    partial.mkdir()
    np.save(partial / "0000.npy", np.ones((1, 6), dtype=np.float32))
    adapter = WaymoAdapter(root=dataset.parent)
    assert len(adapter.list_scenes()) == 3
    assert len(adapter.warnings) == 1 and "partial" in adapter.warnings[0]
    assert str(partial / f"{partial.name}.pkl") in adapter.warnings[0]


@pytest.mark.parametrize("mutation,match", [
    (lambda record: record["point_cloud"].update(lidar_sequence="wrong"), "different lidar_sequence"),
    (lambda record: record["metadata"].update(context_name="wrong"), "mismatched context_name"),
    (lambda record: record["metadata"].update(timestamp_micros=1.5), "timestamp_micros"),
    (lambda record: record.update(frame_id="wrong"), "frame_id does not match"),
])
def test_metadata_pairing_validation(dataset, mutation, match):
    path, records = _records(dataset, dataset.alpha)
    mutation(records[0])
    path.write_bytes(pickle.dumps(records))
    with pytest.raises(DatasetError, match=match):
        WaymoAdapter(root=dataset.parent).list_frames(dataset.alpha)


@pytest.mark.parametrize("array", [np.ones((2, 5), np.float32), np.ones((2, 6), np.float64),
                                  np.full((2, 6), np.nan, np.float32)])
def test_malformed_cloud_rejected(dataset, array):
    np.save(dataset.processed / dataset.alpha / "0000.npy", array)
    with pytest.raises(DatasetError, match=r"float32 \(N,6\)"):
        WaymoAdapter(root=dataset.parent).load_frame("waymo:alpha:1000000")


def test_missing_cloud_bad_pose_and_unsupported_pickle_are_actionable(dataset):
    cloud = dataset.processed / dataset.alpha / "0000.npy"
    cloud.unlink()
    adapter = WaymoAdapter(root=dataset.parent)
    with pytest.raises(DatasetError, match="finish OpenPCDet preprocessing") as failure:
        adapter.load_frame("waymo:alpha:1000000")
    assert str(cloud) in str(failure.value)
    np.save(cloud, np.ones((1, 6), dtype=np.float32))
    path, records = _records(dataset, dataset.alpha)
    records[1]["pose"] = np.zeros((4, 4))
    path.write_bytes(pickle.dumps(records))
    with pytest.raises(DatasetError, match="vehicle-to-world pose"):
        WaymoAdapter(root=dataset.parent).load_frame("waymo:alpha:1000000")
    path.write_bytes(pickle.dumps(Path("unsupported-metadata-global")))
    with pytest.raises(DatasetError, match="Unsupported Waymo metadata pickle global"):
        WaymoAdapter(root=dataset.parent).list_frames(dataset.alpha)


def test_invalid_roots_splits_and_unknown_identifiers(dataset, tmp_path):
    with pytest.raises(DatasetError, match="requires a dataset root"):
        WaymoAdapter(root=None)
    with pytest.raises(DatasetError, match="does not exist"):
        WaymoAdapter(root=tmp_path / "missing")
    with pytest.raises(DatasetError, match="not found"):
        WaymoAdapter(root=tmp_path)
    adapter = WaymoAdapter(root=dataset.parent)
    with pytest.raises(DatasetError, match="Unknown Waymo split"):
        adapter.list_scenes_for_split("mini_val")
    with pytest.raises(DatasetError, match="Unknown Waymo scene"):
        adapter.list_frames("missing")
    with pytest.raises(DatasetError, match="Unknown Waymo frame"):
        adapter.load_frame("waymo:unknown:1")


def _native_fixture(dataset, monkeypatch, *, mismatch=False):
    raw = dataset.parent / "waymo_og_format_unpacked" / "training"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{dataset.alpha}.tfrecord").write_bytes(b"fixture-record-decoded-by-fake-backend")
    calls = []
    module = ModuleType("dataset_utils.waymo")

    class Reader:
        def __init__(self, root):
            calls.append(("open", root))

        def read_frame(self, context, index, timestamp_us=None):
            calls.append(("frame", context, index, timestamp_us))
            return object()

    xyz = np.array([[5, 2, 3], [4, 5, 6]], dtype=np.float64)
    if mismatch:
        xyz = xyz[::-1]
    module.NativeWaymoReader = Reader
    module.decode_lidar = lambda frame: SimpleNamespace(points=xyz)
    module.colorize_points = lambda frame, points: (np.array([[1., 0, 0], [0, 1., 0]]), np.array([True, False]))
    parent = ModuleType("dataset_utils")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "dataset_utils", parent)
    monkeypatch.setitem(sys.modules, "dataset_utils.waymo", module)
    return calls


def test_camera_rgb_lazy_native_pairing_and_invalid_point_correspondence(dataset, monkeypatch):
    calls = _native_fixture(dataset, monkeypatch)
    adapter = WaymoAdapter(root=dataset.processed)
    assert adapter.supports_camera_rgb and calls == []
    frame = adapter.load_frame("waymo:alpha:1500000")
    assert frame.rgb_colors is None and calls == []
    prepared = adapter.prepare_frame(frame, camera_rgb=True)
    assert calls[1] == ("frame", "alpha", 5, 1_500_000)
    assert prepared.points is frame.points and prepared.boxes is frame.boxes
    np.testing.assert_array_equal(prepared.rgb_colors, [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(prepared.rgb_valid_mask, [True, False])
    assert adapter.prepare_frame(prepared, camera_rgb=True) is prepared
    with pytest.raises(DatasetError, match="acquisition-time"):
        adapter.prepare_frame(frame, point_timing=True)
    _native_fixture(dataset, monkeypatch, mismatch=True)
    mismatched = WaymoAdapter(root=dataset.parent)
    frame = mismatched.load_frame(frame.info.token)
    with pytest.raises(DatasetError, match="correspondence differs"):
        mismatched.prepare_frame(frame, camera_rgb=True)


def test_legacy_native_constructor_headers_cloud_and_camera_rgb(tmp_path, monkeypatch):
    raw = tmp_path / "waymo_og_format_unpacked"
    (raw / "testing").mkdir(parents=True)
    record_path = raw / "testing/segment-native_with_camera_labels.tfrecord"
    record_path.write_bytes(b"native-fixture")
    module = ModuleType("dataset_utils.waymo")
    calls = []
    pointset = SimpleNamespace(points=np.array([[1., 2, 3]]), intensity=np.array([29.5]))
    native = SimpleNamespace(vehicle_to_world=np.eye(4), boxes=(), points=lambda: pointset)

    class Reader:
        def __init__(self, root):
            self.native_root = raw

        def list_sequences(self):
            return (SimpleNamespace(context_name="native", path=record_path, split="testing"),)

        def list_frame_headers(self, context):
            calls.append(("headers", context))
            return (SimpleNamespace(frame_index=0, timestamp_us=100), SimpleNamespace(frame_index=1, timestamp_us=200))

        def read_frame(self, context, index, timestamp_us=None):
            calls.append(("frame", context, index, timestamp_us))
            return native

    module.NativeWaymoReader = Reader
    module.colorize_points = lambda frame, points: (np.array([[.1, .2, .3]]), np.array([True]))
    parent = ModuleType("dataset_utils")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "dataset_utils", parent)
    monkeypatch.setitem(sys.modules, "dataset_utils.waymo", module)
    adapter = NativeWaymoAdapter(root=raw)
    assert isinstance(adapter, NativeWaymoAdapter) and calls == []
    assert adapter.available_splits == ("all", "test")
    assert len(adapter.list_scenes_for_split("test")) == 1
    with pytest.raises(DatasetError, match="No native Waymo"):
        adapter.list_scenes_for_split("val")
    frames = adapter.list_frames(adapter.list_scenes()[0].token)
    assert [info.token for info in frames] == ["waymo:native:100", "waymo:native:200"]
    frame = adapter.load_frame(frames[1].token)
    assert calls[-1] == ("frame", "native", 1, 200)
    assert frame.boxes == () and frame.lidar_timestamp_us == 200
    np.testing.assert_array_equal(frame.points, [[1, 2, 3]])
    prepared = adapter.prepare_frame(frame, camera_rgb=True)
    np.testing.assert_allclose(prepared.rgb_colors, [[.1, .2, .3]])
    assert prepared.rgb_valid_mask.all() and not adapter.supports_point_timing


def test_factory_never_falls_back_from_malformed_processed_metadata(dataset, monkeypatch):
    _native_fixture(dataset, monkeypatch)
    metadata = dataset.processed / dataset.alpha / f"{dataset.alpha}.pkl"
    metadata.write_bytes(b"invalid-pickle")
    adapter = create_dataset("waymo", root=dataset.parent)
    assert isinstance(adapter, SharedPointCloudAdapter)
    assert isinstance(adapter.loader.processed_adapter, WaymoAdapter)
    with pytest.raises(DatasetError, match="Cannot read Waymo sequence metadata"):
        adapter.list_frames("alpha")


@pytest.mark.parametrize("timestamps,match", [((200, 100), "increase chronologically"), ((100, 100), "Duplicate timestamp")])
def test_native_malformed_timestamps_do_not_publish_partial_indexes(timestamps, match):
    adapter = NativeWaymoAdapter.__new__(NativeWaymoAdapter)
    adapter._sequences = {"segment-native": SimpleNamespace(context_name="native")}
    headers = [SimpleNamespace(frame_index=index, timestamp_us=timestamp) for index, timestamp in enumerate(timestamps)]
    adapter._reader = SimpleNamespace(list_frame_headers=lambda context: headers)
    adapter._frames_by_scene, adapter._frame_locations = {}, {}
    adapter._scenes = (SceneInfo("segment-native", "native"),)
    with pytest.raises(DatasetError, match=match):
        adapter.list_frames("segment-native")
    assert adapter._frame_locations == {} and adapter._frames_by_scene == {}
    # A corrected reader result can be indexed without stale first-frame keys.
    headers[:] = [SimpleNamespace(frame_index=0, timestamp_us=100), SimpleNamespace(frame_index=1, timestamp_us=200)]
    assert [frame.timestamp_us for frame in adapter.list_frames("segment-native")] == [100, 200]
