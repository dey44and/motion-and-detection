"""Lazy reader for OpenPCDet's per-sequence Waymo NPY/NumPy metadata format.

Only the selected sequence metadata is decoded; aggregate infos pickles and
the TensorFlow/Waymo SDK are unnecessary. Points and boxes are expressed in
the reference vehicle frame, whose world transform is stored in ``pose``.

Format reference: OpenPCDet ``pcdet/datasets/waymo/waymo_utils.py``.
"""

from __future__ import annotations

import importlib
import pickle
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DatasetError
from ..models import BoundingBox3D, PointCloudFrame as FrameData, FrameInfo, SceneInfo

WAYMO_CLASSES = ("Vehicle", "Pedestrian", "Sign", "Cyclist")
DEFAULT_PROCESSED_TAG = "waymo_processed_data_v0_5_0_full_with_corr"
_SEQUENCE_CACHE_SIZE = 2
_MAX_SEQUENCE_METADATA_BYTES = 256 * 1024 * 1024


class _NumpyMetadataUnpickler(pickle.Unpickler):
    """Allow only NumPy reconstruction primitives, never arbitrary globals."""

    def find_class(self, module: str, name: str):
        if (module, name) in {("numpy", "ndarray"), ("numpy", "dtype")}:
            return getattr(np, name)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {"_reconstruct", "scalar"}:
            core = "numpy._core" if hasattr(np, "_core") else "numpy.core"
            return getattr(importlib.import_module(core + ".multiarray"), name)
        if module in {"numpy.core.numeric", "numpy._core.numeric"} and name == "_frombuffer":
            core = "numpy._core" if hasattr(np, "_core") else "numpy.core"
            return getattr(importlib.import_module(core + ".numeric"), name)
        raise DatasetError(
            f"Unsupported Waymo metadata pickle global {module}.{name}. "
            "Use OpenPCDet per-sequence metadata containing NumPy arrays and ordinary Python values."
        )


def _context_name(sequence_name: str) -> str:
    result = sequence_name.split("segment-", 1)[-1]
    return result.removesuffix("_with_camera_labels")


def _is_sequence(name: str) -> bool:
    return name.startswith(("segment-", "training_segment-", "validation_segment-", "testing_segment-"))


def _integer(value: Any, context: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise DatasetError(f"Invalid {context}: expected a nonnegative integer.")
    return int(value)


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _pose(value: Any, context: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetError(f"Invalid {context}: expected a finite rigid 4x4 vehicle-to-world pose.") from exc
    if (
        matrix.shape != (4, 4) or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)
    ):
        raise DatasetError(f"Invalid {context}: expected a finite rigid 4x4 vehicle-to-world pose.")
    return _readonly(matrix)


@dataclass(frozen=True)
class _SequenceFrame:
    info: FrameInfo
    sample_idx: int
    record: dict[str, Any]


class WaymoAdapter:
    """Read existing OpenPCDet Waymo preprocessing without changing its data.

    ``root`` may be the dataset parent, ``waymo_pcdet_workspace``, or an explicit
    processed-data directory. The full processed tag is preferred automatically;
    an explicit directory or non-nuScenes ``version`` selects another tag.
    ``warnings`` reports skipped partial sequence directories.
    """

    def __init__(self, *, root: str | Path | None, version: str = "v1.0-mini"):
        if root is None:
            raise DatasetError("Waymo requires a dataset root: choose waymo_pcdet_workspace or its processed-data directory.")
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise DatasetError(f"Waymo dataset directory does not exist: {self.root}")
        self.processed_root, self.workspace_root = self._discover(self.root, version)
        self.version = self.processed_root.name
        self._sequence_cache: OrderedDict[str, tuple[_SequenceFrame, ...]] = OrderedDict()
        self._frames_by_scene: dict[str, tuple[FrameInfo, ...]] = {}
        self._frame_locations: dict[str, tuple[str, int]] = {}
        self._additional_classes: set[str] = set()
        self._split_members = self._read_splits()

        paths = {}
        partial = []
        try:
            directories = sorted(self.processed_root.iterdir(), key=lambda path: path.name)
            for directory in directories:
                if not directory.is_dir() or not _is_sequence(directory.name):
                    continue
                metadata = directory / f"{directory.name}.pkl"
                if metadata.is_file():
                    paths[directory.name] = metadata
                else:
                    partial.append(str(metadata))
        except OSError as exc:
            raise DatasetError(f"Cannot inspect Waymo processed directory {self.processed_root}: {exc}") from exc
        if not paths:
            detail = f" Missing sequence metadata, for example {partial[0]}." if partial else ""
            raise DatasetError(
                f"No complete OpenPCDet Waymo sequences found in {self.processed_root}.{detail} "
                "Each segment directory needs its matching segment-name.pkl and frame NPY files."
            )
        self._metadata_paths = paths
        self._context_to_scene = {}
        for name in paths:
            context = _context_name(name)
            if context in self._context_to_scene:
                raise DatasetError(f"Multiple processed Waymo sequences share context {context!r}.")
            self._context_to_scene[context] = name
        self._scenes = tuple(SceneInfo(
            token=name, name=_context_name(name),
            description=f"Waymo OpenPCDet sequence · {self.processed_root.name}. Frame count is indexed when selected.",
        ) for name in paths)
        self.warnings = tuple([
            f"Skipped {len(partial)} partial Waymo sequence directories without per-sequence metadata. "
            f"First missing file: {partial[0]}. Choose a complete processed directory or finish preprocessing."
        ] if partial else [])
        self._native_root, self._native_contexts = self._discover_native_records()
        self._native_reader = None

    def _discover_native_records(self) -> tuple[Path | None, frozenset[str]]:
        """Find paired raw records by filenames only, without decoding frames."""
        for ancestor in (self.root, *self.root.parents):
            candidate = ancestor if ancestor.name == "waymo_og_format_unpacked" else ancestor / "waymo_og_format_unpacked"
            if not candidate.is_dir():
                continue
            contexts = set()
            try:
                for split in ("training", "validation", "testing"):
                    folder = candidate / split
                    if folder.is_dir():
                        contexts.update(_context_name(path.stem) for path in folder.glob("*.tfrecord"))
            except OSError as exc:
                raise DatasetError(f"Cannot inspect native Waymo record directory {candidate}: {exc}") from exc
            if contexts:
                return candidate, frozenset(contexts)
        return None, frozenset()

    @staticmethod
    def _discover(root: Path, version: str) -> tuple[Path, Path]:
        def sequence_directory(path):
            return any(child.is_dir() and _is_sequence(child.name) for child in path.iterdir())

        try:
            if sequence_directory(root):
                return root, root.parent
            workspace = root / "waymo_pcdet_workspace" if (root / "waymo_pcdet_workspace").is_dir() else root
            requested = version if isinstance(version, str) and version and not version.startswith("v1.0-") else None
            if requested is not None:
                if Path(requested).name != requested or requested in {".", ".."}:
                    raise DatasetError("Waymo processed version must be a directory tag, not a path.")
                selected = workspace / requested
                if selected.is_dir():
                    return selected, workspace
                raise DatasetError(f"Requested Waymo processed directory does not exist: {selected}")
            preferred = workspace / DEFAULT_PROCESSED_TAG
            if preferred.is_dir():
                return preferred, workspace
            candidates = sorted(path for path in workspace.iterdir()
                                if path.is_dir() and path.name.startswith("waymo_processed_data_")
                                and "gt_database" not in path.name)
            if len(candidates) == 1:
                return candidates[0], workspace
            if len(candidates) > 1:
                raise DatasetError(
                    f"Several Waymo processed directories exist in {workspace}; choose one as the dataset root "
                    "or set its directory tag as the version."
                )
        except OSError as exc:
            raise DatasetError(f"Cannot inspect Waymo dataset root {root}: {exc}") from exc
        raise DatasetError(
            f"OpenPCDet Waymo preprocessing not found at {root}. Choose waymo_pcdet_workspace, "
            f"its {DEFAULT_PROCESSED_TAG} directory, or the dataset parent. "
            "Legacy Waymo/ frame pickles and raw TFRecords require a different adapter."
        )

    def _read_splits(self) -> dict[str, tuple[str, ...]]:
        result = {}
        split_root = self.workspace_root / "ImageSets"
        for name in ("train", "train_full", "train_1of16", "val", "test"):
            path = split_root / f"{name}.txt"
            if not path.is_file():
                continue
            try:
                names = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if Path(line).name != line:
                        raise DatasetError(f"Invalid sequence name in Waymo split file {path}: {line!r}")
                    context = _context_name(line.removesuffix(".tfrecord"))
                    if not context:
                        raise DatasetError(f"Invalid empty context in Waymo split file {path}: {line!r}")
                    names.append(context)
                if len(set(names)) != len(names):
                    raise DatasetError(f"Duplicate sequence names in Waymo split file {path}.")
                result[name] = tuple(names)
            except (OSError, UnicodeError) as exc:
                raise DatasetError(f"Cannot read Waymo split file {path}: {exc}") from exc
        return result

    @property
    def supports_point_timing(self) -> bool:
        return False

    @property
    def class_names(self) -> tuple[str, ...]:
        return WAYMO_CLASSES + tuple(sorted(self._additional_classes.difference(WAYMO_CLASSES)))

    @property
    def supports_camera_rgb(self) -> bool:
        return bool(self._native_contexts.intersection(self._context_to_scene))

    @property
    def available_splits(self) -> tuple[str, ...]:
        return ("all",) + tuple(name for name in ("train", "val", "test", "train_full", "train_1of16")
                                if name in self._split_members or name == "train" and "train_full" in self._split_members)

    def list_scenes(self) -> tuple[SceneInfo, ...]:
        return self._scenes

    def list_scenes_for_split(self, split: str = "all") -> tuple[SceneInfo, ...]:
        if not isinstance(split, str):
            raise DatasetError("Waymo split must be all, train, val, test, train_full or train_1of16.")
        name = split.strip().lower()
        if name == "all":
            return self._scenes
        if name not in {"train", "val", "test", "train_full", "train_1of16"}:
            raise DatasetError(f"Unknown Waymo split {split!r}. Choose all, train, val, test, train_full or train_1of16.")
        if name == "train" and "train_full" in self._split_members:
            name = "train_full"
        elif (name == "train" and "train" in self._split_members and "train_1of16" in self._split_members
              and set(self._split_members["train"]) == set(self._split_members["train_1of16"])):
            raise DatasetError(
                "Waymo train.txt contains the reduced train_1of16 selection and train_full.txt is missing. "
                "Choose train_1of16 explicitly or restore the full training split file."
            )
        if name not in self._split_members:
            raise DatasetError(f"Waymo split file is missing: {self.workspace_root / 'ImageSets' / (name + '.txt')}")
        requested = set(self._split_members[name])
        selected = tuple(scene for scene in self._scenes if scene.name in requested)
        if not selected:
            raise DatasetError(
                f"No complete processed Waymo sequences for split {split!r} are installed in {self.processed_root}. "
                "Check the processed directory and ImageSets split files."
            )
        return selected

    def _load_sequence(self, scene_token: str) -> tuple[_SequenceFrame, ...]:
        if scene_token in self._sequence_cache:
            self._sequence_cache.move_to_end(scene_token)
            return self._sequence_cache[scene_token]
        path = self._metadata_paths.get(scene_token)
        if path is None:
            raise DatasetError(f"Unknown Waymo scene {scene_token!r}.")
        try:
            if path.stat().st_size > _MAX_SEQUENCE_METADATA_BYTES:
                raise DatasetError(f"Waymo per-sequence metadata exceeds 256 MiB: {path}. Use segment metadata, not an aggregate infos pickle.")
            with path.open("rb") as stream:
                records = _NumpyMetadataUnpickler(stream).load()
        except DatasetError:
            raise
        except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError, OverflowError) as exc:
            raise DatasetError(f"Cannot read Waymo sequence metadata {path}: {exc}") from exc
        if not isinstance(records, (list, tuple)) or not records:
            raise DatasetError(f"Invalid Waymo metadata {path}: expected a nonempty list of frame records.")
        indexed = []
        seen_indices, seen_timestamps = set(), set()
        context_name = _context_name(scene_token)
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("point_cloud"), dict) or not isinstance(record.get("metadata"), dict):
                raise DatasetError(f"Invalid Waymo frame in {path}: expected point_cloud and metadata dictionaries.")
            cloud, metadata = record["point_cloud"], record["metadata"]
            if cloud.get("lidar_sequence") != scene_token:
                raise DatasetError(f"Waymo frame in {path} references a different lidar_sequence: {cloud.get('lidar_sequence')!r}.")
            if metadata.get("context_name") != context_name:
                raise DatasetError(f"Waymo frame in {path} has a mismatched context_name: {metadata.get('context_name')!r}.")
            sample_idx = _integer(cloud.get("sample_idx"), "Waymo original sample_idx")
            timestamp = _integer(metadata.get("timestamp_micros"), "Waymo timestamp_micros")
            if sample_idx in seen_indices or timestamp in seen_timestamps:
                raise DatasetError(f"Duplicate Waymo sample_idx or timestamp_micros in {path}.")
            seen_indices.add(sample_idx)
            seen_timestamps.add(timestamp)
            frame_id = record.get("frame_id")
            if frame_id is not None and frame_id != f"{scene_token}_{sample_idx:03d}":
                raise DatasetError(f"Waymo frame_id does not match its sequence/sample_idx in {path}: {frame_id!r}.")
            indexed.append((sample_idx, timestamp, record))
        indexed.sort(key=lambda item: item[0])
        if any(right[1] <= left[1] for left, right in zip(indexed, indexed[1:])):
            raise DatasetError(f"Waymo timestamps do not increase with original sample indices in {path}.")
        frames = tuple(_SequenceFrame(
            FrameInfo(f"waymo:{context_name}:{timestamp}", scene_token, display_index, timestamp), sample_idx, record
        ) for display_index, (sample_idx, timestamp, record) in enumerate(indexed))
        self._frames_by_scene[scene_token] = tuple(frame.info for frame in frames)
        self._frame_locations.update((frame.info.token, (scene_token, frame.sample_idx)) for frame in frames)
        self._scenes = tuple(replace(scene, frame_count=len(frames)) if scene.token == scene_token else scene for scene in self._scenes)
        self._sequence_cache[scene_token] = frames
        if len(self._sequence_cache) > _SEQUENCE_CACHE_SIZE:
            self._sequence_cache.popitem(last=False)
        return frames

    def list_frames(self, scene_token: str) -> tuple[FrameInfo, ...]:
        if scene_token not in self._frames_by_scene:
            self._load_sequence(scene_token)
        return self._frames_by_scene[scene_token]

    def _boxes(self, record: dict[str, Any], frame_token: str) -> tuple[BoundingBox3D, ...]:
        annotations = record.get("annos")
        if annotations is None:
            return ()
        if not isinstance(annotations, dict):
            raise DatasetError(f"Invalid Waymo annotations for {frame_token}: expected a dictionary.")
        try:
            values = np.asarray(annotations.get("gt_boxes_lidar"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DatasetError(f"Invalid Waymo gt_boxes_lidar for {frame_token}.") from exc
        if values.ndim != 2 or values.shape[1] not in (7, 9) or not np.isfinite(values).all() or np.any(values[:, 3:6] <= 0):
            raise DatasetError(f"Invalid Waymo gt_boxes_lidar for {frame_token}: expected finite (N,7) or (N,9) XYZ/LWH/yaw boxes with positive dimensions.")
        names = np.asarray(annotations.get("name", []))
        identifiers = np.asarray(annotations.get("obj_ids", [f"{frame_token}:box:{index}" for index in range(len(values))]))
        if names.shape != (len(values),) or identifiers.shape != (len(values),):
            raise DatasetError(f"Waymo annotation names/obj_ids must match gt_boxes_lidar for {frame_token}.")
        boxes = []
        for value, name, identifier in zip(values, names, identifiers):
            if not isinstance(name, str) or not name.strip():
                raise DatasetError(f"Invalid Waymo annotation class for {frame_token}: {name!r}.")
            cosine, sine = np.cos(value[6]), np.sin(value[6])
            rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
            boxes.append(BoundingBox3D(_readonly(value[:3]), _readonly(value[3:6]), _readonly(rotation), str(name), str(identifier)))
            self._additional_classes.add(str(name))
        return tuple(boxes)

    def load_frame(self, frame_token: str) -> FrameData:
        return self._load_frame_data(frame_token)

    def _load_frame_data(self, frame_token: str, *, include_attributes: bool = False):
        location = self._frame_locations.get(frame_token)
        if location is None and isinstance(frame_token, str) and frame_token.startswith("waymo:"):
            context = frame_token[len("waymo:"):].rsplit(":", 1)[0]
            scene_token = self._context_to_scene.get(context)
            if scene_token is not None:
                self._load_sequence(scene_token)
                location = self._frame_locations.get(frame_token)
        if location is None:
            raise DatasetError(f"Unknown Waymo frame {frame_token!r}.")
        scene_token, sample_idx = location
        frame = next(item for item in self._load_sequence(scene_token) if item.sample_idx == sample_idx)
        path = self.processed_root / scene_token / f"{sample_idx:04d}.npy"
        try:
            scan = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, TypeError) as exc:
            raise DatasetError(f"Cannot read Waymo point cloud {path}: {exc}. Restore this frame or finish OpenPCDet preprocessing.") from exc
        if not isinstance(scan, np.ndarray):
            if hasattr(scan, "close"):
                scan.close()
            raise DatasetError(f"Invalid Waymo point cloud {path}: expected a NumPy array, not an archive.")
        if scan.ndim != 2 or scan.shape[1] != 6 or scan.dtype.kind != "f" or scan.dtype.itemsize != 4 or not np.isfinite(scan).all():
            raise DatasetError(f"Invalid Waymo point cloud {path}: expected finite float32 (N,6) XYZ/intensity/elongation/NLZ values.")
        boxes = self._boxes(frame.record, frame_token)
        result = FrameData(
            info=frame.info, points=_readonly(scan[:, :3]), boxes=boxes,
            intensity=_readonly(scan[:, 3]), lidar_timestamp_us=frame.info.timestamp_us,
            reference_sensor_to_world=_pose(frame.record.get("pose"), f"Waymo pose for {frame_token}"),
        )

        if include_attributes:
            return result, {"intensity": result.intensity, "elongation": _readonly(scan[:, 4]), "nlz": _readonly(scan[:, 5])}
        return result

    def prepare_frame(self, frame: FrameData, *, camera_rgb: bool = False, point_timing: bool = False) -> FrameData:
        if point_timing:
            raise DatasetError("Waymo acquisition-time point display is not available for this processed-data adapter.")
        if not camera_rgb or frame.rgb_colors is not None:
            return frame
        location = self._frame_locations.get(frame.info.token)
        if location is None:
            raise DatasetError("Cannot match this frame to the selected Waymo processed dataset.")
        scene_token, sample_idx = location
        context = _context_name(scene_token)
        if self._native_root is None or context not in self._native_contexts:
            raise DatasetError(
                f"Camera RGB needs the original TFRecord for Waymo context {context}. "
                "Keep waymo_og_format_unpacked/{training,validation,testing} beside waymo_pcdet_workspace."
            )
        try:
            from dataset_utils.waymo import NativeWaymoReader, colorize_points, decode_lidar
            if self._native_reader is None:
                self._native_reader = NativeWaymoReader(self._native_root)
            native_frame = self._native_reader.read_frame(context, sample_idx, timestamp_us=frame.info.timestamp_us)
            native_points = decode_lidar(native_frame)
            points = np.asarray(native_points.points)
            if points.shape != frame.points.shape or not np.allclose(points, frame.points, atol=.02, rtol=0):
                raise DatasetError(
                    "Waymo processed/native point correspondence differs in count, order or XYZ coordinates "
                    f"for context {context}, original frame {sample_idx}. Camera colors were not attached. "
                    "Use matching all-five-LiDAR, both-return preprocessing; point filtering or reordering needs an explicit mapping."
                )
            colors, valid = colorize_points(native_frame, native_points)
        except DatasetError:
            raise
        except ImportError as exc:
            raise DatasetError("Waymo Camera RGB requires the shared dataset-utils package; install -e './dataset_utils' with the viewer.") from exc
        except Exception as exc:
            raise DatasetError(f"Cannot prepare Waymo Camera RGB for context {context}, original frame {sample_idx}: {exc}") from exc
        colors, valid = np.asarray(colors), np.asarray(valid)
        if (colors.shape != frame.points.shape or not np.isfinite(colors).all()
                or np.any((colors < 0) | (colors > 1)) or valid.shape != (len(frame.points),) or valid.dtype != np.bool_):
            raise DatasetError("Native Waymo camera colors returned invalid RGB values or a mismatched validity mask.")
        return replace(frame, rgb_colors=_readonly(colors), rgb_valid_mask=_readonly(valid))


class NativeWaymoAdapter:
    """Read an explicitly selected native TFRecord root, including test data.

    The usual processed-data path stays the default when both representations
    exist. This adapter shares the native reader already used for Camera RGB.
    """

    def __init__(self, *, root: str | Path | None, version: str = "v1.0-mini"):
        if root is None:
            raise DatasetError("Waymo requires a dataset root.")
        try:
            from dataset_utils.waymo import NativeWaymoReader
            self._reader = NativeWaymoReader(root)
        except ImportError as exc:
            raise DatasetError("Native Waymo viewing requires dataset-utils; install -e './dataset_utils' with the viewer.") from exc
        except Exception as exc:
            raise DatasetError(f"Cannot open native Waymo records: {exc}") from exc
        self.root = self._reader.native_root
        sequences = self._reader.list_sequences()
        if not sequences:
            raise DatasetError(f"No native Waymo TFRecord sequences found at {self.root}.")
        self._sequences = {sequence.path.stem: sequence for sequence in sequences}
        self._context_to_scene = {sequence.context_name: sequence.path.stem for sequence in sequences}
        self._scenes = tuple(SceneInfo(token=token, name=sequence.context_name,
                                      description=f"Native Waymo TFRecord · {sequence.split}.")
                             for token, sequence in self._sequences.items())
        self._frames_by_scene: dict[str, tuple[FrameInfo, ...]] = {}
        self._frame_locations: dict[str, tuple[str, int]] = {}
        self._native_cache = OrderedDict()
        self._additional_classes = set()
        self.warnings = ()

    @property
    def supports_point_timing(self) -> bool:
        return False

    @property
    def class_names(self) -> tuple[str, ...]:
        return WAYMO_CLASSES + tuple(sorted(self._additional_classes.difference(WAYMO_CLASSES)))

    @property
    def supports_camera_rgb(self) -> bool:
        return True

    @property
    def available_splits(self) -> tuple[str, ...]:
        present = {sequence.split for sequence in self._sequences.values()}
        return ("all",) + tuple(short for short, full in (("train", "training"), ("val", "validation"), ("test", "testing"))
                                if full in present)

    def list_scenes(self) -> tuple[SceneInfo, ...]:
        return self._scenes

    def list_scenes_for_split(self, split: str = "all") -> tuple[SceneInfo, ...]:
        name = split.strip().lower() if isinstance(split, str) else ""
        if name == "all":
            return self._scenes
        full = {"train": "training", "val": "validation", "test": "testing"}.get(name)
        if full is None:
            raise DatasetError(f"Unknown native Waymo split {split!r}; choose all, train, val or test.")
        scenes = tuple(scene for scene in self._scenes if self._sequences[scene.token].split == full)
        if not scenes:
            raise DatasetError(f"No native Waymo {split!r} sequences are installed at {self.root}.")
        return scenes

    def list_frames(self, scene_token: str) -> tuple[FrameInfo, ...]:
        if scene_token not in self._sequences:
            raise DatasetError(f"Unknown native Waymo scene {scene_token!r}.")
        if scene_token not in self._frames_by_scene:
            sequence = self._sequences[scene_token]
            try:
                headers = self._reader.list_frame_headers(sequence.context_name)
            except Exception as exc:
                raise DatasetError(f"Cannot index native Waymo sequence {sequence.context_name}: {exc}") from exc
            frames, locations = [], {}
            for index, header in enumerate(headers):
                original = _integer(header.frame_index, "native Waymo frame index")
                timestamp = _integer(header.timestamp_us, "native Waymo timestamp")
                info = FrameInfo(f"waymo:{sequence.context_name}:{timestamp}", scene_token, index, timestamp)
                if info.token in self._frame_locations or info.token in locations:
                    raise DatasetError(f"Duplicate timestamp in native Waymo sequence {sequence.context_name}.")
                if frames and timestamp <= frames[-1].timestamp_us:
                    raise DatasetError(f"Native Waymo timestamps must increase chronologically in sequence {sequence.context_name}.")
                locations[info.token] = (scene_token, original)
                frames.append(info)
            self._frame_locations.update(locations)
            self._frames_by_scene[scene_token] = tuple(frames)
            self._scenes = tuple(replace(scene, frame_count=len(frames)) if scene.token == scene_token else scene for scene in self._scenes)
        return self._frames_by_scene[scene_token]

    def _read_native(self, token):
        if token not in self._frame_locations and isinstance(token, str) and token.startswith("waymo:"):
            context = token[len("waymo:"):].rsplit(":", 1)[0]
            scene = self._context_to_scene.get(context)
            if scene is not None:
                self.list_frames(scene)
        if token not in self._frame_locations:
            raise DatasetError(f"Unknown native Waymo frame {token!r}.")
        scene, original = self._frame_locations[token]
        info = next(info for info in self._frames_by_scene[scene] if info.token == token)
        if token not in self._native_cache:
            try:
                frame = self._reader.read_frame(self._sequences[scene].context_name, original, timestamp_us=info.timestamp_us)
                self._native_cache[token] = frame, frame.points()
            except Exception as exc:
                raise DatasetError(f"Cannot decode native Waymo frame {token}: {exc}") from exc
            if len(self._native_cache) > 2:
                self._native_cache.popitem(last=False)
        self._native_cache.move_to_end(token)
        return info, *self._native_cache[token]

    def load_frame(self, frame_token: str) -> FrameData:
        info, native, points = self._read_native(frame_token)
        boxes = []
        for box in native.boxes:
            cosine, sine = np.cos(box.heading), np.sin(box.heading)
            rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
            boxes.append(BoundingBox3D(_readonly(box.center), _readonly(box.size), _readonly(rotation), box.class_name, box.identifier))
            self._additional_classes.add(box.class_name)
        return FrameData(info, _readonly(points.points), tuple(boxes), intensity=_readonly(points.intensity),
                         lidar_timestamp_us=info.timestamp_us,
                         reference_sensor_to_world=_pose(native.vehicle_to_world, f"native Waymo pose for {frame_token}"))

    def prepare_frame(self, frame: FrameData, *, camera_rgb: bool = False, point_timing: bool = False) -> FrameData:
        if point_timing:
            raise DatasetError("Waymo acquisition-time point display is not available in this viewer.")
        if not camera_rgb or frame.rgb_colors is not None:
            return frame
        _, native, points = self._read_native(frame.info.token)
        try:
            from dataset_utils.waymo import colorize_points
            colors, valid = colorize_points(native, points)
        except Exception as exc:
            raise DatasetError(f"Cannot prepare native Waymo Camera RGB: {exc}") from exc
        colors, valid = np.asarray(colors), np.asarray(valid)
        if (colors.shape != frame.points.shape or not np.isfinite(colors).all()
                or np.any((colors < 0) | (colors > 1)) or valid.shape != (len(frame.points),) or valid.dtype != np.bool_):
            raise DatasetError("Native Waymo camera colors returned invalid RGB values or a mismatched validity mask.")
        return replace(frame, rgb_colors=_readonly(colors), rgb_valid_mask=_readonly(valid))


def open_waymo(*, root: str | Path | None, version: str = "v1.0-mini") -> WaymoAdapter | NativeWaymoAdapter:
    """Prefer processed data; select native data only when no processed layout exists.

    A present but malformed processed layout reports its error rather than
    silently switching to a different representation.
    """
    if root is None:
        return WaymoAdapter(root=root, version=version)
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        return WaymoAdapter(root=root, version=version)
    try:
        workspace = path / "waymo_pcdet_workspace" if (path / "waymo_pcdet_workspace").is_dir() else path
        processed = (path.name.startswith("waymo_processed_data_")
                     or any(child.is_dir() and (_is_sequence(child.name) or child.name.startswith("waymo_processed_data_"))
                            for child in workspace.iterdir()))
    except OSError as exc:
        raise DatasetError(f"Cannot inspect Waymo root {path}: {exc}") from exc
    if processed:
        return WaymoAdapter(root=root, version=version)
    return NativeWaymoAdapter(root=root, version=version)
