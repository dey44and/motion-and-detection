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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DatasetError, DatasetValidationError, UnknownIdentifierError
from ..models import (
    AnnotationAlignment, AnnotationSet, AnnotationStatus, AnnotationTiming,
    Box3D, FrameInfo, Modality, PointCloudFrame,
)

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
    return matrix


@dataclass(frozen=True)
class _SequenceFrame:
    info: FrameInfo
    sample_idx: int
    record: dict[str, Any]


class ProcessedWaymoReader:
    """Read existing OpenPCDet Waymo preprocessing without changing its data.

    ``root`` may be the dataset parent, ``waymo_pcdet_workspace``, or an explicit
    processed-data directory. The full processed tag is preferred automatically;
    an explicit directory or ``processed_tag`` selects another preparation.
    ``warnings`` reports skipped partial sequence directories.
    """

    def __init__(self, root: str | Path, *, processed_tag: str | None = None,
                 classes: tuple[str, ...] = WAYMO_CLASSES):
        if root is None:
            raise DatasetError("Waymo requires a dataset root: choose waymo_pcdet_workspace or its processed-data directory.")
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise DatasetError(f"Waymo dataset directory does not exist: {self.root}")
        self.processed_root, self.workspace_root = self._discover(self.root, processed_tag)
        self.source_id = f"waymo:openpcdet:xyz-intensity-elongation-nlz:{self.processed_root.name}"
        self._classes = frozenset(classes)
        self._sequence_cache: OrderedDict[str, tuple[_SequenceFrame, ...]] = OrderedDict()
        self._frames_by_scene: dict[str, tuple[FrameInfo, ...]] = {}
        self._frame_locations: dict[str, tuple[str, int]] = {}
        self.split_members = self._read_splits()

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
        self.warnings = tuple([
            f"Skipped {len(partial)} partial Waymo sequence directories without per-sequence metadata. "
            f"First missing file: {partial[0]}. Choose a complete processed directory or finish preprocessing."
        ] if partial else [])

    @staticmethod
    def _discover(root: Path, processed_tag: str | None) -> tuple[Path, Path]:
        def sequence_directory(path):
            return any(child.is_dir() and _is_sequence(child.name) for child in path.iterdir())

        try:
            if sequence_directory(root):
                return root, root.parent
            workspace = root / "waymo_pcdet_workspace" if (root / "waymo_pcdet_workspace").is_dir() else root
            requested = processed_tag
            if requested is not None:
                if Path(requested).name != requested or requested in {".", ".."}:
                    raise DatasetError("Waymo processed_tag must be a directory tag, not a path.")
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
                    "or set processed_tag."
                )
        except OSError as exc:
            raise DatasetError(f"Cannot inspect Waymo dataset root {root}: {exc}") from exc
        raise DatasetError(
            f"OpenPCDet Waymo preprocessing not found at {root}. Choose waymo_pcdet_workspace, "
            f"its {DEFAULT_PROCESSED_TAG} directory, or the dataset parent. "
            "Legacy Waymo/ frame pickles are a different representation; native TFRecords use NativeWaymoReader."
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
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(self._context_to_scene)

    def close(self) -> None:
        """Release sequence metadata; sensor files are opened only per read."""
        self._sequence_cache.clear()
        self._frames_by_scene.clear()
        self._frame_locations.clear()

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
            FrameInfo(f"waymo:{context_name}:{timestamp}", context_name, display_index, timestamp,
                      frozenset((Modality.POINT_CLOUD,))), sample_idx, record
        ) for display_index, (sample_idx, timestamp, record) in enumerate(indexed))
        self._frames_by_scene[scene_token] = tuple(frame.info for frame in frames)
        self._frame_locations.update((frame.info.frame_id, (scene_token, frame.sample_idx)) for frame in frames)
        self._sequence_cache[scene_token] = frames
        if len(self._sequence_cache) > _SEQUENCE_CACHE_SIZE:
            self._sequence_cache.popitem(last=False)
        return frames

    def list_frames(self, scene_id: str) -> tuple[FrameInfo, ...]:
        scene_token = self._context_to_scene.get(scene_id)
        if scene_token is None:
            raise UnknownIdentifierError(f"Unknown Waymo scene {scene_id!r}.")
        if scene_token not in self._frames_by_scene:
            self._load_sequence(scene_token)
        return self._frames_by_scene[scene_token]

    def _boxes(self, record: dict[str, Any], frame_token: str) -> tuple[Box3D, ...]:
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
        for index, (value, name, identifier) in enumerate(zip(values, names, identifiers)):
            if not isinstance(name, str) or not name.strip():
                raise DatasetError(f"Invalid Waymo annotation class for {frame_token}: {name!r}.")
            cosine, sine = np.cos(value[6]), np.sin(value[6])
            rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
            if name not in self._classes:
                raise DatasetValidationError(f"Undeclared Waymo class {name!r}; declare it in additional_classes.")
            boxes.append(Box3D(
                annotation_id=str(identifier) or f"{frame_token}:box:{index}",
                instance_id=(str(identifier) or None) if "obj_ids" in annotations else None,
                class_id=str(name), center=value[:3], size=value[3:6], rotation=rotation,
            ))
        return tuple(boxes)

    def load_point_cloud(self, frame_token: str) -> PointCloudFrame:
        location = self._frame_locations.get(frame_token)
        if location is None and isinstance(frame_token, str) and frame_token.startswith("waymo:"):
            context = frame_token[len("waymo:"):].rsplit(":", 1)[0]
            scene_token = self._context_to_scene.get(context)
            if scene_token is not None:
                self._load_sequence(scene_token)
                location = self._frame_locations.get(frame_token)
        if location is None:
            raise UnknownIdentifierError(f"Unknown Waymo frame {frame_token!r}.")
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
        reference = f"{frame_token}:{self.source_id}:vehicle-reference"
        provided = frame.record.get("annos") is not None
        status = (AnnotationStatus.AVAILABLE if provided else
                  AnnotationStatus.WITHHELD if frame.info.scene_id in self.split_members.get("test", ()) else
                  AnnotationStatus.NOT_PROVIDED)
        return PointCloudFrame(
            info=frame.info, source_id=self.source_id, reference_frame_id=reference,
            points=scan[:, :3],
            attributes={"intensity": scan[:, 3], "elongation": scan[:, 4], "nlz": scan[:, 5]},
            annotations=AnnotationSet(
                reference, status, boxes, AnnotationTiming(alignment=AnnotationAlignment.REFERENCE_TIME)),
            world_frame_id=f"waymo:{frame.info.scene_id}:world",
            T_world_from_reference=_pose(frame.record.get("pose"), f"Waymo pose for {frame_token}"),
            scan_start_us=frame.info.timestamp_us,
        )


def has_processed_layout(root: Path) -> bool:
    """Recognize a processed source by directory names without reading payloads.

    Recognition intentionally includes incomplete layouts: malformed processed
    data must report its error rather than silently falling back to native data.
    """
    workspace = root / "waymo_pcdet_workspace" if (root / "waymo_pcdet_workspace").is_dir() else root
    return root.name.startswith("waymo_processed_data_") or any(
        child.is_dir() and (_is_sequence(child.name) or
                            child.name.startswith("waymo_processed_data_") and "gt_database" not in child.name)
        for child in workspace.iterdir()
    )
