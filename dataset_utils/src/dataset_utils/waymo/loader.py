"""Normalized Waymo loader over native TFRecords and OpenPCDet point rows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..config import DatasetConfig
from ..dataset import DatasetLoader
from ..models import (
    AcquisitionGeometry, AnnotationAlignment, AnnotationSet, AnnotationStatus,
    AnnotationTiming, Box3D, CameraFrameSet, CameraObservation, Capability,
    ClassInfo, DatasetMetadata, FrameInfo, Modality, PointCloudFrame,
    PointColors, ProcessingProvenance, ProjectionResult,
    SceneInfo, SensorInfo,
)
from ..errors import (
    DataUnavailableError, DatasetConfigurationError, DatasetError,
    DatasetIOError, DatasetValidationError, UnknownIdentifierError,
)
from .camera import OPTICAL_TO_WAYMO, WaymoCameraProjection
from .processed import WAYMO_CLASSES, ProcessedWaymoReader, has_processed_layout
from .range_image import colorize_points, decode_lidar
from .reader import NativeWaymoReader, find_native_root

_NATIVE_SOURCE = "waymo:native:all-lidars-both-returns"
_CAMERAS = (
    (2, "CAM_FRONT_LEFT", "Front left", "front_left"),
    (1, "CAM_FRONT", "Front", "front"),
    (3, "CAM_FRONT_RIGHT", "Front right", "front_right"),
    (4, "CAM_SIDE_LEFT", "Side left", "side_left"),
    (5, "CAM_SIDE_RIGHT", "Side right", "side_right"),
)
_LASERS = ((1, "LIDAR_TOP"), (2, "LIDAR_FRONT"), (3, "LIDAR_SIDE_LEFT"),
           (4, "LIDAR_SIDE_RIGHT"), (5, "LIDAR_REAR"))
_SPLIT_ALIASES = {"training": "train", "validation": "val", "testing": "test"}


def _call(function, *args, **kwargs):
    """Translate storage parser failures into the shared dataset errors."""
    try:
        return function(*args, **kwargs)
    except (DatasetConfigurationError, DatasetIOError, DatasetValidationError,
            UnknownIdentifierError, DataUnavailableError):
        raise
    except (DatasetError, OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        if function in (NativeWaymoReader, ProcessedWaymoReader) and any(fragment in str(error) for fragment in (
            "TFRecords not found", "preprocessing not found", "Requested Waymo processed directory does not exist",
            "Several Waymo processed directories",
        )):
            raise DatasetConfigurationError(str(error)) from error
        cause = error
        while cause is not None:
            if isinstance(cause, OSError) and getattr(cause, "errno", None) is not None:
                raise DatasetIOError(str(error)) from error
            cause = cause.__cause__
        raise DatasetValidationError(str(error)) from error


def _directory(path, name):
    if path is None or not Path(path).is_dir():
        raise DatasetConfigurationError(f"Waymo {name} directory does not exist: {path}")
    return Path(path)


def _reference(frame_id):
    return f"{frame_id}:vehicle-reference"


def _world(context):
    return f"waymo:{context}:world"


def _rotation(heading):
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])


class _ReferenceProjection:
    """A native shutter/distortion model consuming reference-vehicle points."""

    def __init__(self, camera, vehicle_to_world):
        self._model = WaymoCameraProjection(camera)
        self._world = np.array(vehicle_to_world, dtype=np.float64, copy=True)
        self._world.setflags(write=False)

    def project_reference(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise DatasetValidationError("Projection input must be finite (N,3) reference points.")
        uv, depth = self._model.project_world(points @ self._world[:3, :3].T + self._world[:3, 3])
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
        return ProjectionResult(uv=uv, depth=depth, valid_mask=valid)


class WaymoLoader(DatasetLoader):
    """One contract for native cameras and native or processed point clouds.

    Formats are ``auto``, ``native`` and ``openpcdet``. Named source paths are
    ``native`` and ``processed``. Options are ``processed_tag`` and
    ``additional_classes``; the latter declares custom profile categories
    without eagerly opening every sequence metadata pickle.
    """

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        if config.storage_format not in {"auto", "native", "openpcdet"}:
            raise DatasetConfigurationError("Waymo storage_format must be auto, native or openpcdet.")
        if config.release not in {None, "1.4.2", "v1.4.2"}:
            raise DatasetConfigurationError(f"Unsupported Waymo release {config.release!r}; use 1.4.2 or omit release.")
        if set(config.source_paths).difference({"processed", "native"}):
            raise DatasetConfigurationError("Waymo source_paths accepts only processed and native.")
        if set(config.options).difference({"processed_tag", "additional_classes"}):
            raise DatasetConfigurationError("Waymo options accepts only processed_tag and additional_classes.")
        tag = config.options.get("processed_tag")
        if tag is not None and (not isinstance(tag, str) or not tag or Path(tag).name != tag or tag in {".", ".."}):
            raise DatasetConfigurationError("processed_tag must be a nonempty directory name.")
        additional = config.options.get("additional_classes", ())
        if isinstance(additional, str) or not isinstance(additional, (tuple, list)) or any(
            not isinstance(name, str) or not name.strip() for name in additional
        ) or len(set(additional)) != len(additional):
            raise DatasetConfigurationError("additional_classes must be a sequence of distinct nonempty names.")
        self._class_names = tuple(dict.fromkeys((*WAYMO_CLASSES, *additional)))
        self._processed = None
        self._reader = None
        root = config.root
        processed_path = config.source_paths.get("processed")
        native_path = config.source_paths.get("native")
        if processed_path is not None:
            _directory(processed_path, "processed")
        if native_path is not None:
            _directory(native_path, "native")
        if root is None and processed_path is None and native_path is None:
            raise DatasetConfigurationError("Waymo needs root or an explicit processed/native source path.")

        use_processed = (config.storage_format == "openpcdet"
                         or config.storage_format == "auto" and (
                             processed_path is not None or tag is not None
                             or root is not None and _call(has_processed_layout, _directory(root, "root"))))
        if use_processed:
            self._processed = _call(
                ProcessedWaymoReader, _directory(processed_path or root, "processed"),
                processed_tag=tag, classes=self._class_names,
            )
        elif config.storage_format == "native" and tag is not None:
            raise DatasetConfigurationError("processed_tag is not used by native storage.")

        if native_path is not None:
            self._reader = _call(NativeWaymoReader, native_path)
        elif self._processed is None:
            self._reader = _call(NativeWaymoReader, _directory(root, "native"))
        else:
            paired = _call(find_native_root, self._processed.root)
            if paired is not None:
                self._reader = _call(NativeWaymoReader, paired)
        self._native_sequences = {} if self._reader is None else {
            sequence.context_name: sequence for sequence in self._reader.list_sequences()
        }
        if tag is not None and self._processed is None:
            raise DatasetConfigurationError("processed_tag requires an OpenPCDet point source.")
        if self._reader is not None and not self._native_sequences:
            raise DatasetConfigurationError("Waymo native source contains no TFRecord sequences.")
        self._processed_scenes = frozenset() if self._processed is None else frozenset(self._processed.scene_ids)
        self._scene_ids = tuple(sorted(set(self._processed_scenes) | set(self._native_sequences)))
        if not self._scene_ids:
            raise DatasetConfigurationError("No Waymo sequences are installed.")
        self._native_points = OrderedDict()
        self._frames = {Modality.POINT_CLOUD: {}, Modality.CAMERAS: {}}
        self._by_id = {Modality.POINT_CLOUD: {}, Modality.CAMERAS: {}}
        self._native_indices = {}
        self._splits = self._split_catalog()
        capabilities = {Capability.POINT_CLOUD, Capability.BOXES_3D}
        if self._reader is not None:
            capabilities.update((Capability.CAMERAS, Capability.RGB_COLORING, Capability.ACQUISITION_GEOMETRY))
        sensors = [SensorInfo(sensor_id=name, modality=Modality.POINT_CLOUD, display_name=name) for _, name in _LASERS]
        if self._reader is not None:
            sensors.extend(SensorInfo(sensor_id=sensor_id, modality=Modality.CAMERAS, display_name=label, view_role=role)
                           for _, sensor_id, label, role in _CAMERAS)
        self._metadata = DatasetMetadata(
            dataset_id=f"waymo:{'1.4.2' if config.release else 'perception'}", display_name="Waymo Open Dataset",
            release="1.4.2" if config.release else None, capabilities=frozenset(capabilities),
            classes=tuple(ClassInfo(class_id=name, native_name=name, display_name=name) for name in self._class_names),
            sensors=tuple(sensors), splits=tuple(self._splits), clock_domain="unix_microseconds",
            feature_coverage={Capability.ACQUISITION_GEOMETRY: "partial"} if self._reader is not None else {},
        )

    @property
    def metadata(self):
        return self._metadata

    @property
    def reader(self):
        return self._reader

    def _split_catalog(self):
        native = {short: {context for context, seq in self._native_sequences.items() if seq.split == full}
                  for full, short in _SPLIT_ALIASES.items()}
        declared = {} if self._processed is None else self._processed.split_members
        result = {"all": frozenset(self._scene_ids)}
        for name in ("train", "val", "test"):
            members = set(native[name])
            if name == "train" and "train_full" in declared:
                members.update(declared["train_full"])
            elif name == "train" and not members and "train" in declared:
                if "train_1of16" not in declared or set(declared["train"]) != set(declared["train_1of16"]):
                    members.update(declared["train"])
            elif name != "train":
                members.update(declared.get(name, ()))
            if members or name in declared or name == "train" and "train_full" in declared:
                # Do not advertise reduced train.txt as the complete train split.
                if name == "train" and not members:
                    continue
                result[name] = frozenset(members)
        for name in ("train_full", "train_1of16"):
            if name in declared:
                result[name] = frozenset(declared[name])
        return result

    def _has_points(self, context):
        return context in (self._processed_scenes if self._processed is not None else self._native_sequences)

    def _list_scenes(self, *, split="all"):
        selected = self._splits[split]
        result = []
        for context in self._scene_ids:
            if context not in selected:
                continue
            counts = {}
            for mode, present in ((Modality.POINT_CLOUD, self._has_points(context)), (Modality.CAMERAS, context in self._native_sequences)):
                counts[mode] = len(self._frames[mode][context]) if context in self._frames[mode] else None if present else 0
            result.append(SceneInfo(scene_id=context, display_name=context,
                                    description="Waymo OpenPCDet points" if self._processed else "Waymo native TFRecord",
                                    frame_counts=counts))
        return tuple(result)

    def _list_frames(self, scene_id, *, modality):
        if scene_id not in self._scene_ids:
            raise UnknownIdentifierError(f"Unknown Waymo scene {scene_id!r}.")
        if scene_id in self._frames[modality]:
            return self._frames[modality][scene_id]
        observations = []
        if modality == Modality.POINT_CLOUD and self._processed is not None:
            frames = (() if scene_id not in self._processed_scenes else
                      _call(self._processed.list_frames, scene_id))
            self._frames[modality][scene_id] = frames
            self._by_id[modality].update((frame.frame_id, frame) for frame in frames)
            return frames
        if scene_id in self._native_sequences:
            headers = _call(self._reader.list_frame_headers, scene_id)
            for header in headers:
                token = f"waymo:{scene_id}:{header.timestamp_us}"
                observations.append((token, header.timestamp_us))
                self._native_indices[token] = header.frame_index
        frames = tuple(FrameInfo(frame_id=token, scene_id=scene_id, index=index, timestamp_us=timestamp,
                                 modalities=frozenset({modality}))
                       for index, (token, timestamp) in enumerate(observations))
        if any(right.timestamp_us <= left.timestamp_us for left, right in zip(frames, frames[1:])):
            raise DatasetValidationError(f"Waymo timestamps do not increase in scene {scene_id}.")
        self._frames[modality][scene_id] = frames
        self._by_id[modality].update((frame.frame_id, frame) for frame in frames)
        return frames

    def _resolve(self, frame_id, modality):
        if frame_id not in self._by_id[modality]:
            context = frame_id[len("waymo:"):].rsplit(":", 1)[0] if isinstance(frame_id, str) and frame_id.startswith("waymo:") else None
            if context not in self._scene_ids:
                raise UnknownIdentifierError(f"Unknown Waymo frame {frame_id!r}.")
            self.list_frames(context, modality=modality)
            if frame_id not in self._by_id[modality]:
                other = Modality.CAMERAS if modality == Modality.POINT_CLOUD else Modality.POINT_CLOUD
                if Capability(other.value) in self.metadata.capabilities:
                    self.list_frames(context, modality=other)
                    if frame_id in self._by_id[other]:
                        raise DataUnavailableError(f"Waymo frame {frame_id} has no {modality.value} observation in the selected profile.")
                raise UnknownIdentifierError(f"Unknown Waymo frame {frame_id!r}.")
        return self._by_id[modality][frame_id]

    def _native(self, info, *, include_points=True, include_cameras=True):
        if info.scene_id not in self._native_sequences:
            raise DataUnavailableError(f"Native Waymo TFRecord for context {info.scene_id} is unavailable.")
        if info.frame_id not in self._native_indices:
            self.list_frames(info.scene_id, modality=Modality.CAMERAS)
        if info.frame_id not in self._native_indices:
            raise DataUnavailableError(f"Native Waymo frame {info.frame_id} is unavailable.")
        return _call(self._reader.read_frame, info.scene_id, self._native_indices[info.frame_id], timestamp_us=info.timestamp_us,
                     include_points=include_points, include_cameras=include_cameras)

    def _pointset(self, info):
        if info.frame_id not in self._native_points:
            native = self._native(info, include_cameras=False)
            if not native.lasers:
                raise DataUnavailableError(f"Waymo frame {info.frame_id} contains no LiDAR observations.")
            self._native_points[info.frame_id] = _call(decode_lidar, native)
            if len(self._native_points) > 2:
                self._native_points.popitem(last=False)
        self._native_points.move_to_end(info.frame_id)
        return self._native_points[info.frame_id]

    def _status(self, context, *, has_boxes):
        if has_boxes:
            return AnnotationStatus.AVAILABLE
        if context in self._splits.get("test", ()):
            return AnnotationStatus.WITHHELD
        if self._native_sequences[context].split in {"training", "validation"}:
            return AnnotationStatus.AVAILABLE
        return AnnotationStatus.NOT_PROVIDED

    def _native_annotations(self, info, native, *, camera_name=None):
        reference = _reference(info.frame_id)
        source_boxes = []
        alignments = []
        synced = False
        names = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
        for index, box in enumerate(native.boxes):
            selected = box
            alignment = AnnotationAlignment.REFERENCE_TIME
            if camera_name is not None and box.camera_synced_box is not None and box.most_visible_camera_name == names[camera_name]:
                selected, alignment, synced = box.camera_synced_box, AnnotationAlignment.CAMERA_SYNCHRONIZED, True
            if box.class_name not in self._class_names:
                raise DatasetValidationError(f"Undeclared Waymo native class {box.class_name!r}.")
            source_boxes.append((index, box, selected, alignment))
            alignments.append(alignment)
        if camera_name is not None and not synced:
            return None
        mixed = len(set(alignments)) > 1
        timing = AnnotationTiming(alignment=AnnotationAlignment.MIXED if mixed else alignments[0] if alignments else AnnotationAlignment.REFERENCE_TIME)
        boxes = tuple(Box3D(
            annotation_id=box.identifier or f"box:{index}", instance_id=box.identifier or None, class_id=box.class_name,
            center=selected.center, size=selected.size, rotation=_rotation(selected.heading),
            timing_override=AnnotationTiming(alignment=alignment) if mixed else None,
        ) for index, box, selected, alignment in source_boxes)
        status = self._status(info.scene_id, has_boxes=bool(boxes))
        return AnnotationSet(reference_frame_id=reference, status=status, boxes=boxes, timing=timing)

    def _load_point_cloud(self, frame_id):
        info = self._resolve(frame_id, Modality.POINT_CLOUD)
        if self._processed is not None:
            return _call(self._processed.load_point_cloud, frame_id)
        reference = _reference(frame_id)
        native = self._native(info, include_cameras=False)
        points = self._pointset(info)
        attributes = {name: getattr(points, name) for name in ("intensity", "elongation", "nlz", "return_id", "lidar_id")}
        sensor_names = np.array(["", *(name for _, name in _LASERS)])
        attributes["sensor_id"] = sensor_names[points.lidar_id]
        return PointCloudFrame(info=info, source_id=_NATIVE_SOURCE, reference_frame_id=reference,
            points=points.points, attributes=attributes, annotations=self._native_annotations(info, native),
            world_frame_id=_world(info.scene_id), T_world_from_reference=native.vehicle_to_world,
            scan_start_us=info.timestamp_us)

    def _load_cameras(self, frame_id):
        info = self._resolve(frame_id, Modality.CAMERAS)
        native = self._native(info, include_points=False)
        if not native.cameras:
            raise DataUnavailableError(f"Waymo frame {frame_id} contains no camera observations.")
        reference = _reference(frame_id)
        by_id = {camera.name: camera for camera in native.cameras}
        inverse = np.linalg.inv(native.vehicle_to_world)
        cameras = []
        for number, sensor_id, label, role in _CAMERAS:
            camera = by_id.get(number)
            if camera is None:
                continue
            timestamp = round(camera.trigger_time_s * 1_000_000) if camera.trigger_time_s > 0 else None
            pose_time = round(camera.pose_timestamp_s * 1_000_000) if camera.pose_timestamp_s > 0 else None
            interval = ((timestamp, round(camera.readout_done_time_s * 1_000_000))
                        if timestamp is not None and camera.readout_done_time_s >= camera.trigger_time_s else None)
            intrinsics = np.array([[camera.intrinsic[0], 0., camera.intrinsic[2]],
                                   [0., camera.intrinsic[1], camera.intrinsic[3]], [0., 0., 1.]])
            cameras.append(CameraObservation(sensor_id=sensor_id, image=camera.image_rgb, timestamp_us=timestamp,
                timestamp_kind="camera_trigger" if timestamp is not None else None, exposure_interval_us=interval,
                pose_timestamp_us=pose_time, intrinsics=intrinsics,
                T_reference_from_camera=inverse @ camera.vehicle_to_world @ camera.extrinsic @ OPTICAL_TO_WAYMO,
                projection_model=_ReferenceProjection(camera, native.vehicle_to_world),
                annotations=self._native_annotations(info, native, camera_name=number)))
        return CameraFrameSet(info=info, source_id=_NATIVE_SOURCE, reference_frame_id=reference,
                              cameras=tuple(cameras), annotations=self._native_annotations(info, native),
                              world_frame_id=_world(info.scene_id), T_world_from_reference=native.vehicle_to_world)

    def _prepare_point_cloud(self, frame, *, request):
        if request.timestamps:
            # The base capability guard normally handles this before entry.
            raise DataUnavailableError("Waymo does not supply scalar point acquisition timestamps.")
        info = frame.info
        native = self._native(info, include_cameras=request.rgb and frame.rgb is None)
        pose = frame.T_world_from_reference
        if (pose is None or not np.allclose(pose[:3, :3], native.vehicle_to_world[:3, :3], rtol=0, atol=1e-5)
                or not np.allclose(pose[:3, 3], native.vehicle_to_world[:3, 3], rtol=0, atol=.02)):
            raise DatasetValidationError(
                f"Waymo processed/native reference pose correspondence differs for {info.frame_id}. "
                "Camera colors/acquisition geometry require matching vehicle poses (2 cm translation tolerance)."
            )
        points = self._pointset(info)
        if points.points.shape != frame.points.shape or not np.allclose(points.points, frame.points, rtol=0, atol=.02):
            raise DatasetValidationError(
                f"Waymo processed/native point correspondence differs in count, order or XYZ for {info.frame_id}. "
                "Camera colors/acquisition geometry require matching all-five-LiDAR, both-return rows or an explicit mapping."
            )
        changes = {}
        if request.rgb and frame.rgb is None:
            if not native.cameras:
                raise DataUnavailableError(f"Waymo frame {info.frame_id} has no camera images for RGB.")
            colors, valid = _call(colorize_points, native, points)
            changes["rgb"] = PointColors(values=colors, valid_mask=valid,
                provenance=ProcessingProvenance(method_id="waymo.native_return_camera_associations",
                    source_ids=(frame.source_id, _NATIVE_SOURCE), assumptions=("Native per-return image links; no additional occlusion inference.",)))
        if request.acquisition_geometry and frame.acquisition is None:
            has_poses = any(laser.name == 1 and any(item.range_image_pose_compressed for item in laser.returns)
                            for laser in native.lasers)
            if points.raw_points is None or points.raw_valid_mask is None or not has_poses:
                raise DataUnavailableError(f"Waymo frame {info.frame_id} has no TOP per-pixel acquisition geometry.")
            changes["acquisition"] = AcquisitionGeometry(points=points.raw_points, valid_mask=points.raw_valid_mask,
                reference_frame_id=frame.reference_frame_id + ":acquisition-vehicle",
                provenance=ProcessingProvenance(method_id="waymo.top_pixel_poses", source_ids=(_NATIVE_SOURCE,),
                    assumptions=("Only TOP rows with pixel poses are valid; side-lidar rows retain reference placeholders.",
                                 "Per-pixel poses do not provide scalar point timestamps.")))
        return replace(frame, **changes)

    def _close(self):
        self._native_points.clear()
        if self._reader is not None:
            self._reader.close()
        if self._processed is not None:
            self._processed.close()
