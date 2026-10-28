"""Small file-backed provider: one scene, NPZ clouds, RGB images, optional boxes.

This is an executable teaching example, not a universal interchange format.
The JSON catalog is read at construction; sensor payloads are loaded on demand.
Geometry uses metres and vehicle axes (X forward, Y left, Z up). Each frame
has its own local reference. Clouds and boxes share it; projection explicitly
converts it into the camera's optical axes.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from dataset_utils import (
    AnnotationSet, AnnotationStatus, Box3D, CameraFrameSet, CameraObservation,
    Capability, ClassInfo, DataUnavailableError, DatasetConfig,
    DatasetConfigurationError, DatasetIOError, DatasetLoader, DatasetMetadata,
    DatasetValidationError, FrameInfo, Modality, PinholeProjection,
    PointCloudFrame, SceneInfo, SensorInfo, UnknownIdentifierError,
)


def _read_manifest(path):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetIOError(f"Cannot read research manifest {path}: {error}") from error
    except (ValueError, UnicodeError) as error:
        raise DatasetValidationError(f"Invalid research manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise DatasetValidationError("Research manifest must be a JSON object.")
    for key in ("dataset_id", "scene_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise DatasetValidationError(f"Research manifest needs nonempty {key}.")
    records = manifest.get("frames")
    if not isinstance(records, list) or not records:
        raise DatasetValidationError("Research manifest needs a nonempty frames list.")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("frame_id"), str) or not record["frame_id"]:
            raise DatasetValidationError("Each research frame needs a nonempty frame_id.")
        if not {"points", "image"}.intersection(record):
            raise DatasetValidationError(f"Frame {record['frame_id']!r} has no payload.")
        for key in ("points", "image"):
            if key in record and (not isinstance(record[key], str) or not record[key]):
                raise DatasetValidationError(f"Research {key} path must be a nonempty string.")
    return manifest


class ResearchDatasetLoader(DatasetLoader):
    """Adapt the documented example manifest without importing any viewer."""

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        if (config.root is None or config.release is not None
                or config.storage_format not in {"auto", "native"}
                or config.source_paths or config.options):
            raise DatasetConfigurationError(
                "Research example requires root; accepts auto/native format and no release, sources or options."
            )
        manifest = _read_manifest(config.root / "manifest.json")
        self._scene_id = manifest["scene_id"]
        self._records = {}
        for record in manifest["frames"]:
            frame_id = record["frame_id"]
            if frame_id in self._records:
                raise DatasetValidationError(f"Duplicate example frame ID: {frame_id!r}.")
            self._records[frame_id] = record
        modes = {"points": Modality.POINT_CLOUD, "image": Modality.CAMERAS}
        self._frames = {}
        for modality in modes.values():
            infos = []
            for frame_id, record in self._records.items():
                present = frozenset(mode for key, mode in modes.items() if key in record)
                if not present:
                    raise DatasetValidationError(f"Frame {frame_id!r} has no payload.")
                if modality in present:
                    infos.append(FrameInfo(frame_id, self._scene_id, len(infos), record.get("timestamp_us"), present))
            self._frames[modality] = tuple(infos)
        self._infos = {mode: {info.frame_id: info for info in infos} for mode, infos in self._frames.items()}
        capabilities = {Capability(mode.value) for mode, infos in self._frames.items() if infos}
        if any("boxes" in record for record in self._records.values()):
            capabilities.add(Capability.BOXES_3D)
        self._metadata = DatasetMetadata(
            dataset_id=manifest["dataset_id"], display_name="Research dataset example",
            capabilities=capabilities, classes=(ClassInfo("object", "object", "Object"),),
            sensors=tuple(SensorInfo(key, mode, key.title(), "front" if key == "image" else None)
                          for key, mode in modes.items() if self._frames[mode]),
            splits=("all", "review"), clock_domain="scene-relative microseconds",
        )
        self._scene = SceneInfo(self._scene_id, self._scene_id,
                                frame_counts={mode: len(infos) for mode, infos in self._frames.items()})
        self._intrinsics = manifest.get("intrinsics")

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def _list_scenes(self, *, split: str = "all") -> tuple[SceneInfo, ...]:
        # Both declared splits intentionally contain this example's one scene.
        return (self._scene,)

    def _list_frames(self, scene_id: str, *, modality: Modality) -> tuple[FrameInfo, ...]:
        if scene_id != self._scene_id:
            raise UnknownIdentifierError(f"Unknown research scene {scene_id!r}.")
        return self._frames[modality]

    def _record(self, frame_id: str, modality: Modality) -> tuple[dict, FrameInfo]:
        if frame_id not in self._records:
            raise UnknownIdentifierError(f"Unknown research frame {frame_id!r}.")
        if frame_id not in self._infos[modality]:
            raise DataUnavailableError(f"Research frame {frame_id!r} lacks {modality.value}.")
        return self._records[frame_id], self._infos[modality][frame_id]

    def _annotations(self, frame_id: str, record: dict) -> AnnotationSet:
        # Absent boxes mean not provided; an explicit [] means available-empty.
        reference = f"vehicle:{frame_id}"
        if "boxes" not in record:
            return AnnotationSet(reference)
        try:
            boxes = tuple(Box3D(box["annotation_id"], "object", box["center"], box["size"], box["rotation"])
                          for box in record["boxes"])
        except (KeyError, TypeError) as error:
            raise DatasetValidationError(f"Malformed research boxes in frame {frame_id!r}: {error}") from error
        return AnnotationSet(reference, AnnotationStatus.AVAILABLE, boxes)

    def _load_point_cloud(self, frame_id: str) -> PointCloudFrame:
        record, info = self._record(frame_id, Modality.POINT_CLOUD)
        # The normalized record copies arrays before this archive is closed.
        path = self.config.root / record["points"]
        with np.load(path, allow_pickle=False) as archive:
            if "points" not in archive:
                raise DatasetValidationError(f"Research cloud {path} needs a 'points' array.")
            attributes = {"intensity": archive["intensity"]} if "intensity" in archive else {}
            return PointCloudFrame(info, "research:npz-xyz-v1", f"vehicle:{frame_id}",
                                   archive["points"], self._annotations(frame_id, record), attributes)

    def _load_cameras(self, frame_id: str) -> CameraFrameSet:
        record, info = self._record(frame_id, Modality.CAMERAS)
        with Image.open(self.config.root / record["image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        # Optical X-right/Y-down/Z-forward -> vehicle X-forward/Y-left/Z-up.
        # This is an explicit proper rotation, not an inferred camera model.
        pose = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                         [0., -1., 0., 0.], [0., 0., 0., 1.]])
        projection = None if self._intrinsics is None else PinholeProjection(self._intrinsics, pose)
        camera = CameraObservation("image", rgb, timestamp_us=info.timestamp_us,
                                   timestamp_kind="synthetic simultaneous exposure" if info.timestamp_us is not None else None,
                                   intrinsics=self._intrinsics, T_reference_from_camera=pose,
                                   projection_model=projection)
        return CameraFrameSet(info, "research:rgb-v1", f"vehicle:{frame_id}", (camera,),
                              self._annotations(frame_id, record))
