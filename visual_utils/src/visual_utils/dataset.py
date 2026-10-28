"""The common dataset interface used by the camera, cloud, and detection viewers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import DatasetConfig
from .errors import DatasetClosedError, DatasetConfigurationError, DatasetIOError, UnsupportedCapabilityError
from .models import (
    CameraFrameSet, Capability, DatasetMetadata, FrameInfo, Modality,
    PointCloudFrame, PointCloudPreparation, SceneInfo,
)


class DatasetLoader(ABC):
    """Adapt native storage to shared records.

    Implement metadata and the _list_*/_load_* methods below. Public methods
    handle lifecycle, splits and capabilities. Calls are synchronous; viewers
    schedule them on their own loading worker.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self._closed = False

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Classes, ordered sensors, splits and features of this installation."""

    def list_scenes(self, *, split: str = "all") -> tuple[SceneInfo, ...]:
        self._ensure_open()
        if split not in self.metadata.splits:
            raise DatasetConfigurationError(f"Unsupported split {split!r} for {self.metadata.dataset_id}.")
        return self._list_scenes(split=split)

    @abstractmethod
    def _list_scenes(self, *, split: str) -> tuple[SceneInfo, ...]:
        """List metadata without decoding images or clouds."""

    def list_frames(self, scene_id: str, *, modality: Modality) -> tuple[FrameInfo, ...]:
        modality = Modality(modality)
        self._require(Capability(modality.value))
        return self._list_frames(scene_id, modality=modality)

    @abstractmethod
    def _list_frames(self, scene_id: str, *, modality: Modality) -> tuple[FrameInfo, ...]:
        """Return ordered observations with stable IDs and dense display indices."""

    def load_point_cloud(self, frame_id: str) -> PointCloudFrame:
        self._require(Capability.POINT_CLOUD)
        try:
            return self._load_point_cloud(frame_id)
        except OSError as error:
            raise DatasetIOError(f"{self.metadata.dataset_id}: cloud {frame_id}: {error}") from error

    def _load_point_cloud(self, frame_id: str) -> PointCloudFrame:
        raise UnsupportedCapabilityError("This dataset does not implement point-cloud loading.")

    def load_cameras(self, frame_id: str) -> CameraFrameSet:
        self._require(Capability.CAMERAS)
        try:
            return self._load_cameras(frame_id)
        except OSError as error:
            raise DatasetIOError(f"{self.metadata.dataset_id}: cameras {frame_id}: {error}") from error

    def _load_cameras(self, frame_id: str) -> CameraFrameSet:
        raise UnsupportedCapabilityError("This dataset does not implement camera loading.")

    def prepare_point_cloud(self, frame: PointCloudFrame, *, request: PointCloudPreparation) -> PointCloudFrame:
        """Add requested data, preserving baseline rows and existing fields.

        Pass the unchanged observation returned by this loader. Providers own
        point correspondence; this interface does not police caller mutation.
        """
        self._ensure_open()
        missing = ((request.rgb and frame.rgb is None, Capability.RGB_COLORING),
                   (request.timestamps and frame.timing is None, Capability.POINT_TIMESTAMPS),
                   (request.acquisition_geometry and frame.acquisition is None, Capability.ACQUISITION_GEOMETRY))
        for needed, capability in missing:
            if needed:
                self._require(capability)
        if not any(needed for needed, _ in missing):
            return frame
        try:
            return self._prepare_point_cloud(frame, request=request)
        except OSError as error:
            raise DatasetIOError(f"{self.metadata.dataset_id}: prepare {frame.info.frame_id}: {error}") from error

    def _prepare_point_cloud(self, frame: PointCloudFrame, *, request: PointCloudPreparation) -> PointCloudFrame:
        raise UnsupportedCapabilityError("This dataset does not implement optional point preparation.")

    def _ensure_open(self) -> None:
        if self._closed:
            raise DatasetClosedError("Dataset loader is closed.")

    def _require(self, capability: Capability) -> None:
        self._ensure_open()
        if capability not in self.metadata.capabilities:
            raise UnsupportedCapabilityError(f"{self.metadata.dataset_id} does not offer {capability.value}.")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._close()

    def _close(self) -> None:
        """Release provider resources after pending reads have finished."""

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
