"""The adapter boundary between dataset formats and the camera viewer."""

from abc import ABC, abstractmethod

from dataset_utils.errors import DatasetError

from .models import CameraInfo, FrameData, FrameInfo, SceneInfo


class DatasetAdapter(ABC):
    def close(self) -> None:
        """Release an adapter after its caller finishes all pending reads."""

    @property
    def available_splits(self) -> tuple[str, ...]:
        return ("all",)

    def list_scenes_for_split(self, split: str = "all") -> tuple[SceneInfo, ...]:
        if split == "all":
            return self.list_scenes()
        raise DatasetError(f"This dataset does not support split {split!r}.")

    @abstractmethod
    def list_scenes(self) -> tuple[SceneInfo, ...]:
        """Return scenes without decoding their camera images."""

    @abstractmethod
    def list_frames(self, scene_token: str) -> tuple[FrameInfo, ...]:
        """Return chronologically ordered keyframes with zero-based indices."""

    @abstractmethod
    def load_frame(self, frame_token: str) -> FrameData:
        """Read RGB camera images, calibrated exposure poses and world boxes."""

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        """Return all selectable annotation classes in a stable order."""

    @property
    def camera_layout(self) -> tuple[CameraInfo, ...]:
        """Grid slots, including cameras absent from individual frames.

        A dataset with another camera arrangement can override this property;
        grid dimensions are derived from the supplied row and column indices.
        """
        return ()
