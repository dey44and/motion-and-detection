"""The adapter boundary between dataset formats and the camera viewer."""

from abc import ABC, abstractmethod

from .models import CameraInfo, FrameData, FrameInfo, SceneInfo


class DatasetError(RuntimeError):
    """A dataset could not be read or has inconsistent metadata."""


class DatasetAdapter(ABC):
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
