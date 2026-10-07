"""The narrow adapter interface needed by the visualization application."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import FrameData, FrameInfo, SceneInfo


class DatasetError(RuntimeError):
    """A dataset cannot be read or has inconsistent metadata."""


class DatasetAdapter(ABC):
    """Implement this interface to support another dataset without UI changes."""

    @abstractmethod
    def list_scenes(self) -> tuple[SceneInfo, ...]:
        """Return scenes in a stable order; loading points is deferred."""

    @abstractmethod
    def list_frames(self, scene_token: str) -> tuple[FrameInfo, ...]:
        """Return chronologically ordered frames with zero-based indices."""

    @abstractmethod
    def load_frame(self, frame_token: str) -> FrameData:
        """Read a frame's sensor-coordinate points and bounding boxes."""

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        """Return selectable annotation classes in a stable order."""

    @property
    def supports_camera_rgb(self) -> bool:
        """Whether camera images and calibration can supply point colors."""
        return False

    @property
    def supports_point_timing(self) -> bool:
        """Whether acquisition times and uncompensated points can be prepared."""
        return False

    def prepare_frame(
        self, frame: FrameData, *, camera_rgb: bool = False, point_timing: bool = False
    ) -> FrameData:
        """Prepare optional frame attributes without imposing them on adapters.

        The viewer calls this on its loading worker, only for requested modes.
        Existing adapters keep working without camera or motion support.
        """
        if camera_rgb and frame.rgb_colors is None:
            raise DatasetError("This dataset does not provide camera RGB point colors.")
        if point_timing and frame.raw_points is None:
            raise DatasetError("This dataset does not provide point acquisition times.")
        return frame
