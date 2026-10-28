"""The prediction-source interface is independent of the selected dataset."""

from abc import ABC, abstractmethod

from .models import PredictionFrame, PredictionInfo


class PredictionSource(ABC):
    """Read canonical predictions independently of dataset storage and GUI.

    The historical term ``token`` means the exact normalized dataset frame ID;
    it need not be a nuScenes token. All boxes in a source share the declared
    coordinate system. Sources used by the desktop application must support
    sequential reads on its loading worker and metadata access on the UI thread.
    """
    @property
    @abstractmethod
    def info(self) -> PredictionInfo:
        """Describe the source coordinate system and declared processing stage."""

    @property
    @abstractmethod
    def frame_tokens(self) -> frozenset[str]:
        """All covered dataset frame IDs, including explicitly empty frames."""

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        """Selectable class names in stable order."""

    @abstractmethod
    def get_frame(self, token: str) -> PredictionFrame | None:
        """Read a frame; return None only when its token is absent."""

    @abstractmethod
    def close(self) -> None:
        """Release this handle; repeated closes are safe."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
