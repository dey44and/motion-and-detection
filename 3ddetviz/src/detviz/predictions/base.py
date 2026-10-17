"""The prediction-source interface is independent of the selected dataset."""

from abc import ABC, abstractmethod

from .models import PredictionFrame, PredictionInfo


class PredictionSource(ABC):
    @property
    @abstractmethod
    def info(self) -> PredictionInfo:
        """Describe the source coordinate system and declared processing stage."""

    @property
    @abstractmethod
    def frame_tokens(self) -> frozenset[str]:
        """All covered sample tokens, including explicitly empty frames."""

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
