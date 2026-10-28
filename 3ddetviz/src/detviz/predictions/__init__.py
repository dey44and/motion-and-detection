"""Prediction-source registry, canonical models, and coordinate normalization."""

from .base import PredictionSource
from .demo import DemoPredictionSource, make_demo_predictions
from .factory import PredictionOpener, available_prediction_formats, open_predictions, register_prediction_format
from .models import Detection, PredictionCancelled, PredictionError, PredictionFrame, PredictionInfo
from .memory import InMemoryPredictionSource
from .normalize import normalize_frame
from .nuscenes_json import open_nuscenes_json
from .pcdet_pickle import open_pcdet_pickle

register_prediction_format("nuscenes_json", open_nuscenes_json)
register_prediction_format("pcdet_pickle", open_pcdet_pickle)

__all__ = [
    "Detection", "PredictionFrame", "PredictionInfo", "PredictionError", "PredictionCancelled",
    "PredictionSource", "InMemoryPredictionSource", "DemoPredictionSource", "make_demo_predictions", "normalize_frame",
    "PredictionOpener", "available_prediction_formats", "open_predictions", "register_prediction_format",
]
