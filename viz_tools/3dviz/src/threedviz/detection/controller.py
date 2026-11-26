"""Compose canonical observations and predictions without wrapping a dataset."""

from __future__ import annotations

from dataclasses import dataclass, replace

from visual_utils import PointCloudFrame

from .predictions import PredictionFrame, PredictionSource, normalize_frame
from .predictions.demo import make_demo_predictions
from .processing.filters import FilterConfig, FilterResult, filter_predictions


@dataclass(frozen=True)
class ReviewFrame:
    observation: PointCloudFrame
    predictions: PredictionFrame | None = None
    filtered: FilterResult | None = None


def refilter(frame: ReviewFrame, config: FilterConfig) -> ReviewFrame:
    """Apply the current policy to original predictions, preserving the cloud."""
    return replace(frame, filtered=None if frame.predictions is None else filter_predictions(frame.predictions, config))


def compose_review(
    observation: PointCloudFrame, *, source: PredictionSource | None = None,
    config: FilterConfig | None = None, demo: bool = False,
) -> ReviewFrame:
    """Join one frame by exact ID; caller owns the loader and prediction source.

    Run on the loading worker when the source performs IO or filtering is costly.
    Missing predictions remain distinct from an explicitly empty prediction frame.
    """
    predictions = make_demo_predictions(observation) if demo else None
    if source is not None:
        raw = source.get_frame(observation.info.frame_id)
        if raw is not None:
            predictions = normalize_frame(raw, observation, coordinate_frame=source.info.coordinate_frame)
    return refilter(ReviewFrame(observation, predictions), config or FilterConfig())
