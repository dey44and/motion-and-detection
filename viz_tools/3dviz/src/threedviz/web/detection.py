"""Browser state for the detection review pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

from threedviz.detection.controller import ReviewFrame, compose_review, refilter
from threedviz.detection.predictions import PredictionError, PredictionSource, open_predictions
from threedviz.detection.processing.filters import FilterConfig
from threedviz.processing import box_corners, class_color


@dataclass(frozen=True)
class DetectionReview:
    filters: FilterConfig
    source: PredictionSource | None = None
    path: str | None = None
    demo: bool = False
    review: ReviewFrame | None = None
    error: str | None = None

    @classmethod
    def initial(cls, frame, *, demo=False):
        return cls(FilterConfig(), demo=demo).for_frame(frame)

    def for_frame(self, frame):
        if self.source is None and self.error is not None:
            return self
        try:
            review = compose_review(frame, source=self.source, config=self.filters, demo=self.demo and self.source is None)
        except PredictionError as error:
            return replace(self, review=None, error=str(error))
        return replace(self, review=review, error=None)

    def with_filters(self, values):
        filters = replace(self.filters, **values)
        review = None if self.review is None else refilter(self.review, filters)
        return replace(self, filters=filters, review=review)

    def with_source(self, frame, path, format, *, cache_dir, progress, cancelled):
        resolved = str(Path(path).expanduser().resolve()) if path else None
        source = None if resolved is None else open_predictions(
            resolved, format, cache_dir=cache_dir, progress=progress, cancelled=cancelled,
        )
        try:
            return replace(self, source=source, path=resolved, error=None).for_frame(frame)
        except Exception:
            if source is not None:
                source.close()
            raise

    def public(self, metadata):
        classes = {}
        for item in metadata.classes:
            group = item.display_group or item.display_name
            classes[item.class_id] = _class(item.class_id, item.display_name, group)
        if self.review is not None:
            for box in self.review.observation.annotations.boxes:
                if box.class_id not in classes:
                    classes[box.class_id] = _class(box.class_id, box.class_id, box.class_id)
        source = None
        if self.source is not None:
            info = self.source.info
            source = {"name": info.name, "path": self.path, "format": info.format, "stage": info.stage}
            for name in self.source.class_names:
                if name not in classes:
                    classes[name] = _class(name, name, name)
        elif self.demo and self.error is None:
            source = {"name": "Demo predictions", "path": None, "format": "demo", "stage": "pre_nms"}

        counts, boxes = None, []
        filtered = None if self.review is None else self.review.filtered
        if self.error:
            status, message = "invalid", self.error
        elif source is None:
            status, message = "detached", "Attach a detection export in Sources."
        elif filtered is None:
            status, message = "missing", "This frame is absent from the detection export."
        else:
            counts = {name: getattr(filtered, f"{name}_count")
                      for name in ("loaded", "score", "spatial", "pre_nms", "nms", "output")}
            if filtered.loaded_count == 0:
                status, message = "empty", "The export contains an empty result for this frame."
            elif filtered.output_count == 0:
                status, message = "filtered", "All detections were removed by the current filters."
            else:
                status, message = "ready", f"{filtered.output_count:,} detections retained."
            for item in filtered.detections:
                box = item.box
                if box.class_id not in classes:
                    classes[box.class_id] = _class(box.class_id, box.class_id, box.class_id)
                boxes.append({"id": item.identifier, "class_id": box.class_id, "group": classes[box.class_id]["group"],
                              "corners": box_corners(box).tolist(), "score": item.score})
        filters = {field.name: getattr(self.filters, field.name) for field in fields(self.filters)}
        filters["per_class_scores"] = dict(self.filters.per_class_scores)
        return {"available": True, "status": status, "message": message, "source": source,
                "filters": filters, "counts": counts, "classes": list(classes.values()), "boxes": boxes}


def _class(class_id, name, group):
    return {"id": class_id, "name": name, "group": group, "color": list(class_color(group))}
