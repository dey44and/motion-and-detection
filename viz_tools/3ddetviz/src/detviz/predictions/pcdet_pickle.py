"""Import saved OpenPCDet NumPy predictions without loading Torch or pcdet."""

from __future__ import annotations

import importlib
import math
import pickle
import sqlite3
from pathlib import Path

import numpy as np

from .models import PredictionError
from .store import (
    CacheWriter, SQLitePredictionSource, cache_path, canonical_row,
    check_cancelled, notify, open_cached, read_sidecar, source_info,
)


class NumpyPredictionUnpickler(pickle.Unpickler):
    """Limit pickle globals to the NumPy array/scalar reconstruction primitives."""

    def find_class(self, module: str, name: str):
        if (module, name) in {("numpy", "ndarray"), ("numpy", "dtype")}:
            return getattr(np, name)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {"_reconstruct", "scalar"}:
            core = "numpy._core" if hasattr(np, "_core") else "numpy.core"
            return getattr(importlib.import_module(core + ".multiarray"), name)
        if module in {"numpy.core.numeric", "numpy._core.numeric"} and name == "_frombuffer":
            core = "numpy._core" if hasattr(np, "_core") else "numpy.core"
            return getattr(importlib.import_module(core + ".numeric"), name)
        raise PredictionError(
            f"Unsupported pickle global {module}.{name}. Export NumPy arrays and primitive metadata; "
            "Torch tensors and model objects are not accepted."
        )


def _class_mapping(metadata: dict, sidecar: dict) -> tuple[str, ...] | None:
    declarations = {}
    for source in (metadata, metadata.get("detviz", {}), sidecar, sidecar.get("detviz", {})):
        if isinstance(source, dict):
            declarations.update(source)
    names = declarations.get("class_names")
    if names is None:
        return None
    if not isinstance(names, (list, tuple)) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise PredictionError("class_names metadata must be an ordered list of nonempty class names.")
    if len(set(names)) != len(names):
        raise PredictionError("class_names metadata must not contain duplicates.")
    return tuple(names)


def _frame_rows(entry: object, class_names: tuple[str, ...] | None):
    if not isinstance(entry, dict):
        raise PredictionError("Each OpenPCDet prediction frame must be a dictionary.")
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        raise PredictionError("OpenPCDet frame metadata must be a dictionary.")
    token = metadata.get("token", entry.get("sample_token"))
    if not isinstance(token, str) or not token:
        raise PredictionError("OpenPCDet frames require metadata.token (the exact dataset frame ID).")
    if "sample_token" in entry and entry["sample_token"] != token:
        raise PredictionError(f"OpenPCDet frame {token!r} has inconsistent sample tokens.")
    try:
        boxes = np.asarray(entry.get("boxes_lidar", entry.get("pred_boxes")), dtype=np.float64)
        scores = np.asarray(entry.get("score", entry.get("pred_scores")))
    except (TypeError, ValueError, OverflowError) as exc:
        raise PredictionError(f"Invalid OpenPCDet box/score arrays in frame {token!r}.") from exc
    if boxes.ndim != 2 or boxes.shape[1] not in (7, 9) or not np.isfinite(boxes).all():
        raise PredictionError("boxes_lidar must be finite with shape (N, 7) or (N, 9).")
    if scores.shape != (len(boxes),):
        raise PredictionError("OpenPCDet score array must have shape (N,) matching boxes_lidar.")
    names = entry.get("name")
    if names is not None:
        names = np.asarray(names)
        if names.shape != (len(boxes),):
            raise PredictionError("OpenPCDet name array must have shape (N,) matching boxes_lidar.")
    elif len(boxes):
        frame_mapping = _class_mapping(metadata, {}) or class_names
        if frame_mapping is None:
            raise PredictionError("Numeric prediction labels require explicit class_names metadata or a .detviz.json sidecar.")
        labels = np.asarray(entry.get("pred_labels"))
        if labels.shape != (len(boxes),) or labels.dtype.kind not in "iu" or np.any((labels < 1) | (labels > len(frame_mapping))):
            raise PredictionError("OpenPCDet pred_labels must be 1-based integers within the class_names mapping.")
        names = [frame_mapping[int(label) - 1] for label in labels]
    else:
        names = []
    rows = []
    for index, box in enumerate(boxes):
        cosine, sine = math.cos(box[6]), math.sin(box[6])
        rows.append(canonical_row(
            token, index, center=box[:3], size=box[3:6],
            rotation=[[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]],
            class_name=str(names[index]) if isinstance(names[index], np.str_) else names[index],
            score=scores[index], velocity=None if boxes.shape[1] == 7 else [*box[7:9], 0.0],
        ))
    return token, rows


def open_pcdet_pickle(path, *, cache_dir=None, progress=None, cancelled=None):
    path = Path(path).expanduser().resolve()
    writer = None
    try:
        sidecar, sidecar_path = read_sidecar(path)
        target = cache_path(path, "pcdet_pickle", cache_dir, sidecar_path, cancelled)
        source = open_cached(path, target, progress, cancelled)
        if source is not None:
            return source
        check_cancelled(cancelled)
        notify(progress, 0, f"Reading NumPy predictions: {path.name}")
        with path.open("rb") as stream:
            document = NumpyPredictionUnpickler(stream).load()
        check_cancelled(cancelled)
        metadata = {}
        if isinstance(document, dict):
            metadata = document.get("metadata", document.get("detviz", {}))
            entries = document.get("frames", document.get("predictions"))
            if not isinstance(metadata, dict):
                raise PredictionError("OpenPCDet export metadata must be a dictionary.")
        else:
            entries = document
        if not isinstance(entries, (list, tuple)):
            raise PredictionError("OpenPCDet pickle must contain a frame list, or an envelope with 'frames'.")
        class_names = _class_mapping(metadata, sidecar)
        writer = CacheWriter(target)
        for index, entry in enumerate(entries):
            check_cancelled(cancelled)
            token, rows = _frame_rows(entry, class_names)
            writer.add(token, rows)
            if index % 16 == 0:
                notify(progress, min((index + 1) / max(len(entries), 1), 0.99), f"Imported {index + 1:,} frames")
        check_cancelled(cancelled)
        writer.finish(source_info(path, "pcdet_pickle", "lidar", {"detviz": metadata}, sidecar))
        notify(progress, 1, f"Imported {writer.count:,} frames")
        return SQLitePredictionSource(target)
    except PredictionError:
        raise
    except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError, sqlite3.Error) as exc:
        raise PredictionError(f"Cannot import OpenPCDet predictions {path}: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()
