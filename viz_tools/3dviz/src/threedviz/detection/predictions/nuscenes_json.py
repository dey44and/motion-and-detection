"""Stream official nuScenes result JSON, retaining one frame in memory."""

from __future__ import annotations

import time
import sqlite3
from pathlib import Path

from .models import PredictionError
from .store import (
    CacheWriter, SQLitePredictionSource, cache_path, canonical_row,
    check_cancelled, finite_vector, notify, open_cached, quaternion_rotation,
    read_sidecar, source_info,
)


def _rows(token: str, entries: object) -> list:
    if not isinstance(entries, list):
        raise PredictionError(f"Predictions for {token!r} must be an array, including empty frames.")
    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PredictionError(f"Detection {token}:{index} must be an object.")
        if entry.get("sample_token") != token:
            raise PredictionError(f"Detection {token}:{index} has an inconsistent sample_token.")
        size = finite_vector(entry.get("size"), 3, "nuScenes width, length, height")
        velocity = None
        if "velocity" in entry:
            velocity = finite_vector(entry["velocity"], 2, "nuScenes world XY velocity") + [0.0]
        result.append(canonical_row(
            token, index, center=entry.get("translation"), size=[size[1], size[0], size[2]],
            rotation=quaternion_rotation(entry.get("rotation")), class_name=entry.get("detection_name"),
            score=entry.get("detection_score"), velocity=velocity,
        ))
    return result


def _stream_frames(stream, metadata: dict, cancelled=None):
    import ijson
    from ijson.common import ObjectBuilder

    builder = None
    depth = 0
    destination = ""
    pending_token = None
    saw_results = False
    for count, (prefix, event, value) in enumerate(ijson.parse(stream, use_float=True)):
        if count % 4096 == 0:
            check_cancelled(cancelled)
        if builder is not None:
            builder.event(event, value)
            depth += (event in {"start_map", "start_array"}) - (event in {"end_map", "end_array"})
            if depth == 0:
                if destination == "frame":
                    yield pending_token, builder.value
                else:
                    metadata[destination] = builder.value
                builder = None
            continue
        if prefix == "results":
            if event == "start_map":
                if saw_results:
                    raise PredictionError("Duplicate top-level results object.")
                saw_results = True
            elif event == "map_key":
                pending_token = value
            elif event != "end_map":
                raise PredictionError("nuScenes JSON 'results' must be an object keyed by sample token.")
        elif pending_token is not None and prefix == f"results.{pending_token}":
            if event != "start_array":
                raise PredictionError(f"Predictions for {pending_token!r} must be an array.")
            destination = "frame"
            builder = ObjectBuilder()
            builder.event(event, value)
            depth = 1
        elif prefix in {"meta", "detviz"}:
            if event == "start_map":
                destination = prefix
                builder = ObjectBuilder()
                builder.event(event, value)
                depth = 1
            elif event not in {"map_key", "end_map"}:
                metadata[prefix] = value
    if not saw_results:
        raise PredictionError("nuScenes prediction JSON requires a top-level 'results' object.")


def open_nuscenes_json(path, *, cache_dir=None, progress=None, cancelled=None):
    import ijson

    path = Path(path).expanduser().resolve()
    writer = None
    try:
        sidecar, sidecar_path = read_sidecar(path)
        target = cache_path(path, "nuscenes_json", cache_dir, sidecar_path, cancelled)
        source = open_cached(path, target, progress, cancelled)
        if source is not None:
            return source
        writer = CacheWriter(target)
        metadata = {}
        size = max(path.stat().st_size, 1)
        previous_update = 0.0
        notify(progress, 0, f"Importing {path.name}")
        with path.open("rb") as stream:
            for token, entries in _stream_frames(stream, metadata, cancelled):
                check_cancelled(cancelled)
                writer.add(token, _rows(token, entries))
                now = time.monotonic()
                if now - previous_update >= 0.1:
                    notify(progress, min(stream.tell() / size, 0.99), f"Imported {writer.count:,} frames")
                    previous_update = now
        check_cancelled(cancelled)
        writer.finish(source_info(path, "nuscenes_json", "world", metadata, sidecar))
        notify(progress, 1, f"Imported {writer.count:,} frames")
        return SQLitePredictionSource(target)
    except PredictionError:
        raise
    except (OSError, ValueError, RuntimeError, sqlite3.Error, ijson.JSONError) as exc:
        raise PredictionError(f"Cannot import nuScenes predictions {path}: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()
