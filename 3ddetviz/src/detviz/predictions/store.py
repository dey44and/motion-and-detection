"""Atomic SQLite caches keep multi-gigabyte exports out of viewer memory."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import threading
import zlib
from contextlib import closing
from pathlib import Path
from typing import Callable

import numpy as np

from dataset_utils import Box3D, DatasetValidationError

from .base import PredictionSource
from .models import Detection, PredictionCancelled, PredictionError, PredictionFrame, PredictionInfo

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]
_CACHE_VERSION = 2
_PRODUCER_NOTE = (
    "Typical nuScenes/OpenPCDet evaluation files already contain score-filtered and NMS-filtered detections. "
    "The processing stage is unknown unless explicitly declared; filters operate on the saved boxes."
)


def check_cancelled(cancelled: Cancelled | None) -> None:
    if cancelled is not None and cancelled():
        raise PredictionCancelled("Prediction import cancelled.")


def notify(progress: Progress | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(max(0.0, min(float(fraction), 1.0)), message)


def read_sidecar(path: Path) -> tuple[dict, Path | None]:
    candidates = (Path(str(path) + ".detviz.json"), path.with_suffix(".detviz.json"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if candidate.stat().st_size > 1_048_576:
                raise PredictionError(f"Prediction metadata sidecar exceeds 1 MiB: {candidate}")
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PredictionError(f"Cannot read prediction metadata {candidate}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise PredictionError("Prediction metadata sidecar must contain a JSON object.")
        return metadata, candidate
    return {}, None


def source_info(path: Path, format_name: str, coordinate_frame: str, embedded: dict, sidecar: dict) -> PredictionInfo:
    metadata = dict(embedded)
    declarations = {}
    for source in (embedded.get("meta", {}), embedded.get("detviz", {}), sidecar):
        if isinstance(source, dict):
            declarations.update(source)
            if isinstance(source.get("detviz"), dict):
                declarations.update(source["detviz"])
    declared_coordinates = declarations.get("coordinate_frame", coordinate_frame)
    if declared_coordinates != coordinate_frame:
        raise PredictionError(
            f"{format_name} stores {coordinate_frame!r} boxes; metadata declares incompatible "
            f"coordinate_frame {declared_coordinates!r}."
        )
    metadata.update({"source_path": str(path), "sidecar": sidecar, "producer_note": _PRODUCER_NOTE})
    return PredictionInfo(
        name=str(declarations.get("name", path.name)), format=format_name,
        coordinate_frame=coordinate_frame, stage=declarations.get("stage", "unknown"), metadata=metadata,
    )


def cache_path(path: Path, format_name: str, cache_dir: str | Path | None, sidecar_path: Path | None, cancelled=None) -> Path:
    stat = path.stat()
    identity = [str(path.resolve()), format_name, _CACHE_VERSION, stat.st_dev, stat.st_ino,
                stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    if sidecar_path is not None:
        identity.append(sidecar_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = (
        Path(cache_dir).expanduser() if cache_dir is not None
        else Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "3ddetviz" / "predictions"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{path.stem[:32]}-{digest}.sqlite3"


def finite_vector(value: object, count: int, name: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise PredictionError(f"{name} requires {count} finite numbers.") from exc
    if len(result) != count or not all(math.isfinite(number) for number in result):
        raise PredictionError(f"{name} requires {count} finite numbers.")
    return result


def score_value(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PredictionError("Detection score must be finite and in [0, 1].") from exc
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(result) or not 0 <= result <= 1:
        raise PredictionError("Detection score must be finite and in [0, 1].")
    return result


def quaternion_rotation(value: object) -> list[list[float]]:
    quaternion = finite_vector(value, 4, "WXYZ quaternion")
    norm = math.sqrt(sum(number * number for number in quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise PredictionError("Detection quaternion must have nonzero finite norm.")
    w, x, y, z = (number / norm for number in quaternion)
    return [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]


def canonical_row(token: str, index: int, *, center, size, rotation, class_name, score, velocity=None) -> list:
    if not isinstance(class_name, str) or not class_name.strip():
        raise PredictionError("Detection class name must be a nonempty string.")
    center = finite_vector(center, 3, "Detection center")
    size = finite_vector(size, 3, "Detection length, width, height")
    if any(dimension <= 0 for dimension in size):
        raise PredictionError("Detection dimensions must be positive length, width, height.")
    if velocity is not None:
        velocity = finite_vector(velocity, 3, "Detection velocity")
    return [center, size, rotation, class_name, score_value(score), f"{token}:{index}", velocity]


def frame_from_rows(token: str, rows: list) -> PredictionFrame:
    return PredictionFrame(token, tuple(
        Detection(Box3D(row[5], row[3], row[0], row[1], row[2]),
                  row[4], row[5], row[6])
        for row in rows
    ))


class SQLitePredictionSource(PredictionSource):
    """Connections live for one query, allowing safe worker/main-thread access."""

    def __init__(self, path: str | Path):
        self.cache_path = Path(path)
        self._closed = False
        self._lock = threading.RLock()
        try:
            with closing(self._connect()) as connection:
                metadata = json.loads(connection.execute("SELECT value FROM metadata WHERE key='source'").fetchone()[0])
                if metadata["cache_version"] != _CACHE_VERSION:
                    raise PredictionError("Prediction cache version is incompatible.")
                self._info = PredictionInfo(**metadata["info"])
                self._class_names = tuple(metadata["class_names"])
                self._frame_tokens = frozenset(row[0] for row in connection.execute("SELECT token FROM frames"))
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            raise PredictionError(f"Cannot read prediction cache {self.cache_path}: {exc}") from exc

    def _connect(self):
        return sqlite3.connect(self.cache_path.resolve().as_uri() + "?mode=ro", uri=True)

    @property
    def info(self):
        return self._info

    @property
    def class_names(self):
        return self._class_names

    @property
    def frame_tokens(self):
        return self._frame_tokens

    def get_frame(self, token: str) -> PredictionFrame | None:
        with self._lock:
            if self._closed:
                raise PredictionError("Prediction source is closed.")
            try:
                with closing(self._connect()) as connection:
                    record = connection.execute("SELECT payload FROM frames WHERE token=?", (token,)).fetchone()
            except sqlite3.Error as exc:
                raise PredictionError(f"Cannot read cached predictions for {token!r}: {exc}") from exc
        if record is None:
            return None
        try:
            return frame_from_rows(token, json.loads(zlib.decompress(record[0])))
        except (DatasetValidationError, ValueError, TypeError, IndexError, zlib.error) as exc:
            raise PredictionError(f"Invalid cached predictions for {token!r}: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            self._closed = True


class CacheWriter:
    def __init__(self, target: Path):
        self.target = target
        descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        os.close(descriptor)
        self.temporary = Path(name)
        self.connection = sqlite3.connect(self.temporary)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("CREATE TABLE frames(token TEXT PRIMARY KEY, payload BLOB NOT NULL)")
        self.connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.class_names = set()
        self.count = 0
        self.published = False

    def add(self, token: str, rows: list) -> None:
        if not isinstance(token, str) or not token:
            raise PredictionError("Prediction sample token must be a nonempty string.")
        try:
            payload = zlib.compress(json.dumps(rows, separators=(",", ":"), allow_nan=False).encode(), level=3)
            self.connection.execute("INSERT INTO frames VALUES (?,?)", (token, payload))
        except sqlite3.IntegrityError as exc:
            raise PredictionError(f"Duplicate prediction frame {token!r}.") from exc
        self.class_names.update(row[3] for row in rows)
        self.count += 1
        if self.count % 128 == 0:
            self.connection.commit()

    def finish(self, info: PredictionInfo) -> None:
        record = {
            "cache_version": _CACHE_VERSION,
            "info": {"name": info.name, "format": info.format, "coordinate_frame": info.coordinate_frame,
                     "stage": info.stage, "metadata": dict(info.metadata)},
            "class_names": sorted(self.class_names),
        }
        self.connection.execute("INSERT INTO metadata VALUES ('source',?)", (json.dumps(record, allow_nan=False),))
        self.connection.commit()
        self.connection.close()
        os.replace(self.temporary, self.target)
        self.published = True

    def close(self) -> None:
        self.connection.close()
        if not self.published:
            self.temporary.unlink(missing_ok=True)


def open_cached(path: Path, target: Path, progress: Progress | None, cancelled: Cancelled | None):
    check_cancelled(cancelled)
    if target.is_file():
        try:
            source = SQLitePredictionSource(target)
        except PredictionError:
            target.unlink(missing_ok=True)
        else:
            notify(progress, 1, f"Loaded cached predictions: {path.name}")
            return source
    return None
