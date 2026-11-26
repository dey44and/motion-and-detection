"""A local, streaming index of native nuScenes metadata tables."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any

from ..errors import DatasetError, DatasetValidationError


_SCHEMA_VERSION = 1
_BATCH_SIZE = 8192
_SOURCES = ("sample", "instance", "category", "ego_pose", "sample_data", "sample_annotation")
_OPTIONAL = frozenset({"instance", "category", "sample_annotation"})

_SCHEMA = """
CREATE TABLE cache_info (signature TEXT NOT NULL, pose_count INTEGER NOT NULL, sample_data_count INTEGER NOT NULL);
CREATE TABLE ego_pose (ordinal INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE, record TEXT NOT NULL);
CREATE TABLE sample_data (
    ordinal INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE, scene_token TEXT,
    calibration_token TEXT, is_key_frame INTEGER NOT NULL, record TEXT NOT NULL
);
CREATE TABLE sample_annotation (
    ordinal INTEGER PRIMARY KEY, sample_token TEXT NOT NULL, class_name TEXT NOT NULL, record TEXT NOT NULL
);
CREATE TABLE calibration_tokens (token TEXT PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE annotation_classes (name TEXT PRIMARY KEY) WITHOUT ROWID;
"""


class MetadataCacheError(RuntimeError):
    """The local metadata cache cannot be opened, queried, or written."""


@dataclass(frozen=True)
class MetadataSourceSnapshot:
    root: Path
    sources: tuple[tuple[str, tuple | None], ...]

    def source(self, name: str) -> Path | None:
        identity = dict(self.sources)[name]
        return None if identity is None else Path(identity[0])

    @property
    def signature(self) -> str:
        return json.dumps([_SCHEMA_VERSION, str(self.root), self.sources], separators=(",", ":"))


def metadata_source_snapshot(table_root: Path) -> MetadataSourceSnapshot:
    """Fingerprint index inputs and the tables used to classify or group rows."""
    root = Path(table_root).resolve()
    sources = []
    for name in _SOURCES:
        path = root / f"{name}.json"
        try:
            resolved = path.resolve(strict=True)
            info = resolved.stat()
        except FileNotFoundError as error:
            if name not in _OPTIONAL:
                raise DatasetError(f"Cannot read nuScenes table {path}: {error}") from error
            identity = None
        except (OSError, RuntimeError) as error:
            raise DatasetError(f"Cannot read nuScenes table {path}: {error}") from error
        else:
            if not stat.S_ISREG(info.st_mode):
                raise DatasetValidationError(f"Invalid nuScenes table {path}: expected a regular file.")
            identity = (str(resolved), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        sources.append((name, identity))
    return MetadataSourceSnapshot(root, tuple(sources))


def _records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import ijson
    except ImportError as error:
        raise MetadataCacheError("Building a nuScenes metadata index requires ijson.") from error

    try:
        with path.open("rb") as stream:
            while (prefix := stream.peek(1)) and not (prefix.lstrip(b" \t\r\n")):
                stream.read(len(prefix))
            if not prefix.lstrip(b" \t\r\n").startswith(b"["):
                raise DatasetValidationError(f"Invalid nuScenes table {path}: expected a JSON array of records.")
            for record in ijson.items(stream, "item", use_float=True):
                if not isinstance(record, dict):
                    raise DatasetValidationError(f"Invalid nuScenes table {path}: expected a JSON array of records.")
                yield record
    except OSError as error:
        raise DatasetError(f"Cannot read nuScenes table {path}: {error}") from error
    except (ijson.JSONError, UnicodeError) as error:
        raise DatasetValidationError(f"Cannot read nuScenes table {path}: {error}") from error


def _token(record: dict, table: str, field: str = "token") -> str:
    token = record.get(field)
    if not isinstance(token, str) or not token:
        raise DatasetValidationError(f"Invalid {table}.json record: expected a nonempty {field}.")
    return token


def _encode(record: dict) -> str:
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False)


def _insert_rows(connection, table, sql, rows, progress, *, unique_tokens=False) -> int:
    count = 0
    for batch in iter(lambda: list(islice(rows, _BATCH_SIZE)), []):
        if unique_tokens:
            connection.execute("SAVEPOINT token_batch")
        try:
            connection.executemany(sql, batch)
        except sqlite3.IntegrityError:
            if not unique_tokens:
                raise
            connection.execute("ROLLBACK TO token_batch")
            for record in batch:
                try:
                    connection.execute(sql, record)
                except sqlite3.IntegrityError as error:
                    raise DatasetValidationError(f"Duplicate token {record[1]!r} in {table}.json.") from error
            raise
        if unique_tokens:
            connection.execute("RELEASE token_batch")
        count += len(batch)
        if progress is not None and count % (_BATCH_SIZE * 8) == 0:
            progress(f"Indexing nuScenes {table} ({count:,} records)")
    return count


def _build(temporary, source, sample_scene_tokens, annotation_class, progress):
    with closing(sqlite3.connect(temporary)) as db:
        db.executescript(_SCHEMA)
        db.execute("BEGIN")
        if progress is not None:
            progress("Indexing nuScenes ego_pose")
        pose_count = _insert_rows(
            db, "ego_pose", "INSERT INTO ego_pose VALUES (?, ?, ?)",
            ((ordinal, _token(record, "ego_pose"), _encode(record))
             for ordinal, record in enumerate(_records(source.source("ego_pose")), 1)),
            progress, unique_tokens=True,
        )

        def sample_data_rows():
            for ordinal, record in enumerate(_records(source.source("sample_data")), 1):
                sample = record.get("sample_token")
                calibration = record.get("calibrated_sensor_token")
                yield (
                    ordinal, _token(record, "sample_data"),
                    sample_scene_tokens.get(sample) if isinstance(sample, str) else None,
                    calibration if isinstance(calibration, str) else None,
                    bool(record.get("is_key_frame", False)), _encode(record),
                )

        if progress is not None:
            progress("Indexing nuScenes sample_data")
        data_count = _insert_rows(
            db, "sample_data", "INSERT INTO sample_data VALUES (?, ?, ?, ?, ?, ?)",
            sample_data_rows(), progress, unique_tokens=True,
        )

        def annotations(path):
            for ordinal, record in enumerate(_records(path), 1):
                class_name = annotation_class(record)
                sample = _token(record, "sample_annotation", "sample_token")
                if not isinstance(class_name, str) or not class_name:
                    raise DatasetValidationError(f"Invalid category for annotation {record.get('token')!r}.")
                yield ordinal, sample, class_name, _encode(record)

        if (path := source.source("sample_annotation")) is not None:
            if progress is not None:
                progress("Indexing nuScenes sample_annotation")
            _insert_rows(
                db, "sample_annotation", "INSERT INTO sample_annotation VALUES (?, ?, ?, ?)",
                annotations(path), progress,
            )

        if progress is not None:
            progress("Finalizing nuScenes metadata index")
        db.execute("CREATE INDEX sample_data_by_scene ON sample_data (scene_token, ordinal)")
        db.execute("CREATE INDEX sample_data_keyframes ON sample_data (ordinal) WHERE is_key_frame = 1")
        db.execute("CREATE INDEX sample_annotation_by_sample ON sample_annotation (sample_token, ordinal)")
        db.execute("INSERT INTO calibration_tokens SELECT DISTINCT calibration_token FROM sample_data WHERE calibration_token IS NOT NULL")
        db.execute("INSERT INTO annotation_classes SELECT DISTINCT class_name FROM sample_annotation")
        db.execute("INSERT INTO cache_info VALUES (?, ?, ?)", (source.signature, pose_count, data_count))
        db.commit()


class _TokenRecords(Mapping[str, dict[str, Any]]):
    def __init__(self, index: MetadataIndex, table: str, count: int):
        self._index = index
        self._table = table
        self._count = count

    def __getitem__(self, token: str) -> dict[str, Any]:
        if not isinstance(token, str):
            raise KeyError(token)
        with closing(self._index._rows(f"SELECT record FROM {self._table} WHERE token = ?", (token,))) as rows:
            result = next(rows, None)
        if result is None:
            raise KeyError(token)
        return self._index._decode(result[0])

    def __iter__(self) -> Iterator[str]:
        return (token for token, in self._index._rows(f"SELECT token FROM {self._table} ORDER BY ordinal"))

    def __len__(self) -> int:
        return self._count


class MetadataIndex:
    """Read one completed metadata index using serialized queries."""

    def __init__(self, path: Path, db: sqlite3.Connection, pose_count: int, data_count: int):
        self.path = path
        self._db = db
        self.poses: Mapping[str, dict[str, Any]] = _TokenRecords(self, "ego_pose", pose_count)
        self.sample_data: Mapping[str, dict[str, Any]] = _TokenRecords(self, "sample_data", data_count)

    def _rows(self, query, parameters=()):
        try:
            with closing(self._db.execute(query, parameters)) as cursor:
                yield from cursor
        except sqlite3.Error as error:
            raise MetadataCacheError(f"Cannot query nuScenes metadata cache {self.path}: {error}") from error

    def _decode(self, record: str) -> dict[str, Any]:
        try:
            result = json.loads(record)
        except (TypeError, ValueError) as error:
            raise MetadataCacheError(f"Invalid record in nuScenes metadata cache {self.path}.") from error
        if not isinstance(result, dict):
            raise MetadataCacheError(f"Invalid record in nuScenes metadata cache {self.path}.")
        return result

    def keyframes(self) -> Iterator[dict[str, Any]]:
        return (self._decode(row) for row, in self._rows("SELECT record FROM sample_data WHERE is_key_frame = 1 ORDER BY ordinal"))

    def scene_sample_data(self, scene_token: str) -> Iterator[dict[str, Any]]:
        return (self._decode(row) for row, in self._rows(
            "SELECT record FROM sample_data WHERE scene_token = ? ORDER BY ordinal", (scene_token,),
        ))

    def calibration_tokens(self) -> tuple[str, ...]:
        return tuple(token for token, in self._rows("SELECT token FROM calibration_tokens ORDER BY token"))

    def annotation_classes(self) -> tuple[str, ...]:
        return tuple(name for name, in self._rows("SELECT name FROM annotation_classes ORDER BY name"))

    def annotations(self, frame_id: str) -> tuple[tuple[dict[str, Any], str], ...]:
        return tuple((self._decode(record), name) for record, name in self._rows(
            "SELECT record, class_name FROM sample_annotation WHERE sample_token = ? ORDER BY ordinal", (frame_id,),
        ))

    def close(self) -> None:
        self._db.close()


def _open_cached(path: Path, signature: str) -> MetadataIndex | None:
    if not path.is_file():
        return None
    db = None
    try:
        db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        db.execute("PRAGMA query_only=ON")
        row = db.execute("SELECT signature, pose_count, sample_data_count FROM cache_info").fetchone()
        if row is not None and row[0] == signature:
            result = MetadataIndex(path, db, row[1], row[2])
            db = None
            return result
    except sqlite3.DatabaseError:
        # A damaged or incomplete cache can be rebuilt from the unchanged JSON.
        return None
    finally:
        if db is not None:
            db.close()
    return None


@contextmanager
def _build_lock(path: Path, progress):
    # The tiny SQLite database is a lock file only. A process exit releases its
    # OS-backed transaction lock, including an exit during a network read.
    with closing(sqlite3.connect(path, timeout=1)) as lock:
        while True:
            try:
                lock.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                if getattr(error, "sqlite_errorcode", None) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) and "locked" not in str(error):
                    raise
                if progress is not None:
                    progress("Waiting for the nuScenes metadata index")
            else:
                break
        yield


def open_metadata_index(
    table_root: Path, sample_scene_tokens: Mapping[str, str], annotation_class: Callable[[dict], str], *,
    progress: Callable[[str], None] | None = None, source_snapshot: MetadataSourceSnapshot | None = None,
) -> MetadataIndex:
    """Open a reusable index, streaming the source tables on a cache miss."""
    source = metadata_source_snapshot(table_root)
    if source_snapshot is not None and source != source_snapshot:
        raise DatasetValidationError("nuScenes metadata changed while the dataset was opening. Reopen the dataset.")
    signature = source.signature
    root_digest = hashlib.sha256(str(source.root).encode()).hexdigest()[:24]
    digest = hashlib.sha256(signature.encode()).hexdigest()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    root = (cache_home / "visual-utils" / "nuscenes").expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
        path = root / f"{source.root.name[:32]}-{digest}.sqlite3"
        if (cached := _open_cached(path, signature)) is not None:
            return cached
        temporary = root / f"{root_digest}.partial"
        with _build_lock(root / f"{root_digest}.lock", progress):
            if (cached := _open_cached(path, signature)) is not None:
                return cached
            try:
                temporary.unlink(missing_ok=True)
                temporary.with_name(temporary.name + "-journal").unlink(missing_ok=True)
                _build(temporary, source, sample_scene_tokens, annotation_class, progress)
                if source != metadata_source_snapshot(table_root):
                    raise DatasetValidationError("nuScenes metadata changed during indexing. Reopen the dataset.")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
                temporary.with_name(temporary.name + "-journal").unlink(missing_ok=True)
            if (cached := _open_cached(path, signature)) is not None:
                return cached
            raise MetadataCacheError(f"Cannot validate the completed nuScenes metadata index {path}.")
    except (OSError, sqlite3.Error) as error:
        raise MetadataCacheError(f"Cannot access nuScenes metadata cache {root}: {error}") from error
