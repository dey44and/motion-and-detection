"""Session-owned dataset workers, independent of HTTP and desktop rendering.

Each tab has one serial worker and one committed frame/packet. Jobs prepare a
replacement completely before committing it. Failed loads retain the prior
view, and closing a session prevents late results from becoming visible.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
import secrets
from threading import Event, RLock
import time
from typing import Callable

import numpy as np

from visual_utils import (
    Capability, DatasetConfig, DatasetConfigurationError, DatasetLoader,
    Modality, PointCloudPreparation, UnsupportedCapabilityError, load_dataset_config,
)
from threedviz.dataloader import available_datasets, create_dataset
from threedviz.processing import COLOR_SCHEMES, class_color, colorize_points

from .wire import encode_frame


_LOG = logging.getLogger(__name__)


class BrowserError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class LaunchOptions:
    dataset: str = "nuscenes"
    root: str | Path | None = None
    version: str = "v1.0-mini"
    split: str = "all"
    scene: str | None = None
    frame: int = 0
    autoload: bool = False
    config: DatasetConfig | None = None
    config_path: str | Path | None = None

    def public(self):
        root = self.config.root if self.config is not None else self.root
        return {
            "dataset": self.dataset,
            "root": str(root or ""),
            "version": (self.config.release or "") if self.config is not None else self.version,
            "split": self.split, "scene": self.scene, "frame": self.frame,
            "autoload": self.autoload,
            "config_path": None if self.config_path is None else str(self.config_path),
            "configured": self.config is not None,
        }


@dataclass
class _Connection:
    loader: DatasetLoader
    dataset: dict
    split: str
    scenes: tuple
    scene_id: str
    frames: tuple
    frame_index: int
    frame: object
    rgb: bool
    raw: bool
    revision: int
    packet: bytes


def _supports_raw(metadata):
    return (Capability.ACQUISITION_GEOMETRY in metadata.capabilities
            and metadata.feature_coverage.get(Capability.ACQUISITION_GEOMETRY, "full") != "partial")


def _groups(metadata):
    return {item.class_id: item.display_group or item.display_name for item in metadata.classes}


def _state(connection):
    if connection is None:
        return None
    metadata = connection.loader.metadata
    return {
        "dataset": dict(connection.dataset),
        "metadata": {
            "display_name": metadata.display_name,
            "capabilities": sorted(cap.value for cap in metadata.capabilities),
            "feature_coverage": {cap.value: value for cap, value in metadata.feature_coverage.items()},
            "classes": [
                {"id": item.class_id, "name": item.display_name,
                 "group": item.display_group or item.display_name,
                 "color": list(class_color(item.display_group or item.display_name))}
                for item in metadata.classes
            ],
            "splits": list(metadata.splits),
        },
        "split": connection.split,
        "scenes": [
            {"id": item.scene_id, "name": item.display_name, "description": item.description,
             "count": item.frame_counts.get(Modality.POINT_CLOUD)}
            for item in connection.scenes
        ],
        "scene_id": connection.scene_id,
        "frames": [
            {"id": item.frame_id, "index": item.index,
             "timestamp_us": None if item.timestamp_us is None else str(item.timestamp_us)}
            for item in connection.frames
        ],
        "frame_index": connection.frame_index, "revision": connection.revision,
        "has_frame": True, "display": {"rgb": connection.rgb, "raw": connection.raw},
    }


def _close_loader(loader):
    if loader is not None:
        try:
            loader.close()
        except Exception:
            _LOG.exception("Dataset cleanup failed")


class BrowserSession:
    """One isolated tab; only its worker touches dataset readers."""

    def __init__(self, options: LaunchOptions, factory: Callable, *, history_limit: int = 8):
        self.id = secrets.token_urlsafe(24)
        self.options = options
        self._factory = factory
        self._history_limit = history_limit
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="3dviz-browser")
        self._closed = False
        self._disposed = Event()
        self._connection = None
        self._jobs = OrderedDict()
        self._active_job = None
        self._revision = 0
        self.last_access = time.monotonic()

    def touch(self):
        with self._lock:
            self._check_open()
            self.last_access = time.monotonic()

    def _check_open(self):
        if self._closed:
            raise BrowserError(404, "This browser session has closed. Reload the page to reconnect.")

    def snapshot(self):
        with self._lock:
            self._check_open()
            return _state(self._connection)

    def packet(self, revision: int):
        with self._lock:
            self._check_open()
            if self._connection is None:
                raise BrowserError(409, "Load a dataset before requesting a frame.")
            if revision != self._connection.revision:
                raise BrowserError(409, "The requested frame revision is no longer current.")
            return self._connection.packet

    def start(self, values: dict):
        values = dict(values)
        op = values.get("op")
        if op not in {"connect", "select", "prepare"}:
            raise BrowserError(422, "Unknown browser operation.")
        common = {"op", "rgb", "raw"}
        allowed = common | ({"dataset", "root", "version", "config_path", "split", "scene_id", "frame_index"}
                            if op == "connect" else {"split", "scene_id", "frame_index"} if op == "select" else set())
        if values.keys() - allowed:
            raise BrowserError(422, "This operation contains fields belonging to a different operation.")
        with self._lock:
            self._check_open()
            if self._active_job is not None:
                raise BrowserError(409, "This session is already loading. Wait for the current operation.")
            if op != "connect" and self._connection is None:
                raise BrowserError(409, "Connect a dataset before selecting or preparing a frame.")
            job_id = secrets.token_urlsafe(18)
            self._jobs[job_id] = {"id": job_id, "status": "running", "stage": "Queued"}
            while len(self._jobs) > self._history_limit:
                self._jobs.popitem(last=False)
            self._active_job = job_id
            self._executor.submit(self._run, job_id, values)
            return {"id": job_id}

    def job(self, job_id):
        with self._lock:
            self._check_open()
            try:
                result = deepcopy(self._jobs[job_id])
            except KeyError:
                raise BrowserError(404, "This loading operation was not found.") from None
            return result

    def _stage(self, job_id, text):
        with self._lock:
            if self._closed:
                raise BrowserError(404, "Browser session closed during loading.")
            self._jobs[job_id]["stage"] = text

    def _run(self, job_id, values):
        candidate = None
        previous = self._connection
        try:
            if values["op"] == "connect":
                candidate = self._connect(job_id, values)
            else:
                candidate = self._select_or_prepare(job_id, values, previous)
            state = _state(candidate)
            with self._lock:
                if self._closed:
                    return
                self._connection = candidate
                self._revision = candidate.revision
                self._jobs[job_id].update(status="done", stage="Ready", state=state)
                self._active_job = None
            if previous is not None and previous.loader is not candidate.loader:
                _close_loader(previous.loader)
        except Exception as error:
            with self._lock:
                if not self._closed:
                    self._jobs[job_id].update(status="error", stage="Loading failed", error=str(error))
                    self._active_job = None
        finally:
            if candidate is not None and candidate is not self._connection:
                if previous is None or candidate.loader is not previous.loader:
                    _close_loader(candidate.loader)
            with self._lock:
                if self._active_job == job_id:
                    self._active_job = None

    def _source(self, values):
        fixed = self.options.config
        if fixed is not None:
            expected = self.options.public()
            for key in ("dataset", "root", "version", "config_path"):
                if key in values and values[key] is not None and str(values[key]) != str(expected[key] or ""):
                    raise DatasetConfigurationError("This launch uses a fixed dataset configuration; source fields cannot change.")
            return self.options.dataset, fixed, "", None
        name = str(values.get("dataset") or self.options.dataset).strip().lower()
        config_path = values.get("config_path")
        if config_path:
            if values.get("root") or values.get("version"):
                raise DatasetConfigurationError("Use a configuration file or root/version, rather than both.")
            if name == "demo":
                raise DatasetConfigurationError("Demo data does not need a dataset configuration.")
            return name, load_dataset_config(config_path), "", None
        # Launch fields are bootstrap defaults; a later connection must not
        # inherit an earlier provider's source directory or release.
        root = values.get("root")
        version = values.get("version") or "v1.0-mini"
        return name, None, version, None if not root else Path(root).expanduser()

    def _connect(self, job_id, values):
        self._stage(job_id, "Opening dataset")
        name, config, version, root = self._source(values)
        loader = self._factory(name, config=config) if config is not None else self._factory(name, root=root, version=version)
        try:
            metadata = loader.metadata
            if Capability.POINT_CLOUD not in metadata.capabilities:
                raise UnsupportedCapabilityError("This dataset does not provide point clouds.")
            split = values.get("split") or "all"
            self._stage(job_id, "Listing scenes")
            scenes = self._scenes(loader, split)
            scene = values.get("scene_id")
            selected = self._scene(scenes, scene)
            frames = loader.list_frames(selected.scene_id, modality=Modality.POINT_CLOUD)
            index = values.get("frame_index", 0)
            info = self._frame_info(frames, index)
            rgb = bool(values.get("rgb", False)) and Capability.RGB_COLORING in metadata.capabilities
            raw = bool(values.get("raw", False)) and _supports_raw(metadata)
            self._stage(job_id, "Reading point cloud")
            frame = loader.load_point_cloud(info.frame_id)
            frame = self._prepare(job_id, loader, frame, rgb, raw)
            dataset = {"name": name, "root": str(loader.config.root or ""),
                       "version": (config.release or "") if config is not None else metadata.release or loader.config.release or "",
                       "configured": config is not None}
            return self._candidate(job_id, loader, dataset, split, scenes, selected.scene_id, frames, index, frame, rgb, raw)
        except Exception:
            _close_loader(loader)
            raise

    @staticmethod
    def _scenes(loader, split):
        scenes = tuple(scene for scene in loader.list_scenes(split=split)
                       if scene.frame_counts.get(Modality.POINT_CLOUD) != 0)
        if not scenes:
            raise ValueError("This split contains no scenes with point clouds.")
        return scenes

    @staticmethod
    def _scene(scenes, wanted):
        if wanted is None or wanted == "":
            return scenes[0]
        for item in scenes:
            if wanted in (item.scene_id, item.display_name):
                return item
        raise ValueError(f"Scene {wanted!r} was not found in this split.")

    @staticmethod
    def _frame_info(frames, index):
        if type(index) is not int or not 0 <= index < len(frames):
            if not frames:
                raise ValueError("This scene contains no point-cloud frames.")
            raise ValueError(f"Frame index must be between 0 and {len(frames) - 1}.")
        return frames[index]

    def _prepare(self, job_id, loader, frame, rgb, raw):
        metadata = loader.metadata
        if rgb and Capability.RGB_COLORING not in metadata.capabilities:
            raise UnsupportedCapabilityError("Camera RGB is unavailable for this dataset.")
        if raw and not _supports_raw(metadata):
            raise UnsupportedCapabilityError("Complete acquisition geometry is unavailable for this dataset.")
        need_rgb = rgb and frame.rgb is None
        need_raw = raw and frame.acquisition is None
        if need_rgb or need_raw:
            self._stage(job_id, "Preparing camera RGB" if need_rgb else "Preparing acquisition geometry")
            frame = loader.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=need_rgb, acquisition_geometry=need_raw))
        if rgb and frame.rgb is None:
            raise ValueError("The dataset did not provide the requested camera RGB colors.")
        if raw and (frame.acquisition is None or not frame.acquisition.valid_mask.all()):
            raise ValueError("Acquisition geometry must cover every point in this scan.")
        return frame

    def _select_or_prepare(self, job_id, values, connection):
        loader = connection.loader
        rgb = values.get("rgb", connection.rgb)
        raw = values.get("raw", connection.raw)
        if values["op"] == "prepare":
            split, scenes, scene_id, frames, index, frame = (
                connection.split, connection.scenes, connection.scene_id,
                connection.frames, connection.frame_index, connection.frame,
            )
        else:
            split = values.get("split") or connection.split
            if split != connection.split:
                self._stage(job_id, "Listing scenes")
                scenes = self._scenes(loader, split)
            else:
                scenes = connection.scenes
            wanted = values.get("scene_id")
            if wanted is None:
                wanted = connection.scene_id if any(item.scene_id == connection.scene_id for item in scenes) else None
            selected = self._scene(scenes, wanted)
            scene_id = selected.scene_id
            changed_scene = scene_id != connection.scene_id
            frames = loader.list_frames(scene_id, modality=Modality.POINT_CLOUD) if changed_scene else connection.frames
            index = values.get("frame_index", 0 if changed_scene else connection.frame_index)
            info = self._frame_info(frames, index)
            if info.frame_id == connection.frame.info.frame_id:
                frame = connection.frame
            else:
                self._stage(job_id, "Reading point cloud")
                frame = loader.load_point_cloud(info.frame_id)
        frame = self._prepare(job_id, loader, frame, rgb, raw)
        return self._candidate(job_id, loader, connection.dataset, split, scenes, scene_id, frames, index, frame, rgb, raw)

    def _candidate(self, job_id, loader, dataset, split, scenes, scene_id, frames, index, frame, rgb, raw):
        if frame.info.frame_id != frames[index].frame_id or frame.info.scene_id != scene_id:
            raise ValueError("The dataset returned a point cloud for a different observation.")
        revision = self._revision + 1
        self._stage(job_id, "Transferring geometry")
        packet = encode_frame(frame, revision, rgb=rgb, raw=raw, class_groups=_groups(loader.metadata))
        return _Connection(loader, dataset, split, scenes, scene_id, frames, index, frame, rgb, raw, revision, packet)

    def close(self, *, wait=False):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._executor.submit(self._dispose)
        # The disposal task follows every pending read on this same worker.
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _dispose(self):
        with self._lock:
            connection, self._connection = self._connection, None
            self._jobs.clear()
        if connection is not None:
            _close_loader(connection.loader)
        self._disposed.set()


class SessionManager:
    """Bounded tab sessions, with one frame/packet and short job history each."""

    def __init__(self, options: LaunchOptions | None = None, *, factory=create_dataset,
                 catalog=available_datasets, max_sessions=8, idle_timeout=1800.0):
        if max_sessions < 1 or idle_timeout <= 0:
            raise ValueError("Session limits must be positive.")
        self.options = options or LaunchOptions()
        self._factory, self._catalog = factory, catalog
        self._max_sessions, self._idle_timeout = max_sessions, idle_timeout
        self._sessions = {}
        self._closing = []
        self._lock = RLock()
        self._closed = False

    def bootstrap(self):
        samples = np.column_stack((np.linspace(0, 1, 256), np.zeros((256, 2))))
        return {"datasets": list(self._catalog()), "initial": self.options.public(),
                "palettes": {name: colorize_points(samples, name, 1.0).tolist() for name in COLOR_SCHEMES}}

    def create(self):
        with self._lock:
            if self._closed:
                raise BrowserError(503, "The viewer is shutting down.")
            self._reap()
            if len(self._sessions) + len(self._closing) >= self._max_sessions:
                raise BrowserError(429, "Too many viewer tabs are open. Close an existing tab first.")
            session = BrowserSession(self.options, self._factory)
            self._sessions[session.id] = session
            return session

    def _reap(self):
        self._closing = [session for session in self._closing if not session._disposed.is_set()]
        now = time.monotonic()
        for session_id, session in tuple(self._sessions.items()):
            if now - session.last_access > self._idle_timeout:
                self._sessions.pop(session_id)
                session.close()
                self._closing.append(session)

    def expire(self):
        with self._lock:
            self._reap()

    def get(self, session_id):
        with self._lock:
            self._reap()
            try:
                session = self._sessions[session_id]
            except KeyError:
                raise BrowserError(404, "This viewer session expired. Reload the page to reconnect.") from None
            session.touch()
            return session

    def delete(self, session_id):
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.close()
                self._closing.append(session)

    def close(self):
        with self._lock:
            self._closed = True
            sessions = [*self._sessions.values(), *self._closing]
            self._sessions.clear()
            self._closing.clear()
        for session in sessions:
            session.close(wait=True)
