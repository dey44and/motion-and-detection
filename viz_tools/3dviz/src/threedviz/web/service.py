"""Session-owned dataset and prediction workers.

Each tab has one serial worker and one committed frame/packet. Jobs prepare a
replacement completely before committing it. Failed loads retain the prior
view, and closing a session prevents late results from becoming visible.
"""

from __future__ import annotations

from base64 import b64encode
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
import logging
from pathlib import Path
import secrets
from threading import Event, RLock
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from visual_utils import (
    Capability, DatasetConfig, DatasetConfigurationError, DatasetError, DatasetLoader, DataUnavailableError,
    Modality, PointCloudFrame, PointCloudPreparation, UnsupportedCapabilityError, load_dataset_config,
)
from threedviz.dataloader import available_datasets, create_dataset
from threedviz.detection.predictions import PredictionError, available_prediction_formats
from threedviz.processing import COLOR_SCHEMES, class_color, colorize_points

from .detection import DetectionReview
from .wire import encode_frame

if TYPE_CHECKING:
    from .cameras import CameraReview
    from .segmentation import FrameSegmentation, PredictionRun


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
    predictions: str | Path | None = None
    prediction_format: str = "auto"
    cache_dir: str | Path | None = None

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
            "prediction_path": None if self.predictions is None else str(self.predictions),
            "prediction_format": self.prediction_format,
        }


@dataclass(frozen=True)
class _Segmentation:
    mode: str = "off"
    run: PredictionRun | None = None
    result: FrameSegmentation | None = None
    generation: int = 0

    def for_frame(self, loader, frame):
        from .segmentation import frame_segmentation

        result = frame_segmentation(loader, frame, self.mode, self.run)
        return replace(self, result=result)

    def public(self, metadata):
        state = self.result.public()
        return {
            **state, "message": state["reason"] or "",
            "available": Capability.LIDAR_SEGMENTATION in metadata.capabilities,
            "mode": self.mode, "path": None if self.run is None else str(self.run.path),
            "format": "auto" if self.run is None else self.run.format, "generation": self.generation,
            "run": None if self.run is None else self.run.public(),
        }


@dataclass(frozen=True)
class _Connection:
    loader: DatasetLoader
    dataset: dict
    split: str
    scenes: tuple
    scene_id: str
    frames: tuple
    frame_index: int
    frame: PointCloudFrame
    rgb: bool
    raw: bool
    revision: int
    packet: bytes
    detection: DetectionReview
    segmentation: _Segmentation
    camera_rig: tuple[dict, ...] = ()
    camera_review: CameraReview | None = None
    camera_review_dirty: bool = False


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
        "detection": connection.detection.public(metadata),
        "segmentation": connection.segmentation.public(metadata),
        "camera_rig": list(connection.camera_rig),
        "camera_review": (None if connection.camera_review is None or connection.camera_review_dirty
                          else connection.camera_review.public()),
    }


def _close_resource(resource):
    if resource is not None:
        try:
            resource.close()
        except Exception:
            _LOG.exception("Viewer resource cleanup failed")


def _close_replaced(connection, retained=None):
    if connection is None:
        return
    if retained is None or connection.detection.source is not retained.detection.source:
        _close_resource(connection.detection.source)
    if retained is None or connection.loader is not retained.loader:
        _close_resource(connection.loader)


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
        self._camera_preview_job = None
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

    def semantic_labels(self, revision: int, generation: int, *, native: bool = False):
        with self._lock:
            self._check_open()
            connection = self._connection
            if connection is None or connection.revision != revision or connection.segmentation.generation != generation:
                raise BrowserError(409, "The requested LiDAR labels are no longer current.")
            result = connection.segmentation.result
            labels = result.native_labels if native else result.labels
            if labels is None:
                raise BrowserError(404, "The requested LiDAR labels are unavailable for this scan.")
            return labels.tobytes()

    def _camera_frame(self, revision: int):
        self._check_open()
        connection = self._connection
        if connection is None or connection.revision != revision:
            raise BrowserError(409, "The requested camera frame is no longer current.")
        return connection

    def camera_previews(self, revision: int):
        with self._lock:
            connection = self._camera_frame(revision)
            if not any(marker.get("corner_rays") for marker in connection.camera_rig):
                return {}
            if self._camera_preview_job is None:
                self._camera_preview_job = self._executor.submit(self._read_camera_previews, revision)
            pending = self._camera_preview_job
        previews = pending.result()
        with self._lock:
            self._camera_frame(revision)
        return previews

    def _read_camera_previews(self, revision: int):
        from .cameras import build_camera_thumbnails

        with self._lock:
            connection = self._camera_frame(revision)
        try:
            thumbnails = build_camera_thumbnails(
                connection.loader, connection.frame, revision=revision, review=connection.camera_review,
            )
        except DatasetError as error:
            _LOG.debug("Camera previews are unavailable for %s: %s", connection.frame.info.frame_id, error)
            return {}
        calibrated = {marker["id"] for marker in connection.camera_rig if marker.get("corner_rays")}
        return {
            camera_id: b64encode(thumbnails.image(camera_id)).decode("ascii")
            for camera_id in thumbnails.camera_ids if camera_id in calibrated
        }

    def camera_asset(self, revision: int, camera_id: str, *, projected_points: bool = False):
        with self._lock:
            connection = self._camera_frame(revision)
            review = connection.camera_review
            if review is None or connection.camera_review_dirty:
                raise BrowserError(404, "Open the camera review for the current frame first.")
            try:
                return review.points(camera_id) if projected_points else review.image(camera_id)
            except (KeyError, DataUnavailableError):
                raise BrowserError(404, "The requested camera asset is unavailable for this frame.") from None

    def start(self, values: dict):
        values = dict(values)
        op = values.get("op")
        operations = {
            "connect": {"dataset", "root", "version", "config_path", "split", "scene_id", "frame_index", "rgb", "raw"},
            "select": {"split", "scene_id", "frame_index", "rgb", "raw"},
            "prepare": {"rgb", "raw"},
            "detection": {"revision", "prediction_path", "prediction_format", "filters"},
            "segmentation": {"revision", "segmentation_mode", "segmentation_path", "segmentation_format"},
            "cameras": {"revision", "camera_points"},
        }
        if op not in operations:
            raise BrowserError(422, "Unknown browser operation.")
        if values.keys() - (operations[op] | {"op"}):
            raise BrowserError(422, "This operation contains fields belonging to a different operation.")
        with self._lock:
            self._check_open()
            if self._active_job is not None:
                raise BrowserError(409, "This session is already loading. Wait for the current operation.")
            if op != "connect" and self._connection is None:
                raise BrowserError(409, "Connect a dataset before changing the view.")
            if op in {"detection", "segmentation", "cameras"}:
                if "revision" not in values:
                    raise BrowserError(422, "An overlay operation requires a frame revision.")
                if values["revision"] != self._connection.revision:
                    raise BrowserError(409, "The requested frame revision is no longer current.")
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
            elif values["op"] == "detection":
                candidate = self._detection(job_id, values, previous)
            elif values["op"] == "segmentation":
                candidate = self._segmentation(job_id, values, previous)
            elif values["op"] == "cameras":
                candidate = self._cameras(job_id, values, previous)
            else:
                candidate = self._select_or_prepare(job_id, values, previous)
            state = _state(candidate)
            with self._lock:
                if self._closed:
                    return
                self._connection = candidate
                if candidate.revision != self._revision:
                    self._camera_preview_job = None
                self._revision = candidate.revision
                self._jobs[job_id].update(status="done", stage="Ready", state=state)
                self._active_job = None
            _close_replaced(previous, candidate)
        except Exception as error:
            with self._lock:
                if not self._closed:
                    self._jobs[job_id].update(status="error", stage="Loading failed", error=str(error))
                    self._active_job = None
        finally:
            if candidate is not None and candidate is not self._connection:
                _close_replaced(candidate, previous)
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
        source = {"config": config} if config is not None else {"root": root, "version": version}
        if self._factory is create_dataset:
            source["progress"] = lambda message: self._stage(job_id, message)
        loader = self._factory(name, **source)
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
            detection = DetectionReview.initial(frame, demo=name == "demo")
            segmentation = _Segmentation().for_frame(loader, frame)
            candidate = self._candidate(job_id, loader, dataset, split, scenes, selected.scene_id, frames, index, frame,
                                        rgb, raw, detection, segmentation)
            if self._connection is None and name == self.options.dataset and self.options.predictions is not None:
                try:
                    detection = self._attach_detections(
                        job_id, detection, frame, self.options.predictions, self.options.prediction_format,
                    )
                except PredictionError as error:
                    detection = replace(detection, review=None, error=str(error))
                candidate = replace(candidate, detection=detection)
            return candidate
        except Exception:
            _close_resource(loader)
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
        detection, segmentation = connection.detection, connection.segmentation
        if frame.info.frame_id != connection.frame.info.frame_id:
            if detection.source is not None:
                self._stage(job_id, "Reading detections")
            detection = detection.for_frame(frame)
            if segmentation.mode != "off":
                self._stage(job_id, "Reading LiDAR labels")
            segmentation = segmentation.for_frame(loader, frame)
        return self._candidate(job_id, loader, connection.dataset, split, scenes, scene_id, frames, index, frame,
                               rgb, raw, detection, segmentation)

    def _detection(self, job_id, values, connection):
        detection = connection.detection
        if "filters" in values:
            self._stage(job_id, "Filtering detections")
            detection = detection.with_filters(values["filters"])
        if "prediction_path" in values:
            detection = self._attach_detections(
                job_id, detection, connection.frame, values["prediction_path"].strip(),
                values.get("prediction_format", "auto"),
            )
        return replace(connection, detection=detection, camera_review_dirty=True)

    def _attach_detections(self, job_id, detection, frame, path, format):
        self._stage(job_id, "Opening detection export")
        return detection.with_source(
            frame, path, format, cache_dir=self.options.cache_dir,
            progress=lambda _fraction, message: self._stage(job_id, message),
            cancelled=lambda: self._closed,
        )

    def _segmentation(self, job_id, values, connection):
        from .segmentation import load_prediction_run

        loader = connection.loader
        segmentation = connection.segmentation
        if Capability.LIDAR_SEGMENTATION not in loader.metadata.capabilities:
            raise UnsupportedCapabilityError("LiDAR segmentation is unavailable for this dataset.")
        if "segmentation_path" in values:
            path = values["segmentation_path"].strip()
            self._stage(job_id, "Opening LiDAR segmentation export")
            run = None if not path else load_prediction_run(
                path, release=loader.metadata.release or loader.config.release,
                format=values.get("segmentation_format", "auto"),
            )
            segmentation = replace(segmentation, run=run)
        self._stage(job_id, "Reading LiDAR labels")
        segmentation = replace(
            segmentation, mode=values.get("segmentation_mode", segmentation.mode),
            generation=segmentation.generation + 1,
        ).for_frame(loader, connection.frame)
        return replace(connection, segmentation=segmentation)

    def _cameras(self, job_id, values, connection):
        from .cameras import build_camera_review

        detection = connection.detection.review
        retained = () if detection is None or detection.filtered is None else detection.filtered.detections
        review = build_camera_review(
            connection.loader, connection.frame, retained, revision=connection.revision,
            include_points=values.get("camera_points", False), previous=connection.camera_review,
            progress=lambda message: self._stage(job_id, message),
        )
        return replace(connection, camera_review=review, camera_review_dirty=False)

    @staticmethod
    def _camera_rig(loader, frame):
        metadata = loader.metadata
        if Capability.CAMERAS not in metadata.capabilities:
            return ()
        try:
            poses = loader.camera_rig(frame)
        except DatasetError as error:
            _LOG.debug("Camera mounting data is unavailable for %s: %s", frame.info.frame_id, error)
            return ()
        sensors = {sensor.sensor_id: sensor for sensor in metadata.sensors if sensor.modality == Modality.CAMERAS}
        return tuple({
            "id": pose.sensor_id,
            "name": sensors[pose.sensor_id].display_name if pose.sensor_id in sensors else pose.sensor_id,
            "position": pose.position.tolist(), "forward": pose.forward.tolist(),
            **({"corner_rays": pose.corner_rays.tolist()} if pose.corner_rays is not None else {}),
        } for pose in poses)

    def _candidate(self, job_id, loader, dataset, split, scenes, scene_id, frames, index, frame,
                   rgb, raw, detection, segmentation):
        if frame.info.frame_id != frames[index].frame_id or frame.info.scene_id != scene_id:
            raise ValueError("The dataset returned a point cloud for a different observation.")
        revision = self._revision + 1
        self._stage(job_id, "Transferring geometry")
        packet = encode_frame(frame, revision, rgb=rgb, raw=raw, class_groups=_groups(loader.metadata))
        prior = self._connection
        rig = (prior.camera_rig if prior is not None and prior.loader is loader and prior.frame.info == frame.info
               else self._camera_rig(loader, frame))
        return _Connection(loader, dataset, split, scenes, scene_id, frames, index, frame, rgb, raw, revision, packet,
                           detection, segmentation, rig)

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
            self._camera_preview_job = None
            self._jobs.clear()
        _close_replaced(connection)
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
                "prediction_formats": list(available_prediction_formats()),
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
