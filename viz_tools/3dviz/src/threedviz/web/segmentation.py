"""Browser segmentation overlays from native labels and local experiment exports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import pickle
from types import MappingProxyType
from zipfile import BadZipFile, ZipFile

import numpy as np

from visual_utils import (
    Capability, DataUnavailableError, DatasetError, DatasetLoader, DatasetValidationError, PointCloudFrame,
)
from visual_utils.numpy_pickle import RestrictedNumpyUnpickler
from visual_utils.nuscenes.lidarseg import EVALUATION_CLASSES, read_label_file, validate_labels


_LABEL_SUFFIX = "_lidarseg.bin"
_PICKLE_SUFFIXES = {".pkl", ".pickle"}


@dataclass(frozen=True)
class _SonataFrame:
    sample_token: str
    values: object


@dataclass(frozen=True)
class PredictionRun:
    path: Path
    format: str
    name: str
    release: str | None
    split: str | None
    _scan_entries: Mapping[str, Path | str | _SonataFrame] = field(repr=False)
    _scans_by_sample: Mapping[str, str] = field(repr=False)

    def public(self) -> dict:
        return {
            "path": str(self.path), "format": self.format, "name": self.name,
            "release": self.release, "split": self.split, "frame_count": len(self._scan_entries),
        }

    def labels_for(self, sample_token: str, scan_token: str, point_count: int) -> np.ndarray:
        if (declared_scan := self._scans_by_sample.get(sample_token)) is not None and declared_scan != scan_token:
            raise DatasetValidationError(
                f"Prediction for sample {sample_token} refers to LIDAR_TOP scan {declared_scan}; expected {scan_token}."
            )
        entry = self._scan_entries.get(scan_token)
        if entry is None:
            raise DataUnavailableError(f"This run has no prediction for LIDAR_TOP scan {scan_token}.")
        if isinstance(entry, _SonataFrame):
            if entry.sample_token != sample_token:
                raise DatasetValidationError(
                    f"Prediction for scan {scan_token} refers to sample {entry.sample_token}; expected {sample_token}."
                )
            values, minimum = entry.values, 0
        else:
            minimum = 1
            try:
                if isinstance(entry, Path):
                    values = read_label_file(entry, point_count)
                else:
                    with ZipFile(self.path) as archive, archive.open(entry) as stream:
                        values = np.frombuffer(stream.read(point_count + 1), dtype=np.uint8)
            except (OSError, RuntimeError, KeyError, BadZipFile) as error:
                raise DatasetValidationError(f"Cannot read prediction for LIDAR_TOP scan {scan_token}: {error}") from error
        return validate_labels(
            values, point_count, context=f"Prediction for scan {scan_token}",
            minimum=minimum, maximum=len(EVALUATION_CLASSES) - 1,
        )


def _sonata_run(path: Path, release: str | None) -> PredictionRun:
    try:
        with path.open("rb") as stream:
            document = RestrictedNumpyUnpickler(stream).load()
    except (OSError, EOFError, pickle.PickleError, ValueError, TypeError) as error:
        raise DatasetValidationError(f"Cannot import Sonata predictions {path}: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
        raise DatasetValidationError("Sonata predictions require a metadata dictionary and a frames list.")
    metadata = document["metadata"]
    expected = {
        "format": "nuscenes_lidarseg", "schema_version": 1, "dataset": "nuscenes", "sensor": "LIDAR_TOP",
        "point_order": "original_lidar_scan", "label_dtype": "uint8", "ignore_label": 0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise DatasetValidationError(f"Sonata export metadata requires {key}={value!r}.")
    class_ids = list(range(1, len(EVALUATION_CLASSES)))
    class_names = [name for name, _ in EVALUATION_CLASSES[1:]]
    if metadata.get("class_ids") != class_ids or metadata.get("class_names") != class_names:
        raise DatasetValidationError("Sonata classes must use the nuScenes evaluation IDs 1–16 and their canonical names.")
    run_release = metadata.get("dataset_version")
    if not isinstance(run_release, str) or not run_release:
        raise DatasetValidationError("Sonata export metadata must name its nuScenes dataset version.")
    if release and run_release != release:
        raise DatasetValidationError(f"Sonata run targets nuScenes {run_release}; the connected dataset is {release}.")
    frames = document.get("frames")
    if not isinstance(frames, (list, tuple)) or not frames:
        raise DatasetValidationError("Sonata predictions must contain a nonempty frames list.")
    entries, samples = {}, {}
    for index, frame in enumerate(frames):
        frame_metadata = frame.get("metadata") if isinstance(frame, dict) else None
        if not isinstance(frame_metadata, dict):
            raise DatasetValidationError(f"Sonata frame {index} must contain a metadata dictionary.")
        sample, scan = frame_metadata.get("token"), frame_metadata.get("sample_data_token")
        if not all(isinstance(token, str) and token for token in (sample, scan)):
            raise DatasetValidationError(f"Sonata frame {index} must identify both its sample and LIDAR_TOP scan.")
        if scan in entries or sample in samples:
            raise DatasetValidationError(f"Sonata frame {index} repeats an existing sample or LIDAR_TOP scan token.")
        entries[scan] = _SonataFrame(sample, frame.get("pred_labels"))
        samples[sample] = scan
    split, name = metadata.get("split"), metadata.get("name")
    return PredictionRun(
        path, "sonata", name if isinstance(name, str) and name else path.stem,
        run_release, split if isinstance(split, str) and split else None,
        MappingProxyType(entries), MappingProxyType(samples),
    )


def _official_run(path: Path, release: str | None) -> PredictionRun:
    entries, splits = {}, set()

    def add(filename: PurePosixPath, entry: Path | str):
        token = filename.name.removesuffix(_LABEL_SUFFIX)
        if not token or token == filename.name:
            return
        if token in entries:
            raise DatasetValidationError(f"Official predictions contain duplicate labels for scan {token}.")
        entries[token] = entry
        if filename.parent.name in {"train", "val", "test", "mini_train", "mini_val"}:
            splits.add(filename.parent.name)

    try:
        if path.is_dir():
            official_root = path / "lidarseg" if (path / "lidarseg").is_dir() else path
            for file in (*official_root.glob(f"*{_LABEL_SUFFIX}"), *official_root.glob(f"*/*{_LABEL_SUFFIX}")):
                if file.is_file():
                    add(PurePosixPath(file), file)
        elif path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                for item in archive.infolist():
                    if not item.is_dir():
                        add(PurePosixPath(item.filename), item.filename)
        else:
            raise DatasetValidationError("Choose an extracted nuScenes lidarseg submission folder or ZIP archive.")
    except (OSError, BadZipFile) as error:
        raise DatasetValidationError(f"Cannot read nuScenes lidarseg submission {path}: {error}") from error
    if not entries:
        raise DatasetValidationError(f"No <sample_data_token>{_LABEL_SUFFIX} files were found in {path}.")
    split = next(iter(splits)) if len(splits) == 1 else None
    return PredictionRun(path, "official", path.stem, release, split, MappingProxyType(entries), MappingProxyType({}))


def load_prediction_run(path: str | Path, *, release: str | None = None, format: str = "auto") -> PredictionRun:
    source = Path(path).expanduser().resolve()
    if format == "auto":
        format = "sonata" if source.suffix.lower() in _PICKLE_SUFFIXES else "official"
    if format == "sonata":
        return _sonata_run(source, release)
    if format == "official":
        return _official_run(source, release)
    raise DatasetValidationError(f"Unknown LiDAR segmentation prediction format {format!r}.")


@dataclass(frozen=True, eq=False)
class FrameSegmentation:
    source: str
    status: str
    scan_token: str | None = None
    reason: str | None = None
    labels: np.ndarray | None = None
    native_labels: np.ndarray | None = None

    def public(self) -> dict:
        counts = None if self.labels is None else np.bincount(self.labels, minlength=len(EVALUATION_CLASSES))
        return {
            "source": self.source, "status": self.status, "reason": self.reason, "scan_token": self.scan_token,
            "classes": [
                {"id": index, "name": name, "color": [component / 255 for component in color],
                 "count": 0 if counts is None else int(counts[index])}
                for index, (name, color) in enumerate(EVALUATION_CLASSES)
            ],
        }


def frame_segmentation(
    loader: DatasetLoader, frame: PointCloudFrame, source: str, run: PredictionRun | None = None,
) -> FrameSegmentation:
    if source not in {"off", "ground_truth", "prediction"}:
        raise ValueError(f"Unknown LiDAR segmentation source {source!r}.")
    if source == "off":
        return FrameSegmentation(source, "off")
    if Capability.LIDAR_SEGMENTATION not in loader.metadata.capabilities:
        return FrameSegmentation(source, "unsupported", reason="This dataset does not provide LiDAR segmentation.")
    scan_token = None
    try:
        scan_token = loader.lidar_scan_token(frame.info.frame_id)
        if source == "ground_truth":
            segmentation = loader.load_lidarseg_ground_truth(frame)
            if segmentation.scan_token != scan_token:
                raise DatasetValidationError(f"Ground-truth labels refer to a different LIDAR_TOP scan than {scan_token}.")
            labels = validate_labels(
                segmentation.labels, len(frame.points), context=f"Ground truth for scan {scan_token}",
                maximum=len(EVALUATION_CLASSES) - 1,
            )
            return FrameSegmentation(source, "ready", scan_token, labels=labels, native_labels=segmentation.native_labels)
        if run is None:
            raise DataUnavailableError("Attach a LiDAR segmentation prediction run in Sources.")
        labels = run.labels_for(frame.info.frame_id, scan_token, len(frame.points))
        return FrameSegmentation(source, "ready", scan_token, labels=labels)
    except DataUnavailableError as error:
        return FrameSegmentation(source, "missing", scan_token, str(error))
    except (DatasetError, ValueError, TypeError) as error:
        return FrameSegmentation(source, "invalid", scan_token, str(error))
