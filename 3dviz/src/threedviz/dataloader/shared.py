"""Generic bridge from the shared loader contract to the 3D renderer."""

from collections import OrderedDict
from dataclasses import fields, replace
import weakref

import numpy as np

from dataset_utils import DatasetConfig, create_loader
from dataset_utils.contracts.models import Capability, Modality, PointCloudPreparation
from dataset_utils.errors import DatasetValidationError, UnsupportedCapabilityError

from .base import DatasetAdapter
from .models import BoundingBox3D, FrameData, FrameInfo, SceneInfo


def legacy_config(name, root, version):
    # The old launch API supplies the nuScenes default to every provider.
    release = None if version in (None, "", "v1.0-mini") else version
    return DatasetConfig(root=root, release=release)


class SharedPointCloudAdapter(DatasetAdapter):
    """Display any registered point-cloud loader without dataset-specific UI."""

    def __init__(self, loader):
        if Capability.POINT_CLOUD not in loader.metadata.capabilities:
            loader.close()
            raise UnsupportedCapabilityError("The selected dataset does not provide point-cloud observations.")
        self.loader = loader
        self._classes = {item.class_id: item.display_group or item.display_name for item in loader.metadata.classes}
        self._normalized_frames = OrderedDict()
        self._issued_points = {}
        self._frame_lists = {}
        self._scene_lists = {}
        self._scene_records = {}

    @property
    def class_names(self):
        return tuple(dict.fromkeys(self._classes.values()))

    @property
    def available_splits(self):
        return self.loader.metadata.splits

    @property
    def supports_camera_rgb(self):
        return Capability.RGB_COLORING in self.loader.metadata.capabilities

    @property
    def supports_point_timing(self):
        return Capability.POINT_TIMESTAMPS in self.loader.metadata.capabilities

    @property
    def supports_acquisition_geometry(self):
        metadata = self.loader.metadata
        return (Capability.ACQUISITION_GEOMETRY in metadata.capabilities
                and metadata.feature_coverage.get(Capability.ACQUISITION_GEOMETRY, "full") != "partial")

    def list_scenes(self):
        return self.list_scenes_for_split("all")

    def list_scenes_for_split(self, split="all"):
        scenes = []
        for item in self.loader.list_scenes(split=split):
            if item.frame_counts.get(Modality.POINT_CLOUD) == 0:
                continue
            scene = SceneInfo(item.scene_id, item.display_name, item.description,
                              item.frame_counts.get(Modality.POINT_CLOUD) or 0)
            if self._scene_records.get(item.scene_id) != scene:
                self._scene_records[item.scene_id] = scene
            scenes.append(self._scene_records[item.scene_id])
        result = tuple(scenes)
        if self._scene_lists.get(split) != result:
            self._scene_lists[split] = result
        return self._scene_lists[split]

    def list_frames(self, scene_token):
        # Refreshing scene counts need not invalidate stable frame metadata.
        if scene_token not in self._frame_lists:
            self._frame_lists[scene_token] = tuple(
                FrameInfo(info.frame_id, info.scene_id, info.index, info.timestamp_us)
                for info in self.loader.list_frames(scene_token, modality=Modality.POINT_CLOUD))
        return self._frame_lists[scene_token]

    def _convert(self, native):
        boxes = tuple(BoundingBox3D(box.center, box.size, box.rotation,
                                   self._classes[box.class_id], box.annotation_id)
                      for box in native.annotations.boxes)
        acquisition = native.acquisition
        raw = None
        if acquisition is not None:
            if acquisition.valid_mask.all():
                raw = acquisition.points
        timing = native.timing
        times = timing.timestamps_us if timing is not None and timing.valid_mask.all() else None
        reference_time = native.reference_pose_timestamp_us
        return FrameData(
            info=FrameInfo(native.info.frame_id, native.info.scene_id, native.info.index, native.info.timestamp_us),
            points=native.points, boxes=boxes, intensity=native.attributes.get("intensity"),
            lidar_timestamp_us=reference_time if reference_time is not None else native.info.timestamp_us,
            ring_indices=native.attributes.get("ring_indices", native.attributes.get("ring")),
            point_timestamps_us=times, raw_points=raw,
            rgb_colors=None if native.rgb is None else native.rgb.values,
            rgb_valid_mask=None if native.rgb is None else native.rgb.valid_mask,
            reference_sensor_to_world=native.T_world_from_reference,
            annotation_status=native.annotations.status.value,
        )

    def _remember(self, native):
        self._normalized_frames[native.info.frame_id] = native
        self._normalized_frames.move_to_end(native.info.frame_id)
        while len(self._normalized_frames) > 2:
            self._normalized_frames.popitem(last=False)

    def load_frame(self, frame_token):
        native = self.loader.load_point_cloud(frame_token)
        self._remember(native)
        converted = self._convert(native)
        identity = id(converted.points)
        issued = self._issued_points
        issued[identity] = weakref.ref(converted.points, lambda _reference: issued.pop(identity, None))
        return converted

    def prepare_frame(self, frame, *, camera_rgb=False, point_timing=False):
        issued = self._issued_points.get(id(frame.points))
        if issued is None or issued() is not frame.points:
            raise DatasetValidationError("Preparation requires unchanged point rows issued by this adapter.")
        needs_rgb = camera_rgb and frame.rgb_colors is None
        needs_raw = point_timing and frame.raw_points is None
        if not needs_rgb and not needs_raw:
            return frame
        if needs_raw and not self.supports_acquisition_geometry:
            raise UnsupportedCapabilityError("Full acquisition-time geometry is unavailable; a partial scan cannot be shown as fully uncompensated.")
        native = self._normalized_frames.get(frame.info.token)
        if native is None:
            native = self.loader.load_point_cloud(frame.info.token)
        if native.points.shape != frame.points.shape or not np.array_equal(native.points, frame.points):
            raise DatasetValidationError("Point rows changed before optional dataset preparation.")
        prepared = self.loader.prepare_point_cloud(native, request=PointCloudPreparation(
            rgb=needs_rgb, timestamps=needs_raw and self.supports_point_timing,
            acquisition_geometry=needs_raw,
        ))
        if needs_raw and (prepared.acquisition is None or not prepared.acquisition.valid_mask.all()):
            raise UnsupportedCapabilityError("Acquisition-time geometry covers only part of this scan.")
        self._remember(prepared)
        converted = self._convert(prepared)
        # Preserve subclass fields used by the independent detection viewer.
        values = {field.name: getattr(converted, field.name) for field in fields(FrameData)}
        values["points"], values["boxes"] = frame.points, frame.boxes
        return replace(frame, **values)

    def close(self):
        self.loader.close()

    def __getattr__(self, name):
        # Compatibility access for existing integrations inspecting reader
        # caches. Runtime loading above uses only the public shared contract.
        loader = self.__dict__.get("loader")
        if loader is None:
            raise AttributeError(name)
        for candidate in (loader, getattr(loader, "processed_adapter", None), getattr(loader, "reader", None)):
            if candidate is not None and hasattr(candidate, name):
                return getattr(candidate, name)
        native_reader = getattr(loader, "native_reader", None)
        if native_reader is not None:
            reader = native_reader(Modality.POINT_CLOUD)
            if hasattr(reader, name):
                return getattr(reader, name)
        raise AttributeError(name)


def open_shared(name, root=None, version="v1.0-mini", *, config=None):
    loader = create_loader(name, config if config is not None else legacy_config(name, root, version))
    try:
        return SharedPointCloudAdapter(loader)
    except Exception:
        loader.close()
        raise
