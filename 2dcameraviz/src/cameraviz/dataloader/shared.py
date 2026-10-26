"""Generic camera bridge for any registered shared dataset loader."""

from dataset_utils import DatasetConfig, create_loader
from dataset_utils.contracts.models import Capability, Modality
from dataset_utils.errors import UnsupportedCapabilityError
import numpy as np

from .base import DatasetAdapter
from .models import BoundingBox3D, CameraFrame, CameraInfo, FrameData, FrameInfo, SceneInfo


_ROLE_SLOTS = {
    "front_left": (0, 0), "front": (0, 1), "front_right": (0, 2),
    "back_right": (1, 0), "rear_right": (1, 0),
    "back": (1, 1), "rear": (1, 1),
    "back_left": (1, 2), "rear_left": (1, 2),
    "side_left": (1, 0), "side_right": (1, 2),
}


def camera_layout(sensors):
    cameras = [sensor for sensor in sensors if sensor.modality == Modality.CAMERAS]
    columns = min(3, max(1, len(cameras)))
    occupied, slots = set(), {}
    for sensor in cameras:
        role = getattr(sensor.view_role, "value", sensor.view_role)
        slot = _ROLE_SLOTS.get(role)
        if slot is not None and slot not in occupied:
            slots[sensor.sensor_id] = slot
            occupied.add(slot)
    for sensor in cameras:
        if sensor.sensor_id not in slots:
            index = 0
            while divmod(index, columns) in occupied:
                index += 1
            slots[sensor.sensor_id] = divmod(index, columns)
            occupied.add(slots[sensor.sensor_id])
    return tuple(CameraInfo(sensor.sensor_id, sensor.display_name, *slots[sensor.sensor_id]) for sensor in cameras)


class ProjectionBridge:
    def __init__(self, model, world_to_reference=None):
        self.model = model
        self.world_to_reference = world_to_reference

    def project_world(self, points):
        if self.world_to_reference is not None:
            transform = self.world_to_reference
            points = points @ transform[:3, :3].T + transform[:3, 3]
        result = self.model.project_reference(points)
        # Behind-camera signed depth remains available for near-plane clipping.
        uv = np.where(result.valid_mask[:, None], result.uv, np.nan)
        return uv, result.depth


class SharedCameraAdapter(DatasetAdapter):
    def __init__(self, loader):
        if Capability.CAMERAS not in loader.metadata.capabilities:
            loader.close()
            raise UnsupportedCapabilityError("The selected dataset does not provide camera observations.")
        self.loader = loader
        self._classes = {item.class_id: item.display_group or item.display_name for item in loader.metadata.classes}
        self._camera_layout = camera_layout(loader.metadata.sensors)
        self._frame_lists = {}

    @property
    def camera_layout(self):
        return self._camera_layout

    @property
    def class_names(self):
        return tuple(dict.fromkeys(self._classes.values()))

    @property
    def available_splits(self):
        return self.loader.metadata.splits

    def list_scenes(self):
        return self.list_scenes_for_split("all")

    def list_scenes_for_split(self, split="all"):
        return tuple(SceneInfo(info.scene_id, info.display_name, info.description,
                               info.frame_counts.get(Modality.CAMERAS) or 0)
                     for info in self.loader.list_scenes(split=split)
                     if info.frame_counts.get(Modality.CAMERAS) != 0)

    def list_frames(self, scene_token):
        if scene_token not in self._frame_lists:
            self._frame_lists[scene_token] = tuple(
                FrameInfo(info.frame_id, info.scene_id, info.index, info.timestamp_us)
                for info in self.loader.list_frames(scene_token, modality=Modality.CAMERAS))
        return self._frame_lists[scene_token]

    def _boxes(self, annotations, transform=None):
        if annotations is None:
            return ()
        return tuple(BoundingBox3D(box.center if transform is None else transform[:3, :3] @ box.center + transform[:3, 3],
                                   box.size, box.rotation if transform is None else transform[:3, :3] @ box.rotation,
                                   self._classes[box.class_id], box.annotation_id)
                     for box in annotations.boxes)

    def load_frame(self, frame_token):
        native = self.loader.load_cameras(frame_token)
        world = native.T_world_from_reference
        inverse_world = None if world is None else np.linalg.inv(world)
        by_sensor = {camera.sensor_id: camera for camera in native.cameras}
        cameras = []
        for info in self._camera_layout:
            camera = by_sensor.get(info.channel)
            if camera is None:
                continue
            cameras.append(CameraFrame(
                info=info, image=camera.image, intrinsics=camera.intrinsics,
                sensor_to_world=(camera.T_reference_from_camera if world is None or camera.T_reference_from_camera is None
                                 else world @ camera.T_reference_from_camera), timestamp_us=camera.timestamp_us,
                projection_model=None if camera.projection_model is None else ProjectionBridge(camera.projection_model, inverse_world),
                boxes=None if camera.annotations is None else self._boxes(camera.annotations, world),
                allow_pinhole=False,
                annotation_status=(camera.annotations if camera.annotations is not None else native.annotations).status.value,
            ))
        info = native.info
        return FrameData(FrameInfo(info.frame_id, info.scene_id, info.index, info.timestamp_us),
                         tuple(cameras), self._boxes(native.annotations, world), native.annotations.status.value)

    def close(self):
        self.loader.close()

    def __getattr__(self, name):
        loader = self.__dict__.get("loader")
        if loader is None:
            raise AttributeError(name)
        for candidate in (loader, getattr(loader, "reader", None)):
            if candidate is not None and hasattr(candidate, name):
                return getattr(candidate, name)
        native_reader = getattr(loader, "native_reader", None)
        if native_reader is not None:
            reader = native_reader(Modality.CAMERAS)
            if hasattr(reader, name):
                return getattr(reader, name)
        raise AttributeError(name)


def open_shared(name, root=None, version="v1.0-mini", *, config=None):
    release = None if version in (None, "", "v1.0-mini") else version
    loader = create_loader(name, config if config is not None else DatasetConfig(root=root, release=release))
    try:
        return SharedCameraAdapter(loader)
    except Exception:
        loader.close()
        raise
