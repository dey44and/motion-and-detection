"""Camera grid placement is presentation metadata, separate from observations."""

from collections.abc import Sequence
from dataclasses import dataclass

from visual_utils import Modality, SensorInfo


@dataclass(frozen=True)
class CameraSlot:
    sensor_id: str
    label: str
    row: int
    column: int


_ROLE_SLOTS = {
    "front_left": (0, 0), "front": (0, 1), "front_right": (0, 2),
    "back_right": (1, 0), "rear_right": (1, 0),
    "back": (1, 1), "rear": (1, 1),
    "back_left": (1, 2), "rear_left": (1, 2),
    "side_left": (1, 0), "side_right": (1, 2),
}


def camera_layout(sensors: Sequence[SensorInfo]) -> tuple[CameraSlot, ...]:
    """Place known roles first; fill remaining slots in stable sensor order."""
    cameras = [sensor for sensor in sensors if sensor.modality == Modality.CAMERAS]
    columns = min(3, max(1, len(cameras)))
    occupied, slots = set(), {}
    for sensor in cameras:
        slot = _ROLE_SLOTS.get(sensor.view_role)
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
    return tuple(CameraSlot(sensor.sensor_id, sensor.display_name, *slots[sensor.sensor_id]) for sensor in cameras)
