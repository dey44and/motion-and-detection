"""Deterministic synthetic camera views for using the viewer without a dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .base import DatasetAdapter, DatasetError
from .models import BoundingBox3D, CameraFrame, FrameData, FrameInfo, SceneInfo
from .nuscenes import CAMERA_LAYOUT, DETECTION_CLASSES

_WIDTH, _HEIGHT = 960, 540
_FOCAL = 600.0
_INTRINSICS = np.array([[_FOCAL, 0, _WIDTH / 2], [0, _FOCAL, _HEIGHT / 2], [0, 0, 1]])
_YAW_DEGREES = (55, 0, -55, 125, 180, -125)
_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
])
_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
_PALETTE = (
    (71, 122, 169), (186, 130, 63), (208, 170, 63), (126, 101, 153), (152, 117, 79),
    (134, 143, 151), (84, 148, 139), (173, 104, 113), (161, 136, 85), (208, 137, 72),
)
_SIZES = (
    (4.3, 1.8, 1.5), (6.5, 2.3, 2.7), (4.0, 2.5, 2.4), (8.0, 2.5, 2.8), (6.0, 2.4, 2.7),
    (2.2, 0.45, 0.8), (1.8, 0.7, 1.25), (1.7, 0.55, 1.2), (0.65, 0.65, 1.75), (0.45, 0.45, 0.7),
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def _project(points: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    optical = (points - pose[:3, 3]) @ pose[:3, :3]
    pixels_h = optical @ _INTRINSICS.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels, optical[:, 2]


def _view(pose: np.ndarray, boxes: tuple[BoundingBox3D, ...], camera_index: int) -> np.ndarray:
    rows = np.arange(_HEIGHT, dtype=float)[:, None, None]
    top = np.array([190, 212, 227], dtype=float)
    horizon = np.array([226, 233, 234], dtype=float)
    colors = top + (horizon - top) * np.minimum(rows / (_HEIGHT / 2), 1)
    colors = np.repeat(colors, _WIDTH, axis=1)
    ground = rows[:, 0, 0] >= _HEIGHT / 2
    colors[ground] = np.array([92, 101, 109]) + (rows[ground] - _HEIGHT / 2) * np.array([.025, .025, .025])
    image = Image.fromarray(colors.clip(0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    # A simple skyline and lane markings keep the generated images readable.
    for column in range(0, _WIDTH, 96):
        height = 24 + ((column // 96 + camera_index * 3) % 5) * 11
        shade = 143 + (column // 96 % 3) * 9
        draw.rectangle([column, _HEIGHT // 2 - height, column + 74, _HEIGHT // 2], fill=(shade, shade + 8, shade + 10))
    for shift in (-.55, .55):
        draw.line([(_WIDTH / 2 + shift * 22, _HEIGHT / 2 + 15), (_WIDTH / 2 + shift * 660, _HEIGHT)], fill=(205, 203, 179), width=4)
    faces = []
    for index, box in enumerate(boxes):
        corners = (_SIGNS * box.size / 2) @ box.rotation.T + box.center
        pixels, depths = _project(corners, pose)
        for face_index, face in enumerate(_FACES):
            selected = np.array(face)
            if np.any(depths[selected] <= .1) or not np.isfinite(pixels[selected]).all():
                continue
            polygon = pixels[selected]
            if polygon[:, 0].max() < 0 or polygon[:, 0].min() >= _WIDTH or polygon[:, 1].max() < 0 or polygon[:, 1].min() >= _HEIGHT:
                continue
            factor = .72 + .055 * face_index
            color = tuple(int(value * factor) for value in _PALETTE[index % len(_PALETTE)])
            faces.append((depths[selected].mean(), polygon, color))
    for _, polygon, color in sorted(faces, key=lambda face: face[0], reverse=True):
        draw.polygon([tuple(point) for point in polygon], fill=color)
    array = np.array(image, dtype=np.uint8)
    array.setflags(write=False)
    return array


class DemoAdapter(DatasetAdapter):
    def __init__(self, *, root: str | Path | None = None, version: str = "demo"):
        self._scenes = (
            SceneInfo("demo-intersection", "Demo intersection", "Generated street views with traffic in all six directions.", 24),
            SceneInfo("demo-roundabout", "Demo roundabout", "Generated camera images and moving annotations; no dataset files required.", 24),
        )
        self._frames_by_scene = {
            scene.token: tuple(FrameInfo(f"{scene.token}-{index:03d}", scene.token, index, 1_000_000 + index * 500_000)
                               for index in range(scene.frame_count))
            for scene in self._scenes
        }
        self._frames = {frame.token: frame for frames in self._frames_by_scene.values() for frame in frames}

    @property
    def class_names(self) -> tuple[str, ...]:
        return DETECTION_CLASSES

    @property
    def camera_layout(self):
        return CAMERA_LAYOUT

    def list_scenes(self) -> tuple[SceneInfo, ...]:
        return self._scenes

    def list_frames(self, scene_token: str) -> tuple[FrameInfo, ...]:
        try:
            return self._frames_by_scene[scene_token]
        except KeyError:
            raise DatasetError(f"Unknown demo scene {scene_token!r}.") from None

    def load_frame(self, frame_token: str) -> FrameData:
        try:
            info = self._frames[frame_token]
        except KeyError:
            raise DatasetError(f"Unknown demo frame {frame_token!r}.") from None
        scene_phase = .24 if info.scene_token == "demo-roundabout" else 0.0
        boxes = []
        for index, class_name in enumerate(DETECTION_CLASSES):
            angle = index * 2 * np.pi / len(DETECTION_CLASSES) + scene_phase
            distance = 12.0 + index % 3 * 2.5 + np.sin(info.index * .12 + index) * .8
            size = np.array(_SIZES[index])
            center = np.array([np.cos(angle) * distance, np.sin(angle) * distance, size[2] / 2])
            boxes.append(BoundingBox3D(center, size, _rotation_z(angle + .25), class_name, f"demo-box-{index}"))
        boxes_tuple = tuple(boxes)
        cameras = []
        for index, (camera_info, yaw_degrees) in enumerate(zip(CAMERA_LAYOUT, _YAW_DEGREES)):
            angle = np.deg2rad(yaw_degrees)
            forward = np.array([np.cos(angle), np.sin(angle), 0])
            right = np.array([np.sin(angle), -np.cos(angle), 0])
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack([right, [0, 0, -1], forward])
            pose[:3, 3] = [forward[0] * .3, forward[1] * .3, 1.6]
            cameras.append(CameraFrame(camera_info, _view(pose, boxes_tuple, index), _INTRINSICS, pose, info.timestamp_us + index * 1000))
        return FrameData(info, tuple(cameras), boxes_tuple)
