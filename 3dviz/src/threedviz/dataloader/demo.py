"""Deterministic synthetic lidar scenes for trying the viewer without downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import DatasetAdapter, DatasetError
from .models import BoundingBox3D, FrameData, FrameInfo, SceneInfo
from .nuscenes import DETECTION_CLASSES


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class DemoAdapter(DatasetAdapter):
    """Two scenes with 36 frames each, generated lazily from fixed RNG seeds."""

    def __init__(self, *, root: str | Path | None = None, version: str = "v1.0-mini"):
        self._scenes = (
            SceneInfo("demo-urban", "Demo: city street", "Synthetic traffic, pedestrians and roadside objects.", 36),
            SceneInfo("demo-highway", "Demo: highway", "Synthetic vehicles and a roadside construction area.", 36),
        )
        self._frames_by_scene = {
            scene.token: tuple(
                FrameInfo(f"{scene.token}-{index:03d}", scene.token, index, index * 500_000)
                for index in range(scene.frame_count)
            ) for scene in self._scenes
        }
        self._frames = {
            frame.token: frame for frames in self._frames_by_scene.values() for frame in frames
        }

    @property
    def class_names(self) -> tuple[str, ...]:
        return DETECTION_CLASSES

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
        highway = info.scene_token == "demo-highway"
        rng = np.random.default_rng(1050 + (1000 if highway else 0) + info.index)
        ground_z = -1.8
        shift = 3 * np.sin(info.index / 8)
        objects = [
            ("car", (14 + shift, -3.5), (4.5, 1.9, 1.6), 0.1),
            ("truck", (29 - shift, 3.5), (7.0, 2.5, 3.2), 0.0),
            ("motorcycle" if highway else "bicycle", (9, 6.0), (2.0, 0.7, 1.4), 0.25),
            ("barrier", (20, -8.0), (4.0, 0.5, 1.0), 0.0),
            ("traffic_cone", (13, -7.5), (0.5, 0.5, 0.9), 0.0),
        ]
        if highway:
            objects += [
                ("construction_vehicle", (37, -10), (6.0, 3.0, 3.5), -0.3),
                ("trailer", (-20 + shift, 3.5), (10.0, 2.5, 3.0), 0.0),
            ]
        else:
            objects += [
                ("bus", (-17 + shift, 3.5), (10.0, 2.6, 3.2), 0.0),
                ("pedestrian", (6.0 + shift / 2, 8.5), (0.6, 0.6, 1.75), 0.1),
                ("pedestrian", (10.0, -10.0), (0.6, 0.6, 1.65), -0.2),
            ]

        boxes = tuple(
            BoundingBox3D(
                center=np.array([xy[0], xy[1], ground_z + size[2] / 2]),
                size=np.array(size),
                rotation=_yaw_matrix(yaw),
                class_name=class_name,
                token=f"{info.scene_token}-object-{index}",
            ) for index, (class_name, xy, size, yaw) in enumerate(objects)
        )
        ground = np.column_stack((
            rng.uniform(-45, 55, 18_000),
            rng.uniform(-20, 20, 18_000),
            rng.normal(ground_z, 0.025, 18_000),
        ))
        clouds = [ground]
        for box in boxes:
            local = rng.uniform(-0.5, 0.5, (850, 3))
            face_axis = rng.integers(0, 3, len(local))
            local[np.arange(len(local)), face_axis] = rng.choice([-0.5, 0.5], len(local))
            clouds.append((local * box.size) @ box.rotation.T + box.center)
        for side in (-1, 1):
            count = 2600
            clouds.append(np.column_stack((
                rng.uniform(-35, 50, count),
                rng.normal(side * (17 if highway else 14), 0.03, count),
                rng.uniform(ground_z, 2.5 if highway else 7.0, count),
            )))
        points = np.ascontiguousarray(np.concatenate(clouds), dtype=np.float32)
        intensity = rng.uniform(20, 220, len(points)).astype(np.float32)
        return FrameData(info=info, points=points, boxes=boxes, intensity=intensity)
