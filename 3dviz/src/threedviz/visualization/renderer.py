"""Render normalized frames without knowing their source dataset."""

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from threedviz.dataloader import FrameData
from threedviz.processing import BOX_EDGES, box_corners, class_color, colorize_points


@dataclass
class RenderSettings:
    point_size: float = 3.0
    color_scheme: str = "gray"
    max_distance: float = 80.0
    line_width: float = 2.0
    selected_classes: set[str] = field(default_factory=set)
    class_colors: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    use_raw_points: bool = False


class SceneRenderer:
    """Bridge between an Open3D viewport and the common dataset models.

    All methods must run on the GUI thread. Style changes preserve the camera;
    only an explicit reset or a scene change fits the camera to the geometry.
    """

    def __init__(self, widget: gui.SceneWidget, scaling: float = 1.0):
        self.widget = widget
        self.scaling = scaling
        self.settings = RenderSettings()
        self.frame: FrameData | None = None
        self._box_names: list[str] = []
        self._cloud = o3d.geometry.PointCloud()
        self.widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        # Preserve the white background and the chosen RGB colors without
        # photographic tone mapping or exposure changes.
        self.widget.scene.view.set_post_processing(False)
        self.widget.scene.show_axes(True)
        self.widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def set_frame(self, frame: FrameData, *, reset_camera: bool = False):
        # Validate and construct all geometry before replacing a working frame.
        points = self._display_points(frame)
        colors = self._point_colors(frame, points)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        boxes = self._box_geometries(frame)
        scene = self.widget.scene
        if scene.has_geometry("points"):
            scene.remove_geometry("points")
        if len(points):
            scene.add_geometry("points", cloud, self._point_material())
        self._replace_boxes(boxes)
        self.frame, self._cloud = frame, cloud
        if reset_camera:
            self.reset_camera()
        self.widget.force_redraw()

    def _display_points(self, frame):
        if self.settings.use_raw_points:
            if frame.raw_points is None:
                raise ValueError("Acquisition-time points have not been prepared for this frame.")
            points = np.asarray(frame.raw_points, dtype=np.float64)
            if points.shape != frame.points.shape:
                raise ValueError("Acquisition-time points must match the point cloud shape.")
            return points
        return frame.points

    def _point_colors(self, frame, points):
        if self.settings.color_scheme != "camera_rgb":
            return colorize_points(points, self.settings.color_scheme, self.settings.max_distance)
        if frame.rgb_colors is None:
            raise ValueError("Camera RGB colors have not been prepared for this frame.")
        colors = np.asarray(frame.rgb_colors, dtype=np.float64)
        if colors.shape != points.shape or not np.isfinite(colors).all():
            raise ValueError("Camera RGB colors must have shape (N, 3) and contain finite values.")
        if np.any((colors < 0) | (colors > 1)):
            raise ValueError("Camera RGB colors must be between zero and one.")
        fallback = colorize_points(points, "gray", self.settings.max_distance)
        if frame.rgb_valid_mask is None:
            return colors
        valid = np.asarray(frame.rgb_valid_mask)
        if valid.shape != (len(points),) or valid.dtype != np.bool_:
            raise ValueError("Camera RGB validity must be a boolean array with shape (N,).")
        return np.where(valid[:, None], colors, fallback)

    def _point_material(self):
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = self.settings.point_size * self.scaling
        return material

    def update_points(self, *, recolor: bool = False):
        if self.frame is None:
            return
        scene = self.widget.scene
        if recolor:
            colors = self._point_colors(self.frame, self._display_points(self.frame))
            self._cloud.colors = o3d.utility.Vector3dVector(colors)
            if scene.has_geometry("points"):
                scene.remove_geometry("points")
            if len(self.frame.points):
                scene.add_geometry("points", self._cloud, self._point_material())
        elif scene.has_geometry("points"):
            scene.modify_geometry_material("points", self._point_material())
        self.widget.force_redraw()

    def update_boxes(self):
        if self.frame is None:
            return
        self._replace_boxes(self._box_geometries(self.frame))
        self.widget.force_redraw()

    def _box_geometries(self, frame):
        grouped: dict[str, list[np.ndarray]] = {}
        for box in frame.boxes:
            corners = box_corners(box)
            if box.class_name in self.settings.selected_classes:
                grouped.setdefault(box.class_name, []).append(corners)
        geometries = []
        # One geometry per class keeps draw calls small even with many boxes.
        for class_name, corners in grouped.items():
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(np.concatenate(corners))
            edges = np.concatenate(
                [BOX_EDGES + 8 * index for index in range(len(corners))]
            )
            lines.lines = o3d.utility.Vector2iVector(edges)
            color = self.settings.class_colors.get(class_name, class_color(class_name))
            lines.paint_uniform_color(color)
            material = rendering.MaterialRecord()
            material.shader = "unlitLine"
            material.line_width = self.settings.line_width * self.scaling
            name = f"boxes:{class_name}"
            geometries.append((name, lines, material))
        return geometries

    def _replace_boxes(self, geometries):
        scene = self.widget.scene
        for name in self._box_names:
            scene.remove_geometry(name)
        self._box_names.clear()
        for name, lines, material in geometries:
            scene.add_geometry(name, lines, material)
            self._box_names.append(name)

    def reset_camera(self):
        if self.frame is None:
            return
        geometry = []
        points = self._display_points(self.frame)
        if len(points):
            geometry.append(points)
        geometry.extend(box_corners(box) for box in self.frame.boxes)
        if geometry:
            points = np.concatenate(geometry)
            lower, upper = points.min(axis=0), points.max(axis=0)
            # Give an empty/flat scan a usable camera and clipping range.
            center = (lower + upper) / 2.0
            extent = np.maximum(upper - lower, [2.0, 2.0, 2.0])
            bounds = o3d.geometry.AxisAlignedBoundingBox(
                center - extent / 2.0, center + extent / 2.0
            )
        else:
            center = np.zeros(3)
            extent = np.array([20.0, 20.0, 10.0])
            bounds = o3d.geometry.AxisAlignedBoundingBox(-extent / 2.0, extent / 2.0)
        self.widget.setup_camera(60.0, bounds, center)
        distance = max(float(extent.max()), 10.0)
        eye = center + [0.0, -0.9 * distance, 0.7 * distance]
        self.widget.scene.camera.look_at(center, eye, [0.0, 0.0, 1.0])
        self.widget.force_redraw()
