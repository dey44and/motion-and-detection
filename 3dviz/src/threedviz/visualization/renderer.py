"""Render normalized frames without knowing their source dataset."""

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from threedviz.dataloader import BoundingBox3D, FrameData
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


@dataclass(frozen=True)
class BoxLayer:
    """Independently styled boxes in the current frame's sensor coordinates.

    A layer uses one RGB color across all classes. Class selection still gates
    each box; ``visible`` independently controls the entire layer. An omitted
    width follows the renderer's common bounding-box line-width setting.
    """

    name: str
    boxes: tuple[BoundingBox3D, ...]
    color: tuple[float, float, float]
    visible: bool = True
    line_width: float | None = None


_UNCHANGED_LAYERS = object()
_COORDINATE_AXES = "coordinate_axes"
_AXES_LENGTH_METRES = 3.0


def _vector3d(values):
    # Open3D's Eigen binding requests a writable float64 buffer even though
    # it copies the values. Shared loader arrays remain immutable.
    array = np.asarray(values, dtype=np.float64, order="C")
    return o3d.utility.Vector3dVector(array if array.flags.writeable else array.copy())


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
        self._box_layers: tuple[BoxLayer, ...] | None = None
        self._box_names: list[str] = []
        self._cloud = o3d.geometry.PointCloud()
        self._axes_visible = True
        self._axes_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(size=_AXES_LENGTH_METRES)
        self._axes_material = rendering.MaterialRecord()
        self._axes_material.shader = "defaultUnlit"
        self.widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        # Preserve the white background and the chosen RGB colors without
        # photographic tone mapping or exposure changes.
        self.widget.scene.view.set_post_processing(False)
        # Open3D's built-in axes are rebuilt at the full scene extent when
        # toggled back on. A scan can make them hundreds of metres long, and
        # its accumulated bounds also retain geometry from earlier frames.
        self.widget.scene.show_axes(False)
        self.widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def set_frame(
        self, frame: FrameData, *, reset_camera: bool = False,
        box_layers: tuple[BoxLayer, ...] | None | object = _UNCHANGED_LAYERS,
    ):
        """Replace a frame, optionally replacing its box layers atomically.

        Omit ``box_layers`` to retain the current configuration. Pass ``None``
        to use the ordinary per-class-colored ``frame.boxes``; pass a tuple to
        render exactly those layers, including an empty tuple to hide all boxes.
        """
        # Validate and construct all geometry before replacing a working frame.
        layers = self._box_layers if box_layers is _UNCHANGED_LAYERS else self._validate_layers(box_layers)
        points = self._display_points(frame)
        colors = self._point_colors(frame, points)
        cloud = o3d.geometry.PointCloud()
        cloud.points = _vector3d(points)
        cloud.colors = _vector3d(colors)
        boxes = self._box_geometries(frame, layers)
        scene = self.widget.scene
        if scene.has_geometry("points"):
            scene.remove_geometry("points")
        if len(points):
            scene.add_geometry("points", cloud, self._point_material())
        self._replace_boxes(boxes)
        self.frame, self._cloud, self._box_layers = frame, cloud, layers
        self._update_axes()
        if reset_camera:
            self.reset_camera()
        self.widget.force_redraw()

    @property
    def axes_visible(self) -> bool:
        return self._axes_visible

    def show_axes(self, visible: bool):
        """Toggle fixed metre-scale sensor axes without changing the camera.

        The same mesh is retained across toggles and frame/style changes. It
        does not contribute to the bounds used by ``reset_camera``.
        """
        if not isinstance(visible, (bool, np.bool_)):
            raise TypeError("Axes visibility must be boolean.")
        self._axes_visible = bool(visible)
        self._update_axes()
        self.widget.force_redraw()

    def _update_axes(self):
        if self.frame is None:
            return
        scene = self.widget.scene
        if not scene.has_geometry(_COORDINATE_AXES):
            scene.add_geometry(_COORDINATE_AXES, self._axes_geometry, self._axes_material, False)
        scene.show_geometry(_COORDINATE_AXES, self._axes_visible)

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
            self._cloud.colors = _vector3d(colors)
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
        self._replace_boxes(self._box_geometries(self.frame, self._box_layers))
        self.widget.force_redraw()

    @property
    def box_layers(self) -> tuple[BoxLayer, ...] | None:
        return self._box_layers

    @staticmethod
    def _validate_layers(layers):
        if layers is None:
            return None
        try:
            layers = tuple(layers)
        except TypeError as exc:
            raise ValueError("Box layers must be a tuple of BoxLayer objects or None.") from exc
        names = set()
        for layer in layers:
            if not isinstance(layer, BoxLayer):
                raise ValueError("Box layers must contain BoxLayer objects.")
            if not isinstance(layer.name, str) or not layer.name.strip():
                raise ValueError("Box layer names must be nonempty strings.")
            if layer.name in names:
                raise ValueError(f"Duplicate box layer name {layer.name!r}.")
            names.add(layer.name)
            color = np.asarray(layer.color, dtype=float)
            if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
                raise ValueError("Box layer colors must contain three finite RGB values between zero and one.")
            if not isinstance(layer.visible, (bool, np.bool_)):
                raise ValueError("Box layer visibility must be boolean.")
            if layer.line_width is not None:
                try:
                    valid_width = (
                        not isinstance(layer.line_width, (bool, np.bool_))
                        and np.isscalar(layer.line_width)
                        and np.isfinite(layer.line_width) and layer.line_width > 0
                    )
                except (TypeError, ValueError):
                    valid_width = False
                if not valid_width:
                    raise ValueError("Box layer line width must be positive and finite.")
            for box in layer.boxes:
                if not isinstance(box, BoundingBox3D):
                    raise ValueError("Box layers must contain BoundingBox3D objects.")
                box_corners(box)
        return layers

    def set_box_layers(self, layers: tuple[BoxLayer, ...] | None):
        """Update box geometry only, retaining the point cloud and camera.

        Invalid layers leave both the current layer settings and visible scene
        intact. Use ``set_frame(..., box_layers=...)`` to change frame and layers
        together without displaying layers from the previous frame.
        """
        layers = self._validate_layers(layers)
        geometries = self._box_geometries(self.frame, layers) if self.frame is not None else None
        if geometries is not None:
            self._replace_boxes(geometries)
        self._box_layers = layers
        self.widget.force_redraw()

    def _box_geometries(self, frame, layers=None):
        geometries = []
        sources = ((None, frame.boxes),) if layers is None else tuple((layer, layer.boxes) for layer in layers)
        for layer, boxes in sources:
            grouped: dict[str, list[np.ndarray]] = {}
            for box in boxes:
                corners = box_corners(box)
                if box.class_name in self.settings.selected_classes and (layer is None or layer.visible):
                    grouped.setdefault(box.class_name, []).append(corners)
            # One geometry per layer/class keeps draw calls small while
            # preserving independent layer controls and class filtering.
            for class_name, corners in grouped.items():
                lines = o3d.geometry.LineSet()
                lines.points = o3d.utility.Vector3dVector(np.concatenate(corners))
                edges = np.concatenate([BOX_EDGES + 8 * index for index in range(len(corners))])
                lines.lines = o3d.utility.Vector2iVector(edges)
                color = (self.settings.class_colors.get(class_name, class_color(class_name))
                         if layer is None else layer.color)
                lines.paint_uniform_color(color)
                material = rendering.MaterialRecord()
                material.shader = "unlitLine"
                width = (self.settings.line_width if layer is None or layer.line_width is None else layer.line_width)
                material.line_width = width * self.scaling
                name = f"boxes:{class_name}" if layer is None else f"layer:{layer.name}:{class_name}"
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
        if self._box_layers is not None:
            geometry.extend(box_corners(box) for layer in self._box_layers if layer.visible for box in layer.boxes)
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
