"""Editable camera pose shared by the point-cloud and detection windows."""

import atexit
from dataclasses import asdict, dataclass
import json

import glfw
import numpy as np
from open3d.visualization import gui, rendering


_clipboard_ready = False


def _clipboard():
    # Open3D's Linux text widgets keep a private clipboard per window.
    global _clipboard_ready
    if not _clipboard_ready:
        if not glfw.init():
            raise RuntimeError("The system clipboard is unavailable.")
        _clipboard_ready = True
        atexit.register(glfw.terminate)
    return glfw


def _rotation_matrix(angles):
    """Camera-to-reference rotation, Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    x, y, z = np.radians(angles)
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def _rotation_angles(matrix):
    horizontal = np.hypot(matrix[0, 0], matrix[1, 0])
    pitch = np.arctan2(-matrix[2, 0], horizontal)
    if horizontal > 1e-6:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return tuple(float(value) for value in np.degrees([roll, pitch, yaw]))


@dataclass(frozen=True)
class CameraView:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    pivot: tuple[float, float, float]
    fov: float
    near: float
    far: float

    def __post_init__(self):
        for name in ("position", "rotation", "pivot"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not np.isfinite(values).all():
                raise ValueError(f"{name.capitalize()} requires three finite numbers.")
            object.__setattr__(self, name, values)
        for name in ("fov", "near", "far"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name.capitalize()} must be finite.")
            object.__setattr__(self, name, value)
        if not 0 < self.fov < 180:
            raise ValueError("Vertical field of view must be between 0 and 180 degrees.")
        if not 0 < self.near < self.far:
            raise ValueError("Clipping distances must satisfy 0 < near < far.")

    @classmethod
    def capture(cls, viewport):
        camera = viewport.scene.camera
        model = camera.get_model_matrix()
        projection = camera.get_projection_matrix()
        fov = np.degrees(2 * np.arctan(1 / projection[1, 1]))
        return cls(model[:3, 3], _rotation_angles(model[:3, :3]),
                   viewport.center_of_rotation, fov, camera.get_near(), camera.get_far())

    def apply(self, viewport):
        rotation = _rotation_matrix(self.rotation)
        eye = np.asarray(self.position)
        viewport.look_at(eye - rotation[:, 2], eye, rotation[:, 1])
        viewport.center_of_rotation = np.asarray(self.pivot)
        viewport.scene.camera.set_projection(
            self.fov, max(1, viewport.frame.width) / max(1, viewport.frame.height),
            self.near, self.far, rendering.Camera.FovType.Vertical,
        )
        viewport.force_redraw()


class CameraPanel:
    def __init__(self, viewport, font_size):
        self.viewport = viewport
        self._signature = None
        self._dirty = False
        self._updating = False
        self.widget = gui.Vert(0.25 * font_size, gui.Margins(*([0.7 * font_size] * 4)))
        self.widget.background_color = gui.Color(0.08, 0.08, 0.08, 0.5)
        title = gui.Label("Full view")
        title.tooltip = "Copy the current view, then paste it into another window and Apply."
        self.widget.add_child(title)
        self.view_edit = gui.TextEdit()
        self.view_edit.tooltip = "The complete camera view, ready to copy or replace."
        self.view_edit.set_on_text_changed(self._on_edit)
        self.widget.add_child(self.view_edit)
        buttons = gui.Horiz(0.4 * font_size)
        for label, callback in (("Copy view", self.copy_view), ("Paste view", self.paste_view)):
            button = gui.Button(label)
            button.set_on_clicked(callback)
            buttons.add_child(button)
        self.apply_button = gui.Button("Apply")
        self.apply_button.enabled = False
        self.apply_button.set_on_clicked(self.apply)
        buttons.add_child(self.apply_button)
        self.widget.add_child(buttons)
        self.message = gui.Label("Load a scene to copy its view.")
        self.widget.add_child(self.message)
        self.widget.enabled = False

    def _on_edit(self, _text):
        if not self._updating:
            self._dirty = True
            self.apply_button.enabled = True
            self.message.text = "Apply to use this view."

    def _show(self, view):
        self._updating = True
        try:
            self.view_edit.text_value = json.dumps(asdict(view), separators=(",", ":"))
        finally:
            self._updating = False
        self._dirty = False
        self.apply_button.enabled = False
        self.message.text = "Copy this view, or paste one and Apply."

    def sync(self):
        if _clipboard_ready:
            glfw.poll_events()  # Serve clipboard requests from other processes.
        camera = self.viewport.scene.camera
        signature = np.concatenate((camera.get_model_matrix().ravel(),
                                    camera.get_projection_matrix().ravel(),
                                    self.viewport.center_of_rotation))
        if np.array_equal(signature, self._signature):
            return False
        self._signature = signature
        self.widget.enabled = True
        if self._dirty:
            return False
        self._show(CameraView.capture(self.viewport))
        return True

    def apply(self):
        try:
            view = CameraView(**json.loads(self.view_edit.text_value))
        except (ValueError, TypeError, OverflowError) as error:
            self.message.text = str(error)
            return
        view.apply(self.viewport)
        self._show(CameraView.capture(self.viewport))

    def copy_view(self):
        try:
            view = CameraView.capture(self.viewport)
            text = json.dumps(asdict(view), separators=(",", ":"))
            _clipboard().set_clipboard_string(None, text)
            self._show(view)
            self.message.text = "View copied to the system clipboard."
        except RuntimeError as error:
            self.message.text = str(error)

    def paste_view(self):
        try:
            text = _clipboard().get_clipboard_string(None)
            if not text:
                raise ValueError("The clipboard is empty.")
            view = CameraView(**json.loads(text))
        except (ValueError, TypeError, OverflowError, RuntimeError) as error:
            self.message.text = str(error)
            return
        self._show(view)
        self._dirty = True
        self.apply_button.enabled = True
        self.message.text = "View pasted · Apply to move the camera"
