"""Camera poses retain roll, lens settings and editable drafts across views."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("open3d")
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

from open3d.visualization import rendering

from threedviz.ui import camera as camera_module
from threedviz.ui.camera import CameraPanel, CameraView


def _perspective(fov, aspect, near, far):
    scale = 1 / np.tan(np.radians(fov) / 2)
    return np.array([
        [scale / aspect, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ])


class _Camera:
    def __init__(self, position, rotation, fov, aspect, near, far):
        self.model = np.eye(4)
        self.model[:3, :3] = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
        self.model[:3, 3] = position
        self.projection = _perspective(fov, aspect, near, far)
        self.near, self.far = near, far
        self.projection_calls = []

    def get_model_matrix(self):
        return self.model.copy()

    def get_projection_matrix(self):
        return self.projection.copy()

    def get_near(self):
        return self.near

    def get_far(self):
        return self.far

    def set_projection(self, fov, aspect, near, far, fov_type):
        self.projection_calls.append((fov, aspect, near, far, fov_type))
        self.projection = _perspective(fov, aspect, near, far)
        self.near, self.far = near, far


class _Viewport:
    """Implement look_at geometrically, independently of Euler conversion."""

    def __init__(self, *, position=(10, -4, 7), rotation=(23, -31, 61),
                 pivot=(1, 2, -3), fov=53, near=0.2, far=900,
                 width=1200, height=600):
        self.frame = SimpleNamespace(width=width, height=height)
        self.scene = SimpleNamespace(camera=_Camera(position, rotation, fov, width / height, near, far))
        self.center_of_rotation = np.array(pivot, dtype=float)
        self.look_at_calls = []
        self.redraws = 0

    def look_at(self, target, eye, up):
        self.look_at_calls.append(tuple(np.array(value, copy=True) for value in (target, eye, up)))
        backward = eye - target
        backward /= np.linalg.norm(backward)
        right = np.cross(up, backward)
        right /= np.linalg.norm(right)
        upward = np.cross(backward, right)
        self.scene.camera.model[:3, :3] = np.column_stack((right, upward, backward))
        self.scene.camera.model[:3, 3] = eye
        self.center_of_rotation = np.array(target, copy=True)

    def force_redraw(self):
        self.redraws += 1


def _assert_same_view(actual, expected):
    np.testing.assert_allclose(actual.position, expected.position, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.pivot, expected.pivot, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", actual.rotation, degrees=True).as_matrix(),
        Rotation.from_euler("xyz", expected.rotation, degrees=True).as_matrix(),
        rtol=0, atol=1e-12,
    )
    assert (actual.fov, actual.near, actual.far) == pytest.approx(
        (expected.fov, expected.near, expected.far), rel=1e-12, abs=1e-12,
    )


def _edit(panel, text):
    # Native TextEdit setters do not emit user input callbacks.
    panel.view_edit.text_value = text
    panel._on_edit(text)


@pytest.fixture
def clipboard(monkeypatch):
    class Clipboard:
        value = None

        def set_clipboard_string(self, window, text):
            assert window is None
            self.value = text.encode("utf-8")

        def get_clipboard_string(self, window):
            assert window is None
            return self.value

    clipboard = Clipboard()
    monkeypatch.setattr(camera_module, "_clipboard", lambda: clipboard)
    return clipboard


@pytest.mark.parametrize("rotation", [(23, -31, 61), (37, 90, -24), (-29, -90, 43)])
def test_capture_preserves_full_pose_including_roll_and_gimbal_lock(rotation):
    viewport = _Viewport(rotation=rotation)
    expected = CameraView((10, -4, 7), rotation, (1, 2, -3), 53, 0.2, 900)

    captured = CameraView.capture(viewport)

    _assert_same_view(captured, expected)
    assert viewport.look_at_calls == []
    assert viewport.scene.camera.projection_calls == []
    assert viewport.redraws == 0
    # At a singular pitch, equivalent Euler triples are valid. Reapplying must
    # recover the original orientation rather than depend on one decomposition.
    target = _Viewport()
    captured.apply(target)
    np.testing.assert_allclose(target.scene.camera.model, viewport.scene.camera.model, atol=1e-12)


def test_apply_uses_camera_to_reference_axes_and_keeps_pivot_independent_of_look_target():
    viewport = _Viewport(width=1250, height=500)
    view = CameraView((8, -3, 12), (35, -42, 72), (4, 5, -6), 47, 0.4, 750)
    rotation = Rotation.from_euler("xyz", view.rotation, degrees=True).as_matrix()

    view.apply(viewport)

    assert len(viewport.look_at_calls) == 1
    target, eye, up = viewport.look_at_calls[0]
    np.testing.assert_allclose(eye, view.position)
    np.testing.assert_allclose(target, np.array(view.position) - rotation[:, 2])
    np.testing.assert_allclose(up, rotation[:, 1])
    np.testing.assert_allclose(viewport.scene.camera.model[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(viewport.center_of_rotation, view.pivot)
    assert viewport.scene.camera.projection_calls == [
        (47, 2.5, 0.4, 750, rendering.Camera.FovType.Vertical),
    ]
    assert viewport.redraws == 1
    _assert_same_view(CameraView.capture(viewport), view)


def test_panel_applies_full_precision_json_view_using_local_aspect_ratio():
    viewport = _Viewport(width=600, height=1000)
    panel = CameraPanel(viewport, 14)
    assert not panel.widget.enabled
    assert not panel.apply_button.enabled
    assert panel.sync()
    assert panel.widget.enabled
    assert not panel.sync()
    expected = CameraView((8.12345678912, -3, 12), (35.12345678912, -42, 72),
                          (4, 5.12345678912, -6), 47.12345678912, 0.412345678912, 750)
    _edit(panel, json.dumps(asdict(expected)))
    assert panel.apply_button.enabled

    panel.apply()

    _assert_same_view(CameraView.capture(viewport), expected)
    _assert_same_view(CameraView(**json.loads(panel.view_edit.text_value)), expected)
    assert viewport.scene.camera.projection_calls[-1][1] == pytest.approx(0.6)
    assert not panel._dirty
    assert not panel.apply_button.enabled


@pytest.mark.parametrize("component", ["pose", "lens", "pivot"])
def test_panel_refreshes_clean_json_when_live_camera_changes(component):
    viewport = _Viewport()
    panel = CameraPanel(viewport, 14)
    panel.sync()
    position, rotation, pivot, fov, near, far = (10, -4, 7), (23, -31, 61), (1, 2, -3), 53, 0.2, 900
    if component == "pose":
        position, rotation = (-8, 7, 9), (-17, 28, 39)
        viewport.scene.camera.model[:3, :3] = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
        viewport.scene.camera.model[:3, 3] = position
    elif component == "lens":
        fov, near, far = 68, 0.6, 1800
        viewport.scene.camera.projection = _perspective(fov, 2, near, far)
        viewport.scene.camera.near, viewport.scene.camera.far = near, far
    else:
        pivot = (2, 4, 6)
        viewport.center_of_rotation = np.array(pivot, dtype=float)

    assert panel.sync()

    expected = CameraView(position, rotation, pivot, fov, near, far)
    _assert_same_view(CameraView(**json.loads(panel.view_edit.text_value)), expected)
    assert not panel._dirty
    assert not panel.apply_button.enabled
    assert not panel.sync()


def test_live_updates_preserve_unapplied_json_draft():
    viewport = _Viewport()
    panel = CameraPanel(viewport, 14)
    panel.sync()
    draft = '{"unfinished":'
    _edit(panel, draft)
    latest = CameraView((-8, 7, 9), (-17, 28, 39), (2, 4, 6), 68, 0.6, 1800)
    latest.apply(viewport)

    assert not panel.sync()
    assert panel.view_edit.text_value == draft
    assert panel._dirty
    assert panel.apply_button.enabled
    assert not panel.sync()


@pytest.mark.parametrize("invalid", [
    "{broken",
    "[]",
    '{"position":[1,2,3]}',
    {"position": [1, 2]},
    {"position": [1, float("nan"), 3]},
    {"rotation": [1, float("inf"), 3]},
    {"pivot": ["invalid", 2, 3]},
    {"fov": 0},
    {"fov": 180},
    {"near": 0},
    {"near": 2, "far": 2},
    {"near": 10, "far": 2},
    {"far": float("inf")},
])
def test_invalid_json_apply_leaves_camera_and_draft_unchanged(invalid):
    viewport = _Viewport()
    panel = CameraPanel(viewport, 14)
    panel.sync()
    before = CameraView.capture(viewport)
    text = json.dumps({**asdict(before), **invalid}) if isinstance(invalid, dict) else invalid
    _edit(panel, text)
    edit_message = panel.message.text

    panel.apply()

    _assert_same_view(CameraView.capture(viewport), before)
    assert viewport.look_at_calls == []
    assert viewport.scene.camera.projection_calls == []
    assert viewport.redraws == 0
    assert panel.view_edit.text_value == text
    assert panel._dirty
    assert panel.apply_button.enabled
    assert panel.message.text != edit_message


def test_clipboard_copy_refreshes_draft_from_live_view_and_paste_requires_apply(clipboard):
    source = _Viewport()
    panel = CameraPanel(source, 14)
    panel.sync()
    _edit(panel, '{"unwanted draft":')
    latest = CameraView((-8, 7, 9), (-17, 28, 39), (2, 4, 6), 68, 0.6, 1800)
    latest.apply(source)

    panel.copy_view()

    _assert_same_view(CameraView(**json.loads(clipboard.value)), latest)
    _assert_same_view(CameraView(**json.loads(panel.view_edit.text_value)), latest)
    assert not panel._dirty
    assert not panel.apply_button.enabled
    assert source.redraws == 1

    destination = _Viewport(width=600, height=1000)
    other_panel = CameraPanel(destination, 14)
    other_panel.sync()
    before = CameraView.capture(destination)

    other_panel.paste_view()

    _assert_same_view(CameraView.capture(destination), before)
    assert destination.look_at_calls == []
    assert destination.scene.camera.projection_calls == []
    assert destination.redraws == 0
    _assert_same_view(CameraView(**json.loads(other_panel.view_edit.text_value)), latest)
    assert other_panel._dirty
    assert other_panel.apply_button.enabled
    assert not other_panel.sync()

    other_panel.apply()

    _assert_same_view(CameraView.capture(destination), latest)
    assert destination.scene.camera.projection_calls[-1][1] == pytest.approx(0.6)
    assert destination.redraws == 1
    assert not other_panel.apply_button.enabled


@pytest.mark.parametrize("value", [
    None,
    b"",
    b"not JSON",
    b"\xff",
    b"[]",
    b'{"position":[1,2,3]}',
    b'{"position":[1,2,3],"rotation":[0,0,0],"pivot":[0,0,0],"fov":45,"near":5,"far":1}',
])
def test_invalid_clipboard_paste_preserves_camera_and_existing_draft(clipboard, value):
    viewport = _Viewport()
    panel = CameraPanel(viewport, 14)
    panel.sync()
    draft = '{"unfinished":'
    _edit(panel, draft)
    edit_message = panel.message.text
    before = CameraView.capture(viewport)
    clipboard.value = value

    panel.paste_view()

    _assert_same_view(CameraView.capture(viewport), before)
    assert viewport.look_at_calls == []
    assert viewport.scene.camera.projection_calls == []
    assert viewport.redraws == 0
    assert panel.view_edit.text_value == draft
    assert panel._dirty
    assert panel.apply_button.enabled
    assert panel.message.text != edit_message


@pytest.mark.parametrize("action", ["copy_view", "paste_view"])
@pytest.mark.parametrize("stage", ["initialize", "access"])
def test_unavailable_clipboard_reports_error_and_preserves_draft(monkeypatch, action, stage):
    def unavailable(*_args):
        raise RuntimeError("The system clipboard is unavailable.")

    if stage == "initialize":
        monkeypatch.setattr(camera_module, "_clipboard", unavailable)
    else:
        api = SimpleNamespace(set_clipboard_string=unavailable, get_clipboard_string=unavailable)
        monkeypatch.setattr(camera_module, "_clipboard", lambda: api)
    viewport = _Viewport()
    panel = CameraPanel(viewport, 14)
    panel.sync()
    draft = '{"unfinished":'
    _edit(panel, draft)
    before = CameraView.capture(viewport)

    getattr(panel, action)()

    _assert_same_view(CameraView.capture(viewport), before)
    assert viewport.look_at_calls == []
    assert viewport.scene.camera.projection_calls == []
    assert viewport.redraws == 0
    assert panel.view_edit.text_value == draft
    assert panel._dirty
    assert panel.apply_button.enabled
    assert panel.message.text == "The system clipboard is unavailable."
