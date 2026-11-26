"""Local HTTP transport validates inputs and keeps each tab's state isolated."""

import json
import struct
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from threedviz.web.server import create_app, run_server
from threedviz.web.service import SessionManager


@pytest.fixture
def client():
    with TestClient(create_app(dataset="demo", autoload=True)) as client:
        yield client


def _connect(client, sid, **values):
    return _job(client, sid, op="connect", dataset="demo", **values)


def _job(client, sid, **values):
    response = client.post(f"/api/sessions/{sid}/jobs", json=values)
    assert response.status_code == 202, response.text
    job = response.json()["id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/sessions/{sid}/jobs/{job}")
        result = response.json()
        if result["status"] != "running":
            assert result["status"] == "done", result
            return result["state"]
        time.sleep(.005)
    pytest.fail("Browser job did not finish")


def test_http_demo_end_to_end_and_revision_guard(client):
    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200 and bootstrap.headers["cache-control"] == "no-store"
    assert bootstrap.json()["initial"]["dataset"] == "demo"
    response = client.post("/api/sessions")
    assert response.status_code == 201
    sid = response.json()["id"]
    assert client.get(f"/api/sessions/{sid}").json() is None
    state = _connect(client, sid, frame_index=5)
    assert state == client.get(f"/api/sessions/{sid}").json()
    assert state["frame_index"] == 5 and state["dataset"]["name"] == "demo"
    assert state["metadata"]["classes"][0]["group"] == "car"
    packet = client.get(f"/api/sessions/{sid}/frame.bin", params={"revision": state["revision"]})
    assert packet.status_code == 200 and packet.headers["content-type"] == "application/octet-stream"
    length, = struct.unpack_from("<I", packet.content)
    header = json.loads(packet.content[4:4 + length])
    assert header["count"] == 30000 and header["frame"]["index"] == 5
    assert header["frame"]["timestamp_us"] == "2500000"
    assert len(header["boxes"]) == 8
    assert client.get(f"/api/sessions/{sid}/frame.bin", params={"revision": state["revision"] + 1}).status_code == 409
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/sessions/{sid}").status_code == 204


def test_http_sessions_are_isolated_and_invalid_identifiers_are_bounded(client):
    first = client.post("/api/sessions").json()["id"]
    second = client.post("/api/sessions").json()["id"]
    _connect(client, first)
    assert client.get(f"/api/sessions/{second}").json() is None
    assert client.get("/api/sessions/missing").status_code == 404
    assert client.get(f"/api/sessions/{first}/jobs/missing").status_code == 404
    assert client.get(f"/api/sessions/{first}/frame.bin").status_code == 422
    assert client.get(f"/api/sessions/{first}/frame.bin?revision=0").status_code == 422


@pytest.mark.parametrize("format", ["nuscenes_json", "pcdet_pickle"])
def test_browser_detection_edits_only_boxes_and_matches_review_pipeline(client, tmp_path, monkeypatch, format):
    import pickle
    import numpy as np

    from threedviz.dataloader import DemoLoader
    from threedviz.detection.controller import compose_review
    from threedviz.detection.predictions import open_predictions
    from threedviz.detection.processing.filters import FilterConfig
    from threedviz.processing import box_corners

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    sid = client.post("/api/sessions").json()["id"]
    state = _connect(client, sid)
    revision = state["revision"]
    token, empty_token = (row["id"] for row in state["frames"][:2])
    if format == "nuscenes_json":
        path = tmp_path / "predictions.json"
        box = {"sample_token": token, "translation": [10, 20, 3], "size": [2, 4, 1.5],
               "rotation": [1, 0, 0, 0], "detection_name": "car", "velocity": [0, 0]}
        path.write_text(json.dumps({"results": {
            token: [dict(box, detection_score=0.8), dict(box, detection_score=0.6)], empty_token: [],
        }}))
    else:
        path = tmp_path / "predictions.pkl"
        path.write_bytes(pickle.dumps([
            {"metadata": {"token": token}, "name": ["car", "car"], "score": [0.8, 0.6],
             "boxes_lidar": [[10, 20, 3, 4, 2, 1.5, 0]] * 2},
            {"metadata": {"token": empty_token}, "score": [], "boxes_lidar": np.empty((0, 7))},
        ]))

    frame_url = f"/api/sessions/{sid}/frame.bin?revision={revision}"
    original_cloud = client.get(frame_url).content
    filters = {"nms_mode": "class", "nms_iou_threshold": 0.2}
    state = _job(client, sid, op="detection", revision=revision, prediction_path=str(path), filters=filters)
    assert state["revision"] == revision and client.get(frame_url).content == original_cloud
    layer = state["detection"]
    with DemoLoader() as loader, open_predictions(path) as predictions:
        reference = compose_review(loader.load_point_cloud(token), source=predictions, config=FilterConfig(**filters))
    assert layer["status"] == "ready" and layer["source"]["format"] == format
    assert layer["counts"] == {"loaded": 2, "score": 2, "spatial": 2, "pre_nms": 2, "nms": 1, "output": 1}
    assert [item["id"] for item in layer["boxes"]] == [item.identifier for item in reference.filtered.detections]
    np.testing.assert_allclose(layer["boxes"][0]["corners"], box_corners(reference.filtered.detections[0].box))
    state = _job(client, sid, op="detection", revision=revision, filters={"max_detections": 0})
    assert state["detection"]["status"] == "filtered" and client.get(frame_url).content == original_cloud
    state = _job(client, sid, op="select", frame_index=1)
    assert state["detection"]["status"] == "empty"
    assert client.post(f"/api/sessions/{sid}/jobs", json={
        "op": "detection", "revision": revision, "filters": {"score_threshold": 0},
    }).status_code == 409
    state = _job(client, sid, op="select", frame_index=2)
    assert state["detection"]["status"] == "missing"


@pytest.mark.parametrize("valid", [True, False])
def test_launch_export_and_cache_are_used_when_connecting(valid, tmp_path):
    path = tmp_path / "prediction.export"
    cache = tmp_path / "indexes"
    if valid:
        path.write_text(json.dumps({"results": {"demo-urban-000": [], "demo-urban-001": []}}))
    with TestClient(create_app(dataset="demo", predictions=path, prediction_format="nuscenes_json",
                               cache_dir=cache)) as client:
        boot = client.get("/api/bootstrap").json()
        assert boot["initial"]["prediction_path"] == str(path)
        assert boot["initial"]["prediction_format"] == "nuscenes_json"
        assert {"nuscenes_json", "pcdet_pickle"} <= set(boot["prediction_formats"])
        sid = client.post("/api/sessions").json()["id"]
        state = _connect(client, sid)
        assert state["detection"]["available"]
        assert state["detection"]["status"] == ("empty" if valid else "invalid")
        if valid:
            assert state["detection"]["source"]["path"] == str(path)
            assert list(cache.rglob("*.sqlite3"))
        else:
            assert state["detection"]["source"] is None
            assert str(path) in state["detection"]["message"]
        packet = client.get(f"/api/sessions/{sid}/frame.bin", params={"revision": state["revision"]})
        assert packet.status_code == 200
        state = _job(client, sid, op="select", frame_index=1)
        assert state["detection"]["status"] == ("empty" if valid else "invalid")
        reconnected = _connect(client, sid)
        assert reconnected["detection"]["source"]["format"] == "demo"


def test_camera_previews_and_lazy_review_assets_match_frame_revision():
    from base64 import b64decode
    from dataclasses import replace
    from io import BytesIO

    import numpy as np
    from PIL import Image
    from threedviz.dataloader import DemoLoader
    from visual_utils import (
        CameraFrameSet, CameraObservation, CameraPose, Capability, Modality, PinholeProjection, SensorInfo,
    )

    pose = np.eye(4)
    pose[:3, :3] = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    pose[:3, 3] = [1.2, 0, -0.3]
    intrinsics = np.array([[120., 0., 160.], [0., 120., 90.], [0., 0., 1.]])
    corners = np.array([[1., 4 / 3, .75], [1., -4 / 3, .75], [1., -4 / 3, -.75], [1., 4 / 3, -.75]])

    class CameraFixture(DemoLoader):
        reads = 0

        def __init__(self):
            super().__init__()
            self._metadata = replace(
                self.metadata, capabilities=self.metadata.capabilities | {Capability.CAMERAS},
                sensors=(*self.metadata.sensors, SensorInfo("rig/front", Modality.CAMERAS, "Front", "front")),
            )

        def _camera_rig(self, frame):
            return (CameraPose("rig/front", pose[:3, 3], pose[:3, 2], corner_rays=corners),)

        def _load_cameras(self, frame_id):
            self.reads += 1
            frame = self._load_point_cloud(frame_id)
            info = replace(frame.info, modalities=frame.info.modalities | {Modality.CAMERAS})
            camera = CameraObservation(
                "rig/front", np.full((180, 320, 3), 99, dtype=np.uint8),
                intrinsics=intrinsics, T_reference_from_camera=pose,
                projection_model=PinholeProjection(intrinsics, pose),
            )
            return CameraFrameSet(
                info, frame.source_id, frame.reference_frame_id, (camera,), frame.annotations,
                world_frame_id=frame.world_frame_id, T_world_from_reference=frame.T_world_from_reference,
            )

    loader = CameraFixture()
    with TestClient(create_app(factory=lambda *_args, **_kwargs: loader)) as client:
        sid = client.post("/api/sessions").json()["id"]
        base = f"/api/sessions/{sid}"
        state = _connect(client, sid)
        revision = state["revision"]
        assert state["camera_review"] is None and loader.reads == 0
        assert state["camera_rig"] == [{"id": "rig/front", "name": "Front",
                                        "position": [1.2, 0, -0.3], "forward": [1., 0, 0],
                                        "corner_rays": corners.tolist()}]
        image_url = f"{base}/camera-image/rig%2Ffront?revision={revision}"
        points_url = f"{base}/camera-points/rig%2Ffront?revision={revision}"
        preview_url = f"{base}/camera-previews?revision={revision}"
        cloud_url = f"{base}/frame.bin?revision={revision}"
        original_cloud = client.get(cloud_url).content
        assert client.get(image_url).status_code == 404
        previews = client.get(preview_url)
        assert previews.status_code == 200 and loader.reads == 1
        assert list(previews.json()) == ["rig/front"]
        with Image.open(BytesIO(b64decode(previews.json()["rig/front"]))) as preview:
            assert preview.size == (256, 144)
        assert client.get(preview_url).content == previews.content and loader.reads == 1
        assert client.get(base).json()["camera_review"] is None and client.get(image_url).status_code == 404
        state = _job(client, sid, op="cameras", revision=revision)
        field = state["camera_review"]["views"][0]
        assert state["revision"] == revision and field["point_count"] is None
        assert field["id"] == "rig/front" and field["projection"]
        assert field["ground_truth"] and field["predictions"]
        response = client.get(image_url)
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        with Image.open(BytesIO(response.content)) as image:
            assert image.size == (320, 180)
        assert client.get(points_url).status_code == 404
        state = _job(client, sid, op="cameras", revision=revision, camera_points=True)
        points = client.get(points_url)
        assert points.status_code == 200 and loader.reads == 2
        rows = list(struct.iter_unpack("<ffI", points.content))
        assert len(rows) == state["camera_review"]["views"][0]["point_count"] > 0
        assert all(0 <= u < 320 and 0 <= v < 180 and row < 30000 for u, v, row in rows)
        assert client.get(cloud_url).content == original_cloud
        state = _job(client, sid, op="detection", revision=revision, filters={"max_detections": 0})
        assert state["camera_review"] is None and client.get(image_url).status_code == 404
        state = _job(client, sid, op="cameras", revision=revision, camera_points=True)
        assert state["camera_review"]["views"][0]["predictions"] == [] and loader.reads == 2
        assert client.get(points_url).content == points.content
        state = _job(client, sid, op="select", frame_index=1)
        assert state["camera_review"] is None
        assert client.get(image_url).status_code == client.get(points_url).status_code == 409
        assert client.get(preview_url).status_code == 409
        assert client.get(f"{base}/camera-previews?revision={state['revision']}").status_code == 200
        assert loader.reads == 3


@pytest.mark.parametrize("values", [
    {"op": "other"}, {"op": "connect", "frame_index": -1},
    {"op": "connect", "frame_index": 1.5}, {"op": "connect", "frame_index": True},
    {"op": "connect", "rgb": "false"}, {"op": "connect", "raw": 1},
    {"op": "connect", "dataset": ["demo"]}, {"op": "connect", "extra": "ignored?"},
    {"op": "prepare", "scene_id": "anything"},
    {"op": "segmentation", "revision": 1, "segmentation_mode": "other"},
    {"op": "detection", "revision": True}, {"op": "connect", "prediction_path": "/data"},
    {"op": "cameras", "revision": 1, "camera_points": "false"},
])
def test_http_strict_input_validation(client, values):
    sid = client.post("/api/sessions").json()["id"]
    response = client.post(f"/api/sessions/{sid}/jobs", json=values)
    assert response.status_code == 422
    assert client.get(f"/api/sessions/{sid}").json() is None


@pytest.mark.parametrize("headers", [
    {"Origin": "https://unrelated.example"}, {"Origin": "null"},
    {"Origin": "http://localhost:8080"}, {"Sec-Fetch-Site": "cross-site"},
])
def test_cross_origin_mutations_are_rejected(client, headers):
    assert client.post("/api/sessions", headers=headers).status_code == 403


def test_same_origin_mutations_and_loopback_hosts_are_allowed(client):
    assert client.post("/api/sessions", headers={"Origin": "http://testserver"}).status_code == 201
    for host in ("localhost:8080", "127.0.0.1:8080", "[::1]:8080"):
        assert client.get("/api/bootstrap", headers={"Host": host}).status_code == 200
        assert client.post("/api/sessions", headers={"Host": host, "Origin": "http://" + host}).status_code == 201


@pytest.mark.parametrize("host", ["attacker.example", "192.168.1.5:8080", "localhost.attacker.example", "localhost:bad", "user@localhost"])
def test_nonlocal_or_malformed_hosts_are_rejected(client, host):
    assert client.get("/api/bootstrap", headers={"Host": host}).status_code == 400


def test_session_limit_is_reported_and_lifespan_closes_resources():
    manager = SessionManager(max_sessions=1)
    with TestClient(create_app(manager=manager)) as client:
        assert client.post("/api/sessions").status_code == 201
        assert client.post("/api/sessions").status_code == 429
    with pytest.raises(Exception, match="shutting down"):
        manager.create()


@pytest.mark.parametrize("host,port", [("0.0.0.0", 8080), ("example.com", 8080), ("127.0.0.1", 0), ("127.0.0.1", 65536), ("127.0.0.1", True)])
def test_server_does_not_bind_public_interfaces_or_invalid_ports(host, port):
    with pytest.raises(ValueError):
        run_server(host=host, port=port)


def test_web_transport_imports_without_desktop_or_dataset_backends():
    code = '''
import sys
from threedviz.web.server import create_app
assert create_app(dataset="demo")
for prefix in ("open3d", "glfw", "pandas", "detviz", "tkinter", "tensorflow",
               "visual_utils.nuscenes", "visual_utils.waymo"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
