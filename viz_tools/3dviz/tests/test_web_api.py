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
    response = client.post(f"/api/sessions/{sid}/jobs", json={"op": "connect", "dataset": "demo", **values})
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
    pytest.fail("demo connect job did not finish")


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


@pytest.mark.parametrize("values", [
    {"op": "other"}, {"op": "connect", "frame_index": -1},
    {"op": "connect", "frame_index": 1.5}, {"op": "connect", "frame_index": True},
    {"op": "connect", "rgb": "false"}, {"op": "connect", "raw": 1},
    {"op": "connect", "dataset": ["demo"]}, {"op": "connect", "extra": "ignored?"},
    {"op": "prepare", "scene_id": "anything"},
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
for prefix in ("open3d", "glfw", "tensorflow", "visual_utils.nuscenes", "visual_utils.waymo"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
