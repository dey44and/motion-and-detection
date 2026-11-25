"""Row identity and label validation for browser prediction overlays."""

import pickle
from dataclasses import replace
import time
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from threedviz.web.segmentation import frame_segmentation, load_prediction_run
from visual_utils import Capability, DatasetValidationError, LidarSegmentation
from visual_utils.nuscenes.lidarseg import EVALUATION_CLASSES


def sonata_export(path):
    metadata = {
        "format": "nuscenes_lidarseg", "schema_version": 1, "dataset": "nuscenes", "dataset_version": "v1.0-trainval",
        "sensor": "LIDAR_TOP", "point_order": "original_lidar_scan", "label_dtype": "uint8", "ignore_label": 0,
        "class_ids": list(range(1, 17)), "class_names": [name for name, _ in EVALUATION_CLASSES[1:]],
    }
    frames = [{
        "metadata": {"token": "sample", "sample_data_token": "scan"},
        "pred_labels": np.array([16, 0, 1], dtype=np.uint8),
    }]
    path.write_bytes(pickle.dumps({"metadata": metadata, "frames": frames}))
    return path


def fixture_frame(sample="sample", scan="scan", point_count=3):
    loader = SimpleNamespace(
        lidar_scan_token=lambda frame_id: scan,
        metadata=SimpleNamespace(capabilities={Capability.LIDAR_SEGMENTATION}),
    )
    frame = SimpleNamespace(info=SimpleNamespace(frame_id=sample), points=np.zeros((point_count, 3)))
    return loader, frame


def test_sonata_checks_both_tokens_and_counts_in_original_order(tmp_path):
    path = sonata_export(tmp_path / "predictions.pkl")
    with pytest.raises(DatasetValidationError, match="targets nuScenes"):
        load_prediction_run(path, release="v1.0-mini")
    run = load_prediction_run(path, release="v1.0-trainval")
    assert run.public()["frame_count"] == 1
    overlay = frame_segmentation(*fixture_frame(), "prediction", run)
    assert overlay.status == "ready"
    np.testing.assert_array_equal(overlay.labels, [16, 0, 1])
    assert overlay.public()["classes"][0]["name"] == "ignored"
    assert overlay.public()["classes"][0]["count"] == 1
    assert frame_segmentation(*fixture_frame(scan="wrong"), "prediction", run).status == "invalid"
    assert frame_segmentation(*fixture_frame(sample="wrong"), "prediction", run).status == "invalid"
    assert frame_segmentation(*fixture_frame(point_count=2), "prediction", run).status == "invalid"
    missing = frame_segmentation(*fixture_frame(sample="next", scan="next-scan"), "prediction", run)
    assert missing.status == "missing" and missing.labels is None and missing.scan_token == "next-scan"


@pytest.mark.parametrize("token, values", [("wrong", [4, 4, 4]), ("scan", [4]), ("scan", [4, 4, 17])])
def test_ground_truth_does_not_color_unmatched_or_invalid_provider_labels(token, values):
    loader, frame = fixture_frame()
    labels = np.asarray(values, dtype=np.uint8)
    loader.load_lidarseg_ground_truth = lambda frame: LidarSegmentation(token, labels, labels)
    overlay = frame_segmentation(loader, frame, "ground_truth")
    assert overlay.status == "invalid" and overlay.labels is None


@pytest.mark.parametrize("archive", (False, True))
def test_official_submission_uses_sample_data_token_and_validates_ids(tmp_path, archive):
    if archive:
        path = tmp_path / "submission.zip"
        with ZipFile(path, "w") as stream:
            stream.writestr("submission/lidarseg/val/scan_lidarseg.bin", bytes([1, 3, 16]))
            stream.writestr("submission/lidarseg/val/invalid_lidarseg.bin", bytes([1, 0, 16]))
    else:
        path = tmp_path
        folder = path / "lidarseg/val"
        folder.mkdir(parents=True)
        (folder / "scan_lidarseg.bin").write_bytes(bytes([1, 3, 16]))
        (folder / "invalid_lidarseg.bin").write_bytes(bytes([1, 0, 16]))
    run = load_prediction_run(path, format="official")
    assert run.public()["split"] == "val"
    overlay = frame_segmentation(*fixture_frame(), "prediction", run)
    np.testing.assert_array_equal(overlay.labels, [1, 3, 16])
    invalid = frame_segmentation(*fixture_frame(scan="invalid"), "prediction", run)
    assert invalid.status == "invalid" and invalid.labels is None


def test_label_endpoint_rejects_old_overlay_and_frame_requests(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from threedviz.dataloader.demo import DemoLoader
    from threedviz.web.server import create_app
    from threedviz.web.service import SessionManager

    class SegmentationFixture(DemoLoader):
        def __init__(self):
            super().__init__()
            self._metadata = replace(self.metadata, capabilities=self.metadata.capabilities | {Capability.LIDAR_SEGMENTATION})

        def _lidar_scan_token(self, frame_id):
            return f"scan-{frame_id}"

        def _load_lidarseg_ground_truth(self, frame):
            count = len(frame.points)
            return LidarSegmentation(
                self.lidar_scan_token(frame.info.frame_id),
                np.full(count, 4 + frame.info.index, dtype=np.uint8),
                np.full(count, 17 + frame.info.index, dtype=np.uint8),
            )

    manager = SessionManager(factory=lambda *args, **kwargs: SegmentationFixture())
    with TestClient(create_app(manager=manager)) as client:
        session_id = client.post("/api/sessions").json()["id"]
        base = f"/api/sessions/{session_id}"

        def run_job(**payload):
            response = client.post(f"{base}/jobs", json=payload)
            assert response.status_code == 202, response.text
            identifier = response.json()["id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                job = client.get(f"{base}/jobs/{identifier}").json()
                if job["status"] != "running":
                    assert job["status"] == "done", job
                    return job["state"]
                time.sleep(.005)
            pytest.fail("Browser job did not finish.")

        def labels(state, **extra):
            return client.get(f"{base}/segmentation.bin", params={
                "revision": state["revision"], "generation": state["segmentation"]["generation"], **extra,
            })

        state = run_job(op="connect", dataset="demo")
        revision = state["revision"]
        original_point_buffer = client.get(f"{base}/frame.bin", params={"revision": revision}).content
        ground_truth = run_job(op="segmentation", revision=revision, segmentation_mode="ground_truth")
        eval_bytes = labels(ground_truth).content
        assert len(eval_bytes) == 30_000 and set(eval_bytes) == {4}
        assert set(labels(ground_truth, native="true").content) == {17}

        folder = tmp_path / "lidarseg/val"
        folder.mkdir(parents=True)
        scan = ground_truth["segmentation"]["scan_token"]
        (folder / f"{scan}_lidarseg.bin").write_bytes(bytes([16]) * len(eval_bytes))
        prediction = run_job(op="segmentation", revision=revision, segmentation_mode="prediction", segmentation_path=str(tmp_path))
        assert prediction["revision"] == revision
        assert labels(ground_truth).status_code == 409
        assert set(labels(prediction).content) == {16}
        assert client.get(f"{base}/frame.bin", params={"revision": revision}).content == original_point_buffer

        next_frame = run_job(op="select", frame_index=1)
        assert labels(prediction).status_code == 409
        assert next_frame["segmentation"]["status"] == "missing"
        assert labels(next_frame).status_code == 404
