"""Browser sessions commit atomically and own serialized reader lifetimes."""

from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from visual_utils import (
    AcquisitionGeometry, AnnotationSet, AnnotationStatus, Capability, DatasetConfig,
    DatasetMetadata, PointCloudFrame, PointColors, ProcessingProvenance,
)
from threedviz.dataloader.demo import DemoLoader
from threedviz.web.service import BrowserError, LaunchOptions, SessionManager


class FixtureLoader(DemoLoader):
    def __init__(self, config, name="fixture"):
        super().__init__(config)
        self._metadata = replace(self._metadata, dataset_id=name, display_name=name,
                                 release="test-1", capabilities=self._metadata.capabilities | {
                                     Capability.RGB_COLORING, Capability.ACQUISITION_GEOMETRY},
                                 feature_coverage={Capability.ACQUISITION_GEOMETRY: "full"})
        self.calls = []
        self.close_count = 0
        self.block_read = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_read = self.fail_prepare = self.partial = False

    def _load_point_cloud(self, frame_id):
        self.calls.append(("read", frame_id, threading.current_thread().name))
        if self.block_read:
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("fixture reader was not released")
        if self.fail_read:
            raise ValueError("fixture cloud failure")
        info = self._frames[frame_id]
        points = np.array([[0., 1., 2.], [10., 0., -1.], [12., -2., .5]]) + [info.index, 0., 0.]
        return PointCloudFrame(info, "fixture:source", "fixture:reference", points,
                               AnnotationSet("fixture:reference", AnnotationStatus.AVAILABLE))

    def _prepare_point_cloud(self, frame, *, request):
        self.calls.append(("prepare", request, threading.current_thread().name))
        if self.fail_prepare:
            raise ValueError("fixture RGB failure")
        extra = {}
        if request.rgb:
            extra["rgb"] = PointColors(np.array([[0., 0., 0.], [1., .5, .25], [.2, .3, .4]]),
                                       np.array([True, True, False]), ProcessingProvenance("fixture:rgb"))
        if request.acquisition_geometry:
            extra["acquisition"] = AcquisitionGeometry(
                frame.points + [0., 0., 1.], np.array([True, True, not self.partial]),
                frame.reference_frame_id, ProcessingProvenance("fixture:raw"))
        return replace(frame, **extra)

    def _close(self):
        self.calls.append(("close", None, threading.current_thread().name))
        self.close_count += 1


class Factory:
    def __init__(self):
        self.loaders = []
        self.next_setup = None

    def __call__(self, name, root=None, version=None, *, config=None):
        if name == "invalid":
            raise ValueError("fixture source failure")
        loader = FixtureLoader(config or DatasetConfig(root=root, release=version), name)
        if self.next_setup:
            setup, self.next_setup = self.next_setup, None
            setup(loader)
        self.loaders.append(loader)
        return loader


@pytest.fixture
def factory():
    return Factory()


@pytest.fixture
def manager(factory):
    manager = SessionManager(factory=factory, catalog=lambda: ("fixture",))
    yield manager
    manager.close()


def finish(session, values):
    job = session.start(values)["id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = session.job(job)
        if result["status"] != "running":
            return result
        time.sleep(.005)
    pytest.fail("browser loading job did not finish")


def connect(session, **extra):
    result = finish(session, {"op": "connect", "dataset": "fixture", **extra})
    assert result["status"] == "done", result
    return result["state"]


def test_tabs_have_independent_loaders_and_committed_frames(manager, factory):
    first, second = manager.create(), manager.create()
    one = connect(first, frame_index=1)
    two = connect(second, frame_index=2)
    assert first.id != second.id and len(factory.loaders) == 2
    assert one["frame_index"] == 1 and two["frame_index"] == 2
    connect(first, frame_index=3)
    assert second.snapshot()["frame_index"] == 2
    assert factory.loaders[0].close_count == 1
    assert factory.loaders[1].close_count == 0
    assert all(call[2].startswith("3dviz-browser") for item in factory.loaders for call in item.calls)


def test_read_and_prepare_failures_leave_state_and_packet_unchanged(manager, factory):
    session = manager.create()
    state = connect(session)
    packet = session.packet(state["revision"])
    factory.loaders[0].fail_read = True
    result = finish(session, {"op": "select", "frame_index": 1})
    assert result["status"] == "error" and "cloud failure" in result["error"]
    assert session.snapshot() == state and session.packet(state["revision"]) == packet
    factory.loaders[0].fail_read = False
    factory.loaders[0].fail_prepare = True
    result = finish(session, {"op": "prepare", "rgb": True})
    assert result["status"] == "error" and "RGB failure" in result["error"]
    assert session.snapshot() == state and session.packet(state["revision"]) == packet


def test_failed_new_connection_closes_candidate_and_preserves_prior_view(manager, factory):
    session = manager.create()
    state = connect(session)
    factory.next_setup = lambda loader: setattr(loader, "fail_read", True)
    result = finish(session, {"op": "connect", "dataset": "other"})
    assert result["status"] == "error"
    assert session.snapshot() == state
    assert factory.loaders[0].close_count == 0 and factory.loaders[1].close_count == 1
    result = finish(session, {"op": "connect", "dataset": "invalid"})
    assert result["status"] == "error" and session.snapshot() == state


def test_optional_preparation_is_retained_across_frames_and_scenes(manager, factory):
    session = manager.create()
    connect(session, rgb=True, raw=True)
    result = finish(session, {"op": "select", "frame_index": 4})
    assert result["status"] == "done" and result["state"]["display"] == {"rgb": True, "raw": True}
    result = finish(session, {"op": "select", "scene_id": "demo-highway"})
    assert result["status"] == "done" and result["state"]["frame_index"] == 0
    assert result["state"]["scene_id"] == "demo-highway"
    assert result["state"]["display"] == {"rgb": True, "raw": True}
    reads = [item for item in factory.loaders[0].calls if item[0] == "read"]
    prepares = [item for item in factory.loaders[0].calls if item[0] == "prepare"]
    assert len(reads) == len(prepares) == 3
    result = finish(session, {"op": "prepare", "rgb": False, "raw": False})
    assert result["state"]["display"] == {"rgb": False, "raw": False}
    assert len([item for item in factory.loaders[0].calls if item[0] == "prepare"]) == 3


def test_partial_raw_preparation_rolls_back_without_losing_rgb(manager, factory):
    session = manager.create()
    state = connect(session, rgb=True)
    factory.loaders[0].partial = True
    result = finish(session, {"op": "prepare", "raw": True})
    assert result["status"] == "error" and "every point" in result["error"]
    assert session.snapshot() == state


def test_concurrent_job_is_rejected_and_close_waits_for_pending_read(manager, factory):
    session = manager.create()
    connect(session)
    loader = factory.loaders[0]
    loader.block_read = True
    session.start({"op": "select", "frame_index": 1})
    assert loader.entered.wait(2)
    with pytest.raises(BrowserError) as error:
        session.start({"op": "prepare", "rgb": True})
    assert error.value.status == 409
    manager.delete(session.id)
    assert loader.close_count == 0
    with pytest.raises(BrowserError):
        session.snapshot()
    loader.release.set()
    session.close(wait=True)
    assert loader.close_count == 1
    assert [item[0] for item in loader.calls] == ["read", "read", "close"]
    assert session._connection is None


def test_closed_first_connection_never_commits_and_closes_new_loader(manager, factory):
    session = manager.create()
    factory.next_setup = lambda loader: setattr(loader, "block_read", True)
    session.start({"op": "connect", "dataset": "fixture"})
    deadline = time.monotonic() + 2
    while not factory.loaders and time.monotonic() < deadline:
        time.sleep(.005)
    loader = factory.loaders[0]
    assert loader.entered.wait(2)
    manager.delete(session.id)
    loader.release.set()
    session.close(wait=True)
    assert loader.close_count == 1 and session._connection is None


def test_done_jobs_keep_their_own_revision_and_old_packet_revisions_are_rejected(manager):
    session = manager.create()
    first = session.start({"op": "connect", "dataset": "fixture"})["id"]
    deadline = time.monotonic() + 2
    while session.job(first)["status"] == "running" and time.monotonic() < deadline:
        time.sleep(.005)
    old = session.job(first)
    assert old["status"] == "done"
    finish(session, {"op": "select", "frame_index": 1})
    assert session.job(first)["state"]["revision"] == old["state"]["revision"] == 1
    with pytest.raises(BrowserError) as error:
        session.packet(1)
    assert error.value.status == 409
    assert session.snapshot()["revision"] == 2


def test_launch_navigation_is_only_a_bootstrap_default(factory):
    options = LaunchOptions(dataset="fixture", scene="missing-in-new-dataset", frame=999, split="val")
    manager = SessionManager(options, factory=factory)
    try:
        state = connect(manager.create())
        assert state["scene_id"] == "demo-urban" and state["frame_index"] == 0 and state["split"] == "all"
        assert manager.bootstrap()["initial"]["scene"] == "missing-in-new-dataset"
    finally:
        manager.close()


def test_configured_launch_is_locked_and_reuses_exact_configuration(factory, tmp_path):
    config = DatasetConfig(root=tmp_path, release="fixture", options={"provider_option": "retained"})
    manager = SessionManager(LaunchOptions(dataset="fixture", config=config), factory=factory)
    try:
        session = manager.create()
        state = connect(session)
        assert factory.loaders[0].config is config and state["dataset"]["configured"]
        for changes in ({"dataset": "other"}, {"root": "/elsewhere"}, {"version": "other"}, {"config_path": "/tmp/other.json"}):
            result = finish(session, {"op": "connect", **changes})
            assert result["status"] == "error" and "fixed dataset" in result["error"]
            assert session.snapshot() == state
        assert manager.bootstrap()["initial"]["configured"] is True
    finally:
        manager.close()


def test_configured_source_with_automatic_release_can_reload_locked_form(factory, tmp_path):
    config = DatasetConfig(root=tmp_path)
    manager = SessionManager(LaunchOptions(dataset="fixture", config=config), factory=factory)
    try:
        session = manager.create()
        initial = manager.bootstrap()["initial"]
        state = connect(session, root=initial["root"], version=initial["version"])
        assert state["dataset"]["version"] == initial["version"] == ""
        source = state["dataset"]
        result = finish(session, {"op": "connect", "dataset": source["name"],
                                  "root": source["root"], "version": source["version"]})
        assert result["status"] == "done"
        assert all(loader.config is config for loader in factory.loaders)
    finally:
        manager.close()


def test_new_connection_does_not_inherit_launch_provider_root_or_release(factory, tmp_path):
    manager = SessionManager(LaunchOptions(dataset="other", root=tmp_path, version="old-release"), factory=factory)
    try:
        session = manager.create()
        connect(session)
        assert factory.loaders[0].config.root is None
        assert factory.loaders[0].config.release != "old-release"
    finally:
        manager.close()


def test_done_connection_cleanup_cannot_erase_a_queued_job(manager, factory):
    session = manager.create()
    connect(session)
    old = factory.loaders[0]
    entered, release = threading.Event(), threading.Event()
    close_hook = old._close

    def slow_close():
        entered.set()
        assert release.wait(3)
        close_hook()

    old._close = slow_close
    replacement = session.start({"op": "connect", "dataset": "other"})["id"]
    assert entered.wait(2)
    assert session.job(replacement)["status"] == "done"
    new = factory.loaders[1]
    new.block_read = True
    queued = session.start({"op": "select", "frame_index": 1})["id"]
    assert session.job(queued)["status"] == "running"
    release.set()
    assert new.entered.wait(2)
    try:
        with pytest.raises(BrowserError) as error:
            session.start({"op": "prepare", "rgb": True})
        assert error.value.status == 409
    finally:
        new.release.set()


def test_job_history_and_session_resources_are_bounded(factory):
    manager = SessionManager(factory=factory, max_sessions=1, idle_timeout=.01)
    try:
        first = manager.create()
        with pytest.raises(BrowserError) as error:
            manager.create()
        assert error.value.status == 429
        connect(first)
        for index in range(12):
            finish(first, {"op": "select", "frame_index": index})
        assert len(first._jobs) == 8
        time.sleep(.02)
        manager.expire()
        first.close(wait=True)
        second = manager.create()
        assert second.id != first.id and not manager._closing
        manager.delete(second.id)
        second.close(wait=True)
        manager.create()
        assert not manager._closing
    finally:
        manager.close()


def test_bootstrap_palettes_are_exact_and_do_not_require_desktop(manager):
    bootstrap = manager.bootstrap()
    assert bootstrap["datasets"] == ["fixture"]
    assert set(bootstrap["palettes"]) == {"gray", "viridis", "concerto"}
    concerto = np.array(bootstrap["palettes"]["concerto"])
    np.testing.assert_allclose(concerto[0], np.array([0x23, 0x1f, 0x62]) / 255)
    np.testing.assert_allclose(concerto[-1], np.array([0x56, 0xba, 0x62]) / 255)
    assert concerto.shape == (256, 3)


@pytest.mark.parametrize("values", [{"op": "select"}, {"op": "prepare", "rgb": True}])
def test_navigation_requires_a_connected_session(manager, values):
    with pytest.raises(BrowserError) as error:
        manager.create().start(values)
    assert error.value.status == 409


def test_wrong_operation_fields_and_unknown_operation_are_rejected(manager):
    session = manager.create()
    for values in ({"op": "other"}, {"op": "prepare", "frame_index": 2}, {"op": "select", "root": "/data"}):
        with pytest.raises(BrowserError) as error:
            session.start(values)
        assert error.value.status == 422
