"""Generic point bridge capabilities, ownership and partial-data behavior."""

from dataclasses import replace

import numpy as np
import pytest

from dataset_utils import DatasetConfig, DatasetLoader
from dataset_utils.contracts.models import (
    AcquisitionGeometry, Capability, DatasetMetadata, FrameInfo, Modality,
    PointCloudFrame, ProcessingProvenance, SceneInfo,
)
from dataset_utils.errors import DatasetValidationError, UnsupportedCapabilityError
from threedviz.dataloader.shared import SharedPointCloudAdapter


class GeometryOnly(DatasetLoader):
    def __init__(self, coverage="full", *, partial=False):
        super().__init__(DatasetConfig())
        self.partial = partial
        self.requests = []
        self._metadata = DatasetMetadata(
            dataset_id="geometry-only", display_name="Geometry only",
            capabilities=frozenset((Capability.POINT_CLOUD, Capability.ACQUISITION_GEOMETRY)),
            feature_coverage={Capability.ACQUISITION_GEOMETRY: coverage})

    @property
    def metadata(self):
        return self._metadata

    def list_scenes(self, *, split="all"):
        return (SceneInfo("scene", "Scene"),)

    def list_frames(self, scene_id, *, modality):
        return tuple(FrameInfo(str(index), scene_id, index, None, frozenset((Modality.POINT_CLOUD,)))
                     for index in range(3))

    def load_point_cloud(self, frame_id):
        index = int(frame_id)
        return PointCloudFrame(self.list_frames("scene", modality=Modality.POINT_CLOUD)[index],
                               "points", "reference", np.array([[index, 0., 0], [index, 1., 0]]))

    def prepare_point_cloud(self, frame, *, request):
        self.requests.append(request)
        mask = np.array([True, not self.partial])
        return replace(frame, acquisition=AcquisitionGeometry(
            frame.points - [1., 0, 0], mask, "reference@acquisition", ProcessingProvenance("pixel-poses")))


@pytest.mark.parametrize("coverage", ["full", "varies"])
def test_geometry_can_be_prepared_without_scalar_timestamps_after_cache_eviction(coverage):
    loader = GeometryOnly(coverage)
    adapter = SharedPointCloudAdapter(loader)
    frame = adapter.load_frame("0")
    adapter.load_frame("1")
    adapter.load_frame("2")
    assert "0" not in adapter._normalized_frames
    assert adapter.supports_acquisition_geometry and not adapter.supports_point_timing
    prepared = adapter.prepare_frame(frame, point_timing=True)
    np.testing.assert_array_equal(prepared.raw_points, frame.points - [1., 0, 0])
    assert prepared.point_timestamps_us is None
    assert prepared.reference_sensor_to_world is None
    assert prepared.points is frame.points
    assert loader.requests[-1].acquisition_geometry
    assert not loader.requests[-1].timestamps
    assert adapter.prepare_frame(prepared, point_timing=True) is prepared
    adapter.close()


def test_partial_geometry_is_not_exposed_as_a_fully_uncompensated_scan():
    adapter = SharedPointCloudAdapter(GeometryOnly("partial", partial=True))
    assert not adapter.supports_acquisition_geometry
    frame = adapter.load_frame("0")
    with pytest.raises(UnsupportedCapabilityError, match="partial scan"):
        adapter.prepare_frame(frame, point_timing=True)
    adapter.close()


def test_unexpected_partial_frame_and_foreign_or_reordered_inputs_are_rejected():
    adapter = SharedPointCloudAdapter(GeometryOnly(partial=True))
    frame = adapter.load_frame("0")
    with pytest.raises(UnsupportedCapabilityError, match="part of this scan"):
        adapter.prepare_frame(frame, point_timing=True)
    assert frame.raw_points is None
    foreign = SharedPointCloudAdapter(GeometryOnly()).load_frame("0")
    for invalid in (foreign, replace(frame, points=frame.points[::-1])):
        with pytest.raises(DatasetValidationError, match="issued by this adapter"):
            adapter.prepare_frame(invalid, point_timing=True)
    adapter.close()
