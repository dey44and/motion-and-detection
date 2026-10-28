"""Optional display requests operate directly on canonical observations."""

from dataclasses import replace

import numpy as np
import pytest

from visual_utils import (
    AcquisitionGeometry, Capability, PointColors, ProcessingProvenance,
    UnsupportedCapabilityError,
)
from threedviz.dataloader import DemoLoader


@pytest.fixture
def viewer():
    pytest.importorskip("open3d")
    from threedviz.ui.app import ViewerWindow
    return ViewerWindow.__new__(ViewerWindow)


class OptionalData(DemoLoader):
    def __init__(self, coverage="full", *, partial=False):
        super().__init__()
        self.requests = []
        self.partial = partial
        self._metadata = replace(
            self.metadata,
            capabilities=self.metadata.capabilities | {Capability.RGB_COLORING, Capability.ACQUISITION_GEOMETRY},
            feature_coverage={Capability.ACQUISITION_GEOMETRY: coverage},
        )

    def _prepare_point_cloud(self, frame, *, request):
        self.requests.append(request)
        values = {}
        if request.rgb:
            values["rgb"] = PointColors(np.full(frame.points.shape, .5), np.ones(len(frame.points), dtype=bool),
                                        ProcessingProvenance("synthetic-rgb"))
        if request.acquisition_geometry:
            mask = np.ones(len(frame.points), dtype=bool)
            mask[-1] = not self.partial
            values["acquisition"] = AcquisitionGeometry(
                frame.points - [1., 0, 0], mask, "lidar@acquisition", ProcessingProvenance("synthetic-poses"))
        return replace(frame, **values)


@pytest.mark.parametrize("first", ["rgb", "acquisition"])
def test_preparing_one_display_feature_preserves_the_other_without_reloading(viewer, first):
    with OptionalData() as loader:
        frame = loader.load_point_cloud("demo-urban-000")
        rgb_first = first == "rgb"
        prepared = viewer._prepare_frame(loader, frame, "camera_rgb" if rgb_first else "gray", not rgb_first)
        for index in range(1, 5):
            loader.load_point_cloud(f"demo-urban-{index:03d}")
        result = viewer._prepare_frame(loader, prepared, "camera_rgb", True)
        assert result.rgb is not None and result.acquisition is not None
        np.testing.assert_array_equal(result.points, frame.points)
        assert result.annotations is frame.annotations
        if rgb_first:
            assert result.rgb is prepared.rgb
        else:
            assert result.acquisition is prepared.acquisition
        assert all(not request.timestamps for request in loader.requests)
        assert viewer._prepare_frame(loader, result, "camera_rgb", True) is result


@pytest.mark.parametrize("coverage,partial", [("partial", False), ("full", True)])
def test_partial_acquisition_geometry_cannot_be_displayed_as_a_complete_scan(viewer, coverage, partial):
    with OptionalData(coverage, partial=partial) as loader:
        frame = loader.load_point_cloud("demo-urban-000")
        with pytest.raises(UnsupportedCapabilityError, match="geometry"):
            viewer._prepare_frame(loader, frame, "gray", True)
        assert frame.acquisition is None
