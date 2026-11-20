"""Binary browser geometry preserves coordinates, identities and validity."""

from dataclasses import replace
import json
import struct

import numpy as np
import pytest

from visual_utils import (
    AcquisitionGeometry, AnnotationSet, AnnotationStatus, Box3D, FrameInfo,
    Modality, PointCloudFrame, PointColors, ProcessingProvenance,
)
from threedviz.web.wire import encode_frame


def _decode(packet):
    length, = struct.unpack_from("<I", packet)
    assert length % 4 == 0
    header = json.loads(packet[4:4 + length])
    payload = memoryview(packet)[4 + length:]
    arrays = {}
    for item in header["buffers"]:
        assert item["offset"] % 4 == 0
        data = payload[item["offset"]:item["offset"] + item["length"]]
        arrays[item["name"]] = np.frombuffer(data, dtype=item["dtype"]).reshape(item["shape"])
    return header, arrays


def _frame(points=None, *, boxes=()):
    info = FrameInfo("frame-α", "scene", 3, 2**60 + 123, frozenset({Modality.POINT_CLOUD}))
    if points is None:
        points = np.array([[1_000_000.001, 2_000_000.002, 3.003],
                           [1_000_000.011, 2_000_000.012, 3.013]], dtype=np.float64)
    return PointCloudFrame(info, "source", "reference", points,
                           AnnotationSet("reference", AnnotationStatus.AVAILABLE, boxes))


def test_binary_origin_rebasing_preserves_centimetres_and_original_distance():
    frame = _frame()
    header, arrays = _decode(encode_frame(frame, 7))
    assert header["protocol"] == 1 and header["revision"] == 7
    assert header["count"] == 2 and arrays["positions"].dtype == np.dtype("<f4")
    np.testing.assert_allclose(arrays["positions"] + header["origin"], frame.points, atol=1e-8, rtol=0)
    np.testing.assert_allclose(arrays["distance"], np.linalg.norm(frame.points, axis=1), rtol=1e-7)
    assert np.linalg.norm(arrays["positions"][0]) < .01
    assert header["frame"]["timestamp_us"] == str(2**60 + 123)
    assert header["frame"]["id"] == "frame-α"
    assert header["source_id"] == "source" and header["reference_frame_id"] == "reference"
    assert header["annotation_status"] == "available"
    assert set(arrays) == {"positions", "distance"}


def test_rgb_mask_preserves_real_black_pixels_and_raw_geometry_row_correspondence():
    frame = _frame(np.array([[0., 0., 0.], [4., 0., 0.], [8., 0., 0.]]))
    provenance = ProcessingProvenance("fixture")
    raw = frame.points + [0., 1., 0.]
    frame = replace(frame,
                    rgb=PointColors(np.array([[0., 0., 0.], [1., .5, .25], [.1, .2, .3]]),
                                    np.array([True, False, True]), provenance),
                    acquisition=AcquisitionGeometry(raw, np.ones(3, bool), "reference", provenance))
    header, arrays = _decode(encode_frame(frame, 1, rgb=True, raw=True))
    np.testing.assert_allclose(arrays["positions"] + header["origin"], raw)
    np.testing.assert_allclose(arrays["distance"], np.linalg.norm(raw, axis=1))
    np.testing.assert_array_equal(arrays["rgb_mask"], [1, 0, 1])
    np.testing.assert_array_equal(arrays["rgb"][0], [0, 0, 0])
    np.testing.assert_array_equal(arrays["rgb"][1], [255, 128, 64])
    assert header["rgb_count"] == 2 and header["display"] == {"rgb": True, "raw": True}
    # Source records remain immutable, including their original reference data.
    np.testing.assert_array_equal(frame.points[:, 1], [0., 0., 0.])


def test_boxes_are_reference_coordinates_and_contribute_to_camera_bounds():
    box = Box3D("box-id", "native.class", np.array([100., 0., 1.]), np.array([6., 2., 2.]), np.eye(3))
    frame = _frame(np.array([[0., 0., 0.]]), boxes=(box,))
    header, _ = _decode(encode_frame(frame, 1, class_groups={"native.class": "car"}))
    assert header["bounds"] == {"min": [0., -1., 0.], "max": [103., 1., 2.]}
    annotation = header["boxes"][0]
    assert annotation["id"] == "box-id" and annotation["class_id"] == "native.class"
    assert annotation["group"] == "car"
    np.testing.assert_allclose(np.mean(annotation["corners"], axis=0), box.center)


def test_empty_withheld_cloud_is_valid_and_has_usable_bounds():
    frame = _frame(np.empty((0, 3)))
    frame = replace(frame, annotations=AnnotationSet("reference", AnnotationStatus.WITHHELD))
    header, arrays = _decode(encode_frame(frame, 1))
    assert header["count"] == 0 and header["annotation_status"] == "withheld"
    assert arrays["positions"].shape == (0, 3) and arrays["distance"].shape == (0,)
    assert header["origin"] == [0., 0., 0.]
    assert header["bounds"] == {"min": [-1., -1., -1.], "max": [1., 1., 1.]}


@pytest.mark.parametrize("raw,rgb,message", [(True, False, "Acquisition"), (False, True, "RGB")])
def test_unprepared_features_are_rejected(raw, rgb, message):
    with pytest.raises(ValueError, match=message):
        encode_frame(_frame(), 1, raw=raw, rgb=rgb)


def test_partial_acquisition_geometry_cannot_be_mislabelled_as_raw_scan():
    frame = _frame()
    frame = replace(frame, acquisition=AcquisitionGeometry(
        frame.points, np.array([True, False]), "reference", ProcessingProvenance("partial")))
    with pytest.raises(ValueError, match="every point"):
        encode_frame(frame, 1, raw=True)


def test_oversized_or_unrepresentable_frame_fails_explicitly():
    with pytest.raises(ValueError, match="size limit"):
        encode_frame(_frame(), 1, max_bytes=20)
    with pytest.raises(ValueError, match="float32"):
        encode_frame(_frame(np.array([[1e100, 0., 0.]])), 1)
