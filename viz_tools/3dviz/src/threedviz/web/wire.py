"""Lossless row correspondence and compact, origin-rebased browser geometry.

Protocol 1 is a uint32 little-endian JSON header length, the UTF-8 header padded
with spaces to a four-byte boundary, and typed payload buffers. Buffer offsets
are relative to the payload. Coordinates stay in the dataset reference frame;
only the rendering positions are rebased before conversion to float32.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import struct

import numpy as np

from visual_utils import PointCloudFrame
from threedviz.processing import box_corners


MAX_PACKET_BYTES = 512 * 1024 * 1024


def encode_frame(
    frame: PointCloudFrame, revision: int, *, rgb: bool = False,
    raw: bool = False, class_groups: Mapping[str, str] | None = None,
    max_bytes: int = MAX_PACKET_BYTES,
) -> bytes:
    """Encode every point and annotation without modifying source arrays.

    RGB validity is separate from its pixel values, so an unmatched point and
    a physically black point remain distinguishable. Timestamps use decimal
    strings to preserve microseconds beyond JavaScript's safe integer range.
    """
    if raw:
        if frame.acquisition is None or not frame.acquisition.valid_mask.all():
            raise ValueError("Acquisition geometry must cover every point in this scan.")
        points = np.asarray(frame.acquisition.points, dtype=np.float64)
    else:
        points = np.asarray(frame.points, dtype=np.float64)
    if rgb and frame.rgb is None:
        raise ValueError("Camera RGB colors have not been prepared for this frame.")

    groups = class_groups or {}
    boxes = []
    lower = points.min(axis=0) if len(points) else None
    upper = points.max(axis=0) if len(points) else None
    for box in frame.annotations.boxes:
        corners = box_corners(box)
        lower = corners.min(axis=0) if lower is None else np.minimum(lower, corners.min(axis=0))
        upper = corners.max(axis=0) if upper is None else np.maximum(upper, corners.max(axis=0))
        boxes.append({
            "id": box.annotation_id, "class_id": box.class_id,
            "group": groups.get(box.class_id, box.class_id), "corners": corners.tolist(),
        })
    if lower is None:
        lower, upper = np.full(3, -1.0), np.full(3, 1.0)
    origin = lower / 2.0 + upper / 2.0
    rebased = points - origin
    # hypot avoids overflowing intermediate squares of large finite inputs.
    distance = np.hypot(np.hypot(points[:, 0], points[:, 1]), points[:, 2])
    limit = np.finfo(np.float32).max
    if not np.isfinite(rebased).all() or (np.abs(rebased) > limit).any() or (distance > limit).any():
        raise ValueError("This point cloud exceeds the browser geometry's float32 range.")

    # Check the predictable payload size before making the byte copies.
    estimated = len(points) * (16 + (4 if frame.rgb is not None else 0))
    if estimated > max_bytes:
        raise ValueError("This frame exceeds the browser packet size limit.")
    buffers, chunks, offset = [], [], 0

    def append(name, values, dtype):
        nonlocal offset
        padding = (-offset) % 4
        if padding:
            chunks.append(b"\0" * padding)
            offset += padding
        array = np.asarray(values, dtype=dtype, order="C")
        encoded = array.tobytes(order="C")
        buffers.append({
            "name": name, "dtype": dtype, "shape": list(array.shape),
            "offset": offset, "length": len(encoded),
        })
        chunks.append(encoded)
        offset += len(encoded)

    append("positions", rebased, "<f4")
    append("distance", distance, "<f4")
    rgb_count = 0
    if frame.rgb is not None:
        append("rgb", np.rint(frame.rgb.values * 255).clip(0, 255), "u1")
        append("rgb_mask", frame.rgb.valid_mask, "u1")
        rgb_count = int(np.count_nonzero(frame.rgb.valid_mask))
    info = frame.info
    header = {
        "protocol": 1, "revision": revision, "count": len(points),
        "origin": origin.tolist(), "bounds": {"min": lower.tolist(), "max": upper.tolist()},
        "frame": {"id": info.frame_id, "scene_id": info.scene_id, "index": info.index,
                  "timestamp_us": None if info.timestamp_us is None else str(info.timestamp_us)},
        "source_id": frame.source_id, "reference_frame_id": frame.reference_frame_id,
        "annotation_status": frame.annotations.status.value, "boxes": boxes,
        "rgb_count": rgb_count, "display": {"rgb": rgb, "raw": raw}, "buffers": buffers,
    }
    encoded_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    encoded_header += b" " * ((-len(encoded_header)) % 4)
    if 4 + len(encoded_header) + offset > max_bytes:
        raise ValueError("This frame exceeds the browser packet size limit.")
    return struct.pack("<I", len(encoded_header)) + encoded_header + b"".join(chunks)
