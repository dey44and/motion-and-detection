# dataset-utils

Shared nuScenes and Waymo sensor data for `3dviz`, `2dcameraviz`, and `3ddetviz`.
The distribution is named `dataset-utils`; Python imports use `dataset_utils`.
The package contains dataset reading, common data models, calibration/pose
transforms, point timing, camera RGB projection and split definitions. It has
no GUI or renderer imports and can be used from an ordinary Python program.

The implemented [version-1 loader contract](docs/loader_contract.md) defines
the shared extension API, capabilities, records, and conformance requirements.
Both viewers use generic adapters over this API; existing low-level reader
imports remain available for compatibility.

NumPy and Pillow support the common processing paths; protobuf and CRC32C
support native Waymo records. Open3D, Tkinter, Matplotlib, TensorFlow, the
nuScenes/Waymo SDKs and detector frameworks are not required by this package.

From the repository root:

```bash
python -m pip install -e './dataset_utils[dev]'
```

Each viewer can be installed separately alongside the shared package:

```bash
python -m pip install -e './dataset_utils' -e './3dviz[dev]'
python -m pip install -e './dataset_utils' -e './2dcameraviz[dev]'
```

`3ddetviz` additionally uses `3dviz`'s existing renderer and window, so its
local-source installation includes `-e './3dviz'` as documented in that tool.

## Shared loader API

Use the same factory for either dataset and for third-party loaders:

```python
from dataset_utils import DatasetConfig, Modality, PointCloudPreparation, create_loader

config = DatasetConfig(root="/datasets/nuscenes", release="v1.0-mini")
with create_loader("nuscenes", config) as dataset:
    scene = dataset.list_scenes(split="val")[0]
    info = dataset.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD)[0]
    frame = dataset.load_point_cloud(info.frame_id)
    colored = dataset.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True))
    camera_set = dataset.load_cameras(info.frame_id)
    print(colored.rgb.valid_mask.sum(), len(camera_set.cameras))
```

Capabilities describe the configured sources. RGB, point timestamps, and
acquisition geometry are independent; masks and provenance describe coverage
and whether timing is recorded or estimated. Annotation status distinguishes
available-empty, not-provided, and withheld ground truth. Normalized arrays are
immutable snapshots and remain valid after cache eviction or loader close.

Waymo accepts `storage_format="auto"`, `"native"`, or `"openpcdet"`. Supply
`source_paths={"processed": "/data/processed", "native": "/data/original"}`
when the representations are stored independently. Automatic directory discovery
remains a convenience. Its options are `processed_tag` and `additional_classes`;
nuScenes accepts `auto`/`native`, an optional `native` source, and no options.
Unknown options, sources, and incompatible releases produce configuration errors.

Register a new backend once with `register_loader(name, constructor)` before
launching the chosen viewer in that process. The constructor receives an
immutable `DatasetConfig` and returns `DatasetLoader`. The applicable viewers
discover it through the shared registry, using sensor roles/order for camera
layout and class display groups for filtering. No dataset-specific UI code or
second per-viewer registration is needed. Automatic external-package discovery
is outside version 1; see the contract for required methods and error behavior.

The examples below expose the lower-level compatibility readers. Their record
types differ from the normalized factory API above; new extensions should
implement the contract rather than copy a viewer adapter.

## nuScenes input and API

`NuScenesReader` reads the native JSON tables directly. Select an extracted
dataset root containing `v1.0-mini`, `v1.0-trainval` or `v1.0-test`, or the
metadata version directory itself. Sensor files can remain under the dataset
root or version directory. `sensor="lidar"` indexes `LIDAR_TOP` keyframes;
`sensor="camera"` indexes camera keyframes and works with camera-only extracts.
Images and point clouds are loaded when their frame is requested.
Both payload-reading methods remain available; `sensor` chooses the scene/frame
enumeration. Camera mode retains keyframe metadata, loading full sweep indexes
only if LiDAR timing or RGB preparation is explicitly requested.

```python
from dataset_utils.nuscenes import NuScenesReader

lidar = NuScenesReader(root="/datasets/nuscenes", version="v1.0-mini", sensor="lidar")
scene = lidar.list_scenes_for_split("val")[0]
info = lidar.list_frames(scene.token)[0]
frame = lidar.load_frame(info.token)
colored = lidar.prepare_frame(frame, camera_rgb=True)

cameras = NuScenesReader(root="/datasets/nuscenes", version="v1.0-mini", sensor="camera")
camera_frame = cameras.load_camera_frame(cameras.list_frames(scene.token)[0].token)
```

The reader exposes scene/frame lists, annotation class names and official
offline split membership. On mini, `train` and `val` select `mini_train` and
`mini_val`; release/split mismatches report an error. Annotation-free releases
remain useful. Ten detection class aliases are retained alongside additional
original categories.

Shared `SceneInfo`, `FrameInfo`, `BoundingBox3D` and `PointCloudFrame` models live
in `dataset_utils.models`. Point-cloud boxes use reference sensor coordinates,
local axes are length/width/height, and `reference_sensor_to_world` preserves the
LiDAR reference pose. `load_camera_frame` returns world-coordinate boxes and
camera RGB images, intrinsics and each exposure's sensor-to-world pose.
Frontend adapters add their layout and presentation models to these values.

The nuScenes backend also owns the nominal HDL-32E acquisition-time estimate,
pose interpolation, undoing existing ego compensation and camera RGB sampling.
This follows the nominal approach described in the
[NCore nuScenes guide](https://nvidia.github.io/ncore/conversions/nuscenes/nuscenes.html),
with no empirical fitting. Camera selection uses point acquisition estimates
and exposure poses; projection uses undistorted pinhole intrinsics, a sparse
LiDAR occlusion check and bilinear image sampling. Decoded images use a bounded
cache. RGB and timing are optional `prepare_frame` requests, so ordinary LiDAR
viewing does not decode camera images or estimate point times.

The timing remains an estimate. Static undo/reapply transforms cancel; moving
objects are not motion compensated, and sparse occlusion checks cannot establish
complete camera visibility. See [3dviz's detailed timing conventions](../3dviz/README.md#camera-rgb-and-point-timing).

## Native Waymo input

Select either the dataset parent, `waymo_og_format_unpacked`, or a directory
containing sequence TFRecords. Standard split directories are discovered by
filename, without reading their sensor payloads:

```text
waymo_1.4.2/
  waymo_og_format_unpacked/
    training/
      segment-<context>_with_camera_labels.tfrecord
    validation/
      segment-<context>_with_camera_labels.tfrecord
    testing/
      segment-<context>_with_camera_labels.tfrecord
```

`train`, `val`, and `test` alias `training`, `validation`, and `testing`.
Omitting the split selects all discovered sequences. A processed workspace can
also locate a sibling `waymo_og_format_unpacked` directory. Duplicate context
names are rejected.

## Waymo Python API

```python
from dataset_utils.waymo import (
    NativeWaymoReader, decode_lidar, colorize_points, WaymoCameraProjection,
)

reader = NativeWaymoReader('/datasets/waymo_1.4.2', split='val')
sequence = reader.list_sequences()[0]
headers = reader.list_frame_headers(sequence.context_name)
header = headers[0]
frame = reader.read_frame(
    sequence.context_name, header.frame_index, timestamp_us=header.timestamp_us,
)
points = decode_lidar(frame)  # Equivalent to frame.points().
rgb, valid = colorize_points(frame, points)
```

Sequence entries expose `context_name`, `path`, and `split`. Frame headers expose
`frame_index` and `timestamp_us`; `list_frames(context)` aliases
`list_frame_headers(context)`, and `frame_timestamps(context)` returns only the
timestamps. Indices refer to original records within the selected sequence.

The reader indexes record offsets by checking length headers and seeking past
payloads. Header enumeration reads only the small protobuf context/timestamp
fields, seeking past JPEG and range-image bytes. Selecting a frame validates
its complete TFRecord payload CRC, parses the needed protobuf fields, and
decodes its camera images. The cache holds at most two decoded frames. Every
selected frame must match the requested context and optional exact timestamp.

`NativeFrame` provides:

- `context_name`, `timestamp_us`, and reference `vehicle_to_world` transform.
- `cameras`: original RGB images, intrinsic vector, camera-to-vehicle extrinsic,
  camera-specific vehicle pose, pose timestamp, linear/angular velocity,
  exposure/readout timing, and rolling shutter direction.
- `boxes`: native `Vehicle`, `Pedestrian`, `Sign`, and `Cyclist` labels with
  center, length/width/height, heading radians, and identifier. Camera-synchronized
  boxes and their most-visible camera are retained when populated.
- Compressed laser returns and calibration for lazy point decoding, plus native
  camera labels and projected LiDAR label rectangles when present.

Frame timestamps denote the beginning of the TOP scan. The reference frame
vehicle pose is approximately near the middle of the scan and must not be
treated as a pose sampled at that start timestamp. Each camera instead supplies
its own vehicle pose and corresponding `pose_timestamp_s`.

`WaymoCameraProjection(camera).project_world(points)` supports the native
distortion and rolling shutter model for projecting world-coordinate geometry
onto an original image. The camera's native axes are forward/left/up;
`OPTICAL_TO_WAYMO` converts conventional optical right/down/forward axes.

## Point order and camera RGB

`NativePoints.points` are reference-vehicle XYZ. Rows follow the official
OpenPCDet conversion order: sorted laser ID, return 0 followed by return 1, then
row-major range-image pixels whose range is positive. Both returns retain their
own intensity, elongation, no-label-zone value, laser ID, return ID, and native
camera projection array `[camera, u, v, camera, u, v]`. No NLZ filtering or
point reordering is applied.

Range decoding uses beam inclinations, the sensor azimuth correction and
extrinsic, then the TOP per-pixel vehicle poses and inverse reference vehicle
pose. `raw_points` contain acquisition-vehicle coordinates only where
`raw_valid_mask` is true: TOP points with provided per-pixel poses. Other rows
retain reference coordinates and have a false mask.

Camera RGB samples the original images through these native per-return camera
associations. It prefers the first valid association and uses the second when
the first camera or pixel is unavailable. It never substitutes first-return
projections for second-return points. Unmatched points receive gray RGB and a
false validity mask. Colors are floating-point RGB in `[0, 1]`.

These dataset correspondences avoid fitting another acquisition-time or
pinhole approximation. They do not establish complete image visibility or
correct moving-object motion. No additional occlusion model is applied.
Consumers attaching these colors to an independently processed cloud must
first verify matching frame identity, point count, and XYZ row order within a
strict tolerance. Matching only point count is insufficient.

## Package boundaries

```text
dataset_utils/src/dataset_utils/
  models.py        # shared scene/frame/box/point-cloud values
  errors.py        # dataset errors independent of a frontend
  nuscenes/        # tables, calibration, images/LiDAR, timing, RGB and splits
  waymo/           # native records, range images, camera model and associations
```

Storage and sensor geometry belong here. Frontend `dataloader` packages remain
small adapter/factory boundaries; they configure camera labels/layouts and map
shared values into the frontend's presentation contract. Point-cloud distance
colors, bounding-box overlays, rendering, UI controls and navigation belong to
their viewers. Prediction import, confidence filtering and NMS remain in
`3ddetviz`. Existing frontend imports can re-export shared models and numerical
helpers without maintaining another implementation.

Adding another frontend should reuse these readers without adding GUI imports
to `dataset_utils`. Supporting another dataset can add a backend module and
register a frontend adapter without changing the existing readers.

## Validation and references

```bash
python -m pytest dataset_utils/tests
```

Synthetic tests cover TFRecord CRCs, truncated records, record/matrix size
limits, protobuf matrix parsing, signed integers, missing returns, return-specific
camera associations, full pose transforms, identity mismatches, lazy headers,
and the two-frame cache. TFRecord length CRCs are checked during indexing;
complete payload CRCs are checked when records are selected. A pure-Python
CRC32C fallback exists; the declared `google-crc32c` wheel provides the fast path.
nuScenes tests cover native table indexing, full quaternion/pose transforms,
camera-only and annotation-free extracts, official splits, nominal point timing
and camera projection. Viewer tests separately cover their adapters and GUI.

Real validation on the first validation context
`10203656353524179475_7625_000_7645_000` decoded 126,504 points in the same order
as its processed OpenPCDet cloud, with maximum XYZ difference about 1.02 mm.
Native associations colored 72,480 points in that frame.

The descriptor subset follows the Apache-2.0 Waymo Open Dataset
[dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/dataset.proto)
and [label.proto](https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/label.proto),
pinned at v1.6.1 and validated against this v1.4.2 data. Unknown map, keypoint,
segmentation, and other unused fields are discarded. Range equations follow
[Waymo's range image utilities](https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/utils/range_image_utils.py).
The camera model follows the
[reference camera implementation](https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/wdl_limited/camera/camera_model.cc).
The local NVIDIA NCore implementation in
`tools/data_converter/waymo/waymo_range_image_utils.py` and its matrix parser
provided additional guidance on TOP pose handling and efficient matrix decoding.

Upstream license text and attribution are retained in `third_party/` and in
built distributions, including official nuScenes scene memberships and the
Waymo camera model's separate BSD terms.
