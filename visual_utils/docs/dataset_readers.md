# 📚 Native dataset readers

Use the shared `DatasetLoader` when connecting data to a viewer. For research
that needs access to dataset-specific sensor records, the lower-level nuScenes
and Waymo readers are also public. This guide describes their storage layouts,
geometry, and processing assumptions.

**[Package overview](../README.md)** ·
**[nuScenes](#nuscenes)** ·
**[Waymo](#waymo)** ·
**[Dataset interface](loader_contract.md)** ·
**[Custom dataset tutorial](custom_datasets.md)**

<a id="nuscenes"></a>
## 🚗 Read nuScenes

`NuScenesReader` reads native JSON tables directly, without the nuScenes SDK.
Point it at an extracted dataset root containing `v1.0-mini`, `v1.0-trainval`,
or `v1.0-test`, or at the metadata version directory itself. Sensor files may
remain under the dataset root or version directory.

Select `sensor="lidar"` to enumerate `LIDAR_TOP` keyframes, or `sensor="camera"`
to enumerate camera keyframes, including camera-only extracts. Both payload
methods remain available regardless of the enumeration mode. Camera mode keeps
keyframe metadata and loads full sweep indexes only when LiDAR timing or RGB
preparation is requested.

```python
from visual_utils.nuscenes import NuScenesReader

lidar = NuScenesReader(root="/datasets/nuscenes", version="v1.0-mini", sensor="lidar")
scene = lidar.list_scenes_for_split("val")[0]
info = lidar.list_frames(scene.scene_id)[0]
cloud = lidar.load_frame(info.frame_id)
colored = lidar.prepare_frame(cloud, camera_rgb=True)

cameras = NuScenesReader(root="/datasets/nuscenes", version="v1.0-mini", sensor="camera")
camera_info = cameras.list_frames(scene.scene_id)[0]
camera_set = cameras.load_camera_frame(camera_info.frame_id)
```

Scene and frame catalogs expose official offline split membership. On mini,
`train` and `val` select `mini_train` and `mini_val`. Release/split mismatches
raise an error. Annotation-free releases are also supported. Native category
IDs remain intact, and loader metadata supplies detection display groups
alongside the additional original categories.

The reader returns the shared records directly:

| Payload | Reference and contents |
| --- | --- |
| `PointCloudFrame` | Points and annotations in the reference sensor coordinates, LWH box dimensions, and `T_world_from_reference` for the LiDAR pose. |
| `CameraFrameSet` | World-reference annotations, RGB images, and explicit camera poses and projection models. |
| Optional point fields | `rgb`, `timing`, and `acquisition`, with masks and provenance. |

### 🕒 Interpret timing and RGB

The backend owns the nominal HDL-32E acquisition-time estimate, pose
interpolation, undoing existing ego compensation, and camera RGB sampling.
It follows the nominal approach in the
[NCore nuScenes guide](https://nvidia.github.io/ncore/conversions/nuscenes/nuscenes.html)
without empirical fitting.

Camera selection uses estimated point acquisition times and exposure poses.
Projection combines undistorted pinhole intrinsics, a sparse LiDAR occlusion
check, and bilinear image sampling. Decoded images use a bounded cache. RGB and
timing are optional preparation requests, so ordinary LiDAR viewing does not
decode camera images or estimate point times.

Timing remains an estimate. Static undo/reapply transforms cancel, moving
objects are not motion compensated, and sparse occlusion checks cannot
establish complete camera visibility. See
[3dviz's scientific interpretation](../../viz_tools/3dviz/README.md#controls-and-scientific-interpretation)
before drawing conclusions from the display.

<a id="waymo"></a>
## 🚙 Read native Waymo

Select the dataset parent, `waymo_og_format_unpacked`, or a directory containing
sequence TFRecords. Standard split directories are discovered by filename,
without reading sensor payloads:

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

```python
from visual_utils.waymo import NativeWaymoReader, colorize_points, decode_lidar

reader = NativeWaymoReader("/datasets/waymo_1.4.2", split="val")
sequence = reader.list_sequences()[0]
header = reader.list_frame_headers(sequence.context_name)[0]
frame = reader.read_frame(
    sequence.context_name,
    header.frame_index,
    timestamp_us=header.timestamp_us,
)
points = decode_lidar(frame)  # Equivalent to frame.points().
rgb, valid = colorize_points(frame, points)
```

Sequence entries expose `context_name`, `path`, and `split`. Frame headers
expose `frame_index` and `timestamp_us`. Use `list_frames(context)` as an alias
for `list_frame_headers(context)`, or `frame_timestamps(context)` when you only
need timestamps. Indices identify original records within the selected sequence.

The reader indexes offsets by validating length headers and seeking past
payloads. Header enumeration reads the small context and timestamp fields,
skipping JPEG and range-image bytes. Selecting a frame validates its full
TFRecord payload CRC, parses the required protobuf fields, and decodes camera
images. A bounded cache retains at most two decoded frames. Selected frames
must match the requested context and any supplied exact timestamp.

### 📦 Inspect a native frame

`NativeFrame` retains the sensor information needed for dataset-specific work:

| Field | Contents |
| --- | --- |
| Frame identity and pose | `context_name`, `timestamp_us`, and the reference `vehicle_to_world` transform. |
| `cameras` | Original RGB images, intrinsic vector, camera-to-vehicle extrinsic, camera-specific vehicle pose and timestamp, linear/angular velocity, exposure/readout timing, and rolling shutter direction. |
| `boxes` | Native `Vehicle`, `Pedestrian`, `Sign`, and `Cyclist` labels with center, LWH dimensions, heading in radians, and identifier. Camera-synchronized boxes and their most-visible camera are retained when present. |
| Laser returns and labels | Compressed returns and calibration for lazy point decoding, plus camera labels and projected LiDAR label rectangles when present. |

Frame timestamps denote the beginning of the TOP scan. The reference vehicle
pose is approximately near the middle of the scan, so it must not be treated as
a pose sampled at that start timestamp. Each camera supplies its own vehicle
pose and corresponding `pose_timestamp_s`.

`WaymoCameraProjection(camera).project_world(points)` projects world-coordinate
geometry into an original image using native distortion and rolling shutter.
Import it from `visual_utils.waymo`. Native camera axes are forward/left/up,
and `OPTICAL_TO_WAYMO` converts conventional optical right/down/forward axes.

### 🌈 Preserve point order and camera RGB

`NativePoints.points` contains XYZ in the reference vehicle frame. Rows follow
the official OpenPCDet conversion order: sorted laser ID, return 0 followed by
return 1, then row-major range-image pixels with positive range. Both returns
retain their own intensity, elongation, no-label-zone value, laser ID, return
ID, and camera projection array `[camera, u, v, camera, u, v]`. No NLZ filtering
or point reordering is applied.

Range decoding uses beam inclinations, sensor azimuth correction, and the
sensor extrinsic, followed by TOP per-pixel vehicle poses and the inverse
reference vehicle pose. `raw_points` gives acquisition-vehicle coordinates
where `raw_valid_mask` is true, which covers TOP points with provided per-pixel
poses. Other rows retain reference coordinates with a false mask.

Camera RGB uses each return's native camera associations to sample the original
images. It prefers the first valid association and uses the second when the
first camera or pixel is unavailable. Second-return points keep their own
projections. Unmatched points receive gray RGB with a false validity mask.
Colors are floating-point RGB in `[0, 1]`.

These correspondences avoid fitting another acquisition-time or pinhole
approximation. They do not establish complete image visibility or correct
moving-object motion, and no additional occlusion model is applied.

Before attaching these colors to an independently processed cloud, verify
frame identity, point count, and XYZ row order within a strict tolerance.
Matching only the point count is insufficient.

## 🧪 Understand the validation

Run the shared tests from the repository root:

```bash
python -m pip install -e './visual_utils[dev]'
python -m pytest visual_utils/tests
```

Waymo synthetic tests cover TFRecord CRCs, truncated records, record/matrix size
limits, protobuf matrix parsing, signed integers, missing returns, return-specific
camera associations, full pose transforms, identity mismatches, lazy headers,
and the two-frame cache. Length CRCs are checked during indexing and full payload
CRCs when a record is selected. The declared `google-crc32c` dependency provides
a fast implementation, with a pure-Python fallback also available.

nuScenes tests cover native table indexing, full quaternion and pose transforms,
camera-only and annotation-free extracts, official splits, nominal point timing,
and camera projection. Viewer tests cover their own integration and UI behavior.

A recorded real-data check on validation context
`10203656353524179475_7625_000_7645_000` decoded 126,504 points in the same order
as its processed OpenPCDet cloud, with a maximum XYZ difference of about
1.02 mm. Native associations colored 72,480 points in that frame. This is a
specific correspondence check, rather than a claim of coverage across all data.

## 🔗 Sources and attribution

The Waymo descriptor subset follows the Apache-2.0
[dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/dataset.proto)
and [label.proto](https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/label.proto),
pinned at v1.6.1 and validated against this v1.4.2 data. Unknown map, keypoint,
segmentation, and other unused fields are discarded.

Range equations follow
[Waymo's range image utilities](https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/utils/range_image_utils.py).
The camera model follows the
[reference camera implementation](https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/wdl_limited/camera/camera_model.cc).
The NVIDIA NCore implementation in
`tools/data_converter/waymo/waymo_range_image_utils.py`
and its matrix parser also informed TOP pose handling and efficient matrix decoding.

[Upstream notices](../third_party/NOTICE.txt) and license text are included in
`third_party/` and in built distributions. They include official nuScenes scene
memberships and the Waymo camera model's separate BSD terms.
