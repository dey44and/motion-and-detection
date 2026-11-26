# 📐 Dataset interface

Implement one dataset adapter to connect your storage to `3dviz` and Python
clients. The adapter returns shared point-cloud and camera records with
explicit scientific conventions.

**[Package overview](../README.md)** ·
**[Runnable tutorial](custom_datasets.md)** ·
**[Architecture](../../docs/VISUALIZATION_ARCHITECTURE.md)**

The interface lives in [`dataset.py`](../src/visual_utils/dataset.py). Records
live in [`models.py`](../src/visual_utils/models.py). Import public types from
`visual_utils`.

## 🔧 Implement an adapter

Subclass `DatasetLoader`, call `super().__init__(config)`, and implement:

```python
@property
def metadata(self) -> DatasetMetadata: ...

def _list_scenes(self, *, split: str) -> tuple[SceneInfo, ...]: ...

def _list_frames(self, scene_id: str, *, modality: Modality) -> tuple[FrameInfo, ...]: ...

# Implement the payloads your dataset supplies:
def _load_point_cloud(self, frame_id: str) -> PointCloudFrame: ...
def _load_cameras(self, frame_id: str) -> CameraFrameSet: ...

# Optional:
def _camera_rig(self, frame: PointCloudFrame) -> tuple[CameraPose, ...]: ...
def _prepare_point_cloud(self, frame, *, request: PointCloudPreparation): ...
def _lidar_scan_token(self, frame_id: str) -> str: ...
def _load_lidarseg_ground_truth(self, frame: PointCloudFrame) -> LidarSegmentation: ...
def _close(self) -> None: ...
```

Callers use the public methods without underscores. Those methods perform
small lifecycle, split, and capability checks before calling your implementation.
The base class uses ordinary method delegation, so the execution path is visible
in the source.

Catalog methods read metadata without decoding clouds or images. Payload
methods load the requested observation. Methods are synchronous, so callers
serialize reads and wait for pending work before closing. The
[custom dataset tutorial](custom_datasets.md) shows a complete implementation.

## 📦 Return shared records

| Record | Purpose |
| --- | --- |
| `DatasetMetadata` | Dataset identity, classes, ordered sensors, supported splits and capabilities. |
| `SceneInfo` | Stable scene ID, display name and optional per-modality frame counts. |
| `FrameInfo` | Stable frame ID, scene ID, zero-based display index, optional timestamp and modalities. |
| `PointCloudFrame` | XYZ points, attributes, annotations, spatial reference and optional prepared data. |
| `LidarSegmentation` | Exact LiDAR scan token, evaluation IDs, and original native IDs for each point. |
| `CameraFrameSet` | Ordered RGB observations, annotations and their shared spatial reference. |
| `CameraPose` | Camera mount position, unit forward direction, and optional calibrated corner rays in the point cloud's reference. |
| `Box3D` | Annotation/class IDs, center, LWH size and full rotation. |

A point-only adapter needs only the point-cloud payload. Python clients can
use camera-only adapters with only the camera payload. The `3dviz` workspace
requires a point cloud. Metadata describes configured sources, and unsupported
operations raise `UnsupportedCapabilityError`.

Class IDs retain native identity. `display_name` and optional `display_group`
control labels and grouped checkboxes without rewriting boxes. Camera sensor
order and optional `view_role` supply layout hints that the viewer resolves into
its own layout.

Frame IDs are opaque identifiers, independent of display indices. Sparse native
indices keep their original identity. Modality catalogs can have different
lengths and indices. A loaded frame's index belongs to that modality's catalog.
Unknown timestamps and counts remain `None`, rather than invented zeros.

## 📏 Keep geometry and time explicit

- Coordinates use metres, and box dimensions are `(length, width, height)`.
- `rotation` maps box-local coordinates into the enclosing reference.
- `T_target_from_source` maps source to target using homogeneous column vectors.
- Reference IDs identify anchored coordinates, including the observation/pose.
  Equal axis labels at different times do not establish the same reference.
- Known timestamps are integer microseconds in the documented clock domain.
  Nominal frame time, camera exposure time and pose time remain distinct.
- World poses are optional. Supply `world_frame_id` and
  `T_world_from_reference` together when known. Leave both absent when unknown.

The shared records allow documented native axes. Current 3D height controls,
horizontal distance and BEV NMS use Z-up and horizontal XY. Convert other native
axes consistently when targeting those tools.

Each camera's `projection_model.project_reference(points)` maps shared-reference
points into its original image and returns `ProjectionResult(uv, depth,
valid_mask)`. Optical axes are X-right/Y-down/Z-forward. Use `PinholeProjection`
for undistorted images. Custom distortion or rolling shutter models implement
the same small protocol. An image without calibration remains viewable.

Projection validity means a finite, positive-depth projection, not image bounds
or proven visibility. Retain signed depth for behind-camera points when defined
so the viewer can clip box edges correctly. Invalid UV may be NaN.

`camera_rig(frame)` gives the viewer camera mount markers from calibration
metadata. The frame comes from the same loader. Each marker uses the camera
`sensor_id`, its position in metres, and its optical +Z direction expressed as
a unit vector in the cloud reference. Use the mounted rig at the cloud's
reference pose. Separately, `load_cameras` uses the image's exposure pose for
projection. Rig lookup leaves images unopened and returns only available,
calibrated markers. Providers without rig data return an empty tuple.

When the image extent and lens calibration are available, `corner_rays` holds
four vectors in top-left, top-right, bottom-right, bottom-left order. They
correspond to image bounds `(0, 0)`, `(width, 0)`, `(width, height)`, and
`(0, height)`. Each vector has dot product 1 with `forward`. The expression
`position + depth * corner_rays` places the quad on a plane at that optical
depth in metres. `3dviz` uses these rays for the clickable camera preview.

## 🎨 Describe annotations and optional processing

`AnnotationSet` distinguishes `AVAILABLE` (including zero boxes), `NOT_PROVIDED`
and `WITHHELD`. Points and default boxes share a reference. Camera-specific
annotations can override the set, including an available-empty override.
Annotation timing can identify reference-time, camera-synchronized, estimated,
unknown or mixed values. Preserve per-box timing when a set is mixed.

Point preparation requests three independent features: RGB, scalar point times,
and acquisition geometry. Results include row-aligned values, validity masks
and processing provenance. Estimated timing stays marked estimated. Valid black
RGB is distinct from a missing color. Partial acquisition geometry must not be
presented as a fully uncompensated scan.

Preparation receives the unchanged frame from that loader and returns a new
record, typically with `dataclasses.replace`. Preserve baseline rows, attributes,
annotations and existing derived fields. Record constructors validate numerical
shapes and geometry. Provider tests verify row preservation and preparation
semantics, while read-only arrays communicate the snapshot's intended use.

Processed/native point pairing still verifies actual correspondence before
attaching colors or acquisition geometry. Equal point counts alone are
insufficient because colors and geometry must describe the same physical points.

`LIDAR_SEGMENTATION` indicates a provider can identify the exact native scan
with `lidar_scan_token(frame_id)`. The scan token can differ from the catalog
frame ID. `load_lidarseg_ground_truth(frame)` receives the unchanged cloud
from that provider and returns one native and one evaluation ID per original
point row. Raise `DataUnavailableError` for withheld or missing labels and
`DatasetValidationError` for a wrong scan, wrong row count, or invalid IDs.
The nuScenes provider uses its 16 challenge classes plus ignored ID `0` for
evaluation and retains the full 32-class native IDs. Prediction runs attach
through the viewer and are matched to the scan token at review time.

<a id="configuration-registration-and-errors"></a>
## ⚙️ Configure and register a loader

`DatasetConfig` contains `root`, optional `release`, `storage_format`, named
`source_paths`, and provider-specific `options`. Providers reject unsupported
settings. `load_dataset_config(path)` reads the same fields from JSON. Relative
roots resolve against the JSON location, and relative named sources resolve
against that root when present. Without a root, named sources resolve against
the JSON directory. `3dviz` accepts `--dataset-config FILE`.

Register a constructor with `register_loader("name", MyDataset)`. Installed
packages may expose a `visual_utils.loaders` entry point instead. The registry
constructs the loader once. Frame processing then uses the loader directly.
See the [custom dataset example](custom_datasets.md).

## 🔄 Manage lifetime and errors

Use `with create_loader(...) as dataset` or call `close()`. Returned arrays are
read-only snapshots and remain usable after close. Override `_close()` for
provider resources. Inherited `close()` is idempotent.

Errors distinguish unsupported capabilities, unknown IDs, known unavailable
data, source I/O and malformed geometry. Missing/corrupt selected files must not
be disguised as empty observations. Programming errors remain visible rather
than being caught by a generic wrapper around every provider method.
