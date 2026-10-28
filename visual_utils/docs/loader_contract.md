# Dataset interface

One dataset adapter converts native storage into shared Python records. The
camera, point-cloud and detection viewers consume those records directly.
There are no viewer-specific frame models or runtime frame fingerprints.

The interface lives in [`dataset.py`](../src/visual_utils/dataset.py). Records
live in [`models.py`](../src/visual_utils/models.py). Import public types from
`visual_utils`.

## Implementing an adapter

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
def _prepare_point_cloud(self, frame, *, request: PointCloudPreparation): ...
def _close(self) -> None: ...
```

Callers use the public methods without underscores. Those methods perform
small lifecycle, split and capability checks before calling the corresponding
implementation. This is ordinary method delegation; no subclass methods are
rewritten or decorated automatically.

Catalog methods read metadata without decoding clouds or images. Payload
methods load the requested observation. Methods are synchronous; callers
serialize reads and wait for pending work before closing.

## Records needed by the viewers

| Record | Purpose |
| --- | --- |
| `DatasetMetadata` | Dataset identity, classes, ordered sensors, supported splits and capabilities. |
| `SceneInfo` | Stable scene ID, display name and optional per-modality frame counts. |
| `FrameInfo` | Stable frame ID, scene ID, zero-based display index, optional timestamp and modalities. |
| `PointCloudFrame` | XYZ points, attributes, annotations, spatial reference and optional prepared data. |
| `CameraFrameSet` | Ordered RGB observations, annotations and their shared spatial reference. |
| `Box3D` | Annotation/class IDs, center, LWH size and full rotation. |

A point-only adapter needs only the point-cloud payload. A camera-only adapter
needs only the camera payload. Metadata describes the configured sources;
unsupported operations raise `UnsupportedCapabilityError`.

Class IDs retain native identity. `display_name` and optional `display_group`
control labels and grouped checkboxes without rewriting boxes. Camera sensor
order and optional `view_role` supply layout hints; they contain no GUI slots.

Frame IDs are opaque identifiers, independent of display indices. Sparse native
indices keep their original identity. Modality catalogs can have different
lengths and indices; a loaded frame's index belongs to that modality's catalog.
Unknown timestamps and counts remain `None`, rather than invented zeros.

## Geometry and time

- Coordinates use metres; box dimensions are `(length, width, height)`.
- `rotation` maps box-local coordinates into the enclosing reference.
- `T_target_from_source` maps source to target using homogeneous column vectors.
- Reference IDs identify anchored coordinates, including the observation/pose.
  Equal axis labels at different times do not establish the same reference.
- Known timestamps are integer microseconds in the documented clock domain.
  Nominal frame time, camera exposure time and pose time remain distinct.
- World poses are optional. Supply `world_frame_id` and
  `T_world_from_reference` together; do not fabricate identity world poses.

The shared records allow documented native axes. Current 3D height controls,
horizontal distance and BEV NMS use Z-up and horizontal XY. Convert other native
axes consistently when targeting those tools.

Each camera's `projection_model.project_reference(points)` maps shared-reference
points into its original image and returns `ProjectionResult(uv, depth,
valid_mask)`. Optical axes are X-right/Y-down/Z-forward. Use `PinholeProjection`
for undistorted images; custom distortion/rolling-shutter models implement the
same small protocol. An image without calibration remains viewable.

Projection validity means a finite, positive-depth projection, not image bounds
or proven visibility. Retain signed depth for behind-camera points when defined
so the viewer can clip box edges correctly. Invalid UV may be NaN.

## Annotations and optional processing

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
shapes and geometry. Tests check each provider's preparation semantics; runtime
code does not hash arrays to detect a caller changing them.

Processed/native point pairing still verifies actual correspondence before
attaching colors or acquisition geometry. Equal point counts alone are
insufficient. This checks scientific alignment between sources, not ownership.

## Configuration, registration and errors

`DatasetConfig` contains `root`, optional `release`, `storage_format`, named
`source_paths`, and provider-specific `options`. Providers reject unsupported
settings. `load_dataset_config(path)` reads the same fields from JSON; relative
roots resolve against the JSON location and relative named sources against its
root when present. All three commands accept `--dataset-config FILE`.

Register a constructor with `register_loader("name", MyDataset)`. Installed
packages may expose a `visual_utils.loaders` entry point instead. The registry
is only a construction convenience; it is not involved in frame processing.
See the [custom dataset example](custom_datasets.md).

Use `with create_loader(...) as dataset` or call `close()`. Returned arrays are
read-only snapshots and remain usable after close. Override `_close()` for
provider resources; inherited `close()` is idempotent.

Errors distinguish unsupported capabilities, unknown IDs, known unavailable
data, source I/O and malformed geometry. Missing/corrupt selected files must not
be disguised as empty observations. Programming errors remain visible rather
than being caught by a generic wrapper around every provider method.
