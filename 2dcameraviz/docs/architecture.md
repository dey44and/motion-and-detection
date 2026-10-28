# Camera viewer architecture

The camera viewer has one scientific data model: the records exported by
`dataset_utils`. Dataset-specific file formats, calibration and timing belong
in a `DatasetLoader`. Presentation code receives its observations unchanged.

```text
DatasetLoader.load_cameras(frame_id)
    ↓ CameraFrameSet: images, annotations, explicit camera models
project_boxes(frame)
    ↓ sensor ID → ProjectedBox tuples in native image pixels
CameraPanel
    ↓ aspect-preserving images and styled overlays
Tk window
```

## The small interface

The viewer needs these shared operations:

```python
metadata                                      # classes, ordered sensors, splits, capabilities
list_scenes(split="all")                      # tuple[SceneInfo, ...]
list_frames(scene_id, modality=Modality.CAMERAS)  # tuple[FrameInfo, ...]
load_cameras(frame_id)                         # CameraFrameSet
close()
```

A provider implements `metadata`, `_list_scenes`, `_list_frames` and
`_load_cameras`. The base class owns public lifecycle, split and capability
handling. Implement `_close` if the provider keeps resources open. See the
[shared interface](../../dataset_utils/docs/loader_contract.md) and
[complete example](../../dataset_utils/docs/custom_datasets.md).

`cameraviz.dataloader.create_dataset` opens the shared loader and checks camera
support. The synthetic demo implements the same interface. The viewer has no
second dataset registry, DTO hierarchy, per-dataset adapter or implicit reader
attribute forwarding.

## Data and presentation

| Shared record | Used by the viewer |
| --- | --- |
| `DatasetMetadata` | Split names; sensor order and view roles; class display groups and labels. |
| `SceneInfo` | Scene ID, display name, description and camera frame count. |
| `FrameInfo` | Stable frame ID, display index and optional logical timestamp. |
| `CameraFrameSet` | Camera observations and default annotation set in a declared reference. |
| `CameraObservation` | Sensor ID, RGB image, explicit projection model and optional annotation override. |
| `Box3D` | Native class and annotation IDs, centre, LWH size and full rotation. |

Only two records belong to presentation: `CameraSlot` describes a grid cell;
`ProjectedBox` contains native-pixel line segments and a 2D enclosure. Native
class IDs stay in both input boxes and projected output. The UI maps them to
display groups for checkbox state and colors.

Camera rig placement is in `visualization/layout.py`. Recognized roles reserve
their grid positions; duplicate, unknown or absent roles fill free positions in
sensor order. Missing observations retain their declared slot. Layout has no
coordinate-frame or calibration responsibility.

## Projection and reference frames

`Box3D` and `ProjectionModel.project_reference(points)` already share the
`CameraFrameSet.reference_frame_id`. Projection uses those coordinates directly.
An optional world transform is metadata for other consumers; the viewer does
not convert boxes to world and invert that conversion for projection.

An explicit `PinholeProjection` uses exact near-plane clipping before perspective
division, followed by viewport clipping. Native nonlinear models use sampled
curved edges and refined near-plane intersections. Undefined projection gaps
are not joined. The geometry retains pitch, roll and yaw. Pixel centres run
from zero through width/height minus one.

A camera-specific annotation set replaces the default set for that camera,
including an explicitly available-empty set. Otherwise, the camera uses the
frame's default annotations. Status labels distinguish available, not-provided
and withheld annotations. An image without a model remains viewable without
projected boxes; intrinsics alone do not authorize a pinhole assumption.

Known times remain signed microseconds in the provider's clock domain; unknown
times remain `None`. Logical frame time, exposure time, pose time and annotation
time are separate fields. Display indices are never used as timestamps.

## Ownership and execution

The window owns one loader. Its single background worker serializes catalog
reads, image loading and projection; Tk widgets are updated on the main thread.
Scene/frame selection operates on stable IDs while sliders use dense display
indices. A failed read leaves the previous displayed frame available.

Dataset replacement closes the old loader after reads finish. Window shutdown
queues cleanup behind pending work. Script callers should use a context manager
and finish pending reads before closing. Providers remain synchronous and own
their payload caches; the viewer adds no hidden synchronization or hashing.

Tests cover analytical pinhole geometry, full rigid transforms, nonlinear
visibility/clipping, native Waymo models and camera-specific annotations. Tk
checks additionally cover grouped class styling, navigation, source changes,
missing cameras, unavailable annotations and cleanup. See the
[README](../README.md#interpretation-and-validation) for commands.
