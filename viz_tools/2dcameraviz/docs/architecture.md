# 🏗️ Camera viewer architecture

The camera viewer uses the scientific records exported by `visual_utils` from
loading through projection. A `DatasetLoader` owns dataset-specific storage,
calibration and timing, while presentation code receives its observations
unchanged.

**[Interface](#the-loader-interface)** · **[Records](#records-and-presentation)** · **[Projection](#projection-and-reference-frames)** · **[Lifecycle](#ownership-and-execution)** · **[Validation](#validation-and-next-reading)**

The execution path follows those responsibilities:

```text
DatasetLoader.load_cameras(frame_id)
    ↓ CameraFrameSet: images, annotations, explicit camera models
project_boxes(frame)
    ↓ sensor ID → ProjectedBox tuples in native image pixels
CameraPanel
    ↓ aspect-preserving images and styled overlays
Tk window
```

<a id="the-loader-interface"></a>
## 🔌 The loader interface

The viewer needs a small set of shared operations:

```python
metadata                                        # classes, sensors, splits, capabilities
list_scenes(split="all")                         # tuple[SceneInfo, ...]
list_frames(scene_id, modality=Modality.CAMERAS)   # tuple[FrameInfo, ...]
load_cameras(frame_id)                            # CameraFrameSet
close()
```

A provider implements `metadata`, `_list_scenes`, `_list_frames` and
`_load_cameras`. The base class handles public lifecycle, split and capability
checks before delegating to those hooks. Providers that keep resources open
also implement `_close`. The [shared interface](../../../visual_utils/docs/loader_contract.md)
and [complete example](../../../visual_utils/docs/custom_datasets.md) show the
provider side in detail.

`cameraviz.dataloader.create_dataset` opens that shared loader and checks camera
support. The synthetic demo implements the same interface, so it follows the
same projection and display path. The viewer uses this single registry and
record model throughout, with direct method calls to the loader.

<a id="records-and-presentation"></a>
## 🧩 Records and presentation

Shared records retain the sensor and annotation meaning established by the
provider. The UI reads the fields it needs for navigation and display:

| Shared record | Used by the viewer |
| --- | --- |
| `DatasetMetadata` | Split names, sensor order, view roles, class display groups and labels. |
| `SceneInfo` | Scene ID, display name, description and camera frame count. |
| `FrameInfo` | Stable frame ID, display index and optional logical timestamp. |
| `CameraFrameSet` | Camera observations and default annotations in a declared reference. |
| `CameraObservation` | Sensor ID, RGB image, explicit projection model and optional annotation override. |
| `Box3D` | Native class and annotation IDs, centre, LWH size and full rotation. |

Two records describe presentation output. `CameraSlot` identifies a grid cell,
and `ProjectedBox` contains native-pixel line segments and a 2D enclosure.
Native class IDs stay in both input boxes and projected output. The UI resolves
display groups when selecting checkbox state and colors.

Grid placement lives in `visualization/layout.py`. Recognized camera roles
reserve their positions, then duplicate, unknown or absent roles fill free
positions in sensor order. Missing observations retain their declared slot.
This lets a provider describe a rig through sensor metadata while the viewer
chooses its visual arrangement.

<a id="projection-and-reference-frames"></a>
## 📐 Projection and reference frames

`Box3D` geometry and `ProjectionModel.project_reference(points)` already share
`CameraFrameSet.reference_frame_id`, so projection uses those coordinates
directly. An optional world transform remains available to other consumers.
There is no world-coordinate round trip in the camera display path.

An explicit `PinholeProjection` uses exact near-plane clipping before perspective
division, followed by viewport clipping. Native nonlinear models instead use
sampled curved edges and refined near-plane intersections. Undefined projection
gaps stay separate. Both paths retain pitch, roll and yaw, and use pixel centres
from zero through width/height minus one.

For each camera, an annotation override replaces the default set, including an
explicitly available-empty set. Otherwise, projection uses the frame's default
annotations. Status labels retain the distinction between available,
not-provided and withheld ground truth. Images without a model remain viewable
without projected boxes because intrinsics alone do not establish a pinhole
assumption.

Known times remain signed microseconds in the provider's clock domain, while
unknown times remain `None`. Logical frame, exposure, pose and annotation times
are separate fields. Display indices select observations and never substitute
for timestamps. These distinctions preserve the meaning of the source data
while the UI changes layout or appearance.

<a id="ownership-and-execution"></a>
## 🔄 Ownership and execution

The window owns one loader. Its single background worker serializes catalog
reads, image loading and projection, while the main thread updates Tk widgets.
Scene and frame selection use stable IDs, and sliders use dense display indices.
A failed read leaves the previous displayed frame available.

Replacing a dataset closes the old loader after its reads finish. Window
shutdown queues cleanup behind pending work. Script callers use the same
ownership rule through a context manager and finish pending reads before
closing. Providers stay synchronous and own their payload caches. The window
supplies scheduling around those synchronous dataset operations.

<a id="validation-and-next-reading"></a>
## 🧪 Validation and next reading

Analytical pinhole checks establish geometry, full rigid transforms and clipping
expectations. Native Waymo fixtures cover nonlinear projection and
camera-specific annotations. Tk checks add grouped class styling, navigation,
source changes, missing cameras, unavailable annotations and cleanup.

Use the [README validation commands](../README.md#interpretation-and-validation)
to run those checks. For a new backend, continue with the
[shared dataset interface](../../../visual_utils/docs/loader_contract.md) and
[custom-dataset tutorial](../../../visual_utils/docs/custom_datasets.md).
