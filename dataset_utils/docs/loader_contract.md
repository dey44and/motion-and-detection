# Dataset loader contract, version 1

Status: implemented by dataset-utils 0.2.0. The normalized public API is exported
from `dataset_utils` and `dataset_utils.contracts`. Built-in `NuScenesLoader`
and `WaymoLoader` implement this contract. Existing native readers and legacy
viewer imports remain compatibility interfaces.

## 1. Boundary and responsibilities

A dataset extension implements one `DatasetLoader` and registers its constructor
once in the shared `dataset_utils` registry. Both viewers use generic adapters
over that interface. Registration executes in each application process; it is
not persistent state shared between independent processes.

The loader owns storage access, native identifiers, sensor metadata, calibration,
coordinate conversion, annotations, and dataset-specific timing/RGB preparation.
Native readers such as `NativeWaymoReader` can remain implementation helpers.
Waymo native and OpenPCDet representations sit behind the same public contract.

The viewers own widgets, camera-grid placement, colors, line widths, navigation,
rendering, and background-worker scheduling. Loader records contain no widgets,
renderer objects, or grid row/column coordinates.

Version 1 supports point-cloud observations, camera observations, and optional
3D ground truth. Supporting another dataset using these observations requires no
viewer source changes. A genuinely new payload type can require an additional
contract capability and renderer; this contract does not promise to render
every sensor modality automatically.

## 2. Abstract interface

The public base class is `dataset_utils.DatasetLoader`, implemented using
`abc.ABC`. Only the common metadata/catalog operations are abstract:

```python
class DatasetLoader(ABC):
    def __init__(self, config: DatasetConfig): ...

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata: ...

    @abstractmethod
    def list_scenes(self, *, split: str = "all") -> tuple[SceneInfo, ...]: ...

    @abstractmethod
    def list_frames(
        self, scene_id: str, *, modality: Modality
    ) -> tuple[FrameInfo, ...]: ...

    def load_point_cloud(self, frame_id: str) -> PointCloudFrame: ...

    def load_cameras(self, frame_id: str) -> CameraFrameSet: ...

    def prepare_point_cloud(
        self, frame: PointCloudFrame, *, request: PointCloudPreparation
    ) -> PointCloudFrame: ...

    def close(self) -> None: ...
```

`Modality` has `POINT_CLOUD` and `CAMERAS` values in version 1. At least one
must be supported. An unimplemented payload operation raises
`UnsupportedCapabilityError`; it never returns `None` or an empty placeholder
to disguise an unsupported modality.
Subclasses call the base initializer to retain configuration and initialize
lifecycle state before opening provider resources.

The base preparation implementation returns the input unchanged for an empty
request or when all requested derived fields are already present. Otherwise it
raises `UnsupportedCapabilityError`. A supported requested field must be
returned or its failure must be reported explicitly.

The base `close()` marks the loader closed, performs no provider-specific
resource cleanup, and is idempotent. Providers add cleanup while preserving
that state transition. Close also runs when leaving the loader's context
manager. Catalog, payload, and preparation operations after close raise
`DatasetClosedError`; the immutable metadata snapshot can still be inspected.
Returned records remain usable after cache eviction, later loads, and close.

## 3. Configuration and construction

The factory API is:

```python
def register_loader(
    name: str, constructor: Callable[[DatasetConfig], DatasetLoader]
) -> None: ...

def available_loaders() -> tuple[str, ...]: ...

def create_loader(name: str, config: DatasetConfig) -> DatasetLoader: ...
```

Names are nonempty, case-normalized registry keys. Duplicate registrations are
rejected. Unknown names and constructors returning another type are rejected.
Every constructed loader declares `metadata.contract_version == 1`; unsupported
contract versions are rejected before use.

`DatasetConfig` is immutable and has these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `root` | `Path \| None` | Main dataset root; optional for providers that need no filesystem root. |
| `release` | `str \| None` | Dataset-defined release, with no global nuScenes default. |
| `storage_format` | `str` | Provider-defined format, default `"auto"`. |
| `source_paths` | `Mapping[str, Path]` | Explicit named sources, such as `processed` and `native`. |
| `options` | `Mapping[str, object]` | Provider-declared scalar/sequence/mapping configuration values. |

Mappings and nested option containers are copied into recursively immutable
snapshots; mutable caller objects are never retained as configuration. Option
values are JSON-like scalars and string-keyed mappings/sequences, excluding
arbitrary runtime objects and non-finite numbers. The factory expands user-home
paths and resolves the main root against the working directory at construction.
Relative source paths resolve against `root`, or against that same working
directory when no root exists. Version 1 specifies local filesystem paths.

The provider validates its release, format, source names, and option names and
types. Unknown supplied configuration is an error, not an ignored setting.
Explicit paths/formats take priority over discovery; a malformed explicitly
selected source must not silently switch to a different representation.

For example, a Waymo configuration can explicitly supply `processed` and
`native` sources. `waymo_pcdet_workspace` and `waymo_og_format_unpacked` remain
discovery conveniences rather than required public API directory names.
Split selection is a catalog operation and is separate from release selection.

## 4. Metadata and capabilities

`DatasetMetadata` is immutable for an open loader and contains:

| Field | Meaning |
| --- | --- |
| `contract_version` | Integer `1`. |
| `dataset_id` | Stable identity of the logical dataset/release, independent of its machine path. |
| `display_name`, `release` | Human-readable dataset name and optional release. |
| `capabilities` | Frozen set of the capabilities below for the configured sources. |
| `splits` | Stable tuple of supported split names, including `"all"`. |
| `classes` | Stable tuple of `ClassInfo` records. |
| `sensors` | Stable tuple of `SensorInfo` records. |
| `clock_domain` | Documented timestamp basis, such as Unix or scene-relative; unknown is allowed. |
| `feature_coverage` | Optional capability-to-`full`/`partial`/`varies` hints; per-point masks remain authoritative. |

Capabilities are independent:

| Capability | Promise |
| --- | --- |
| `POINT_CLOUD` | Can load point-cloud observations. |
| `CAMERAS` | Can load one or more camera observations. |
| `BOXES_3D` | Can supply 3D ground truth where it is provided. |
| `RGB_COLORING` | Can supply per-point RGB, using native colors, native associations, or projection. |
| `POINT_TIMESTAMPS` | Can supply recorded or estimated scalar point timestamps. |
| `ACQUISITION_GEOMETRY` | Can supply acquisition-time coordinates, possibly for a masked subset. |

Capabilities describe an implementation operating on its configured sources.
They do not guarantee every frame, sensor, or point has that information.
RGB does not require scalar point timestamps, and acquisition geometry does not
imply scalar point timestamps. Advertising a capability requires implementing
its operation or providing its data in the loaded record.
`RGB_COLORING`, `POINT_TIMESTAMPS`, and `ACQUISITION_GEOMETRY` each require
`POINT_CLOUD`; no camera-image capability is required for native point colors.

`ClassInfo` has `class_id`, `native_name`, `display_name`, and optional
`display_group`. IDs preserve native category identity and are stable within
`dataset_id`; all returned boxes reference a declared class ID. Display names
can coincide. For example, nuScenes adult/child pedestrian categories keep
distinct native IDs while `display_group="pedestrian"` permits a single UI
checkbox controlling those IDs. Existing bus/pedestrian aliases must not erase
per-box native category identity. These hints do not impose a cross-dataset
ontology or automatically map detection-file classes.

`SensorInfo` has `sensor_id`, `modality`, `display_name`, and optional
`view_role`. Its position in the sensor tuple supplies a stable preferred order.
Camera roles can identify front/side/rear views; unknown roles are valid. Generic
viewer adapters use these hints and a deterministic fallback layout, without
hardcoding a dataset name or requiring a new camera-layout adapter per dataset.
Recognized role hints are `front_left`, `front`, `front_right`, `side_left`,
`side_right`, `rear_left`, `rear`, and `rear_right`. Duplicate, unknown, or absent
roles use the stable sensor order and fallback placement rather than failing.

## 5. Scene and frame identity

`SceneInfo` has `scene_id`, `display_name`, `description`, and
`frame_counts: Mapping[Modality, int | None]`. Counts are per modality; `None`
means not indexed/unknown, while zero means confirmed empty. Counts can be
resolved by returning an updated immutable scene record after indexing.

`FrameInfo` has `frame_id`, `scene_id`, `index`, `timestamp_us`, and
`modalities`. IDs are opaque, stable, nonempty strings. A frame's identity is
the pair `(dataset_id, frame_id)`. Consumers never parse IDs to obtain a native
file index, timestamp, or source path.

`list_frames(scene_id, modality=...)` returns only logical frames declared to
contain that modality. Camera availability means at least one camera, not a
complete rig. The same logical observation keeps the same ID and nominal
timestamp across modality lists and storage representations. Independent
asynchronous sensor observations need not share IDs or list lengths.
An unsupported modality raises `UnsupportedCapabilityError`; a supported
modality with no observations in a known scene returns an empty tuple.

`index` is dense and zero-based within the returned scene/modality list.
It can differ between the camera and point-cloud lists. It is never a native
storage offset. Sparse original indices and filtered catalogs retain their
native identity and must still select the correct source files/records.
The `info.index` in a returned payload belongs to that payload's modality list,
regardless of which list the caller previously enumerated. Preparation preserves
the input `info` unchanged.

Ordering is deterministic and follows the dataset's native chronological order
where known. Repeated timestamps are allowed with a stable native tie-breaker.
Unknown timestamps remain `None`; indices must not be converted into invented
times. Membership filtering preserves IDs and order.

`list_scenes(split="all")` returns the installed catalog. A valid split with
no installed members returns an empty tuple; an unknown or incompatible split
raises `DatasetConfigurationError`. No split silently falls back to `all`.

## 6. Time and coordinate conventions

Known timestamps are integer microseconds within the declared clock domain;
absolute UTC is not required. Normalize comparable sensor clocks before
returning them. Unknown or unconvertible times remain `None`, and must not be
compared as if they were synchronized.

Frame nominal time, sensor exposure time, scan interval, and geometry-reference
pose time are distinct. Records expose them separately where available.
In particular, Waymo's scan-start timestamp must not be reused as the timestamp
of its approximately mid-scan reference vehicle pose. If the latter timestamp
is unknown, `reference_pose_timestamp_us` is `None`.

Every geometry payload declares `reference_frame_id`, with documented axes and
handedness. Coordinates use metres; box dimensions are `(length, width, height)`.
Box rotations are proper `3 x 3` matrices mapping box-local coordinates into
the declared reference. A transform named `T_target_from_source` maps source
coordinates into target coordinates, using homogeneous column-vector notation.

Reference IDs are scoped to `dataset_id` and identify an anchored coordinate
system, including its pose/reference observation. The label `vehicle` at two
different times must not imply one identical spatial reference. The anchor can
use a frame/pose identity even when its exact timestamp is unknown. Matching
reference IDs mean the same origin, axes, and anchor; shared axis labels alone
are insufficient for combining geometry. Optional world transforms also name
their target with `world_frame_id`; transforms into different world references
cannot be combined without an explicit mapping.

Point-cloud points and their default boxes share the cloud's declared reference.
A camera set can use a scene-world or a declared local reference. World poses
are optional: absent world knowledge is `None`, never a fabricated identity
world pose. An identity is valid only when the two declared references coincide.

Sensor-native optical/vehicle/LiDAR axis differences are handled explicitly by
the provider and its transforms. A frontend does not assume every dataset's
point reference is a LiDAR frame or a vehicle frame.

## 7. Point-cloud records

`PointCloudFrame` contains:

- `info: FrameInfo` and `source_id: str` identifying the selected representation.
- `reference_frame_id`, `reference_pose_timestamp_us`, and optional
  `world_frame_id`/`T_world_from_reference` supplied together.
- Optional `scan_start_us` and `scan_end_us`, separately from the nominal frame time.
- `points`: finite floating-point `(N, 3)` XYZ, allowing `N == 0`.
- `attributes`: immutable mapping of row-aligned `(N, ...)` arrays. Intensity,
  ring, sensor/return identifiers, elongation, and other native features remain
  optional and preserve their original values. Attribute meanings/units and
  any integer-to-sensor-ID mapping are documented by the provider.
- `annotations: AnnotationSet`, in the cloud reference.
- Optional derived `rgb: PointColors`, `timing: PointTiming`, and
  `acquisition: AcquisitionGeometry` records described below.

`source_id` is an opaque baseline representation/profile identity within the
logical dataset: it distinguishes native versus processed data and row-affecting
choices such as sensor/return selection, filtering, or compensation. It is not
a machine path or a synonym for `dataset_id`. Relocating identical data does
not change it; changing the baseline representation does. An ID match alone is
not proof of point correspondence or permission to prepare a foreign frame.

The selected storage representation defines the baseline point rows. Removing
invalid native range-image placeholders during decoding is permitted. UI
filtering, downsampling, and display coloring do not belong in the loader.
Preparing optional data never changes the baseline point count, order, values,
frame identity, attributes, or default annotations.

All array records are read-only and own or retain the storage needed for their
lifetime. Loading another frame must not overwrite an earlier returned array.

## 8. Camera records and projection

`CameraFrameSet` contains `info`, `source_id`, `reference_frame_id`, nullable
`reference_pose_timestamp_us`, optional `world_frame_id`/`T_world_from_reference`
supplied together, a `cameras` tuple of `CameraObservation` records, and default
`annotations` in the set reference. Camera tuple order follows sensor metadata.
The tuple is nonempty for a successful camera load; a known frame with no
camera observations raises `DataUnavailableError` instead.

Each camera observation contains:

- Stable `sensor_id` referencing metadata.
- A nonempty, read-only `image: uint8 (H, W, 3)` RGB array in its native orientation.
- Nullable `timestamp_us` and a documented `timestamp_kind` identifying its native
  image-time meaning; nullable `exposure_interval_us` when an exposure envelope
  is actually known. The separate nullable `pose_timestamp_us` describes the
  camera pose. Unknown values remain `None`; no midpoint is fabricated.
- Optional `intrinsics` and `T_reference_from_camera` at the stated pose time.
  Camera coordinates use optical axes: X right, Y down, Z forward.
- Optional `projection_model`, mapping points in the set's declared reference
  directly into this original image.
- Optional camera-specific `annotations` overriding the default set; `None`
  means use the default set, while an available empty override means zero boxes.

Images without calibration are valid camera data. Every projectable camera
supplies a `ProjectionModel`, including a reusable pinhole implementation for
ordinary undistorted images. Intrinsics alone never authorize an implicit
pinhole assumption for an original distorted/rolling-shutter image.

The small projection interface uses a Python `Protocol`:

```python
class ProjectionModel(Protocol):
    def project_reference(self, points: np.ndarray) -> ProjectionResult: ...
```

Input is finite `(N, 3)` points in the declared set reference. `ProjectionResult`
contains `uv: float (N, 2)`, `depth: float (N,)` in metres along optical forward,
and `valid_mask: bool (N,)`. Valid means a finite, converged projection with
positive depth. Outside-image coordinates can remain valid so box edges can be
clipped; image bounds and occlusion are separate checks. Invalid UV values can
be NaN and must not be used. Pixels use zero-based `(u, v)` with integer pixel
centres; array sampling indexes `[v, u]`.

Where camera-space evaluation is defined, `depth` remains finite and signed,
including for behind-camera points whose validity is false. This supports
near-plane clipping. Failed/undefined evaluations can return NaN depth with a
false mask. Consumers must subdivide or discard undefined projection intervals,
not connect valid endpoints across them as if the whole interval were valid.

Distortion, rolling shutter, exposure poses, and native camera axes belong in
the projection model. Generic frontends do not reconstruct a simpler camera
from dataset names. RGB preparation can instead use native point-to-image
associations without rerunning this projection interface.

## 9. Ground-truth annotations

`AnnotationSet` contains `status`, `reference_frame_id`, `timing: AnnotationTiming`,
and a `boxes` tuple of `Box3D` records. `AnnotationTiming` has nullable
`timestamp_us` and an `alignment` value: `REFERENCE_TIME`, `CAMERA_SYNCHRONIZED`,
`ESTIMATED`, `UNKNOWN`, or `MIXED`. A mixed set has no single timestamp.
Each box has `annotation_id`, `class_id`, finite `center`, positive LWH `size`,
proper `rotation`, optional `instance_id`, and optional `timing_override`.
The override is an `AnnotationTiming`; `None` inherits the set timing. Each box
in a mixed set must provide its own non-mixed timing, whose timestamp can still
be unknown when the alignment relation alone is known.

Annotation IDs are unique within one set and stable on repeated reads of that
observation. Their complete identity includes dataset and frame IDs. They are
not implicitly cross-frame tracking IDs: Waymo's label ID can populate both
fields, while nuScenes annotation tokens and instance tokens populate
`annotation_id` and `instance_id` respectively. Camera overrides reuse matching
annotation IDs when they describe the same logical boxes.

Status is one of:

- `AVAILABLE`: annotations are known; an empty tuple means zero objects.
- `NOT_PROVIDED`: annotations are not supplied for this observation/source.
- `WITHHELD`: annotations are intentionally withheld for this release/split.

The last two statuses require empty boxes and are distinct from zero objects.
A malformed or missing declared annotation file raises an error rather than
becoming an empty annotation set. Absence of annotations never makes an otherwise
valid point-cloud or camera payload unsupported.

Annotation timestamps/alignment remain distinct from sensor exposure times.
Camera-synchronized native boxes can override the default set for their
designated camera; other cameras retain the original annotation reference time.
Both default and camera-specific sets use `CameraFrameSet.reference_frame_id`.
An override containing synchronized and unchanged boxes is explicitly mixed;
one camera timestamp must not be assigned falsely to every box in that set.
No object-motion compensation is claimed unless the provider actually supplies it.

## 10. Optional point preparation

`PointCloudPreparation` is immutable and contains three independent booleans,
all defaulting to false: `rgb`, `timestamps`, and `acquisition_geometry`.
Requests are explicit so ordinary reads do not decode unrelated images or
estimate acquisition times.

Derived records are row-aligned with the baseline points:

| Record | Required values |
| --- | --- |
| `PointColors` | `values`: finite float `(N,3)` RGB in `[0,1]`; `valid_mask`: bool `(N,)`; `provenance`. |
| `PointTiming` | `timestamps_us`: int64 `(N,)`; `valid_mask`: bool `(N,)`; `quality`; `provenance`. |
| `AcquisitionGeometry` | `points`: finite float `(N,3)` acquisition coordinates; `valid_mask`: bool `(N,)`; `reference_frame_id`; `provenance`. |

`ProcessingProvenance` has a stable `method_id`, a tuple of `source_ids`, and
a tuple of documented `assumptions`. Dataset-specific method names are allowed;
for example, nominal HDL-32E estimation and native Waymo camera associations
remain distinguishable.

Invalid row values are placeholders and must never be interpreted without their
mask. Missing derived records are `None`, rather than plausible-looking values.
Timing quality is `RECORDED` only when all valid timestamps came from recorded
point timing; otherwise it is `ESTIMATED`. The provenance identifies the method,
source inputs, and relevant assumptions. Estimated values must never be labeled
recorded. Frame time alone is insufficient to claim per-point timing.

Acquisition coordinates refer to the acquisition-time version of a stated
sensor/rig reference, consistent across valid rows. Per-return acquisition poses
can supply these coordinates without supplying scalar timestamps. Mixed sensor
frames must be converted to a common declared reference or kept unavailable.
For this per-point representation, its reference ID identifies the acquisition
frame family for this observation; it does not claim that all returns share a
single rigid pose. Default boxes remain in the baseline reference.

`rgb.valid_mask` means a usable color sample, not proven physical visibility.
Native associations and geometric reprojection identify their different methods.
Zero RGB matches is a valid result when the required inputs exist. Missing
requested inputs cannot be disguised as zero matches. RGB fallback gray and
distance color maps are viewer choices.
The same rule applies to timing/acquisition masks: all-false masks are valid
only with valid supporting inputs and no eligible points. Known missing inputs
raise `DataUnavailableError`; a declared missing/corrupt file raises the
appropriate I/O/validation error.

Preparation accepts a baseline frame from the same loader/configuration, or a
previous preparation result from it. Identity, source, and point correspondence
must be checked before attaching derived arrays. Pairing independent processed
and native clouds requires a verified mapping/order; matching counts alone is
insufficient. Reordered or filtered inputs without a mapping must be rejected.

Requests are atomic: failure returns no partially prepared record and does not
mutate the input. Existing unrequested derived data is preserved. Repeated
requests are semantically equivalent and can reuse cached immutable results.
Requested fields are minimum guarantees; valid native/cached fields or helper
fields derived for that request may also be retained. An empty request performs
no optional work.
Partial coverage is explicit in masks. A consumer must not silently present
partially uncompensated geometry as a fully uncompensated scan.

## 11. Availability, errors, and loading behavior

Metadata, scene listing, and frame listing may read/index metadata, but do not
decode sensor payloads. Payload methods read only requested modalities/features.
A camera-only extract must initialize and operate without LiDAR files; ordinary
point-cloud viewing must operate without camera images. Invalid metadata solely
for an inactive modality does not block the active view; it raises when used.

Frame modality declarations are metadata knowledge, not a promise that every
referenced file has been checked. A known absent modality/feature, an unsupported
operation, and an expected-but-missing file are different outcomes.

All loader errors derive from `DatasetError`:

| Error | Meaning |
| --- | --- |
| `DatasetConfigurationError` | Invalid roots, release, split, format, source name, or option. |
| `UnsupportedCapabilityError` | This configured loader does not offer the operation. |
| `UnknownIdentifierError` | Unknown scene or frame ID. |
| `DataUnavailableError` | Known frame lacks the requested modality/feature. |
| `DatasetIOError` | A declared source cannot be accessed/read, including missing selected files. |
| `DatasetValidationError` | Corrupt/inconsistent source data, invalid geometry, or failed correspondence. |
| `DatasetClosedError` | A dataset operation was called after close. |

Errors identify the relevant dataset, scene/frame/sensor, path, or option and
retain the original exception cause. Unexpected I/O/corruption is never converted
silently into an empty cloud, absent camera, or unavailable annotation status.

A camera set may omit channels that are known absent. A channel declared for
that frame whose file cannot be loaded is an error, not an omitted camera.

Methods are synchronous. Callers serialize calls per loader instance; the
contract does not require concurrent-call safety. The UI owns its worker/event
loop. Providers bound their caches and must not load an entire sensor dataset
to enumerate a scene. The caller waits for its in-flight operations to finish
before calling close; close does not manage the application's worker threads
and must not run concurrently with a read.

## 12. Dataset-specific interpretation

| Source | RGB | Scalar point timestamps | Acquisition geometry |
| --- | --- | --- | --- |
| nuScenes | Geometric camera projection; nominal timing is explicitly estimated. | Estimated nominal HDL-32E timing when required inputs exist. | Estimated undoing of existing compensation when required inputs exist. |
| Native Waymo | Native camera associations retained separately for both LiDAR returns. | Not inferred from TOP per-pixel poses; unavailable unless genuinely supplied. | TOP pose-based coordinates can be exposed with a partial mask. |
| OpenPCDet Waymo | Available when native pairing/correspondence is verified or a valid mapping exists. | No fabricated timestamps from point order or frame index. | Advertised only if implemented with the required supporting data/mapping. |
| Point-cloud-only dataset | Optional; native colors need no camera-image capability. | Optional. | Optional. |
| Camera-only dataset | Point coloring unsupported without point observations. | Unsupported without point observations. | Unsupported without point observations. |

Camera and ground-truth capabilities reflect configured source availability;
public test data can return images/clouds with `WITHHELD` ground truth.

## 13. Extension and conformance

A contributor implements the required catalog methods, the payload operations
their capabilities advertise, and any optional processing. They register a
constructor accepting the common config:

```python
from dataset_utils import create_loader, register_loader
from my_dataset import MyDatasetLoader

register_loader("my_dataset", MyDatasetLoader)
loader = create_loader("my_dataset", config)
```

Explicit registration before launching an application is the version-1 plugin
mechanism. External-package auto-discovery is not required for this contract.
The frontends must consult the shared registry and capabilities, including
generic sensor/layout handling, rather than require separate dataset
registrations or dataset-specific imports in each viewer.

Conformance checks must cover:

1. A point-only and a camera-only test provider each need one backend and one
   registration; every applicable viewer opens them without viewer source edits.
2. Metadata enumeration is lazy, deterministic, and correctly handles unknown
   counts, sparse original indices, modality-specific indices, and stable IDs.
3. Nonidentity rotations/translations, sensor axes, reference frames, and
   exposure/pose timestamps are preserved; absent world poses remain absent.
4. Available-empty, not-provided, and withheld annotations remain distinguishable.
5. Partial RGB/timing/acquisition masks work, including valid black pixels;
   native associations and estimated timing have correct provenance.
6. Preparation is atomic, immutable, row-preserving, and rejects wrong-source
   or unverified reordered processed/native correspondence.
7. Missing/corrupt declared files differ from unsupported/absent data, while
   unrelated modalities do not prevent valid loads.
8. Camera projection supports distorted/shutter-aware and uncalibrated-image
   cases without guessing pinhole calibration. Near-plane-crossing boxes remain
   visible where defined, and undefined projection intervals are not bridged.
9. Cache eviction and idempotent close preserve returned arrays; cleanup and
   closed-instance behavior follow the lifecycle rules.
10. Shared base/models/registry import without either viewer or a GUI framework.

## 14. Compatibility boundary

`create_loader` returns normalized `NuScenesLoader` or `WaymoLoader` instances,
or an explicitly registered extension. Both viewers' normal dataset factories
use generic presentation bridges over these loaders. Legacy per-viewer demo
and test adapters can still be explicitly registered through their old APIs.

The low-level `NuScenesReader`, `NativeWaymoReader`, and OpenPCDet storage helper
remain available. Their native/legacy records are not the normalized contract
records. In particular, `dataset_utils.models` retains legacy compatibility
models, while normalized models are exported from `dataset_utils` and
`dataset_utils.contracts.models`. Existing 3D imports remain aliases for the
legacy records so the detection viewer and existing integrations keep working.

Version-1 behavior is the conformance target. Optional additions can be backward
compatible; changes to method meanings, reference conventions, or required
record fields require an explicit contract-version change.
