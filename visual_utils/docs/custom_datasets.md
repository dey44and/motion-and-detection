# Add a custom dataset once, use it across viewers

Implement a `DatasetLoader` that converts your storage into the normalized
contract. The same provider works in every viewer supporting its modalities.
This guide uses a small, executable provider stored in
[`examples/custom_dataset`](../examples/custom_dataset/). It reads actual NPZ
clouds, PNG images and a JSON catalog; its implementation imports no viewer.

## Run the reference package

From the repository root, with the visualization packages installed:

```bash
python -m pip install -e './visual_utils/examples/custom_dataset'
python -m research_dataset.generate /tmp/research-dataset-example
2dcameraviz --dataset research --root /tmp/research-dataset-example --split review
3dviz --dataset research --root /tmp/research-dataset-example --split review
3ddetviz --dataset research --root /tmp/research-dataset-example --split review
```

Use a new output directory; the generator refuses to overwrite an existing one.
The generated data are synthetic and deterministic, with sparse frame IDs
`scan-15` and `scan-39`, a calibrated camera, one annotated box per frame and
1000 points per cloud. The camera image is a flat background; it illustrates
file loading and geometric overlays rather than a reconstructed scene. Detection
review initially shows ground truth without predictions; prediction loading is
an independent step described below.

Installation exposes this standard entry point:

```toml
[project.entry-points."visual_utils.loaders"]
research = "research_dataset:ResearchDatasetLoader"
```

All three viewer parsers discover the installed name automatically. Only the
selected provider is imported. Entry points export a callable accepting one
`DatasetConfig`; their modules must not call `register_loader` for the same name.
Names are case-normalized and collisions are errors. Avoid `demo`, which the
viewer applications reserve for their generated examples. Restart a running
process after installing or uninstalling a provider.

For a notebook, script, or uninstalled local module, explicit registration works
in the current process instead:

```python
from visual_utils import DatasetConfig, create_loader, register_loader
from my_dataset import MyDatasetLoader

register_loader("my_lab", MyDatasetLoader)
with create_loader("my_lab", DatasetConfig(root="/datasets/my_lab")) as loader:
    print(loader.metadata)
```

Registration in a script does not persist into a separately launched shell
command. Use an entry point for ordinary installed CLI use. Do not combine
explicit registration with an installed entry point of the same name.

## Implement the provider

Read the [reference implementation](../examples/custom_dataset/src/research_dataset/__init__.py)
alongside the [complete contract](loader_contract.md). Its structure is:

1. Call `DatasetLoader.__init__(config)`. Validate release, storage format, named
   sources and provider-specific options. Read catalog metadata, not payloads.
2. Return a stable `DatasetMetadata` snapshot describing classes, ordered sensors,
   supported splits and capabilities for the configured sources.
3. Implement `_list_scenes(split=...)` and `_list_frames(scene_id, modality=...)`.
   Return tuples; use stable opaque IDs and dense modality-specific display indices.
4. Implement `_load_point_cloud` and/or `_load_cameras` for the advertised modalities.
   Use the shared immutable records, retaining optional data as `None` when unknown.
5. Implement `_prepare_point_cloud` only for additional capabilities your data
   support. Preserve baseline rows and provide validity masks and provenance.
6. Override `_close()` when files, databases or native resources remain open.
   Inherit `close()` so the shared lifecycle remains idempotent.

Callers use the public methods without underscores. The base class delegates
explicitly to your methods and handles closed state, splits and capabilities.
Shared records validate numerical shapes and geometry at construction. Your
provider and its tests establish calibration, timing and row-preserving optional
processing; no hashes or automatic wrappers enforce these on every call.

The example catalog has these fields:

| Location | Fields |
| --- | --- |
| Manifest | `dataset_id`, `scene_id`, nonempty `frames`, optional `intrinsics`. |
| Frame | `frame_id`, optional integer `timestamp_us`, `points` and/or `image`, optional `boxes`. |
| Box | `annotation_id`, `center` XYZ, `size` LWH, proper `rotation` matrix. |
| NPZ | Floating `(N,3)` `points`; optional row-aligned `intensity`. |

Payload paths resolve against the dataset root. Every box uses the declared
`object` class. Missing `boxes` means not provided; `boxes: []` means available
with zero objects. Images without `intrinsics` remain viewable without overlays.
The example uses a local vehicle reference (X forward, Y left, Z up) for
clouds and boxes, and an explicit nonidentity rotation into the undistorted
camera's optical coordinates (X right, Y down, Z forward). Current 3D height,
horizontal-distance and BEV tools require this Z-up/XY-horizontal convention;
convert other native conventions in your provider, including box rotations and
associated transforms. Each frame has a distinct anchored reference and no
known world transform. It does not invent
per-point timing, acquisition geometry or RGB coloring.

This compact catalog is a teaching format. Adapt the provider to your native
storage rather than converting every dataset to it. For multiple asynchronous
sensors, expose independent modality catalogs when appropriate. Supply your
actual projection model for distortion or rolling shutter; intrinsics alone do
not justify a pinhole assumption.

## Configure and consume it

Every viewer accepts the same strict JSON configuration. For example, place
this in `/tmp/research-view.json`:

```json
{
  "root": "research-dataset-example",
  "storage_format": "native"
}
```

Then run any of:

```bash
2dcameraviz --dataset research --dataset-config /tmp/research-view.json --split review
3dviz --dataset research --dataset-config /tmp/research-view.json --split review
3ddetviz --dataset research --dataset-config /tmp/research-view.json --split review
```

The root resolves relative to the JSON file. Named sources resolve relative to
that root, or to the file directory when root is absent. Allowed top-level keys
are exactly `root`, `release`, `storage_format`, `source_paths` and `options`.
Duplicate keys, non-finite option numbers and unknown fields are errors.
Do not also pass `--root` or `--version`. Configured source controls stay fixed
for the desktop session, while scene/frame/split navigation remains available.

For direct Python consumption:

```python
from visual_utils import Modality, create_loader, load_dataset_config

with create_loader("research", load_dataset_config("/tmp/research-view.json")) as loader:
    scene = loader.list_scenes(split="review")[0]
    info = loader.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD)[0]
    cloud = loader.load_point_cloud(info.frame_id)
    camera_set = loader.load_cameras(info.frame_id)
    print(cloud.points.shape, camera_set.cameras[0].image.shape)
```

Applicable viewers can also be launched in the same process after registering
a provider. Each `run_app` accepts `dataset=`, `config=`, `split=` and
`autoload=True`; see the individual package README for imports and examples.

## Add predictions independently

`3ddetviz` joins predictions by the exact normalized frame ID. Its built-in
OpenPCDet and nuScenes readers are file adapters, not required dataset formats.
For your own outputs, implement a `PredictionSource` or use
`InMemoryPredictionSource`; see the
[prediction-source guide](../../viz_tools/3ddetviz/docs/prediction_contract.md).
Predictions already expressed in the cloud reference use the historical
coordinate label `lidar`, even when that reference is not physically a LiDAR.
World predictions require the loader's real world transform. Missing world
poses must never be replaced with identity transforms to make an overlay appear.

## Validate your interpretation

The [cross-package example tests](../../tests/visualization/test_custom_dataset.py)
demonstrate lazy catalog access, sparse IDs, error categories, immutable payload
lifetimes and using one shared provider for all three viewers. Extend that pattern
with nonidentity transforms, asynchronous cameras, unknown times, empty scenes,
withheld annotations and your own sensor edge cases. Check numerical projection
against independently known calibration points before relying on visual alignment.

Providers are synchronous and need not support concurrent calls. Serialize
operations per instance and wait for reads to finish before closing. Keep sensor
caches bounded. A context manager owns the loader; returned arrays remain usable
after close. See [contribution checks](../../CONTRIBUTING.md) for test commands.
