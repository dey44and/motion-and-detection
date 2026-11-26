# 🔌 Add your own dataset

Connect your storage to `3dviz` and Python by implementing `DatasetLoader`.
This tutorial starts with a runnable example, then explains how to adapt it to
your own files and sensor conventions.

**[Package overview](../README.md)** ·
**[Interface reference](loader_contract.md)** ·
**[Example source](../examples/custom_dataset/src/research_dataset/__init__.py)**

The [example package](../examples/custom_dataset/) reads NPZ point clouds, PNG
images, and a JSON catalog. It depends only on `visual_utils` and imports no viewer.

## 🚀 Run the example

From the repository root, with `visual_utils` and `3dviz` installed:

```bash
python -m pip install -e './visual_utils/examples/custom_dataset'
python -m research_dataset.generate /tmp/research-dataset-example
3dviz --dataset research --root /tmp/research-dataset-example --split review
```

Use a new output directory, as the generator refuses to overwrite an existing one.
The generated data are synthetic and deterministic, with sparse frame IDs
`scan-15` and `scan-39`, a calibrated camera, one annotated box per frame and
1000 points per cloud. Click the **Image** camera preview in the 3D scene to
inspect the generated image and projected box. Detection review initially
shows ground truth. You can attach predictions after connecting the dataset.

## 📦 Make the loader discoverable

Installation exposes this standard entry point:

```toml
[project.entry-points."visual_utils.loaders"]
research = "research_dataset:ResearchDatasetLoader"
```

`3dviz` discovers the installed name automatically and imports the selected
provider. Entry points export a callable accepting one `DatasetConfig`. Their
modules must not also call `register_loader` for that name. Names are
case-normalized and collisions are errors. `demo` is reserved for the generated
viewer example. Restart a running process after installing or uninstalling a
provider.

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

## 🔧 Implement your provider

Read the [reference implementation](../examples/custom_dataset/src/research_dataset/__init__.py)
alongside the [complete contract](loader_contract.md). Its structure is:

1. Call `DatasetLoader.__init__(config)`. Validate release, storage format, named
   sources and provider-specific options. Read catalog metadata, not payloads.
2. Return a stable `DatasetMetadata` snapshot describing classes, ordered sensors,
   supported splits and capabilities for the configured sources.
3. Implement `_list_scenes(split=...)` and `_list_frames(scene_id, modality=...)`.
   Return tuples with stable opaque IDs and dense modality-specific display indices.
4. Implement `_load_point_cloud` and/or `_load_cameras` for the advertised modalities.
   Use the shared immutable records, retaining optional data as `None` when unknown.
5. Implement `_camera_rig` for clickable camera previews in the 3D scene.
   Return calibrated mount positions and image corner rays from metadata.
   [Camera record conventions](loader_contract.md)
6. Implement `_prepare_point_cloud` only for additional capabilities your data
   support. Preserve baseline rows and provide validity masks and provenance.
7. Override `_close()` when files, databases or native resources remain open.
   Inherit `close()` so the shared lifecycle remains idempotent.

Callers use the public methods without underscores. The base class delegates
explicitly to your methods and handles closed state, splits and capabilities.
Shared records validate numerical shapes and geometry at construction. Your
provider and its tests establish calibration, timing, and row-preserving optional
processing. This keeps the runtime path short and the dataset semantics explicit.

### 🗂️ Understand the example format

The example catalog has these fields:

| Location | Fields |
| --- | --- |
| Manifest | `dataset_id`, `scene_id`, nonempty `frames`, optional `intrinsics`. |
| Frame | `frame_id`, optional integer `timestamp_us`, `points` and/or `image`, optional `boxes`. |
| Box | `annotation_id`, `center` XYZ, `size` LWH, proper `rotation` matrix. |
| NPZ | Floating `(N,3)` `points`, with optional row-aligned `intensity`. |

Payload paths resolve against the dataset root. Every box uses the declared
`object` class. Missing `boxes` means annotations were not provided, while
`boxes: []` means they are available with zero objects. Images without
`intrinsics` remain accessible through `load_cameras`.

### 📏 Preserve sensor conventions

The example uses a local vehicle reference (X forward, Y left, Z up) for
clouds and boxes, and an explicit nonidentity rotation into the undistorted
camera's optical coordinates (X right, Y down, Z forward). Current 3D height,
horizontal-distance and BEV tools require Z-up with horizontal XY. Convert other
native conventions in your provider, including box rotations and associated
transforms. Each example frame has a distinct anchored reference and no known
world transform. Per-point timing, acquisition geometry, and RGB coloring are
left absent because the example does not supply them.

This compact catalog is a teaching format. Adapt the provider to your native
storage rather than converting every dataset to it. For multiple asynchronous
sensors, expose independent modality catalogs when appropriate. Supply your
actual projection model for distortion or rolling shutter. Intrinsics alone do
not justify a pinhole assumption.

## ⚙️ Reuse a dataset configuration

`3dviz` and Python clients use the same JSON configuration. Save this example
as `/tmp/research-view.json`.

```json
{
  "root": "research-dataset-example",
  "storage_format": "native"
}
```

Launch the workspace with the configuration file.

```bash
3dviz --dataset research --dataset-config /tmp/research-view.json --split review
```

The root resolves relative to the JSON file. Named sources resolve relative to
that root, or to the file directory when root is absent. Allowed top-level keys
are exactly `root`, `release`, `storage_format`, `source_paths` and `options`.
Duplicate keys, non-finite option numbers and unknown fields are errors.
Pass `--dataset-config` on its own. Configured source controls stay fixed for
that launch. Split, scene, and frame navigation remain available.

### 🐍 Load the same data in Python

```python
from visual_utils import Modality, create_loader, load_dataset_config

with create_loader("research", load_dataset_config("/tmp/research-view.json")) as loader:
    scene = loader.list_scenes(split="review")[0]
    info = loader.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD)[0]
    cloud = loader.load_point_cloud(info.frame_id)
    camera_set = loader.load_cameras(info.frame_id)
    print(cloud.points.shape, camera_set.cameras[0].image.shape)
```

After registering a provider in the current process, launch the workspace with
`threedviz.web.server.run_server`. It accepts `dataset=`, `config=`, `split=`, and
`autoload=True`.

## 🎯 Add predictions independently

`3dviz` joins predictions by the exact normalized frame ID. Its built-in
OpenPCDet and nuScenes prediction readers load experimental exports.
For your own outputs, implement a `PredictionSource` or use
`InMemoryPredictionSource`, as shown in the
[prediction-source guide](../../viz_tools/3dviz/docs/prediction_contract.md).
Predictions already expressed in the cloud reference use the historical
coordinate label `lidar`, even when that reference is not physically a LiDAR.
World predictions require the loader's real world transform. Missing world
poses must never be replaced with identity transforms to make an overlay appear.

## 🧪 Validate your interpretation

The shared loader and browser registration tests exercise record validation,
payload lifetimes, and registered providers. Extend those patterns with
nonidentity transforms, asynchronous cameras, unknown times, empty scenes,
withheld annotations, and your own sensor edge cases. Check numerical projection
against independently known calibration points before relying on visual alignment.
[Loader contract tests](../tests/test_contract.py) · [Viewer dataset tests](../../viz_tools/3dviz/tests/test_datasets.py)

Providers are synchronous and need not support concurrent calls. Serialize
operations per instance and wait for reads to finish before closing. Keep sensor
caches bounded. A context manager owns the loader, and returned arrays remain
usable after close. [Viewer test commands](../../viz_tools/3dviz/README.md#validation)
