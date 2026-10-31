# 🔎 3ddetviz

Inspect 3D detections alongside ground truth, with reversible filters and the
shared [3dviz](../3dviz/README.md) renderer. Open an existing export, compare its
saved proposals, and keep the point cloud and viewpoint while adjusting filters.

**[Quick start](#install-and-launch) · [Open data](#open-a-dataset) · [Predictions](#prediction-files) · [Review controls](#review-controls-and-scientific-conventions) · [Extensions](#caches-and-extension-points)**

<a id="install-and-launch"></a>

## 🚀 Quick start

Use Python 3.10–3.12 with a desktop session and an OpenGL-capable display. From
the repository root, install the packages and launch the generated demo:

```bash
python -m pip install -e './visual_utils' -e './viz_tools/3dviz' \
  -e './viz_tools/3ddetviz[dev]'
3ddetviz --demo
```

The demo provides deterministic clouds, ground truth and overlapping detection
proposals without dataset files. To connect your own data, start with
`3ddetviz` and use the connection form. `3ddetviz --help` and
`python -m detviz --help` describe the launch options without opening a GUI.

<a id="open-a-dataset"></a>

## 🗂️ Open a dataset

For a nuScenes validation export, supply the dataset and prediction paths:

```bash
3ddetviz --root /datasets/nuscenes --version v1.0-mini --split val \
  --predictions /results/results_nusc.json
```

The root can be the dataset parent or its metadata version directory. Native
JSON tables and LiDAR files support ordinary viewing. Camera coloring also
needs the corresponding images. Torch, OpenPCDet and the nuScenes SDK are
unnecessary. Full nuScenes metadata uses in-memory indexes, so measure startup
memory in your environment before scaling from mini.

`--scene` accepts a scene name or ID, and `--frame` selects a zero-based frame
index. `--dataroot` is an alias for `--root`. You can open ground truth first,
then attach or replace an export with **Load predictions**.

The default split is `val` for nuScenes and demo data, and `all` for other
providers. On nuScenes mini, `val` and `train` select the official `mini_val`
and `mini_train` scenes. Use `test` with `v1.0-test`. Other split names are
validated by the selected provider.

For multiple sources or provider-specific options, all three viewers accept the
same [dataset configuration](../../visual_utils/docs/custom_datasets.md):

```bash
3ddetviz --dataset waymo --dataset-config dataset.json --split val \
  --predictions /results/results.pkl
```

Relative configuration paths resolve against the JSON file. Named sources
resolve against its root when one is supplied. `--dataset-config` cannot be
combined with `--root` or an explicit `--version`. These connection settings
stay fixed for the window's lifetime, while scene, frame and split navigation
remain available.

<a id="prediction-files"></a>

## 📦 Load prediction files

Frame IDs must match exactly. An absent prediction frame means coverage is
unknown for that frame. A present frame containing zero detections is an
explicitly empty result. Preserve the class IDs expected by the prediction
format, and map custom model labels explicitly in your importer. For example,
nuScenes results use `car` while ground truth retains `vehicle.car`. Display
groups align their labels, while confidence overrides and NMS use the prediction
class IDs.

**nuScenes results JSON** contains a top-level `results` mapping by sample ID.
Each detection supplies `sample_token`, world `translation`, `size` in
`(width, length, height)` order, WXYZ `rotation`, `detection_name` and
`detection_score`. Optional velocity is world XY. Import converts dimensions
to LWH, then normalization uses the cloud's real `T_world_from_reference`
pose. Unknown world poses are an error.

**OpenPCDet NumPy pickle** contains a list of frame dictionaries or an envelope
with `frames`. Each frame needs `metadata.token` and either `boxes_lidar`/`score`
or `pred_boxes`/`pred_scores`. Boxes are `(N, 7)` or `(N, 9)` arrays containing
`x, y, z, length, width, height, yaw[, vx, vy]` in reference coordinates.
Supply names in `name`, or one-based `pred_labels` with ordered `class_names`
metadata. Empty frames still need correctly shaped arrays. The importer accepts
NumPy arrays and primitive metadata, and rejects Torch tensors and model objects.

Suffix detection selects the built-in formats. Use
`--prediction-format nuscenes_json`, `pcdet_pickle`, or a registered custom
format when an export has an unfamiliar suffix.

Processing stage is `unknown` unless the producer declares `pre_nms` or
`post_nms` in embedded `detviz` metadata or a `result.pkl.detviz.json` sidecar:

```json
{"name": "Model proposals", "stage": "pre_nms", "coordinate_frame": "lidar"}
```

Use `world` for nuScenes JSON. Contradictory coordinate declarations are
rejected. Keep the producer's thresholds and candidate limits with the export
so reviewers can understand which proposals were saved.

<a id="review-controls-and-scientific-conventions"></a>

## 🎛️ Review controls and interpretation

Ground truth and detections have independent visibility and colors, green and
orange by default. Class checkboxes affect both layers. Labels come from dataset
metadata while boxes retain their native class IDs.

| Control | Behavior |
| --- | --- |
| Confidence | The initial threshold is `0.1`. **Reset** removes it. Per-class overrides replace the global threshold for those native class IDs. |
| Spatial limits | Filter each box center by XY distance and optional Z bounds. |
| Candidate and output limits | Limit candidates before NMS and results afterward. Zero retains no detections, and the UI uses `-1` for unlimited. |
| Additional NMS | Choose off, per-class, class-agnostic or CenterPoint head groups. |
| Presets | Save or load filters, colors and line width as JSON. Optional bounds use `null`, and preset `schema_version` is `1`. |

Filtering follows confidence → spatial limits → stable descending score order
→ pre-NMS candidate limit → NMS → output limit. Threshold boundaries are
inclusive. NMS suppresses overlaps strictly greater than its IoU threshold,
and equal scores retain source order. Sidebar counts describe these stages
before class or layer visibility is applied.

Filters always start from the imported predictions. Relaxing a threshold can
restore a box hidden by the viewer, while preserving the cloud and viewpoint.
It cannot recover proposals removed before export.

Spatial controls assume horizontal reference XY and +Z upward. Distance is
the box center's XY radius, and height is its center's Z coordinate. NMS uses
the convex footprints of fully oriented cuboids projected onto XY. Its overlap
measure is **BEV IoU**, rather than 3D volume IoU or a detector's CUDA
implementation. Providers must convert clouds, annotations, calibration and
predictions consistently for these controls to have their usual driving
interpretation. The historical coordinate label `lidar` means this cloud
reference, including for non-LiDAR sources.

Grouped NMS joins truck/construction vehicle, bus/trailer, motorcycle/bicycle,
and pedestrian/traffic cone. Car and barrier remain separate, and other classes
form their own groups. Group settings and confidence overrides use native class
IDs.

Point colors offer gray, viridis, concerto and Camera RGB when supported.
Distance palettes use Euclidean distance from the displayed reference origin,
with a fixed 0–80 metre scale by default. The maximum is adjustable, and farther
points use the endpoint color. Point appearance, acquisition geometry, axes and
mouse controls come from 3dviz. Its
[controls and scientific interpretation](../3dviz/README.md#controls-and-scientific-interpretation)
explain the palette, camera and timing conventions.

<a id="caches-and-extension-points"></a>

## 🧩 Caches and extensions

JSON imports stream one frame at a time into a SQLite index. Navigation then
reads only the selected frame. Pickle's first import loads the complete pickle
before indexing. Caches default to `$XDG_CACHE_HOME/3ddetviz/predictions`, or
`~/.cache/3ddetviz/predictions` when that variable is unset. Override the
location with `--cache-dir`.

Cache identity includes the resolved source path, filesystem identity, size,
modification and change timestamps, sidecar contents and cache schema. Cache
hits do not reread the export payload or scan the entire database. Finish
writing exports before opening them. If your storage preserves file metadata
across content changes, delete the old cache to force rebuilding. Interrupted
imports never publish a partial index, and caches can be removed while the
viewer is closed.

To support a new dataset,
[implement one DatasetLoader](../../visual_utils/docs/custom_datasets.md).
The point-cloud window consumes it directly, so no detection adapter is needed.
All geometry uses `visual_utils.PointCloudFrame` and `visual_utils.Box3D`.
`ReviewFrame` composes the observation, original predictions and filtered
results, while `compose_review` and `refilter` work without a GUI.

For custom model outputs, follow the [prediction-source guide](docs/prediction_contract.md).
Implement `PredictionSource`, or use `InMemoryPredictionSource` for predictions
already produced by Python. File importers register independently through
`register_prediction_format`.

## 🧪 Validation and next steps

Run the package checks from the repository root:

```bash
python -m pytest -q viz_tools/3ddetviz/tests
DETVIZ_DESKTOP_TESTS=1 LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a \
  python -m pytest -q viz_tools/3ddetviz/tests/test_desktop.py
```

The CPU suite uses synthetic data. Native integration also checks rendering,
navigation, filtering, presets, source replacement and failure recovery, and
requires working display drivers. Visual inspection and synthetic regressions
do not establish detector benchmark accuracy.

Continue with the [prediction-source guide](docs/prediction_contract.md) for a
runnable Python example, or the [contribution checks](../../CONTRIBUTING.md) for
cross-package validation.
