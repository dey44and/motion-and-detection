# 3ddetviz

Review 3D detections beside ground truth using the [3dviz](../3dviz/README.md)
renderer. Dataset readers return the same canonical `visual_utils.PointCloudFrame`
and `Box3D` records used by the other viewers. Prediction imports, coordinate
normalization and reversible filters are separate from dataset loading.

## Install and launch

Use Python 3.10–3.12, a desktop session and an OpenGL-capable display. From the
repository root:

```bash
python -m pip install -e './visual_utils' -e './viz_tools/3dviz' -e './viz_tools/3ddetviz[dev]'
3ddetviz --demo
```

The demo generates deterministic clouds, ground truth and overlapping proposals
without dataset files. For a nuScenes export:

```bash
3ddetviz --root /datasets/nuscenes --version v1.0-mini --split val \
  --predictions /results/results_nusc.json
```

The root may be the dataset parent or metadata version directory. Native JSON
tables and LiDAR files are sufficient for ordinary viewing; optional camera
coloring needs the corresponding images. Torch, OpenPCDet and the nuScenes SDK
are unnecessary. Full nuScenes metadata uses in-memory indexes; measure startup
memory in your dataset environment before scaling from mini.

`--scene` accepts a scene name or ID; `--frame` is zero-based. `--dataroot` aliases
`--root`. Starting without a root opens the connection form. A dataset can be
opened without predictions, then **Load predictions** attaches or replaces a
source. `3ddetviz --help` and `python -m detviz --help` work without opening a GUI.

The default split is `val` for nuScenes/demo and `all` for other providers.
nuScenes mini aliases `val`/`train` to official `mini_val`/`mini_train`; use `test`
with `v1.0-test`. Custom split names are validated by the selected provider.

Every viewer accepts the same [dataset configuration](../../visual_utils/docs/custom_datasets.md):

```bash
3ddetviz --dataset waymo --dataset-config dataset.json --split val \
  --predictions /results/results.pkl
```

Configuration paths resolve relative to the JSON file; named sources resolve
relative to its root when supplied. Do not combine `--dataset-config` with
`--root` or an explicit `--version`. Connection settings remain fixed for the
window's lifetime. Use `--prediction-format nuscenes_json`, `pcdet_pickle`, or a
registered custom format when suffix detection is inappropriate.

## Review controls and scientific conventions

Ground truth and detections have independent visibility and colors, green and
orange by default. Class checkboxes affect both layers; labels come from dataset
metadata while boxes retain native class IDs. Point appearance, camera RGB,
acquisition geometry, axes and mouse controls come from 3dviz. See its
[camera and timing conventions](../3dviz/README.md#camera-rgb-and-point-timing).

Detection filters run in this order:

1. Confidence, including optional per-class overrides.
2. Box-center distance and height limits.
3. Stable descending score order and an optional pre-NMS candidate limit.
4. Optional NMS and an output limit.

Threshold boundaries are inclusive. NMS suppresses overlaps strictly greater
than its IoU threshold; equal scores retain source order. The initial score
threshold is `0.1`; **Reset** removes it. Candidate limits accept zero, and the
UI uses `-1` for unlimited. Filters retain the original imported predictions,
so relaxing a viewer threshold restores boxes hidden by the viewer. They cannot
recover proposals removed before export. Sidebar counts precede class and layer
visibility; filter changes preserve the cloud and viewpoint.

Spatial controls assume horizontal reference XY and +Z upward. Distance is the
box center's XY radius; height is its center's Z coordinate. NMS uses convex
footprints of the oriented cuboid projected onto XY: **BEV IoU**, not 3D volume
IoU or a detector's CUDA implementation. Providers must convert native clouds,
annotations, calibration and predictions consistently for these controls to have
their usual driving interpretation. The historical prediction coordinate label
`lidar` means this cloud reference, including for non-LiDAR sources.

NMS modes are off, per-class, class-agnostic, or CenterPoint head groups.
Default groups join truck/construction vehicle, bus/trailer, motorcycle/bicycle,
and pedestrian/traffic cone; car and barrier remain separate. Other classes form
separate groups. Group settings and confidence overrides use native class IDs.
Save/load JSON presets for filters, colors and line width; optional bounds use
`null`, and preset `schema_version` is `1`.

## Prediction files

**nuScenes results JSON** contains a top-level `results` mapping by sample ID.
Each detection supplies `sample_token`, world `translation`, `(width,length,height)`
`size`, WXYZ `rotation`, `detection_name` and `detection_score`. Optional velocity
is world XY. Import converts dimensions to LWH; normalization uses the cloud's
real `T_world_from_reference` pose. Unknown world poses are an error.

**OpenPCDet NumPy pickle** contains a list of frame dictionaries or an envelope
with `frames`. Each frame needs `metadata.token` and `boxes_lidar`/`score` or
`pred_boxes`/`pred_scores`. Boxes are `(N,7)` or `(N,9)` arrays containing
`x,y,z,length,width,height,yaw[,vx,vy]` in reference coordinates. Supply names in
`name`, or one-based `pred_labels` with ordered `class_names` metadata. Empty
frames still require correctly shaped arrays. The importer accepts NumPy and
primitive metadata, and rejects Torch tensors and model objects.

IDs match exactly; absent frames differ from explicitly empty predictions.
Export native dataset class IDs, or map model labels in your prediction importer.
Processing stage is `unknown` unless `pre_nms` or `post_nms` is declared in
embedded `detviz` metadata or a `result.pkl.detviz.json` sidecar:

```json
{"name": "Model proposals", "stage": "pre_nms", "coordinate_frame": "lidar"}
```

Use `world` for nuScenes JSON; contradictory declarations are rejected. Preserve
producer thresholds and candidate limits with the export for interpretation.

## Caches and extension points

JSON imports stream one frame at a time into a SQLite index; navigation reads
only the selected frame. Pickle's first import loads the full pickle before
indexing. Caches default to `$XDG_CACHE_HOME/3ddetviz/predictions` or
`~/.cache/3ddetviz/predictions`; override with `--cache-dir`.

Cache identity uses the resolved source path, filesystem identity, size,
modification/change timestamps, sidecar contents and cache schema. Cache hits
do not reread the export payload or scan the entire database. Finish writing
exports before opening them. If your storage preserves file metadata across
content changes, delete the old cache to force rebuilding. Interrupted imports
never publish a partial index; caches can be removed while the viewer is closed.

[Implement one DatasetLoader](../../visual_utils/docs/custom_datasets.md) for a new
dataset. The point-cloud window consumes it directly; there is no detection
adapter to implement. `ReviewFrame` composes an observation, original predictions
and filtered results. `compose_review` and `refilter` work without a GUI.

[Prediction extensions](docs/prediction_contract.md) implement `PredictionSource`
or use `InMemoryPredictionSource` for Python-generated outputs. File importers
register separately with `register_prediction_format`. All geometry uses the
single public `visual_utils.Box3D` type; prediction records add scores and IDs.

Run checks from the repository root:

```bash
python -m pytest -q viz_tools/3ddetviz/tests
DETVIZ_DESKTOP_TESTS=1 LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a \
  python -m pytest -q viz_tools/3ddetviz/tests/test_desktop.py
```

The CPU suite uses synthetic data. Native integration additionally checks
rendering, navigation, filtering, presets, source replacement and failure
recovery; it requires working display drivers. Visual inspection and synthetic
regressions do not establish detector benchmark accuracy.
