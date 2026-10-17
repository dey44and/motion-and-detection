# 3ddetviz

A desktop viewer for comparing 3D detections with dataset ground truth. It uses the existing [3dviz](../3dviz/README.md) point-cloud viewer, adding prediction loading, validation-split selection, and reversible detection filters. Ground truth and detections have separate visibility controls and uniform colors: green for ground truth and orange for detections by default.

## Install and launch

Use Python 3.10–3.12 with a desktop session and an OpenGL-capable display. From `motion-and-detection`:

```bash
python3 -m venv 3ddetviz/.venv
source 3ddetviz/.venv/bin/activate
python -m pip install -e './3dviz' -e './3ddetviz[dev]'
3ddetviz --demo
```

The demo generates clouds, ground truth, overlapping detection proposals and confidence scores without dataset files. For a nuScenes validation export:

```bash
3ddetviz \
  --root ../pc-corrector/data/nuscenes/v1.0-trainval \
  --version v1.0-trainval \
  --split val \
  --predictions /path/to/results_nusc.json
```

Replace the root with your extracted nuScenes location. The parent dataset root and the metadata version directory are both accepted, following 3dviz's dataset-layout rules. Native nuScenes JSON tables and `samples/LIDAR_TOP` files are sufficient; loading predictions does not require Torch, OpenPCDet or the nuScenes SDK.

Full nuScenes metadata currently uses 3dviz's in-memory indexes. In the local full trainval check, opening the dataset took about one minute and the complete GUI process peaked around 8.4 GiB of memory. Prediction-file streaming is bounded independently; its low memory use does not include the dataset metadata or renderer. Mini is suitable for a quick first launch.

```bash
3ddetviz --root /path/to/nuscenes --version v1.0-mini --split val \
  --predictions /path/to/result.pkl --scene scene-0103 --frame 3
3ddetviz --help
python -m detviz --help
```

`--scene` accepts a scene name or token; `--frame` is zero-based. `--dataroot` aliases `--root`. `--demo` and `--dataset demo` both open the generated demonstration. Launching without a root opens the connection form. A dataset can be loaded without predictions for ground-truth inspection; **Load predictions** attaches or replaces its prediction source later.

The default split is `val`. On `v1.0-mini`, `val` and `train` select the official `mini_val` and `mini_train` scenes. On `v1.0-trainval`, they select the full official splits. Use `--split test` with `v1.0-test`; `all` keeps every installed scene. Split membership is bundled for offline use, and incompatible release/split combinations report an error. Prediction matching always uses sample tokens rather than frame positions; absent prediction frames are distinguished from frames explicitly containing zero detections.

## Review controls

The left panel selects the scene and frame, controls point appearance, and configures detection filtering. The white 3D viewport supports mouse rotation and scroll-wheel zoom. Existing 3dviz controls remain available: point size, gray/jet/turbo distance coloring, optional camera RGB, estimated acquisition-time points, axes, and camera reset. See [3dviz's camera and timing requirements](../3dviz/README.md#camera-rgb-and-point-timing) for the optional image files and limitations.

- **Ground truth and detections:** show or hide each layer independently and change either layer's color. Each layer uses one color across all classes.
- **Classes:** shared checkboxes control which classes appear in both layers. Each row reports ground-truth and filtered-detection counts. One line-width slider applies to both layers.
- **Minimum confidence:** adjust the global score threshold with the slider or number field. The initial threshold is `0.1`; reset removes the viewer's score threshold.
- **Spatial limits:** set the minimum horizontal distance from the reference LiDAR origin and optionally enable a maximum distance. Minimum and maximum box-center z height can each be enabled independently. Values use metres and test the box center rather than the cuboid's nearest edge.
- **Candidate/output limits:** retain the highest scoring candidates before NMS and cap the final output. In the UI, `-1` means unlimited; `0` intentionally retains zero predictions.
- **Additional NMS:** choose **Off — as saved**, **Per class**, **CenterPoint head groups**, or **Class agnostic**, and set the BEV IoU overlap threshold.
- **Presets:** save/load confidence, spatial limits, candidate limits, NMS settings, the two colors and box line width in a JSON file.

Filtering runs in this order: confidence → spatial limits → score ordering → pre-NMS limit → NMS → output limit. Threshold boundaries are inclusive; NMS suppresses overlaps strictly greater than its IoU threshold. Equal scores retain their source order. Sidebar stage counts describe this processing pipeline before class/layer visibility is applied. Filtering changes detection boxes only; it preserves the cloud and ground-truth annotations. Appearance and filter updates retain the current 3D viewpoint.

NMS uses oriented bird's-eye-view footprints in reference LiDAR XY, with full box orientation retained. The overlap measure is **BEV IoU**, rather than 3D volume IoU or the detector's CUDA implementation. Grouped NMS compares `truck` with `construction_vehicle`, `bus` with `trailer`, `motorcycle` with `bicycle`, and `pedestrian` with `traffic_cone`; `car` and `barrier` each have their own group. Unlisted classes form separate groups. Groups can be customized in a preset.

Per-class confidence overrides are supported through preset JSON; each override replaces the global threshold for that class. For example:

```json
{
  "schema_version": 1,
  "filters": {
    "score_threshold": 0.15,
    "per_class_scores": {"pedestrian": 0.3, "traffic_cone": 0.25},
    "max_distance": 80.0,
    "min_height": -4.0,
    "max_height": 3.0,
    "nms_mode": "class",
    "nms_iou_threshold": 0.2,
    "pre_nms_limit": 5000,
    "max_detections": 500
  },
  "appearance": {
    "gt_color": [0.08, 0.55, 0.22],
    "detection_color": [0.9, 0.25, 0.03],
    "line_width": 2.0
  }
}
```

Optional filter bounds and limits use `null` for unlimited in preset files. Candidate/output limits accept zero to retain zero predictions, consistently with programmatic `FilterConfig` instances. The UI represents an unlimited limit as `-1`.

## Prediction formats and processing stage

**nuScenes results JSON** uses the official top-level `results` mapping, keyed by sample token. Each detection supplies `sample_token`, world `translation`, `size` in `(width, length, height)`, WXYZ `rotation`, `detection_name`, and `detection_score`; world XY `velocity` is optional. The loader converts sizes to `(length, width, height)` and uses each frame's reference LiDAR pose to transform world predictions into the cloud coordinate system.

**OpenPCDet NumPy pickle** accepts a list of per-frame dictionaries, or an envelope with `frames`. Each frame needs `metadata.token` and either `boxes_lidar`/`score` or `pred_boxes`/`pred_scores`. Boxes have shape `(N, 7)` or `(N, 9)` and store `(x, y, z, length, width, height, yaw[, vx, vy])` in reference LiDAR coordinates. Supply class names in `name`, or 1-based `pred_labels` with an explicit ordered `class_names` mapping in metadata. Empty frames must still contain correctly shaped empty arrays. The importer accepts NumPy reconstruction primitives and plain metadata, and rejects Torch tensors and model objects.

The viewer retains the original imported detections and recomputes filters from them whenever controls change. Relaxing a threshold can restore a box that the viewer hid. It cannot restore boxes removed before the prediction file was written. Standard evaluation exports may already have score filtering, candidate limits and NMS; **Off — as saved** disables additional viewer NMS.

Processing stage is `unknown` unless the producer declares `pre_nms` or `post_nms`. A declaration describes the saved data; it does not recover removed candidates. Add a sidecar named `result.pkl.detviz.json` (or `result.detviz.json`), for example:

```json
{
  "name": "Validation candidates",
  "stage": "pre_nms",
  "coordinate_frame": "lidar",
  "class_names": ["car", "truck", "pedestrian"]
}
```

Use `coordinate_frame: "world"` for nuScenes JSON; a contradictory declaration is rejected. An embedded `detviz` metadata object can also carry these declarations. Preserve the producer's score floor and candidate limits with the export so reviewers can interpret its remaining candidates.

### Exporting pre-NMS candidates from pc-corrector

From the `pc-corrector` repository, run the existing evaluation command with the additional export flags for the supported CenterHead path:

```bash
python tools/test.py \
  --cfg_file /path/to/model.yaml \
  --ckpt /path/to/checkpoint.pth \
  --save_pre_nms \
  --pre_nms_score_thresh 0.001
```

This additionally writes `result_pre_nms.pkl` and its `result_pre_nms.pkl.detviz.json` sidecar, declaring the `pre_nms` stage. Existing final-result files remain available. Load the candidate pickle through `--predictions /path/to/result_pre_nms.pkl` in 3ddetviz.

These are decoded candidates captured before NMS, after the head's top-K selection and spatial gates. The export score floor and model/configuration candidate caps still limit which boxes can be recovered; the export does not contain every dense output logit. The export path supports CenterHead with `nms_gpu` or `nms_normal_gpu`; circle NMS and other detection heads are rejected. The export integration has automated checks, but full model inference with a real checkpoint must be validated in the detector's configured environment.

## Large exports and caches

JSON is parsed incrementally with `ijson`, keeping one frame's predictions in memory during import and writing a per-frame SQLite index. Navigation then reads only the selected frame. Decoded clouds and camera images also use bounded caches. Import progress appears in the window; loading and filtering run on its worker thread.

Caches default to `$XDG_CACHE_HOME/3ddetviz/predictions`, or `~/.cache/3ddetviz/predictions` when that variable is unset. Override the location with `--cache-dir /path/to/cache`. File identity, modification metadata, sidecar contents and cache-format version determine reuse. Changed exports create a new index; interrupted imports do not publish a partial cache. Cache files can be removed when the viewer is closed and will be rebuilt as needed.

Pickle must be deserialized before its frames can be indexed. Its initial import therefore has a one-time memory cost proportional to the complete pickle. Subsequent launches reuse the SQLite index. Prefer JSON for a very large export when its first-import memory requirements matter.

## Extension points and tests

```text
src/detviz/
├── controller.py      # dataset/prediction composition and preserved frame data
├── predictions/       # immutable detections, source registry and format importers
├── processing/        # pure filters and deterministic CPU BEV NMS
├── ui/                # review controls layered onto the shared 3dviz window
└── cli.py             # launch options with lazy graphical imports
```

Dataset support uses `threedviz.dataloader.DatasetAdapter` and its registry; adapters can implement `list_scenes_for_split` and expose `FrameData.reference_sensor_to_world` for world-coordinate detections. Existing point-cloud adapters remain usable with `all` and LiDAR-coordinate predictions. Prediction support is separate: implement `PredictionSource` and register an opener with `register_prediction_format`. Rendered `BoxLayer` objects are independent of import formats.

Run the CPU tests after installing the development dependencies:

```bash
python -m pytest 3ddetviz/tests
python -m pytest 3dviz/tests
```

The loaders, controller, filters and CLI tests use synthetic fixtures without a dataset download or display. Opening the native viewer requires a display and working OpenGL drivers; verify `3ddetviz --demo` interactively when changing rendering behavior.

Native integration checks cover the actual window, independent layers, filters, camera preservation, point-color modes, navigation, presets and failure recovery. On Linux, run the two viewer suites in separate processes:

```bash
DETVIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest 3ddetviz/tests
THREEDVIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest 3dviz/tests
```
