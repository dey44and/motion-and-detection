# ☁️ 3dviz

Inspect autonomous-driving point clouds, 3D ground truth, predicted detections,
camera-derived point colors, and nuScenes LiDAR segmentation in one local
browser workspace.

Dataset reading belongs to [`visual-utils`](../../visual_utils/README.md).
Its `DatasetLoader` and canonical observations are shared with research scripts
and [2dcameraviz](../2dcameraviz/README.md). This package provides point colors,
box geometry, detection filters, camera navigation, and asynchronous controls.

**[Quick start](#install-and-launch)** · **[Data and configuration](#dataset-sources)** · **[Controls and interpretation](#controls-and-scientific-interpretation)** · **[Research and extensions](#research-api-and-extension-boundary)** · **[Validation](#validation)**

<a id="install-and-launch"></a>

## 🚀 Quick start

Use Python 3.10–3.12 and a browser with WebGL 2. From the repository root,
install the shared library and viewer, then open the deterministic demo:

```bash
python -m pip install -e './visual_utils' -e './viz_tools/3dviz'
3dviz --demo
```

The server opens [http://localhost:8080](http://localhost:8080) in your browser.
Use `--no-open-browser` to start just the server, `--port` to choose a different
port, and Ctrl+C to stop it. The demo generates its own observations. Running
`3dviz` without a root opens **Sources** for dataset selection. To list the
command-line options:

```bash
python -m threedviz --help
```

The workspace keeps dataset, split, and scene selectors above the viewport,
with the frame scrubber, step controls, and frame jump below it. The collapsible
inspector has **Scene**, **Detection**, and **LiDAR segmentation** tabs. Switch
tabs to inspect the same frame and camera view. **Sources** holds dataset settings
and prediction attachments. The camera panel supports editable JSON and preserves
unapplied drafts while you navigate the viewport.

Dataset roots and configuration paths refer to the machine running the server.
`--dataset-config` fixes the connection settings for that launch. Ordinary
launches can change the dataset connection in the page.

Point size, distance colors, box styles, and camera navigation update in the
browser. The dataset provider prepares camera RGB with a coverage mask. Camera
view JSON stores position, rotation, pivot, field of view, and clipping in
reference coordinates.

Browser points use instanced 20-face icosahedrons with subtle face lighting.
Point size sets their screen diameter in pixels. The selected point colors
supply the base colors before lighting.

The Python package bundles JavaScript, styles, and Source Sans Pro for offline
use. The server runs without a display and independently of OpenPCDet, CUDA,
TensorFlow, and dataset SDKs.

### Detection and LiDAR segmentation

In **Sources**, attach a nuScenes results JSON or OpenPCDet NumPy pickle as the
detection export. You can also attach one at launch:

```bash
3dviz --dataset nuscenes --root /datasets/nuscenes --version v1.0-trainval \
  --split val --predictions /results/result.pkl
```

`--prediction-format` overrides extension-based detection with `nuscenes_json`,
`pcdet_pickle`, or a registered format. `--cache-dir` sets the directory for
indexed detection exports. The default continues to use existing caches in
`$XDG_CACHE_HOME/3ddetviz/predictions` or `~/.cache/3ddetviz/predictions`.

In **Detection**, switch the solid ground-truth and dashed prediction layers
independently. Confidence is available directly. Under **Advanced**, set
per-class confidence overrides, XY and Z limits, candidate and output limits,
and BEV NMS. The stage counts follow confidence, spatial filtering, pre-NMS
limit, NMS, and output limit. Import or export JSON presets to share review
settings, including existing 3ddetviz presets. The viewer distinguishes an
absent prediction frame, an empty result, and a result filtered to zero.

For nuScenes LiDAR segmentation, install the official labels alongside the
dataset. Ground truth uses `<root>/<release>/lidarseg.json` and the label files
under `<root>/lidarseg/<release>/`. To review predictions, attach a Sonata
pickle such as `result_lidarseg.pkl`, or a nuScenes submission folder or ZIP.
Select **Ground truth** or **Prediction** in the **LiDAR segmentation** tab.
Both use the same evaluation-class palette and show ignored points separately.
Each label file must match the exact LiDAR scan and contain one ID per original
point row. When labels are unavailable or invalid, the viewer keeps the cloud
visible in neutral colors and reports the reason. Export fields, coordinate
conventions, and custom readers are documented in the
[prediction contract](docs/prediction_contract.md).

Existing `3dviz --web` commands still serve the workspace and retain explicit
browser opening via `--open-browser`. Launch detection review with `3dviz`
using the prediction flags shown above.

<a id="dataset-sources"></a>

## 🗂️ Data and configuration

Connect directly to nuScenes or Waymo with a dataset root:

```bash
3dviz --dataset nuscenes --root /datasets/nuscenes --version v1.0-mini --split val
3dviz --dataset nuscenes --root /datasets/nuscenes --version v1.0-trainval --split val
3dviz --dataset waymo --root /datasets/waymo --split val
```

**nuScenes** accepts the extracted dataset root or a metadata version directory.
Install `samples/LIDAR_TOP` to view keyframes. Optional RGB coloring also needs
the referenced `samples/CAM_*` and `sweeps/CAM_*` images. The reader transforms
boxes into the LiDAR reference frame and uses local length, width and height
for dimensions. On trainval and test, the first connection builds a local
metadata index. Later connections reuse it, with sensor files and label
blobs still read from the dataset. See the cache location, disk space, and
configuration options in the [native reader guide](../../visual_utils/docs/dataset_readers.md#cache-nuscenes-metadata).

**Waymo** accepts native TFRecords or an OpenPCDet processed workspace. Automatic
selection prefers processed clouds when both representations are present.
Select the TFRecord directory explicitly to view native clouds. Processed point
files retain their original row order and are paired with native records for
optional camera colors. Sparse original frame indices remain distinct from
dense display indices. The [native reader guide](../../visual_utils/docs/dataset_readers.md)
describes directory layouts, split discovery, correspondence checks and formats.

Use these options to choose the initial view:

| Option | Meaning |
| --- | --- |
| `--root`, `--dataroot` | Equivalent paths to the dataset source. |
| `--version` | Dataset release, with `v1.0-mini` as the nuScenes default. |
| `--split` | Dataset-defined split. By default, nuScenes and demo launches with `--predictions` select `val`. Other launches select `all`. |
| `--scene` | Initial scene ID or display name. |
| `--frame` | Zero-based display index within the selected scene. |

### ⚙️ Configure separate sources

For separate data directories or provider options, save a configuration such as
`/datasets/waymo.json`:

```json
{
  "release": "1.4.2",
  "storage_format": "openpcdet",
  "source_paths": {
    "processed": "./processed",
    "native": "./tfrecords"
  }
}
```

Then launch the viewer with that file:

```bash
3dviz --dataset waymo --dataset-config /datasets/waymo.json --split val
```

Relative paths resolve against the JSON file's directory. When `root` is
supplied, that root anchors relative named sources instead. `--dataset-config`
replaces `--root` and `--version`, and combining them reports an error. The
connection fields stay fixed for this launch, while splits, scenes and frames
remain selectable. Both viewers accept the same configuration.

<a id="controls-and-scientific-interpretation"></a>

## 🎛️ Controls and scientific interpretation

Choose observations using the dataset, scene, and frame controls. Adjust the
display in the inspector and navigate the cloud in the viewport.

| Control | Behavior |
| --- | --- |
| Split, scene and frame | Choose an observation or step through frames with previous/next. |
| Point colors | Choose `gray`, `viridis` or `concerto` distance coloring and adjust the maximum distance. |
| Camera RGB | Use image colors when supported. Unmatched points use gray, and the sidebar reports valid color coverage. |
| Undo ego motion compensation | Display complete acquisition geometry when available. Boxes retain their reference-time pose. |
| Boxes and points | Filter boxes by class, change their colors and line width, and adjust point size. |
| Camera navigation | Drag to rotate, right-drag to pan, scroll to zoom, or reset the camera. |
| Coordinate axes | Toggle fixed 3-metre axes with X in red, Y in green and Z in blue. Toggling preserves the viewpoint and leaves camera fitting unchanged. |

Distance colors use Euclidean distance from the displayed point-cloud reference
origin, which need not be the ego origin. The scale defaults to 0–80 metres.
Its adjustable maximum stays fixed across frames, and farther points use the
endpoint color. Gray keeps a dark-to-light range suitable for the white
background. Camera RGB uses image colors instead of the distance palette.

Concerto interpolates this palette from near to far:
`#231F62` → `#3D87AE` → `#7EDFD0` → `#D95EA5` → `#EC599D` → `#F28C88` →
`#F4AA79` → `#56BA62`.

RGB, scalar point timestamps and acquisition geometry are independent features.
A dataset can provide acquisition geometry without scalar timestamps. Partial
geometry remains available in canonical observations, but cannot be displayed
as a fully uncompensated scan. Waymo's partial per-pixel pose coverage therefore
leaves that control disabled.

nuScenes per-point timestamps are nominal HDL-32E estimates, rather than recorded
measurements. Camera colors use each exposure's calibration and pose, estimated
acquisition times, and approximate sparse occlusion checks. Moving objects are
not motion compensated, and a color-validity mask does not prove physical
visibility. Timing quality and processing provenance retain these assumptions
on the canonical observations. See the [native reader guide](../../visual_utils/docs/dataset_readers.md)
for timing, coordinate conventions and limitations.

<a id="research-api-and-extension-boundary"></a>

## 🧩 Research API and extensions

A new dataset implements `visual_utils.DatasetLoader` once. It supplies metadata,
ordered scene/frame catalogs and observations in its declared reference frames.
Install a provider exposing a `visual_utils.loaders` entry point to use it with
`3dviz --dataset NAME`. For local work, call `visual_utils.register_loader`
before launching the viewer in the same process. The [custom dataset guide](../../visual_utils/docs/custom_datasets.md)
walks through both approaches.

The viewer factory returns the shared loader directly. Research scripts can read
and prepare the same observations as the browser workspace:

```python
from visual_utils import DatasetConfig, Modality, PointCloudPreparation
from threedviz.dataloader import create_dataset

with create_dataset("nuscenes", config=DatasetConfig(
    root="/datasets/nuscenes", release="v1.0-mini",
)) as dataset:
    scene = dataset.list_scenes(split="val")[0]
    info = dataset.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD)[0]
    frame = dataset.load_point_cloud(info.frame_id)
    colored = dataset.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True))
    print(colored.points.shape, colored.annotations.status, colored.rgb.valid_mask.sum())
```

All consumers share one point-cloud record representation. Rendering uses its
points, annotations, optional RGB and acquisition geometry directly. Native
class IDs remain in boxes, and an explicit display-label map controls grouping
and styling. Providers own correspondence and preserve baseline rows and
existing features during preparation.

Each browser tab owns a session that serializes loading and preparation on
one worker. Failed loads retain the current frame. The session closes replaced
datasets and cleans up after pending reads finish. In Python, serialize reads
and use a context manager to own the loader.

The package follows that boundary:

```text
src/threedviz/
  dataloader/       dataset selection and canonical synthetic demo
  detection/       prediction readers, detection filters and review presets
  processing/      dataset-independent geometry and colors
  web/             local API, background jobs, binary frames and bundled browser UI
  cli.py           arguments and local server launch
```

Point positions, optional colors, and segmentation labels travel as binary
arrays. The server rebases rendering positions before conversion to float32
and retains the source coordinates in canonical observations.

To develop the browser frontend, use the checked-in npm lockfile:

```bash
cd viz_tools/3dviz/web_frontend
npm ci
npm run build
```

The build type-checks the TypeScript sources and writes the bundled assets into
`src/threedviz/web/static` for Python packaging. Commit source and rebuilt assets
together. Frontend dependencies are needed only for development.

<a id="validation"></a>

## ✅ Validation

From the repository root, run the viewer tests:

```bash
python -m pip install -e './visual_utils' -e './viz_tools/3dviz[dev]'
python -m pytest viz_tools/3dviz/tests
```

Tests cover shared-provider selection, geometry, detection filters and presets,
segmentation alignment, and data transfer to the browser. With Playwright and
Chrome or Chromium installed, run the browser integration suite:

```bash
THREEDVIZ_BROWSER_TESTS=1 python -m pytest viz_tools/3dviz/tests/test_web_browser.py
```

Use `THREEDVIZ_BROWSER_BINARY` if Chrome or Chromium is outside your PATH.
See the [custom dataset guide](../../visual_utils/docs/custom_datasets.md)
for provider development and the [contribution guide](../../CONTRIBUTING.md)
for repository-wide validation.
