# ☁️ 3dviz

Inspect autonomous-driving point clouds, 3D ground truth and camera-derived
colors in a standalone desktop viewer. Browse scenes and frames, adjust their
appearance, and inspect acquisition geometry where the dataset supports it.

Dataset reading belongs to [`visual-utils`](../../visual_utils/README.md).
Its `DatasetLoader` and canonical observations are shared with research scripts,
[2dcameraviz](../2dcameraviz/README.md) and [3ddetviz](../3ddetviz/README.md).
This package provides point colors, box geometry, camera navigation and
asynchronous desktop controls.

**[Quick start](#install-and-launch)** · **[Data and configuration](#dataset-sources)** · **[Controls and interpretation](#controls-and-scientific-interpretation)** · **[Research and extensions](#research-api-and-extension-boundary)** · **[Validation](#validation)**

<a id="install-and-launch"></a>

## 🚀 Quick start

Use Python 3.10–3.12 with an OpenGL-capable desktop. From the repository root,
install the shared library and viewer, then open the deterministic demo:

```bash
python -m pip install -e './visual_utils' -e './viz_tools/3dviz[dev]'
3dviz --demo
```

The demo needs no dataset files. Running `3dviz` without a root leaves the
connection form ready for input. To explore the command-line options:

```bash
python -m threedviz --help
```

CLI help imports neither the renderer nor native dataset readers. The viewer
runs independently of OpenPCDet, CUDA, TensorFlow and dataset SDKs. Opening a
window requires a display and working OpenGL drivers, while dataset reading,
geometry and color utilities remain usable without a desktop.

### 🛠️ Startup troubleshooting

If importing Open3D fails with `numpy.dtype size changed`, check NumPy and pandas
in the environment used by the `3dviz` command. Older pandas wheels cannot run
with NumPy 2. This package requires pandas 2.2.2 or later within the 2.x series.
Upgrade pandas in that same environment, for example:

```bash
uv pip install --python .venv/bin/python 'pandas>=2.2.2,<3'
```

Use the path to your viewer environment if it differs from `.venv`.

On Linux, a Conda-based interpreter can also load an older C++ runtime than the
system Mesa driver needs. If imports succeed but creating a window crashes,
select the system runtime for that launch, using its path on your machine:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/3dviz --demo
```

<a id="dataset-sources"></a>

## 🗂️ Data and configuration

Connect directly to nuScenes or Waymo with a dataset root:

```bash
3dviz --dataset nuscenes --root /datasets/nuscenes --version v1.0-mini --split val
3dviz --dataset waymo --root /datasets/waymo --split val
```

**nuScenes** accepts the extracted dataset root or a metadata version directory.
Install `samples/LIDAR_TOP` to view keyframes. Optional RGB coloring also needs
the referenced `samples/CAM_*` and `sweeps/CAM_*` images. The reader uses native
JSON tables directly, without devkit preprocessing. It transforms boxes into
the LiDAR reference frame and uses local length, width and height for dimensions.

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
| `--split` | Dataset-defined split, defaulting to `all`. The connected dataset supplies the sidebar choices. |
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
remain selectable. All three viewers accept the same configuration.

<a id="controls-and-scientific-interpretation"></a>

## 🎛️ Controls and scientific interpretation

Use the sidebar to choose observations and appearance, then navigate the cloud
in the viewport:

| Control | Behavior |
| --- | --- |
| Split, scene and frame | Choose an observation or step through frames with previous/next. |
| Point colors | Choose `gray`, `viridis` or `concerto` distance coloring and adjust the maximum distance. |
| Camera RGB | Use image colors when supported. Unmatched points use gray, and the sidebar reports valid color coverage. |
| Undo ego motion compensation | Display complete acquisition geometry when available. Boxes retain their reference-time pose. |
| Boxes and points | Filter boxes by class, change their colors and line width, and adjust point size. |
| Camera navigation | Drag to rotate, right-drag to pan, scroll to zoom, or reset the camera. |
| Coordinate axes | Toggle fixed 3-metre axes: X red, Y green and Z blue. Toggling preserves the viewpoint, and axes do not affect camera fitting. |

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

The viewer factory returns the shared loader directly, so a research script can
read and prepare the same observations as the desktop application:

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

The window serializes loading and preparation on one worker. Stale results
cannot replace a newer selection. Replaced datasets and datasets returned after
window closure are closed after pending reads finish. Programmatic clients
should also serialize reads and cleanup, using a context manager to own the
loader.

The package follows that boundary:

```text
src/threedviz/
  dataloader/       dataset selection and canonical synthetic demo
  processing/      dataset-independent geometry and colors
  visualization/   Open3D rendering of canonical observations
  ui/              controls and loading worker
  cli.py           arguments and lazy desktop launch
```

<a id="validation"></a>

## ✅ Validation and next steps

From the repository root, run the viewer tests:

```bash
python -m pytest viz_tools/3dviz/tests
```

Tests cover deterministic demo data, shared-provider selection, geometry, colors,
canonical rendering and box layers. With Xvfb and working OpenGL, opt into the
native window test:

```bash
THREEDVIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest viz_tools/3dviz/tests/test_desktop.py
```

The native test covers navigation, styles, camera axes, optional preparation,
error recovery and configured sources. Also check real nuScenes or Waymo data
and the installed display driver before using a scientific result based on
visual alignment.

Continue with the [custom dataset guide](../../visual_utils/docs/custom_datasets.md)
to add a provider, [3ddetviz](../3ddetviz/README.md) to compare predictions, or the
[contribution guide](../../CONTRIBUTING.md) for repository-wide validation.
