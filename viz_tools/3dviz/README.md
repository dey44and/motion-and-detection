# 3dviz

A desktop viewer for point clouds and 3D ground truth. Dataset reading belongs
to [`visual-utils`](../../visual_utils/README.md); this package owns point colors,
box geometry, camera navigation and asynchronous desktop controls. The same
`DatasetLoader` and canonical observations can be used from a research script,
[2dcameraviz](../2dcameraviz/README.md) and [3ddetviz](../3ddetviz/README.md).

## Install and launch

Use Python 3.10–3.12 with an OpenGL-capable desktop. From the repository root:

```bash
python -m pip install -e './visual_utils' -e './viz_tools/3dviz[dev]'
3dviz --demo
3dviz --dataset nuscenes --root /datasets/nuscenes --version v1.0-mini --split val
3dviz --dataset waymo --root /datasets/waymo --split val
python -m threedviz --help
```

The deterministic demo needs no dataset files. Launching without a root leaves
an empty connection form. `--scene` accepts a scene ID or display name;
`--frame` is a zero-based display index. `--root` and `--dataroot` are equivalent.
`--split` defaults to `all` and the connected dataset supplies available splits.
CLI help imports neither the renderer nor native dataset readers.

The viewer is independent of OpenPCDet, CUDA, TensorFlow and dataset SDKs.
Opening a window requires a display and working OpenGL drivers. Dataset reading,
geometry and color utilities remain usable without a desktop.

If Open3D import fails with `numpy.dtype size changed`, check the NumPy and
pandas versions in the environment used by the `3dviz` command. Older pandas
wheels do not support NumPy 2; this package requires pandas 2.2.2 or later.
Upgrade pandas in that same environment, for example:

```bash
uv pip install --python .venv/bin/python 'pandas>=2.2.2,<3'
```

Use the path to your viewer environment if it differs from `.venv`.

On Linux, a Conda-based interpreter can also load an older C++ runtime than the
system Mesa driver needs. If imports succeed but creating a window crashes,
select the system runtime for that launch (using its path on your machine):

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/3dviz --demo
```

## Dataset sources

nuScenes accepts the extracted dataset root or a metadata version directory.
Install `samples/LIDAR_TOP` to view keyframes; optional RGB coloring also needs
the referenced `samples/CAM_*` and `sweeps/CAM_*` images. Native JSON tables are
read directly, without devkit preprocessing. Boxes are transformed into the
LiDAR reference frame and dimensions use local length, width and height.

Waymo accepts native TFRecords or an OpenPCDet processed workspace. Automatic
selection prefers processed clouds when both representations are present.
Explicitly select the TFRecord directory to view native clouds. Processed point
files retain their original row order and are paired with native records for
optional camera colors. Sparse original frame indices remain distinct from
dense display indices. See the [shared source documentation](../../visual_utils/README.md)
for directory layouts, split discovery, correspondence checks and file formats.

For separate data directories or provider options, save a configuration:

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

```bash
3dviz --dataset waymo --dataset-config /datasets/waymo.json --split val
```

Relative paths are resolved against the configuration file, or its `root` when
supplied. `--dataset-config` replaces `--root` and `--version`; combining them
reports an error. The connection fields remain fixed for a configured launch;
splits, scenes and frames remain selectable. All three viewers accept the same
configuration.

## Controls and scientific interpretation

- Select a split, scene and frame; use previous/next to step through frames.
- Choose gray, viridis or concerto distance colors. The scale defaults to
  0–80 metres; its adjustable maximum stays fixed across frames, and farther
  points use the endpoint color. Distance is Euclidean distance from the
  displayed point-cloud reference origin, which need not be the ego origin.
- Choose Camera RGB when supported. Unmatched points use gray and the sidebar
  reports valid color coverage.
- Use Undo ego motion compensation when complete acquisition geometry is
  available. Boxes retain the reference-time pose.
- Filter boxes by class, change their colors and line width, and adjust point size.
- Drag to rotate, right-drag to pan, scroll to zoom, or reset the camera.
- Toggle fixed 3-metre axes: X red, Y green and Z blue. Axes do not affect
  camera fitting and toggling them preserves the viewpoint.

Concerto interpolates this palette in order from near to far:
`#231F62` → `#3D87AE` → `#7EDFD0` → `#D95EA5` → `#EC599D` → `#F28C88` →
`#F4AA79` → `#56BA62`. Gray retains its dark-to-light range for the white
background. Camera RGB uses image colors rather than the distance palette.

RGB, scalar point timestamps and acquisition geometry are independent features.
A dataset may provide acquisition geometry without scalar timestamps. Partial
geometry is retained in canonical observations, but cannot be displayed as a
fully uncompensated scan. Waymo's partial per-pixel pose coverage therefore does
not enable that control.

nuScenes per-point timestamps are nominal HDL-32E estimates, rather than recorded
measurements. Camera colors use each exposure's calibration and pose, estimated
acquisition times, and approximate sparse occlusion checks. Moving objects are
not motion compensated; a color-validity mask does not prove physical visibility.
The [shared dataset documentation](../../visual_utils/README.md) describes timing,
coordinate conventions and limitations. These assumptions remain attached to
canonical observations through timing quality and processing provenance.

## Research API and extension boundary

A new dataset implements `visual_utils.DatasetLoader` once. It supplies metadata,
ordered scene/frame catalogs and observations in its declared reference frames.
Install a provider exposing a `visual_utils.loaders` entry point to use it from
the ordinary `3dviz --dataset NAME` command. For local work, call
`visual_utils.register_loader` before launching the viewer in the same process.
See the [custom dataset guide](../../visual_utils/docs/custom_datasets.md).

The viewer factory returns the shared loader directly:

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

There is one point-cloud record representation. Rendering consumes its points,
annotations, optional RGB and acquisition geometry directly. Native class IDs
remain in boxes; an explicit display-label map controls grouping and styling.
No per-viewer dataset registry, frame reconstruction cache, reader forwarding,
content hashing or ownership tracking is required. Providers own correspondence
and preserve baseline rows and existing features during preparation.

The window serializes loading and preparation on one worker. Stale results cannot
replace a newer selection. Replaced datasets and datasets returned after window
closure are closed after pending reads finish. Programmatic clients should also
serialize reads and cleanup and use a context manager.

```text
src/threedviz/
  dataloader/       dataset selection and canonical synthetic demo
  processing/       dataset-independent geometry and colors
  visualization/    Open3D rendering of canonical observations
  ui/               controls and loading worker
  cli.py            arguments and lazy desktop launch
```

## Validation

```bash
python -m pytest viz_tools/3dviz/tests
THREEDVIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest viz_tools/3dviz/tests/test_desktop.py
```

Tests cover deterministic demo data, shared-provider selection, geometry, colors,
canonical rendering and box layers. The opt-in native test covers navigation,
styles, camera axes, optional preparation, error recovery and configured sources.
Real nuScenes/Waymo data and the installed display driver should also be checked
before using a scientific result based on visual alignment.
