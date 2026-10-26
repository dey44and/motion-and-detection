# 2dcameraviz

A standalone desktop viewer for autonomous driving camera images and projected
ground truth bounding boxes. Its window runs independently of `3dviz`, using
NumPy, Pillow and Python's Tkinter GUI. Both viewers use the GUI-independent
[`dataset-utils`](../dataset_utils/README.md) package for nuScenes tables,
calibration and native Waymo data. Python imports use `dataset_utils`.

## Install and run

Requires Python 3.10–3.12. From the `motion-and-detection` repository root:

```bash
python -m pip install -e './dataset_utils' -e './2dcameraviz[dev]'
2dcameraviz --demo
2dcameraviz --dataroot /path/to/nuscenes --version v1.0-mini
```

This installs the camera viewer and its shared data reader; `3dviz` and Open3D
are not required. Use a separate virtual environment for each viewer if you
want their installed dependency versions to remain independent.

Tkinter comes with many Python installations. On Debian/Ubuntu, install it with
`sudo apt install python3-tk`. A graphical desktop is required to open the
window; `2dcameraviz --help` and `--app-version` also work without a display.

Launching `2dcameraviz` without arguments opens the window ready to connect a
dataset. You can also launch it as `python -m cameraviz`.

```bash
2dcameraviz --root /path/to/nuscenes --version v1.0-trainval \
  --scene scene-0001 --frame 3
```

`--root` aliases `--dataroot`. `--scene` accepts a scene name or token, and
`--frame` selects a zero-based frame index. `--demo` provides synthetic images
and boxes without downloading a dataset. `--split` defaults to `all`; the loaded
adapter determines the split choices available in the menu. CLI help does not
import Tk, Open3D, TensorFlow or the Waymo SDK.

## Window and overlays

The left menu selects scenes and frames and controls class visibility, box
colors, and line width in screen pixels. Choose projected 3D wireframes or
enclosing 2D rectangles. nuScenes images appear on the right in this layout:

| Front left | Front | Front right |
| --- | --- | --- |
| Back right | Back | Back left |

All images keep their original orientation. Scene/frame loading runs in a
background worker to keep the controls responsive. Shared sensor metadata
supplies camera roles and preferred order; the generic bridge derives the
layout, including Waymo's five-view arrangement below.

## Waymo input

Select the supplied dataset parent or its `waymo_og_format_unpacked` directory:

```bash
2dcameraviz --dataset waymo --root /home/cristian/Datasets/waymo_1.4.2 --split val
2dcameraviz --dataset waymo \
  --root /home/cristian/Datasets/waymo_1.4.2/waymo_og_format_unpacked --split test
```

Waymo uses the original Perception TFRecords under `training/`, `validation/`
and `testing/`. No preprocessing or JPEG extraction is required. The adapter
indexes frame timestamps lazily, then reads the selected frame's camera images,
calibrations and annotations. `train`, `val`, `test` and `all` select available
native splits. The nuScenes Version field is hidden for Waymo; omit `--version`.
TensorFlow, the Waymo SDK, OpenPCDet and NCore are not runtime requirements.

Waymo has five camera views. The lower middle grid cell is left unused:

| Front left | Front | Front right |
| --- | --- | --- |
| Side left | | Side right |

The side cameras keep their correct names and are not presented as rear cameras.
Original JPEG orientation and dimensions are preserved. Native test frames can
display images without unavailable 3D annotations. The legacy frame-pickle
`Waymo/` tree and `kitti_format/` are not camera inputs for this adapter.

Waymo overlays use the native lens-distortion model and camera pose, velocity,
trigger/readout times and rolling-shutter direction. This follows the
[official Waymo camera model](https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/wdl_limited/camera/camera_model.cc),
with the camera's forward/left/up axes converted explicitly to optical axes.
Curved projected edges are sampled before clipping into the image. A bare
pinhole projection is insufficient for these original JPEGs.

When the native annotation provides a camera-synchronized box, that box is used
for its designated most-visible camera. Other views use the reference-time
3D annotation and retain any object-motion timing difference. Projected boxes
do not estimate image occlusion. These are projected LiDAR annotations, rather
than separately annotated 2D camera boxes.

[NCore's Waymo guide](https://nvidia.github.io/ncore/conversions/waymo/waymo.html)
describes the coordinate conventions and an alternative conversion workflow.
This viewer reads the original TFRecords directly through the shared
`dataset-utils` package; conversion into NCore format is unnecessary.

## nuScenes input

Use an extracted nuScenes mini, trainval, or test release with its native JSON
tables and camera keyframe images. You may select either the dataset root or
the release metadata directory:

```text
nuscenes/
  v1.0-mini/
    scene.json
    sample.json
    sample_data.json
    sensor.json
    calibrated_sensor.json
    ego_pose.json
    sample_annotation.json    # optional ground truth
    instance.json             # optional ground truth category lookup
    category.json             # optional ground truth category lookup
  samples/
    CAM_FRONT/
    CAM_FRONT_LEFT/
    CAM_FRONT_RIGHT/
    CAM_BACK/
    CAM_BACK_LEFT/
    CAM_BACK_RIGHT/
```

LiDAR files are not needed. Images load when a frame is selected. Releases
without ground truth, including the public test split, can still display
camera images.

Boxes are transformed from nuScenes world coordinates into each camera's
optical coordinates using that image's actual recorded ego pose, calibrated
sensor transform, and pinhole intrinsics. The viewer projects the resulting
3D box edges and derives the optional 2D enclosure from the visible projection.
These are projected 3D annotations, rather than independent 2D annotations.
The shared factory opens `NuScenesLoader`, which uses the native nuScenes
reader for table/image reading and calibration. Shared sensor metadata supplies
the rig labels and view roles. A camera-only extract does not require
LiDAR files or the 3dviz package.

nuScenes annotations refer to the sample timestamp while individual camera
exposures can occur at different times. Using each camera's actual pose accounts
for ego motion; moving objects can retain a timing mismatch because annotation
motion is not inferred. Box wireframes show geometry and do not estimate
occlusion by surfaces in the image.

## Package structure and other datasets

```text
src/cameraviz/
  dataloader/      # frontend camera models, thin reader adapters and factory
  processing/     # dataset-independent box projection/clipping and geometry
  visualization/  # image fitting and overlay presentation
  ui/             # Tk window, controls, image layout, and asynchronous loading
  cli.py          # standalone command-line entry point
```

New datasets implement `dataset_utils.DatasetLoader` and register their
constructor once with `dataset_utils.register_loader`. The implemented
[version-1 loader contract](../dataset_utils/docs/loader_contract.md), available
in `dataset-utils` 0.2, defines metadata, scene/split catalogs, modality-specific
frame lists, payloads, optional preparation, and errors. Constructors receive
an immutable `DatasetConfig`. A camera-only provider needs no point-cloud files
or point-cloud implementation.

The generic camera bridge uses `SensorInfo.view_role` and sensor order to place
views, with a deterministic fallback for unfamiliar roles. Class display groups
or names supply UI labels. Providers return RGB images, declared reference
frames, annotations, and an explicit projection model for each projectable
camera. That model handles native distortion/shutter effects and can accompany
camera-specific annotation overrides. Uncalibrated images remain viewable
without projected boxes. No per-dataset `CameraInfo` layout or frontend
registration is needed.

Register before calling `main(argv)` in the same process. This runnable launcher
uses an alias for the existing nuScenes backend; substitute your own
`DatasetLoader` class when adding another dataset:

```python
from dataset_utils import register_loader
from dataset_utils.nuscenes.loader import NuScenesLoader
from cameraviz.cli import main

register_loader("local_nuscenes", NuScenesLoader)
raise SystemExit(main([
    "--dataset", "local_nuscenes", "--root", "/path/to/nuscenes",
    "--version", "v1.0-mini", "--split", "val",
]))
```

The same backend registration is discoverable by the point viewer when it
supports point clouds. Each application process performs its registration
before launch. Neither application needs dataset-specific UI code.

For programmatic access, pass explicit source locations through the common
configuration to the frontend factory:

```python
from dataset_utils import DatasetConfig
from cameraviz.dataloader import create_dataset

adapter = create_dataset("waymo", config=DatasetConfig(
    release="1.4.2",
    storage_format="native",
    source_paths={"native": "/data/waymo/tfrecords"},
))
try:
    print([scene.name for scene in adapter.list_scenes_for_split("val")])
finally:
    adapter.close()
```

`DatasetConfig` also supports a main `root` and provider-declared `options`;
unknown settings are rejected. The Python `config=` API is separate from the
launcher's root/version form.

Normalized records and the projection protocol are exported from
`dataset_utils` and defined in `dataset_utils.contracts.models`. New backends
return those records. The older `dataset_utils.models` types remain reader and
3D-frontend compatibility records. This viewer keeps its strict display models
in `cameraviz.dataloader.models`; its generic bridge performs the conversion.

`DatasetAdapter`, `CameraInfo`, and `register_dataset` remain available for
legacy integrations, demo data, and fixtures. Shared backends own nuScenes and
Waymo reading, calibration, timing, RGB preparation, and splits. The camera
frontend owns its image grid, overlay styles, Tk window, and worker scheduling;
it does not import Open3D.

## Verification

From the repository root after installing the development dependencies:

```bash
python -m pytest 2dcameraviz/tests
```

Desktop integration checks require Tk and an available display. On Linux they
can run under Xvfb:

```bash
CAMERA_VIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest 2dcameraviz/tests
```

The tests use synthetic images and metadata; they do not require nuScenes or
Waymo downloads. Shared native-reader tests are available under `dataset_utils/tests`.
