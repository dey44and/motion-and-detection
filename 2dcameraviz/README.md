# 2dcameraviz

A desktop viewer for RGB camera rigs and projected 3D annotations. It uses
NumPy, Pillow and Tkinter; neither Open3D nor the training stack is required.
The viewer consumes the same [`dataset_utils`](../dataset_utils/README.md)
records as the point-cloud and detection viewers.

## Install and run

Requires Python 3.10–3.12. From the repository root:

```bash
python -m pip install -e './dataset_utils' -e './2dcameraviz[dev]'
2dcameraviz --demo
2dcameraviz --dataset nuscenes --root /data/nuscenes --version v1.0-mini
2dcameraviz --dataset waymo --root /data/waymo --split val
```

Use `python -m cameraviz` as an alternative launcher. On Debian/Ubuntu, Tkinter
is supplied by `python3-tk`. Opening the window requires a graphical display;
`--help` and `--app-version` work without one. Launching without arguments opens
an empty window with dataset controls.

`--scene` selects a scene ID or display name; `--frame` selects its zero-based
display index. `--root` aliases `--dataroot`. `--split` defaults to `all`; the
loaded dataset supplies its available split names. Arrow keys step through
frames. Scene and frame loading run on one background worker.

## Views and annotations

The window preserves each image's native orientation and aspect ratio. Choose
3D wireframes or enclosing 2D rectangles, then control class visibility, colors
and line width. Camera roles and ordering come from dataset sensor metadata.
Known roles use the usual front/rear/side grid; unknown roles receive distinct
slots in stable order. A missing camera keeps its placeholder.

Native class IDs remain intact in observations and projected geometry. Class
display groups affect only the controls and overlay styles. Available-empty,
not-provided and withheld ground truth have distinct status labels. A camera
without an explicit projection model displays its image without box overlays.

## Dataset inputs

**nuScenes:** select an extracted mini, trainval or test root, or its release
metadata directory. Native JSON tables and camera keyframe images are required;
LiDAR files are unnecessary. The backend uses each camera's recorded pose and
pinhole calibration. Annotation-free releases remain viewable. The six cameras
are placed as follows:

| Front left | Front | Front right |
| --- | --- | --- |
| Back right | Back | Back left |

**Waymo:** select the dataset parent, `waymo_og_format_unpacked`, or a directory
of native Perception TFRecords under `training/`, `validation/` and `testing/`.
No JPEG extraction, TensorFlow, Waymo SDK or OpenPCDet installation is required.
Omit the nuScenes `--version` option. The five camera views use:

| Front left | Front | Front right |
| --- | --- | --- |
| Side left | | Side right |

Waymo overlays use the native distortion and rolling-shutter model, including
exposure poses and motion. Camera-synchronized boxes override reference-time
boxes where provided. Curved edges are sampled before image clipping. Processed
OpenPCDet point clouds and `kitti_format/` trees are not camera-image inputs.
See [shared dataset documentation](../dataset_utils/README.md) for layouts and
scientific conventions.

## Configure sources

For named sources or provider options, save a `DatasetConfig` JSON file:

```json
{
  "storage_format": "native",
  "source_paths": {"native": "/data/waymo/tfrecords"}
}
```

```bash
2dcameraviz --dataset waymo --dataset-config waymo.json --split val
```

The JSON accepts `root`, `release`, `storage_format`, `source_paths` and `options`.
Relative roots resolve against the JSON directory; relative named sources
resolve against that root, or the JSON directory if there is no root. Unknown
fields and duplicate keys are errors. Provider-specific options are validated
by the provider. `--dataset-config` excludes explicit root/version flags and
`--demo`; it locks source controls while leaving navigation and styling usable.

## Use your own dataset

Implement `dataset_utils.DatasetLoader` in your package. Return canonical
`SceneInfo`, `FrameInfo` and `CameraFrameSet` records; implement the protected
catalog and camera-loading hooks. Supply ordered sensors, native class metadata
and an explicit projection model when calibration is known. A camera-only
provider needs no point-cloud methods.

Expose the constructor through your package's `pyproject.toml`:

```toml
[project.entry-points."dataset_utils.loaders"]
my_dataset = "my_dataset:MyDatasetLoader"
```

After installation, launch `2dcameraviz --dataset my_dataset --root /data/mine`.
For an in-process script, call `dataset_utils.register_loader` before launching.
There is one shared registry; no camera-specific adapter is needed. The name
`demo` is reserved by the viewer. Follow the
[custom-dataset tutorial](../dataset_utils/docs/custom_datasets.md) for a runnable
provider used by all three viewers.

For Python processing, the camera factory returns the shared loader itself:

```python
from dataset_utils import DatasetConfig, Modality
from cameraviz.dataloader import create_dataset
from cameraviz.processing import project_boxes

with create_dataset("nuscenes", config=DatasetConfig(
    root="/data/nuscenes", release="v1.0-mini",
)) as dataset:
    scene = dataset.list_scenes(split="val")[0]
    info = dataset.list_frames(scene.scene_id, modality=Modality.CAMERAS)[0]
    frame = dataset.load_cameras(info.frame_id)
    overlays = project_boxes(frame)
    print({sensor_id: len(boxes) for sensor_id, boxes in overlays.items()})
```

`create_dataset` accepts either `config=` or convenience root/version arguments.
The caller closes the loader after pending reads finish. Python desktop callers
can use `cameraviz.ui.app.run_app(dataset="my_dataset", config=config, autoload=True)`.
The [architecture guide](docs/architecture.md) describes the direct data flow.

## Interpretation and validation

Boxes and camera models use the camera set's declared reference coordinates;
the viewer does not transform them through an invented world frame. Box sizes
are length/width/height in metres. Recorded exposure time and annotation time
remain separate; moving objects can retain a timing mismatch. Projections do
not infer image occlusion or independently annotated 2D boxes. Nonlinear edge
sampling is finite, so inspect numerical calibration as well as rendered images.

```bash
python -m pytest 2dcameraviz/tests
CAMERA_VIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest 2dcameraviz/tests
```

Tests use generated data, analytical projection expectations and native camera
fixtures. Desktop checks exercise the actual Tk window, overlays, navigation,
source replacement and recovery from failed reads.

Version 0.2 consumes the shared 0.3 records directly. The previous duplicate
camera models and adapter API are preserved in
[`archive/visualization_before_simplification`](../archive/visualization_before_simplification/)
for comparison; new integrations use `dataset_utils`.
