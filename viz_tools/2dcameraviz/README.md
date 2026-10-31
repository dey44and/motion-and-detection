# 📷 2dcameraviz

Explore RGB camera rigs and projected 3D annotations in a desktop window.
`2dcameraviz` runs with NumPy, Pillow and Tkinter, independently of Open3D and
the training stack. It shares [`visual_utils`](../../visual_utils/README.md)
records with the point-cloud and detection viewers.

**[Quick start](#quick-start)** · **[Datasets](#open-a-dataset)** · **[Controls](#controls-and-interpretation)** · **[Extend](#use-your-own-dataset)** · **[Validate](#interpretation-and-validation)**

<a id="quick-start"></a>
## 🚀 Quick start

Use Python 3.10–3.12 and run these commands from the repository root:

```bash
python -m pip install -e './visual_utils' -e './viz_tools/2dcameraviz[dev]'
2dcameraviz --demo
```

The demo generates its own images and annotations, so you can try the controls
before connecting a dataset. Use `python -m cameraviz` as an alternative launcher.
Launching without arguments opens an empty window with dataset controls.

Opening the window requires a graphical display. On Debian/Ubuntu, Tkinter is
supplied by `python3-tk`. The `--help` and `--app-version` commands also work
without a display.

<a id="open-a-dataset"></a>
## 🗂️ Open a dataset

Once you have explored the demo, select a native dataset root:

```bash
2dcameraviz --dataset nuscenes --root /data/nuscenes --version v1.0-mini
2dcameraviz --dataset waymo --root /data/waymo --split val
```

**nuScenes** accepts an extracted mini, trainval or test root, or its release
metadata directory. Native JSON tables and camera keyframe images are required.
LiDAR files are unnecessary. The backend uses each camera's recorded pose and
pinhole calibration, and annotation-free releases remain viewable. Its six
camera views use this layout:

| Front left | Front | Front right |
| --- | --- | --- |
| Back right | Back | Back left |

**Waymo** accepts the dataset parent, `waymo_og_format_unpacked`, or a directory
of native Perception TFRecords under `training/`, `validation/` and `testing/`.
It reads the original camera images without JPEG extraction, TensorFlow, the
Waymo SDK or OpenPCDet. Omit the nuScenes `--version` option. The five views use:

| Front left | Front | Front right |
| --- | --- | --- |
| Side left | | Side right |

Waymo overlays use the native distortion and rolling-shutter model, including
exposure poses and motion. Camera-synchronized boxes override reference-time
boxes where provided, and curved edges are sampled before image clipping.
Processed OpenPCDet clouds and `kitti_format/` trees are not camera-image inputs.
The [native reader guide](../../visual_utils/docs/dataset_readers.md) gives the full layouts
and scientific conventions.

## ⚙️ Configure sources

The root/version arguments cover simple layouts. For named sources or provider
options, save a `DatasetConfig` JSON file such as `waymo.json`:

```json
{
  "storage_format": "native",
  "source_paths": {"native": "/data/waymo/tfrecords"}
}
```

Then open the same viewer with that configuration:

```bash
2dcameraviz --dataset waymo --dataset-config waymo.json --split val
```

The accepted fields are `root`, `release`, `storage_format`, `source_paths` and
`options`. Relative roots resolve against the JSON directory. Relative named
sources resolve against that root, or the JSON directory when there is no root.
Unknown fields and duplicate keys are errors, and each provider validates its
own options.

Use `--dataset-config` on its own instead of explicit root/version flags or
`--demo`. Configured source controls stay fixed for that window, while
navigation and overlay styling remain available.

<a id="controls-and-interpretation"></a>
## 🎛️ Controls and interpretation

Choose a split, scene and frame in the sidebar, then adjust the overlays while
keeping each image's native orientation and aspect ratio. Scene and frame reads
run on one background worker so the controls remain responsive.

| Task | Control or argument |
| --- | --- |
| Select a dataset split | Split menu or `--split`, which defaults to `all`. Available names come from the loaded dataset. |
| Open a scene | Scene menu or `--scene`, accepting its ID or display name. |
| Select a frame | Frame slider or zero-based `--frame`. Left/right arrows step through frames. |
| Set the dataset root | Root field or `--root`, also available as `--dataroot`. |
| Change box presentation | Choose projected 3D wireframes or enclosing 2D rectangles, then set class visibility, colors and line width. |

Camera roles and ordering come from sensor metadata. Familiar roles use the
front/rear/side grid, while unknown roles receive distinct slots in stable
order. A missing camera keeps its placeholder. A camera without an explicit
projection model still displays its image, with no box overlay.

Native class IDs remain intact in observations and projected geometry. Display
groups affect only controls and styles. Status labels distinguish ground truth
that is available but empty, not provided, or withheld.

To interpret an overlay, keep its spatial and temporal references in mind.
Boxes and camera models use the camera set's declared reference coordinates,
and box sizes are length/width/height in metres. Exposure time and annotation
time remain separate, so moving objects can retain a timing mismatch.
Projections do not infer image occlusion or provide independently annotated 2D
boxes. Nonlinear edge sampling is finite, so check numerical calibration
alongside rendered alignment.

<a id="use-your-own-dataset"></a>
## 🔌 Use your own dataset

Implement `visual_utils.DatasetLoader` in your package and return shared
`SceneInfo`, `FrameInfo` and `CameraFrameSet` records. The protected catalog and
camera-loading hooks supply ordered sensors, native class metadata and an
explicit projection model when calibration is known. A camera-only provider
needs no point-cloud methods.

Expose its constructor through your package's `pyproject.toml`:

```toml
[project.entry-points."visual_utils.loaders"]
my_dataset = "my_dataset:MyDatasetLoader"
```

After installation, launch `2dcameraviz --dataset my_dataset --root /data/mine`.
For an in-process script, call `visual_utils.register_loader` before launching.
The shared registry makes a separate camera adapter unnecessary. The name
`demo` is reserved by the viewer. Follow the
[custom-dataset tutorial](../../visual_utils/docs/custom_datasets.md) for a
runnable provider used by all three viewers.

The same loader also supports Python processing without opening a window:

```python
from visual_utils import DatasetConfig, Modality
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
The context manager closes the loader after the synchronous reads finish.
Desktop callers can use
`cameraviz.ui.app.run_app(dataset="my_dataset", config=config, autoload=True)`.
The [architecture guide](docs/architecture.md) explains how these shared records
reach the image panels.

<a id="interpretation-and-validation"></a>
## 🧪 Validation and next reading

Run the generated-data and analytical projection checks from the repository root:

```bash
python -m pytest viz_tools/2dcameraviz/tests
CAMERA_VIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest viz_tools/2dcameraviz/tests
```

Native camera fixtures exercise calibration and projection behavior. The opt-in
Tk checks also exercise overlays, navigation, source replacement and recovery
from failed reads.

For deeper work, continue with the [camera architecture](docs/architecture.md),
[shared dataset interface](../../visual_utils/docs/loader_contract.md) or
[custom-dataset tutorial](../../visual_utils/docs/custom_datasets.md).

Version 0.2 consumes the shared 0.3 records directly. The previous camera models
and adapter API remain in the
[implementation archive](../../archive/visualization_before_simplification/)
for comparison. New integrations use `visual_utils`.
