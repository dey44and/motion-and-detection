# 3dviz

A standalone desktop viewer for autonomous driving point clouds and 3D ground truth boxes. The left panel selects the dataset, scene and frame and controls appearance; the right panel displays the scene on a white background. Drag with the mouse to rotate and use the scroll wheel to zoom.

## Install and launch

Use Python 3.10–3.12 and a desktop session with an OpenGL-capable display. From the `motion-and-detection` directory:

```bash
python3 -m venv 3dviz/.venv
source 3dviz/.venv/bin/activate
python -m pip install -e './3dviz[dev]'
3dviz --demo
```

The synthetic demo runs without downloading a dataset. Open the dataset connection form with `3dviz`, or load nuScenes directly:

```bash
3dviz --dataroot /path/to/nuscenes --version v1.0-mini
3dviz --root /path/to/nuscenes --version v1.0-trainval --scene scene-0001 --frame 3
python -m threedviz --help
```

`--scene` accepts a scene name or token, and `--frame` is a zero-based frame index. `--root` is an alias for `--dataroot`. `--dataset demo` is equivalent to `--demo`. Launching without a dataset root leaves the connection form ready for input. CLI help does not import Open3D or require a display.

The viewer is installed independently of the parent OpenPCDet project; its runtime dependencies are NumPy, Matplotlib, Open3D and Pillow. A terminal-only session can run the loader, processing code and tests, but opening the window needs a display and working OpenGL drivers.

## Dataset layout and scope

Point `--dataroot` at the extracted nuScenes root, containing the metadata release directory and LiDAR sample files:

```text
nuscenes/
├── v1.0-mini/                  # or v1.0-trainval / v1.0-test
│   ├── scene.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── calibrated_sensor.json
│   ├── ego_pose.json
│   ├── sensor.json
│   ├── category.json
│   ├── instance.json
│   └── sample_annotation.json
└── samples/
    └── LIDAR_TOP/
        └── *.pcd.bin
```

The adapter reads the official JSON tables and `LIDAR_TOP` binary files directly. No nuScenes devkit installation or OpenPCDet preprocessing is required. It loads one LiDAR keyframe at a time and transforms global annotation boxes into the sensor frame using both ego and sensor poses. The internal box size convention is `(length, width, height)`; nuScenes metadata uses `(width, length, height)`.

Annotated mini and trainval releases provide ground truth boxes. The test release can display clouds without boxes because its ground truth annotations are withheld. Ten nuScenes detection classes use their familiar aliases, such as `car` and `pedestrian`; other annotation categories retain their original category names. See the official [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md) and [test annotation policy](https://www.nuscenes.org/tracking?externalData=all).

The viewer displays ground truth from individual LiDAR keyframes and uses camera images for optional RGB point colors. Prediction files, accumulated sweeps, camera image panels and radar visualization are outside the current scope.

## Controls

- Select a scene and frame; previous/next controls step through keyframes.
- Under **Point colors**, choose grayscale, jet or turbo coloring by distance, or **Camera RGB** when the dataset has camera metadata.
- Set the distance color range in metres. Colors use a fixed zero-to-range scale across frames, and distances beyond the selected range use the far-end color.
- Use **Undo ego motion compensation** to inspect the estimated acquisition-time points. Boxes remain at the sweep reference time.
- Toggle bounding boxes by class and select a color for each class.
- Adjust box line width, show or hide axes, and reset the camera.
- Drag to rotate the scene and scroll to zoom.

Dataset and frame loading run on a single background worker. The window stays responsive, and outdated requests do not replace the latest selection. Rendering uses Open3D's [SceneWidget](https://www.open3d.org/docs/release/python_api/open3d.visualization.gui.SceneWidget.html).

## Camera RGB and point timing

Camera RGB is prepared on demand. Install the camera images referenced by `sample_data.json`, including `samples/CAM_*/` and `sweeps/CAM_*/`; the closest exposure can be a non-keyframe image. Distance coloring continues to work with LiDAR files alone. If a required camera file is missing, the viewer reports its path and keeps the previous display.

nuScenes point files contain XYZ, intensity and ring, without recorded per-point timestamps. The viewer follows the **nominal** HDL-32E approach from [NCore's nuScenes converter](https://github.com/NVIDIA/ncore/blob/dde366e7e8e9e7dbb9b1278488936005a7dee4e2/tools/data_converter/nuscenes/converter.py): a clockwise 1085-column model, analytical beam geometry, and 4× column resolution for alignment. It estimates acquisition times from ordered 32-beam firing columns and interpolated ego poses, with no empirical model fitting or multi-frame optimization. The previous LiDAR sweep timestamp supplies the start; the current timestamp supplies the reference/end. Columns use `column / number_of_model_columns`, so the final column precedes the sweep end. Beam firing offsets affect nominal geometry; they do not add individual ring timestamps.

The adapter reverses the existing nuScenes ego compensation using linear translation interpolation and quaternion SLERP. For RGB projection it maps returns into world coordinates, then into each selected camera's own exposure pose and undistorted pinhole image. Per point, the nearest exposure is selected independently for each camera channel. Projection rejects points behind a camera, outside its image, or behind a nearer LiDAR return in the same pixel. Overlapping views prefer the optical axis, then the smaller time gap. RGB sampling is bilinear; unmatched points use distance-based gray, and the sidebar reports coverage. Images are decoded lazily into a bounded cache.

These acquisition times are estimates. Undoing and reapplying the same ego transform cancels for static geometry; timing chiefly improves exposure selection. Moving objects are not compensated, and calibration, nominal scan geometry and sparse occlusion checks limit alignment. nuScenes cameras use global shutter, so no rolling shutter correction is applied. See [NCore's timing and sensor conventions](https://nvidia.github.io/ncore/conversions/nuscenes/nuscenes.html). The viewer retains all points, including scan-edge columns that NCore's conversion can trim.

## Package structure and adding datasets

```text
src/threedviz/
├── dataloader/       # dataset-neutral models, adapter interface and factory
├── processing/       # colors, box geometry, pose interpolation and projection
├── visualization/    # Open3D scene and material management
├── ui/               # window, controls and asynchronous loading
└── cli.py            # launch options, with lazy graphical imports
```

`DatasetAdapter` is the boundary between dataset storage and the viewer. An adapter exposes `list_scenes`, `list_frames`, `load_frame` and `class_names`, returning the common `SceneInfo`, `FrameInfo`, `FrameData` and `BoundingBox3D` models. Points and boxes must share the same coordinate frame, distances use metres, rotations are 3×3 matrices, and box sizes use `(length, width, height)`.

Implement the adapter for another dataset and register its constructor with the factory. Constructors receive `root` and `version` keyword arguments:

```python
from threedviz.dataloader import create_dataset, register_dataset
from my_dataset_adapter import MyDatasetAdapter  # implements DatasetAdapter

register_dataset("my_dataset", MyDatasetAdapter)
dataset = create_dataset("my_dataset", root="/path/to/data", version="release-1")
```

The renderer and UI consume the shared models and do not need dataset-specific changes. Register custom adapters before launching `threedviz.cli.main()` to make their names available in the dataset selector and CLI choices.

Camera and timing support are optional adapter capabilities: override `supports_camera_rgb`, `supports_point_timing` and `prepare_frame(frame, camera_rgb=..., point_timing=...)`. Preparation populates the common frame's `rgb_colors`, `rgb_valid_mask`, `point_timestamps_us` and `raw_points` fields. Existing adapters need no changes to keep distance coloring.

## Tests

```bash
python -m pytest 3dviz/tests
```

Loader, geometry, coloring, pose interpolation, camera projection and CLI tests use synthetic data and do not need the full dataset or a desktop session. Interactive rendering should also be checked with `3dviz --demo` and a real nuScenes release on a machine with a display.

An optional integration test opens the actual native window and checks loading, navigation, rendering, appearance controls and error recovery. Run it on a desktop with `THREEDVIZ_DESKTOP_TESTS=1 python -m pytest 3dviz/tests`, or on Linux with a virtual display:

```bash
THREEDVIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest 3dviz/tests
```
