# 3dviz

A standalone desktop viewer for autonomous driving point clouds and 3D ground truth boxes. The left panel selects the dataset, scene and frame and controls appearance; the right panel displays the scene on a white background. Drag with the mouse to rotate and use the scroll wheel to zoom.

For comparing prediction files against ground truth, use the companion [3ddetviz](../3ddetviz/README.md). It shares this viewer's dataset loading, point controls, geometry and camera navigation, with independent ground-truth and detection layers. The `3dviz` launcher retains its existing class colors and controls.

## Install and launch

Use Python 3.10–3.12 and a desktop session with an OpenGL-capable display. From the `motion-and-detection` directory:

```bash
python3 -m venv 3dviz/.venv
source 3dviz/.venv/bin/activate
python -m pip install -e './dataset_utils' -e './3dviz[dev]'
3dviz --demo
```

The synthetic demo runs without downloading a dataset. Open the dataset connection form with `3dviz`, or load nuScenes directly:

```bash
3dviz --dataroot /path/to/nuscenes --version v1.0-mini
3dviz --root /path/to/nuscenes --version v1.0-trainval --scene scene-0001 --frame 3
python -m threedviz --help
```

`--scene` accepts a scene name or token, and `--frame` is a zero-based frame index. `--root` is an alias for `--dataroot`. `--split` selects a dataset split and defaults to `all`; the sidebar also offers the loaded adapter's available splits. `--dataset demo` is equivalent to `--demo`. Launching without a dataset root leaves the connection form ready for input. CLI help does not import Open3D, TensorFlow or the Waymo SDK, and does not require a display.

The viewer is installed independently of the parent OpenPCDet project. Its rendering dependencies are NumPy, Matplotlib, Open3D and Pillow. The GUI-independent [`dataset-utils`](../dataset_utils/README.md) package in `dataset_utils/` owns nuScenes table reading, calibration, nominal timing, RGB projection and splits, plus native Waymo decoding. Python imports use `dataset_utils`. TensorFlow, either dataset SDK and the detector framework are not runtime requirements. A terminal-only session can run the loader, processing code and tests, but opening the window needs a display and working OpenGL drivers.

## nuScenes input

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

The shared factory opens `NuScenesLoader`, which uses `dataset_utils.nuscenes.NuScenesReader` for official JSON tables and `LIDAR_TOP` binaries. No nuScenes devkit installation or OpenPCDet preprocessing is required. It loads one LiDAR keyframe at a time and transforms global annotation boxes into the sensor frame using both ego and sensor poses. The internal box size convention is `(length, width, height)`; nuScenes metadata uses `(width, length, height)`.

Annotated mini and trainval releases provide ground truth boxes. The test release can display clouds without boxes because its ground truth annotations are withheld. Ten nuScenes detection classes use their familiar aliases, such as `car` and `pedestrian`; other annotation categories retain their original category names. See the official [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md) and [test annotation policy](https://www.nuscenes.org/tracking?externalData=all).

The viewer displays ground truth from individual LiDAR keyframes and uses camera images for optional RGB point colors. Prediction files, accumulated sweeps, camera image panels and radar visualization are outside this launcher's scope.

## Waymo input

Use the supplied dataset parent for the normal OpenPCDet point-cloud path:

```bash
3dviz --dataset waymo --root /home/cristian/Datasets/waymo_1.4.2 --split val
```

Waymo does not use the nuScenes `--version` option; the Version field is hidden for this dataset. The parent root discovers this layout:

```text
waymo_1.4.2/
  waymo_pcdet_workspace/
    ImageSets/
      train_full.txt
      train_1of16.txt
      val.txt
      test.txt
    waymo_processed_data_v0_5_0_full_with_corr/
      segment-<context>_with_camera_labels/
        segment-<context>_with_camera_labels.pkl
        0000.npy
        0001.npy
        ...
  waymo_og_format_unpacked/
    training/*.tfrecord
    validation/*.tfrecord
    testing/*.tfrecord
```

The dataset parent, OpenPCDet workspace and explicit processed-data directory are accepted. Automatic discovery prefers `waymo_processed_data_v0_5_0_full_with_corr`; select a different processed directory explicitly to inspect another preparation. The adapter indexes sequence names and split files, then loads only the selected sequence's metadata, with a bounded cache. It does not load multi-gigabyte aggregate `*_infos_*.pkl` files. Partial directories missing sequence metadata are skipped; the adapter's `warnings` property reports their count and an example path. Missing selected-frame clouds report their path.

NPY files must contain float32 `(N,6)` XYZ, intensity, elongation and NLZ values in the reference vehicle frame. All valid rows are retained, including NLZ points, with original intensity values. Ground-truth boxes use `(x,y,z,length,width,height,yaw[,vx,vy])`; `Vehicle`, `Pedestrian`, `Sign` and `Cyclist` remain separate classes. Vehicle-to-world poses and exact frame timestamps are preserved. See [OpenPCDet's Waymo preparation](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/waymo/waymo_utils.py).

For `train`, `ImageSets/train_full.txt` takes priority over `train.txt`, which can contain a reduced selection. `train_1of16` explicitly selects the reduced split; `train_full`, `val`, `test` and `all` are available where their split files/data exist. Sparse original frame indices select their corresponding NPY filenames rather than being renumbered; the UI frame index is the position in the installed sequence. Initial scene discovery leaves frame counts unknown until that sequence is indexed.

For native clouds, including test data absent from the processed workspace, select the TFRecord directory explicitly:

```bash
3dviz --dataset waymo \
  --root /home/cristian/Datasets/waymo_1.4.2/waymo_og_format_unpacked --split test
```

Native loading decodes all five LiDARs and both returns, retains the reference vehicle pose, and displays test clouds without unavailable ground truth. When both native and processed data exist under a parent root, processed data remains the default. A malformed processed dataset reports its error rather than silently changing representations. The legacy `Waymo/{train,val,test}/{lidar,annos}` frame-pickle format and the KITTI conversion are separate representations and are not directly supported by this adapter.

### Waymo Camera RGB

Select **Camera RGB** under **Point colors**. Original TFRecords supply camera JPEGs and the native point-to-image associations for each LiDAR return. Processed data is paired with native data by context, original frame index and exact timestamp. Before colors are attached, the adapter checks point count and ordering and requires each XYZ component to agree within `0.02` metres. Reordered or filtered preprocessing reports a correspondence error instead of attaching unrelated colors. Native clouds use their own decoded point associations directly.

The paired TFRecords must remain under `waymo_og_format_unpacked/{training,validation,testing}` beside the workspace. Missing associations use the normal gray fallback, with coverage reported in the sidebar. On the supplied dataset's first validation frame, the verified result is `72,480 / 126,504` colored points; camera coverage varies by frame. Camera RGB is prepared only when requested for processed clouds.

Waymo range images provide TOP per-pixel poses, which the shared decoder uses when producing reference-frame points. Other LiDARs lack the same acquisition pose information. This viewer currently disables **Undo ego motion compensation** for Waymo and does not invent scalar point timestamps or present partial acquisition-time coordinates as a complete scan. No extra conversion into NCore format is required; [NCore's Waymo conversion guide](https://nvidia.github.io/ncore/conversions/waymo/waymo.html) provides useful coordinate-convention background.

## Controls

- Select a split, scene and frame; previous/next controls step through frames.
- Under **Point colors**, choose grayscale, jet or turbo coloring by distance, or **Camera RGB** when the dataset supports RGB point colors.
- Set the distance color range in metres. Colors use a fixed zero-to-range scale across frames, and distances beyond the selected range use the far-end color.
- Where supported, use **Undo ego motion compensation** to inspect acquisition-time points. Boxes remain at the sweep reference time. This control is disabled for Waymo.
- Toggle bounding boxes by class and select a color for each class.
- Adjust box line width, show or hide axes, and reset the camera.
- Drag to rotate the scene and scroll to zoom.

Dataset and frame loading run on a single background worker. The window stays responsive, and outdated requests do not replace the latest selection. Rendering uses Open3D's [SceneWidget](https://www.open3d.org/docs/release/python_api/open3d.visualization.gui.SceneWidget.html).

Coordinate axes are a fixed 3-metre reference at the cloud reference origin: X red, Y green, Z blue. Toggling them preserves their size and the current camera; they do not affect camera fitting. The same axes are used in `3ddetviz`.

## Camera RGB and point timing

For nuScenes, Camera RGB is prepared on demand. Install the camera images referenced by `sample_data.json`, including `samples/CAM_*/` and `sweeps/CAM_*/`; the closest exposure can be a non-keyframe image. Distance coloring continues to work with LiDAR files alone. If a required camera file is missing, the viewer reports its path and keeps the previous display.

nuScenes point files contain XYZ, intensity and ring, without recorded per-point timestamps. The viewer follows the **nominal** HDL-32E approach from [NCore's nuScenes converter](https://github.com/NVIDIA/ncore/blob/dde366e7e8e9e7dbb9b1278488936005a7dee4e2/tools/data_converter/nuscenes/converter.py): a clockwise 1085-column model, analytical beam geometry, and 4× column resolution for alignment. It estimates acquisition times from ordered 32-beam firing columns and interpolated ego poses, with no empirical model fitting or multi-frame optimization. The previous LiDAR sweep timestamp supplies the start; the current timestamp supplies the reference/end. Columns use `column / number_of_model_columns`, so the final column precedes the sweep end. Beam firing offsets affect nominal geometry; they do not add individual ring timestamps.

The adapter reverses the existing nuScenes ego compensation using linear translation interpolation and quaternion SLERP. For RGB projection it maps returns into world coordinates, then into each selected camera's own exposure pose and undistorted pinhole image. Per point, the nearest exposure is selected independently for each camera channel. Projection rejects points behind a camera, outside its image, or behind a nearer LiDAR return in the same pixel. Overlapping views prefer the optical axis, then the smaller time gap. RGB sampling is bilinear; unmatched points use distance-based gray, and the sidebar reports coverage. Images are decoded lazily into a bounded cache.

These acquisition times are estimates. Undoing and reapplying the same ego transform cancels for static geometry; timing chiefly improves exposure selection. Moving objects are not compensated, and calibration, nominal scan geometry and sparse occlusion checks limit alignment. nuScenes cameras use global shutter, so no rolling shutter correction is applied. See [NCore's timing and sensor conventions](https://nvidia.github.io/ncore/conversions/nuscenes/nuscenes.html). The viewer retains all points, including scan-edge columns that NCore's conversion can trim.

## Package structure and adding datasets

```text
src/threedviz/
├── dataloader/       # frontend adapter interface/factory and shared-model exports
├── processing/       # point colors/box geometry; compatibility exports for shared math
├── visualization/    # Open3D scene and material management
├── ui/               # window, controls and asynchronous loading
└── cli.py            # launch options, with lazy graphical imports
```

New datasets implement `dataset_utils.DatasetLoader` and register their
constructor once with `dataset_utils.register_loader`. The implemented
[version-1 loader contract](../dataset_utils/docs/loader_contract.md), available
in `dataset-utils` 0.2, defines metadata, scene/split catalogs, modality-specific
frame lists, point/camera payloads, optional preparation, and errors. A constructor
receives an immutable `DatasetConfig`. Implement the operations advertised by
its capabilities; point-only and camera-only providers are supported.

Both viewers discover the shared registration through generic bridges. Class
IDs remain native in backend records; `ClassInfo.display_group` or
`display_name` supplies the familiar UI labels. Sensor roles and preferred order
supply the camera viewer's layout. There is no second dataset registration or
camera-layout implementation in either application.

Register before calling the chosen viewer's `main(argv)` in the same process.
For example, this launcher registers a working alias for the existing nuScenes
backend; replace `NuScenesLoader` with your own `DatasetLoader` implementation
for another dataset:

```python
from dataset_utils import register_loader
from dataset_utils.nuscenes.loader import NuScenesLoader
from threedviz.cli import main

register_loader("local_nuscenes", NuScenesLoader)
raise SystemExit(main([
    "--dataset", "local_nuscenes", "--root", "/path/to/nuscenes",
    "--version", "v1.0-mini", "--split", "val",
]))
```

For programmatic access, the frontend factory accepts the common configuration
through `config=`. Explicit sources can live in separate directories:

```python
from dataset_utils import DatasetConfig
from threedviz.dataloader import create_dataset

adapter = create_dataset("waymo", config=DatasetConfig(
    release="1.4.2",
    storage_format="openpcdet",
    source_paths={
        "processed": "/data/waymo/processed",
        "native": "/data/waymo/tfrecords",
    },
))
try:
    print([scene.name for scene in adapter.list_scenes_for_split("val")])
finally:
    adapter.close()
```

`root`, `release`, `storage_format`, named `source_paths`, and provider-declared
`options` belong to `DatasetConfig`; providers validate their supported values.
This Python API is separate from the launcher's root/version form. RGB,
acquisition geometry, and scalar point timestamps are independent capabilities.
The generic bridge enables full acquisition-time display only when full geometry
is supported, and preserves the baseline cloud while preparing optional data.

Normalized extension records are exported from `dataset_utils` and defined in
`dataset_utils.contracts.models`. The older `dataset_utils.models` records are
compatibility types: `threedviz.dataloader.FrameData` still aliases that module's
legacy `PointCloudFrame`, and the generic bridge converts normalized records for
the existing renderer and detection viewer. New providers return normalized
contract records, not these legacy frame types.

The frontend `DatasetAdapter` and `register_dataset` API remains available for
legacy integrations, demo data, and local fixtures. Existing reader and
processing imports also remain compatibility exports. Dataset parsing,
calibration, timing, RGB preparation, and split handling live in the shared
backends; the frontend owns rendering, controls, and worker scheduling.

## Tests

```bash
python -m pytest 3dviz/tests
```

Loader, geometry, coloring, pose interpolation, camera projection and CLI tests use synthetic data and do not need the full dataset or a desktop session. Interactive rendering should also be checked with `3dviz --demo` and a real nuScenes release on a machine with a display.

An optional integration test opens the actual native window and checks loading, navigation, rendering, appearance controls and error recovery. Run it on a desktop with `THREEDVIZ_DESKTOP_TESTS=1 python -m pytest 3dviz/tests`, or on Linux with a virtual display:

```bash
THREEDVIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest 3dviz/tests
```
