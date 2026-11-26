# 🔭 Visualization tools

Use `3dviz` to inspect point clouds, 3D detections, LiDAR segmentation, and
camera images in one browser workspace. It uses the dataset interface in
`visual_utils` for nuScenes, Waymo, and custom providers.

**[3dviz guide](3dviz/README.md)** ·
**[Camera review](3dviz/README.md#camera-images)** ·
**[Detection review](3dviz/README.md#detection-and-lidar-segmentation)** ·
**[Shared dataset utilities](../visual_utils/README.md)**

## 🚀 Try the demo

Use Python 3.10–3.12. From the repository root, install the shared package and
viewer, then launch the demo.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './visual_utils' -e './viz_tools/3dviz'
3dviz --demo
```

The demo generates its observations locally. `3dviz` requires a browser with
WebGL 2 and installs independently of OpenPCDet and CUDA.

## 📂 Open your data

Open a native nuScenes mini extract in the workspace.

```bash
3dviz --dataset nuscenes --root /datasets/nuscenes --version v1.0-mini --split val
```

Click a calibrated camera preview in the 3D scene to review images with
ground truth, filtered predictions, or projected LiDAR points.
[Camera review guide](3dviz/README.md#camera-images)

For layouts with separate sources or provider options, put the settings in a
JSON file and pass `--dataset-config FILE`. Run `3dviz --help` for the arguments.
[Configuration guide](../visual_utils/README.md)

Predictions are configured separately from observations. The
[review guide](3dviz/README.md#detection-and-lidar-segmentation) explains supported exports, frame matching,
and the meaning of review filters. RGB, timing, and acquisition displays also
depend on the sources available in your dataset. Read the viewer's
interpretation notes before using an overlay as evidence of alignment.

## 🔌 Connect a custom dataset

Implement one `DatasetLoader` and return shared records for point clouds,
cameras, or both. Register it through the `visual_utils.loaders` entry-point
group to make its name available to `3dviz`.

The [custom dataset tutorial](../visual_utils/docs/custom_datasets.md) includes
a complete file-backed provider, generated sample data, and a launch command.
For a custom detector output, add a separate
[prediction source](3dviz/docs/prediction_contract.md).

## 📚 Explore the design

| Topic | Guide |
| --- | --- |
| Shared data records, calibration, and lifecycle. | [Dataset interface](../visual_utils/docs/loader_contract.md) |
| nuScenes and native Waymo storage and decoding. | [Native readers](../visual_utils/docs/dataset_readers.md) |
| Package responsibilities and the data flow. | [Visualization architecture](../docs/VISUALIZATION_ARCHITECTURE.md) |
| Local checks and browser tests. | [3dviz validation](3dviz/README.md#validation) |
