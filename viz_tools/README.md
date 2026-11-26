# 🔭 Visualization tools

Explore point clouds, 3D detections, and LiDAR segmentation in the browser
with `3dviz`. Review camera images and projected annotations in the separate
`2dcameraviz` desktop viewer. Both use the dataset interface in `visual_utils`.

**[Camera viewer](2dcameraviz/README.md)** ·
**[Point-cloud viewer](3dviz/README.md)** ·
**[Detection review](3dviz/README.md#detection-and-lidar-segmentation)** ·
**[Shared dataset utilities](../visual_utils/README.md)**

## 🧭 Choose a tool

| Tool | What you can inspect | Python package |
| --- | --- | --- |
| [2dcameraviz](2dcameraviz/README.md) | Multi-camera RGB images with projected 3D ground truth. | `cameraviz` |
| [3dviz](3dviz/README.md) | Point clouds, distance or camera colors, acquisition geometry, ground-truth and predicted boxes, and LiDAR segmentation. | `threedviz` |

Each viewer supports nuScenes and Waymo through the shared loaders. Custom
datasets declare which modalities and optional features are available.

## 🚀 Try the demos

Use Python 3.10–3.12. From the repository root, install the shared package and
both viewers:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './visual_utils' \
  -e './viz_tools/2dcameraviz' \
  -e './viz_tools/3dviz'
```

If you already have an environment for a viewer, activate that environment.
Launch either demo:

```bash
3dviz --demo
2dcameraviz --demo
```

The demos generate their observations locally. `3dviz` requires a browser with
WebGL 2. `2dcameraviz` requires a graphical display and Tkinter. Both viewers
install independently of OpenPCDet and CUDA.

## 📂 Open your data

Both commands share dataset selection and configuration. Open a native nuScenes
mini extract in the point-cloud viewer:

```bash
3dviz --dataset nuscenes --root /datasets/nuscenes --version v1.0-mini --split val
```

Use `2dcameraviz` with the same dataset arguments to inspect the cameras.
For layouts with separate sources or provider
options, put the settings in a JSON file and pass `--dataset-config FILE`.
The [configuration guide](../visual_utils/README.md) explains the shared fields.
Run a command with `--help` for its viewer-specific arguments.

Predictions are configured separately from observations. The
[review guide](3dviz/README.md#detection-and-lidar-segmentation) explains supported exports, frame matching,
and the meaning of review filters. RGB, timing, and acquisition displays also
depend on the sources available in your dataset. Read the viewer's
interpretation notes before using an overlay as evidence of alignment.

## 🔌 Connect a custom dataset

Implement one `DatasetLoader` and return shared records for point clouds,
cameras, or both. Register it through the `visual_utils.loaders` entry-point
group to make its name available to the viewer commands.

The [custom dataset tutorial](../visual_utils/docs/custom_datasets.md) includes
a complete file-backed provider, generated sample data, and launch commands for
both viewers. For a custom detector output, add a separate
[prediction source](3dviz/docs/prediction_contract.md).

## 📚 Explore the design

| Topic | Guide |
| --- | --- |
| Shared data records, calibration, and lifecycle. | [Dataset interface](../visual_utils/docs/loader_contract.md) |
| nuScenes and native Waymo storage and decoding. | [Native readers](../visual_utils/docs/dataset_readers.md) |
| Package responsibilities and the data flow. | [Visualization architecture](../docs/VISUALIZATION_ARCHITECTURE.md) |
| Local checks, desktop tests, and packaging. | [Contributing](../CONTRIBUTING.md) |
