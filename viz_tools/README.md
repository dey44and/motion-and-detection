# 🔭 Visualization tools

Explore camera images, point clouds, and 3D detections with three focused
desktop viewers. They share the dataset interface in `visual_utils`, so a
dataset adapter can serve every viewer that supports its modalities.

**[Camera viewer](2dcameraviz/README.md)** ·
**[Point-cloud viewer](3dviz/README.md)** ·
**[Detection review](3ddetviz/README.md)** ·
**[Shared dataset utilities](../visual_utils/README.md)**

## 🧭 Choose a tool

| Tool | What you can inspect | Python package |
| --- | --- | --- |
| [2dcameraviz](2dcameraviz/README.md) | Multi-camera RGB images with projected 3D ground truth. | `cameraviz` |
| [3dviz](3dviz/README.md) | Point clouds, ground truth, distance or camera colors, and acquisition geometry. | `threedviz` |
| [3ddetviz](3ddetviz/README.md) | Ground truth alongside predictions, with class/score filters and geometric NMS. | `detviz` |

Each viewer supports nuScenes and Waymo through the shared loaders. Custom
datasets declare which modalities and optional features are available. Detection
review uses `3dviz` for point-cloud rendering and adds its own prediction sources
and review controls.

## 🚀 Try the demos

Use Python 3.10–3.12 and an environment with desktop display support. From the
repository root, install the shared package and the viewers you want to use.
To try all three:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './visual_utils' \
  -e './viz_tools/2dcameraviz' \
  -e './viz_tools/3dviz' \
  -e './viz_tools/3ddetviz'
```

If you already have an environment for a viewer, activate that environment and
skip creating another one. Then launch any demo:

```bash
2dcameraviz --demo
3dviz --demo
3ddetviz --demo
```

The demos generate their observations locally, so no dataset download is needed.
The viewers install independently of the repository's OpenPCDet training stack
and CUDA extensions. Their individual guides include smaller installation
commands, controls, and platform-specific troubleshooting.

## 📂 Open your data

All three commands share dataset selection and configuration. For example,
open a native nuScenes mini extract in the point-cloud viewer:

```bash
3dviz --dataset nuscenes --root /datasets/nuscenes --version v1.0-mini --split val
```

Use `2dcameraviz` or `3ddetviz` with the same dataset arguments to inspect that
dataset in another viewer. For layouts with separate sources or provider
options, put the settings in a JSON file and pass `--dataset-config FILE`.
The [configuration guide](../visual_utils/README.md) explains the shared fields.
Run a command with `--help` for its viewer-specific arguments.

Predictions are configured separately from observations. The
[detection guide](3ddetviz/README.md) explains supported exports, frame matching,
and the meaning of review filters. RGB, timing, and acquisition displays also
depend on the sources available in your dataset. Read the viewer's
interpretation notes before using an overlay as evidence of alignment.

## 🔌 Connect a custom dataset

Implement one `DatasetLoader` and return shared records for point clouds,
cameras, or both. Register it through the `visual_utils.loaders` entry-point
group to make its name available to the viewer commands.

The [custom dataset tutorial](../visual_utils/docs/custom_datasets.md) includes
a complete file-backed provider, generated sample data, and launch commands for
all three viewers. For a custom detector output, add a separate
[prediction source](3ddetviz/docs/prediction_contract.md).

## 📚 Explore the design

| Topic | Guide |
| --- | --- |
| Shared data records, calibration, and lifecycle. | [Dataset interface](../visual_utils/docs/loader_contract.md) |
| nuScenes and native Waymo storage and decoding. | [Native readers](../visual_utils/docs/dataset_readers.md) |
| Package responsibilities and the data flow. | [Visualization architecture](../docs/VISUALIZATION_ARCHITECTURE.md) |
| Local checks, desktop tests, and packaging. | [Contributing](../CONTRIBUTING.md) |
