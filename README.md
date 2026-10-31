# 3D geometry and detection tools

Tools for inspecting autonomous-driving sensor data, ground truth, and 3D
detections. The visualization packages run independently of the repository's
OpenPCDet training stack and CUDA extensions.

**[Visualization tools](viz_tools/README.md)** ·
**[Shared dataset utilities](visual_utils/README.md)** ·
**[Custom dataset tutorial](visual_utils/docs/custom_datasets.md)** ·
**[Contributing](CONTRIBUTING.md)**

| Package | Python import / command | Purpose |
| --- | --- | --- |
| [visual-utils](visual_utils/README.md) | `visual_utils` | GUI-independent dataset contract, nuScenes/Waymo readers, transforms and optional point preparation. |
| [2dcameraviz](viz_tools/2dcameraviz/README.md) | `cameraviz` / `2dcameraviz` | Multi-camera RGB inspection and projected 3D ground truth. |
| [3dviz](viz_tools/3dviz/README.md) | `threedviz` / `3dviz` | Point clouds, ground truth, RGB and acquisition-geometry inspection. |
| [3ddetviz](viz_tools/3ddetviz/README.md) | `detviz` / `3ddetviz` | Prediction review, class/score filters and geometric NMS, using the 3D viewer. |

## 🚀 Install and try

Use Python 3.10–3.12. From the repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './visual_utils' \
  -e './viz_tools/2dcameraviz' \
  -e './viz_tools/3dviz' \
  -e './viz_tools/3ddetviz'
2dcameraviz --demo
3dviz --demo
3ddetviz --demo
```

## 🔌 Bring your own dataset

Implement one `DatasetLoader` in your own package. It adapts native storage to
shared metadata and sensor records used directly by all three viewers. There
is one frame/box model at that boundary. Declare point clouds, cameras, or both
through capabilities.
Register the constructor in a script, or publish an installed
`visual_utils.loaders` entry point so it appears in the viewer commands.
Prediction-file adapters are a separate extension interface.

Start with the [runnable custom-dataset tutorial](visual_utils/docs/custom_datasets.md),
then consult the [loader contract](visual_utils/docs/loader_contract.md) and
[prediction-source contract](viz_tools/3ddetviz/docs/prediction_contract.md).


## Acknowledgments

<img src="docs/open_mmlab.png" align="right" width="20%">

This repository builds on `OpenPCDet`, an open source project for LiDAR-based
3D object detection.
Dataset-specific notices are retained under [visual_utils/third_party](visual_utils/third_party/NOTICE.txt).
