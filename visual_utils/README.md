# 🧩 visual-utils

Load a dataset once and use it across the camera, point-cloud, and detection
viewers. `visual_utils` translates nuScenes, Waymo, or your own storage into
shared Python records, with explicit calibration, coordinate references, and
timing. It works in scripts and notebooks without importing a GUI.

**[Visualization tools](../viz_tools/README.md)** ·
**[Dataset interface](docs/loader_contract.md)** ·
**[Custom dataset tutorial](docs/custom_datasets.md)** ·
**[Native readers](docs/dataset_readers.md)**

## 🚀 Get started

Use Python 3.10–3.12. From the repository root, install the shared package into
your active environment:

```bash
python -m pip install -e './visual_utils'
```

For a native nuScenes mini extract, the same loader gives you scene metadata,
a point cloud, and the corresponding camera observations:

```python
from visual_utils import DatasetConfig, Modality, create_loader

config = DatasetConfig(root="/datasets/nuscenes", release="v1.0-mini")
with create_loader("nuscenes", config) as dataset:
    scene = dataset.list_scenes(split="val")[0]
    info = dataset.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD)[0]
    cloud = dataset.load_point_cloud(info.frame_id)
    cameras = dataset.load_cameras(info.frame_id)
    print(cloud.points.shape, len(cameras.cameras))
```

Catalog calls read metadata. Images and point clouds are decoded when requested.
The context manager closes the loader, and returned observations remain usable
after it closes. If you do not have a dataset available, try the
[small generated example](docs/custom_datasets.md) to explore the same interface.

## 🔭 Choose a viewer

Install only the viewer you need alongside `visual_utils`. Run these commands
from the repository root in the environment where you installed the shared
package.

| Tool | Use it for |
| --- | --- |
| [2dcameraviz](../viz_tools/2dcameraviz/README.md) | Camera images and projected 3D ground truth. |
| [3dviz](../viz_tools/3dviz/README.md) | Point clouds, boxes, colors, and acquisition geometry. |
| [3ddetviz](../viz_tools/3ddetviz/README.md) | Predictions, confidence filters, and geometric NMS. |

For example, install and launch the point-cloud viewer:

```bash
python -m pip install -e './viz_tools/3dviz'
3dviz --demo
```

Detection review reuses the 3D renderer, which is why its installation includes
`3dviz`. Each viewer guide provides its installation command, and all three use
the same dataset records.

## 📂 Configure your data

Built-in loaders accept a `DatasetConfig`. Choose the provider and storage
layout that match your data:

| Provider | Storage formats | Named sources | Provider options |
| --- | --- | --- | --- |
| `nuscenes` | `auto`, `native` | `native` | None. Select `v1.0-mini`, `v1.0-trainval`, or `v1.0-test` with `release`. |
| `waymo` | `auto`, `native`, `openpcdet` | `native`, `processed` | `processed_tag`, `additional_classes`. Release may be omitted or set to `1.4.2` / `v1.4.2`. |

Waymo can discover native and processed directories automatically. When they
live separately, provide both paths through `source_paths`. The
[native reader guide](docs/dataset_readers.md) explains the supported layouts,
sensor conventions, and lower-level APIs.

For a configuration you can reuse across viewers, save a JSON file:

```json
{
  "root": "../datasets/waymo",
  "storage_format": "openpcdet",
  "source_paths": {"native": "../waymo-original"},
  "options": {"processed_tag": "default"}
}
```

Pass it to any viewer with `--dataset-config FILE`, or read it in Python with
`load_dataset_config(FILE)`. A relative root resolves against the JSON file's
directory. Relative named sources resolve against that root, or against the
file's directory when the root is absent.

The accepted fields are `root`, `release`, `storage_format`, `source_paths`, and
`options`. Unknown fields, duplicate keys, and non-finite option numbers are
rejected. Each provider also validates its supported settings. See the
[configuration reference](docs/loader_contract.md#configuration-registration-and-errors)
for the full behavior.

## 🎨 Request optional point data

RGB colors, per-point timestamps, and acquisition geometry are independent
features. Request them only when your analysis needs them and the configured
sources support them. For example, after loading a nuScenes cloud:

```python
from visual_utils import PointCloudPreparation

# Reuse the nuScenes configuration and frame ID from the first example.
with create_loader("nuscenes", config) as dataset:
    cloud = dataset.load_point_cloud(info.frame_id)
    colored = dataset.prepare_point_cloud(cloud, request=PointCloudPreparation(rgb=True))
    print(colored.rgb.valid_mask.sum())
```

Preparation returns a record with the original point order and the requested
derived fields. Validity masks describe coverage, and provenance distinguishes
recorded values from estimates. Check those fields before interpreting a
colored or uncompensated view. Likewise, an empty annotation set differs from
ground truth that was not provided or was withheld.

The [dataset interface](docs/loader_contract.md) defines these conventions.
[3dviz's interpretation guide](../viz_tools/3dviz/README.md#controls-and-scientific-interpretation)
explains how they affect the display.

## 🔌 Add your own dataset

Implement `DatasetLoader` in your own package and return the shared
`PointCloudFrame` and/or `CameraFrameSet` records. Declare the available
modalities, classes, sensors, and optional features in metadata. A point-only
provider can support both 3D viewers, while a camera provider can support the
camera viewer.

For an installed package, expose your constructor through an entry point:

```toml
[project.entry-points."visual_utils.loaders"]
my_dataset = "my_package:MyDatasetLoader"
```

The constructor accepts one `DatasetConfig` and returns a `DatasetLoader`.
Viewer commands discover the name automatically. `available_loaders()` lists
built-in, registered, and installed names without importing plugin code, and
only the selected provider is imported during construction. Existing names
cannot be overridden.

For a script or notebook, `register_loader(name, constructor)` registers a
provider in that process instead. The [custom dataset tutorial](docs/custom_datasets.md)
walks through a complete file-backed implementation and shows both approaches.

## 🧭 Find the right guide

| Your next step | Read |
| --- | --- |
| Understand records, geometry, capabilities, and lifecycle. | [Dataset interface](docs/loader_contract.md) |
| Connect your own storage to the viewers. | [Custom dataset tutorial](docs/custom_datasets.md) |
| Work directly with nuScenes or native Waymo decoding. | [Native dataset readers](docs/dataset_readers.md) |
| Understand which package should own a feature. | [Visualization architecture](../docs/VISUALIZATION_ARCHITECTURE.md) |
| Run checks or contribute a change. | [Contribution guide](../CONTRIBUTING.md) |

To run the shared package's tests:

```bash
python -m pip install -e './visual_utils[dev]'
python -m pytest visual_utils/tests
```

Dataset-specific source references and validation notes live with the
[reader documentation](docs/dataset_readers.md). Upstream licenses and
attribution are preserved in [third_party](third_party/NOTICE.txt) and in built
distributions.
