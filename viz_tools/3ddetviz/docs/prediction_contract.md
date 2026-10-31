# 🧩 Prediction sources

Connect model outputs to detection review through exact dataset frame IDs.
A `visual_utils.DatasetLoader` supplies the observation, a `PredictionSource`
supplies detections, and `compose_review` joins them. `refilter` changes policy
while preserving the observation and original predictions.

**[Python example](#predictions-produced-directly-by-python) · [Records](#records-and-coordinates) · [Source lifetime](#source-interface-and-lifetime) · [File importers](#file-importers)**

<a id="predictions-produced-directly-by-python"></a>

## 🐍 Review predictions from Python

Start with `InMemoryPredictionSource` when your model already produces Python
records. This complete CPU example uses the generated dataset without opening
a GUI or creating an export file:

```python
import numpy as np

from visual_utils import Box3D, Modality
from threedviz.dataloader import create_dataset
from detviz.controller import compose_review
from detviz.predictions import (
    Detection, InMemoryPredictionSource, PredictionFrame, PredictionInfo,
)
from detviz.processing import FilterConfig

with create_dataset("demo") as dataset:
    scene = dataset.list_scenes()[0]
    frames = dataset.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD)
    frame_id = frames[0].frame_id
    observation = dataset.load_point_cloud(frame_id)
    box = Box3D("model:0", "car", [5., 0., 1.], [4., 2., 1.5], np.eye(3))

    with InMemoryPredictionSource(
        [PredictionFrame(frame_id, (Detection(box, 0.9, "model:0"),))],
        info=PredictionInfo("My model", "memory", "lidar", stage="pre_nms"),
    ) as predictions:
        review = compose_review(
            observation,
            source=predictions,
            config=FilterConfig(score_threshold=0.5),
        )
        print(review.filtered.output_count)  # 1
        assert review.observation is observation
```

The source eagerly consumes its iterable, so choose an indexed source for large
exports. Prediction and dataset frame IDs must agree exactly. The renderer
receives the original `PointCloudFrame` with separate ground-truth and prediction
box layers.

<a id="records-and-coordinates"></a>

## 📐 Records and coordinates

The following records define the boundary between an importer, the review
controller and the renderer:

| Record | Meaning |
| --- | --- |
| `PredictionInfo` | Source name, format, coordinate frame, processing stage and provenance metadata. |
| `PredictionFrame.sample_token` | Exact `FrameInfo.frame_id`. The historical name also supports arbitrary dataset IDs. |
| `PredictionFrame.detections` | Immutable tuple. An empty tuple means this frame is covered and has zero saved detections. |
| `Detection.box` | Canonical `visual_utils.Box3D` with a geometric center in metres, positive LWH dimensions and a proper rotation from box axes into the declared frame. |
| `Detection.score` | Finite confidence in `[0, 1]`. Importers convert logits explicitly. |
| `Detection.identifier` | Nonempty, stable ID that is unique within the frame. |
| `Detection.velocity` | Optional XYZ vector in metres per second, using the same axes as the box center. |

`Box3D` owns geometry validation and immutable arrays. `Detection` adds score,
identity and optional velocity validation without copying a validated box.
Source `class_id` values stay intact, and the viewer maps them to metadata
display labels. Importers own any mapping from custom model labels. In nuScenes,
results retain `car` while ground truth retains `vehicle.car`, and display groups
align their labels. Per-class thresholds and NMS use the prediction IDs.
Unknown prediction class IDs remain selectable as additional classes. Per-class
settings therefore remain tied to the source's class vocabulary.

The coordinate declaration determines how a prediction joins the observation:

| Declaration | Interpretation |
| --- | --- |
| `lidar` | The observation's reference coordinates, including for non-LiDAR sources. These predictions need no transformation or reconstruction. |
| `world` | The dataset world frame. A real `PointCloudFrame.T_world_from_reference` is required. Normalization applies its inverse to centers and rotations, and rotates velocities. |

Normalization leaves sizes and scores unchanged. Missing world poses cannot be
replaced with identity transforms. Distance, height and BEV NMS assume horizontal
XY and +Z upward. To give those controls their usual driving interpretation,
convert clouds, boxes, calibration and predictions consistently into a Z-up
reference. The viewer does not infer an up axis.

<a id="source-interface-and-lifetime"></a>

## 🔌 Source interface and lifetime

Implement `PredictionSource` with stable metadata and these operations:

| Member | Contract |
| --- | --- |
| `info` | A `PredictionInfo`. Coordinates are `world` or `lidar`, and stage is `pre_nms`, `post_nms` or `unknown`. |
| `frame_tokens` | A `frozenset[str]` of covered dataset IDs, including explicitly empty frames. |
| `class_names` | Native class IDs in stable order. The historical property name is retained. |
| `get_frame(id)` | Return the matching `PredictionFrame`, or `None` only for an absent ID. Raise `PredictionError` for unavailable or corrupt data. |
| `close()` | Release resources, allow repeated calls and reject later reads. Metadata may remain readable. |

The caller owns the loader and source independently through their context
managers. `ReviewFrame` owns neither. The desktop serializes source reads and
filtering on its loading worker, while the UI can read stable metadata. It
closes replaced sources after their pending work is finished.

Programmatic callers must also finish pending reads before closing a source.
Built-in SQLite sources support concurrent queries through short-lived read-only
connections. Custom sources do not need to support concurrent reads.

<a id="file-importers"></a>

## 📂 Add a file importer

Use a file importer when predictions should be opened through the CLI or shared
as exports. An opener follows the public `PredictionOpener` protocol. It accepts
an existing `Path` and the keyword-only arguments `cache_dir=None`,
`progress=None` and `cancelled=None`, then returns a `PredictionSource`.

In your launcher, register the opener before calling `main()` so its format is
available to the CLI parser:

```python
from detviz.predictions import register_prediction_format
from detviz.cli import main
from my_project.predictions import open_predictions

register_prediction_format("local", open_predictions)
raise SystemExit(main([
    "--dataset", "my_dataset", "--dataset-config", "dataset.json",
    "--split", "holdout", "--predictions", "results.local",
    "--prediction-format", "local",
]))
```

Install or register the dataset independently through the
[shared extension interface](../../../visual_utils/docs/custom_datasets.md).
Format names are case-insensitive. Duplicates, `auto` and built-in aliases are
reserved. Automatic suffix detection recognizes built-in JSON and pickle
formats, while custom formats need their registered name.

Progress callbacks receive a fraction in `[0, 1]` and a short message. During a
lengthy import, poll the optional cancellation callback, raise
`PredictionCancelled` when requested, and remove unpublished temporary
artifacts. If you cache imports, include file revision, import options, relevant
metadata and cache schema in the identity, then publish completed caches
atomically. Built-in local caches use file metadata rather than scanning the
full export on every launch. Finish writing exports before opening them.

## 🧪 Validate your source

Exercise exact and absent IDs, empty frames, duplicate IDs, class mapping,
stage provenance, nonidentity world transforms, malformed geometry and repeated
cleanup. Indexed sources also need cancellation and cache-invalidation checks.
Keep exported thresholds and candidate limits with the data, because viewer
filtering cannot recover removed boxes or establish benchmark performance.

Return to the [viewer guide](../README.md) for file schemas, review controls and
NMS conventions. The [dataset extension guide](../../../visual_utils/docs/custom_datasets.md)
explains how the same provider supplies observations to all three viewers.
