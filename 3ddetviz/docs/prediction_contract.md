# Prediction sources

A dataset supplies canonical observations through `dataset_utils.DatasetLoader`.
A prediction source supplies detections by exact frame ID. `compose_review` joins
one observation and prediction frame; `refilter` changes policy without changing
the observation or original detections. The renderer receives the original
`PointCloudFrame` plus separate ground-truth and prediction box layers.

## Records and coordinates

| Record | Meaning |
| --- | --- |
| `PredictionInfo` | Source name, format, coordinate frame, processing stage and provenance metadata. |
| `PredictionFrame.sample_token` | Exact `FrameInfo.frame_id`; this historical name supports arbitrary dataset IDs. |
| `PredictionFrame.detections` | Immutable tuple; empty means covered with zero saved detections. |
| `Detection.box` | Canonical `dataset_utils.Box3D`, with geometric center in metres, positive LWH dimensions and a proper rotation from box axes into the declared frame. |
| `Detection.score` | Finite confidence in `[0,1]`; importers convert logits explicitly. |
| `Detection.identifier` | Nonempty stable ID unique within the frame. |
| `Detection.velocity` | Optional XYZ vector in metres per second using the same axes as the center. |

`Box3D` owns geometry validation and immutable arrays. `Detection` adds score,
identity and optional velocity validation without copying a validated box.
Native `class_id` values remain intact; the viewer maps them to metadata display
labels. Importers own model-label mapping. Unknown prediction IDs are selectable
as additional classes. Per-class thresholds and NMS groups use native IDs.

`coordinate_frame="lidar"` means the observation's reference, even for a
non-LiDAR source. These predictions need no transformation or reconstruction.
`"world"` requires a real `PointCloudFrame.T_world_from_reference`; normalization
applies its inverse to centers and rotations and rotates velocities. Sizes and
scores remain unchanged. Missing world poses cannot be replaced by identity.

Distance, height and BEV NMS assume horizontal XY and +Z upward. For their usual
driving interpretation, the provider must convert clouds, boxes, calibration
and predictions consistently into a Z-up reference. No up axis is inferred.

## Source interface and lifetime

Implement `PredictionSource` with:

| Member | Contract |
| --- | --- |
| `info` | `PredictionInfo`; coordinates are `world`/`lidar`, stage is `pre_nms`/`post_nms`/`unknown`. |
| `frame_tokens` | `frozenset[str]` of covered dataset IDs, including empty frames. |
| `class_names` | Native class IDs in stable order; the historical property name is retained. |
| `get_frame(id)` | Matching `PredictionFrame`; `None` only for an absent ID. Raise `PredictionError` for unavailable/corrupt data. |
| `close()` | Release resources, safely repeatable; reject later reads. Metadata may remain readable. |

The caller owns the loader and source independently, using their context
managers. `ReviewFrame` owns neither. The desktop serializes source reads and
filtering on its loading worker and closes replaced sources. Finish pending reads
before closing. Built-in SQLite sources also support concurrent queries through
short-lived read-only connections; custom sources need not support concurrency.

## Predictions produced directly by Python

This complete CPU example needs no GUI or file importer:

```python
import numpy as np
from dataset_utils import Box3D, Modality
from threedviz.dataloader import create_dataset
from detviz.controller import compose_review
from detviz.predictions import (
    Detection, InMemoryPredictionSource, PredictionFrame, PredictionInfo,
)
from detviz.processing import FilterConfig

with create_dataset("demo") as dataset:
    scene = dataset.list_scenes()[0]
    frame_id = dataset.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD)[0].frame_id
    observation = dataset.load_point_cloud(frame_id)
    box = Box3D("model:0", "car", [5., 0., 1.], [4., 2., 1.5], np.eye(3))
    with InMemoryPredictionSource(
        [PredictionFrame(frame_id, (Detection(box, 0.9, "model:0"),))],
        info=PredictionInfo("My model", "memory", "lidar", stage="pre_nms"),
    ) as predictions:
        review = compose_review(observation, source=predictions,
                                config=FilterConfig(score_threshold=0.5))
        print(review.filtered.output_count)  # 1
        assert review.observation is observation
```

The in-memory source eagerly consumes its iterable; use an indexed source for
large exports. Detection and dataset frame IDs must agree exactly.

## File importers

An opener implements the public `PredictionOpener` protocol: an existing `Path`,
keyword-only `cache_dir=None`, `progress=None`, and `cancelled=None`, returning a
`PredictionSource`. Register it before launching the CLI in that process:

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

Install/register the dataset independently through the
[shared extension interface](../../dataset_utils/docs/custom_datasets.md).
Names are case-insensitive; duplicates, `auto` and built-in aliases are reserved.
Automatic suffix detection recognizes built-in JSON/pickle formats only.

Progress callbacks receive a fraction in `[0,1]` and a short message. Poll optional
cancellation during lengthy imports, raise `PredictionCancelled` and remove
unpublished temporary artifacts. If caching, include file revision, import
options, relevant metadata and cache schema in the identity. Publish completed
caches atomically. Local built-in caches use file metadata rather than scanning
the full export on every launch; exports must be complete before opening.

Test exact/absent IDs, empty frames, duplicate IDs, class mapping, stage provenance,
nonidentity world transforms, malformed geometry and repeated cleanup. Indexed
sources also need cancellation and invalidation tests. Preserve exported
thresholds and candidate limits: viewer filtering cannot recover removed boxes
or establish benchmark performance.
