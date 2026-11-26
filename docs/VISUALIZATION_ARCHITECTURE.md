# 🏗️ Visualization architecture

One dataset adapter connects native storage to shared records. Both
viewers consume those records directly, while detection review adds an
independent prediction source. This guide explains where each responsibility
belongs when you extend the code.

**[Viewer overview](../viz_tools/README.md)** ·
**[Dataset interface](../visual_utils/docs/loader_contract.md)** ·
**[Custom dataset tutorial](../visual_utils/docs/custom_datasets.md)**

The data flows through one shared interface:

```text
nuScenes / Waymo / your storage
              |
        DatasetLoader adapter
              |
         shared records
          /          \
   Tk camera    browser workspace
     viewer     (cloud + predictions + lidarseg)
```

`DatasetLoader` adapts native storage and scientific conventions into the
records a viewer needs. Each dataset implements it once.
The records carry scene, frame, and box information through to the renderer.

## 🗂️ Know where code belongs

| Component | Responsibility |
| --- | --- |
| `visual_utils/dataset.py` | Small public interface, with explicit calls to provider implementation methods. |
| `visual_utils/models.py` | Shared records, shapes, coordinate/time metadata and geometric validation. |
| `visual_utils/nuscenes`, `waymo` | Native decoding, catalogs, calibration, pairing, RGB and acquisition processing. |
| `visual_utils/config.py`, `registry.py` | Configuration and optional name-based construction. |
| `cameraviz` | Camera slots, box projection/clipping, display styles and Tk navigation. |
| `threedviz.web` | Local sessions, frame transfer and the bundled Three.js workspace for point clouds, detections, and segmentation. |
| `threedviz.detection` | Prediction readers, coordinate normalization, filters, BEV NMS, and review presets. |
| `threedviz.processing` | Shared box geometry and point color palettes. |

The nuScenes and processed Waymo readers return shared records directly.
Waymo's format-specific TFRecord/range-image structures remain inside its
decoder because they describe the stored sensor format. Dataset authors and viewers
use the public records from `visual_utils`.

Detection review composes an observation, predictions and a filter result.
The observation remains the shared cloud record, and the caller owns the
loader and prediction source separately. Detection filter changes update
the box layer without replacing the frame packet or camera view.

## 🛤️ Follow a frame through the code

1. Construct a loader with `DatasetConfig`.
2. Read its metadata, scenes and frame catalog.
3. Load a `PointCloudFrame` or `CameraFrameSet`.
4. Request optional point processing if the selected display mode needs it.
5. Render the observation, resolving display labels and colors in the UI.

The public methods call `_list_scenes`, `_list_frames`, `_load_point_cloud`,
`_load_cameras`, `_load_lidar_segmentation`, and optional `_prepare_point_cloud`.
The base class handles closed state, splits, and capabilities,
leaving storage and scientific processing to the provider.

Camera projection works directly in the camera set's declared reference.
Class IDs remain unchanged in the records. Display groups are looked up when
building controls, filtering displayed classes and choosing styles.

## 🧪 Keep checks tied to data correctness

Record constructors validate numerical shapes and rotations. Row-aligned masks,
source availability errors, and processed/native point correspondence checks
help prevent a plausible-looking but wrong visualization.

Waymo TFRecord CRC32C checks belong to its storage format. Prediction caches use
file metadata and small sidecar/schema identities, so opening a valid local
export does not reread the entire file just to hash it. Treat
exports as immutable while open and rebuild the cache after unusual external
storage changes that preserve all file identity metadata.

Providers return read-only records and are tested for row preservation and
correct optional processing. Runtime loading and preparation rely on those
semantics without hashing clouds, annotations, or derived arrays on each call.

## 📏 Preserve scientific meaning and lifetime

Metres, LWH sizes, explicit references and integer-microsecond timestamps are
shared conventions. Current 3D height/distance/BEV operations require Z-up with
horizontal XY. Camera models explicitly handle optical axes. Unknown world
poses, calibration, timing and annotations remain unknown. RGB, scalar point
times and acquisition geometry are independent, with masks and provenance.

Dataset methods are synchronous. The UI owns background scheduling, serializes
reads and waits for pending work before cleanup. Use a context manager in scripts.
Returned observations outlive provider caches and closure.

The previous implementation is preserved in
[`archive/visualization_before_simplification`](../archive/visualization_before_simplification/SNAPSHOT.md).
It records the former Python interfaces for historical reference.

## 🔗 Read further

- [Adapter pattern](https://refactoring.guru/design-patterns/adapter): translate native interfaces behind a small consumer-facing boundary.
- [TFRecord format](https://www.tensorflow.org/tutorials/load_data/tfrecord): length and payload CRCs are storage-integrity checks.

The [dataset interface](../visual_utils/docs/loader_contract.md) gives field
conventions, and the [custom dataset tutorial](../visual_utils/docs/custom_datasets.md)
shows a complete provider. The [prediction interface](../viz_tools/3dviz/docs/prediction_contract.md)
covers independent result loading.
