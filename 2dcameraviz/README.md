# 2dcameraviz

A standalone desktop viewer for autonomous driving camera images and projected
ground truth bounding boxes. It runs independently of `3dviz`, using NumPy,
Pillow, and Python's Tkinter GUI.

## Install and run

Requires Python 3.10–3.12. From the `motion-and-detection` repository root:

```bash
python -m pip install -e './2dcameraviz[dev]'
2dcameraviz --demo
2dcameraviz --dataroot /path/to/nuscenes --version v1.0-mini
```

Tkinter comes with many Python installations. On Debian/Ubuntu, install it with
`sudo apt install python3-tk`. A graphical desktop is required to open the
window; `2dcameraviz --help` and `--app-version` also work without a display.

Launching `2dcameraviz` without arguments opens the window ready to connect a
dataset. You can also launch it as `python -m cameraviz`.

```bash
2dcameraviz --root /path/to/nuscenes --version v1.0-trainval \
  --scene scene-0001 --frame 3
```

`--root` aliases `--dataroot`. `--scene` accepts a scene name or token, and
`--frame` selects a zero-based frame index. `--demo` provides synthetic images
and boxes without downloading a dataset.

## Window and overlays

The left menu selects scenes and frames and controls class visibility, box
colors, and line width in screen pixels. Choose projected 3D wireframes or
enclosing 2D rectangles. Camera images appear on the right in this fixed layout:

| Front left | Front | Front right |
| --- | --- | --- |
| Back left | Back | Back right |

All images keep their original orientation. Scene/frame loading runs in a
background worker to keep the controls responsive.

## nuScenes input

Use an extracted nuScenes mini, trainval, or test release with its native JSON
tables and camera keyframe images. You may select either the dataset root or
the release metadata directory:

```text
nuscenes/
  v1.0-mini/
    scene.json
    sample.json
    sample_data.json
    sensor.json
    calibrated_sensor.json
    ego_pose.json
    sample_annotation.json    # optional ground truth
    instance.json             # optional ground truth category lookup
    category.json             # optional ground truth category lookup
  samples/
    CAM_FRONT/
    CAM_FRONT_LEFT/
    CAM_FRONT_RIGHT/
    CAM_BACK/
    CAM_BACK_LEFT/
    CAM_BACK_RIGHT/
```

LiDAR files are not needed. Images load when a frame is selected. Releases
without ground truth, including the public test split, can still display
camera images.

Boxes are transformed from nuScenes world coordinates into each camera's
optical coordinates using that image's actual recorded ego pose, calibrated
sensor transform, and pinhole intrinsics. The viewer projects the resulting
3D box edges and derives the optional 2D enclosure from the visible projection.
These are projected 3D annotations, rather than independent 2D annotations.

nuScenes annotations refer to the sample timestamp while individual camera
exposures can occur at different times. Using each camera's actual pose accounts
for ego motion; moving objects can retain a timing mismatch because annotation
motion is not inferred. Box wireframes show geometry and do not estimate
occlusion by surfaces in the image.

## Package structure and other datasets

```text
src/cameraviz/
  dataloader/      # dataset-independent models, adapters, and factory registry
  processing/     # calibrated box projection and geometry
  visualization/  # image fitting and overlay presentation
  ui/             # Tk window, controls, image layout, and asynchronous loading
  cli.py          # standalone command-line entry point
```

To add a dataset, implement `DatasetAdapter` and register its constructor with
`register_dataset`. Supply scenes, frame metadata, camera images/calibrations,
world-coordinate boxes, selectable classes, and a `camera_layout` of
`CameraInfo` entries. The UI and projection code use this shared contract.
Box sizes use length, width, height; camera transforms map optical camera
coordinates into the world. Camera images are RGB NumPy arrays.

## Verification

From the repository root after installing the development dependencies:

```bash
python -m pytest 2dcameraviz/tests
```

Desktop integration checks require Tk and an available display. On Linux they
can run under Xvfb:

```bash
CAMERA_VIZ_DESKTOP_TESTS=1 xvfb-run -a python -m pytest 2dcameraviz/tests
```

The tests use synthetic images and metadata; they do not require a nuScenes
download.
