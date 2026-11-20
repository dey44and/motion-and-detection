# nuScenes dataset research notes

These notes cover the nuScenes database structure and the geometry needed for research with multiple sensors. The starting dataset is `/home/cristian/andrei-work/motion-and-detection/data/nuscenes/v1.0-mini`. Official schema definitions and devkit behavior establish the dataset conventions; observations about this local copy are identified separately. Sources and local files were checked on 16 November 2026.

## Question 1 Intrinsics for LiDAR and radar

**In nuScenes, `camera_intrinsic` applies only to cameras. LiDAR and radar records contain an empty list, `[]`.** The field exists in their records because every modality uses the same `calibrated_sensor` table. The official schema explicitly defines it as camera calibration and specifies that it is empty for other sensors. [Official schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#calibrated_sensor)

### Calibration records and table relationships

A `calibrated_sensor` record describes a sensor calibrated on a vehicle. `translation` and `rotation` describe its placement relative to the ego vehicle body frame. The camera-specific field describes image projection. [Official tutorial, section 10](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html#10.-calibrated_sensor)

| Field | Meaning |
| --- | --- |
| `token` | Identifier of this calibration record. |
| `sensor_token` | Reference to `sensor.json`, which supplies the channel and modality. |
| `translation` | Sensor origin in the ego frame, in meters, ordered `[x, y, z]`. |
| `rotation` | Sensor orientation relative to the ego frame, as quaternion `[w, x, y, z]`. |
| `camera_intrinsic` | A 3×3 camera matrix, or `[]` for LiDAR/radar. |

The database associates a measurement with its calibration through `sample_data.calibrated_sensor_token`. Follow that reference, then `calibrated_sensor.sensor_token`, to obtain its sensor channel and modality. This relationship is implemented by the devkit when it builds its database indexes. [Official devkit, `__make_reverse_index__`](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)

### What camera intrinsics do

For a point already expressed in camera coordinates, the pinhole camera model gives

$$
Z_C\begin{bmatrix}u\\v\\1\end{bmatrix}
=K\begin{bmatrix}X_C\\Y_C\\Z_C\end{bmatrix},
\qquad
K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.
$$

Here, `f_x` and `f_y` are focal lengths in pixel units; `c_x` and `c_y` are principal-point coordinates. Equivalently, `u = f_x X_C / Z_C + c_x` and `v = f_y Y_C / Z_C + c_y`. Intrinsics map viewing directions to pixels. They do not supply the depth of an arbitrary image pixel. [OpenCV camera model](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

The nuScenes schema states that its camera images are undistorted and rectified. Use the supplied matrix for those released images; image resizing or cropping requires a corresponding update to the pixel-coordinate mapping. [Official schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#calibrated_sensor), [OpenCV camera model](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

### Why point clouds use no camera intrinsic matrix

The released LiDAR and radar files already provide Cartesian point coordinates in their respective sensor frames:

| Modality | Released representation relevant here | `camera_intrinsic` |
| --- | --- | --- |
| Camera | Image pixels; projection of a camera-frame point uses `K`. | 3×3 matrix |
| LiDAR | Binary point records containing `x, y, z, intensity, ring index`. | `[]` |
| Radar | PCD point records containing `x, y, z`, velocities, and other attributes. | `[]` |

The official `LidarPointCloud.from_file` reads five stored values per point and retains the first four in its point-cloud object. `RadarPointCloud.from_file` parses the PCD fields. Neither reader uses a camera intrinsic matrix to reconstruct the released coordinates. [Official point-cloud readers](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py)

An empty `camera_intrinsic` does not mean that a physical range sensor has no internal calibration. The nuScenes paper discusses intrinsic and extrinsic calibration across the sensor suite. For a concrete example of internal calibration, the Velodyne ROS driver's converter handles manufacturer-provided parameters for individual lasers, including angular and distance corrections. Those parameters describe the sensor's measurement process and are different from a camera projection matrix. nuScenes does not publish them in `camera_intrinsic`. [nuScenes paper, Appendix A](https://arxiv.org/html/1903.11027v5#A1), [Velodyne driver calibration converter](https://github.com/ros-drivers/velodyne/blob/ros2/velodyne_pointcloud/scripts/gen_calibration.py)

For ordinary nuScenes point-cloud loading and alignment, start from the supplied coordinates and apply the required frame transformations. Research on raw measurement formation or internal calibration would need additional sensor-model information.

### Extrinsics and projection between sensors

All three modalities need extrinsics to place their coordinate frames relative to the vehicle. For column vectors, nuScenes applies the stored calibration as

$$
p_E=R_{E\leftarrow S}p_S+t_{E\leftarrow S}.
$$

`S` denotes the sensor frame and `E` the ego frame. Build `R` from the stored quaternion and take `t` from `translation`. The inverse mapping is `p_S = R^T (p_E - t)`. The devkit confirms this direction by rotating a point cloud with the stored quaternion and then adding the translation. [Official devkit, `map_pointcloud_to_image`](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)

When projecting a LiDAR or radar point onto a camera image, the full coordinate sequence is

```text
Source sensor point
  -> ego frame at the point-cloud timestamp       source extrinsics
  -> global frame                               source ego pose
  -> ego frame at the image timestamp            inverse camera ego pose
  -> camera frame                               inverse camera extrinsics
  -> image pixel                                camera intrinsics and division by depth
```

This is the sequence implemented by `map_pointcloud_to_image`. **The intrinsic matrix belongs to the destination camera.** The ego poses account for vehicle motion between the measurements; this transformation alone does not compensate for an object's own motion. Points must also have positive camera depth and project inside the image bounds. [Official devkit projection implementation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)

### Observations about the local dataset

The supplied path is the dataset root in this installation. Its metadata tables are one directory deeper:

```text
data/nuscenes/v1.0-mini/
  samples/
  sweeps/
  maps/
  v1.0-mini/
    calibrated_sensor.json
    sensor.json
    sample_data.json
```

Consequently, the devkit arguments for this layout are `version="v1.0-mini"` and `dataroot="/home/cristian/andrei-work/motion-and-detection/data/nuscenes/v1.0-mini"`. The devkit locates the tables by appending the version to the data root. [Official devkit, `table_root`](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)

The local inspection originally found two loading issues. Both checks now pass:

1. `calibrated_sensor.json`, line 208, previously contained `"camera_intrinsic": [f`, which caused a standard JSON parser to fail. The stray character is no longer present, and the current file parses without adjustment.
2. `sensor.json`, line 33, has been corrected to give `LIDAR_TOP` the token `dc8b396651c05aedbb9cdaae573bb567`. This value was verified against `/home/cristian/Datasets/nuScenes/v1.0-mini/sensor.json` on 16 November 2026. It matches the references in all ten LiDAR calibration records.

The initial inspection removed the stray `f` from a string in memory only. The counts below were subsequently confirmed by directly parsing the current files. After the correction to `sensor.json`, all 120 `calibrated_sensor.sensor_token` references resolve, and the complete sensor table matches the reference mini sensor table.

| Modality | Sensor channels | Calibration records | Intrinsic values |
| --- | ---: | ---: | --- |
| Camera | 6 | 60 | All 60 are 3×3 matrices. |
| LiDAR | 1 | 10 | All 10 are `[]`. |
| Radar | 5 | 50 | All 50 are `[]`. |

All 31,206 `sample_data.calibrated_sensor_token` references resolve to one of the 120 calibration records. These checks cover the three inspected tables, rather than establishing the integrity of the entire dataset.

For example, the local `RADAR_FRONT` calibration with token `f4d2a6c281f34a7eb8bb033d82321f79` contains `translation: [3.412, 0.0, 0.5]`, a rotation quaternion, and `camera_intrinsic: []`. The local `LIDAR_TOP` calibration with token `a183049901c24361a6b0b11b8013137c` also has `camera_intrinsic: []`.

## Question 2 Vehicle poses and the global frame

**`ego_pose.json` records the position and orientation of the ego vehicle at particular timestamps, expressed in the global coordinate system of the log's map.** Here, the ego vehicle is the vehicle carrying the sensors. Its pose allows measurements taken from a moving platform to be placed in a shared map frame. [Official schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#ego_pose)

### What translation and rotation describe

The ego-frame origin is the midpoint of the vehicle's rear axle. Its position and orientation change as the vehicle moves through the map. [nuScenes paper, Appendix A](https://arxiv.org/html/1903.11027v5#A1)

| Field | Meaning |
| --- | --- |
| `token` | Identifier used by `sample_data.ego_pose_token` to select this pose. |
| `translation` | Position of the ego-frame origin in global map coordinates, in meters, ordered `[x, y, z]`. |
| `rotation` | Orientation of the ego frame relative to the global frame, as a quaternion ordered `[w, x, y, z]`. |
| `timestamp` | Time associated with the pose. The devkit treats nuScenes timestamps as microseconds since the Unix epoch. |

Translation is a position in the map, rather than a displacement since the preceding record. Rotation is the vehicle's orientation in that map, rather than a rotation increment. The quaternion components are not Euler angles. [Official schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#ego_pose), [Official devkit timestamp handling](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py)

The schema describes localization in the x–y plane and specifies that translation's `z` is always zero. Treat that value as part of the dataset's localization convention, rather than a measured geographic altitude. It does not imply a yaw-only quaternion: the local example below has nonzero quaternion x and y components. [Official schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#ego_pose)

### How vehicle pose differs from sensor calibration

| Table | What its rotation and translation describe | Direct transformation |
| --- | --- | --- |
| `calibrated_sensor` | The placement of a sensor relative to the vehicle, for the applicable calibration record. | Sensor frame → ego frame |
| `ego_pose` | The position and orientation of the vehicle in the map at a timestamp. | Ego frame → global frame |

For column vectors, the second transformation is

$$
p_G=R_{G\leftarrow E}(t)\,p_E+t_{G\leftarrow E}(t).
$$

`p_E` is a point expressed in ego coordinates. `p_G` is the same point expressed in global coordinates. Construct `R` from the pose quaternion and use the pose translation as `t`. To go from global coordinates back into ego coordinates, use `p_E = R^T (p_G - t)`. This direction is confirmed by the devkit's direct rotation and translation when moving a cloud from ego to global. [Official devkit, `map_pointcloud_to_image`](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)

Combining calibration and vehicle pose gives

$$
T_{G\leftarrow S}(t)
=T_{G\leftarrow E}(t)\,T_{E\leftarrow S}.
$$

The rightmost matrix acts first: sensor coordinates become ego coordinates, then global coordinates. The calibration is selected through `sample_data.calibrated_sensor_token`; the vehicle pose is selected through `sample_data.ego_pose_token`.

### Why poses matter for research

Each sensor measurement links to the vehicle pose for its timestamp. The official tutorial describes a one-to-one correspondence between `sample_data` and `ego_pose` records. Use that association when measurements have different acquisition times, including measurements belonging to the same annotated sample. [Official tutorial, section 11](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html#11.-ego_pose)

The poses allow point clouds from different times to be accumulated in one frame and allow point-cloud and camera measurements to be aligned across timestamps. The devkit implements these transformations in `from_file_multisweep` and `map_pointcloud_to_image`. Transforming by vehicle poses accounts for the vehicle's change of frame; it does not by itself remove an object's own motion. [Official multisweep implementation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py), [Official camera projection implementation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)

The paper describes localization against an offline LiDAR map using LiDAR and odometry. These records are localization estimates in a Cartesian map frame. Latitude and longitude, or comparisons between different map coordinate systems, require an appropriate geographic or map transformation. [nuScenes paper, localization](https://arxiv.org/html/1903.11027v5#S2)

### Observations about the local poses

The first record in the local `ego_pose.json` contains:

```json
{
  "token": "5ace90b379af485b9dcb1584b01e7212",
  "timestamp": 1532402927814384,
  "rotation": [
    0.5731787718287827,
    -0.0015811634307974854,
    0.013859363182046986,
    -0.8193116095230444
  ],
  "translation": [410.77878632230204, 1179.4673290964536, 0.0]
}
```

The vehicle's reference origin is therefore at approximately `(410.779, 1179.467, 0)` meters in the relevant map frame. The quaternion supplies its orientation. This pose is linked to a `RADAR_FRONT` measurement at the same timestamp.

Direct checks of this local copy found 31,206 pose records and 31,206 `sample_data` records. Every measurement references an existing pose; all pose references are unique, every linked timestamp matches, and every pose translation has `z = 0`. These checks cover the pose table and its measurement links.
