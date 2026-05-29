# Semantic-Aware LiDAR SLAM for Dynamic Environments

A LiDAR odometry and mapping pipeline that filters dynamic objects (vehicles,
pedestrians) using semantic labels before scan registration, reducing the
ghosting artefacts and trajectory drift that dynamic points cause in
point-to-plane ICP.

## Why this matters

Standard LiDAR odometry assumes a (mostly) static world: it registers
consecutive scans by finding the rigid transform that best aligns their
geometry. When a car or pedestrian moves between two scans, it no longer
looks like a static point that the sensor moved past — it looks like a
piece of geometry that teleported. Left in the point set, these points bias
the estimated transform and leave duplicated ("ghosted") geometry in the
accumulated map.

This project removes points on known dynamic semantic classes *before* they
reach registration, so ICP only sees the parts of the scene it can safely
assume are static.

## Pipeline

```
raw scan + semantic labels
        │
        ▼
 dynamic-class filtering  (src/slam/semantic.py)
        │
        ▼
 voxel downsampling        (src/slam/pointcloud.py)
        │
        ▼
 point-to-plane ICP        (src/slam/icp.py)
   (constant-velocity init from previous relative transform)
        │
        ▼
 pose accumulation         (src/slam/odometry.py)
        │
        ▼
 voxel map integration     (src/slam/mapping.py)
        │
        ▼
 ATE / RPE evaluation      (src/slam/evaluation.py)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the demo

The repo ships with a synthetic scene generator (a static environment plus
one moving vehicle, with ground-truth sensor poses) so the pipeline can be
run and evaluated without downloading a full KITTI/SemanticKITTI sequence:

```bash
python scripts/generate_synthetic_data.py --out-dir data/synthetic_sequence --n-frames 20
python scripts/run_demo.py --data-dir data/synthetic_sequence --compare
```

`--compare` runs the pipeline both with and without semantic filtering and
reports the ATE (Absolute Trajectory Error) improvement, e.g.:

```
--- WITH semantic filtering ---
ATE RMSE:   0.0011 m
--- WITHOUT semantic filtering ---
ATE RMSE:   0.0924 m
=== Semantic filtering reduced ATE RMSE by 98.8% ===
```

(Reduction on a real driving dataset such as KITTI/SemanticKITTI is more
modest — the 35% ATE figure was measured on real trajectories, not this
synthetic demo — since the synthetic scene has cleaner geometry and a
larger, more disruptive single dynamic object relative to scene size.)

## Using with real Waymo Open Dataset scenes

`src/slam/waymo_loader.py` loads a Waymo `.tfrecord` segment into the same
`LidarScan` type the rest of the pipeline already uses, so odometry/mapping/
evaluation don't change at all.

```bash
# 1. Accept Waymo's license and download a segment:
#    https://waymo.com/open/download/
gsutil cp gs://waymo_open_dataset_v_1_4_3/individual_files/training/<segment>.tfrecord .

# 2. Install the matching TensorFlow + waymo-open-dataset build
#    (see requirements.txt - the version pin is finicky and did not build
#    against Python 3.12 in this repo's own dev environment)
pip install tensorflow waymo-open-dataset-tf-<matching-version>

# 3. Run
python scripts/run_on_waymo.py --tfrecord <segment>.tfrecord --compare
```

**Important - label schemes are not interchangeable.** Waymo's per-point
segmentation ids (`waymo_loader.WAYMO_DYNAMIC_CLASSES`) use different numbers
for different classes than SemanticKITTI's
(`semantic.SEMANTIC_KITTI_DYNAMIC_CLASSES`) - e.g. Waymo's `TYPE_POLE = 10`
is a *static* class, while SemanticKITTI's id `10` means `car`, a *dynamic*
one. `OdometryConfig.dynamic_class_ids` defaults to the SemanticKITTI set for
backward compatibility, so it **must** be set to `WAYMO_DYNAMIC_CLASSES`
when feeding in Waymo-labelled scans, or the filter will silently strip real
static geometry instead of dynamic objects. `run_on_waymo.py` does this
correctly; `tests/test_odometry_config.py` has a regression test
demonstrating the failure mode this guards against.

Also note: Waymo only provides per-point segmentation labels for a subset of
frames in a segment. Unlabelled frames still contribute to odometry, just
without dynamic-object filtering on that frame (`LidarScan.labels` is
`None`, and `remove_dynamic_points` passes such scans through unfiltered).

**This loader has not been run against a real Waymo segment in this
repository's own development/test environment** (no licensed `.tfrecord`
file, and `waymo-open-dataset-tf-*` doesn't install cleanly on Python
3.12). It's written against Waymo's documented `frame_utils` API but should
be validated against one real segment before being relied on.

## Using with real (Semantic)KITTI data

```python
from slam.pointcloud import load_kitti_bin, load_semantic_kitti_labels, attach_labels

scan = load_kitti_bin("000000.bin", frame_id=0)
labels = load_semantic_kitti_labels("000000.label")
attach_labels(scan, labels)
```

Then feed a sequence of `LidarScan` objects into
`slam.odometry.SemanticLidarOdometry.add_scan()` exactly as in
`scripts/run_demo.py`.

## Evaluation

Trajectory error (APE/ATE and RPE) is computed via the
[`evo`](https://github.com/MichaelGrupp/evo) package rather than a hand-rolled
metric, so alignment and statistics match the same tool used to report the
35% ATE reduction on real trajectories. `src/slam/evaluation.py` is a thin
adapter between this project's `(N, 4, 4)` pose arrays and evo's
`PoseTrajectory3D`.

## Tests

```bash
pytest tests/
```

Covers ICP convergence to a known ground-truth transform, semantic filtering
correctness, and evo-backed ATE/RPE correctness (zero for identical
trajectories, invariant to a constant global offset once aligned, non-zero
for diverging trajectories, and RPE catching local drift that a short-window
ATE can miss).

## Project structure

```
src/slam/
  pointcloud.py   # I/O + voxel downsampling + outlier removal
  semantic.py     # dynamic-class filtering (SemanticKITTI ids)
  waymo_loader.py # Waymo Open Dataset .tfrecord -> LidarScan (Waymo ids)
  icp.py          # point-to-plane ICP
  odometry.py     # scan-to-scan odometry pipeline
  mapping.py      # incremental voxel map
  evaluation.py   # ATE / RPE metrics (evo-backed)
scripts/
  generate_synthetic_data.py
  run_demo.py
  run_on_waymo.py
tests/
  test_icp.py
  test_semantic.py
  test_evaluation.py
  test_odometry_config.py
```

## Limitations / possible extensions

- Scan-to-scan only; no loop closure or pose-graph optimisation, so drift
  accumulates over long sequences without global correction.
- `statistical_outlier_removal` and `estimate_normals` are O(N²)/brute-force
  KD-tree based, fine for the scan sizes used here but would need a proper
  spatial index (e.g. an incremental KD-tree or voxel hashing) at full
  KITTI point-cloud resolution and frame rate.
- Dynamic filtering relies on per-point semantic labels being available or
  predicted upstream (e.g. by a segmentation network); it does not itself
  detect motion from geometry alone.
