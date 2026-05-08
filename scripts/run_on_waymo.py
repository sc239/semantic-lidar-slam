"""
Run the semantic-aware LiDAR odometry pipeline against a real Waymo Open
Dataset segment instead of the synthetic demo sequence.

NOT independently tested against a real .tfrecord in this environment (see
the warning at the top of src/slam/waymo_loader.py) - validate against one
real segment before relying on the numbers it prints.

Setup:
    1. Accept the Waymo Open Dataset license and download a segment:
       https://waymo.com/open/download/
       gsutil cp gs://waymo_open_dataset_v_1_4_3/individual_files/training/<segment>.tfrecord .
    2. pip install tensorflow waymo-open-dataset-tf-<version matching your TF>
    3. python scripts/run_on_waymo.py --tfrecord <segment>.tfrecord

Only frames that have per-point segmentation labels get dynamic-object
filtering (Waymo labels a subset of frames per segment); unlabelled frames
still contribute to odometry, just without filtering on that frame.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.evaluation import absolute_trajectory_error
from slam.mapping import VoxelMap
from slam.odometry import OdometryConfig, SemanticLidarOdometry
from slam.waymo_loader import WAYMO_DYNAMIC_CLASSES, load_waymo_sequence


def run(tfrecord_path: str, max_frames: int | None, out_dir: str, filter_dynamic: bool):
    scans, gt_poses = load_waymo_sequence(tfrecord_path, max_frames=max_frames)

    n_labelled = sum(1 for s in scans if s.labels is not None)
    print(f"Loaded {len(scans)} frames ({n_labelled} with per-point segmentation labels)")

    # dynamic_class_ids MUST be the Waymo set here, not the module default
    # (SemanticKITTI) - label id numbers are not comparable across schemes,
    # e.g. Waymo TYPE_POLE=10 vs. SemanticKITTI car=10. Getting this wrong
    # silently strips static geometry instead of dynamic objects.
    config = OdometryConfig(
        voxel_size=0.3,
        max_correspondence_distance=1.0,
        filter_dynamic_points=filter_dynamic,
        dynamic_class_ids=WAYMO_DYNAMIC_CLASSES,
    )
    odom = SemanticLidarOdometry(config)
    vmap = VoxelMap(voxel_size=0.2)

    total_dynamic_removed = 0
    for scan in scans:
        result = odom.add_scan(scan)  # dynamic-point removal happens inside, using WAYMO_DYNAMIC_CLASSES
        vmap.integrate(scan.points, result.pose)
        total_dynamic_removed += result.n_dynamic_points_removed

    est_trajectory = odom.get_trajectory()
    ate = absolute_trajectory_error(est_trajectory, gt_poses)

    label = "WITH semantic filtering" if filter_dynamic else "WITHOUT semantic filtering"
    print(f"\n--- {label} ---")
    print(f"Dynamic points removed (total across sequence): {total_dynamic_removed}")
    print(f"ATE RMSE:   {ate.rmse:.4f} m")
    print(f"ATE mean:   {ate.mean:.4f} m")
    print(f"ATE median: {ate.median:.4f} m")
    print(f"ATE max:    {ate.max:.4f} m")

    os.makedirs(out_dir, exist_ok=True)
    suffix = "filtered" if filter_dynamic else "unfiltered"
    np.savetxt(
        os.path.join(out_dir, f"waymo_trajectory_{suffix}.txt"),
        est_trajectory[:, :3, 3],
        header="x y z (estimated, map frame)",
    )
    vmap.compact()
    vmap.save(os.path.join(out_dir, f"waymo_map_{suffix}.xyz"))
    return ate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", required=True, help="Path to a Waymo .tfrecord segment")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out-dir", default="results_waymo")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both with and without semantic filtering and compare ATE.",
    )
    args = parser.parse_args()

    if args.compare:
        ate_filtered = run(args.tfrecord, args.max_frames, args.out_dir, filter_dynamic=True)
        ate_unfiltered = run(args.tfrecord, args.max_frames, args.out_dir, filter_dynamic=False)
        improvement = 100 * (1 - ate_filtered.rmse / ate_unfiltered.rmse)
        print(f"\n=== Semantic filtering reduced ATE RMSE by {improvement:.1f}% ===")
    else:
        run(args.tfrecord, args.max_frames, args.out_dir, filter_dynamic=True)
