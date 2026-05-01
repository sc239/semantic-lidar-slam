"""
End-to-end demo: run semantic-aware LiDAR odometry over the synthetic
sequence, evaluate ATE against ground truth (with and without semantic
filtering, to show its effect), and save the resulting map + trajectory.

Usage:
    python scripts/generate_synthetic_data.py --out-dir data/synthetic_sequence
    python scripts/run_demo.py --data-dir data/synthetic_sequence
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.evaluation import absolute_trajectory_error
from slam.mapping import VoxelMap
from slam.odometry import OdometryConfig, SemanticLidarOdometry
from slam.pointcloud import LidarScan


def load_sequence(data_dir: str) -> list[LidarScan]:
    files = sorted(glob.glob(os.path.join(data_dir, "frame_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No frames found in {data_dir}. Run generate_synthetic_data.py first."
        )

    scans, gt_poses = [], []
    for i, f in enumerate(files):
        data = np.load(f)
        scans.append(
            LidarScan(
                points=data["points"],
                labels=data["labels"],
                frame_id=i,
            )
        )
        gt_poses.append(data["gt_pose"])
    return scans, np.stack(gt_poses)


def run(data_dir: str, filter_dynamic: bool, out_dir: str) -> None:
    scans, gt_poses = load_sequence(data_dir)

    config = OdometryConfig(
        voxel_size=0.3,
        max_correspondence_distance=1.0,
        filter_dynamic_points=filter_dynamic,
    )
    odom = SemanticLidarOdometry(config)
    vmap = VoxelMap(voxel_size=0.2)

    total_dynamic_removed = 0
    for scan in scans:
        result = odom.add_scan(scan)
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
        os.path.join(out_dir, f"trajectory_{suffix}.txt"),
        est_trajectory[:, :3, 3],
        header="x y z (estimated, map frame)",
    )
    vmap.compact()
    vmap.save(os.path.join(out_dir, f"map_{suffix}.xyz"))

    return ate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/synthetic_sequence")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both with and without semantic filtering and compare ATE.",
    )
    args = parser.parse_args()

    if args.compare:
        ate_filtered = run(args.data_dir, filter_dynamic=True, out_dir=args.out_dir)
        ate_unfiltered = run(args.data_dir, filter_dynamic=False, out_dir=args.out_dir)
        improvement = 100 * (1 - ate_filtered.rmse / ate_unfiltered.rmse)
        print(f"\n=== Semantic filtering reduced ATE RMSE by {improvement:.1f}% ===")
    else:
        run(args.data_dir, filter_dynamic=True, out_dir=args.out_dir)
