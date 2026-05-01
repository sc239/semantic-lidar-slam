"""
Generate a synthetic LiDAR sequence for testing/demoing the pipeline without
needing to download a full KITTI/SemanticKITTI sequence.

The scene consists of:
  - a static "environment" point cloud sampled from box surfaces (walls,
    ground plane, a couple of pillars) that stays fixed in the world frame
  - a small number of "dynamic" clusters (representing cars/pedestrians)
    that translate over time and are semantically labelled as dynamic
  - a sensor that moves along a smooth path through the scene

Each frame is saved as an .npz with points, labels, and the ground-truth
sensor pose, so the demo script can both run odometry and evaluate ATE
against a known-correct trajectory.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation

STATIC_LABEL = 40      # "road"-like static class id (SemanticKITTI: road=40)
DYNAMIC_LABEL = 252     # "moving-car" class id (SemanticKITTI convention)


def sample_box_surface(center, size, n_points, rng):
    """Sample points on the 6 faces of an axis-aligned box (hollow shell)."""
    half = np.array(size) / 2
    pts = []
    for axis in range(3):
        for sign in (-1, 1):
            n = n_points // 6
            local = rng.uniform(-half, half, size=(n, 3))
            local[:, axis] = sign * half[axis]
            pts.append(local + center)
    return np.vstack(pts)


def build_static_environment(rng, n_points=6000):
    ground = np.column_stack(
        [
            rng.uniform(-20, 20, n_points // 2),
            rng.uniform(-20, 20, n_points // 2),
            np.zeros(n_points // 2) + rng.normal(0, 0.01, n_points // 2),
        ]
    )
    walls = sample_box_surface(center=(0, 0, 3), size=(40, 40, 6), n_points=n_points // 2, rng=rng)
    pillar1 = sample_box_surface(center=(5, 3, 1.5), size=(1, 1, 3), n_points=500, rng=rng)
    pillar2 = sample_box_surface(center=(-6, -2, 1.5), size=(1, 1, 3), n_points=500, rng=rng)
    return np.vstack([ground, walls, pillar1, pillar2])


def build_dynamic_object(center, size=(1.8, 4.0, 1.5), n_points=300, rng=None):
    return sample_box_surface(center=center, size=size, n_points=n_points, rng=rng)


def sensor_pose(t: float) -> np.ndarray:
    """Ground-truth sensor pose at normalized time t in [0, 1]: a gentle
    S-curve path through the static scene, moving mostly along +x.
    """
    x = 12 * t - 6
    y = 3.0 * np.sin(1.5 * t)
    yaw = np.arctan2(np.gradient([np.sin(1.5 * s) for s in [t - 1e-3, t + 1e-3]])[0] * 3.0, 1.0)

    R = Rotation.from_euler("z", yaw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, 0.5]
    return T


def generate_sequence(out_dir: str, n_frames: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    static_points = build_static_environment(rng)

    for i in range(n_frames):
        t = i / (n_frames - 1)
        pose = sensor_pose(t)

        # One dynamic vehicle driving past in the opposite direction.
        dyn_center = np.array([6 - 10 * t, -4 + 8 * t, 0.75])
        dynamic_points = build_dynamic_object(dyn_center, rng=rng)

        world_points = np.vstack([static_points, dynamic_points])
        labels = np.concatenate(
            [
                np.full(len(static_points), STATIC_LABEL, dtype=np.uint32),
                np.full(len(dynamic_points), DYNAMIC_LABEL, dtype=np.uint32),
            ]
        )

        # Transform world points into the (noisy) sensor frame for this frame.
        R, tvec = pose[:3, :3], pose[:3, 3]
        sensor_points = (world_points - tvec) @ R
        sensor_points += rng.normal(0, 0.01, sensor_points.shape)  # sensor noise

        np.savez(
            os.path.join(out_dir, f"frame_{i:04d}.npz"),
            points=sensor_points.astype(np.float32),
            labels=labels,
            gt_pose=pose,
        )

    print(f"Wrote {n_frames} synthetic frames to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/synthetic_sequence")
    parser.add_argument("--n-frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_sequence(args.out_dir, args.n_frames, args.seed)
