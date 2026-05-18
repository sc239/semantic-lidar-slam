"""
Trajectory evaluation: Absolute Pose Error (APE, commonly called ATE for its
translational component) and Relative Pose Error (RPE), computed via the
`evo` package (https://github.com/MichaelGrupp/evo) rather than a hand-rolled
implementation.

evo handles trajectory alignment (Umeyama/Horn, no scale for metric LiDAR
data), correspondence, and the statistic computation itself, so this module
is a thin adapter that converts this project's (N, 4, 4) pose arrays into
evo's `PoseTrajectory3D` objects and back.

Install: `pip install evo` (already listed in requirements.txt).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D


@dataclass
class TrajectoryErrorResult:
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    sse: float
    per_frame_error: np.ndarray


def _poses_to_evo_trajectory(poses: np.ndarray, timestamps: np.ndarray | None = None) -> PoseTrajectory3D:
    """Convert an (N, 4, 4) array of homogeneous poses into an evo PoseTrajectory3D."""
    if timestamps is None:
        timestamps = np.arange(poses.shape[0], dtype=np.float64)
    return PoseTrajectory3D(poses_se3=[p for p in poses], timestamps=timestamps)


def absolute_trajectory_error(
    estimated_poses: np.ndarray,
    ground_truth_poses: np.ndarray,
    align: bool = True,
    pose_relation: metrics.PoseRelation = metrics.PoseRelation.translation_part,
) -> TrajectoryErrorResult:
    """Compute Absolute Pose Error (APE) between estimate and ground truth
    using evo's implementation.

    Args:
        estimated_poses: (N, 4, 4) estimated trajectory.
        ground_truth_poses: (N, 4, 4) ground-truth trajectory, same length
            and frame correspondence as the estimate (already synchronised).
        align: if True, rigidly align the estimate onto ground truth (Umeyama,
            no scale) before computing error, factoring out the arbitrary
            global reference frame.
        pose_relation: which component of pose error to measure. Defaults to
            translation only, which is what's conventionally called "ATE".
    """
    if estimated_poses.shape[0] != ground_truth_poses.shape[0]:
        raise ValueError("Estimated and ground-truth trajectories must have equal length")

    traj_est = _poses_to_evo_trajectory(estimated_poses)
    traj_ref = _poses_to_evo_trajectory(ground_truth_poses)

    if align:
        traj_est.align(traj_ref, correct_scale=False)

    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data((traj_ref, traj_est))

    stats = ape_metric.get_all_statistics()

    return TrajectoryErrorResult(
        rmse=stats["rmse"],
        mean=stats["mean"],
        median=stats["median"],
        std=stats["std"],
        min=stats["min"],
        max=stats["max"],
        sse=stats["sse"],
        per_frame_error=np.array(ape_metric.error),
    )


def relative_pose_error(
    estimated_poses: np.ndarray,
    ground_truth_poses: np.ndarray,
    delta: int = 1,
    pose_relation: metrics.PoseRelation = metrics.PoseRelation.translation_part,
) -> TrajectoryErrorResult:
    """Compute Relative Pose Error (RPE) over a fixed frame delta using evo.
    Complements APE/ATE by measuring local drift rather than global
    trajectory consistency.
    """
    if estimated_poses.shape[0] != ground_truth_poses.shape[0]:
        raise ValueError("Estimated and ground-truth trajectories must have equal length")

    traj_est = _poses_to_evo_trajectory(estimated_poses)
    traj_ref = _poses_to_evo_trajectory(ground_truth_poses)

    rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=metrics.Unit.frames)
    rpe_metric.process_data((traj_ref, traj_est))

    stats = rpe_metric.get_all_statistics()

    return TrajectoryErrorResult(
        rmse=stats["rmse"],
        mean=stats["mean"],
        median=stats["median"],
        std=stats["std"],
        min=stats["min"],
        max=stats["max"],
        sse=stats["sse"],
        per_frame_error=np.array(rpe_metric.error),
    )
