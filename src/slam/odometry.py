"""
Scan-to-scan semantic-aware LiDAR odometry.

Pipeline per incoming scan:
    1. remove dynamic-class points (semantic.remove_dynamic_points)
    2. voxel downsample for speed / uniform density
    3. estimate normals on the reference (previous) scan
    4. register with point-to-plane ICP, initialised from a constant-velocity
       motion model
    5. accumulate the estimated relative transform into the global trajectory
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .icp import estimate_normals, point_to_plane_icp
from .pointcloud import LidarScan, voxel_downsample
from .semantic import remove_dynamic_points


@dataclass
class OdometryConfig:
    voxel_size: float = 0.3
    max_correspondence_distance: float = 1.0
    max_iterations: int = 30
    use_constant_velocity_init: bool = True
    filter_dynamic_points: bool = True


@dataclass
class OdometryFrameResult:
    frame_id: int
    pose: np.ndarray            # (4, 4) global pose (map frame)
    relative_transform: np.ndarray  # (4, 4) transform from previous to current scan
    fitness: float
    rmse: float
    n_dynamic_points_removed: int


class SemanticLidarOdometry:
    """Incrementally estimates sensor trajectory from a stream of LidarScans."""

    def __init__(self, config: OdometryConfig | None = None):
        self.config = config or OdometryConfig()
        self.global_pose = np.eye(4)
        self.trajectory: list[np.ndarray] = [self.global_pose.copy()]
        self.results: list[OdometryFrameResult] = []

        self._prev_points: np.ndarray | None = None
        self._prev_normals: np.ndarray | None = None
        self._last_relative_transform = np.eye(4)

    def _preprocess(self, scan: LidarScan) -> tuple[np.ndarray, int]:
        n_before = len(scan)
        if self.config.filter_dynamic_points:
            scan = remove_dynamic_points(scan)
        n_removed = n_before - len(scan)

        points = voxel_downsample(scan.points, self.config.voxel_size)
        return points, n_removed

    def add_scan(self, scan: LidarScan) -> OdometryFrameResult:
        points, n_removed = self._preprocess(scan)

        if self._prev_points is None:
            # First scan: it defines the map origin.
            self._prev_points = points
            self._prev_normals = estimate_normals(points)
            result = OdometryFrameResult(
                frame_id=scan.frame_id,
                pose=self.global_pose.copy(),
                relative_transform=np.eye(4),
                fitness=1.0,
                rmse=0.0,
                n_dynamic_points_removed=n_removed,
            )
            self.results.append(result)
            return result

        init = self._last_relative_transform if self.config.use_constant_velocity_init else np.eye(4)

        icp_result = point_to_plane_icp(
            source=points,
            target=self._prev_points,
            target_normals=self._prev_normals,
            init_transform=init,
            max_iterations=self.config.max_iterations,
            max_correspondence_distance=self.config.max_correspondence_distance,
        )

        self._last_relative_transform = icp_result.transform
        self.global_pose = self.global_pose @ icp_result.transform
        self.trajectory.append(self.global_pose.copy())

        self._prev_points = points
        self._prev_normals = estimate_normals(points)

        result = OdometryFrameResult(
            frame_id=scan.frame_id,
            pose=self.global_pose.copy(),
            relative_transform=icp_result.transform,
            fitness=icp_result.fitness,
            rmse=icp_result.rmse,
            n_dynamic_points_removed=n_removed,
        )
        self.results.append(result)
        return result

    def get_trajectory(self) -> np.ndarray:
        """Return the estimated trajectory as an (N, 4, 4) array of poses."""
        return np.stack(self.trajectory, axis=0)
