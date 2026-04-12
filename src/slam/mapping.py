"""
Incremental map accumulation.

Transforms each registered scan into the global map frame using the pose
estimated by SemanticLidarOdometry, and periodically voxel-downsamples the
accumulated map to bound memory growth over long sequences.
"""

from __future__ import annotations

import numpy as np

from .pointcloud import voxel_downsample


class VoxelMap:
    def __init__(self, voxel_size: float = 0.2, compact_every: int = 10):
        self.voxel_size = voxel_size
        self.compact_every = compact_every
        self._points = np.empty((0, 3), dtype=np.float32)
        self._frames_since_compact = 0

    def integrate(self, points_sensor_frame: np.ndarray, pose: np.ndarray) -> None:
        """Transform points into the map frame using `pose` and add them."""
        R, t = pose[:3, :3], pose[:3, 3]
        points_map = points_sensor_frame @ R.T + t
        self._points = np.vstack([self._points, points_map])

        self._frames_since_compact += 1
        if self._frames_since_compact >= self.compact_every:
            self.compact()

    def compact(self) -> None:
        if self._points.shape[0] == 0:
            return
        self._points = voxel_downsample(self._points, self.voxel_size)
        self._frames_since_compact = 0

    @property
    def points(self) -> np.ndarray:
        return self._points

    def save(self, path: str) -> None:
        """Save the accumulated map as a simple ASCII XYZ point cloud."""
        np.savetxt(path, self._points, fmt="%.4f")
