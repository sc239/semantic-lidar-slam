"""
Point cloud I/O and preprocessing.

Supports:
  - KITTI-style raw binary scans (.bin, float32 x,y,z,intensity)
  - Optional per-point semantic labels (.label, uint32 KITTI semantic-kitti format)
  - Simple voxel-grid downsampling and statistical outlier removal
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class LidarScan:
    """A single LiDAR scan with optional per-point semantic labels."""

    points: np.ndarray                 # (N, 3) float32 xyz in sensor frame
    intensity: np.ndarray | None = None  # (N,) float32
    labels: np.ndarray | None = None     # (N,) uint32 semantic class id
    timestamp: float = 0.0
    frame_id: int = 0

    def __len__(self) -> int:
        return self.points.shape[0]

    def filtered(self, mask: np.ndarray) -> "LidarScan":
        """Return a new LidarScan containing only points where mask is True."""
        return LidarScan(
            points=self.points[mask],
            intensity=self.intensity[mask] if self.intensity is not None else None,
            labels=self.labels[mask] if self.labels is not None else None,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
        )


def load_kitti_bin(path: str, frame_id: int = 0, timestamp: float = 0.0) -> LidarScan:
    """Load a KITTI-style .bin point cloud (N x 4: x, y, z, intensity)."""
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return LidarScan(
        points=raw[:, :3].copy(),
        intensity=raw[:, 3].copy(),
        frame_id=frame_id,
        timestamp=timestamp,
    )


def load_semantic_kitti_labels(path: str) -> np.ndarray:
    """Load SemanticKITTI .label file. Lower 16 bits encode the semantic class id."""
    raw = np.fromfile(path, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint32)


def attach_labels(scan: LidarScan, labels: np.ndarray) -> LidarScan:
    if labels.shape[0] != len(scan):
        raise ValueError(
            f"Label count ({labels.shape[0]}) does not match point count ({len(scan)})"
        )
    scan.labels = labels
    return scan


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample a point cloud to (approximately) one point per voxel.

    Keeps the point closest to each voxel's centroid rather than a naive
    first-hit, which gives more stable geometry for ICP normal estimation.
    """
    if voxel_size <= 0:
        return points

    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    # Encode 3D voxel index into a single key for grouping.
    keys = (
        voxel_idx[:, 0].astype(np.int64) * 73856093
        ^ voxel_idx[:, 1].astype(np.int64) * 19349663
        ^ voxel_idx[:, 2].astype(np.int64) * 83492791
    )
    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_points = points[order]

    boundaries = np.nonzero(np.diff(sorted_keys))[0] + 1
    groups = np.split(np.arange(sorted_points.shape[0]), boundaries)

    out = np.empty((len(groups), 3), dtype=points.dtype)
    for i, g in enumerate(groups):
        pts = sorted_points[g]
        centroid = pts.mean(axis=0)
        closest = np.argmin(np.sum((pts - centroid) ** 2, axis=1))
        out[i] = pts[closest]
    return out


def statistical_outlier_removal(
    points: np.ndarray, k: int = 12, std_ratio: float = 2.0
) -> np.ndarray:
    """Remove points whose mean distance to their k nearest neighbours is an
    outlier relative to the global distribution. Pure-numpy, O(N^2) — fine for
    the scan sizes used in this project's tests/demo (a proper KD-tree should
    be used for full-resolution KITTI scans).
    """
    if points.shape[0] <= k:
        return points

    from scipy.spatial import cKDTree  # local import: optional heavy dependency

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)  # includes self at index 0
    mean_dists = dists[:, 1:].mean(axis=1)

    mu, sigma = mean_dists.mean(), mean_dists.std()
    threshold = mu + std_ratio * sigma
    return points[mean_dists < threshold]
