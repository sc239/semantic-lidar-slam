"""
Point-to-plane Iterative Closest Point (ICP) registration.

Point-to-plane ICP minimises the sum of squared distances between each
source point and the *tangent plane* of its corresponding target point
(rather than the point-to-point distance). This converges faster and is
more robust on locally-planar structure such as walls, roads and building
facades, which is why it is the standard choice for LiDAR scan matching.

Reference: Chen & Medioni, "Object modelling by registration of multiple
range images," Image and Vision Computing, 1992.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class ICPResult:
    transform: np.ndarray        # (4, 4) homogeneous transform, source -> target
    fitness: float                # fraction of source points with a valid correspondence
    rmse: float                   # point-to-plane RMSE at the final iteration
    n_iterations: int
    converged: bool


def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate per-point normals via PCA over the k nearest neighbours."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k, points.shape[0]))
    normals = np.empty_like(points)

    for i, neighbours in enumerate(idx):
        neighbourhood = points[neighbours]
        centered = neighbourhood - neighbourhood.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # eigenvector of smallest eigenvalue

    return normals


def _to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def point_to_plane_icp(
    source: np.ndarray,
    target: np.ndarray,
    target_normals: np.ndarray | None = None,
    init_transform: np.ndarray | None = None,
    max_iterations: int = 30,
    max_correspondence_distance: float = 1.0,
    tolerance: float = 1e-6,
) -> ICPResult:
    """Register `source` onto `target` using point-to-plane ICP.

    Args:
        source: (N, 3) source point cloud.
        target: (M, 3) target point cloud.
        target_normals: (M, 3) precomputed target normals. Estimated from
            `target` via PCA if not supplied.
        init_transform: (4, 4) initial guess (e.g. from a constant-velocity
            motion model). Identity if not supplied.
        max_correspondence_distance: correspondences farther apart than this
            are rejected as outliers (handles partial overlap / dynamic
            objects that slipped through semantic filtering).

    Returns:
        ICPResult with the estimated source->target transform.
    """
    if target_normals is None:
        target_normals = estimate_normals(target)

    T = init_transform.copy() if init_transform is not None else np.eye(4)
    tree = cKDTree(target)

    prev_rmse = np.inf
    converged = False
    fitness = 0.0
    rmse = np.inf
    it = 0

    for it in range(1, max_iterations + 1):
        R, t = T[:3, :3], T[:3, 3]
        transformed = source @ R.T + t

        dists, idx = tree.query(transformed, k=1)
        valid = dists < max_correspondence_distance
        if valid.sum() < 6:
            # Not enough inliers to solve a well-posed 6-DoF system.
            converged = False
            break

        fitness = float(valid.mean())

        p = transformed[valid]                 # transformed source points
        q = target[idx[valid]]                 # corresponding target points
        n = target_normals[idx[valid]]          # corresponding target normals

        # Linearised point-to-plane system (small-angle approximation):
        # for each correspondence, n . ((p + [alpha,beta,gamma] x p + [tx,ty,tz]) - q) = 0
        cross = np.cross(p, n)
        A = np.hstack([cross, n])               # (K, 6): [alpha, beta, gamma, tx, ty, tz]
        b = np.einsum("ij,ij->i", n, q - p)      # (K,)

        # Damped least squares for numerical stability on near-degenerate geometry.
        damping = 1e-6 * np.eye(6)
        try:
            x, *_ = np.linalg.lstsq(A.T @ A + damping, A.T @ b, rcond=None)
        except np.linalg.LinAlgError:
            converged = False
            break

        alpha, beta, gamma, tx, ty, tz = x
        d_rot = Rotation.from_rotvec([alpha, beta, gamma]).as_matrix()
        d_T = _to_homogeneous(d_rot, np.array([tx, ty, tz]))

        T = d_T @ T
        rmse = float(np.sqrt(np.mean((np.einsum("ij,ij->i", n, p - q)) ** 2)))

        if abs(prev_rmse - rmse) < tolerance:
            converged = True
            break
        prev_rmse = rmse

    return ICPResult(
        transform=T,
        fitness=fitness,
        rmse=rmse,
        n_iterations=it,
        converged=converged,
    )
