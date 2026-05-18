import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.icp import point_to_plane_icp


def make_planar_cloud(rng, n=2000):
    """A simple synthetic scene: a ground plane plus two intersecting walls,
    which gives well-conditioned normals for point-to-plane ICP.
    """
    ground = np.column_stack(
        [rng.uniform(-5, 5, n // 2), rng.uniform(-5, 5, n // 2), np.zeros(n // 2)]
    )
    wall = np.column_stack(
        [rng.uniform(-5, 5, n // 2), np.full(n // 2, 5.0), rng.uniform(0, 3, n // 2)]
    )
    return np.vstack([ground, wall]).astype(np.float64)


def test_icp_recovers_known_transform():
    rng = np.random.default_rng(42)
    target = make_planar_cloud(rng)

    true_R = Rotation.from_euler("z", 8, degrees=True).as_matrix()
    true_t = np.array([0.3, -0.15, 0.0])
    source = (target - true_t) @ true_R  # source = R^-1 (target - t), so target = source @ R^T + t

    result = point_to_plane_icp(
        source=source,
        target=target,
        max_iterations=50,
        max_correspondence_distance=2.0,
    )

    est_R = result.transform[:3, :3]
    est_t = result.transform[:3, 3]

    rot_error_deg = np.degrees(
        np.arccos(np.clip((np.trace(true_R.T @ est_R) - 1) / 2, -1.0, 1.0))
    )
    trans_error = np.linalg.norm(true_t - est_t)

    assert result.fitness > 0.9
    assert rot_error_deg < 0.5, f"rotation error too large: {rot_error_deg:.3f} deg"
    assert trans_error < 0.05, f"translation error too large: {trans_error:.4f} m"


def test_icp_reports_low_fitness_on_disjoint_clouds():
    rng = np.random.default_rng(0)
    target = make_planar_cloud(rng)
    source = target + np.array([500.0, 500.0, 500.0])  # far away, no overlap

    result = point_to_plane_icp(
        source=source,
        target=target,
        max_iterations=10,
        max_correspondence_distance=1.0,
    )
    assert not result.converged or result.fitness < 0.1


if __name__ == "__main__":
    test_icp_recovers_known_transform()
    test_icp_reports_low_fitness_on_disjoint_clouds()
    print("All tests passed.")
