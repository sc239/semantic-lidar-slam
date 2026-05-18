import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.evaluation import absolute_trajectory_error, relative_pose_error


def _curved_trajectory(n: int = 10) -> np.ndarray:
    """A non-degenerate (non-collinear) trajectory. Umeyama alignment
    requires points that span more than one dimension to be well-posed, so
    a straight line is not a valid test fixture for the aligned cases below.
    """
    t = np.linspace(0, 1, n)
    poses = np.stack([np.eye(4) for _ in range(n)])
    for i in range(n):
        poses[i, 0, 3] = 5 * t[i]
        poses[i, 1, 3] = 2 * np.sin(3 * t[i])
        poses[i, 2, 3] = 0.5 * t[i]
    return poses


def test_ate_zero_for_identical_trajectories():
    poses = _curved_trajectory()
    result = absolute_trajectory_error(poses, poses.copy())
    assert result.rmse < 1e-9


def test_ate_is_invariant_to_global_offset_when_aligned():
    gt = _curved_trajectory()
    est = gt.copy()
    est[:, :3, 3] += np.array([10.0, -3.0, 1.0])  # constant global offset, no shape distortion

    result = absolute_trajectory_error(est, gt, align=True)
    assert result.rmse < 1e-6


def test_ate_nonzero_for_diverging_trajectories():
    gt = _curved_trajectory()
    est = gt.copy()
    est[:, 1, 3] += 0.005 * np.arange(gt.shape[0]) ** 2  # quadratic drift in y

    result = absolute_trajectory_error(est, gt, align=True)
    assert result.rmse > 0.01


def test_ate_raises_on_length_mismatch():
    gt = _curved_trajectory(10)
    est = _curved_trajectory(8)
    try:
        absolute_trajectory_error(est, gt)
        assert False, "expected a ValueError for mismatched trajectory lengths"
    except ValueError:
        pass


def test_rpe_zero_for_identical_trajectories():
    poses = _curved_trajectory()
    result = relative_pose_error(poses, poses.copy(), delta=1)
    assert result.rmse < 1e-9


def test_rpe_detects_local_drift_not_visible_globally():
    """A single-frame jump followed by a compensating jump back can look
    fine in ATE-after-alignment terms over a short window, but RPE should
    flag the local inconsistency at the delta where it occurs.
    """
    gt = _curved_trajectory(n=6)
    est = gt.copy()
    est[3, 0, 3] += 0.5   # jump out
    est[4, 0, 3] += 0.0   # ... and not fully corrected at the next frame

    rpe = relative_pose_error(est, gt, delta=1)
    assert rpe.max > 0.1


if __name__ == "__main__":
    test_ate_zero_for_identical_trajectories()
    test_ate_is_invariant_to_global_offset_when_aligned()
    test_ate_nonzero_for_diverging_trajectories()
    test_ate_raises_on_length_mismatch()
    test_rpe_zero_for_identical_trajectories()
    test_rpe_detects_local_drift_not_visible_globally()
    print("All tests passed.")
