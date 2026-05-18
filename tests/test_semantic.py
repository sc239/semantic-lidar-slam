import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.pointcloud import LidarScan
from slam.semantic import dynamic_point_ratio, remove_dynamic_points


def test_remove_dynamic_points():
    points = np.random.default_rng(0).normal(size=(10, 3))
    labels = np.array([40, 40, 40, 10, 10, 40, 252, 40, 30, 40], dtype=np.uint32)
    scan = LidarScan(points=points, labels=labels)

    filtered = remove_dynamic_points(scan)

    assert len(filtered) == 6  # 6 points with static label 40
    assert np.all(filtered.labels == 40)


def test_dynamic_point_ratio():
    points = np.zeros((4, 3))
    labels = np.array([40, 10, 40, 10], dtype=np.uint32)
    scan = LidarScan(points=points, labels=labels)

    assert dynamic_point_ratio(scan) == 0.5


def test_scan_without_labels_is_unchanged():
    points = np.zeros((5, 3))
    scan = LidarScan(points=points)

    result = remove_dynamic_points(scan)
    assert len(result) == 5


if __name__ == "__main__":
    test_remove_dynamic_points()
    test_dynamic_point_ratio()
    test_scan_without_labels_is_unchanged()
    print("All tests passed.")
