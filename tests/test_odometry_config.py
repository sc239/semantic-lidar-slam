import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.odometry import OdometryConfig, SemanticLidarOdometry
from slam.pointcloud import LidarScan
from slam.semantic import SEMANTIC_KITTI_DYNAMIC_CLASSES

# A label scheme where id 10 means something static (mirrors the real
# Waymo/SemanticKITTI mismatch: Waymo TYPE_POLE = 10, SemanticKITTI car = 10).
FAKE_SCHEME_DYNAMIC_CLASSES = {1, 2}  # e.g. "moving-thing-a", "moving-thing-b"


def _scan_with_labels(labels):
    points = np.random.default_rng(0).normal(size=(len(labels), 3))
    return LidarScan(points=points, labels=np.array(labels, dtype=np.uint32))


def test_default_config_uses_semantic_kitti_ids():
    assert OdometryConfig().dynamic_class_ids == SEMANTIC_KITTI_DYNAMIC_CLASSES


def test_wrong_label_scheme_would_misfilter_static_points():
    """Demonstrates *why* dynamic_class_ids must be overridden per dataset:
    label id 10 is SemanticKITTI 'car' but would be a static class in a
    different scheme. Using the default id set on foreign labels removes
    the wrong points.
    """
    scan = _scan_with_labels([10, 10, 40, 40])  # id 10 = static in FAKE_SCHEME, dynamic in SemanticKITTI

    odom_wrong_scheme = SemanticLidarOdometry(OdometryConfig())  # default = SemanticKITTI ids
    result = odom_wrong_scheme.add_scan(scan)
    assert result.n_dynamic_points_removed == 2  # incorrectly strips the two static-in-reality points

    odom_correct_scheme = SemanticLidarOdometry(
        OdometryConfig(dynamic_class_ids=FAKE_SCHEME_DYNAMIC_CLASSES)
    )
    result2 = odom_correct_scheme.add_scan(scan)
    assert result2.n_dynamic_points_removed == 0  # correctly keeps them, since 10/40 aren't dynamic here


def test_custom_dynamic_class_ids_are_actually_used():
    scan = _scan_with_labels([1, 1, 2, 99, 99])
    config = OdometryConfig(dynamic_class_ids=FAKE_SCHEME_DYNAMIC_CLASSES)
    odom = SemanticLidarOdometry(config)

    result = odom.add_scan(scan)
    assert result.n_dynamic_points_removed == 3  # the three points labelled 1 or 1 or 2


if __name__ == "__main__":
    test_default_config_uses_semantic_kitti_ids()
    test_wrong_label_scheme_would_misfilter_static_points()
    test_custom_dynamic_class_ids_are_actually_used()
    print("All tests passed.")
