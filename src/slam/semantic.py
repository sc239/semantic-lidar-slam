"""
Semantic-label based dynamic object filtering.

Scan registration degrades when moving objects (vehicles, pedestrians,
cyclists) are included in the point set used for ICP: a car that moved
between two scans creates a false correspondence and biases the estimated
transform, and it leaves "ghosting" artefacts (duplicate/smeared geometry)
in the accumulated map. This module removes points belonging to known
dynamic semantic classes before they reach the registration stage.

Class ids follow the SemanticKITTI convention
(https://semantic-kitti.org format), but any labelling scheme can be used by
supplying a custom `dynamic_class_ids` set.
"""

from __future__ import annotations

import numpy as np

# SemanticKITTI class ids that correspond to potentially moving objects.
SEMANTIC_KITTI_DYNAMIC_CLASSES = {
    10,  # car
    11,  # bicycle
    13,  # bus
    15,  # motorcycle
    16,  # on-rails
    18,  # truck
    20,  # other-vehicle
    30,  # person
    31,  # bicyclist
    32,  # motorcyclist
    252,  # moving-car
    253,  # moving-bicyclist
    254,  # moving-person
    255,  # moving-motorcyclist
    256,  # moving-on-rails
    257,  # moving-bus
    258,  # moving-truck
    259,  # moving-other-vehicle
}


def dynamic_mask(
    labels: np.ndarray, dynamic_class_ids: set[int] = SEMANTIC_KITTI_DYNAMIC_CLASSES
) -> np.ndarray:
    """Return a boolean mask that is True for points on a dynamic class."""
    return np.isin(labels, list(dynamic_class_ids))


def remove_dynamic_points(scan, dynamic_class_ids: set[int] | None = None):
    """Return a copy of `scan` with dynamic-class points removed.

    If the scan has no semantic labels, the scan is returned unchanged
    (registration then falls back to purely geometric ICP).
    """
    if scan.labels is None:
        return scan

    ids = dynamic_class_ids if dynamic_class_ids is not None else SEMANTIC_KITTI_DYNAMIC_CLASSES
    mask = ~dynamic_mask(scan.labels, ids)
    return scan.filtered(mask)


def dynamic_point_ratio(scan, dynamic_class_ids: set[int] | None = None) -> float:
    """Fraction of points in the scan belonging to a dynamic class. Useful
    for logging / sanity-checking a sequence before running the full pipeline.
    """
    if scan.labels is None or len(scan) == 0:
        return 0.0
    ids = dynamic_class_ids if dynamic_class_ids is not None else SEMANTIC_KITTI_DYNAMIC_CLASSES
    return float(dynamic_mask(scan.labels, ids).mean())
