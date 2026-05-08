"""
Load frames from a Waymo Open Dataset segment (.tfrecord) into this
project's LidarScan format, so the existing odometry/mapping/evaluation
pipeline can run against real driving data instead of the synthetic demo.

This module is written against the documented Waymo Open Dataset API
(`waymo_open_dataset.utils.frame_utils` /
`waymo_open_dataset.utils.transform_utils`) but has NOT been run against a
real .tfrecord file in this environment: the `waymo-open-dataset-tf-*`
package requires a specific pinned TensorFlow version and did not install
cleanly against this sandbox's Python 3.12. Validate it against one real
segment before trusting it for anything.

Requires:
    pip install waymo-open-dataset-tf-2-12-0   # or whatever build matches
                                                 # your TensorFlow version
    (see https://github.com/waymo-research/waymo-open-dataset for the
    version matrix — the package name/version must match your installed
    TensorFlow, which is a common source of install failures.)

Waymo segments must be downloaded separately after accepting Waymo's
license, e.g.:
    gsutil cp gs://waymo_open_dataset_v_1_4_3/individual_files/training/<segment>.tfrecord .
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .pointcloud import LidarScan

try:
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils
    _WAYMO_AVAILABLE = True
except ImportError:
    _WAYMO_AVAILABLE = False


# Waymo Open Dataset per-point semantic segmentation label ids
# (waymo_open_dataset/protos/segmentation.proto - Segmentation.Type).
# NOTE: these differ from the SemanticKITTI ids used elsewhere in this repo
# (src/slam/semantic.py) - do not mix the two label schemes.
WAYMO_DYNAMIC_CLASSES = {
    1,   # TYPE_CAR
    2,   # TYPE_TRUCK
    3,   # TYPE_BUS
    4,   # TYPE_OTHER_VEHICLE
    5,   # TYPE_MOTORCYCLIST
    6,   # TYPE_CYCLIST
    7,   # TYPE_PEDESTRIAN
    12,  # TYPE_BICYCLE
    13,  # TYPE_MOTORCYCLE
}


def _require_waymo():
    if not _WAYMO_AVAILABLE:
        raise ImportError(
            "waymo_open_dataset / tensorflow are not installed. "
            "Run: pip install tensorflow waymo-open-dataset-tf-<version-matching-your-tf>"
        )


def iter_frames(tfrecord_path: str) -> Iterator["open_dataset.Frame"]:
    """Yield parsed Waymo Frame protos from a .tfrecord segment file."""
    _require_waymo()
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        yield frame


def _extract_point_cloud_and_labels(frame) -> tuple[np.ndarray, np.ndarray | None]:
    """Merge all 5 LiDAR range images into one point cloud in the vehicle
    frame, plus per-point segmentation labels if this frame has them
    (Waymo only labels a subset of frames per segment).
    """
    (
        range_images,
        camera_projections,
        seg_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose
    )
    points_all = np.concatenate(points, axis=0)

    labels_all = None
    if seg_labels:
        point_labels, _ = frame_utils.convert_range_image_to_point_cloud_labels(
            frame, range_images, seg_labels
        )
        if point_labels:
            # column 1 is the semantic class id; column 0 is instance id.
            labels_all = np.concatenate(point_labels, axis=0)[:, 1].astype(np.uint32)

    return points_all.astype(np.float32), labels_all


def _extract_ego_pose(frame) -> np.ndarray:
    """Waymo stores the vehicle-to-global-frame transform as a flat 16-value
    row-major 4x4 matrix on frame.pose.transform.
    """
    return np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)


def load_waymo_sequence(
    tfrecord_path: str, max_frames: int | None = None
) -> tuple[list[LidarScan], np.ndarray]:
    """Load a Waymo segment into a list of LidarScans plus the ground-truth
    ego-vehicle trajectory (in Waymo's global frame), for use with
    slam.odometry.SemanticLidarOdometry and slam.evaluation.

    Frames without per-point segmentation labels will have `scan.labels =
    None`; `remove_dynamic_points` already handles that by passing the scan
    through unfiltered, so mixed labelled/unlabelled sequences work but only
    get dynamic-object filtering on the frames that have labels.
    """
    _require_waymo()

    scans: list[LidarScan] = []
    gt_poses: list[np.ndarray] = []

    for i, frame in enumerate(iter_frames(tfrecord_path)):
        if max_frames is not None and i >= max_frames:
            break

        points, labels = _extract_point_cloud_and_labels(frame)
        scans.append(
            LidarScan(
                points=points,
                labels=labels,
                timestamp=frame.timestamp_micros / 1e6,
                frame_id=i,
            )
        )
        gt_poses.append(_extract_ego_pose(frame))

    return scans, np.stack(gt_poses)
