"""
semantic-lidar-slam
--------------------
A lightweight, semantic-aware LiDAR odometry and mapping pipeline.

Modules:
    pointcloud   - point cloud I/O and preprocessing utilities
    semantic     - semantic-label based dynamic object filtering
    icp          - point-to-plane ICP scan registration
    odometry     - scan-to-scan / scan-to-map odometry pipeline
    mapping      - incremental map accumulation with voxel downsampling
    evaluation   - trajectory error metrics (ATE / RPE)
"""

__version__ = "0.1.0"
