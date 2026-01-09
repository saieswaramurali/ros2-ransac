"""
RANSAC Perception - ROS2 RANSAC-based geometric primitive detection.

This package provides RANSAC algorithms for detecting geometric primitives
(planes, lines) from sensor data (PointCloud2, LaserScan).
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .ransac_core import RANSACPlane, RANSACLine2D, RANSACLine3D
from .point_cloud_handler import PointCloudHandler
from .laser_scan_handler import LaserScanHandler

__all__ = [
    'RANSACPlane',
    'RANSACLine2D', 
    'RANSACLine3D',
    'PointCloudHandler',
    'LaserScanHandler',
]
