"""
LaserScan Message Handler.

This module provides utilities for converting ROS2 LaserScan messages
to numpy arrays for RANSAC processing.
"""

import numpy as np
from typing import Optional, Tuple

from sensor_msgs.msg import LaserScan


class LaserScanHandler:
    """
    Handler for converting between ROS2 LaserScan messages and numpy arrays.
    """
    
    @staticmethod
    def laserscan_to_cartesian(msg: LaserScan) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert LaserScan message to 2D Cartesian coordinates.
        
        Args:
            msg: LaserScan message
            
        Returns:
            Tuple of (points, valid_indices) where points is shape (N, 2)
        """
        # Generate angles for each range measurement
        num_readings = len(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, num_readings)
        
        # Convert ranges to numpy array
        ranges = np.array(msg.ranges, dtype=np.float32)
        
        # Filter valid ranges
        valid_mask = (ranges >= msg.range_min) & (ranges <= msg.range_max) & np.isfinite(ranges)
        valid_indices = np.where(valid_mask)[0]
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        # Convert polar to Cartesian
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        points = np.column_stack([x, y])
        
        return points, valid_indices
    
    @staticmethod
    def laserscan_to_3d(
        msg: LaserScan, 
        z_height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert LaserScan to 3D points at a fixed height.
        
        This is useful for combining with 3D point cloud processing.
        
        Args:
            msg: LaserScan message
            z_height: Z coordinate for all points
            
        Returns:
            Tuple of (points, valid_indices) where points is shape (N, 3)
        """
        points_2d, valid_indices = LaserScanHandler.laserscan_to_cartesian(msg)
        
        # Add Z coordinate
        z = np.full((len(points_2d), 1), z_height, dtype=np.float32)
        points_3d = np.hstack([points_2d, z])
        
        return points_3d, valid_indices
    
    @staticmethod
    def get_intensities(
        msg: LaserScan, 
        valid_indices: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Get intensity values for valid points.
        
        Args:
            msg: LaserScan message
            valid_indices: Indices of valid points from laserscan_to_cartesian
            
        Returns:
            Numpy array of intensities or None if not available
        """
        if len(msg.intensities) == 0:
            return None
        
        intensities = np.array(msg.intensities, dtype=np.float32)
        return intensities[valid_indices]
    
    @staticmethod
    def cartesian_to_polar(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D Cartesian coordinates back to polar (range, angle).
        
        Args:
            points: Numpy array of shape (N, 2) with XY coordinates
            
        Returns:
            Tuple of (ranges, angles)
        """
        ranges = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        angles = np.arctan2(points[:, 1], points[:, 0])
        return ranges, angles
    
    @staticmethod
    def filter_by_range(
        points: np.ndarray,
        min_range: float = 0.0,
        max_range: float = float('inf')
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter points by distance from origin.
        
        Args:
            points: Numpy array of shape (N, 2) or (N, 3)
            min_range: Minimum distance to keep
            max_range: Maximum distance to keep
            
        Returns:
            Tuple of (filtered_points, valid_indices)
        """
        if points.shape[1] == 2:
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        else:
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        
        valid_mask = (distances >= min_range) & (distances <= max_range)
        valid_indices = np.where(valid_mask)[0]
        
        return points[valid_mask], valid_indices
    
    @staticmethod
    def filter_by_angle(
        points: np.ndarray,
        min_angle: float = -np.pi,
        max_angle: float = np.pi
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter 2D points by angle from X-axis.
        
        Args:
            points: Numpy array of shape (N, 2)
            min_angle: Minimum angle in radians
            max_angle: Maximum angle in radians
            
        Returns:
            Tuple of (filtered_points, valid_indices)
        """
        angles = np.arctan2(points[:, 1], points[:, 0])
        valid_mask = (angles >= min_angle) & (angles <= max_angle)
        valid_indices = np.where(valid_mask)[0]
        
        return points[valid_mask], valid_indices
    
    @staticmethod
    def segment_by_gap(
        points: np.ndarray,
        gap_threshold: float = 0.5
    ) -> list:
        """
        Segment points into clusters based on gaps between consecutive points.
        
        Useful for separating objects in laser scan data.
        
        Args:
            points: Numpy array of shape (N, 2), expected to be in scan order
            gap_threshold: Distance threshold to consider as gap
            
        Returns:
            List of numpy arrays, each containing points in a segment
        """
        if len(points) < 2:
            return [points] if len(points) > 0 else []
        
        # Compute distances between consecutive points
        diffs = np.diff(points, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Find gaps
        gap_indices = np.where(distances > gap_threshold)[0]
        
        # Split into segments
        segments = []
        start = 0
        for gap_idx in gap_indices:
            segments.append(points[start:gap_idx + 1])
            start = gap_idx + 1
        
        # Add last segment
        if start < len(points):
            segments.append(points[start:])
        
        return segments
