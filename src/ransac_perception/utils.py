"""
Utility functions for RANSAC perception.
"""

import numpy as np
from typing import Tuple, List
from geometry_msgs.msg import Point, Quaternion, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import math


def plane_to_quaternion(normal: np.ndarray) -> Quaternion:
    """
    Convert plane normal to quaternion for visualization.
    
    The quaternion represents rotation from Z-axis to the plane normal.
    
    Args:
        normal: Unit normal vector [nx, ny, nz]
        
    Returns:
        Quaternion message
    """
    # Default orientation is Z-up
    z_axis = np.array([0, 0, 1])
    
    # Handle case where normal is parallel to Z-axis
    if np.abs(np.dot(normal, z_axis)) > 0.9999:
        if normal[2] > 0:
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        else:
            return Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
    
    # Compute rotation axis (cross product of Z-axis and normal)
    axis = np.cross(z_axis, normal)
    axis = axis / np.linalg.norm(axis)
    
    # Compute rotation angle
    angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
    
    # Convert axis-angle to quaternion
    half_angle = angle / 2
    sin_half = np.sin(half_angle)
    
    return Quaternion(
        x=axis[0] * sin_half,
        y=axis[1] * sin_half,
        z=axis[2] * sin_half,
        w=np.cos(half_angle)
    )


def create_plane_marker(
    plane_coefficients: np.ndarray,
    center: np.ndarray,
    header: Header,
    marker_id: int,
    size: Tuple[float, float] = (1.0, 1.0),
    color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5),
    namespace: str = "ransac_planes"
) -> Marker:
    """
    Create a visualization marker for a detected plane.
    
    Args:
        plane_coefficients: Plane coefficients [a, b, c, d]
        center: Center point of the plane [x, y, z]
        header: ROS2 Header
        marker_id: Unique ID for the marker
        size: (width, height) of the plane marker
        color: RGBA color tuple
        namespace: Marker namespace
        
    Returns:
        Marker message
    """
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    
    # Position at center
    marker.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
    
    # Orientation based on plane normal
    normal = plane_coefficients[:3]
    marker.pose.orientation = plane_to_quaternion(normal)
    
    # Scale (thin plane)
    marker.scale.x = size[0]
    marker.scale.y = size[1]
    marker.scale.z = 0.01  # Thin in Z direction
    
    # Color
    marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    
    # Lifetime (0 = forever)
    marker.lifetime.sec = 0
    marker.lifetime.nanosec = 0
    
    return marker


def create_line_marker_2d(
    line_coefficients: np.ndarray,
    points: np.ndarray,
    inlier_indices: np.ndarray,
    header: Header,
    marker_id: int,
    z_height: float = 0.0,
    color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    line_width: float = 0.05,
    namespace: str = "ransac_lines"
) -> Marker:
    """
    Create a visualization marker for a detected 2D line.
    
    Args:
        line_coefficients: Line coefficients [a, b, c]
        points: 2D points array
        inlier_indices: Indices of inlier points
        header: ROS2 Header
        marker_id: Unique ID for the marker
        z_height: Z coordinate for the line
        color: RGBA color tuple
        line_width: Width of the line marker
        namespace: Marker namespace
        
    Returns:
        Marker message
    """
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    
    # Get inlier points
    inlier_points = points[inlier_indices]
    
    # Find endpoints (min and max along line direction)
    a, b, c = line_coefficients
    direction = np.array([-b, a])  # Perpendicular to normal
    
    # Project points onto line direction
    projections = np.dot(inlier_points, direction)
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    
    # Create endpoints
    start = inlier_points[min_idx]
    end = inlier_points[max_idx]
    
    marker.points = [
        Point(x=float(start[0]), y=float(start[1]), z=z_height),
        Point(x=float(end[0]), y=float(end[1]), z=z_height)
    ]
    
    marker.scale.x = line_width  # Line width
    marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    
    return marker


def create_line_marker_3d(
    line_coefficients: np.ndarray,
    points: np.ndarray,
    inlier_indices: np.ndarray,
    header: Header,
    marker_id: int,
    color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    line_width: float = 0.05,
    namespace: str = "ransac_lines"
) -> Marker:
    """
    Create a visualization marker for a detected 3D line.
    
    Args:
        line_coefficients: Line coefficients [px, py, pz, dx, dy, dz]
        points: 3D points array
        inlier_indices: Indices of inlier points
        header: ROS2 Header
        marker_id: Unique ID for the marker
        color: RGBA color tuple
        line_width: Width of the line marker
        namespace: Marker namespace
        
    Returns:
        Marker message
    """
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    
    # Get inlier points
    inlier_points = points[inlier_indices]
    
    # Extract line parameters
    point_on_line = line_coefficients[:3]
    direction = line_coefficients[3:]
    
    # Project inlier points onto line to find endpoints
    v = inlier_points - point_on_line
    projections = np.dot(v, direction)
    
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    
    # Compute endpoints
    start = point_on_line + min_proj * direction
    end = point_on_line + max_proj * direction
    
    marker.points = [
        Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
        Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))
    ]
    
    marker.scale.x = line_width
    marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    
    return marker


def get_distinct_colors(n: int) -> List[Tuple[float, float, float, float]]:
    """
    Generate n visually distinct colors.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of RGBA tuples
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB (full saturation, full value)
        if hue < 1/6:
            r, g, b = 1.0, hue * 6, 0.0
        elif hue < 2/6:
            r, g, b = 1.0 - (hue - 1/6) * 6, 1.0, 0.0
        elif hue < 3/6:
            r, g, b = 0.0, 1.0, (hue - 2/6) * 6
        elif hue < 4/6:
            r, g, b = 0.0, 1.0 - (hue - 3/6) * 6, 1.0
        elif hue < 5/6:
            r, g, b = (hue - 4/6) * 6, 0.0, 1.0
        else:
            r, g, b = 1.0, 0.0, 1.0 - (hue - 5/6) * 6
        
        colors.append((r, g, b, 0.8))
    
    return colors


def compute_plane_size(
    points: np.ndarray,
    inlier_indices: np.ndarray,
    normal: np.ndarray,
    padding: float = 0.1
) -> Tuple[float, float]:
    """
    Compute appropriate size for plane marker based on inlier extent.
    
    Args:
        points: All points
        inlier_indices: Indices of inlier points
        normal: Plane normal vector
        padding: Extra padding to add to size
        
    Returns:
        Tuple of (width, height) for the plane marker
    """
    inlier_points = points[inlier_indices]
    
    # Create local coordinate system on plane
    z_up = np.array([0, 0, 1])
    if np.abs(np.dot(normal, z_up)) > 0.9:
        z_up = np.array([1, 0, 0])
    
    u = np.cross(normal, z_up)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Project points onto local coordinates
    center = np.mean(inlier_points, axis=0)
    centered = inlier_points - center
    
    u_coords = np.dot(centered, u)
    v_coords = np.dot(centered, v)
    
    width = np.max(u_coords) - np.min(u_coords) + padding * 2
    height = np.max(v_coords) - np.min(v_coords) + padding * 2
    
    return width, height
