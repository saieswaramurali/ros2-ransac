"""
Main RANSAC Perception ROS2 Node.

This node subscribes to sensor data (PointCloud2 or LaserScan) and runs
RANSAC algorithms to detect geometric primitives.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from typing import Optional

from sensor_msgs.msg import PointCloud2, LaserScan
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header

from .ransac_core import RANSACPlane, RANSACLine2D, RANSACLine3D, MultiPlaneRANSAC, RANSACResult
from .point_cloud_handler import PointCloudHandler
from .laser_scan_handler import LaserScanHandler
from .utils import (
    create_plane_marker, 
    create_line_marker_2d, 
    create_line_marker_3d,
    get_distinct_colors,
    compute_plane_size
)


class RANSACPerceptionNode(Node):
    """
    ROS2 Node for RANSAC-based geometric primitive detection.
    
    Supports both PointCloud2 (depth cameras, 3D LiDAR) and 
    LaserScan (2D LiDAR) inputs.
    """
    
    def __init__(self):
        super().__init__('ransac_perception_node')
        
        # Declare parameters
        self._declare_parameters()
        
        # Get parameters
        self.input_mode = self.get_parameter('input_mode').value
        self.detection_type = self.get_parameter('detection_type').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.min_inliers_ratio = self.get_parameter('min_inliers_ratio').value
        self.max_planes = self.get_parameter('max_planes').value
        self.max_points = self.get_parameter('max_points').value
        self.publish_visualization = self.get_parameter('publish_visualization').value
        
        # Initialize RANSAC algorithms
        self._init_ransac()
        
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscriptions based on input mode
        if self.input_mode in ['pointcloud', 'auto']:
            self.pointcloud_sub = self.create_subscription(
                PointCloud2,
                'input/pointcloud',
                self.pointcloud_callback,
                sensor_qos
            )
            self.get_logger().info('Subscribed to input/pointcloud')
        
        if self.input_mode in ['laserscan', 'auto']:
            self.laserscan_sub = self.create_subscription(
                LaserScan,
                'input/scan',
                self.laserscan_callback,
                sensor_qos
            )
            self.get_logger().info('Subscribed to input/scan')
        
        # Publishers
        self.inliers_pub = self.create_publisher(PointCloud2, 'ransac/inliers', 10)
        self.outliers_pub = self.create_publisher(PointCloud2, 'ransac/outliers', 10)
        self.markers_pub = self.create_publisher(MarkerArray, 'ransac/markers', 10)
        
        # For multi-plane, publish segmented clouds with colors
        self.segmented_pub = self.create_publisher(PointCloud2, 'ransac/segmented', 10)
        
        # Marker ID counter
        self.marker_id = 0
        
        self.get_logger().info(
            f'RANSAC Perception Node initialized\n'
            f'  Input mode: {self.input_mode}\n'
            f'  Detection type: {self.detection_type}\n'
            f'  Max iterations: {self.max_iterations}\n'
            f'  Distance threshold: {self.distance_threshold}'
        )
    
    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        self.declare_parameter('input_mode', 'auto')  # 'pointcloud', 'laserscan', 'auto'
        self.declare_parameter('detection_type', 'plane')  # 'plane', 'line', 'multi_plane'
        self.declare_parameter('max_iterations', 1000)
        self.declare_parameter('distance_threshold', 0.01)
        self.declare_parameter('min_inliers_ratio', 0.3)
        self.declare_parameter('max_planes', 5)
        self.declare_parameter('max_points', 50000)  # Subsample if more points
        self.declare_parameter('publish_visualization', True)
    
    def _init_ransac(self):
        """Initialize RANSAC algorithm instances."""
        if self.detection_type == 'plane':
            self.ransac_plane = RANSACPlane(
                max_iterations=self.max_iterations,
                distance_threshold=self.distance_threshold,
                min_inliers_ratio=self.min_inliers_ratio
            )
        elif self.detection_type == 'multi_plane':
            self.ransac_multi = MultiPlaneRANSAC(
                max_planes=self.max_planes,
                max_iterations=self.max_iterations,
                distance_threshold=self.distance_threshold,
                min_inliers_ratio=self.min_inliers_ratio
            )
        elif self.detection_type == 'line':
            self.ransac_line_2d = RANSACLine2D(
                max_iterations=self.max_iterations,
                distance_threshold=self.distance_threshold,
                min_inliers_ratio=self.min_inliers_ratio
            )
            self.ransac_line_3d = RANSACLine3D(
                max_iterations=self.max_iterations,
                distance_threshold=self.distance_threshold,
                min_inliers_ratio=self.min_inliers_ratio
            )
    
    def pointcloud_callback(self, msg: PointCloud2):
        """
        Process incoming PointCloud2 message.
        
        Args:
            msg: PointCloud2 message
        """
        # Convert to numpy array
        points = PointCloudHandler.pointcloud2_to_xyz_fast(msg)
        if points is None:
            self.get_logger().warn('Failed to convert PointCloud2 to numpy array')
            return
        
        # Filter invalid points
        points, valid_indices = PointCloudHandler.filter_invalid_points(points)
        if len(points) < 10:
            self.get_logger().warn('Not enough valid points for RANSAC')
            return
        
        # Subsample if too many points
        if len(points) > self.max_points:
            points, sample_indices = PointCloudHandler.subsample_points(points, self.max_points)
            # Update valid_indices to reflect subsampling
            valid_indices = valid_indices[sample_indices]
        
        self.get_logger().debug(f'Processing {len(points)} points')
        
        # Run RANSAC based on detection type
        if self.detection_type == 'plane':
            self._process_single_plane(points, msg.header)
        elif self.detection_type == 'multi_plane':
            self._process_multi_plane(points, msg.header)
        elif self.detection_type == 'line':
            self._process_line_3d(points, msg.header)
    
    def laserscan_callback(self, msg: LaserScan):
        """
        Process incoming LaserScan message.
        
        Args:
            msg: LaserScan message
        """
        # Convert to 2D Cartesian coordinates
        points, valid_indices = LaserScanHandler.laserscan_to_cartesian(msg)
        if len(points) < 5:
            self.get_logger().warn('Not enough valid points in LaserScan')
            return
        
        self.get_logger().debug(f'Processing {len(points)} laser scan points')
        
        # Create header for output
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id
        
        # Run line RANSAC for 2D data
        if self.detection_type in ['line', 'plane']:
            self._process_line_2d(points, header)
    
    def _process_single_plane(self, points: np.ndarray, header: Header):
        """Process points for single plane detection."""
        result = self.ransac_plane.fit(points)
        
        if result is None:
            self.get_logger().debug('No plane found')
            return
        
        self.get_logger().info(
            f'Plane detected: normal=[{result.coefficients[0]:.3f}, '
            f'{result.coefficients[1]:.3f}, {result.coefficients[2]:.3f}], '
            f'inliers={len(result.inlier_indices)}/{len(points)} '
            f'({result.inlier_ratio:.1%})'
        )
        
        # Publish inliers
        inlier_points = points[result.inlier_indices]
        inliers_msg = PointCloudHandler.xyz_to_pointcloud2(inlier_points, header)
        self.inliers_pub.publish(inliers_msg)
        
        # Publish outliers
        outlier_points = points[result.outlier_indices]
        outliers_msg = PointCloudHandler.xyz_to_pointcloud2(outlier_points, header)
        self.outliers_pub.publish(outliers_msg)
        
        # Publish visualization marker
        if self.publish_visualization:
            self._publish_plane_markers([result], points, header)
    
    def _process_multi_plane(self, points: np.ndarray, header: Header):
        """Process points for multi-plane detection."""
        results = self.ransac_multi.fit(points)
        
        if not results:
            self.get_logger().debug('No planes found')
            return
        
        self.get_logger().info(f'Detected {len(results)} planes')
        
        # Collect all inliers for coloring
        all_inlier_indices = set()
        for result in results:
            all_inlier_indices.update(result.inlier_indices)
        
        # Create colored point cloud
        colors = get_distinct_colors(len(results) + 1)  # +1 for outliers
        point_colors = np.zeros((len(points), 3), dtype=np.uint8)
        
        # Color outliers gray
        outlier_color = np.array([128, 128, 128], dtype=np.uint8)
        point_colors[:] = outlier_color
        
        # Color each plane
        for i, result in enumerate(results):
            color = colors[i]
            rgb = np.array([int(color[0]*255), int(color[1]*255), int(color[2]*255)], dtype=np.uint8)
            point_colors[result.inlier_indices] = rgb
        
        # Publish segmented cloud
        segmented_msg = PointCloudHandler.create_colored_pointcloud(points, point_colors, header)
        self.segmented_pub.publish(segmented_msg)
        
        # Publish markers
        if self.publish_visualization:
            self._publish_plane_markers(results, points, header)
    
    def _process_line_2d(self, points: np.ndarray, header: Header):
        """Process 2D points for line detection."""
        result = self.ransac_line_2d.fit(points)
        
        if result is None:
            self.get_logger().debug('No line found in LaserScan')
            return
        
        self.get_logger().info(
            f'Line detected: coefficients=[{result.coefficients[0]:.3f}, '
            f'{result.coefficients[1]:.3f}, {result.coefficients[2]:.3f}], '
            f'inliers={len(result.inlier_indices)}/{len(points)}'
        )
        
        # Publish markers
        if self.publish_visualization:
            marker_array = MarkerArray()
            
            # Clear previous markers
            clear_marker = Marker()
            clear_marker.header = header
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            
            # Add line marker
            marker = create_line_marker_2d(
                result.coefficients,
                points,
                result.inlier_indices,
                header,
                marker_id=0,
                z_height=0.0
            )
            marker_array.markers.append(marker)
            
            self.markers_pub.publish(marker_array)
    
    def _process_line_3d(self, points: np.ndarray, header: Header):
        """Process 3D points for line detection."""
        result = self.ransac_line_3d.fit(points)
        
        if result is None:
            self.get_logger().debug('No line found')
            return
        
        self.get_logger().info(
            f'3D Line detected: inliers={len(result.inlier_indices)}/{len(points)}'
        )
        
        # Publish inliers
        inlier_points = points[result.inlier_indices]
        inliers_msg = PointCloudHandler.xyz_to_pointcloud2(inlier_points, header)
        self.inliers_pub.publish(inliers_msg)
        
        # Publish markers
        if self.publish_visualization:
            marker_array = MarkerArray()
            
            marker = create_line_marker_3d(
                result.coefficients,
                points,
                result.inlier_indices,
                header,
                marker_id=0
            )
            marker_array.markers.append(marker)
            
            self.markers_pub.publish(marker_array)
    
    def _publish_plane_markers(
        self, 
        results: list, 
        points: np.ndarray, 
        header: Header
    ):
        """Publish visualization markers for detected planes."""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header = header
        clear_marker.ns = "ransac_planes"
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        colors = get_distinct_colors(len(results))
        
        for i, result in enumerate(results):
            center = np.mean(points[result.inlier_indices], axis=0)
            normal = result.coefficients[:3]
            
            # Compute appropriate size
            size = compute_plane_size(points, result.inlier_indices, normal)
            
            marker = create_plane_marker(
                result.coefficients,
                center,
                header,
                marker_id=i,
                size=size,
                color=colors[i]
            )
            marker_array.markers.append(marker)
        
        self.markers_pub.publish(marker_array)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = RANSACPerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
