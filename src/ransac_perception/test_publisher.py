"""
Test Publisher Node for RANSAC Perception.

This node publishes synthetic point cloud and laser scan data
for testing the RANSAC perception pipeline.
"""

import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2, LaserScan
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

from .point_cloud_handler import PointCloudHandler


class TestPublisher(Node):
    """
    Publishes synthetic sensor data for testing RANSAC algorithms.
    """
    
    def __init__(self):
        super().__init__('ransac_test_publisher')
        
        # Declare parameters
        self.declare_parameter('mode', 'pointcloud')  # 'pointcloud', 'laserscan', 'both'
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('noise_level', 0.01)
        self.declare_parameter('outlier_ratio', 0.1)
        
        self.mode = self.get_parameter('mode').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.noise_level = self.get_parameter('noise_level').value
        self.outlier_ratio = self.get_parameter('outlier_ratio').value
        
        # Publishers
        if self.mode in ['pointcloud', 'both']:
            self.pointcloud_pub = self.create_publisher(
                PointCloud2, 
                'input/pointcloud', 
                10
            )
        
        if self.mode in ['laserscan', 'both']:
            self.laserscan_pub = self.create_publisher(
                LaserScan, 
                'input/scan', 
                10
            )
        
        # Timer
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        
        # Random number generator
        self.rng = np.random.default_rng(42)
        
        # Animation state
        self.frame_count = 0
        
        self.get_logger().info(
            f'Test Publisher initialized\n'
            f'  Mode: {self.mode}\n'
            f'  Rate: {self.publish_rate} Hz\n'
            f'  Noise level: {self.noise_level}\n'
            f'  Outlier ratio: {self.outlier_ratio}'
        )
    
    def timer_callback(self):
        """Publish synthetic data."""
        self.frame_count += 1
        
        if self.mode in ['pointcloud', 'both']:
            self._publish_pointcloud()
        
        if self.mode in ['laserscan', 'both']:
            self._publish_laserscan()
    
    def _publish_pointcloud(self):
        """Publish synthetic point cloud with planes."""
        # Create a scene with multiple planes
        points = []
        
        # Ground plane (z = 0)
        n_ground = 2000
        x = self.rng.uniform(-2, 2, n_ground)
        y = self.rng.uniform(-2, 2, n_ground)
        z = np.zeros(n_ground) + self.rng.normal(0, self.noise_level, n_ground)
        points.append(np.column_stack([x, y, z]))
        
        # Wall plane (y = 1.5)
        n_wall = 1000
        x = self.rng.uniform(-2, 2, n_wall)
        z = self.rng.uniform(0, 1.5, n_wall)
        y = np.full(n_wall, 1.5) + self.rng.normal(0, self.noise_level, n_wall)
        points.append(np.column_stack([x, y, z]))
        
        # Tilted plane rotating over time
        angle = self.frame_count * 0.02
        n_tilted = 800
        u = self.rng.uniform(-1, 1, n_tilted)
        v = self.rng.uniform(-1, 1, n_tilted)
        
        # Plane center
        cx, cy, cz = 0, 0, 1.0
        
        # Plane normal (rotating)
        nx = np.sin(angle) * 0.5
        ny = np.cos(angle) * 0.5
        nz = 0.7
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        
        # Create orthogonal basis
        if abs(nz) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        
        right = np.cross(up, [nx, ny, nz])
        right = right / np.linalg.norm(right)
        forward = np.cross([nx, ny, nz], right)
        
        # Generate points on plane
        x = cx + u * right[0] + v * forward[0]
        y = cy + u * right[1] + v * forward[1]
        z = cz + u * right[2] + v * forward[2]
        
        # Add noise
        x += self.rng.normal(0, self.noise_level, n_tilted)
        y += self.rng.normal(0, self.noise_level, n_tilted)
        z += self.rng.normal(0, self.noise_level, n_tilted)
        
        points.append(np.column_stack([x, y, z]))
        
        # Combine all plane points
        all_points = np.vstack(points)
        
        # Add outliers
        n_outliers = int(len(all_points) * self.outlier_ratio)
        outliers = self.rng.uniform(-2, 2, (n_outliers, 3))
        all_points = np.vstack([all_points, outliers])
        
        # Create header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'
        
        # Create and publish message
        msg = PointCloudHandler.xyz_to_pointcloud2(all_points.astype(np.float32), header)
        self.pointcloud_pub.publish(msg)
        
        self.get_logger().debug(f'Published point cloud with {len(all_points)} points')
    
    def _publish_laserscan(self):
        """Publish synthetic laser scan with lines/walls."""
        msg = LaserScan()
        
        # Header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        # Scan parameters
        msg.angle_min = -np.pi
        msg.angle_max = np.pi
        msg.angle_increment = np.pi / 180  # 1 degree resolution
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.1
        msg.range_max = 10.0
        
        n_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment)
        angles = np.linspace(msg.angle_min, msg.angle_max, n_readings)
        
        # Create a room with walls
        ranges = np.full(n_readings, msg.range_max)
        
        # Front wall at y = 3
        # Distance to wall from origin: d = 3 / cos(angle) for |angle| < pi/2
        front_mask = np.abs(angles) < np.pi/2
        front_ranges = 3.0 / np.cos(angles[front_mask])
        ranges[front_mask] = np.minimum(ranges[front_mask], front_ranges)
        
        # Back wall at y = -3
        back_mask = np.abs(angles) > np.pi/2
        back_ranges = 3.0 / np.abs(np.cos(angles[back_mask]))
        ranges[back_mask] = np.minimum(ranges[back_mask], back_ranges)
        
        # Left wall at x = -2
        left_mask = (angles > np.pi/2) | (angles < -np.pi/2)
        left_ranges = 2.0 / np.abs(np.sin(angles + 1e-6))
        left_ranges[~left_mask] = msg.range_max
        ranges = np.minimum(ranges, left_ranges)
        
        # Right wall at x = 4
        right_mask = np.abs(angles) < np.pi/2
        right_ranges = 4.0 / np.abs(np.sin(angles + 1e-6))
        right_ranges[~right_mask] = msg.range_max
        ranges = np.minimum(ranges, right_ranges)
        
        # Add noise
        ranges += self.rng.normal(0, self.noise_level, n_readings)
        
        # Clip to valid range
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        
        # Add some outliers (invalid readings)
        n_outliers = int(n_readings * self.outlier_ratio)
        outlier_indices = self.rng.choice(n_readings, n_outliers, replace=False)
        ranges[outlier_indices] = self.rng.uniform(0, msg.range_min, n_outliers)
        
        msg.ranges = ranges.tolist()
        msg.intensities = []  # No intensity data
        
        self.laserscan_pub.publish(msg)
        
        self.get_logger().debug(f'Published laser scan with {n_readings} readings')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = TestPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
