"""
Main Launch File for RANSAC Perception Node.

Usage:
    ros2 launch ransac_perception ransac.launch.py
    ros2 launch ransac_perception ransac.launch.py detection_type:=multi_plane
    ros2 launch ransac_perception ransac.launch.py input_topic:=/camera/depth/points
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    input_mode_arg = DeclareLaunchArgument(
        'input_mode',
        default_value='auto',
        description='Input mode: pointcloud, laserscan, or auto'
    )
    
    detection_type_arg = DeclareLaunchArgument(
        'detection_type',
        default_value='plane',
        description='Detection type: plane, multi_plane, or line'
    )
    
    distance_threshold_arg = DeclareLaunchArgument(
        'distance_threshold',
        default_value='0.01',
        description='Distance threshold for RANSAC (meters)'
    )
    
    max_iterations_arg = DeclareLaunchArgument(
        'max_iterations',
        default_value='1000',
        description='Maximum RANSAC iterations'
    )
    
    pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/input/pointcloud',
        description='Input PointCloud2 topic'
    )
    
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/input/scan',
        description='Input LaserScan topic'
    )
    
    # Get package share directory for config
    pkg_share = FindPackageShare('ransac_perception')
    config_file = PathJoinSubstitution([pkg_share, 'config', 'ransac_params.yaml'])
    
    # RANSAC perception node
    ransac_node = Node(
        package='ransac_perception',
        executable='ransac_node',
        name='ransac_perception_node',
        output='screen',
        parameters=[
            config_file,
            {
                'input_mode': LaunchConfiguration('input_mode'),
                'detection_type': LaunchConfiguration('detection_type'),
                'distance_threshold': LaunchConfiguration('distance_threshold'),
                'max_iterations': LaunchConfiguration('max_iterations'),
            }
        ],
        remappings=[
            ('input/pointcloud', LaunchConfiguration('pointcloud_topic')),
            ('input/scan', LaunchConfiguration('scan_topic')),
        ]
    )
    
    return LaunchDescription([
        input_mode_arg,
        detection_type_arg,
        distance_threshold_arg,
        max_iterations_arg,
        pointcloud_topic_arg,
        scan_topic_arg,
        ransac_node,
    ])
