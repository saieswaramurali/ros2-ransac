"""
Demo Launch File with Test Publisher.

Launches both the RANSAC perception node and test publisher
for demonstration and testing without real sensors.

Usage:
    ros2 launch ransac_perception demo.launch.py
    ros2 launch ransac_perception demo.launch.py detection_type:=multi_plane
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    detection_type_arg = DeclareLaunchArgument(
        'detection_type',
        default_value='multi_plane',
        description='Detection type: plane, multi_plane, or line'
    )
    
    test_mode_arg = DeclareLaunchArgument(
        'test_mode',
        default_value='pointcloud',
        description='Test publisher mode: pointcloud, laserscan, or both'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='10.0',
        description='Test data publish rate (Hz)'
    )
    
    # Get package share directory for config
    pkg_share = FindPackageShare('ransac_perception')
    config_file = PathJoinSubstitution([pkg_share, 'config', 'ransac_params.yaml'])
    
    # Test publisher node
    test_publisher = Node(
        package='ransac_perception',
        executable='test_publisher',
        name='ransac_test_publisher',
        output='screen',
        parameters=[{
            'mode': LaunchConfiguration('test_mode'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'noise_level': 0.01,
            'outlier_ratio': 0.1,
        }]
    )
    
    # RANSAC perception node (delayed to let test publisher start first)
    ransac_node = TimerAction(
        period=1.0,  # 1 second delay
        actions=[
            Node(
                package='ransac_perception',
                executable='ransac_node',
                name='ransac_perception_node',
                output='screen',
                parameters=[
                    config_file,
                    {
                        'input_mode': 'auto',
                        'detection_type': LaunchConfiguration('detection_type'),
                    }
                ]
            )
        ]
    )
    
    return LaunchDescription([
        detection_type_arg,
        test_mode_arg,
        publish_rate_arg,
        test_publisher,
        ransac_node,
    ])
