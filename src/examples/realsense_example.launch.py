"""
RealSense D435/D455 Depth Camera Example.

This launch file starts the Intel RealSense camera driver and
the RANSAC perception node to detect planes from depth data.

Prerequisites:
    sudo apt install ros-humble-realsense2-camera

Usage:
    ros2 launch ransac_perception realsense_example.launch.py
    ros2 launch ransac_perception realsense_example.launch.py detection_type:=multi_plane
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    detection_type_arg = DeclareLaunchArgument(
        'detection_type',
        default_value='multi_plane',
        description='Detection type: plane, multi_plane, or line'
    )
    
    distance_threshold_arg = DeclareLaunchArgument(
        'distance_threshold',
        default_value='0.02',
        description='Distance threshold for RANSAC (meters)'
    )
    
    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz',
        default_value='true',
        description='Launch RViz2 for visualization'
    )
    
    # Get package share paths
    ransac_pkg = FindPackageShare('ransac_perception')
    config_file = PathJoinSubstitution([ransac_pkg, 'config', 'ransac_params.yaml'])
    
    # RealSense camera node
    # Note: You may need to adjust these parameters for your camera model
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense_camera',
        output='screen',
        parameters=[{
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'pointcloud.enable': True,
            'align_depth.enable': True,
        }]
    )
    
    # RANSAC perception node
    ransac_node = Node(
        package='ransac_perception',
        executable='ransac_node',
        name='ransac_perception_node',
        output='screen',
        parameters=[
            config_file,
            {
                'input_mode': 'pointcloud',
                'detection_type': LaunchConfiguration('detection_type'),
                'distance_threshold': LaunchConfiguration('distance_threshold'),
                'max_points': 30000,  # Limit for real-time performance
            }
        ],
        remappings=[
            ('input/pointcloud', '/camera/depth/color/points'),
        ]
    )
    
    # RViz2 node (optional)
    rviz_config_file = PathJoinSubstitution([ransac_pkg, 'config', 'realsense_rviz.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=LaunchConfiguration('enable_rviz')
    )
    
    return LaunchDescription([
        detection_type_arg,
        distance_threshold_arg,
        enable_rviz_arg,
        realsense_node,
        ransac_node,
        # rviz_node,  # Uncomment when RViz config is created
    ])
