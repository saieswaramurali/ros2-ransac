"""
3D LiDAR Example (Velodyne, Ouster, Livox).

This launch file starts a 3D LiDAR driver and the RANSAC perception
node to detect planes (ground, walls) from point cloud data.

Prerequisites (choose one):
    # For Velodyne
    sudo apt install ros-humble-velodyne
    
    # For Ouster
    sudo apt install ros-humble-ouster-ros
    
    # For Livox
    # Follow Livox ROS2 driver installation from GitHub

Usage:
    ros2 launch ransac_perception lidar_3d_example.launch.py
    ros2 launch ransac_perception lidar_3d_example.launch.py lidar_type:=velodyne
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    lidar_type = LaunchConfiguration('lidar_type').perform(context)
    
    nodes = []
    
    # Get package share path
    ransac_pkg = FindPackageShare('ransac_perception')
    config_file = PathJoinSubstitution([ransac_pkg, 'config', 'ransac_params.yaml'])
    
    # Select LiDAR driver based on type
    if lidar_type == 'velodyne':
        # Velodyne driver node
        lidar_driver = Node(
            package='velodyne_driver',
            executable='velodyne_driver_node',
            name='velodyne_driver',
            output='screen',
            parameters=[{
                'device_ip': '192.168.1.201',
                'model': 'VLP16',
                'rpm': 600.0,
            }]
        )
        # Velodyne pointcloud converter
        lidar_converter = Node(
            package='velodyne_pointcloud',
            executable='velodyne_convert_node',
            name='velodyne_convert',
            output='screen',
            parameters=[{
                'calibration': '',  # Use default calibration
            }]
        )
        nodes.extend([lidar_driver, lidar_converter])
        pointcloud_topic = '/velodyne_points'
        
    elif lidar_type == 'ouster':
        lidar_node = Node(
            package='ouster_ros',
            executable='ouster_driver',
            name='ouster_driver',
            output='screen',
            parameters=[{
                'sensor_hostname': 'os-122312345678.local',
                'lidar_mode': '1024x10',
            }]
        )
        nodes.append(lidar_node)
        pointcloud_topic = '/ouster/points'
        
    elif lidar_type == 'livox':
        lidar_node = Node(
            package='livox_ros2_driver',
            executable='livox_ros2_driver_node',
            name='livox_driver',
            output='screen',
            parameters=[{
                'bd_list': ['0TFDG3B00602171'],  # Replace with your broadcast code
                'frame_id': 'livox_frame',
            }]
        )
        nodes.append(lidar_node)
        pointcloud_topic = '/livox/lidar'
        
    else:
        # Default: use test publisher for demo
        lidar_node = Node(
            package='ransac_perception',
            executable='test_publisher',
            name='test_publisher',
            output='screen',
            parameters=[{
                'mode': 'pointcloud',
                'publish_rate': 10.0,
            }]
        )
        nodes.append(lidar_node)
        pointcloud_topic = '/input/pointcloud'
    
    # RANSAC perception node for plane detection
    ransac_node = Node(
        package='ransac_perception',
        executable='ransac_node',
        name='ransac_perception_node',
        output='screen',
        parameters=[
            config_file,
            {
                'input_mode': 'pointcloud',
                'detection_type': LaunchConfiguration('detection_type').perform(context),
                'distance_threshold': float(LaunchConfiguration('distance_threshold').perform(context)),
                'max_planes': int(LaunchConfiguration('max_planes').perform(context)),
                'max_points': 50000,
            }
        ],
        remappings=[
            ('input/pointcloud', pointcloud_topic),
        ]
    )
    nodes.append(ransac_node)
    
    return nodes


def generate_launch_description():
    # Declare launch arguments
    lidar_type_arg = DeclareLaunchArgument(
        'lidar_type',
        default_value='demo',
        description='LiDAR type: velodyne, ouster, livox, or demo (test data)'
    )
    
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
    
    max_planes_arg = DeclareLaunchArgument(
        'max_planes',
        default_value='5',
        description='Maximum number of planes to detect'
    )
    
    return LaunchDescription([
        lidar_type_arg,
        detection_type_arg,
        distance_threshold_arg,
        max_planes_arg,
        OpaqueFunction(function=launch_setup),
    ])
