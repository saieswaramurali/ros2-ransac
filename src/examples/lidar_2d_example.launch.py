"""
2D LiDAR Example (RPLIDAR, SICK, Hokuyo).

This launch file starts a 2D LiDAR driver and the RANSAC perception
node to detect lines (walls) from laser scan data.

Prerequisites (choose one):
    # For RPLIDAR
    sudo apt install ros-humble-rplidar-ros
    
    # For SICK
    sudo apt install ros-humble-sick-scan2
    
    # For Hokuyo
    sudo apt install ros-humble-urg-node

Usage:
    ros2 launch ransac_perception lidar_2d_example.launch.py
    ros2 launch ransac_perception lidar_2d_example.launch.py lidar_type:=rplidar
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
    if lidar_type == 'rplidar':
        lidar_node = Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': 115200,
                'frame_id': 'laser_frame',
                'angle_compensate': True,
                'scan_mode': 'Standard',
            }]
        )
        scan_topic = '/scan'
    elif lidar_type == 'hokuyo':
        lidar_node = Node(
            package='urg_node',
            executable='urg_node_driver',
            name='urg_node',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyACM0',
                'frame_id': 'laser_frame',
            }]
        )
        scan_topic = '/scan'
    elif lidar_type == 'sick':
        lidar_node = Node(
            package='sick_scan2',
            executable='sick_generic_caller',
            name='sick_scan',
            output='screen',
            parameters=[{
                'scanner_type': 'sick_tim_5xx',
                'hostname': '192.168.0.1',
            }]
        )
        scan_topic = '/scan'
    else:
        # Default: use test publisher for demo
        lidar_node = Node(
            package='ransac_perception',
            executable='test_publisher',
            name='test_publisher',
            output='screen',
            parameters=[{
                'mode': 'laserscan',
                'publish_rate': 10.0,
            }]
        )
        scan_topic = '/input/scan'
    
    nodes.append(lidar_node)
    
    # RANSAC perception node for line detection
    ransac_node = Node(
        package='ransac_perception',
        executable='ransac_node',
        name='ransac_perception_node',
        output='screen',
        parameters=[
            config_file,
            {
                'input_mode': 'laserscan',
                'detection_type': 'line',
                'distance_threshold': float(LaunchConfiguration('distance_threshold').perform(context)),
            }
        ],
        remappings=[
            ('input/scan', scan_topic),
        ]
    )
    nodes.append(ransac_node)
    
    return nodes


def generate_launch_description():
    # Declare launch arguments
    lidar_type_arg = DeclareLaunchArgument(
        'lidar_type',
        default_value='demo',
        description='LiDAR type: rplidar, hokuyo, sick, or demo (test data)'
    )
    
    distance_threshold_arg = DeclareLaunchArgument(
        'distance_threshold',
        default_value='0.05',
        description='Distance threshold for RANSAC line detection (meters)'
    )
    
    return LaunchDescription([
        lidar_type_arg,
        distance_threshold_arg,
        OpaqueFunction(function=launch_setup),
    ])
