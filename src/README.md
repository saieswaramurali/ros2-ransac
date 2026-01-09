# RANSAC Perception

<p align="center">
  <img src="https://img.shields.io/badge/ROS2-Humble-blue" alt="ROS2 Humble">
  <img src="https://img.shields.io/badge/Python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License">
</p>

A ROS2 Python package for RANSAC-based geometric primitive detection in perception pipelines. Supports depth cameras (PointCloud2) and 2D LiDAR (LaserScan) inputs.

## Features

- 🔷 **Plane Detection**: Detect ground planes, walls, and surfaces from 3D point clouds
- 📏 **Line Detection**: Detect linear features in 2D laser scans or 3D point clouds
- 🔄 **Multi-Plane Detection**: Sequentially detect multiple planes in a scene
- 📊 **Visualization**: RViz markers for detected primitives
- ⚡ **Efficient**: NumPy-based implementation for fast processing
- 🔧 **Configurable**: Full parameter configuration via YAML or launch arguments

## Supported Sensors

| Input Type | Message Type | Use Case |
|------------|--------------|----------|
| Depth Cameras | `sensor_msgs/PointCloud2` | Intel RealSense, Kinect, ZED |
| 3D LiDAR | `sensor_msgs/PointCloud2` | Velodyne, Ouster, Livox |
| 2D LiDAR | `sensor_msgs/LaserScan` | RPLIDAR, SICK, Hokuyo |

## Installation

### Prerequisites

- ROS2 Humble (Ubuntu 22.04)
- Python 3.10+
- NumPy

### Build from Source

```bash
# Navigate to your ROS2 workspace
cd ~/ros2_ws/src

# Clone the repository
git clone https://github.com/saieswaramurali/ros2-ransac.git

# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build the package
colcon build --packages-select ransac_perception

# Source the workspace
source install/setup.bash
```

## Quick Start

### Demo with Synthetic Data

```bash
# Launch demo with test data publisher
ros2 launch ransac_perception demo.launch.py
```

### With Real Sensor Data

```bash
# Run with default settings
ros2 run ransac_perception ransac_node

# Or use launch file with topic remapping
ros2 launch ransac_perception ransac.launch.py \
    pointcloud_topic:=/camera/depth/points
```

### Visualize in RViz2

```bash
# In a separate terminal
rviz2 -d $(ros2 pkg prefix ransac_perception)/share/ransac_perception/config/ransac_rviz.rviz
```

## Sensor Examples

Pre-configured launch files for common sensors are provided in the `examples/` folder.

### Intel RealSense (D435/D455)

```bash
# Install RealSense driver
sudo apt install ros-humble-realsense2-camera

# Launch with RealSense
ros2 launch ransac_perception realsense_example.launch.py

# With multi-plane detection
ros2 launch ransac_perception realsense_example.launch.py detection_type:=multi_plane
```

### 2D LiDAR (RPLIDAR, Hokuyo, SICK)

```bash
# Install your LiDAR driver
sudo apt install ros-humble-rplidar-ros  # For RPLIDAR

# Launch with 2D LiDAR
ros2 launch ransac_perception lidar_2d_example.launch.py lidar_type:=rplidar

# Or demo mode (no hardware needed)
ros2 launch ransac_perception lidar_2d_example.launch.py lidar_type:=demo
```

### 3D LiDAR (Velodyne, Ouster, Livox)

```bash
# Install your LiDAR driver
sudo apt install ros-humble-velodyne  # For Velodyne

# Launch with 3D LiDAR
ros2 launch ransac_perception lidar_3d_example.launch.py lidar_type:=velodyne

# Or demo mode (no hardware needed)
ros2 launch ransac_perception lidar_3d_example.launch.py lidar_type:=demo
```

## Topics

### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `input/pointcloud` | `sensor_msgs/PointCloud2` | Input 3D point cloud |
| `input/scan` | `sensor_msgs/LaserScan` | Input 2D laser scan |

### Publications

| Topic | Type | Description |
|-------|------|-------------|
| `ransac/inliers` | `sensor_msgs/PointCloud2` | Inlier points |
| `ransac/outliers` | `sensor_msgs/PointCloud2` | Outlier points |
| `ransac/segmented` | `sensor_msgs/PointCloud2` | Color-coded segmented cloud |
| `ransac/markers` | `visualization_msgs/MarkerArray` | Visualization markers |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_mode` | string | `"auto"` | Input mode: `pointcloud`, `laserscan`, `auto` |
| `detection_type` | string | `"plane"` | Type: `plane`, `multi_plane`, `line` |
| `max_iterations` | int | `1000` | Maximum RANSAC iterations |
| `distance_threshold` | float | `0.01` | Inlier distance threshold (meters) |
| `min_inliers_ratio` | float | `0.3` | Minimum inlier ratio for valid model |
| `max_planes` | int | `5` | Max planes for multi-plane detection |
| `max_points` | int | `50000` | Subsample if more points |
| `publish_visualization` | bool | `true` | Publish RViz markers |

## Usage Examples

### Single Plane Detection (Ground/Floor)

```bash
ros2 run ransac_perception ransac_node --ros-args \
    -p detection_type:=plane \
    -p distance_threshold:=0.02 \
    -r input/pointcloud:=/camera/depth/points
```

### Multi-Plane Detection (Room Segmentation)

```bash
ros2 run ransac_perception ransac_node --ros-args \
    -p detection_type:=multi_plane \
    -p max_planes:=10 \
    -p distance_threshold:=0.03
```

### Line Detection from LaserScan

```bash
ros2 run ransac_perception ransac_node --ros-args \
    -p input_mode:=laserscan \
    -p detection_type:=line \
    -r input/scan:=/scan
```

### In a Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ransac_perception',
            executable='ransac_node',
            parameters=[{
                'detection_type': 'multi_plane',
                'distance_threshold': 0.02,
                'max_planes': 5
            }],
            remappings=[
                ('input/pointcloud', '/camera/depth/points')
            ]
        )
    ])
```

## Algorithm Details

### RANSAC (RANdom SAmple Consensus)

The RANSAC algorithm robustly fits geometric models to data with outliers:

1. **Sample**: Randomly select minimum points to define model (3 for plane, 2 for line)
2. **Fit**: Compute model parameters from sample
3. **Score**: Count inliers (points within `distance_threshold` of model)
4. **Iterate**: Repeat and keep best model
5. **Refine**: Optionally refit model using all inliers

### Plane Equation

Planes are represented as: `ax + by + cz + d = 0`

Where `(a, b, c)` is the unit normal vector and `d` is the signed distance from origin.

### Line Equation (2D)

Lines are represented as: `ax + by + c = 0`

Where `(a, b)` is the unit normal to the line.

## Architecture

```
ransac_perception/
├── ransac_core.py         # Core RANSAC algorithms
├── point_cloud_handler.py # PointCloud2 ↔ NumPy conversion
├── laser_scan_handler.py  # LaserScan ↔ NumPy conversion
├── ransac_node.py         # Main ROS2 node
├── test_publisher.py      # Synthetic data generator
└── utils.py               # Visualization utilities
```

## Performance Tuning

### For Large Point Clouds
- Reduce `max_points` to subsample input
- Decrease `max_iterations` for faster processing

### For Noisy Data
- Increase `distance_threshold` to accommodate noise
- Decrease `min_inliers_ratio` if planes are small

### For Multiple Small Planes
- Use `detection_type: multi_plane`
- Increase `max_planes`
- Decrease `min_inliers_ratio`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**SAI ESWARA M**  
📧 saimurali2005@gmail.com  
🔗 [GitHub](https://github.com/saieswaramurali)

## Acknowledgments

- ROS2 community
- Open3D for inspiration on point cloud processing
- pyRANSAC-3D for algorithm reference

## Citation

If you use this package in your research, please cite the original RANSAC paper:

```bibtex
@article{fischler1981random,
  title={Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography},
  author={Fischler, Martin A and Bolles, Robert C},
  journal={Communications of the ACM},
  volume={24},
  number={6},
  pages={381--395},
  year={1981},
  publisher={ACM New York, NY, USA}
}
```

And optionally cite this ROS2 package:

```bibtex
@software{ransac_perception,
  title = {RANSAC Perception: ROS2 RANSAC-based Geometric Primitive Detection},
  author = {SAI ESWARA M},
  year = {2024},
  url = {https://github.com/saieswaramurali/ros2-ransac}
}
```
