"""
PointCloud2 Message Handler.

This module provides utilities for converting ROS2 PointCloud2 messages
to numpy arrays and back for RANSAC processing.
"""

import numpy as np
from typing import Optional, Tuple, List
import struct

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


# PointField type mappings
PFTYPE_SIZES = {
    PointField.INT8: 1,
    PointField.UINT8: 1,
    PointField.INT16: 2,
    PointField.UINT16: 2,
    PointField.INT32: 4,
    PointField.UINT32: 4,
    PointField.FLOAT32: 4,
    PointField.FLOAT64: 8,
}

PFTYPE_FORMATS = {
    PointField.INT8: 'b',
    PointField.UINT8: 'B',
    PointField.INT16: 'h',
    PointField.UINT16: 'H',
    PointField.INT32: 'i',
    PointField.UINT32: 'I',
    PointField.FLOAT32: 'f',
    PointField.FLOAT64: 'd',
}


class PointCloudHandler:
    """
    Handler for converting between ROS2 PointCloud2 messages and numpy arrays.
    """
    
    @staticmethod
    def pointcloud2_to_xyz(msg: PointCloud2) -> Optional[np.ndarray]:
        """
        Convert PointCloud2 message to numpy array of XYZ coordinates.
        
        Args:
            msg: PointCloud2 message
            
        Returns:
            Numpy array of shape (N, 3) with XYZ coordinates, or None if failed
        """
        # Find x, y, z field offsets
        field_map = {field.name: field for field in msg.fields}
        
        required_fields = ['x', 'y', 'z']
        for field_name in required_fields:
            if field_name not in field_map:
                return None
        
        x_field = field_map['x']
        y_field = field_map['y']
        z_field = field_map['z']
        
        # Determine endianness
        fmt_prefix = '>' if msg.is_bigendian else '<'
        
        # Calculate number of points
        if msg.height == 1:
            # Unorganized point cloud
            n_points = msg.width
        else:
            # Organized point cloud
            n_points = msg.width * msg.height
        
        # Pre-allocate output array
        points = np.zeros((n_points, 3), dtype=np.float32)
        
        # Extract points
        data = bytes(msg.data)
        point_step = msg.point_step
        
        x_offset = x_field.offset
        y_offset = y_field.offset
        z_offset = z_field.offset
        
        x_fmt = fmt_prefix + PFTYPE_FORMATS[x_field.datatype]
        y_fmt = fmt_prefix + PFTYPE_FORMATS[y_field.datatype]
        z_fmt = fmt_prefix + PFTYPE_FORMATS[z_field.datatype]
        
        x_size = PFTYPE_SIZES[x_field.datatype]
        y_size = PFTYPE_SIZES[y_field.datatype]
        z_size = PFTYPE_SIZES[z_field.datatype]
        
        for i in range(n_points):
            base = i * point_step
            points[i, 0] = struct.unpack(x_fmt, data[base + x_offset:base + x_offset + x_size])[0]
            points[i, 1] = struct.unpack(y_fmt, data[base + y_offset:base + y_offset + y_size])[0]
            points[i, 2] = struct.unpack(z_fmt, data[base + z_offset:base + z_offset + z_size])[0]
        
        return points
    
    @staticmethod
    def pointcloud2_to_xyz_fast(msg: PointCloud2) -> Optional[np.ndarray]:
        """
        Fast conversion using numpy for FLOAT32 XYZ fields.
        
        This is much faster than the generic method but only works for
        standard FLOAT32 XYZ point clouds.
        
        Args:
            msg: PointCloud2 message
            
        Returns:
            Numpy array of shape (N, 3) with XYZ coordinates, or None if failed
        """
        # Find x, y, z field offsets
        field_map = {field.name: field for field in msg.fields}
        
        required_fields = ['x', 'y', 'z']
        for field_name in required_fields:
            if field_name not in field_map:
                return None
        
        x_field = field_map['x']
        y_field = field_map['y']
        z_field = field_map['z']
        
        # Check if all fields are FLOAT32
        if not all(f.datatype == PointField.FLOAT32 
                   for f in [x_field, y_field, z_field]):
            # Fall back to generic method
            return PointCloudHandler.pointcloud2_to_xyz(msg)
        
        # Convert data to numpy array
        dtype = np.float32 if not msg.is_bigendian else np.dtype('>f4')
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        
        # Reshape to access individual points
        n_points = msg.width * msg.height if msg.height > 1 else msg.width
        
        # Create view as float32
        points = np.zeros((n_points, 3), dtype=np.float32)
        
        point_step = msg.point_step
        for i, field in enumerate([x_field, y_field, z_field]):
            offset = field.offset
            # Extract bytes for this field
            for j in range(n_points):
                start = j * point_step + offset
                points[j, i] = np.frombuffer(
                    data[start:start + 4].tobytes(), 
                    dtype=np.float32
                )[0]
        
        return points
    
    @staticmethod
    def filter_invalid_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out NaN and Inf values from point cloud.
        
        Args:
            points: Numpy array of shape (N, 3)
            
        Returns:
            Tuple of (filtered_points, valid_indices)
        """
        valid_mask = np.all(np.isfinite(points), axis=1)
        valid_indices = np.where(valid_mask)[0]
        return points[valid_mask], valid_indices
    
    @staticmethod
    def xyz_to_pointcloud2(
        points: np.ndarray,
        header: Header,
        fields: Optional[List[str]] = None
    ) -> PointCloud2:
        """
        Convert numpy array of XYZ coordinates to PointCloud2 message.
        
        Args:
            points: Numpy array of shape (N, 3) with XYZ coordinates
            header: ROS2 Header with timestamp and frame_id
            fields: Optional list of field names (default: ['x', 'y', 'z'])
            
        Returns:
            PointCloud2 message
        """
        if fields is None:
            fields = ['x', 'y', 'z']
        
        n_points = len(points)
        
        # Create PointField descriptors
        point_fields = []
        offset = 0
        for name in fields:
            point_fields.append(PointField(
                name=name,
                offset=offset,
                datatype=PointField.FLOAT32,
                count=1
            ))
            offset += 4  # FLOAT32 is 4 bytes
        
        # Create message
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n_points
        msg.fields = point_fields
        msg.is_bigendian = False
        msg.point_step = len(fields) * 4
        msg.row_step = msg.point_step * n_points
        msg.is_dense = True
        
        # Convert points to bytes
        msg.data = points.astype(np.float32).tobytes()
        
        return msg
    
    @staticmethod
    def create_colored_pointcloud(
        points: np.ndarray,
        colors: np.ndarray,
        header: Header
    ) -> PointCloud2:
        """
        Create PointCloud2 with RGB color information.
        
        Args:
            points: Numpy array of shape (N, 3) with XYZ coordinates
            colors: Numpy array of shape (N, 3) with RGB values (0-255)
            header: ROS2 Header
            
        Returns:
            PointCloud2 message with XYZRGB fields
        """
        n_points = len(points)
        
        # Create PointField descriptors
        point_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Pack RGB into float32
        rgb_packed = np.zeros(n_points, dtype=np.float32)
        colors = colors.astype(np.uint8)
        for i in range(n_points):
            r, g, b = colors[i]
            rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
            rgb_packed[i] = struct.unpack('f', struct.pack('I', rgb_int))[0]
        
        # Combine into single array
        data = np.zeros((n_points, 4), dtype=np.float32)
        data[:, :3] = points.astype(np.float32)
        data[:, 3] = rgb_packed
        
        # Create message
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n_points
        msg.fields = point_fields
        msg.is_bigendian = False
        msg.point_step = 16  # 4 floats * 4 bytes
        msg.row_step = msg.point_step * n_points
        msg.is_dense = True
        msg.data = data.tobytes()
        
        return msg
    
    @staticmethod
    def subsample_points(
        points: np.ndarray, 
        max_points: int = 10000,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly subsample points if there are too many.
        
        Args:
            points: Numpy array of shape (N, 3)
            max_points: Maximum number of points to keep
            random_seed: Optional seed for reproducibility
            
        Returns:
            Tuple of (subsampled_points, selected_indices)
        """
        if len(points) <= max_points:
            return points, np.arange(len(points))
        
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(points), max_points, replace=False)
        return points[indices], indices
