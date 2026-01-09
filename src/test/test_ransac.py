"""
Unit tests for RANSAC algorithms.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ransac_perception.ransac_core import (
    RANSACPlane, 
    RANSACLine2D, 
    RANSACLine3D,
    MultiPlaneRANSAC
)


class TestRANSACPlane:
    """Tests for RANSACPlane algorithm."""
    
    def test_fit_horizontal_plane(self):
        """Test fitting a horizontal plane (z = 0)."""
        # Generate points on z = 0 plane with some noise
        rng = np.random.default_rng(42)
        n_points = 500
        x = rng.uniform(-5, 5, n_points)
        y = rng.uniform(-5, 5, n_points)
        z = rng.normal(0, 0.01, n_points)  # Small noise around z=0
        
        points = np.column_stack([x, y, z])
        
        ransac = RANSACPlane(
            max_iterations=100,
            distance_threshold=0.05,
            min_inliers_ratio=0.8
        )
        
        result = ransac.fit(points)
        
        assert result is not None
        assert result.inlier_ratio > 0.9
        
        # Normal should be close to [0, 0, 1] or [0, 0, -1]
        normal = result.coefficients[:3]
        assert abs(abs(normal[2]) - 1.0) < 0.1
    
    def test_fit_tilted_plane(self):
        """Test fitting a tilted plane."""
        rng = np.random.default_rng(42)
        n_points = 500
        
        # Plane: x + y + z = 1
        # Normal: [1, 1, 1] / sqrt(3)
        u = rng.uniform(-5, 5, n_points)
        v = rng.uniform(-5, 5, n_points)
        
        # Parametric plane
        x = u
        y = v
        z = 1 - u - v + rng.normal(0, 0.01, n_points)
        
        points = np.column_stack([x, y, z])
        
        ransac = RANSACPlane(
            max_iterations=200,
            distance_threshold=0.05,
            min_inliers_ratio=0.8
        )
        
        result = ransac.fit(points)
        
        assert result is not None
        assert result.inlier_ratio > 0.8
    
    def test_with_outliers(self):
        """Test plane detection with outliers."""
        rng = np.random.default_rng(42)
        
        # Ground plane points
        n_inliers = 400
        x = rng.uniform(-5, 5, n_inliers)
        y = rng.uniform(-5, 5, n_inliers)
        z = rng.normal(0, 0.01, n_inliers)
        plane_points = np.column_stack([x, y, z])
        
        # Random outliers
        n_outliers = 100
        outliers = rng.uniform(-5, 5, (n_outliers, 3))
        outliers[:, 2] = rng.uniform(0.5, 2, n_outliers)  # Elevated points
        
        points = np.vstack([plane_points, outliers])
        
        ransac = RANSACPlane(
            max_iterations=200,
            distance_threshold=0.05,
            min_inliers_ratio=0.5
        )
        
        result = ransac.fit(points)
        
        assert result is not None
        # Should detect most of the plane inliers
        assert len(result.inlier_indices) > n_inliers * 0.8


class TestRANSACLine2D:
    """Tests for RANSACLine2D algorithm."""
    
    def test_fit_horizontal_line(self):
        """Test fitting horizontal line y = 0."""
        rng = np.random.default_rng(42)
        n_points = 100
        
        x = rng.uniform(-5, 5, n_points)
        y = rng.normal(0, 0.01, n_points)
        
        points = np.column_stack([x, y])
        
        ransac = RANSACLine2D(
            max_iterations=50,
            distance_threshold=0.05,
            min_inliers_ratio=0.8
        )
        
        result = ransac.fit(points)
        
        assert result is not None
        assert result.inlier_ratio > 0.9
    
    def test_fit_diagonal_line(self):
        """Test fitting diagonal line y = x."""
        rng = np.random.default_rng(42)
        n_points = 100
        
        t = rng.uniform(-5, 5, n_points)
        x = t + rng.normal(0, 0.01, n_points)
        y = t + rng.normal(0, 0.01, n_points)
        
        points = np.column_stack([x, y])
        
        ransac = RANSACLine2D(
            max_iterations=50,
            distance_threshold=0.05,
            min_inliers_ratio=0.8
        )
        
        result = ransac.fit(points)
        
        assert result is not None
        assert result.inlier_ratio > 0.8


class TestRANSACLine3D:
    """Tests for RANSACLine3D algorithm."""
    
    def test_fit_line_along_x_axis(self):
        """Test fitting line along X-axis."""
        rng = np.random.default_rng(42)
        n_points = 100
        
        t = rng.uniform(-5, 5, n_points)
        x = t
        y = rng.normal(0, 0.01, n_points)
        z = rng.normal(0, 0.01, n_points)
        
        points = np.column_stack([x, y, z])
        
        ransac = RANSACLine3D(
            max_iterations=50,
            distance_threshold=0.05,
            min_inliers_ratio=0.8
        )
        
        result = ransac.fit(points)
        
        assert result is not None
        assert result.inlier_ratio > 0.9
        
        # Direction should be close to [1, 0, 0] or [-1, 0, 0]
        direction = result.coefficients[3:]
        assert abs(abs(direction[0]) - 1.0) < 0.1


class TestMultiPlaneRANSAC:
    """Tests for MultiPlaneRANSAC algorithm."""
    
    def test_detect_two_planes(self):
        """Test detecting two perpendicular planes."""
        rng = np.random.default_rng(42)
        
        # Ground plane (z = 0)
        n1 = 300
        x1 = rng.uniform(-3, 3, n1)
        y1 = rng.uniform(-3, 3, n1)
        z1 = rng.normal(0, 0.01, n1)
        plane1 = np.column_stack([x1, y1, z1])
        
        # Wall plane (y = 2)
        n2 = 300
        x2 = rng.uniform(-3, 3, n2)
        y2 = rng.normal(2, 0.01, n2)
        z2 = rng.uniform(0, 2, n2)
        plane2 = np.column_stack([x2, y2, z2])
        
        points = np.vstack([plane1, plane2])
        
        multi_ransac = MultiPlaneRANSAC(
            max_planes=5,
            max_iterations=100,
            distance_threshold=0.05,
            min_inliers_ratio=0.2
        )
        
        results = multi_ransac.fit(points)
        
        assert len(results) >= 2
        
        # Total inliers from both planes should cover most points
        total_inliers = sum(len(r.inlier_indices) for r in results)
        assert total_inliers > len(points) * 0.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
