"""
Core RANSAC Algorithm Implementations.

This module provides RANSAC implementations for detecting geometric primitives:
- RANSACPlane: Fits planes to 3D point clouds
- RANSACLine2D: Fits lines to 2D point sets (for LaserScan)
- RANSACLine3D: Fits lines to 3D point sets
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from abc import ABC, abstractmethod


@dataclass
class RANSACResult:
    """Result of RANSAC fitting."""
    coefficients: np.ndarray  # Model coefficients
    inlier_indices: np.ndarray  # Indices of inlier points
    outlier_indices: np.ndarray  # Indices of outlier points
    inlier_ratio: float  # Ratio of inliers to total points
    num_iterations: int  # Number of iterations performed


class RANSACBase(ABC):
    """Base class for RANSAC algorithms."""
    
    def __init__(
        self,
        max_iterations: int = 1000,
        distance_threshold: float = 0.01,
        min_inliers_ratio: float = 0.3,
        random_seed: Optional[int] = None
    ):
        """
        Initialize RANSAC algorithm.
        
        Args:
            max_iterations: Maximum number of RANSAC iterations
            distance_threshold: Maximum distance for a point to be considered inlier
            min_inliers_ratio: Minimum ratio of inliers for valid model
            random_seed: Optional seed for reproducibility
        """
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.min_inliers_ratio = min_inliers_ratio
        self.rng = np.random.default_rng(random_seed)
    
    @abstractmethod
    def _min_samples(self) -> int:
        """Return minimum number of samples needed to fit model."""
        pass
    
    @abstractmethod
    def _fit_model(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Fit model to sample points. Returns None if fitting fails."""
        pass
    
    @abstractmethod
    def _compute_distances(self, points: np.ndarray, model: np.ndarray) -> np.ndarray:
        """Compute distances from points to model."""
        pass
    
    def fit(self, points: np.ndarray) -> Optional[RANSACResult]:
        """
        Fit model to points using RANSAC.
        
        Args:
            points: Array of shape (N, D) where N is number of points, D is dimension
            
        Returns:
            RANSACResult if successful, None otherwise
        """
        if len(points) < self._min_samples():
            return None
        
        n_points = len(points)
        best_model = None
        best_inlier_count = 0
        best_inliers = None
        
        for iteration in range(self.max_iterations):
            # Randomly sample minimum required points
            sample_indices = self.rng.choice(n_points, self._min_samples(), replace=False)
            sample_points = points[sample_indices]
            
            # Fit model to samples
            model = self._fit_model(sample_points)
            if model is None:
                continue
            
            # Compute distances and find inliers
            distances = self._compute_distances(points, model)
            inliers = distances < self.distance_threshold
            inlier_count = np.sum(inliers)
            
            # Update best model if this one is better
            if inlier_count > best_inlier_count:
                best_model = model
                best_inlier_count = inlier_count
                best_inliers = inliers
                
                # Early termination if we have enough inliers
                if inlier_count / n_points > 0.9:
                    break
        
        # Check if we found a valid model
        if best_model is None or best_inlier_count / n_points < self.min_inliers_ratio:
            return None
        
        inlier_indices = np.where(best_inliers)[0]
        outlier_indices = np.where(~best_inliers)[0]
        
        # Optionally refine model using all inliers
        refined_model = self._fit_model(points[inlier_indices])
        if refined_model is not None:
            best_model = refined_model
        
        return RANSACResult(
            coefficients=best_model,
            inlier_indices=inlier_indices,
            outlier_indices=outlier_indices,
            inlier_ratio=best_inlier_count / n_points,
            num_iterations=iteration + 1
        )


class RANSACPlane(RANSACBase):
    """
    RANSAC algorithm for fitting planes to 3D point clouds.
    
    Plane equation: ax + by + cz + d = 0
    Coefficients are normalized such that (a, b, c) is a unit vector.
    """
    
    def _min_samples(self) -> int:
        return 3  # 3 points define a plane
    
    def _fit_model(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Fit plane to 3 or more points.
        
        Args:
            points: Array of shape (N, 3)
            
        Returns:
            Plane coefficients [a, b, c, d] or None if degenerate
        """
        if len(points) < 3:
            return None
        
        if len(points) == 3:
            # Compute plane from 3 points using cross product
            p1, p2, p3 = points[0], points[1], points[2]
            v1 = p2 - p1
            v2 = p3 - p1
            
            # Normal vector is cross product
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            
            # Check for collinear points
            if norm < 1e-10:
                return None
            
            normal = normal / norm
            d = -np.dot(normal, p1)
            
            return np.array([normal[0], normal[1], normal[2], d])
        else:
            # Use SVD for least squares fit to multiple points
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            
            _, _, vh = np.linalg.svd(centered)
            normal = vh[-1]  # Last row of V^T is normal to plane
            norm = np.linalg.norm(normal)
            
            if norm < 1e-10:
                return None
            
            normal = normal / norm
            d = -np.dot(normal, centroid)
            
            return np.array([normal[0], normal[1], normal[2], d])
    
    def _compute_distances(self, points: np.ndarray, model: np.ndarray) -> np.ndarray:
        """
        Compute distances from points to plane.
        
        Args:
            points: Array of shape (N, 3)
            model: Plane coefficients [a, b, c, d]
            
        Returns:
            Array of distances
        """
        a, b, c, d = model
        # Distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
        # Since we normalize (a, b, c) to unit vector, denominator is 1
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        return distances
    
    def get_plane_normal(self, coefficients: np.ndarray) -> np.ndarray:
        """Get normal vector of plane."""
        return coefficients[:3]
    
    def get_plane_center(self, points: np.ndarray, inlier_indices: np.ndarray) -> np.ndarray:
        """Get center of inlier points."""
        return np.mean(points[inlier_indices], axis=0)


class RANSACLine2D(RANSACBase):
    """
    RANSAC algorithm for fitting lines to 2D point sets.
    
    Line equation: ax + by + c = 0
    Coefficients are normalized such that (a, b) is a unit vector.
    """
    
    def _min_samples(self) -> int:
        return 2  # 2 points define a line
    
    def _fit_model(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Fit line to 2 or more points.
        
        Args:
            points: Array of shape (N, 2)
            
        Returns:
            Line coefficients [a, b, c] or None if degenerate
        """
        if len(points) < 2:
            return None
        
        if len(points) == 2:
            # Compute line from 2 points
            p1, p2 = points[0], points[1]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            
            if norm < 1e-10:
                return None
            
            # Normal to the line direction
            normal = np.array([-direction[1], direction[0]]) / norm
            c = -np.dot(normal, p1)
            
            return np.array([normal[0], normal[1], c])
        else:
            # Use SVD for least squares fit
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            
            _, _, vh = np.linalg.svd(centered)
            direction = vh[0]  # First row is principal direction
            normal = np.array([-direction[1], direction[0]])
            norm = np.linalg.norm(normal)
            
            if norm < 1e-10:
                return None
            
            normal = normal / norm
            c = -np.dot(normal, centroid)
            
            return np.array([normal[0], normal[1], c])
    
    def _compute_distances(self, points: np.ndarray, model: np.ndarray) -> np.ndarray:
        """
        Compute distances from points to line.
        
        Args:
            points: Array of shape (N, 2)
            model: Line coefficients [a, b, c]
            
        Returns:
            Array of distances
        """
        a, b, c = model
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        return distances


class RANSACLine3D(RANSACBase):
    """
    RANSAC algorithm for fitting lines to 3D point sets.
    
    Line is represented as a point on the line and a direction vector.
    Coefficients: [px, py, pz, dx, dy, dz] where (px, py, pz) is a point
    and (dx, dy, dz) is the normalized direction vector.
    """
    
    def _min_samples(self) -> int:
        return 2  # 2 points define a line
    
    def _fit_model(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Fit line to 2 or more points.
        
        Args:
            points: Array of shape (N, 3)
            
        Returns:
            Line coefficients [px, py, pz, dx, dy, dz] or None if degenerate
        """
        if len(points) < 2:
            return None
        
        if len(points) == 2:
            # Compute line from 2 points
            p1, p2 = points[0], points[1]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            
            if norm < 1e-10:
                return None
            
            direction = direction / norm
            return np.concatenate([p1, direction])
        else:
            # Use SVD for least squares fit
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            
            _, _, vh = np.linalg.svd(centered)
            direction = vh[0]  # First row is principal direction
            norm = np.linalg.norm(direction)
            
            if norm < 1e-10:
                return None
            
            direction = direction / norm
            return np.concatenate([centroid, direction])
    
    def _compute_distances(self, points: np.ndarray, model: np.ndarray) -> np.ndarray:
        """
        Compute distances from points to line.
        
        Args:
            points: Array of shape (N, 3)
            model: Line coefficients [px, py, pz, dx, dy, dz]
            
        Returns:
            Array of distances
        """
        point_on_line = model[:3]
        direction = model[3:]
        
        # Vector from point on line to each point
        v = points - point_on_line
        
        # Project v onto direction
        proj_length = np.dot(v, direction)
        proj = np.outer(proj_length, direction)
        
        # Distance is length of perpendicular component
        perpendicular = v - proj
        distances = np.linalg.norm(perpendicular, axis=1)
        
        return distances


class MultiPlaneRANSAC:
    """
    Sequential RANSAC for detecting multiple planes in a point cloud.
    """
    
    def __init__(
        self,
        max_planes: int = 5,
        max_iterations: int = 1000,
        distance_threshold: float = 0.01,
        min_inliers_ratio: float = 0.1,
        min_points_per_plane: int = 100
    ):
        """
        Initialize multi-plane RANSAC.
        
        Args:
            max_planes: Maximum number of planes to detect
            max_iterations: Max iterations per plane detection
            distance_threshold: Inlier distance threshold
            min_inliers_ratio: Min inlier ratio for valid plane
            min_points_per_plane: Minimum points required per plane
        """
        self.max_planes = max_planes
        self.min_points_per_plane = min_points_per_plane
        self.ransac = RANSACPlane(
            max_iterations=max_iterations,
            distance_threshold=distance_threshold,
            min_inliers_ratio=min_inliers_ratio
        )
    
    def fit(self, points: np.ndarray) -> list:
        """
        Detect multiple planes in point cloud.
        
        Args:
            points: Array of shape (N, 3)
            
        Returns:
            List of RANSACResult objects, one per detected plane
        """
        results = []
        remaining_points = points.copy()
        remaining_indices = np.arange(len(points))
        
        for _ in range(self.max_planes):
            if len(remaining_points) < self.min_points_per_plane:
                break
            
            result = self.ransac.fit(remaining_points)
            if result is None:
                break
            
            # Map indices back to original point cloud
            original_inliers = remaining_indices[result.inlier_indices]
            original_outliers = remaining_indices[result.outlier_indices]
            
            results.append(RANSACResult(
                coefficients=result.coefficients,
                inlier_indices=original_inliers,
                outlier_indices=original_outliers,
                inlier_ratio=result.inlier_ratio,
                num_iterations=result.num_iterations
            ))
            
            # Remove inliers for next iteration
            remaining_points = remaining_points[result.outlier_indices]
            remaining_indices = remaining_indices[result.outlier_indices]
        
        return results
