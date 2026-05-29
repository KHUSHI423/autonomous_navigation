"""
3D Scene Reconstruction and Point Cloud Generation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import time
from pathlib import Path

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. Some 3D features will be limited.")

from camera_config import CameraIntrinsics


class Scene3DMapper:
    """
    3D Scene Reconstruction from RGB image and depth map
    """
    
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics = None
    ):
        """
        Initialize 3D mapper
        
        Args:
            camera_intrinsics: Camera parameters (auto-detected if None)
        """
        self.camera = camera_intrinsics
        self.point_cloud = None
        self.colors = None
        self.depth_map = None
        self.rgb_image = None
    
    def set_camera_from_image(self, image: np.ndarray, fov: float = 70.0):
        """Set camera intrinsics based on image size"""
        height, width = image.shape[:2]
        self.camera = CameraIntrinsics.from_image_size(width, height, fov)
        return self.camera
    
    def depth_to_pointcloud(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        downsample_factor: int = 1,
        max_depth: float = 100.0,
        min_depth: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to 3D point cloud
        
        Args:
            rgb_image: RGB image for colors
            depth_map: Depth map in meters
            downsample_factor: Reduce points by this factor
            max_depth: Maximum depth to include
            min_depth: Minimum depth to include
            
        Returns:
            Tuple of (points, colors) arrays
        """
        self.rgb_image = rgb_image
        self.depth_map = depth_map
        
        # Auto-set camera if not set
        if self.camera is None:
            self.set_camera_from_image(rgb_image)
        
        height, width = depth_map.shape[:2]
        
        # Create pixel coordinate grids
        u = np.arange(0, width, downsample_factor)
        v = np.arange(0, height, downsample_factor)
        u, v = np.meshgrid(u, v)
        
        # Get depth values
        z = depth_map[::downsample_factor, ::downsample_factor]
        
        # Create depth mask
        valid_mask = (z > min_depth) & (z < max_depth) & np.isfinite(z)
        
        # Convert to 3D coordinates
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        x = (u - self.camera.cx) * z / self.camera.fx
        y = (v - self.camera.cy) * z / self.camera.fy
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=-1)
        
        # Get colors
        if len(rgb_image.shape) == 3:
            # Convert BGR to RGB if needed
            if rgb_image.shape[2] == 3:
                colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            else:
                colors = rgb_image
            colors = colors[::downsample_factor, ::downsample_factor].astype(np.float32) / 255.0
        else:
            # Grayscale
            gray = rgb_image[::downsample_factor, ::downsample_factor].astype(np.float32) / 255.0
            colors = np.stack([gray, gray, gray], axis=-1)
        
        # Apply mask and flatten
        points = points[valid_mask].reshape(-1, 3)
        colors = colors[valid_mask].reshape(-1, 3)
        
        self.point_cloud = points
        self.colors = colors
        
        print(f"  Generated point cloud with {len(points):,} points")
        
        return points, colors
    
    def create_open3d_pointcloud(self) -> 'o3d.geometry.PointCloud':
        """Convert to Open3D point cloud format"""
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for this function")
        
        if self.point_cloud is None:
            raise ValueError("No point cloud generated. Call depth_to_pointcloud first.")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        
        return pcd
    
    def filter_pointcloud(
        self,
        voxel_size: float = 0.05,
        remove_outliers: bool = True,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> 'o3d.geometry.PointCloud':
        """
        Filter and clean the point cloud
        
        Args:
            voxel_size: Voxel size for downsampling
            remove_outliers: Whether to remove statistical outliers
            nb_neighbors: Number of neighbors for outlier detection
            std_ratio: Standard deviation ratio for outlier detection
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for this function")
        
        pcd = self.create_open3d_pointcloud()
        
        # Voxel downsampling
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            print(f"  After voxel downsampling: {len(pcd.points):,} points")
        
        # Remove outliers
        if remove_outliers:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            print(f"  After outlier removal: {len(pcd.points):,} points")
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2 if voxel_size > 0 else 0.1,
                max_nn=30
            )
        )
        
        return pcd
    
    def save_pointcloud(
        self,
        filepath: str,
        format: str = 'ply'
    ):
        """
        Save point cloud to file
        
        Args:
            filepath: Output file path
            format: 'ply', 'pcd', 'xyz', or 'xyzrgb'
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if HAS_OPEN3D and format in ['ply', 'pcd']:
            pcd = self.create_open3d_pointcloud()
            o3d.io.write_point_cloud(str(filepath), pcd)
        else:
            # Save as text format
            if self.point_cloud is None:
                raise ValueError("No point cloud to save")
            
            with open(filepath, 'w') as f:
                if format == 'xyz':
                    for point in self.point_cloud:
                        f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                else:  # xyzrgb
                    for point, color in zip(self.point_cloud, self.colors):
                        r, g, b = (color * 255).astype(int)
                        f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")
        
        print(f"  Saved point cloud to: {filepath}")
    
    def create_mesh(
        self,
        method: str = 'poisson',
        depth: int = 9
    ) -> 'o3d.geometry.TriangleMesh':
        """
        Create mesh from point cloud using surface reconstruction
        
        Args:
            method: 'poisson' or 'ball_pivoting'
            depth: Octree depth for Poisson reconstruction
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for this function")
        
        pcd = self.filter_pointcloud()
        
        if method == 'poisson':
            print("  Running Poisson surface reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )
            
            # Remove low density vertices
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.1)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        elif method == 'ball_pivoting':
            print("  Running Ball Pivoting surface reconstruction...")
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2, avg_dist * 4]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        
        print(f"  Created mesh with {len(mesh.triangles):,} triangles")
        
        return mesh
    
    def get_ground_plane(self) -> Tuple[np.ndarray, float]:
        """
        Detect ground plane using RANSAC
        
        Returns:
            Tuple of (plane_normal, plane_distance)
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required")
        
        pcd = self.create_open3d_pointcloud()
        
        # Segment plane using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=1000
        )
        
        # plane_model: [a, b, c, d] where ax + by + cz + d = 0
        normal = np.array(plane_model[:3])
        distance = plane_model[3]
        
        print(f"  Ground plane: normal={normal}, distance={distance:.2f}m")
        
        return normal, distance


class MultiViewMapper(Scene3DMapper):
    """
    Multi-view 3D reconstruction (for video or multiple images)
    """
    
    def __init__(self, camera_intrinsics: CameraIntrinsics = None):
        super().__init__(camera_intrinsics)
        self.all_points = []
        self.all_colors = []
        self.frame_count = 0
    
    def add_frame(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        pose: np.ndarray = None
    ):
        """
        Add a frame to the reconstruction
        
        Args:
            rgb_image: RGB image
            depth_map: Depth map
            pose: 4x4 camera pose matrix (optional)
        """
        points, colors = self.depth_to_pointcloud(rgb_image, depth_map)
        
        # Transform points by camera pose if provided
        if pose is not None:
            # Add homogeneous coordinate
            points_h = np.hstack([points, np.ones((len(points), 1))])
            # Transform
            points_transformed = (pose @ points_h.T).T[:, :3]
            points = points_transformed
        
        self.all_points.append(points)
        self.all_colors.append(colors)
        self.frame_count += 1
        
        print(f"  Added frame {self.frame_count}")
    
    def merge_pointclouds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Merge all point clouds into one"""
        if not self.all_points:
            raise ValueError("No frames added")
        
        self.point_cloud = np.vstack(self.all_points)
        self.colors = np.vstack(self.all_colors)
        
        print(f"  Merged {self.frame_count} frames: {len(self.point_cloud):,} total points")
        
        return self.point_cloud, self.colors
