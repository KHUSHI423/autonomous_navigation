"""
3D Visualization for Scene Reconstruction
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from object_3d_detector import Object3D


class Visualizer3D:
    """
    3D Visualization tools for point clouds and detected objects
    """
    
    OBJECT_COLORS = {
        'person': 'yellow',
        'bicycle': 'orange',
        'car': 'green',
        'motorcycle': 'magenta',
        'bus': 'blue',
        'train': 'purple',
        'truck': 'red',
        'traffic light': 'red',
        'fire hydrant': 'cyan',
        'stop sign': 'darkred',
    }
    
    def __init__(self):
        """Initialize visualizer"""
        if not HAS_PLOTLY:
            print("Warning: Plotly not installed. Interactive visualization unavailable.")
    
    def create_interactive_scene(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        objects: List[Object3D] = None,
        title: str = "3D Scene Reconstruction",
        point_size: int = 2,
        downsample: int = 1
    ) -> 'go.Figure':
        """
        Create interactive 3D visualization using Plotly
        
        Args:
            points: Point cloud (N, 3)
            colors: Colors for each point (N, 3) in 0-1 range
            objects: List of detected 3D objects
            title: Plot title
            point_size: Size of points
            downsample: Downsample factor for large point clouds
            
        Returns:
            Plotly Figure object
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive visualization")
        
        # Downsample if needed
        if downsample > 1:
            indices = np.arange(0, len(points), downsample)
            points = points[indices]
            colors = colors[indices]
        
        # Convert colors to RGB strings
        color_strings = [
            f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
            for c in colors
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add point cloud
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=color_strings,
                opacity=0.8
            ),
            name='Point Cloud',
            hoverinfo='text',
            text=[f'({x:.2f}, {y:.2f}, {z:.2f})' for x, y, z in points]
        ))
        
        # Add detected objects as 3D boxes
        if objects:
            for obj in objects:
                if obj.bbox_3d is not None:
                    self._add_3d_box(fig, obj)
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            scene=dict(
                xaxis_title='X (meters)',
                yaxis_title='Y (meters)',
                zaxis_title='Z (depth, meters)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=0, y=-2, z=1)
                )
            ),
            showlegend=True,
            width=1200,
            height=800,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def _add_3d_box(self, fig: 'go.Figure', obj: Object3D):
        """Add a 3D bounding box for an object"""
        corners = obj.bbox_3d
        color = self.OBJECT_COLORS.get(obj.class_name, 'gray')
        
        # Define edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Front
            [4, 5], [5, 6], [6, 7], [7, 4],  # Back
            [0, 4], [1, 5], [2, 6], [3, 7],  # Sides
        ]
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[corners[edge[0], 0], corners[edge[1], 0]],
                y=[corners[edge[0], 1], corners[edge[1], 1]],
                z=[corners[edge[0], 2], corners[edge[1], 2]],
                mode='lines',
                line=dict(color=color, width=4),
                name=f'{obj.class_name} ({obj.distance:.1f}m)',
                showlegend=False,
                hoverinfo='text',
                text=f'{obj.class_name}: {obj.distance:.1f}m'
            ))
        
        # Add center marker
        center = obj.position_3d
        fig.add_trace(go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode='markers+text',
            marker=dict(size=8, color=color, symbol='diamond'),
            text=[f'{obj.class_name}<br>{obj.distance:.1f}m'],
            textposition='top center',
            name=f'{obj.class_name}',
            showlegend=True
        ))
    
    def create_depth_visualization(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        objects: List[Object3D] = None
    ) -> np.ndarray:
        """
        Create a combined RGB + Depth + 3D info visualization
        """
        h, w = rgb_image.shape[:2]
        
        # Create depth colormap
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        # Create side-by-side visualization
        combined = np.hstack([rgb_image, depth_colored])
        
        # Add depth scale bar
        scale_width = 30
        scale_height = h
        scale_bar = np.zeros((scale_height, scale_width, 3), dtype=np.uint8)
        for i in range(scale_height):
            val = int(255 * (scale_height - i) / scale_height)
            scale_bar[i, :] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
        
        combined = np.hstack([combined, scale_bar])
        
        # Add text annotations
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Add "RGB" label
        cv2.putText(combined, "RGB Image", (10, 30), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add "Depth" label
        cv2.putText(combined, "Depth Map", (w + 10, 30), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add depth range
        cv2.putText(combined, f"{depth_map.max():.1f}m", (w * 2 + 5, 25), font, 0.4, (255, 255, 255), 1)
        cv2.putText(combined, f"{depth_map.min():.1f}m", (w * 2 + 5, h - 10), font, 0.4, (255, 255, 255), 1)
        
        return combined
    
    def create_topdown_view(
        self,
        objects: List[Object3D],
        range_x: Tuple[float, float] = (-20, 20),
        range_z: Tuple[float, float] = (0, 50),
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        """
        Create a top-down bird's eye view of detected objects
        
        Args:
            objects: List of detected 3D objects
            range_x: X-axis range in meters (left, right)
            range_z: Z-axis range in meters (near, far)
            image_size: Output image size (width, height)
        """
        width, height = image_size
        
        # Create black background
        topdown = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw grid
        self._draw_grid(topdown, range_x, range_z)
        
        # Draw ego vehicle (camera position)
        ego_x = width // 2
        ego_y = height - 50
        cv2.circle(topdown, (ego_x, ego_y), 15, (0, 200, 0), -1)
        cv2.putText(topdown, "EGO", (ego_x - 15, ego_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw field of view
        fov_angle = 35  # degrees
        fov_length = height - 100
        left_x = ego_x - int(fov_length * np.tan(np.radians(fov_angle)))
        right_x = ego_x + int(fov_length * np.tan(np.radians(fov_angle)))
        cv2.line(topdown, (ego_x, ego_y), (left_x, 50), (50, 50, 50), 1)
        cv2.line(topdown, (ego_x, ego_y), (right_x, 50), (50, 50, 50), 1)
        
        # Draw objects
        for obj in objects:
            if obj.position_3d is None:
                continue
            
            # Convert 3D position to image coordinates
            x_3d = obj.position_3d[0]
            z_3d = obj.position_3d[2]
            
            # Scale to image coordinates
            x_img = int(((x_3d - range_x[0]) / (range_x[1] - range_x[0])) * width)
            y_img = int((1 - (z_3d - range_z[0]) / (range_z[1] - range_z[0])) * height)
            
            # Clip to image bounds
            x_img = np.clip(x_img, 10, width - 10)
            y_img = np.clip(y_img, 10, height - 10)
            
            # Get color
            color_name = self.OBJECT_COLORS.get(obj.class_name, 'white')
            color = self._color_name_to_bgr(color_name)
            
            # Draw object as rectangle (approximate size)
            dims = obj.estimated_dimensions or obj.dimensions
            if dims:
                obj_width = int(dims['width'] / (range_x[1] - range_x[0]) * width)
                obj_length = int(dims['length'] / (range_z[1] - range_z[0]) * height)
                obj_width = max(10, min(obj_width, 50))
                obj_length = max(10, min(obj_length, 80))
                
                cv2.rectangle(
                    topdown,
                    (x_img - obj_width // 2, y_img - obj_length // 2),
                    (x_img + obj_width // 2, y_img + obj_length // 2),
                    color, -1
                )
                cv2.rectangle(
                    topdown,
                    (x_img - obj_width // 2, y_img - obj_length // 2),
                    (x_img + obj_width // 2, y_img + obj_length // 2),
                    (255, 255, 255), 1
                )
            else:
                cv2.circle(topdown, (x_img, y_img), 8, color, -1)
            
            # Add label
            label = f"{obj.class_name[:3]} {obj.distance:.0f}m"
            cv2.putText(topdown, label, (x_img - 20, y_img - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add title
        cv2.putText(topdown, "Bird's Eye View", (10, 25),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        return topdown
    
    def _draw_grid(
        self,
        image: np.ndarray,
        range_x: Tuple[float, float],
        range_z: Tuple[float, float],
        grid_spacing: float = 10.0
    ):
        """Draw distance grid on top-down view"""
        height, width = image.shape[:2]
        
        # Draw horizontal lines (distance markers)
        for z in np.arange(range_z[0], range_z[1], grid_spacing):
            y = int((1 - (z - range_z[0]) / (range_z[1] - range_z[0])) * height)
            cv2.line(image, (0, y), (width, y), (30, 30, 30), 1)
            cv2.putText(image, f"{z:.0f}m", (5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Draw vertical center line
        cv2.line(image, (width // 2, 0), (width // 2, height), (30, 30, 30), 1)
    
    def _color_name_to_bgr(self, color_name: str) -> Tuple[int, int, int]:
        """Convert color name to BGR tuple"""
        colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'darkred': (0, 0, 139),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
        }
        return colors.get(color_name, (255, 255, 255))
    
    def save_interactive_html(
        self,
        fig: 'go.Figure',
        filepath: str
    ):
        """Save interactive Plotly figure as HTML"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(filepath))
        print(f"  Saved interactive 3D view to: {filepath}")
    
    def save_visualization(
        self,
        image: np.ndarray,
        filepath: str
    ):
        """Save visualization image"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved visualization to: {filepath}")


def visualize_open3d(
    points: np.ndarray,
    colors: np.ndarray,
    objects: List[Object3D] = None,
    show_coordinate_frame: bool = True
):
    """
    Visualize using Open3D (blocking window)
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]
    
    # Add coordinate frame
    if show_coordinate_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        geometries.append(coord_frame)
    
    # Add 3D boxes for objects
    if objects:
        for obj in objects:
            if obj.bbox_3d is not None:
                lines = [
                    [0, 1], [1, 2], [2, 3], [3, 0],
                    [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]
                ]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(obj.bbox_3d)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.paint_uniform_color([0, 1, 0])  # Green
                geometries.append(line_set)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Scene",
        width=1200,
        height=800
    )
