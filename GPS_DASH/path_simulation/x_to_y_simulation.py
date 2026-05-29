"""
================================================================================
                    X TO Y PATH SIMULATION
                    3D Campus Navigation Visualization
================================================================================

This script simulates movement from point X to point Y on a 3D campus map.
Uses matplotlib for 3D visualization (no HTML required).

Requirements:
    pip install matplotlib numpy

Controls:
    - Close the window to stop simulation
    - Progress bar shows completion percentage
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from datetime import datetime
import sys

# ==================== Configuration ====================
CONFIG = {
    # Simulation parameters
    "FPS": 30,
    "DURATION": 10,  # seconds
    "PATH_SMOOTHNESS": 100,  # Number of interpolation points
    
    # Campus layout (simplified from image)
    "BUILDING_SCALE": 1.0,
    
    # Colors
    "PATH_COLOR": '#00BFFF',  # Deep sky blue
    "MARKER_COLOR": '#FFFF00',  # Yellow
    "BUILDING_COLOR": '#D3D3D3',  # Light gray
    "GROUND_COLOR": '#F5F5F5',  # Very light gray
    "GRASS_COLOR": '#90EE90',  # Light green
    
    # Camera view
    "ELEVATION": 45,
    "AZIMUTH": 60,
}

# ==================== Campus Map Data ====================
class CampusMap:
    def __init__(self):
        # Define buildings as rectangular prisms (x, y, z, width, depth, height)
        self.buildings = [
            # Large square buildings (left side)
            {'x': -40, 'y': 0, 'w': 25, 'd': 20, 'h': 3},
            {'x': -10, 'y': 0, 'w': 25, 'd': 20, 'h': 3},
            # Right side buildings
            {'x': 25, 'y': 5, 'w': 20, 'd': 15, 'h': 2.5},
            {'x': 25, 'y': 30, 'w': 20, 'd': 15, 'h': 2.5},
            {'x': 50, 'y': 5, 'w': 15, 'd': 15, 'h': 2},
            {'x': 50, 'y': 30, 'w': 15, 'd': 15, 'h': 2},
            # Top buildings
            {'x': -30, 'y': 40, 'w': 30, 'd': 15, 'h': 2},
            {'x': 20, 'y': 50, 'w': 20, 'd': 10, 'h': 2},
            # Circular building (bottom left) - approximated as polygon
            {'x': -35, 'y': -25, 'w': 15, 'd': 15, 'h': 2, 'circular': True},
        ]
        
        # Grass/field areas
        self.grass_areas = [
            {'x': -15, 'y': 25, 'w': 30, 'd': 20},  # Main field
            {'x': 30, 'y': 55, 'w': 15, 'd': 10},   # Small field
        ]
        
        # Start (X) and End (Y/Z) points
        self.start_point = np.array([-35, 0, 0.5])  # X marker position
        self.end_point = np.array([15, 20, 0.5])    # Y/Z marker position
        
    def get_path(self, num_points=100):
        """Generate smooth path from X to Y with curve like in the image"""
        # Create waypoints for curved path
        waypoints = [
            self.start_point.copy(),
            np.array([-20, 0, 0.5]),      # Move right
            np.array([0, 0, 0.5]),        # Continue right
            np.array([15, 0, 0.5]),       # Continue right
            np.array([15, 10, 0.5]),      # Move up
            np.array([15, 20, 0.5]),      # Continue up to end
            self.end_point.copy(),
        ]
        
        # Interpolate between waypoints
        path = []
        for i in range(len(waypoints) - 1):
            segment_points = np.linspace(waypoints[i], waypoints[i+1], num_points // (len(waypoints)-1))
            path.extend(segment_points[:-1])
        path.append(waypoints[-1])
        
        return np.array(path)
    
    def create_building_mesh(self, building):
        """Create 3D mesh for a building"""
        x, y, w, d, h = building['x'], building['y'], building['w'], building['d'], building['h']
        
        # Create vertices for rectangular prism
        vertices = [
            [x - w/2, y - d/2, 0],
            [x + w/2, y - d/2, 0],
            [x + w/2, y + d/2, 0],
            [x - w/2, y + d/2, 0],
            [x - w/2, y - d/2, h],
            [x + w/2, y - d/2, h],
            [x + w/2, y + d/2, h],
            [x - w/2, y + d/2, h],
        ]
        
        return vertices


# ==================== 3D Visualizer ====================
class PathVisualizer3D:
    def __init__(self):
        self.campus = CampusMap()
        self.fig = None
        self.ax = None
        self.path_line = None
        self.marker = None
        self.trail = None
        self.progress_text = None
        self.time_text = None
        
    def setup_figure(self):
        """Create the 3D figure"""
        self.fig = plt.figure(figsize=(14, 10), facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set background colors
        self.ax.set_facecolor(CONFIG['GROUND_COLOR'])
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Configure axes
        self.ax.set_xlim(-60, 80)
        self.ax.set_ylim(-40, 70)
        self.ax.set_zlim(0, 15)
        
        # Remove axis lines for cleaner look
        self.ax.set_axis_off()
        
        # Set view angle
        self.ax.view_init(elev=CONFIG['ELEVATION'], azim=CONFIG['AZIMUTH'])
        
        # Add title
        self.fig.suptitle('️  Campus Navigation: X → Y Path Simulation', 
                         fontsize=16, color='white', fontweight='bold', y=0.98)
        
    def draw_ground(self):
        """Draw ground plane"""
        x = np.linspace(-60, 80, 50)
        y = np.linspace(-40, 70, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.3, color=CONFIG['GROUND_COLOR'])
        
    def draw_grass_areas(self):
        """Draw grass/field areas"""
        for grass in self.campus.grass_areas:
            x, y, w, d = grass['x'], grass['y'], grass['w'], grass['d']
            verts = [
                [x - w/2, y - d/2, 0.01],
                [x + w/2, y - d/2, 0.01],
                [x + w/2, y + d/2, 0.01],
                [x - w/2, y + d/2, 0.01],
            ]
            poly = mpatches.Polygon(
                [[v[0], v[1]] for v in verts],
                facecolor=CONFIG['GRASS_COLOR'],
                alpha=0.4,
                edgecolor='none'
            )
            # Note: Can't directly add 2D patch to 3D axes
            # Drawing as surface instead
            xx = np.linspace(x - w/2, x + w/2, 10)
            yy = np.linspace(y - d/2, y + d/2, 10)
            XX, YY = np.meshgrid(xx, yy)
            ZZ = np.ones_like(XX) * 0.01
            self.ax.plot_surface(XX, YY, ZZ, alpha=0.4, color=CONFIG['GRASS_COLOR'])
        
    def draw_buildings(self):
        """Draw all buildings as 3D blocks"""
        for building in self.campus.buildings:
            x, y, w, d, h = building['x'], building['y'], building['w'], building['d'], building['h']
            
            # Create box vertices
            xx = np.array([x - w/2, x + w/2, x + w/2, x - w/2, x - w/2])
            yy = np.array([y - d/2, y - d/2, y + d/2, y + d/2, y - d/2])
            
            # Bottom face
            self.ax.plot(xx, yy, np.zeros(5), 'k-', alpha=0.3, linewidth=0.5)
            
            # Top face
            self.ax.plot(xx, yy, np.ones(5) * h, '-', 
                        color=CONFIG['BUILDING_COLOR'], alpha=0.8, linewidth=1)
            
            # Vertical edges
            for i in range(4):
                self.ax.plot([xx[i], xx[i]], [yy[i], yy[i]], [0, h], 
                            'k-', alpha=0.3, linewidth=0.5)
            
            # Fill top surface
            if building.get('circular', False):
                # Circular building
                theta = np.linspace(0, 2*np.pi, 30)
                r = min(w, d) / 2
                xc = x
                yc = y
                circle_x = xc + r * np.cos(theta)
                circle_y = yc + r * np.sin(theta)
                circle_z = np.ones_like(theta) * h
                self.ax.plot(circle_x, circle_y, circle_z, 
                            color=CONFIG['BUILDING_COLOR'], alpha=0.9, linewidth=1.5)
                # Fill
                self.ax.plot_surface(
                    np.outer(np.cos(theta), np.ones(1) * r),
                    np.outer(np.sin(theta), np.ones(1) * r),
                    np.ones((30, 1)) * h,
                    alpha=0.7, color=CONFIG['BUILDING_COLOR']
                )
            else:
                # Rectangular building - fill top
                self.ax.plot_surface(
                    np.array([[x-w/2, x+w/2], [x+w/2, x-w/2]]),
                    np.array([[y-d/2, y-d/2], [y+d/2, y+d/2]]),
                    np.array([[h, h], [h, h]]),
                    alpha=0.7, color=CONFIG['BUILDING_COLOR']
                )
    
    def draw_markers(self):
        """Draw X and Y markers"""
        # X marker (start)
        self.ax.scatter(*self.campus.start_point, 
                       c='yellow', s=200, marker='X', 
                       label='Start (X)', edgecolors='orange', linewidths=2)
        
        # Y marker (end)
        self.ax.scatter(*self.campus.end_point, 
                       c='yellow', s=200, marker='*', 
                       label='Destination (Y)', edgecolors='orange', linewidths=2)
        
        # Add text labels
        self.ax.text(self.campus.start_point[0], self.campus.start_point[1], 
                    self.campus.start_point[2] + 2, 'X', 
                    fontsize=14, fontweight='bold', color='yellow',
                    ha='center', va='bottom')
        
        self.ax.text(self.campus.end_point[0], self.campus.end_point[1], 
                    self.campus.end_point[2] + 2, 'Y', 
                    fontsize=14, fontweight='bold', color='yellow',
                    ha='center', va='bottom')
    
    def draw_path(self, path):
        """Draw the complete path"""
        self.ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                    color=CONFIG['PATH_COLOR'], linewidth=4, alpha=0.7,
                    label='Navigation Path')
    
    def init_animation(self, path):
        """Initialize animation elements"""
        # Create path line (initially empty)
        self.path_line, = self.ax.plot([], [], [], 
                                       color=CONFIG['PATH_COLOR'], 
                                       linewidth=4, alpha=0.8)
        
        # Create moving marker
        self.marker = self.ax.scatter([], [], [], 
                                      c='red', s=150, marker='o',
                                      edgecolors='white', linewidths=2)
        
        # Create trail
        self.trail, = self.ax.plot([], [], [], 
                                   color='red', linewidth=2, alpha=0.5)
        
        # Progress text
        self.progress_text = self.fig.text(
            0.02, 0.95, '', 
            transform=self.fig.transFigure,
            fontsize=11, color='white',
            bbox=dict(boxstyle='round', facecolor='#333366', alpha=0.8)
        )
        
        # Time text
        self.time_text = self.fig.text(
            0.98, 0.95, '', 
            transform=self.fig.transFigure,
            fontsize=11, color='cyan',
            ha='right',
            bbox=dict(boxstyle='round', facecolor='#333366', alpha=0.8)
        )
        
        # Legend
        self.ax.legend(loc='upper left', fontsize=10, facecolor='#333366', 
                      edgecolor='white', labelcolor='white')
        
        self.path = path
        self.trail_data_x = []
        self.trail_data_y = []
        self.trail_data_z = []
        
    def update_animation(self, frame):
        """Update animation frame"""
        total_frames = len(self.path)
        progress = frame / total_frames
        
        # Update path line (show completed portion)
        path_idx = int(frame * len(self.path) / total_frames)
        current_path = self.path[:path_idx+1]
        self.path_line.set_data(current_path[:, 0], current_path[:, 1])
        self.path_line.set_3d_properties(current_path[:, 2])
        
        # Update marker position
        if path_idx < len(self.path):
            current_pos = self.path[path_idx]
            self.marker._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
            
            # Update trail
            self.trail_data_x.append(current_pos[0])
            self.trail_data_y.append(current_pos[1])
            self.trail_data_z.append(current_pos[2])
            self.trail.set_data(self.trail_data_x, self.trail_data_y)
            self.trail.set_3d_properties(self.trail_data_z)
        
        # Update progress text
        self.progress_text.set_text(f'Progress: {progress*100:.1f}%  |  Distance: {path_idx}/{len(self.path)} points')
        
        # Update time
        elapsed = frame / CONFIG['FPS']
        self.time_text.set_text(f'Time: {elapsed:.1f}s')
        
        return self.path_line, self.marker, self.trail, self.progress_text, self.time_text
    
    def run_simulation(self):
        """Run the complete simulation"""
        print("\n" + "="*70)
        print("        X TO Y PATH SIMULATION")
        print("="*70)
        print("\n📍 Starting position: X marker")
        print("🎯 Destination: Y marker")
        print(f"⏱️  Simulation duration: {CONFIG['DURATION']} seconds")
        print(f"📊 Path points: {CONFIG['PATH_SMOOTHNESS']}")
        print("\n🚀 Launching visualization...")
        print("   Close the window to stop\n")
        
        # Generate path
        path = self.campus.get_path(CONFIG['PATH_SMOOTHNESS'])
        
        # Setup visualization
        self.setup_figure()
        self.draw_ground()
        self.draw_grass_areas()
        self.draw_buildings()
        self.draw_markers()
        self.draw_path(path)
        self.init_animation(path)
        
        # Create animation
        total_frames = int(CONFIG['DURATION'] * CONFIG['FPS'])
        anim = animation.FuncAnimation(
            self.fig, 
            self.update_animation,
            frames=total_frames,
            interval=1000/CONFIG['FPS'],
            blit=False
        )
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        print("\n✅ Simulation completed!")
        print(f"   Total frames: {total_frames}")
        print(f"   Path length: {len(path)} points")


# ==================== Alternative: Simple Matplotlib Version ====================
class SimplePathVisualizer:
    """Simpler version without animation for basic visualization"""
    
    def __init__(self):
        self.campus = CampusMap()
        
    def visualize(self):
        """Create static 3D visualization"""
        fig = plt.figure(figsize=(12, 10), facecolor='#1a1a2e')
        ax = fig.add_subplot(111, projection='3d')
        
        # Get path
        path = self.campus.get_path(100)
        
        # Draw ground
        ax.set_facecolor(CONFIG['GROUND_COLOR'])
        fig.patch.set_facecolor('#1a1a2e')
        
        # Draw buildings (simplified)
        for building in self.campus.buildings:
            x, y, w, d, h = building['x'], building['y'], building['w'], building['d'], building['h']
            xx = np.array([x-w/2, x+w/2, x+w/2, x-w/2, x-w/2])
            yy = np.array([y-d/2, y-d/2, y+d/2, y+d/2, y-d/2])
            ax.plot(xx, yy, np.ones(5)*h, color=CONFIG['BUILDING_COLOR'], linewidth=1.5)
            for i in range(4):
                ax.plot([xx[i], xx[i]], [yy[i], yy[i]], [0, h], 'k-', alpha=0.3)
        
        # Draw path
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 
               color=CONFIG['PATH_COLOR'], linewidth=4, alpha=0.8, label='Path X→Y')
        
        # Draw markers
        ax.scatter(*self.campus.start_point, c='yellow', s=200, marker='X', 
                  label='Start (X)', edgecolors='orange')
        ax.scatter(*self.campus.end_point, c='yellow', s=200, marker='*', 
                  label='End (Y)', edgecolors='orange')
        
        # Labels
        ax.text(*self.campus.start_point, '  X', fontsize=14, color='yellow', fontweight='bold')
        ax.text(*self.campus.end_point, '  Y', fontsize=14, color='yellow', fontweight='bold')
        
        # Configure
        ax.set_xlim(-60, 80)
        ax.set_ylim(-40, 70)
        ax.set_zlim(0, 10)
        ax.set_axis_off()
        ax.view_init(elev=45, azim=60)
        
        fig.suptitle('Campus Navigation: X to Y Path', fontsize=16, color='white')
        ax.legend(facecolor='#333366', labelcolor='white')
        
        plt.tight_layout()
        plt.show()


# ==================== Main ====================
def main():
    print("\n" + "="*70)
    print("        X TO Y PATH SIMULATION")
    print("="*70)
    print("\n📍 Starting position: X marker")
    print("🎯 Destination: Y marker")
    print(f"⏱️  Simulation duration: {CONFIG['DURATION']} seconds")
    print(f"📊 Path points: {CONFIG['PATH_SMOOTHNESS']}")
    print("\n🚀 Launching 3D visualization...")
    print("   Close the window to stop\n")
    
    # Run animated simulation by default
    visualizer = PathVisualizer3D()
    visualizer.run_simulation()


if __name__ == "__main__":
    main()
