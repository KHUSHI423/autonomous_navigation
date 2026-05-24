"""
3D Icon Generator for Indian Navigation & Mapping Interface
Generates GLB format 3D assets for real-time navigation systems
"""

import numpy as np
import trimesh
import trimesh.creation as creation
from trimesh.transformations import rotation_matrix, translation_matrix, scale_matrix
from pathlib import Path
import os


class ColorPalette:
    """Consistent color palette for all 3D icons"""

    PRIMARY_BLUE = [0.290, 0.565, 0.851, 1.0]  # #4A90D9
    WARM_ORANGE = [0.910, 0.659, 0.486, 1.0]  # #E8A87C
    FOREST_GREEN = [0.239, 0.545, 0.251, 1.0]  # #3D8B40
    ROAD_GRAY = [0.420, 0.447, 0.498, 1.0]  # #6B7280
    VEHICLE_YELLOW = [0.957, 0.769, 0.188, 1.0]  # #F4C430
    VEHICLE_RED = [0.863, 0.271, 0.271, 1.0]  # #DC4545
    SKY_BLUE = [0.529, 0.808, 0.922, 1.0]  # #87CEEB
    LIGHT_GRAY = [0.898, 0.906, 0.922, 1.0]  # #E5E7EB
    DARK_CHARCOAL = [0.216, 0.255, 0.318, 1.0]  # #374151
    WHITE = [1.0, 1.0, 1.0, 1.0]
    BLACK = [0.1, 0.1, 0.1, 1.0]
    SILVER = [0.753, 0.753, 0.753, 1.0]
    BROWN = [0.545, 0.353, 0.169, 1.0]
    BEIGE = [0.961, 0.871, 0.702, 1.0]
    DARK_GREEN = [0.18, 0.42, 0.18, 1.0]


class BaseIconGenerator:
    """Base class for all 3D icon generators"""

    def __init__(self, output_dir="models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _apply_color(self, mesh, color):
        """Apply color to mesh using face_colors"""
        if color:
            rgba = [int(c * 255) if c <= 1.0 else int(c) for c in color[:4]]
            face_colors = np.array([rgba] * len(mesh.faces), dtype=np.uint8)
            mesh.visual.face_colors = face_colors
        return mesh

    def create_box(self, dimensions, position=[0, 0, 0], color=None):
        """Create a box mesh"""
        mesh = creation.box(dimensions=dimensions)
        mesh.apply_translation(position)
        return self._apply_color(mesh, color)

    def create_cylinder(
        self, radius, height, position=[0, 0, 0], color=None, sections=12
    ):
        """Create a cylinder mesh"""
        mesh = creation.cylinder(radius=radius, height=height, sections=sections)
        mesh.apply_translation(position)
        return self._apply_color(mesh, color)

    def create_sphere(self, radius, position=[0, 0, 0], color=None):
        """Create a sphere mesh"""
        mesh = creation.icosphere(radius=radius, subdivisions=1)
        mesh.apply_translation(position)
        return self._apply_color(mesh, color)

    def create_cone(self, radius, height, position=[0, 0, 0], color=None):
        """Create a cone mesh"""
        mesh = creation.cone(radius=radius, height=height)
        mesh.apply_translation(position)
        return self._apply_color(mesh, color)

    def create_torus(self, radius, tube, position=[0, 0, 0], color=None):
        """Create a torus mesh"""
        mesh = creation.torus(major_radius=radius, minor_radius=tube)
        mesh.apply_translation(position)
        return self._apply_color(mesh, color)

    def combine_meshes(self, meshes):
        """Combine multiple meshes into one"""
        return trimesh.util.concatenate(meshes)

    def export_glb(self, mesh, filepath):
        """Export mesh as GLB file"""
        filepath = self.output_dir / filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(file_type="glb", file_obj=str(filepath))
        print(f"[OK] Exported: {filepath}")
        return filepath

    def bevel_edges(self, mesh, radius=0.05):
        """Add bevel to mesh edges for softer look"""
        return mesh


class VehicleGenerator(BaseIconGenerator):
    """Generator for vehicle icons"""

    def car_sedan(self, color=ColorPalette.PRIMARY_BLUE):
        """Create a sedan car model"""
        meshes = []

        # Body base
        body = self.create_box([2.0, 0.8, 4.5], [0, 0.4, 0], color)
        meshes.append(body)

        # Cabin (top part)
        cabin = self.create_box([1.8, 0.7, 2.2], [0, 1.0, -0.3], color)
        meshes.append(cabin)

        # Windows (glass - slightly darker)
        window_color = [c * 0.6 for c in color[:3]] + [1.0]
        windows = self.create_box([1.6, 0.5, 2.0], [0, 1.05, -0.3], window_color)
        meshes.append(windows)

        # Wheels
        wheel_color = ColorPalette.DARK_CHARCOAL
        wheel_positions = [
            [0.9, 0.2, 1.3],
            [-0.9, 0.2, 1.3],
            [0.9, 0.2, -1.3],
            [-0.9, 0.2, -1.3],
        ]
        for pos in wheel_positions:
            wheel = self.create_cylinder(0.35, 0.25, pos, wheel_color, sections=16)
            meshes.append(wheel)

        # Headlights
        light_color = ColorPalette.WHITE
        headlights = [
            self.create_box([0.3, 0.2, 0.1], [0.6, 0.5, 2.25], light_color),
            self.create_box([0.3, 0.2, 0.1], [-0.6, 0.5, 2.25], light_color),
        ]
        meshes.extend(headlights)

        # Taillights
        tail_color = ColorPalette.VEHICLE_RED
        taillights = [
            self.create_box([0.3, 0.2, 0.1], [0.6, 0.5, -2.25], tail_color),
            self.create_box([0.3, 0.2, 0.1], [-0.6, 0.5, -2.25], tail_color),
        ]
        meshes.extend(taillights)

        return self.combine_meshes(meshes)

    def car_suv(self, color=ColorPalette.DARK_CHARCOAL):
        """Create an SUV/Jeep model"""
        meshes = []

        # Body base (taller)
        body = self.create_box([2.0, 0.9, 4.5], [0, 0.45, 0], color)
        meshes.append(body)

        # Cabin (larger, boxier)
        cabin = self.create_box([1.9, 0.9, 2.5], [0, 1.15, 0], color)
        meshes.append(cabin)

        # Windows
        window_color = [c * 0.5 for c in color[:3]] + [1.0]
        windows = self.create_box([1.7, 0.7, 2.3], [0, 1.2, 0], window_color)
        meshes.append(windows)

        # Wheels (bigger)
        wheel_color = ColorPalette.DARK_CHARCOAL
        wheel_positions = [
            [0.95, 0.25, 1.4],
            [-0.95, 0.25, 1.4],
            [0.95, 0.25, -1.4],
            [-0.95, 0.25, -1.4],
        ]
        for pos in wheel_positions:
            wheel = self.create_cylinder(0.4, 0.3, pos, wheel_color, sections=16)
            meshes.append(wheel)

        # Roof rack
        rack = self.create_box([1.7, 0.08, 2.2], [0, 1.65, 0], ColorPalette.ROAD_GRAY)
        meshes.append(rack)

        return self.combine_meshes(meshes)

    def car_hatchback(self, color=ColorPalette.VEHICLE_YELLOW):
        """Create a small hatchback car"""
        meshes = []

        # Body base (shorter, narrower)
        body = self.create_box([1.7, 0.7, 3.8], [0, 0.35, 0], color)
        meshes.append(body)

        # Cabin (sloped rear)
        cabin = self.create_box([1.5, 0.6, 1.8], [0, 0.9, -0.5], color)
        meshes.append(cabin)

        # Windows
        window_color = [c * 0.5 for c in color[:3]] + [1.0]
        windows = self.create_box([1.3, 0.45, 1.6], [0, 0.95, -0.5], window_color)
        meshes.append(windows)

        # Wheels
        wheel_positions = [
            [0.8, 0.18, 1.0],
            [-0.8, 0.18, 1.0],
            [0.8, 0.18, -1.0],
            [-0.8, 0.18, -1.0],
        ]
        for pos in wheel_positions:
            wheel = self.create_cylinder(
                0.3, 0.2, pos, ColorPalette.DARK_CHARCOAL, sections=16
            )
            meshes.append(wheel)

        return self.combine_meshes(meshes)

    def auto_rickshaw(self, color=ColorPalette.VEHICLE_YELLOW):
        """Create Indian auto-rickshaw (3-wheeler)"""
        meshes = []

        # Body base
        body = self.create_box([1.2, 0.6, 2.5], [0, 0.3, 0], color)
        meshes.append(body)

        # Front cabin (driver area)
        front = self.create_box([1.1, 0.8, 1.2], [0, 0.5, 1.2], color)
        meshes.append(front)

        # Canopy (roof)
        roof = self.create_box([1.3, 0.1, 2.8], [0, 1.4, 0], color)
        meshes.append(roof)

        # Windshield
        windshield = self.create_box(
            [1.0, 0.5, 0.1], [0, 1.1, 1.8], ColorPalette.SKY_BLUE
        )
        meshes.append(windshield)

        # Black stripe on front
        stripe = self.create_box([1.15, 0.2, 0.1], [0, 0.6, 1.85], ColorPalette.BLACK)
        meshes.append(stripe)

        # Wheels (3)
        wheel_positions = [[0.55, 0.15, 1.0], [-0.55, 0.15, 1.0], [0, 0.2, -1.0]]
        for pos in wheel_positions:
            radius = 0.35 if pos[2] > 0 else 0.4  # Front wheels smaller
            wheel = self.create_cylinder(
                radius, 0.2, pos, ColorPalette.BLACK, sections=16
            )
            meshes.append(wheel)

        # Headlight
        light = self.create_sphere(0.15, [0, 0.4, 1.9], ColorPalette.WHITE)
        meshes.append(light)

        return self.combine_meshes(meshes)

    def bus_city(self, color=ColorPalette.VEHICLE_RED):
        """Create city bus"""
        meshes = []

        # Main body
        body = self.create_box([2.2, 2.5, 9.0], [0, 1.25, 0], color)
        meshes.append(body)

        # Roof
        roof = self.create_box([2.1, 0.15, 8.8], [0, 2.55, 0], color)
        meshes.append(roof)

        # Windows strip
        window_color = ColorPalette.SKY_BLUE
        windows = self.create_box([2.0, 1.0, 8.5], [0, 1.8, 0], window_color)
        meshes.append(windows)

        # Windows divider
        divider = self.create_box([2.05, 0.1, 8.5], [0, 1.3, 0], ColorPalette.ROAD_GRAY)
        meshes.append(divider)

        # Wheels (6 wheels)
        wheel_positions = [
            [0.95, 0.3, 3.0],
            [-0.95, 0.3, 3.0],
            [0.95, 0.3, 0],
            [-0.95, 0.3, 0],
            [0.95, 0.3, -3.0],
            [-0.95, 0.3, -3.0],
        ]
        for pos in wheel_positions:
            wheel = self.create_cylinder(
                0.45, 0.25, pos, ColorPalette.DARK_CHARCOAL, sections=16
            )
            meshes.append(wheel)

        # Front display
        display = self.create_box(
            [1.5, 0.4, 0.1], [0, 2.0, 4.5], ColorPalette.VEHICLE_YELLOW
        )
        meshes.append(display)

        # Door lines
        for z in [2.0, -1.0]:
            door = self.create_box(
                [2.15, 1.8, 0.05], [0, 1.2, z], ColorPalette.DARK_CHARCOAL
            )
            meshes.append(door)

        return self.combine_meshes(meshes)

    def bus_mini(self, color=ColorPalette.PRIMARY_BLUE):
        """Create mini bus"""
        meshes = []

        # Body
        body = self.create_box([1.8, 1.8, 5.0], [0, 0.9, 0], color)
        meshes.append(body)

        # Roof
        roof = self.create_box([1.7, 0.1, 4.8], [0, 1.85, 0], color)
        meshes.append(roof)

        # Windows
        windows = self.create_box([1.6, 0.8, 4.6], [0, 1.3, 0], ColorPalette.SKY_BLUE)
        meshes.append(windows)

        # Wheels (4)
        wheel_positions = [
            [0.8, 0.25, 1.5],
            [-0.8, 0.25, 1.5],
            [0.8, 0.25, -1.5],
            [-0.8, 0.25, -1.5],
        ]
        for pos in wheel_positions:
            wheel = self.create_cylinder(
                0.35, 0.2, pos, ColorPalette.DARK_CHARCOAL, sections=16
            )
            meshes.append(wheel)

        return self.combine_meshes(meshes)

    def bike_motorcycle(self, color=ColorPalette.DARK_CHARCOAL):
        """Create motorcycle"""
        meshes = []

        # Body/frame
        body = self.create_box([0.3, 0.4, 1.8], [0, 0.5, 0], color)
        meshes.append(body)

        # Tank
        tank = self.create_box([0.25, 0.25, 0.6], [0, 0.6, 0.3], color)
        meshes.append(tank)

        # Seat
        seat = self.create_box([0.25, 0.1, 0.7], [0, 0.65, -0.4], ColorPalette.BLACK)
        meshes.append(seat)

        # Handlebar
        handle = self.create_box(
            [0.6, 0.05, 0.05], [0, 0.75, 0.7], ColorPalette.ROAD_GRAY
        )
        meshes.append(handle)

        # Wheels
        front_wheel = self.create_cylinder(
            0.3, 0.08, [0, 0.3, 0.7], ColorPalette.DARK_CHARCOAL, sections=20
        )
        rear_wheel = self.create_cylinder(
            0.3, 0.08, [0, 0.3, -0.7], ColorPalette.DARK_CHARCOAL, sections=20
        )
        meshes.extend([front_wheel, rear_wheel])

        # Exhaust
        exhaust = self.create_cylinder(
            0.05, 0.8, [0.15, 0.2, -0.3], ColorPalette.SILVER, sections=8
        )
        meshes.append(exhaust)

        # Headlight
        headlight = self.create_sphere(0.1, [0, 0.6, 0.95], ColorPalette.WHITE)
        meshes.append(headlight)

        return self.combine_meshes(meshes)

    def scooter(self, color=ColorPalette.VEHICLE_YELLOW):
        """Create scooter"""
        meshes = []

        # Body (step-through)
        body = self.create_box([0.35, 0.5, 1.5], [0, 0.45, 0], color)
        meshes.append(body)

        # Front panel
        front = self.create_box([0.3, 0.4, 0.4], [0, 0.6, 0.7], color)
        meshes.append(front)

        # Floorboard
        floor = self.create_box([0.35, 0.08, 0.6], [0, 0.2, 0.2], ColorPalette.BLACK)
        meshes.append(floor)

        # Wheels
        front_wheel = self.create_cylinder(
            0.25, 0.1, [0, 0.25, 0.6], ColorPalette.DARK_CHARCOAL, sections=16
        )
        rear_wheel = self.create_cylinder(
            0.28, 0.1, [0, 0.28, -0.6], ColorPalette.DARK_CHARCOAL, sections=16
        )
        meshes.extend([front_wheel, rear_wheel])

        # Headlight
        headlight = self.create_sphere(0.08, [0, 0.65, 0.9], ColorPalette.WHITE)
        meshes.append(headlight)

        return self.combine_meshes(meshes)

    def bicycle(self, color=ColorPalette.PRIMARY_BLUE):
        """Create bicycle"""
        meshes = []

        # Frame
        frame = self.create_box([0.02, 0.02, 1.2], [0, 0.5, 0], color)
        meshes.append(frame)

        # Frame triangles
        top_frame = self.create_box([0.02, 0.4, 0.02], [0, 0.7, 0.3], color)
        seat_post = self.create_box([0.02, 0.25, 0.02], [0, 0.65, -0.3], color)
        meshes.extend([top_frame, seat_post])

        # Wheels (thin rings)
        front_wheel = self.create_torus(
            0.35, 0.02, [0, 0.35, 0.5], ColorPalette.DARK_CHARCOAL
        )
        rear_wheel = self.create_torus(
            0.35, 0.02, [0, 0.35, -0.5], ColorPalette.DARK_CHARCOAL
        )
        meshes.extend([front_wheel, rear_wheel])

        # Handlebar
        handle = self.create_box(
            [0.4, 0.02, 0.02], [0, 0.85, 0.5], ColorPalette.ROAD_GRAY
        )
        meshes.append(handle)

        # Seat
        seat = self.create_box([0.15, 0.05, 0.2], [0, 0.82, -0.35], ColorPalette.BLACK)
        meshes.append(seat)

        return self.combine_meshes(meshes)

    def truck_delivery(self, color=ColorPalette.WHITE):
        """Create delivery truck"""
        meshes = []

        # Cab
        cab = self.create_box([1.8, 1.5, 2.0], [0, 0.75, 1.5], color)
        meshes.append(cab)

        # Cargo box
        cargo = self.create_box([1.8, 2.0, 4.0], [0, 1.0, -1.5], color)
        meshes.append(cargo)

        # Cargo box detail
        cargo_detail = self.create_box(
            [1.7, 1.9, 3.9], [0, 1.0, -1.5], ColorPalette.ROAD_GRAY
        )
        meshes.append(cargo_detail)

        # Windshield
        windshield = self.create_box(
            [1.6, 0.8, 0.1], [0, 1.1, 2.5], ColorPalette.SKY_BLUE
        )
        meshes.append(windshield)

        # Wheels
        wheel_positions = [
            [0.8, 0.3, 1.8],
            [-0.8, 0.3, 1.8],
            [0.8, 0.3, -1.0],
            [-0.8, 0.3, -1.0],
            [0.8, 0.3, -2.5],
            [-0.8, 0.3, -2.5],
        ]
        for pos in wheel_positions:
            wheel = self.create_cylinder(
                0.4, 0.2, pos, ColorPalette.DARK_CHARCOAL, sections=16
            )
            meshes.append(wheel)

        # Headlight
        headlight = self.create_box(
            [0.3, 0.2, 0.1], [0.5, 0.5, 2.55], ColorPalette.WHITE
        )
        meshes.append(headlight)

        return self.combine_meshes(meshes)

    def tempo(self, color=ColorPalette.WHITE):
        """Create Tata Ace type tempo"""
        meshes = []

        # Body
        body = self.create_box([1.5, 1.2, 3.5], [0, 0.6, 0], color)
        meshes.append(body)

        # Cabin
        cabin = self.create_box([1.4, 1.0, 1.5], [0, 1.1, 1.2], color)
        meshes.append(cabin)

        # Cargo
        cargo = self.create_box([1.4, 1.1, 1.8], [0, 0.6, -0.8], ColorPalette.ROAD_GRAY)
        meshes.append(cargo)

        # Wheels (4)
        wheel_positions = [
            [0.65, 0.2, 1.2],
            [-0.65, 0.2, 1.2],
            [0.65, 0.2, -0.8],
            [-0.65, 0.2, -0.8],
        ]
        for pos in wheel_positions:
            wheel = self.create_cylinder(
                0.3, 0.18, pos, ColorPalette.DARK_CHARCOAL, sections=16
            )
            meshes.append(wheel)

        return self.combine_meshes(meshes)

    def tanker(self, color=ColorPalette.VEHICLE_RED):
        """Create fuel tanker"""
        meshes = []

        # Cab
        cab = self.create_box([1.8, 1.8, 1.8], [0, 0.9, 2.0], color)
        meshes.append(cab)

        # Tanker cylinder
        tanker = self.create_cylinder(
            1.0, 5.5, [0, 1.0, -1.5], ColorPalette.VEHICLE_RED, sections=16
        )
        meshes.append(tanker)

        # Tanker ends
        tanker_end1 = self.create_sphere(1.0, [0, 1.0, -4.25], ColorPalette.VEHICLE_RED)
        tanker_end2 = self.create_sphere(1.0, [0, 1.0, 1.25], ColorPalette.VEHICLE_RED)
        meshes.extend([tanker_end1, tanker_end2])

        # Wheels
        wheel_positions = [
            [0.8, 0.35, 2.2],
            [-0.8, 0.35, 2.2],
            [0.8, 0.35, 0],
            [-0.8, 0.35, 0],
            [0.8, 0.35, -2.0],
            [-0.8, 0.35, -2.0],
            [0.8, 0.35, -3.5],
            [-0.8, 0.35, -3.5],
        ]
        for pos in wheel_positions:
            wheel = self.create_cylinder(
                0.45, 0.2, pos, ColorPalette.DARK_CHARCOAL, sections=16
            )
            meshes.append(wheel)

        return self.combine_meshes(meshes)


class BuildingGenerator(BaseIconGenerator):
    """Generator for building icons"""

    def residential_apartment(self, color=ColorPalette.WARM_ORANGE, floors=5):
        """Create apartment building"""
        meshes = []
        height = floors * 0.8

        # Main building
        building = self.create_box([5.0, height, 5.0], [0, height / 2, 0], color)
        meshes.append(building)

        # Windows grid
        window_color = ColorPalette.SKY_BLUE
        window_size = 0.6
        for floor in range(floors):
            y = 0.5 + floor * 0.8
            for x in [-1.5, 0, 1.5]:
                for z in [1.8, -1.8]:
                    window = self.create_box(
                        [window_size, window_size, 0.1], [x, y, 2.5], window_color
                    )
                    meshes.append(window)
                    window2 = self.create_box(
                        [window_size, window_size, 0.1], [x, y, -2.5], window_color
                    )
                    meshes.append(window2)
                for x2 in [1.8, -1.8]:
                    window = self.create_box(
                        [0.1, window_size, window_size], [2.5, y, x2], window_color
                    )
                    meshes.append(window)
                    window2 = self.create_box(
                        [0.1, window_size, window_size], [-2.5, y, x2], window_color
                    )
                    meshes.append(window2)

        # Roof
        roof = self.create_box(
            [5.2, 0.15, 5.2], [0, height + 0.075, 0], ColorPalette.ROAD_GRAY
        )
        meshes.append(roof)

        # Water tank
        tank = self.create_cylinder(
            0.5, 0.8, [1.5, height + 0.55, 1.5], ColorPalette.DARK_CHARCOAL
        )
        meshes.append(tank)

        return self.combine_meshes(meshes)

    def residential_bungalow(self, color=ColorPalette.BEIGE):
        """Create individual house/bungalow"""
        meshes = []

        # Ground floor
        ground = self.create_box([4.0, 0.5, 4.0], [0, 0.25, 0], color)
        meshes.append(ground)

        # First floor
        first = self.create_box([3.5, 0.4, 3.5], [0, 0.7, 0], color)
        meshes.append(first)

        # Roof (pyramid style)
        roof_base = self.create_box([4.2, 0.1, 4.2], [0, 0.9, 0], ColorPalette.BROWN)
        meshes.append(roof_base)

        roof_top = self.create_box([0.1, 0.8, 0.1], [0, 1.4, 0], ColorPalette.BROWN)
        meshes.append(roof_top)

        # Windows
        window_color = ColorPalette.SKY_BLUE
        windows = [
            self.create_box([0.5, 0.5, 0.1], [1.2, 0.3, 2.0], window_color),
            self.create_box([0.5, 0.5, 0.1], [-1.2, 0.3, 2.0], window_color),
            self.create_box([0.5, 0.5, 0.1], [1.2, 0.7, 2.0], window_color),
            self.create_box([0.5, 0.5, 0.1], [-1.2, 0.7, 2.0], window_color),
        ]
        meshes.extend(windows)

        # Door
        door = self.create_box([0.7, 0.8, 0.1], [0, 0.4, 2.0], ColorPalette.BROWN)
        meshes.append(door)

        # Compound wall
        wall = self.create_box([5.0, 0.15, 5.0], [0, 0.075, 0], ColorPalette.LIGHT_GRAY)
        meshes.append(wall)

        return self.combine_meshes(meshes)

    def commercial_shop(self, color=ColorPalette.PRIMARY_BLUE):
        """Create single shop"""
        meshes = []

        # Shop body
        body = self.create_box([3.0, 0.8, 3.0], [0, 0.4, 0], color)
        meshes.append(body)

        # Canopy
        canopy = self.create_box([3.3, 0.1, 1.5], [0, 0.85, 2.0], color)
        meshes.append(canopy)

        # Windows/Display
        display = self.create_box(
            [2.6, 0.5, 0.1], [0, 0.35, 1.5], ColorPalette.SKY_BLUE
        )
        meshes.append(display)

        # Door
        door = self.create_box([0.8, 0.7, 0.1], [0, 0.35, 1.55], ColorPalette.WHITE)
        meshes.append(door)

        # Signboard
        sign = self.create_box(
            [2.0, 0.4, 0.1], [0, 0.95, 1.55], ColorPalette.VEHICLE_YELLOW
        )
        meshes.append(sign)

        return self.combine_meshes(meshes)

    def commercial_mall(self, color=ColorPalette.PRIMARY_BLUE):
        """Create shopping mall"""
        meshes = []

        # Main structure
        main = self.create_box([12.0, 6.0, 10.0], [0, 3.0, 0], color)
        meshes.append(main)

        # Entrance canopy
        canopy = self.create_box([8.0, 0.3, 2.0], [0, 2.8, 5.5], color)
        meshes.append(canopy)

        # Windows
        window_color = ColorPalette.SKY_BLUE
        for y in [1.5, 3.0, 4.5]:
            windows = self.create_box([10.0, 1.0, 0.1], [0, y, 5.0], window_color)
            meshes.append(windows)

        # Entrance
        entrance = self.create_box(
            [3.0, 2.5, 0.1], [0, 1.25, 5.0], ColorPalette.DARK_CHARCOAL
        )
        meshes.append(entrance)

        # Parking area marker
        parking = self.create_box([4.0, 0.05, 3.0], [6.0, 0.025, 0], ColorPalette.WHITE)
        meshes.append(parking)

        # Sign tower
        tower = self.create_box(
            [0.5, 4.0, 0.5], [-5.0, 3.0, 4.0], ColorPalette.WARM_ORANGE
        )
        meshes.append(tower)

        return self.combine_meshes(meshes)

    def commercial_office(self, color=ColorPalette.PRIMARY_BLUE, floors=8):
        """Create office building"""
        meshes = []
        height = floors * 0.9

        # Main building
        building = self.create_box([6.0, height, 6.0], [0, height / 2, 0], color)
        meshes.append(building)

        # Window grid (darker blue tint)
        window_color = [0.2, 0.35, 0.5, 1.0]
        for floor in range(floors):
            y = 0.6 + floor * 0.9
            for x in [-2.0, 0, 2.0]:
                for z in [2.5, -2.5]:
                    window = self.create_box([1.0, 0.5, 0.1], [x, y, 3.0], window_color)
                    meshes.append(window)
                    window2 = self.create_box(
                        [1.0, 0.5, 0.1], [x, y, -3.0], window_color
                    )
                    meshes.append(window2)

        # Roof structure
        roof = self.create_box(
            [6.2, 0.2, 6.2], [0, height + 0.1, 0], ColorPalette.ROAD_GRAY
        )
        meshes.append(roof)

        # Antenna
        antenna = self.create_cylinder(
            0.1, 2.0, [0, height + 1.2, 0], ColorPalette.DARK_CHARCOAL
        )
        meshes.append(antenna)

        return self.combine_meshes(meshes)

    def special_temple(self, color=ColorPalette.WARM_ORANGE):
        """Create temple"""
        meshes = []

        # Base platform
        base = self.create_box([5.0, 0.3, 5.0], [0, 0.15, 0], ColorPalette.BEIGE)
        meshes.append(base)

        # Main shrine
        shrine = self.create_box([2.5, 1.5, 2.5], [0, 1.05, 0], color)
        meshes.append(shrine)

        # Dome
        dome = self.create_sphere(1.5, [0, 2.0, 0], color)
        meshes.append(dome)

        # Spire
        spire = self.create_cone(0.3, 2.0, [0, 3.5, 0], ColorPalette.VEHICLE_YELLOW)
        meshes.append(spire)

        # Entrance
        entrance = self.create_box([0.8, 1.2, 0.1], [0, 0.6, 2.45], ColorPalette.BROWN)
        meshes.append(entrance)

        # Pillars
        pillar_positions = [
            [1.5, 0, 2.0],
            [-1.5, 0, 2.0],
            [1.5, 0, -2.0],
            [-1.5, 0, -2.0],
        ]
        for pos in pillar_positions:
            pillar = self.create_cylinder(
                0.2, 1.3, [pos[0], 0.95, pos[2]], ColorPalette.BEIGE
            )
            meshes.append(pillar)

        return self.combine_meshes(meshes)

    def special_hospital(self, color=ColorPalette.WHITE):
        """Create hospital building"""
        meshes = []

        # Main building
        main = self.create_box([8.0, 4.0, 6.0], [0, 2.0, 0], color)
        meshes.append(main)

        # Emergency bay
        emergency = self.create_box(
            [3.0, 3.0, 2.0], [-4.0, 1.5, 3.0], ColorPalette.VEHICLE_RED
        )
        meshes.append(emergency)

        # Red cross
        cross_v = self.create_box(
            [0.4, 1.5, 0.1], [-4.0, 3.2, 4.1], ColorPalette.VEHICLE_RED
        )
        cross_h = self.create_box(
            [1.2, 0.4, 0.1], [-4.0, 3.2, 4.1], ColorPalette.VEHICLE_RED
        )
        meshes.extend([cross_v, cross_h])

        # Windows
        window_color = ColorPalette.SKY_BLUE
        for y in [1.0, 2.5]:
            for x in [-2.5, 0, 2.5]:
                window = self.create_box([1.0, 0.8, 0.1], [x, y, 3.0], window_color)
                meshes.append(window)

        # Entrance canopy
        canopy = self.create_box(
            [4.0, 0.15, 2.0], [0, 2.0, 3.5], ColorPalette.PRIMARY_BLUE
        )
        meshes.append(canopy)

        # Ambulance bay
        ambulance = self.create_box(
            [2.5, 0.05, 2.0], [-4.0, 0.025, 4.5], ColorPalette.VEHICLE_YELLOW
        )
        meshes.append(ambulance)

        return self.combine_meshes(meshes)

    def special_school(self, color=ColorPalette.WARM_ORANGE):
        """Create school building"""
        meshes = []

        # Main building
        main = self.create_box([10.0, 2.5, 6.0], [0, 1.25, 0], color)
        meshes.append(main)

        # Windows
        window_color = ColorPalette.SKY_BLUE
        for y in [0.8, 2.0]:
            for x in [-3.5, -1.5, 0.5, 2.5]:
                window = self.create_box([1.0, 0.6, 0.1], [x, y, 3.0], window_color)
                meshes.append(window)

        # Clock tower
        tower = self.create_box([1.5, 3.0, 1.5], [4.0, 2.5, 2.0], color)
        meshes.append(tower)
        clock = self.create_cylinder(0.5, 0.1, [4.0, 3.8, 2.8], ColorPalette.WHITE)
        meshes.append(clock)

        # Playground marker
        ground = self.create_box(
            [15.0, 0.02, 10.0], [0, 0.01, -8.0], ColorPalette.FOREST_GREEN
        )
        meshes.append(ground)

        # Flagpole
        pole = self.create_cylinder(
            0.05, 3.0, [-4.0, 1.5, -5.0], ColorPalette.ROAD_GRAY
        )
        meshes.append(pole)

        return self.combine_meshes(meshes)

    def industrial_warehouse(self, color=ColorPalette.ROAD_GRAY):
        """Create warehouse"""
        meshes = []

        # Main structure
        main = self.create_box([12.0, 3.0, 8.0], [0, 1.5, 0], color)
        meshes.append(main)

        # Roof (slanted)
        roof_left = self.create_box(
            [12.5, 0.2, 4.5], [0, 3.2, 2.0], ColorPalette.DARK_CHARCOAL
        )
        roof_right = self.create_box(
            [12.5, 0.2, 4.5], [0, 2.8, -2.0], ColorPalette.DARK_CHARCOAL
        )
        meshes.extend([roof_left, roof_right])

        # Loading dock
        dock = self.create_box([3.0, 0.8, 0.2], [0, 0.4, 4.0], ColorPalette.LIGHT_GRAY)
        meshes.append(dock)

        # Ventilation units
        vent = self.create_box(
            [1.0, 0.5, 1.0], [4.0, 3.5, 2.0], ColorPalette.DARK_CHARCOAL
        )
        meshes.append(vent)

        return self.combine_meshes(meshes)


class InfrastructureGenerator(BaseIconGenerator):
    """Generator for infrastructure icons"""

    def road_lane(self, color=ColorPalette.ROAD_GRAY):
        """Create road lane segment"""
        meshes = []

        # Road surface
        road = self.create_box([3.5, 0.05, 10.0], [0, 0.025, 0], color)
        meshes.append(road)

        # Center line (dashed)
        dash_color = ColorPalette.WHITE
        for z in [-3.5, -1.5, 0.5, 2.5]:
            dash = self.create_box([0.2, 0.06, 1.0], [0, 0.03, z], dash_color)
            meshes.append(dash)

        return self.combine_meshes(meshes)

    def road_intersection(self, color=ColorPalette.ROAD_GRAY):
        """Create road intersection"""
        meshes = []

        # Cross roads
        road_h = self.create_box([10.0, 0.05, 10.0], [0, 0.025, 0], color)
        road_v = self.create_box([10.0, 0.05, 10.0], [0, 0.025, 0], color)
        meshes.extend([road_h, road_v])

        # Zebra crossings
        zebra_color = ColorPalette.WHITE
        for offset in [4.5, -4.5]:
            for x in range(-4, 4):
                stripe = self.create_box(
                    [0.4, 0.06, 0.8], [x * 1.1 + 0.5, 0.03, offset], zebra_color
                )
                meshes.append(stripe)

        return self.combine_meshes(meshes)

    def road_roundabout(self, color=ColorPalette.ROAD_GRAY):
        """Create roundabout"""
        meshes = []

        # Center circle
        center = self.create_cylinder(
            2.0, 0.1, [0, 0.05, 0], ColorPalette.FOREST_GREEN, sections=24
        )
        meshes.append(center)

        # Surrounding road
        outer_road = self.create_cylinder(6.0, 0.05, [0, 0.025, 0], color, sections=32)
        meshes.append(outer_road)

        # Inner hole (visual effect with darker ring)
        inner_road = self.create_cylinder(
            4.0, 0.05, [0, 0.026, 0], ColorPalette.ROAD_GRAY, sections=32
        )
        meshes.append(inner_road)

        return self.combine_meshes(meshes)

    def road_highway(self, color=ColorPalette.ROAD_GRAY):
        """Create highway segment"""
        meshes = []

        # Road surface
        road = self.create_box([7.0, 0.08, 20.0], [0, 0.04, 0], color)
        meshes.append(road)

        # White edge lines
        line_color = ColorPalette.WHITE
        edge1 = self.create_box([0.15, 0.09, 20.0], [3.3, 0.045, 0], line_color)
        edge2 = self.create_box([0.15, 0.09, 20.0], [-3.3, 0.045, 0], line_color)
        meshes.extend([edge1, edge2])

        # Yellow center divider
        divider = self.create_box(
            [0.3, 0.1, 20.0], [0, 0.05, 0], ColorPalette.VEHICLE_YELLOW
        )
        meshes.append(divider)

        return self.combine_meshes(meshes)

    def traffic_signal(self):
        """Create traffic signal pole"""
        meshes = []

        # Pole
        pole = self.create_cylinder(0.1, 3.0, [0, 1.5, 0], ColorPalette.DARK_CHARCOAL)
        meshes.append(pole)

        # Signal box
        box = self.create_box([0.4, 0.8, 0.3], [0, 3.2, 0], ColorPalette.DARK_CHARCOAL)
        meshes.append(box)

        # Red light
        red = self.create_sphere(0.1, [0, 3.4, 0.16], ColorPalette.VEHICLE_RED)
        meshes.append(red)

        # Yellow light
        yellow = self.create_sphere(0.1, [0, 3.2, 0.16], ColorPalette.VEHICLE_YELLOW)
        meshes.append(yellow)

        # Green light
        green = self.create_sphere(0.1, [0, 3.0, 0.16], ColorPalette.FOREST_GREEN)
        meshes.append(green)

        # Base
        base = self.create_box([0.5, 0.15, 0.5], [0, 0.075, 0], ColorPalette.ROAD_GRAY)
        meshes.append(base)

        return self.combine_meshes(meshes)

    def traffic_sign_stop(self):
        """Create stop sign"""
        meshes = []

        # Pole
        pole = self.create_cylinder(0.05, 2.0, [0, 1.0, 0], ColorPalette.ROAD_GRAY)
        meshes.append(pole)

        # Sign (octagon)
        sign = self.create_cylinder(
            0.4, 0.1, [0, 2.3, 0], ColorPalette.VEHICLE_RED, sections=8
        )
        meshes.append(sign)

        # White border
        border = self.create_cylinder(
            0.45, 0.08, [0, 2.3, 0], ColorPalette.WHITE, sections=8
        )
        meshes.append(border)

        # Base
        base = self.create_box([0.3, 0.1, 0.3], [0, 0.05, 0], ColorPalette.ROAD_GRAY)
        meshes.append(base)

        return self.combine_meshes(meshes)

    def traffic_barrier(self, color=ColorPalette.VEHICLE_YELLOW):
        """Create construction barrier"""
        meshes = []

        # Main barrier
        barrier = self.create_box([0.1, 0.8, 2.0], [0, 0.4, 0], color)
        meshes.append(barrier)

        # Stripes
        for z in [-0.6, 0, 0.6]:
            stripe = self.create_box([0.11, 0.15, 0.3], [0, 0.5, z], ColorPalette.BLACK)
            meshes.append(stripe)

        # Legs
        leg1 = self.create_box(
            [0.1, 0.1, 0.3], [0.1, 0.05, 0.7], ColorPalette.ROAD_GRAY
        )
        leg2 = self.create_box(
            [0.1, 0.1, 0.3], [-0.1, 0.05, 0.7], ColorPalette.ROAD_GRAY
        )
        leg3 = self.create_box(
            [0.1, 0.1, 0.3], [0.1, 0.05, -0.7], ColorPalette.ROAD_GRAY
        )
        leg4 = self.create_box(
            [0.1, 0.1, 0.3], [-0.1, 0.05, -0.7], ColorPalette.ROAD_GRAY
        )
        meshes.extend([leg1, leg2, leg3, leg4])

        return self.combine_meshes(meshes)

    def street_light_pole(self):
        """Create street light"""
        meshes = []

        # Pole
        pole = self.create_cylinder(0.08, 4.0, [0, 2.0, 0], ColorPalette.ROAD_GRAY)
        meshes.append(pole)

        # Arm
        arm = self.create_box([1.2, 0.08, 0.08], [0.6, 4.0, 0], ColorPalette.ROAD_GRAY)
        meshes.append(arm)

        # Light fixture
        fixture = self.create_box(
            [0.6, 0.15, 0.4], [1.2, 3.9, 0], ColorPalette.LIGHT_GRAY
        )
        meshes.append(fixture)

        # Light bulb (emissive)
        bulb = self.create_sphere(0.12, [1.2, 3.8, 0], ColorPalette.VEHICLE_YELLOW)
        meshes.append(bulb)

        # Base
        base = self.create_box([0.4, 0.15, 0.4], [0, 0.075, 0], ColorPalette.ROAD_GRAY)
        meshes.append(base)

        return self.combine_meshes(meshes)

    def street_bench(self, color=ColorPalette.BROWN):
        """Create park bench"""
        meshes = []

        # Seat
        seat = self.create_box([1.2, 0.08, 0.4], [0, 0.4, 0], color)
        meshes.append(seat)

        # Backrest
        backrest = self.create_box([1.2, 0.5, 0.08], [0, 0.7, -0.16], color)
        meshes.append(backrest)

        # Legs
        leg1 = self.create_box([0.08, 0.4, 0.35], [0.5, 0.2, 0], ColorPalette.ROAD_GRAY)
        leg2 = self.create_box(
            [0.08, 0.4, 0.35], [-0.5, 0.2, 0], ColorPalette.ROAD_GRAY
        )
        meshes.extend([leg1, leg2])

        return self.combine_meshes(meshes)


class EnvironmentGenerator(BaseIconGenerator):
    """Generator for environment icons"""

    def tree_large(
        self, trunk_color=ColorPalette.BROWN, foliage_color=ColorPalette.FOREST_GREEN
    ):
        """Create large tree (banyan style)"""
        meshes = []

        # Trunk
        trunk = self.create_cylinder(0.3, 1.5, [0, 0.75, 0], trunk_color, sections=12)
        meshes.append(trunk)

        # Trunk detail
        trunk_top = self.create_cylinder(
            0.4, 0.3, [0, 1.65, 0], trunk_color, sections=12
        )
        meshes.append(trunk_top)

        # Foliage (layered spheres)
        foliage1 = self.create_sphere(1.2, [0, 2.5, 0], foliage_color)
        meshes.append(foliage1)

        foliage2 = self.create_sphere(1.0, [0.5, 2.8, 0.3], foliage_color)
        meshes.append(foliage2)

        foliage3 = self.create_sphere(1.0, [-0.4, 2.7, -0.3], foliage_color)
        meshes.append(foliage3)

        # Ground shadow/base
        base = self.create_cylinder(0.8, 0.05, [0, 0.025, 0], trunk_color)
        meshes.append(base)

        return self.combine_meshes(meshes)

    def tree_palm(
        self, trunk_color=ColorPalette.BROWN, foliage_color=ColorPalette.DARK_GREEN
    ):
        """Create palm tree"""
        meshes = []

        # Trunk (curved appearance with cylinder)
        trunk = self.create_cylinder(0.15, 3.0, [0, 1.5, 0], trunk_color, sections=8)
        meshes.append(trunk)

        # Palm fronds
        for angle in [0, 60, 120, 180, 240, 300]:
            rad = np.radians(angle)
            x = np.cos(rad) * 0.8
            z = np.sin(rad) * 0.8
            frond = self.create_box([1.5, 0.05, 0.3], [x, 3.2, z], foliage_color)
            frond.apply_transform(rotation_matrix(np.radians(angle), [0, 1, 0]))
            meshes.append(frond)

        # Top cluster
        top = self.create_sphere(0.4, [0, 3.1, 0], foliage_color)
        meshes.append(top)

        return self.combine_meshes(meshes)

    def tree_small(self, color=ColorPalette.FOREST_GREEN):
        """Create small tree/shrub"""
        meshes = []

        # Trunk
        trunk = self.create_cylinder(
            0.1, 0.8, [0, 0.4, 0], ColorPalette.BROWN, sections=8
        )
        meshes.append(trunk)

        # Foliage
        foliage = self.create_sphere(0.6, [0, 1.2, 0], color)
        meshes.append(foliage)

        # Ground patch
        ground = self.create_cylinder(0.4, 0.03, [0, 0.015, 0], ColorPalette.BROWN)
        meshes.append(ground)

        return self.combine_meshes(meshes)

    def tree_bush(self, color=ColorPalette.DARK_GREEN):
        """Create bush/shrub"""
        meshes = []

        # Main bush
        bush = self.create_sphere(0.5, [0, 0.5, 0], color)
        meshes.append(bush)

        # Smaller additions
        bush1 = self.create_sphere(0.35, [0.3, 0.45, 0.2], color)
        bush2 = self.create_sphere(0.35, [-0.25, 0.42, 0.15], color)
        meshes.extend([bush1, bush2])

        return self.combine_meshes(meshes)

    def green_grass(self, color=ColorPalette.FOREST_GREEN):
        """Create grass patch"""
        mesh = self.create_box([2.0, 0.08, 2.0], [0, 0.04, 0], color)
        return mesh

    def green_flower(
        self, color1=ColorPalette.VEHICLE_RED, color2=ColorPalette.VEHICLE_YELLOW
    ):
        """Create flower bed"""
        meshes = []

        # Grass base
        grass = self.create_box(
            [1.5, 0.05, 1.5], [0, 0.025, 0], ColorPalette.FOREST_GREEN
        )
        meshes.append(grass)

        # Flowers
        flower_positions = [
            [0.3, 0.15, 0.2],
            [-0.2, 0.15, 0.3],
            [0.1, 0.15, -0.3],
            [-0.3, 0.15, -0.2],
            [0.4, 0.15, -0.1],
        ]
        for i, pos in enumerate(flower_positions):
            flower_color = color1 if i % 2 == 0 else color2
            stem = self.create_cylinder(
                0.02, 0.2, [pos[0], 0.1, pos[2]], ColorPalette.FOREST_GREEN
            )
            bloom = self.create_sphere(
                0.08, [pos[0], pos[1] + 0.1, pos[2]], flower_color
            )
            meshes.extend([stem, bloom])

        return self.combine_meshes(meshes)


class HumanGenerator(BaseIconGenerator):
    """Generator for human figures"""

    def pedestrian(self, color=ColorPalette.PRIMARY_BLUE):
        """Create walking pedestrian"""
        meshes = []

        # Head
        head = self.create_sphere(0.12, [0, 1.6, 0], ColorPalette.WARM_ORANGE)
        meshes.append(head)

        # Body
        body = self.create_box([0.25, 0.5, 0.15], [0, 1.15, 0], color)
        meshes.append(body)

        # Arms (swinging)
        arm1 = self.create_box([0.08, 0.35, 0.08], [0.18, 1.0, 0.1], color)
        arm2 = self.create_box([0.08, 0.35, 0.08], [-0.18, 1.0, -0.1], color)
        meshes.extend([arm1, arm2])

        # Legs (walking stance)
        leg1 = self.create_box(
            [0.1, 0.4, 0.1], [0.08, 0.65, 0.1], ColorPalette.DARK_CHARCOAL
        )
        leg2 = self.create_box(
            [0.1, 0.4, 0.1], [-0.08, 0.65, -0.1], ColorPalette.DARK_CHARCOAL
        )
        meshes.extend([leg1, leg2])

        return self.combine_meshes(meshes)

    def standing(self, color=ColorPalette.VEHICLE_RED):
        """Create standing person"""
        meshes = []

        # Head
        head = self.create_sphere(0.12, [0, 1.6, 0], ColorPalette.WARM_ORANGE)
        meshes.append(head)

        # Body
        body = self.create_box([0.25, 0.5, 0.15], [0, 1.15, 0], color)
        meshes.append(body)

        # Arms (at sides)
        arm1 = self.create_box([0.08, 0.35, 0.08], [0.18, 1.0, 0], color)
        arm2 = self.create_box([0.08, 0.35, 0.08], [-0.18, 1.0, 0], color)
        meshes.extend([arm1, arm2])

        # Legs
        leg1 = self.create_box(
            [0.1, 0.4, 0.1], [0.08, 0.65, 0], ColorPalette.DARK_CHARCOAL
        )
        leg2 = self.create_box(
            [0.1, 0.4, 0.1], [-0.08, 0.65, 0], ColorPalette.DARK_CHARCOAL
        )
        meshes.extend([leg1, leg2])

        return self.combine_meshes(meshes)

    def cyclist(self, body_color=ColorPalette.PRIMARY_BLUE):
        """Create cyclist"""
        meshes = []

        # Head
        head = self.create_sphere(0.1, [0, 1.3, 0.3], ColorPalette.WARM_ORANGE)
        meshes.append(head)

        # Body (leaning forward)
        body = self.create_box([0.2, 0.4, 0.12], [0, 1.0, 0.1], body_color)
        meshes.append(body)

        # Arms (on handlebars)
        arms = self.create_box([0.06, 0.25, 0.06], [0, 1.1, 0.45], body_color)
        meshes.append(arms)

        # Legs (pedaling)
        leg1 = self.create_box(
            [0.08, 0.3, 0.08], [0.1, 0.65, 0.2], ColorPalette.DARK_CHARCOAL
        )
        leg2 = self.create_box(
            [0.08, 0.3, 0.08], [-0.1, 0.65, 0.0], ColorPalette.DARK_CHARCOAL
        )
        meshes.extend([leg1, leg2])

        # Bicycle
        frame = self.create_box(
            [0.02, 0.02, 0.8], [0, 0.5, 0.1], ColorPalette.DARK_CHARCOAL
        )
        meshes.append(frame)

        # Wheels
        front_wheel = self.create_torus(
            0.3, 0.02, [0, 0.3, 0.5], ColorPalette.DARK_CHARCOAL
        )
        rear_wheel = self.create_torus(
            0.3, 0.02, [0, 0.3, -0.3], ColorPalette.DARK_CHARCOAL
        )
        meshes.extend([front_wheel, rear_wheel])

        # Handlebar
        handle = self.create_box(
            [0.35, 0.02, 0.02], [0, 0.75, 0.5], ColorPalette.ROAD_GRAY
        )
        meshes.append(handle)

        # Seat
        seat = self.create_box([0.12, 0.04, 0.15], [0, 0.68, -0.1], ColorPalette.BLACK)
        meshes.append(seat)

        return self.combine_meshes(meshes)


class MarkerGenerator(BaseIconGenerator):
    """Generator for map markers"""

    def pin_location(self, color=ColorPalette.VEHICLE_RED):
        """Create location pin marker"""
        meshes = []

        # Pin head
        head = self.create_sphere(0.4, [0, 0.5, 0], color)
        meshes.append(head)

        # Pin point
        point = self.create_cone(0.15, 1.0, [0, -0.2, 0], color)
        meshes.append(point)

        # White circle on top
        circle = self.create_cylinder(
            0.15, 0.05, [0, 0.75, 0], ColorPalette.WHITE, sections=16
        )
        meshes.append(circle)

        return self.combine_meshes(meshes)

    def start_marker(self, color=ColorPalette.FOREST_GREEN):
        """Create start point marker"""
        meshes = []

        # Circle base
        base = self.create_cylinder(0.6, 0.1, [0, 0.05, 0], color, sections=24)
        meshes.append(base)

        # Center dot
        center = self.create_cylinder(
            0.2, 0.15, [0, 0.1, 0], ColorPalette.WHITE, sections=16
        )
        meshes.append(center)

        return self.combine_meshes(meshes)

    def end_marker(self, color=ColorPalette.VEHICLE_RED):
        """Create end point marker"""
        meshes = []

        # Flag base
        base = self.create_box([0.8, 0.1, 0.8], [0, 0.05, 0], ColorPalette.ROAD_GRAY)
        meshes.append(base)

        # Flag pole
        pole = self.create_cylinder(0.05, 1.5, [0, 0.8, 0], ColorPalette.ROAD_GRAY)
        meshes.append(pole)

        # Flag
        flag = self.create_box([0.6, 0.4, 0.05], [0.3, 1.2, 0], color)
        meshes.append(flag)

        return self.combine_meshes(meshes)

    def poi_marker(self, color=ColorPalette.VEHICLE_YELLOW):
        """Create point of interest marker"""
        meshes = []

        # Cone shape
        cone = self.create_cone(0.4, 0.8, [0, 0.5, 0], color)
        meshes.append(cone)

        # White ring
        ring = self.create_torus(0.35, 0.05, [0, 0.3, 0], ColorPalette.WHITE)
        meshes.append(ring)

        return self.combine_meshes(meshes)


def generate_all_assets(output_dir="models"):
    """Generate all 3D assets and export as GLB files"""

    # Initialize generators
    vehicles = VehicleGenerator(output_dir)
    buildings = BuildingGenerator(output_dir)
    infrastructure = InfrastructureGenerator(output_dir)
    environment = EnvironmentGenerator(output_dir)
    humans = HumanGenerator(output_dir)
    markers = MarkerGenerator(output_dir)

    print("=" * 60)
    print("3D ICON GENERATOR - Indian Navigation Interface")
    print("=" * 60)
    print()

    # === VEHICLES ===
    print("Generating Vehicles...")
    vehicle_models = [
        ("veh_car_sedan_blue.glb", vehicles.car_sedan(ColorPalette.PRIMARY_BLUE)),
        ("veh_car_suv_dark.glb", vehicles.car_suv(ColorPalette.DARK_CHARCOAL)),
        (
            "veh_car_hatchback_yellow.glb",
            vehicles.car_hatchback(ColorPalette.VEHICLE_YELLOW),
        ),
        ("veh_car_sedan_silver.glb", vehicles.car_sedan(ColorPalette.SILVER)),
        ("veh_car_suv_white.glb", vehicles.car_suv(ColorPalette.WHITE)),
        ("veh_auto_rickshaw.glb", vehicles.auto_rickshaw()),
        ("veh_bus_city_red.glb", vehicles.bus_city()),
        ("veh_bus_mini_blue.glb", vehicles.bus_mini()),
        ("veh_bike_motorcycle.glb", vehicles.bike_motorcycle()),
        ("veh_scooter_yellow.glb", vehicles.scooter()),
        ("veh_bicycle_blue.glb", vehicles.bicycle()),
        ("veh_truck_delivery.glb", vehicles.truck_delivery()),
        ("veh_tempo_white.glb", vehicles.tempo()),
        ("veh_tanker_red.glb", vehicles.tanker()),
    ]

    for filename, mesh in vehicle_models:
        vehicles.export_glb(mesh, f"vehicles/{filename}")

    print()

    # === BUILDINGS ===
    print("Generating Buildings...")
    building_models = [
        (
            "residential/bld_apartment_orange.glb",
            buildings.residential_apartment(ColorPalette.WARM_ORANGE, 5),
        ),
        (
            "residential/bld_apartment_blue.glb",
            buildings.residential_apartment(ColorPalette.PRIMARY_BLUE, 8),
        ),
        ("residential/bld_bungalow_beige.glb", buildings.residential_bungalow()),
        ("commercial/bld_shop_blue.glb", buildings.commercial_shop()),
        ("commercial/bld_mall_blue.glb", buildings.commercial_mall()),
        ("commercial/bld_office_tower.glb", buildings.commercial_office(floors=12)),
        ("special/bld_temple_orange.glb", buildings.special_temple()),
        ("special/bld_hospital_white.glb", buildings.special_hospital()),
        ("special/bld_school_orange.glb", buildings.special_school()),
        ("industrial/bld_warehouse_gray.glb", buildings.industrial_warehouse()),
    ]

    for filepath, mesh in building_models:
        buildings.export_glb(mesh, f"buildings/{filepath}")

    print()

    # === INFRASTRUCTURE ===
    print("Generating Infrastructure...")
    infra_models = [
        ("roads/inf_road_lane.glb", infrastructure.road_lane()),
        ("roads/inf_intersection.glb", infrastructure.road_intersection()),
        ("roads/inf_roundabout.glb", infrastructure.road_roundabout()),
        ("roads/inf_highway.glb", infrastructure.road_highway()),
        ("traffic/inf_traffic_signal.glb", infrastructure.traffic_signal()),
        ("traffic/inf_sign_stop.glb", infrastructure.traffic_sign_stop()),
        ("traffic/inf_barrier.glb", infrastructure.traffic_barrier()),
        ("street/inf_street_light.glb", infrastructure.street_light_pole()),
        ("street/inf_bench.glb", infrastructure.street_bench()),
    ]

    for filepath, mesh in infra_models:
        infrastructure.export_glb(mesh, f"infrastructure/{filepath}")

    print()

    # === ENVIRONMENT ===
    print("Generating Environment...")
    env_models = [
        ("trees/env_tree_large.glb", environment.tree_large()),
        ("trees/env_tree_palm.glb", environment.tree_palm()),
        ("trees/env_tree_small.glb", environment.tree_small()),
        ("trees/env_bush.glb", environment.tree_bush()),
        ("greenery/env_grass.glb", environment.green_grass()),
        ("greenery/env_flower.glb", environment.green_flower()),
    ]

    for filepath, mesh in env_models:
        environment.export_glb(mesh, f"environment/{filepath}")

    print()

    # === HUMANS ===
    print("Generating Humans...")
    human_models = [
        ("hum_pedestrian_blue.glb", humans.pedestrian(ColorPalette.PRIMARY_BLUE)),
        ("hum_pedestrian_red.glb", humans.pedestrian(ColorPalette.VEHICLE_RED)),
        ("hum_standing.glb", humans.standing()),
        ("hum_cyclist.glb", humans.cyclist()),
    ]

    for filename, mesh in human_models:
        humans.export_glb(mesh, f"humans/{filename}")

    print()

    # === MARKERS ===
    print("Generating Markers...")
    marker_models = [
        ("mrk_pin_location.glb", markers.pin_location()),
        ("mrk_start.glb", markers.start_marker()),
        ("mrk_end.glb", markers.end_marker()),
        ("mrk_poi.glb", markers.poi_marker()),
    ]

    for filename, mesh in marker_models:
        markers.export_glb(mesh, f"markers/{filename}")

    print()
    print("=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(
        f"\nTotal assets generated: {len(vehicle_models) + len(building_models) + len(infra_models) + len(env_models) + len(human_models) + len(marker_models)}"
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    generate_all_assets()
