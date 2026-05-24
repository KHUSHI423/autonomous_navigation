# 3D Icon Inventory for Indian Navigation & Mapping Interface

A comprehensive set of **47 3D assets in GLB format** designed for real-time navigation, autonomous system mapping, and urban visualization interfaces. Optimized for Indian urban environments.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate all 3D assets
python generator.py
```

## Generated Assets (47 Total)

### Vehicles (14 assets)
- `veh_car_sedan_blue.glb` - Standard sedan
- `veh_car_suv_dark.glb` - SUV/Jeep
- `veh_car_hatchback_yellow.glb` - Small car
- `veh_car_sedan_silver.glb` - Silver sedan
- `veh_car_suv_white.glb` - White SUV
- `veh_auto_rickshaw.glb` - Indian 3-wheeler (yellow/black)
- `veh_bus_city_red.glb` - City bus
- `veh_bus_mini_blue.glb` - Mini bus
- `veh_bike_motorcycle.glb` - Motorcycle
- `veh_scooter_yellow.glb` - Scooter
- `veh_bicycle_blue.glb` - Bicycle
- `veh_truck_delivery.glb` - Delivery truck
- `veh_tempo_white.glb` - Tata Ace type tempo
- `veh_tanker_red.glb` - Fuel tanker

### Buildings (10 assets)
#### Residential
- `bld_apartment_orange.glb` - 5-story apartment
- `bld_apartment_blue.glb` - 8-story apartment
- `bld_bungalow_beige.glb` - Individual house

#### Commercial
- `bld_shop_blue.glb` - Single shop
- `bld_mall_blue.glb` - Shopping mall
- `bld_office_tower.glb` - 12-story office building

#### Special
- `bld_temple_orange.glb` - Temple
- `bld_hospital_white.glb` - Hospital
- `bld_school_orange.glb` - School

#### Industrial
- `bld_warehouse_gray.glb` - Warehouse

### Infrastructure (9 assets)
#### Roads
- `inf_road_lane.glb` - Road lane segment
- `inf_intersection.glb` - Crossroads
- `inf_roundabout.glb` - Roundabout
- `inf_highway.glb` - Highway segment

#### Traffic
- `inf_traffic_signal.glb` - Traffic light
- `inf_sign_stop.glb` - Stop sign
- `inf_barrier.glb` - Construction barrier

#### Street
- `inf_street_light.glb` - Street lamp
- `inf_bench.glb` - Park bench

### Environment (6 assets)
#### Trees
- `env_tree_large.glb` - Large tree (banyan style)
- `env_tree_palm.glb` - Palm tree
- `env_tree_small.glb` - Small tree
- `env_bush.glb` - Bush/shrub

#### Greenery
- `env_grass.glb` - Grass patch
- `env_flower.glb` - Flower bed

### Humans (4 assets)
- `hum_pedestrian_blue.glb` - Walking person (blue)
- `hum_pedestrian_red.glb` - Walking person (red)
- `hum_standing.glb` - Standing person
- `hum_cyclist.glb` - Person on bicycle

### Markers (4 assets)
- `mrk_pin_location.glb` - Location pin
- `mrk_start.glb` - Start marker
- `mrk_end.glb` - End marker (flag)
- `mrk_poi.glb` - Point of interest

## Directory Structure

```
models/
├── vehicles/
│   └── 14 GLB files
├── buildings/
│   ├── residential/
│   ├── commercial/
│   ├── special/
│   └── industrial/
├── infrastructure/
│   ├── roads/
│   ├── traffic/
│   └── street/
├── environment/
│   ├── trees/
│   └── greenery/
├── humans/
└── markers/
```

## Usage in Game Engines

### Unity
1. Import GLB files via Assets > Import Package
2. Or drag and drop GLB files into the Hierarchy
3. Adjust scale as needed (base unit = 1 meter)

### Godot
1. Import GLB files via AssetLib or drag into project
2. Use as-is or instance in 3D scenes

### Unreal Engine
1. Use glTF Importer plugin
2. Drag GLB files into Content Browser

### Three.js / WebGL
```javascript
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load('models/vehicles/veh_car_sedan_blue.glb', (gltf) => {
    scene.add(gltf.scene);
});
```

## Color Palette

| Color Name      | Hex Code  | RGB              |
|----------------|-----------|------------------|
| Primary Blue   | #4A90D9   | (74, 144, 217)   |
| Warm Orange    | #E8A87C   | (232, 168, 124)  |
| Forest Green   | #3D8B40   | (61, 139, 64)    |
| Road Gray      | #6B7280   | (107, 114, 128)   |
| Vehicle Yellow | #F4C430   | (244, 196, 48)   |
| Vehicle Red    | #DC4545   | (220, 69, 69)    |
| Sky Blue       | #87CEEB   | (135, 206, 235)  |
| Light Gray     | #E5E7EB   | (229, 231, 235)  |
| Dark Charcoal  | #374151   | (55, 65, 81)     |

## Scale Reference

- 1 unit = 1 meter
- Road width: 3.5 units (single lane)
- Car length: 4.5 units
- Bus length: 10 units
- Person height: 1.7 units
- Tree height: 3-5 units

## Technical Details

- **Format**: GLB (binary glTF 2.0)
- **Coordinate System**: Y-up, right-handed
- **Materials**: PBR with base color
- **Polygon Count**: 50-500 triangles per asset
- **Optimization**: Low-poly for real-time rendering

## Customization

To generate variations or add new assets:

1. Edit `generator.py`
2. Add new methods to existing generator classes
3. Run `python generator.py`

## License

Generated assets are free for commercial and personal use in navigation/mapping applications.

## Credits

Generated using Python with [trimesh](https://trimsh.org/) library.
