# 3D Icon Inventory for Indian Navigation & Mapping Interface

## Project Overview

A comprehensive set of 3D assets (GLB format) designed for real-time navigation, autonomous system mapping, and urban visualization interfaces. Optimized for Indian urban environments with consistent styling and modularity.

## Visual Design Language

### Color Palette

Based on minimalist 3D aesthetic with soft lighting:

- **Primary Blue**: `#4A90D9` - Buildings, primary elements
- **Warm Orange**: `#E8A87C` - Residential buildings, warm tones
- **Forest Green**: `#3D8B40` - Trees, vegetation
- **Road Gray**: `#6B7280` - Roads, paths
- **Vehicle Yellow**: `#F4C430` - Auto-rickshaws, taxis
- **Vehicle Red**: `#DC4545` - Buses, trucks, markers
- **Sky Blue**: `#87CEEB` - Ambient lighting
- **Light Gray**: `#E5E7EB` - Sidewalks, barriers
- **Dark Charcoal**: `#374151` - Shadows, accents

### Lighting & Style

- **Soft ambient occlusion**: Subtle shadows for depth
- **Clean edges**: Beveled corners for modern look
- **Flat shading with subtle gradients**: No hyper-realistic textures
- **Low-poly optimized**: Triangulated meshes under 500 polygons per asset
- **Top-view optimized**: Clear silhouettes from bird's eye perspective

## Asset Naming Conventions

Format: `{category}_{subcategory}_{variant}_{color}_{scale}.glb`

Examples:
- `veh_car_sedan_blue_1x.glb`
- `veh_bus_city_red_2x.glb`
- `bld_residential_apartment_orange_1x.glb`
- `env_tree_palm_green_1x.glb`

## Category Structure

### 1. Vehicles (Category: veh)

#### Cars (subcategory: car)
- `car_sedan` - Standard sedan
- `car_suv` - SUV/Jeep
- `car_hatchback` - Small car
- `car_luxury` - Premium car

#### Auto-Rickshaws (subcategory: auto)
- `auto_3wheeler` - Classic 3-wheeler (yellow/black)
- `auto_electric` - E-rickshaw (green)

#### Buses (subcategory: bus)
- `bus_city` - City bus (red/acqua)
- `bus_mini` - Mini bus
- `bus_double` - Double-decker

#### Two-Wheelers (subcategory: 2w)
- `2w_bike` - Motorcycle
- `2w_scooter` - Scooter
- `2w_cycle` - Bicycle

#### Trucks & Commercial (subcategory: comm)
- `comm_truck` - Delivery truck
- `comm_tempo` - Tata Ace type
- `comm_tanker` - Fuel tanker

### 2. Buildings (Category: bld)

#### Residential (subcategory: res)
- `res_apartment` - Multi-story apartment
- `res_bungalow` - Individual house
- `res_rowhouse` - Row housing
- `res_chawl` - Traditional chawl

#### Commercial (subcategory: comm)
- `comm_shop` - Single shop
- `comm_mall` - Shopping complex
- `comm_office` - Office building
- `comm_hotel` - Hotel

#### Industrial (subcategory: ind)
- `ind_warehouse` - Storage facility
- `ind_factory` - Manufacturing unit
- `ind_power` - Power station

#### Special (subcategory: spec)
- `spec_temple` - Temple
- `spec_mosque` - Mosque
- `spec_church` - Church
- `spec_gurudwara` - Gurudwara
- `spec_hospital` - Hospital
- `spec_school` - School
- `spec_park` - Park pavilion

### 3. Infrastructure (Category: inf)

#### Roads (subcategory: road)
- `road_lane` - Single lane
- `road_intersection` - Crossroads
- `road_roundabout` - Circular junction
- `road_highway` - Highway segment

#### Traffic Elements (subcategory: traffic)
- `traffic_signal` - Traffic light pole
- `traffic_sign_stop` - Stop sign
- `traffic_sign_info` - Information sign
- `traffic_barrier` - Construction barrier

#### Street Furniture (subcategory: street)
- `street_light_pole` - Street light
- `street_bench` - Park bench
- `street_billboard` - Advertisement billboard

### 4. Environment (Category: env)

#### Trees (subcategory: tree)
- `tree_large` - Large tree (banyan, peepal)
- `tree_palm` - Palm tree
- `tree_small` - Small tree
- `tree_bush` - Bush/shrub

#### Greenery (subcategory: green)
- `green_grass` - Grass patch
- `green_flower` - Flower bed

### 5. Humans (Category: hum)

- `hum_pedestrian` - Walking person
- `hum_standing` - Standing person
- `hum_cyclist` - Person on bicycle

### 6. Markers (Category: mrk)

- `mrk_pin` - Location pin
- `mrk_start` - Start point
- `mrk_end` - End point
- `mrk_poi` - Point of interest

## Technical Specifications

### Polygon Budget
- **Vehicles**: 200-500 triangles
- **Buildings**: 100-300 triangles
- **Trees**: 50-150 triangles
- **Roads**: 50-100 triangles
- **People**: 100-200 triangles

### Scale Reference
- **Base unit**: 1 unit = 1 meter
- **Road width**: 3.5 units (single lane)
- **Car length**: 4.5 units
- **Bus length**: 10 units
- **Building height**: Variable (3-100 units)
- **Person height**: 1.7 units

### Export Format
- **Format**: GLB (binary glTF 2.0)
- **Coordinate system**: Y-up, right-handed
- **Units**: Meters
- **Materials**: PBR with base color only (for compatibility)

### File Organization

```
/
├── SPEC.md
├── requirements.txt
├── models/
│   ├── vehicles/
│   │   ├── car_sedan_blue.glb
│   │   ├── car_suv_silver.glb
│   │   └── ...
│   ├── buildings/
│   │   ├── residential/
│   │   ├── commercial/
│   │   └── special/
│   ├── infrastructure/
│   │   ├── roads/
│   │   ├── traffic/
│   │   └── street/
│   ├── environment/
│   │   ├── trees/
│   │   └── greenery/
│   └── humans/
├── generator.py
└── export_all.py
```

## Implementation Notes

- All models use simple geometric primitives (boxes, cylinders, spheres)
- Low-poly aesthetic maintained throughout
- Consistent material properties for easy theming
- Modular design allows dynamic assembly
- Ready for instanced rendering in game engines
