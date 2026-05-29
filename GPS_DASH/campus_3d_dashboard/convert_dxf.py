"""
DXF to JSON Converter for Campus 3D Dashboard
Converts AutoCAD DXF file to Three.js compatible JSON format
"""

import ezdxf
import json
import numpy as np
from pathlib import Path

def read_dxf_file(dxf_path):
    """Read DXF file and extract entities"""
    print(f"📐 Reading DXF file: {dxf_path}")
    
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        entities = {
            'points': [],
            'lines': [],
            'polylines': [],
            'solids': [],
            'text': [],
        }
        
        # Extract entities
        for entity in msp:
            entity_type = entity.dxftype()
            
            if entity_type == 'POINT':
                entities['points'].append({
                    'x': float(entity.dxf.location.x),
                    'y': float(entity.dxf.location.y),
                    'z': float(entity.dxf.location.z) if hasattr(entity.dxf.location, 'z') else 0,
                })
            
            elif entity_type == 'LINE':
                entities['lines'].append({
                    'start': {
                        'x': float(entity.dxf.start.x),
                        'y': float(entity.dxf.start.y),
                        'z': float(entity.dxf.start.z) if hasattr(entity.dxf.start, 'z') else 0,
                    },
                    'end': {
                        'x': float(entity.dxf.end.x),
                        'y': float(entity.dxf.end.y),
                        'z': float(entity.dxf.end.z) if hasattr(entity.dxf.end, 'z') else 0,
                    },
                })
            
            elif entity_type == 'POLYLINE' or entity_type == 'LWPOLYLINE':
                points = []
                for vertex in entity:
                    points.append({
                        'x': float(vertex.dxf.location.x),
                        'y': float(vertex.dxf.location.y),
                        'z': float(vertex.dxf.location.z) if hasattr(vertex.dxf.location, 'z') else 0,
                    })
                entities['polylines'].append({'points': points})
            
            elif entity_type == 'SOLID' or entity_type == '3DFACE':
                vertices = []
                for i in range(4):
                    try:
                        vertex = entity.points[i]
                        vertices.append({
                            'x': float(vertex[0]),
                            'y': float(vertex[1]),
                            'z': float(vertex[2]) if len(vertex) > 2 else 0,
                        })
                    except:
                        break
                if vertices:
                    entities['solids'].append({'vertices': vertices})
            
            elif entity_type == 'TEXT' or entity_type == 'MTEXT':
                entities['text'].append({
                    'text': entity.dxf.text if hasattr(entity.dxf, 'text') else '',
                    'position': {
                        'x': float(entity.dxf.insert.x) if hasattr(entity.dxf, 'insert') else 0,
                        'y': float(entity.dxf.insert.y) if hasattr(entity.dxf, 'insert') else 0,
                        'z': float(entity.dxf.insert.z) if hasattr(entity.dxf, 'insert') else 0,
                    },
                })
        
        print(f"✅ Extracted entities:")
        print(f"   Points: {len(entities['points'])}")
        print(f"   Lines: {len(entities['lines'])}")
        print(f"   Polylines: {len(entities['polylines'])}")
        print(f"   Solids: {len(entities['solids'])}")
        print(f"   Text: {len(entities['text'])}")
        
        return entities
    
    except Exception as e:
        print(f"❌ Error reading DXF: {e}")
        return None

def generate_point_cloud(entities, density=0.5):
    """Generate point cloud from DXF entities"""
    print("\n🔮 Generating point cloud...")
    
    points = []
    colors = []
    
    # Color mapping by entity type
    type_colors = {
        'lines': [0.0, 1.0, 0.8],      # Cyan
        'polylines': [0.0, 0.8, 1.0],  # Blue
        'solids': [0.8, 0.0, 1.0],     # Purple
        'points': [1.0, 1.0, 0.0],     # Yellow
    }
    
    # Sample points from lines
    for line in entities['lines']:
        start = np.array([line['start']['x'], line['start']['y'], line['start']['z']])
        end = np.array([line['end']['x'], line['end']['y'], line['end']['z']])
        
        length = np.linalg.norm(end - start)
        num_points = int(length / density)
        
        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            point = start + t * (end - start)
            points.append(point.tolist())
            colors.append(type_colors['lines'])
    
    # Sample points from polylines
    for polyline in entities['polylines']:
        polyline_points = polyline['points']
        for i in range(len(polyline_points) - 1):
            start = np.array([polyline_points[i]['x'], polyline_points[i]['y'], polyline_points[i]['z']])
            end = np.array([polyline_points[i+1]['x'], polyline_points[i+1]['y'], polyline_points[i+1]['z']])
            
            length = np.linalg.norm(end - start)
            num_points = int(length / density)
            
            for j in range(num_points):
                t = j / max(num_points - 1, 1)
                point = start + t * (end - start)
                points.append(point.tolist())
                colors.append(type_colors['polylines'])
    
    # Sample points from solids (surfaces)
    for solid in entities['solids']:
        vertices = solid['vertices']
        if len(vertices) >= 3:
            # Create triangle fan
            v0 = np.array([vertices[0]['x'], vertices[0]['y'], vertices[0]['z']])
            for i in range(1, len(vertices) - 1):
                v1 = np.array([vertices[i]['x'], vertices[i]['y'], vertices[i]['z']])
                v2 = np.array([vertices[i+1]['x'], vertices[i+1]['y'], vertices[i+1]['z']])
                
                # Sample points in triangle
                for _ in range(int(10 / density)):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    if r1 + r2 > 1:
                        r1 = 1 - r1
                        r2 = 1 - r2
                    
                    point = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
                    points.append(point.tolist())
                    colors.append(type_colors['solids'])
    
    print(f"✅ Generated {len(points)} points")
    
    return {
        'positions': points,
        'colors': colors,
    }

def normalize_coordinates(point_cloud, scale=1.0):
    """Normalize and scale coordinates"""
    print("\n📏 Normalizing coordinates...")
    
    if not point_cloud['positions']:
        return point_cloud
    
    positions = np.array(point_cloud['positions'])
    
    # Center at origin
    center = positions.mean(axis=0)
    positions -= center
    
    # Scale
    max_extent = np.max(np.abs(positions))
    if max_extent > 0:
        positions *= (scale / max_extent)
    
    print(f"   Center: {center.tolist()}")
    print(f"   Scale factor: {scale / max_extent if max_extent > 0 else 1.0}")
    
    point_cloud['positions'] = positions.tolist()
    point_cloud['center'] = center.tolist()
    
    return point_cloud

def save_json(data, output_path):
    """Save to JSON file"""
    print(f"\n💾 Saving to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Saved successfully!")

def main():
    # Input DXF file path
    dxf_path = r"C:\Users\Khushi Tirkey\Downloads\topoexport-EB134F\topoexport_3D_modeling.dxf"
    
    # Output JSON path
    output_dir = Path(__file__).parent / 'public' / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'campus_model.json'
    
    print("=" * 60)
    print("DXF to Three.js JSON Converter")
    print("=" * 60)
    
    # Read DXF
    entities = read_dxf_file(dxf_path)
    
    if entities is None:
        print("\n❌ Failed to read DXF file. Make sure the path is correct.")
        print("   You can also download ezdxf: pip install ezdxf")
        return
    
    # Generate point cloud
    point_cloud = generate_point_cloud(entities, density=0.5)
    
    # Normalize
    point_cloud = normalize_coordinates(point_cloud, scale=50.0)
    
    # Add metadata
    model_data = {
        'metadata': {
            'source': str(dxf_path),
            'generated': str(np.datetime64('now')),
            'num_points': len(point_cloud['positions']),
        },
        'pointCloud': point_cloud,
        'entities': entities,
    }
    
    # Save
    save_json(model_data, output_path)
    
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print(f"   Output: {output_path}")
    print("   Next: Run 'npm install' and 'npm run dev' in the dashboard folder")
    print("=" * 60)

if __name__ == "__main__":
    main()
