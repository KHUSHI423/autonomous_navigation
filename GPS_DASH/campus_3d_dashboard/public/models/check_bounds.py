import json

with open('campus_model.json', 'r') as f:
    d = json.load(f)

positions = d['pointCloud']['positions']
xs = [p[0] for p in positions]
ys = [p[1] for p in positions]
zs = [p[2] for p in positions]

print(f"Total points: {len(positions)}")
print(f"X range: {min(xs):.1f} to {max(xs):.1f} (width: {max(xs)-min(xs):.1f})")
print(f"Y range: {min(ys):.1f} to {max(ys):.1f} (height: {max(ys)-min(ys):.1f})")
print(f"Z range: {min(zs):.3f} to {max(zs):.3f} (elevation: {max(zs)-min(zs):.3f})")
