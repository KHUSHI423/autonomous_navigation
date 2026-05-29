# X to Y Path Simulation

3D campus navigation visualization showing movement from point X to point Y.

## Features

- 3D campus map with buildings and landmarks
- Curved path visualization (matching the reference image)
- Animated simulation with progress tracking
- Interactive 3D view (rotate, zoom, pan)
- Start (X) and Destination (Y) markers

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python x_to_y_simulation.py
```

### Modes

1. **Animated Simulation** - Real-time animation showing movement along the path
2. **Static 3D View** - Static visualization of the complete path

## Controls (Animated Mode)

- **Rotate**: Click and drag to rotate view
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Close Window**: Stop simulation

## Configuration

Edit the `CONFIG` dictionary in the script to customize:

- `FPS`: Animation frame rate (default: 30)
- `DURATION`: Simulation duration in seconds (default: 10)
- `PATH_SMOOTHNESS`: Number of interpolation points (default: 100)
- `ELEVATION` / `AZIMUTH`: Camera view angle

## Output

The simulation displays:
- 3D campus layout with buildings
- Blue curved path from X to Y
- Yellow markers at start and end points
- Real-time progress percentage
- Elapsed time counter
