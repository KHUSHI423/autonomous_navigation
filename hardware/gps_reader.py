"""
=============================================================================
EDGE DRIVE 3D - GPS READER MODULE
=============================================================================
GPS hardware interface for reading position data from USB/Serial GPS modules

Supports:
- USB GPS Dongles (plug-and-play)
- UART GPS modules (NEO-6M, NEO-M8N, ZED-F9P)
- NMEA 0183 protocol parsing
- Simulated GPS for testing without hardware

Author: EdgeDrive3D Team
=============================================================================
"""

import serial
import serial.tools.list_ports
import pynmea2
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict
from datetime import datetime
import math


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GPSReading:
    """GPS position and status data"""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0  # m/s
    heading: float = 0.0  # degrees (0-360, North=0)
    accuracy: float = 5.0  # meters (estimated)
    satellites: int = 0
    fix_quality: int = 0  # 0=invalid, 1=GPS, 2=DGPS
    timestamp: float = field(default_factory=time.time)
    is_valid: bool = False
    
    @property
    def lat_lon(self) -> tuple:
        return (self.latitude, self.longitude)
    
    def to_dict(self) -> Dict:
        return {
            'latitude': round(self.latitude, 8),
            'longitude': round(self.longitude, 8),
            'altitude': round(self.altitude, 2),
            'speed_ms': round(self.speed, 2),
            'speed_kmh': round(self.speed * 3.6, 2),
            'heading': round(self.heading, 2),
            'accuracy_m': round(self.accuracy, 2),
            'satellites': self.satellites,
            'fix_quality': self.fix_quality,
            'is_valid': self.is_valid,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        if not self.is_valid:
            return "GPS: No fix"
        return (f"GPS: {self.latitude:.6f}, {self.longitude:.6f} | "
                f"Alt: {self.altitude:.1f}m | Spd: {self.speed:.1f}m/s | "
                f"Hdg: {self.heading:.1f}° | Sats: {self.satellites}")


# ============================================================================
# GPS RECEIVER CLASS
# ============================================================================

class GPSReceiver:
    """
    GPS receiver for reading NMEA data from serial/USB GPS modules
    
    Usage:
        gps = GPSReceiver(port='COM3', baudrate=9600)
        gps.start()
        
        while True:
            reading = gps.get_position()
            if reading and reading.is_valid:
                print(reading)
            time.sleep(0.1)
    """
    
    def __init__(
        self,
        port: str = 'auto',
        baudrate: int = 9600,
        timeout: float = 2.0,
        simulate: bool = False
    ):
        """
        Initialize GPS receiver
        
        Args:
            port: Serial port (e.g., 'COM3', '/dev/ttyUSB0') or 'auto'
            baudrate: GPS baud rate (typically 9600 or 115200)
            timeout: Read timeout in seconds
            simulate: If True, generate simulated GPS data for testing
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.simulate = simulate
        
        self.serial_conn: Optional[serial.Serial] = None
        self.latest_reading = GPSReading()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable[[GPSReading], None]] = []
        
        # Statistics
        self.total_sentences = 0
        self.valid_sentences = 0
        self.last_update_time = 0
        
        # Auto-detect port if requested
        if port == 'auto':
            self.port = self._auto_detect_port()
    
    def _auto_detect_port(self) -> str:
        """Auto-detect available serial ports"""
        ports = serial.tools.list_ports.comports()
        
        # Look for common GPS device patterns
        gps_patterns = ['usbserial', 'ttyUSB', 'ttyACM', 'FTDI', 'Prolific', 'CH340']
        
        for port in ports:
            # Check device description
            for pattern in gps_patterns:
                if pattern.lower() in port.description.lower():
                    print(f"  ✓ Auto-detected GPS: {port.device} ({port.description})")
                    return port.device
            
            # If no pattern match, return first available non-bluetooth port
            if 'bluetooth' not in port.description.lower():
                print(f"  ✓ Using serial port: {port.device} ({port.description})")
                return port.device
        
        print("  ⚠ No GPS device found, using simulated mode")
        return 'SIMULATED'
    
    def connect(self) -> bool:
        """Connect to GPS device"""
        if self.simulate or self.port == 'SIMULATED':
            print("  ✓ GPS: Simulated mode enabled")
            self.simulate = True
            self.latest_reading.is_valid = True
            return True
        
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"  ✓ GPS: Connected to {self.port} @ {self.baudrate}")
            return True
        except serial.SerialException as e:
            print(f"  ✗ GPS: Connection failed - {e}")
            return False
    
    def disconnect(self):
        """Disconnect from GPS device"""
        self.stop()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("  ✓ GPS: Disconnected")
    
    def start(self):
        """Start background reading thread"""
        if self.running:
            return
        
        if not self.serial_conn and not self.simulate:
            if not self.connect():
                return
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print("  ✓ GPS: Reading started")
    
    def stop(self):
        """Stop background reading thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
    
    def _read_loop(self):
        """Background thread to continuously read GPS data"""
        while self.running:
            try:
                if self.simulate:
                    reading = self._generate_simulated_data()
                else:
                    reading = self._read_nmea_sentence()
                
                if reading:
                    self.latest_reading = reading
                    self.last_update_time = time.time()
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        try:
                            callback(reading)
                        except Exception as e:
                            print(f"  ⚠ GPS callback error: {e}")
                
                time.sleep(0.05)  # 20Hz max
                
            except Exception as e:
                if self.running:
                    print(f"  ⚠ GPS read error: {e}")
                time.sleep(0.5)
    
    def _read_nmea_sentence(self) -> Optional[GPSReading]:
        """Read and parse NMEA sentence from serial"""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None
        
        try:
            line = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
            
            if not line.startswith('$'):
                return None
            
            self.total_sentences += 1
            
            # Parse GGA sentence (position, fix quality)
            if 'GGA' in line:
                try:
                    msg = pynmea2.parse(line)
                    self.valid_sentences += 1
                    
                    reading = GPSReading(
                        latitude=msg.latitude,
                        longitude=msg.longitude,
                        altitude=float(msg.altitude) if msg.altitude else 0.0,
                        fix_quality=int(msg.gps_qual) if msg.gps_qual else 0,
                        satellites=int(msg.num_sats) if msg.num_sats else 0,
                        timestamp=time.time(),
                        is_valid=msg.gps_qual and int(msg.gps_qual) > 0
                    )
                    
                    # Estimate accuracy based on satellites
                    if reading.satellites > 8:
                        reading.accuracy = 2.0
                    elif reading.satellites > 5:
                        reading.accuracy = 5.0
                    else:
                        reading.accuracy = 10.0
                    
                    return reading
                    
                except pynmea2.ParseError:
                    pass
            
            # Parse RMC sentence (speed, heading)
            elif 'RMC' in line:
                try:
                    msg = pynmea2.parse(line)
                    
                    # Update speed and heading
                    if hasattr(msg, 'speed') and msg.speed:
                        # Convert knots to m/s
                        self.latest_reading.speed = float(msg.speed) * 0.514444
                    
                    if hasattr(msg, 'true_course') and msg.true_course:
                        self.latest_reading.heading = float(msg.true_course)
                    
                    self.latest_reading.timestamp = time.time()
                    
                except pynmea2.ParseError:
                    pass
            
            return None
            
        except serial.SerialException:
            return None
        except UnicodeDecodeError:
            return None
    
    def _generate_simulated_data(self) -> GPSReading:
        """Generate simulated GPS data for testing"""
        import math
        
        base_lat = 12.9716  # Bangalore coordinates (example)
        base_lon = 77.5946
        
        # Simulate movement in a circle
        t = time.time()
        radius = 0.0005  # ~50 meters
        
        lat = base_lat + radius * math.sin(t * 0.5)
        lon = base_lon + radius * math.cos(t * 0.5)
        alt = 920.0 + 5.0 * math.sin(t * 0.3)
        
        # Simulate speed and heading
        speed = 5.0 + 2.0 * math.sin(t * 0.2)  # 5 m/s average
        heading = (t * 10) % 360
        
        return GPSReading(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            speed=speed,
            heading=heading,
            accuracy=3.0,
            satellites=12,
            fix_quality=1,
            timestamp=time.time(),
            is_valid=True
        )
    
    def get_position(self) -> Optional[GPSReading]:
        """Get latest GPS position"""
        # Check if data is stale (older than 5 seconds)
        if time.time() - self.last_update_time > 5.0:
            self.latest_reading.is_valid = False
        
        return self.latest_reading if self.latest_reading.is_valid else None
    
    def wait_for_fix(self, timeout: float = 30.0) -> Optional[GPSReading]:
        """Wait for valid GPS fix"""
        start = time.time()
        
        while time.time() - start < timeout:
            reading = self.get_position()
            if reading and reading.is_valid:
                return reading
            time.sleep(0.5)
        
        return None
    
    def on_update(self, callback: Callable[[GPSReading], None]):
        """Register callback for GPS updates"""
        self.callbacks.append(callback)
    
    def get_statistics(self) -> Dict:
        """Get GPS receiver statistics"""
        return {
            'total_sentences': self.total_sentences,
            'valid_sentences': self.valid_sentences,
            'parse_rate': self.valid_sentences / max(1, self.total_sentences) * 100,
            'last_update': self.last_update_time,
            'is_connected': self.serial_conn is not None if not self.simulate else True,
            'simulate_mode': self.simulate
        }


# ============================================================================
# GPS LOGGER (For Recording Trajectories)
# ============================================================================

class GPSLogger:
    """Log GPS data and perception results to file"""
    
    def __init__(self, filepath: str = "gps_trajectory.jsonl"):
        self.filepath = filepath
        self.file = None
        self.total_logged = 0
    
    def start(self):
        """Start logging session"""
        self.file = open(self.filepath, 'w')
        print(f"  ✓ GPS Logger: Recording to {self.filepath}")
    
    def stop(self):
        """Stop logging session"""
        if self.file:
            self.file.close()
            self.file = None
        print(f"  ✓ GPS Logger: Stopped ({self.total_logged} records)")
    
    def log(self, gps: GPSReading, extra_data: Dict = None):
        """Log GPS reading with optional extra data"""
        if not self.file:
            return
        
        import json
        
        record = {
            'timestamp': time.time(),
            'gps': gps.to_dict()
        }
        
        if extra_data:
            record.update(extra_data)
        
        self.file.write(json.dumps(record) + '\n')
        self.file.flush()
        self.total_logged += 1
    
    def replay(self, callback: Callable[[Dict], None]):
        """Replay logged data"""
        import json
        
        with open(self.filepath, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                callback(data)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_gps_devices() -> List[Dict]:
    """List all available serial devices that might be GPS"""
    ports = serial.tools.list_ports.comports()
    devices = []
    
    for port in ports:
        devices.append({
            'device': port.device,
            'description': port.description,
            'hardware_id': port.hwid,
            'is_gps': any(p in port.description.lower() for p in ['usbserial', 'ttyusb', 'ttyacm', 'ftdi'])
        })
    
    return devices


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS coordinates (in meters)
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing from point 1 to point 2 (in degrees)
    
    Returns:
        Bearing in degrees (0-360, North=0)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    
    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360


# ============================================================================
# MAIN (Test GPS)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  GPS Reader Test")
    print("=" * 60)
    
    # List available devices
    print("\nAvailable serial devices:")
    devices = list_gps_devices()
    for dev in devices:
        marker = "📍" if dev['is_gps'] else "  "
        print(f"  {marker} {dev['device']}: {dev['description']}")
    
    # Initialize GPS
    print("\nInitializing GPS receiver...")
    gps = GPSReceiver(port='auto', simulate=True)
    
    # Start reading
    gps.start()
    
    # Wait for fix
    print("\nWaiting for GPS fix (10 seconds)...")
    reading = gps.wait_for_fix(timeout=10.0)
    
    if reading:
        print(f"\n✓ GPS Fix acquired!")
        print(f"  Position: {reading.latitude:.6f}, {reading.longitude:.6f}")
        print(f"  Altitude: {reading.altitude:.1f}m")
        print(f"  Speed: {reading.speed:.2f} m/s ({reading.speed * 3.6:.2f} km/h)")
        print(f"  Heading: {reading.heading:.1f}°")
        print(f"  Satellites: {reading.satellites}")
        print(f"  Accuracy: ±{reading.accuracy:.1f}m")
    else:
        print("\n⚠ No GPS fix acquired")
    
    # Monitor for a few seconds
    print("\nMonitoring GPS updates (press Ctrl+C to stop)...")
    try:
        while True:
            reading = gps.get_position()
            if reading and reading.is_valid:
                print(f"\r{reading}", end="")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        gps.disconnect()
        print("\n\nGPS test complete")
