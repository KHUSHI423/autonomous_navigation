"""
GPS Simulator - Simulates GPS data for testing without hardware
Can output NMEA sentences to virtual serial port or send directly to API
"""

import serial
import serial.tools.list_ports
import time
import random
import math
import json
import requests
from datetime import datetime

# Configuration
SIMULATION_CONFIG = {
    'start_lat': 28.6139,      # Starting latitude (New Delhi)
    'start_lon': 77.2090,      # Starting longitude
    'speed_kmh': 40,           # Simulated speed in km/h
    'update_interval': 1.0,    # Seconds between updates
    'output_mode': 'api',      # 'serial' or 'api'
    'serial_port': 'COM4',     # Virtual serial port for loopback testing
    'baud_rate': 9600,
    'api_url': 'http://localhost:5000/gps/update'
}


class GPSSimulator:
    def __init__(self, config):
        self.config = config
        self.lat = config['start_lat']
        self.lon = config['start_lon']
        self.alt = 216.0
        self.speed = 0.0
        self.heading = random.uniform(0, 360)
        self.satellites = random.randint(6, 12)
        self.running = False
        self.serial_conn = None
        
    def start(self):
        """Start the simulation"""
        print("=" * 50)
        print("GPS Simulator Started")
        print("=" * 50)
        print(f"Mode: {self.config['output_mode']}")
        print(f"Start Position: {self.lat:.6f}, {self.lon:.6f}")
        print(f"Speed: {self.config['speed_kmh']} km/h")
        print("=" * 50)
        
        self.running = True
        
        if self.config['output_mode'] == 'serial':
            self._connect_serial()
        
        try:
            self._simulate()
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the simulation"""
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()
            print("Serial connection closed")
    
    def _connect_serial(self):
        """Connect to virtual serial port"""
        try:
            self.serial_conn = serial.Serial(
                self.config['serial_port'],
                self.config['baud_rate'],
                timeout=1
            )
            time.sleep(2)
            print(f"Connected to {self.config['serial_port']}")
        except Exception as e:
            print(f"Failed to connect to serial port: {e}")
            print("Make sure you have created a virtual serial pair:")
            print("  - Windows: Use com2com or similar tool")
            print("  - Linux: Use socat or tty0tty")
            self.config['output_mode'] = 'api'
    
    def _simulate(self):
        """Main simulation loop"""
        last_update = time.time()
        
        while self.running:
            current_time = time.time()
            
            if current_time - last_update >= self.config['update_interval']:
                self._update_position()
                self._send_data()
                last_update = current_time
            
            time.sleep(0.1)
    
    def _update_position(self):
        """Update simulated GPS position"""
        # Simulate realistic movement
        self.heading += random.uniform(-5, 5)
        self.heading = self.heading % 360
        
        # Convert speed to degrees per second
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        speed_deg_per_sec = self.config['speed_kmh'] / 111.0 / 3600.0
        
        # Update position based on heading
        self.lat += speed_deg_per_sec * math.cos(math.radians(self.heading))
        self.lon += speed_deg_per_sec * math.sin(math.radians(self.heading))
        
        # Add some variation
        self.alt = 216.0 + random.uniform(-5, 5)
        self.speed = self.config['speed_kmh'] + random.uniform(-5, 5)
        self.satellites = random.randint(6, 12)
    
    def _send_data(self):
        """Send GPS data"""
        if self.config['output_mode'] == 'serial':
            self._send_nmea()
        else:
            self._send_api()
    
    def _send_nmea(self):
        """Send NMEA sentences via serial"""
        # Create GPGGA sentence
        gga = self._create_gpgga()
        # Create GPRMC sentence
        rmc = self._create_gprmc()
        
        if self.serial_conn:
            self.serial_conn.write(f"{gga}\r\n".encode())
            self.serial_conn.write(f"{rmc}\r\n".encode())
            print(f"Sent: {gga}")
    
    def _send_api(self):
        """Send data directly to API"""
        data = {
            'latitude': self.lat,
            'longitude': self.lon,
            'altitude': round(self.alt, 1),
            'speed': round(self.speed, 1),
            'heading': round(self.heading, 1),
            'satellites': self.satellites,
            'accuracy': round(random.uniform(1, 5), 1)
        }
        
        try:
            response = requests.post(
                self.config['api_url'],
                json=data,
                timeout=2
            )
            if response.status_code == 200:
                print(f"GPS: {self.lat:.6f}, {self.lon:.6f} | Speed: {self.speed:.1f} km/h | Heading: {self.heading:.1f}°")
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
    
    def _create_gpgga(self):
        """Create GPGGA NMEA sentence"""
        now = datetime.utcnow()
        time_str = now.strftime('%H%M%S.%f')[:9]
        
        # Format latitude: DDMM.MMMM
        lat_deg = int(abs(self.lat))
        lat_min = (abs(self.lat) - lat_deg) * 60
        lat_str = f"{lat_deg:02d}{lat_min:07.4f}"
        lat_dir = 'N' if self.lat >= 0 else 'S'
        
        # Format longitude: DDDMM.MMMM
        lon_deg = int(abs(self.lon))
        lon_min = (abs(self.lon) - lon_deg) * 60
        lon_str = f"{lon_deg:03d}{lon_min:07.4f}"
        lon_dir = 'E' if self.lon >= 0 else 'W'
        
        # Create sentence (without checksum for simplicity)
        gga = f"$GPGGA,{time_str},{lat_str},{lat_dir},{lon_str},{lon_dir},1,{self.satellites:02d},1.0,{self.alt:.1f},M,0.0,M,,"
        
        # Add checksum
        checksum = self._calculate_checksum(gga[1:])
        return f"{gga}*{checksum:02X}"
    
    def _create_gprmc(self):
        """Create GPRMC NMEA sentence"""
        now = datetime.utcnow()
        time_str = now.strftime('%H%M%S.%f')[:9]
        date_str = now.strftime('%d%m%y')
        
        # Format latitude
        lat_deg = int(abs(self.lat))
        lat_min = (abs(self.lat) - lat_deg) * 60
        lat_str = f"{lat_deg:02d}{lat_min:07.4f}"
        lat_dir = 'N' if self.lat >= 0 else 'S'
        
        # Format longitude
        lon_deg = int(abs(self.lon))
        lon_min = (abs(self.lon) - lon_deg) * 60
        lon_str = f"{lon_deg:03d}{lon_min:07.4f}"
        lon_dir = 'E' if self.lon >= 0 else 'W'
        
        # Speed in knots
        speed_knots = self.speed / 1.852
        
        # Create sentence
        rmc = f"$GPRMC,{time_str},A,{lat_str},{lat_dir},{lon_str},{lon_dir},{speed_knots:.1f},{self.heading:.1f},{date_str},,"
        
        # Add checksum
        checksum = self._calculate_checksum(rmc[1:])
        return f"{rmc}*{checksum:02X}"
    
    def _calculate_checksum(self, sentence):
        """Calculate NMEA checksum"""
        checksum = 0
        for char in sentence:
            checksum ^= ord(char)
        return checksum


def list_virtual_ports():
    """List available serial ports"""
    ports = serial.tools.list_ports.comports()
    print("\nAvailable Serial Ports:")
    print("-" * 40)
    for port in ports:
        print(f"  {port.device} - {port.description}")
    print("-" * 40)


def main():
    import sys
    
    print("\n" + "=" * 50)
    print("GPS SIMULATOR")
    print("=" * 50)
    print("\nSelect output mode:")
    print("  1. API Mode (send to Flask server)")
    print("  2. Serial Mode (send to virtual COM port)")
    print("  3. List available ports")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        config = SIMULATION_CONFIG.copy()
        config['output_mode'] = 'api'
        
        # Ask for custom speed
        speed = input(f"Enter speed in km/h (default: {config['speed_kmh']}): ").strip()
        if speed:
            config['speed_kmh'] = float(speed)
        
        print("\nStarting API mode simulation...")
        print("Make sure Flask server is running on http://localhost:5000")
        print("Press Ctrl+C to stop\n")
        time.sleep(2)
        
        simulator = GPSSimulator(config)
        simulator.start()
        
    elif choice == '2':
        list_virtual_ports()
        port = input("Enter COM port (e.g., COM4): ").strip()
        
        if port:
            config = SIMULATION_CONFIG.copy()
            config['output_mode'] = 'serial'
            config['serial_port'] = port
            
            print("\nStarting Serial mode simulation...")
            print("Press Ctrl+C to stop\n")
            time.sleep(2)
            
            simulator = GPSSimulator(config)
            simulator.start()
        else:
            print("No port specified")
            
    elif choice == '3':
        list_virtual_ports()
        
    elif choice == '4':
        print("Exiting...")
        sys.exit(0)
    
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
