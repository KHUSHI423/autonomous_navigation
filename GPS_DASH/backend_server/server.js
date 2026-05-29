const http = require('http');
const express = require('express');
const WebSocket = require('ws');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3001;

// Create HTTP server
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocket.Server({ server, port: PORT });

// Store connected clients
const clients = new Set();

// Store latest sensor data
let latestSensorData = null;

// WebSocket connection handler
wss.on('connection', (ws) => {
  console.log('✅ Client connected');
  clients.add(ws);

  // Send latest data to newly connected client
  if (latestSensorData) {
    ws.send(JSON.stringify(latestSensorData));
  }

  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      latestSensorData = data;
      
      // Log received data
      console.log('\n📡 Received Sensor Data:');
      if (data.data) {
        if (data.data.imu) {
          console.log(`   IMU - Roll: ${data.data.imu.roll.toFixed(2)}°, Pitch: ${data.data.imu.pitch.toFixed(2)}°, Yaw: ${data.data.imu.yaw.toFixed(2)}°`);
        }
        if (data.data.gps) {
          console.log(`   GPS - Lat: ${data.data.gps.lat.toFixed(6)}, Lon: ${data.data.gps.lon.toFixed(6)}, Satellites: ${data.data.gps.satellites}`);
        }
        if (data.data.encoder) {
          console.log(`   Encoder - Angle: ${data.data.encoder.angle}°`);
        }
      }

      // Broadcast to all connected clients (optional - for dashboard frontend)
      clients.forEach((client) => {
        if (client !== ws && client.readyState === WebSocket.OPEN) {
          client.send(message);
        }
      });
    } catch (error) {
      console.error('❌ Error parsing message:', error.message);
    }
  });

  ws.on('close', () => {
    console.log('❌ Client disconnected');
    clients.delete(ws);
  });

  ws.on('error', (error) => {
    console.error('❌ WebSocket error:', error.message);
    clients.delete(ws);
  });
});

// REST API endpoints
app.get('/api/status', (req, res) => {
  res.json({
    status: 'running',
    connectedClients: clients.size,
    hasData: latestSensorData !== null,
    lastUpdate: latestSensorData?.timestamp || null
  });
});

app.get('/api/sensors', (req, res) => {
  if (latestSensorData) {
    res.json(latestSensorData);
  } else {
    res.json({ message: 'No sensor data received yet' });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'OK' });
});

// Start server
server.listen(PORT, '0.0.0.0', () => {
  console.log('\n🚀 GPS Dashboard Backend Server');
  console.log('================================');
  console.log(`📡 WebSocket: ws://0.0.0.0:${PORT}`);
  console.log(`📊 REST API:  http://0.0.0.0:${PORT}/api`);
  console.log(`💚 Health:    http://0.0.0.0:${PORT}/health`);
  console.log('\n⏳ Waiting for ESP32 connection...');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\n👋 Shutting down server...');
  wss.clients.forEach((client) => {
    client.close();
  });
  server.close(() => {
    console.log('✅ Server closed');
    process.exit(0);
  });
});
