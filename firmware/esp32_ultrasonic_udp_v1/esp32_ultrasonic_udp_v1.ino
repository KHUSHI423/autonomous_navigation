/*
=============================================================================
ROBOT CAR V2 - ULTRASONIC SENSOR (ESP32 #2) - UDP V1
=============================================================================
Reads HC-SR04 ultrasonic sensor and sends distance data via UDP to 
motor controller ESP32 (ESP32 #1)

Protocol: Sends "ULTRA:25.50" format to motor ESP32 on port 9002
Motor controller expects: UDP on port 9002 with "ULTRA:distance" format

Wiring:
- HC-SR04 Ultrasonic:
  - VCC -> 5V
  - GND -> GND
  - TRIG -> GPIO 5
  - ECHO -> GPIO 18

Upload to ULTRASONIC ESP32 (ESP32 #2)!
=============================================================================
*/

#include <WiFi.h>
#include <WiFiUDP.h>

/************ YOUR WiFi NETWORK ************/
const char* ssid = "SANJEEVI";        // SAME as motor ESP32
const char* password = "YOUR_WIFI_PASSWORD";   // SAME as motor ESP32
/*********************************************/

// IP of motor controller ESP32 (MUST match ESP32 #1 IP from serial monitor)
const char* MOTOR_ESP_IP = "172.21.121.207";  // ← CHANGE THIS to ESP32 #1 IP
const int ULTRASONIC_PORT = 9002;  // UDP port (matches motor controller)

#define TRIG_PIN 5
#define ECHO_PIN 18

WiFiUDP udp;

// WiFi reconnection
unsigned long lastWiFiCheck = 0;
unsigned long lastUltrasonicSend = 0;
const int WIFI_CHECK_INTERVAL = 5000;     // Check WiFi every 5 seconds
const int ULTRASONIC_INTERVAL = 100;      // Send ultrasonic data every 100ms (10 Hz)

// Statistics
unsigned long packetsSent = 0;
unsigned long lastStatsPrint = 0;

void setup() {    
  Serial.begin(115200);
  Serial.println("\n\n📡 Robot Car V2 - Ultrasonic Sensor ESP32");
  Serial.println("============================================");
  Serial.println("Version: UDP V1");
  Serial.println();

  // Setup ultrasonic pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);  // Ensure trig is low initially

  // Connect to WiFi
  connectToWiFi();

  // Initialize UDP socket
  if (udp.begin(0)) {
    Serial.println("✅ UDP socket initialized");
  } else {
    Serial.println("❌ UDP socket initialization failed!");
  }

  Serial.println("\n📊 Configuration:");
  Serial.print("   Motor ESP IP: ");
  Serial.println(MOTOR_ESP_IP);
  Serial.print("   UDP Port: ");
  Serial.println(ULTRASONIC_PORT);
  Serial.print("   Update Rate: 10 Hz (every 100ms)");
  Serial.println();
  Serial.println("🚀 Starting ultrasonic readings...\n");
}

void connectToWiFi() {
  Serial.print("📶 Connecting to WiFi: ");
  Serial.print(ssid);
  
  WiFi.begin(ssid, password);

  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 40) {
    delay(500);
    Serial.print(".");
    timeout++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi Connected!");
    Serial.print("   IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("   Signal Strength: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    Serial.print("   Sending to: ");
    Serial.print(MOTOR_ESP_IP);
    Serial.print(":");
    Serial.println(ULTRASONIC_PORT);
  } else {
    Serial.println("\n❌ WiFi Connection Failed!");
    Serial.println("   Please check:");
    Serial.println("   1. WiFi credentials are correct");
    Serial.println("   2. Phone hotspot is active");
    Serial.println("   3. ESP32 is within range");
  }
}

float getDistance() {
  // Clear trig pin
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  // Send 10us pulse
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Read echo pulse (timeout after 30ms = ~5m range)
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) {
    return 999.0;  // No echo received (out of range)
  }

  // Calculate distance in cm (speed of sound = 340 m/s = 0.034 cm/us)
  float distance = duration * 0.034 / 2.0;
  
  // Filter out invalid readings
  if (distance < 2.0 || distance > 400.0) {
    return 999.0;
  }
  
  return distance;
}

void checkWiFi() {
  unsigned long now = millis();
  if (now - lastWiFiCheck > WIFI_CHECK_INTERVAL) {
    lastWiFiCheck = now;

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("\n⚠️  WiFi disconnected! Reconnecting...");
      connectToWiFi();
    }
  }
}

void sendUltrasonicData(float distance) {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  // Format: "ULTRA:25.50" (matches motor controller parser)
  String message = "ULTRA:" + String(distance, 2);

  // Send UDP packet
  udp.beginPacket(MOTOR_ESP_IP, ULTRASONIC_PORT);
  udp.print(message);
  int result = udp.endPacket();

  if (result) {
    packetsSent++;
    Serial.print("✅ Sent: ");
    Serial.print(message);
    Serial.print(" cm | Packets: ");
    Serial.println(packetsSent);
  } else {
    Serial.print("❌ Send failed! Distance: ");
    Serial.print(distance);
    Serial.print(" cm | WiFi Status: ");
    Serial.println(WiFi.status());
    
    // Try to reconnect if send fails
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("🔄 WiFi lost, attempting reconnection...");
      connectToWiFi();
    }
  }
}

void printStatistics() {
  unsigned long now = millis();
  if (now - lastStatsPrint > 10000) {  // Every 10 seconds
    lastStatsPrint = now;
    Serial.println("\n📊 === STATISTICS ===");
    Serial.print("   Total packets sent: ");
    Serial.println(packetsSent);
    Serial.print("   WiFi RSSI: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    Serial.print("   WiFi Status: ");
    Serial.println(WiFi.status() == WL_CONNECTED ? "Connected" : "Disconnected");
    Serial.println("   ====================\n");
  }
}

void loop() {
  // Check WiFi connection periodically
  checkWiFi();

  // Send ultrasonic data at fixed interval
  unsigned long now = millis();
  if (now - lastUltrasonicSend >= ULTRASONIC_INTERVAL) {
    lastUltrasonicSend = now;
    
    // Read ultrasonic sensor
    float distance = getDistance();
    
    // Print to serial for debugging
    if (distance < 999.0) {
      Serial.print("📏 Distance: ");
      Serial.print(distance);
      Serial.println(" cm");
    } else {
      Serial.println("📏 Distance: OUT OF RANGE");
    }
    
    // Send to motor controller via UDP
    sendUltrasonicData(distance);
  }

  // Print statistics periodically
  printStatistics();

  // Small delay to prevent watchdog issues
  delay(1);
}
