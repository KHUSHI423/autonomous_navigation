/*
=============================================================================
ESP32 #1 - UDP RECEIVER TEST
=============================================================================
Tests if ESP32 #1 receives UDP on port 9002
Upload to ESP32 #1 and open Serial Monitor (115200 baud)
=============================================================================
*/

#include <WiFi.h>
#include <WiFiUDP.h>

const char* ssid = "SANJEEVI";
const char* password = "YOUR_WIFI_PASSWORD";

WiFiUDP udp;
const int TEST_PORT = 9002;

void setup() {
  Serial.begin(115200);
  Serial.println("\n📡 ESP32 #1 - UDP Receiver Test");
  Serial.println("================================");
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ Connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
  
  // Listen on port 9002
  if (udp.begin(TEST_PORT)) {
    Serial.println("✅ Listening on port " + String(TEST_PORT));
    Serial.println("\nWaiting for UDP packets...\n");
  } else {
    Serial.println("❌ Failed to open port " + String(TEST_PORT));
  }
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    char buffer[256];
    int len = udp.read(buffer, sizeof(buffer));
    buffer[len] = '\0';
    
    Serial.print("📨 Received from ");
    Serial.print(udp.remoteIP());
    Serial.print(":");
    Serial.print(udp.remotePort());
    Serial.print(" | Size: ");
    Serial.print(packetSize);
    Serial.print(" bytes | Data: ");
    Serial.println(buffer);
  }
}
