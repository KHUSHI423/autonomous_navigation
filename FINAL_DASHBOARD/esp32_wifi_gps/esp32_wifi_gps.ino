/*
 * ESP32 WiFi GPS Data Sender
 * Sends GPS data to Flask server via WiFi
 * 
 * Connections:
 * - GPS Module TX -> ESP32 RX (GPIO 16)
 * - GPS Module RX -> ESP32 TX (GPIO 17)
 * - GPS VCC -> 3.3V or 5V (check your GPS module)
 * - GPS GND -> GND
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <HardwareSerial.h>

// ================== CONFIGURATION ==================
const char* WIFI_SSID = "Khushi";        // Change to your WiFi name
const char* WIFI_PASSWORD = "12345678"; // Change to your WiFi password

// Server IP: Your laptop's IP address (find using ipconfig in cmd)
const char* SERVER_IP = "10.213.125.201";  // CHANGE THIS to your laptop IP
const int SERVER_PORT = 5000;

// GPS Serial configuration (GPS module connected to these pins)
#define GPS_RX_PIN 16  // ESP32 pin connected to GPS TX
#define GPS_TX_PIN 17  // ESP32 pin connected to GPS RX
HardwareSerial gpsSerial(1);

// Update interval in milliseconds
const unsigned long UPDATE_INTERVAL = 1000;  // Send data every 1 second

// ================== GLOBAL VARIABLES ==================
float latitude = 0.0;
float longitude = 0.0;
float speed = 0.0;
float altitude = 0.0;
float heading = 0.0;
int satellites = 0;
float accuracy = 1.0;
bool gpsValid = false;

unsigned long lastUpdate = 0;
int wifiRetryCount = 0;
const int MAX_WIFI_RETRIES = 5;

// ================== FUNCTION DECLARATIONS ==================
void connectToWiFi();
void parseGPSData();
void sendGPSDataToServer();
String createJSONPayload();
void printGPSData();

// ================== SETUP ==================
void setup() {
  // Initialize Serial Monitor
  Serial.begin(115200);
  delay(1000);
  
  Serial.println();
  Serial.println("========================================");
  Serial.println("   ESP32 WiFi GPS Data Sender");
  Serial.println("========================================");
  
  // Initialize GPS Serial
  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("GPS Serial initialized");
  
  // Connect to WiFi
  connectToWiFi();
  
  Serial.println("========================================");
  Serial.println("Setup complete!");
  Serial.println("========================================");
}

// ================== MAIN LOOP ==================
void loop() {
  // Parse GPS data from GPS module
  parseGPSData();
  
  // Send data to server at regular intervals
  if (millis() - lastUpdate >= UPDATE_INTERVAL) {
    lastUpdate = millis();
    
    // Only send if we have WiFi connection
    if (WiFi.status() == WL_CONNECTED) {
      sendGPSDataToServer();
    } else {
      Serial.println("WiFi disconnected! Attempting to reconnect...");
      connectToWiFi();
    }
  }
  
  // Small delay to prevent watchdog reset
  delay(10);
}

// ================== FUNCTION DEFINITIONS ==================

/**
 * Connect ESP32 to WiFi network
 */
void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int wifiTimeout = 0;
  const int WIFI_TIMEOUT_MS = 10000;  // 10 seconds timeout
  
  while (WiFi.status() != WL_CONNECTED && wifiTimeout < WIFI_TIMEOUT_MS) {
    delay(500);
    wifiTimeout += 500;
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("✓ WiFi connected successfully!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal Strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    wifiRetryCount = 0;  // Reset retry counter
  } else {
    Serial.println();
    Serial.println("✗ WiFi connection failed!");
    wifiRetryCount++;
    
    if (wifiRetryCount >= MAX_WIFI_RETRIES) {
      Serial.println("Max WiFi retries reached. Restarting...");
      delay(2000);
      ESP.restart();
    }
  }
}

/**
 * Parse NMEA data from GPS module
 */
void parseGPSData() {
  while (gpsSerial.available() > 0) {
    char c = gpsSerial.read();
    
    // Simple NMEA parser for $GPGGA and $GPRMC sentences
    static char buffer[100];
    static int bufferIndex = 0;
    static bool parsing = false;
    
    if (c == '$') {
      bufferIndex = 0;
      parsing = true;
      buffer[bufferIndex++] = c;
    } else if (parsing) {
      if (c == '\n' || c == '\r' || bufferIndex >= 99) {
        buffer[bufferIndex] = '\0';
        parsing = false;
        
        // Parse the complete sentence
        parseNMEASentence(buffer);
      } else {
        buffer[bufferIndex++] = c;
      }
    }
  }
}

/**
 * Parse a complete NMEA sentence
 */
void parseNMEASentence(char* sentence) {
  // Check for GPGGA sentence
  if (strstr(sentence, "$GPGGA") != NULL || strstr(sentence, "$GNGGA") != NULL) {
    parseGPGGA(sentence);
  }
  // Check for GPRMC sentence
  else if (strstr(sentence, "$GPRMC") != NULL || strstr(sentence, "$GNRMC") != NULL) {
    parseGPRMC(sentence);
  }
}

/**
 * Parse $GPGGA sentence for position and altitude
 * Format: $GPGGA,time,lat,N/S,lon,E/Q,fix,sats,HDOP,alt,M,geoid,M,age,station*cs
 */
void parseGPGGA(char* sentence) {
  char* token;
  char* rest = sentence;
  int fieldIndex = 0;
  
  float latDecimal = 0.0;
  float lonDecimal = 0.0;
  
  while ((token = strtok_r(rest, ",", &rest)) != NULL) {
    switch (fieldIndex) {
      case 0:  // Sentence identifier ($GPGGA)
        break;
      case 1:  // UTC Time
        break;
      case 2:  // Latitude (DDMM.MMMM)
        if (strlen(token) > 0) {
          latDecimal = atof(token);
          // Convert from DDMM.MMMM to decimal degrees
          int degrees = (int)(latDecimal / 100);
          float minutes = latDecimal - (degrees * 100);
          latitude = degrees + (minutes / 60.0);
        }
        break;
      case 3:  // N/S indicator
        if (strcmp(token, "S") == 0) {
          latitude = -latitude;
        }
        break;
      case 4:  // Longitude (DDDMM.MMMM)
        if (strlen(token) > 0) {
          lonDecimal = atof(token);
          // Convert from DDDMM.MMMM to decimal degrees
          int degrees = (int)(lonDecimal / 100);
          float minutes = lonDecimal - (degrees * 100);
          longitude = degrees + (minutes / 60.0);
        }
        break;
      case 5:  // E/W indicator
        if (strcmp(token, "W") == 0) {
          longitude = -longitude;
        }
        break;
      case 6:  // Fix quality
        if (strlen(token) > 0) {
          int fixQuality = atoi(token);
          gpsValid = (fixQuality > 0);
        }
        break;
      case 7:  // Number of satellites
        if (strlen(token) > 0) {
          satellites = atoi(token);
        }
        break;
      case 8:  // HDOP (accuracy)
        if (strlen(token) > 0) {
          accuracy = atof(token);
        }
        break;
      case 9:  // Altitude (meters)
        if (strlen(token) > 0) {
          altitude = atof(token);
        }
        break;
    }
    fieldIndex++;
  }
}

/**
 * Parse $GPRMC sentence for speed and heading
 * Format: $GPRMC,time,status,lat,N/S,lon,E/W,speed,course,date,magvar,E/W*cs
 */
void parseGPRMC(char* sentence) {
  char* token;
  char* rest = sentence;
  int fieldIndex = 0;
  
  while ((token = strtok_r(rest, ",", &rest)) != NULL) {
    switch (fieldIndex) {
      case 0:  // Sentence identifier ($GPRMC)
        break;
      case 1:  // UTC Time
        break;
      case 2:  // Status (A=Active, V=Void)
        if (strcmp(token, "A") == 0) {
          gpsValid = true;
        } else {
          gpsValid = false;
        }
        break;
      case 7:  // Speed in knots
        if (strlen(token) > 0) {
          speed = atof(token) * 1.852;  // Convert knots to km/h
        }
        break;
      case 8:  // Course/Heading (degrees)
        if (strlen(token) > 0) {
          heading = atof(token);
        }
        break;
    }
    fieldIndex++;
  }
}

/**
 * Create JSON payload for server
 */
String createJSONPayload() {
  String json = "{";
  json += "\"latitude\":" + String(latitude, 6) + ",";
  json += "\"longitude\":" + String(longitude, 6) + ",";
  json += "\"speed\":" + String(speed, 2) + ",";
  json += "\"altitude\":" + String(altitude, 2) + ",";
  json += "\"satellites\":" + String(satellites) + ",";
  json += "\"heading\":" + String(heading, 2) + ",";
  json += "\"accuracy\":" + String(accuracy, 2) + ",";
  json += "\"status\":\"" + String(gpsValid ? "active" : "no_fix") + "\"";
  json += "}";
  
  return json;
}

/**
 * Send GPS data to Flask server via HTTP POST
 */
void sendGPSDataToServer() {
  // Check if WiFi is connected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("✗ Cannot send: WiFi not connected");
    return;
  }
  
  // Create HTTP client
  HTTPClient http;
  
  // Build server URL
  String url = "http://" + String(SERVER_IP) + ":" + String(SERVER_PORT) + "/gps/update";
  
  Serial.print("Sending GPS data to: ");
  Serial.println(url);
  
  // Begin HTTP connection
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  
  // Create JSON payload
  String jsonPayload = createJSONPayload();
  
  Serial.print("JSON Payload: ");
  Serial.println(jsonPayload);
  
  // Send HTTP POST request
  int httpResponseCode = http.POST(jsonPayload);
  
  // Handle response
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.print("✓ HTTP Response Code: ");
    Serial.println(httpResponseCode);
    Serial.print("Response: ");
    Serial.println(response);
    
    // Print GPS data summary
    printGPSData();
  } else {
    Serial.print("✗ HTTP Error: ");
    Serial.println(httpResponseCode);
    
    // Handle common errors
    if (httpResponseCode == -1) {
      Serial.println("Error: Connection failed (server not reachable)");
    } else if (httpResponseCode == -2) {
      Serial.println("Error: Send failed");
    } else if (httpResponseCode == -11) {
      Serial.println("Error: Invalid URL");
    }
  }
  
  // Close connection
  http.end();
}

/**
 * Print GPS data to Serial Monitor
 */
void printGPSData() {
  Serial.println("----------------------------------------");
  Serial.println("GPS Data Summary:");
  Serial.print("  Latitude:  ");
  Serial.println(latitude, 6);
  Serial.print("  Longitude: ");
  Serial.println(longitude, 6);
  Serial.print("  Speed:     ");
  Serial.print(speed);
  Serial.println(" km/h");
  Serial.print("  Altitude:  ");
  Serial.print(altitude);
  Serial.println(" m");
  Serial.print("  Heading:   ");
  Serial.print(heading);
  Serial.println("°");
  Serial.print("  Satellites: ");
  Serial.println(satellites);
  Serial.print("  Accuracy:  ");
  Serial.print(accuracy);
  Serial.println(" HDOP");
  Serial.print("  Status:    ");
  Serial.println(gpsValid ? "ACTIVE ✓" : "NO FIX ✗");
  Serial.println("----------------------------------------");
}
