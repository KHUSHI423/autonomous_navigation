/*
=============================================================================
ROBOT CAR V2 - MAIN MOTOR CONTROLLER (ESP32 #1)
=============================================================================
Controls motors, receives commands from laptop, receives ultrasonic data from ESP32 #2
WiFi Station mode - connects to your phone WiFi

Features:
- Receives throttle/steering from laptop
- Receives ultrasonic distance from ESP32 #2
- Priority to ultrasonic for close obstacles
- Sends status to laptop

Wiring:
- Driver 1 (Left):  IN1=26, IN2=27, PWM=25 | IN1=14, IN2=13, PWM=33 | STBY=32
- Driver 2 (Right): IN1=4,  IN2=5,  PWM=18 | IN1=19, IN2=21, PWM=22 | STBY=23

Upload to MOTOR CONTROL ESP32!
=============================================================================
*/

#include <WiFi.h>
#include <WebServer.h>
#include <WiFiUDP.h>
#include "soc/soc.h"           // Disable brownout
#include "soc/rtc_cntl_reg.h"  // Disable brownout

/************ YOUR WiFi NETWORK ************/
const char* ssid = "SANJEEVI";        // CHANGE to your phone WiFi
const char* password = "YOUR_WIFI_PASSWORD";   // CHANGE to your WiFi password
/*********************************************/

/************ MOTOR DRIVER PINS ************/
// Driver 1 (LEFT)
#define LF_IN1 26
#define LF_IN2 27
#define LF_PWM 25
#define LR_IN1 14
#define LR_IN2 13
#define LR_PWM 33
#define STBY1 32

// Driver 2 (RIGHT)
#define RF_IN1 4
#define RF_IN2 5
#define RF_PWM 18
#define RR_IN1 19
#define RR_IN2 21
#define RR_PWM 22
#define STBY2 23

#define PWM_FREQ 1000
#define PWM_RES 8

/************ NETWORK ************/
WebServer server(8080);  // Port 8080 for ultrasonic data
WiFiUDP udpCommand;
WiFiUDP udpStatus;
WiFiUDP udpUltrasonic;  // UDP for ultrasonic sensor

const int COMMAND_PORT = 9000;
const int STATUS_PORT = 9001;
const int ULTRASONIC_PORT = 9002;  // UDP port for ultrasonic

/************ STATE ************/
int baseSpeed = 180;
int autonomousSpeed = 0;
float steering = 0.0;
bool autonomousMode = false;
unsigned long lastCommandTime = 0;
const int COMMAND_TIMEOUT = 500;

float ultrasonicDistance = 999.0;  // Default: no obstacle
unsigned long lastUltrasonicTime = 0;

/************ HTML PAGE ************/
String webpage = R"====(
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { text-align:center; background:#0f2027; color:white; font-family:Arial; margin:0; padding:20px; }
h2 { color:#00d2ff; }
button { 
  margin:8px; padding:20px 40px; font-size:20px; 
  background:#00d2ff; border:none; color:#0f2027; 
  border-radius:12px; cursor:pointer; font-weight:bold;
  box-shadow: 0 4px 15px rgba(0,210,255,0.3);
}
button:active { background:#0099cc; transform: scale(0.95); }
input[type=range] { width:80%; margin:15px; }
.status { 
  margin:15px; padding:15px; background:#1a1a2e; 
  border-radius:12px; display:inline-block; min-width:250px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.mode-indicator {
  padding:15px 30px; border-radius:12px; margin:15px;
  font-weight:bold; font-size:22px; display:inline-block;
}
.autonomous { background:#ff4444; color:white; }
.manual { background:#44ff44; color:#0f2027; }
.sensor { font-size:24px; color:#00ff00; margin:10px; }
</style>
</head>
<body>
<h2>🤖 ROBOT CAR V2</h2>

<div id="modeIndicator" class="mode-indicator manual">MANUAL MODE</div>

<div class="status">
  <div class="sensor">📡 Ultrasonic: <span id="ultraVal">--</span> cm</div>
  <div>Speed: <span id="speedVal">180</span></div>
  <div>Mode: <span id="modeVal">MANUAL</span></div>
  <div>Steering: <span id="steerVal">0.00</span></div>
</div>

<br>

<button onclick="setMode('manual')">✋ MANUAL</button>
<button onclick="setMode('auto')">🤖 AUTO</button>

<br><br>

<button onclick="sendCmd('forward')">⬆️ FORWARD</button><br>
<button onclick="sendCmd('left')">⬅️ LEFT</button>
<button onclick="sendCmd('stop')">🛑 STOP</button>
<button onclick="sendCmd('right')">➡️ RIGHT</button><br>
<button onclick="sendCmd('backward')">⬇️ BACKWARD</button>

<br><br>

Speed: <input type="range" min="0" max="255" value="180"
  onchange="sendSpeed(this.value)" oninput="document.getElementById('speedVal').innerText=this.value">

<script>
function sendCmd(cmd){ fetch("/"+cmd); updateStatus(); }
function sendSpeed(val){ fetch("/speed?value="+val); }
function setMode(mode){ fetch("/mode?m="+mode); updateStatus(); }

function updateStatus() {
  fetch("/status").then(r=>r.json()).then(d=>{
    document.getElementById('speedVal').innerText = d.speed;
    document.getElementById('modeVal').innerText = d.mode;
    document.getElementById('steerVal').innerText = d.steering;
    document.getElementById('ultraVal').innerText = d.ultrasonic;
    const modeInd = document.getElementById('modeIndicator');
    if(d.mode === 'AUTONOMOUS') {
      modeInd.className = 'mode-indicator autonomous';
      modeInd.innerText = '🤖 AUTONOMOUS';
    } else {
      modeInd.className = 'mode-indicator manual';
      modeInd.innerText = '✋ MANUAL';
    }
  });
}

setInterval(updateStatus, 300);
</script>

</body>
</html>
)====";

/************ MOTOR CONTROL ************/

void setMotor(int in1, int in2, int pwm, bool d1, bool d2, int speed)
{
  digitalWrite(in1, d1);
  digitalWrite(in2, d2);
  ledcWrite(pwm, speed);
}

void stopMotor(int in1, int in2, int pwm)
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  ledcWrite(pwm, 0);
}

void setAllMotors(bool leftFwd, bool leftBack, bool rightFwd, bool rightBack, int leftSpeed, int rightSpeed)
{
  setMotor(LF_IN1, LF_IN2, LF_PWM, leftFwd, leftBack, leftSpeed);
  setMotor(LR_IN1, LR_IN2, LR_PWM, leftFwd, leftBack, leftSpeed);
  setMotor(RF_IN1, RF_IN2, RF_PWM, rightFwd, rightBack, rightSpeed);
  setMotor(RR_IN1, RR_IN2, RR_PWM, rightFwd, rightBack, rightSpeed);
}

void moveForward(int speed)
{
  setAllMotors(HIGH, LOW, HIGH, LOW, speed, speed);
}

void moveBackward(int speed)
{
  setAllMotors(LOW, HIGH, LOW, HIGH, speed, speed);
}

void turnLeft(int speed)
{
  setAllMotors(LOW, HIGH, HIGH, LOW, speed, speed);
}

void turnRight(int speed)
{
  setAllMotors(HIGH, LOW, LOW, HIGH, speed, speed);
}

void stopMotors()
{
  setAllMotors(LOW, LOW, LOW, LOW, 0, 0);
}

void autonomousDrive(int throttle, float steer)
{
  // Handle reverse (negative throttle)
  if (throttle < 0) {
    // Reverse with steering
    int absThrottle = abs(throttle);

    int leftSpeed = absThrottle;
    int rightSpeed = absThrottle;

    if (steer < 0) {
      // Turning left while reversing - reduce left speed
      leftSpeed = absThrottle * (1.0 - abs(steer));
    } else if (steer > 0) {
      // Turning right while reversing - reduce right speed
      rightSpeed = absThrottle * (1.0 - steer);
    }

    leftSpeed = max(leftSpeed, 30);
    rightSpeed = max(rightSpeed, 30);

    // Reverse: both motors backward
    setAllMotors(LOW, HIGH, LOW, HIGH, leftSpeed, rightSpeed);
    return;
  }

  // Forward or stop
  if (throttle < 10) {
    stopMotors();
    return;
  }

  // All forward turning - differential steering (both wheels same direction)
  int leftSpeed = throttle;
  int rightSpeed = throttle;

  if (steer < 0) {
    // Turning left: reduce LEFT speed (left wheel slower = turn left)
    leftSpeed = throttle * (1.0 - abs(steer));
  } else if (steer > 0) {
    // Turning right: reduce RIGHT speed (right wheel slower = turn right)
    rightSpeed = throttle * (1.0 - steer);
  }

  // Lower minimum for better turning (heavy body)
  leftSpeed = max(leftSpeed, 30);
  rightSpeed = max(rightSpeed, 30);

  setAllMotors(HIGH, LOW, HIGH, LOW, leftSpeed, rightSpeed);
}

/************ WEB SERVER HANDLERS ************/

void handleRoot() {
  server.send(200, "text/html", webpage);
}

void handleCommand(String cmd) {
  autonomousMode = false;
  
  if (cmd == "forward") moveForward(baseSpeed);
  else if (cmd == "backward") moveBackward(baseSpeed);
  else if (cmd == "left") turnLeft(baseSpeed);
  else if (cmd == "right") turnRight(baseSpeed);
  else if (cmd == "stop") stopMotors();
  
  server.send(200, "text/plain", "OK");
}

void handleSpeed() {
  if (server.hasArg("value")) {
    baseSpeed = server.arg("value").toInt();
  }
  server.send(200, "text/plain", "OK");
}

void handleMode() {
  if (server.hasArg("m")) {
    String mode = server.arg("m");
    if (mode == "auto") {
      autonomousMode = true;
      stopMotors();
    } else {
      autonomousMode = false;
      stopMotors();
    }
  }
  server.send(200, "text/plain", "OK");
}

void handleStatus() {
  String json = "{";
  json += "\"speed\":" + String(autonomousMode ? autonomousSpeed : baseSpeed) + ",";
  json += "\"mode\":\"" + String(autonomousMode ? "AUTONOMOUS" : "MANUAL") + "\",";
  json += "\"steering\":" + String(steering, 2) + ",";
  json += "\"ultrasonic\":" + String(ultrasonicDistance, 1);
  json += "}";
  server.send(200, "application/json", json);
}

void handleUltrasonic() {
  if (server.hasArg("value")) {
    ultrasonicDistance = server.arg("value").toFloat();
    lastUltrasonicTime = millis();
    Serial.print("📡 Ultrasonic: ");
    Serial.print(ultrasonicDistance);
    Serial.println(" cm");
  }
  server.send(200, "text/plain", "OK");
}

/************ UDP COMMAND RECEIVER ************/

void processUDPCommand() {
  char packetBuffer[256];
  int packetSize = udpCommand.parsePacket();
  
  if (packetSize > 0) {
    int len = udpCommand.read(packetBuffer, sizeof(packetBuffer));
    if (len > 0) {
      packetBuffer[len] = '\0';
      lastCommandTime = millis();
      
      String command = String(packetBuffer);
      
      if (command == "STOP") {
        stopMotors();
        autonomousMode = true;
      } else if (command == "MANUAL") {
        autonomousMode = false;
        stopMotors();
      } else {
        int colonIndex = command.indexOf(':');
        if (colonIndex > 0) {
          String throttleStr = command.substring(0, colonIndex);
          String steerStr = command.substring(colonIndex + 1);
          
          int throttle = throttleStr.toInt();
          float steer = steerStr.toFloat();
          
          autonomousSpeed = throttle;
          steering = steer;
          autonomousMode = true;
          
          autonomousDrive(throttle, steer);
          
          String status = "OK:" + String(throttle) + ":" + String(steer, 2);
          udpStatus.beginPacket(udpCommand.remoteIP(), STATUS_PORT);
          udpStatus.print(status);
          udpStatus.endPacket();
        }
      }
      
      Serial.print("Command: ");
      Serial.println(command);
    }
  }
}

void sendStatus() {
  static unsigned long lastSend = 0;
  if (millis() - lastSend > 100) {
    String status = "STATUS:";
    status += String(autonomousMode ? "AUTO" : "MANUAL");
    status += ":" + String(autonomousSpeed);
    status += ":" + String(steering, 2);
    status += ":" + String(ultrasonicDistance, 1);

    // Broadcast to entire subnet (e.g., 10.217.106.255)
    // or use last known laptop IP if we received a command
    IPAddress broadcastIP = udpCommand.remoteIP();
    if (broadcastIP == IPAddress(0,0,0,0)) {
      // Broadcast to local subnet so all devices receive it
      broadcastIP = IPAddress(255, 255, 255, 255);
    }
    
    udpStatus.beginPacket(broadcastIP, STATUS_PORT);
    udpStatus.print(status);
    udpStatus.endPacket();
    lastSend = millis();
  }
}

void processUltrasonicUDP() {
  char packetBuffer[256];
  int packetSize = udpUltrasonic.parsePacket();
  
  if (packetSize > 0) {
    int len = udpUltrasonic.read(packetBuffer, sizeof(packetBuffer));
    if (len > 0) {
      packetBuffer[len] = '\0';
      String packet = String(packetBuffer);
      
      // Parse: "ULTRA:25.50"
      if (packet.startsWith("ULTRA:")) {
        ultrasonicDistance = packet.substring(6).toFloat();
        lastUltrasonicTime = millis();
        Serial.print("📡 UDP Ultrasonic: ");
        Serial.print(ultrasonicDistance);
        Serial.println(" cm");
      }
    }
  }
}

void checkCommandTimeout() {
  if (autonomousMode && (millis() - lastCommandTime > COMMAND_TIMEOUT)) {
    stopMotors();
  }
}

/************ SETUP ************/

void setup() {
  // Disable brownout detection (prevents WiFi dropout when motors start)
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  
  Serial.begin(115200);
  Serial.println("\n\n🤖 Robot Car V2 - Motor Controller");
  Serial.println("================================");
  Serial.println("⚠️  Brownout detection DISABLED");
  
  // Motor pins
  pinMode(LF_IN1, OUTPUT); pinMode(LF_IN2, OUTPUT);
  pinMode(LR_IN1, OUTPUT); pinMode(LR_IN2, OUTPUT);
  pinMode(RF_IN1, OUTPUT); pinMode(RF_IN2, OUTPUT);
  pinMode(RR_IN1, OUTPUT); pinMode(RR_IN2, OUTPUT);
  pinMode(STBY1, OUTPUT); pinMode(STBY2, OUTPUT);
  
  digitalWrite(STBY1, HIGH);
  digitalWrite(STBY2, HIGH);
  
  // PWM
  ledcAttach(LF_PWM, PWM_FREQ, PWM_RES);
  ledcAttach(LR_PWM, PWM_FREQ, PWM_RES);
  ledcAttach(RF_PWM, PWM_FREQ, PWM_RES);
  ledcAttach(RR_PWM, PWM_FREQ, PWM_RES);
  
  // WiFi - Connect to your phone
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 30) {
    delay(500);
    Serial.print(".");
    timeout++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Laptop should connect to same WiFi and use this IP");
  } else {
    Serial.println("\n❌ WiFi Connection Failed!");
  }
  
  // Web server
  server.on("/", handleRoot);
  server.on("/forward", [](){ handleCommand("forward"); });
  server.on("/backward", [](){ handleCommand("backward"); });
  server.on("/left", [](){ handleCommand("left"); });
  server.on("/right", [](){ handleCommand("right"); });
  server.on("/stop", [](){ handleCommand("stop"); });
  server.on("/speed", handleSpeed);
  server.on("/mode", handleMode);
  server.on("/status", handleStatus);
  server.on("/distance", handleUltrasonic);
  server.begin();
  Serial.println("Web server started on port 80");
  
  // UDP
  udpCommand.begin(COMMAND_PORT);
  udpStatus.begin(STATUS_PORT);
  udpUltrasonic.begin(ULTRASONIC_PORT);  // Listen for ultrasonic data
  Serial.println("UDP Command listener on port " + String(COMMAND_PORT));
  Serial.println("UDP Ultrasonic listener on port " + String(ULTRASONIC_PORT));
  
  Serial.println("\n✅ Ready!\n");
  stopMotors();
}

/************ LOOP ************/

unsigned long lastWiFiCheck = 0;
const int WIFI_CHECK_INTERVAL = 5000;  // Check every 5 seconds

void loop() {
  // Check WiFi and reconnect if needed
  if (millis() - lastWiFiCheck > WIFI_CHECK_INTERVAL) {
    lastWiFiCheck = millis();
    
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("\n⚠️  WiFi lost! Reconnecting...");
      WiFi.reconnect();
      delay(2000);
      
      if (WiFi.status() == WL_CONNECTED) {
        Serial.println("✅ WiFi reconnected!");
        Serial.print("New IP: ");
        Serial.println(WiFi.localIP());
      } else {
        Serial.println("❌ Reconnection failed, will retry...");
      }
    } else {
      // Print WiFi status occasionally
      static unsigned long lastPrint = 0;
      if (millis() - lastPrint > 10000) {
        lastPrint = millis();
        Serial.print("📶 WiFi OK | RSSI: ");
        Serial.print(WiFi.RSSI());
        Serial.println(" dBm");
      }
    }
  }
  
  server.handleClient();
  processUDPCommand();
  processUltrasonicUDP();  // Check for ultrasonic data
  sendStatus();
  checkCommandTimeout();
  delay(1);
}
