/*
=============================================================================
EDGE DRIVE 3D - ESP32 MOTOR CONTROLLER
=============================================================================
Web-based robot control with automatic decision-making from laptop
Receives commands via HTTP from the perception system

Pins:
  Motor A: AIN1=26, AIN2=27, PWMA=25
  Motor B: BIN1=14, BIN2=13, PWMB=33
  Standby: STBY=32

Author: EdgeDrive3D Team
=============================================================================
*/

#include <WiFi.h>
#include <WebServer.h>

// ============ MOTOR PINS ============
#define AIN1 26
#define AIN2 27
#define PWMA 25

#define BIN1 14
#define BIN2 13
#define PWMB 33

#define STBY 32

// ============ PWM ============
#define PWM_FREQ 1000
#define PWM_RES 8

// ============ WIFI ============
const char* AP_SSID = "EdgeDrive3D_Robot";
const char* AP_PASS = "edgedrive123";

// ============ SERVER ============
WebServer server(80);

// ============ MOTOR SPEED ============
int speedValue = 180;

// ============ AUTO MODE ============
bool autoMode = false;
unsigned long lastCommandTime = 0;
const unsigned long AUTO_TIMEOUT = 2000; // 2 seconds

// ============ HTML INTERFACE ============
String webpage = R"====(
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EdgeDrive3D Robot Control</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      color: white;
      font-family: 'Segoe UI', Arial, sans-serif;
      text-align: center;
      padding: 20px;
      min-height: 100vh;
    }
    h1 {
      font-size: 28px;
      margin-bottom: 10px;
      background: linear-gradient(90deg, #00d2ff, #3a7bd5);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .status {
      font-size: 14px;
      color: #aaa;
      margin-bottom: 30px;
    }
    .control-panel {
      background: rgba(255,255,255,0.1);
      backdrop-filter: blur(10px);
      border-radius: 20px;
      padding: 30px;
      max-width: 400px;
      margin: 0 auto;
      border: 1px solid rgba(255,255,255,0.2);
    }
    .dpad {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin: 20px 0;
    }
    button {
      width: 100%;
      height: 70px;
      font-size: 18px;
      font-weight: 600;
      border: none;
      border-radius: 12px;
      background: linear-gradient(145deg, #3a7bd5, #00d2ff);
      color: white;
      cursor: pointer;
      transition: all 0.2s;
      box-shadow: 0 4px 15px rgba(0,210,255,0.3);
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0,210,255,0.5);
    }
    button:active {
      transform: translateY(0);
    }
    button.stop {
      background: linear-gradient(145deg, #ff416c, #ff4b2b);
      box-shadow: 0 4px 15px rgba(255,75,43,0.3);
    }
    .speed-control {
      margin-top: 25px;
    }
    .speed-control label {
      display: block;
      margin-bottom: 10px;
      font-size: 16px;
    }
    input[type=range] {
      width: 100%;
      height: 8px;
      border-radius: 4px;
      background: rgba(255,255,255,0.2);
      outline: none;
      -webkit-appearance: none;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: linear-gradient(145deg, #3a7bd5, #00d2ff);
      cursor: pointer;
      box-shadow: 0 2px 10px rgba(0,210,255,0.5);
    }
    .speed-value {
      font-size: 24px;
      font-weight: 700;
      margin-top: 10px;
      color: #00d2ff;
    }
    .mode-toggle {
      margin-top: 20px;
      padding: 15px;
      background: rgba(255,255,255,0.05);
      border-radius: 10px;
    }
    .mode-toggle button {
      width: auto;
      padding: 10px 30px;
      height: auto;
    }
    .mode-toggle button.active {
      background: linear-gradient(145deg, #11998e, #38ef7d);
    }
    .info {
      margin-top: 30px;
      font-size: 12px;
      color: #888;
    }
  </style>
</head>
<body>
  <h1>🤖 EdgeDrive3D Robot</h1>
  <div class="status">ESP32 Motor Controller</div>
  
  <div class="control-panel">
    <div class="dpad">
      <div></div>
      <button ontouchstart="sendCmd('forward')" onclick="sendCmd('forward')">▲</button>
      <div></div>
      
      <button ontouchstart="sendCmd('left')" onclick="sendCmd('left')">◀</button>
      <button class="stop" ontouchstart="sendCmd('stop')" onclick="sendCmd('stop')">■</button>
      <button ontouchstart="sendCmd('right')" onclick="sendCmd('right')">▶</button>
      
      <div></div>
      <button ontouchstart="sendCmd('backward')" onclick="sendCmd('backward')">▼</button>
      <div></div>
    </div>
    
    <div class="speed-control">
      <label>Speed</label>
      <input type="range" min="0" max="255" value="180" 
             onchange="sendSpeed(this.value)" 
             oninput="document.getElementById('speedVal').textContent = this.value">
      <div class="speed-value"><span id="speedVal">180</span> / 255</div>
    </div>
    
    <div class="mode-toggle">
      <button id="autoBtn" onclick="toggleAuto()">Auto Mode: OFF</button>
    </div>
    
    <div class="info">
      Control via web interface or autonomous mode from laptop
    </div>
  </div>
  
  <script>
    function sendCmd(cmd) {
      fetch('/' + cmd)
        .then(r => console.log(cmd, r.ok))
        .catch(e => console.error(e));
    }
    
    function sendSpeed(val) {
      fetch('/speed?value=' + val)
        .then(r => console.log('speed:', val, r.ok));
    }
    
    function toggleAuto() {
      fetch('/toggle_auto')
        .then(r => r.text())
        .then(state => {
          const btn = document.getElementById('autoBtn');
          btn.textContent = 'Auto Mode: ' + (state === 'true' ? 'ON' : 'OFF');
          btn.classList.toggle('active', state === 'true');
        });
    }
    
    // Poll auto mode status
    setInterval(() => {
      fetch('/auto_status')
        .then(r => r.text())
        .then(state => {
          const btn = document.getElementById('autoBtn');
          btn.textContent = 'Auto Mode: ' + (state === 'true' ? 'ON' : 'OFF');
          btn.classList.toggle('active', state === 'true');
        });
    }, 2000);
  </script>
</body>
</html>
)====";

// ============ MOTOR FUNCTIONS ============

void setMotorA(bool d1, bool d2, int pwm) {
  digitalWrite(AIN1, d1);
  digitalWrite(AIN2, d2);
  ledcWrite(PWMA, pwm);
}

void setMotorB(bool d1, bool d2, int pwm) {
  digitalWrite(BIN1, d1);
  digitalWrite(BIN2, d2);
  ledcWrite(PWMB, pwm);
}

void moveForward() {
  setMotorA(HIGH, LOW, speedValue);
  setMotorB(HIGH, LOW, speedValue);
}

void moveBackward() {
  setMotorA(LOW, HIGH, speedValue);
  setMotorB(LOW, HIGH, speedValue);
}

void turnLeft() {
  setMotorA(LOW, LOW, 0);
  setMotorB(HIGH, LOW, speedValue);
}

void turnRight() {
  setMotorA(HIGH, LOW, speedValue);
  setMotorB(LOW, LOW, 0);
}

void stopMotors() {
  setMotorA(LOW, LOW, 0);
  setMotorB(LOW, LOW, 0);
}

// ============ SERVER HANDLERS ============

void handleRoot() {
  server.send(200, "text/html", webpage);
}

void handleForward() {
  autoMode = false;
  moveForward();
  server.send(200, "text/plain", "OK");
}

void handleBackward() {
  autoMode = false;
  moveBackward();
  server.send(200, "text/plain", "OK");
}

void handleLeft() {
  autoMode = false;
  turnLeft();
  server.send(200, "text/plain", "OK");
}

void handleRight() {
  autoMode = false;
  turnRight();
  server.send(200, "text/plain", "OK");
}

void handleStop() {
  autoMode = false;
  stopMotors();
  server.send(200, "text/plain", "OK");
}

void handleSpeed() {
  if (server.hasArg("value")) {
    speedValue = server.arg("value").toInt();
    speedValue = constrain(speedValue, 0, 255);
  }
  server.send(200, "text/plain", "OK");
}

void handleToggleAuto() {
  autoMode = !autoMode;
  server.send(200, "text/plain", autoMode ? "true" : "false");
}

void handleAutoStatus() {
  server.send(200, "text/plain", autoMode ? "true" : "false");
}

void handleAutoCommand() {
  // Receive autonomous command from laptop
  if (server.hasArg("action")) {
    String action = server.arg("action");
    int speed = server.hasArg("speed") ? server.arg("speed").toInt() : speedValue;
    
    autoMode = true;
    lastCommandTime = millis();
    
    if (speed != speedValue) {
      speedValue = constrain(speed, 0, 255);
    }
    
    if (action == "forward") moveForward();
    else if (action == "backward") moveBackward();
    else if (action == "left") turnLeft();
    else if (action == "right") turnRight();
    else stopMotors();
    
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Missing action");
  }
}

void handleNotFound() {
  server.send(404, "text/plain", "Not Found");
}

// ============ SETUP ============

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\nEdgeDrive3D ESP32 Controller");
  Serial.println("==========================");
  
  // Motor pins
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);
  
  // PWM
  ledcAttach(PWMA, PWM_FREQ, PWM_RES);
  ledcAttach(PWMB, PWM_FREQ, PWM_RES);
  
  // Stop motors initially
  stopMotors();
  
  // WiFi AP
  Serial.print("Starting AP... ");
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.println("Done");
  
  Serial.print("IP Address: ");
  Serial.println(WiFi.softAPIP());
  Serial.print("SSID: ");
  Serial.println(AP_SSID);
  Serial.print("Password: ");
  Serial.println(AP_PASS);
  
  // Server routes
  server.on("/", handleRoot);
  server.on("/forward", handleForward);
  server.on("/backward", handleBackward);
  server.on("/left", handleLeft);
  server.on("/right", handleRight);
  server.on("/stop", handleStop);
  server.on("/speed", handleSpeed);
  server.on("/toggle_auto", handleToggleAuto);
  server.on("/auto_status", handleAutoStatus);
  server.on("/auto_command", handleAutoCommand);
  server.onNotFound(handleNotFound);
  
  server.begin();
  Serial.println("\nHTTP server started");
  Serial.println("Ready for commands!\n");
}

// ============ LOOP ============

void loop() {
  server.handleClient();
  
  // Auto timeout - stop if no command received
  if (autoMode && millis() - lastCommandTime > AUTO_TIMEOUT) {
    stopMotors();
    autoMode = false;
    Serial.println("Auto timeout - stopped");
  }
  
  delay(2);
}
