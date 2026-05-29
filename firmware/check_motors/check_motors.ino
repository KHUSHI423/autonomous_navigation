#include <WiFi.h>
#include <WebServer.h>

/************ WIFI ************/
const char* ssid = "SANJEEVI";
const char* password = "YOUR_WIFI_PASSWORD";

/************ WEB SERVER ************/
WebServer server(80);

/************ SPEED + TRIM ************/
int speedValue = 150;

int leftTrim  = 0;
int rightTrim = -15;

/************ FAILSAFE ************/
unsigned long lastCommandTime = 0;
const unsigned long failsafeTimeout = 1000;

/************ DRIVER 1 (LEFT SIDE) ************/
#define LF_IN1 26
#define LF_IN2 27
#define LF_PWM 25

#define LR_IN1 14
#define LR_IN2 13
#define LR_PWM 33

#define STBY1 32

/************ DRIVER 2 (RIGHT SIDE) ************/
/* GPIO 5 removed (boot strap pin issue) */

#define RF_IN1 4
#define RF_IN2 5
#define RF_PWM 18

#define RR_IN1 19
#define RR_IN2 21
#define RR_PWM 22

#define STBY2 23

/************ PWM ************/
#define PWM_FREQ 1000
#define PWM_RES 8

/************ HTML UI ************/
String webpage = R"====(
<!DOCTYPE html>
<html>

<head>
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>

*{
  margin:0;
  padding:0;
  box-sizing:border-box;
}

body{
  background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
  min-height:100vh;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  font-family:Arial;
  color:white;
}

h1{
  margin-bottom:20px;
  font-size:32px;
}

.status{
  margin-bottom:20px;
  padding:10px 20px;
  border-radius:20px;
  background:#111;
  color:#00ff99;
}

.dpad{
  position:relative;
  width:220px;
  height:220px;
}

.btn{
  position:absolute;
  width:70px;
  height:70px;
  border:none;
  border-radius:18px;
  font-size:24px;
  font-weight:bold;
  background:#111;
  color:#00ff99;
  box-shadow:0 0 15px rgba(0,255,153,0.4);
}

.btn:active{
  transform:scale(0.95);
}

#up{
  top:0;
  left:50%;
  transform:translateX(-50%);
}

#down{
  bottom:0;
  left:50%;
  transform:translateX(-50%);
}

#left{
  left:0;
  top:50%;
  transform:translateY(-50%);
}

#right{
  right:0;
  top:50%;
  transform:translateY(-50%);
}

#stop{
  left:50%;
  top:50%;
  transform:translate(-50%,-50%);
  background:#ff3355;
  color:white;
  width:60px;
  height:60px;
  font-size:12px;
}

.controls{
  margin-top:30px;
  width:300px;
}

.slider{
  width:100%;
}

.speedText{
  margin-top:10px;
  text-align:center;
  font-size:22px;
  color:#00ff99;
}

.trimSection{
  margin-top:25px;
  padding:15px;
  background:rgba(0,0,0,0.3);
  border-radius:12px;
  width:300px;
}

.trimTitle{
  text-align:center;
  font-size:18px;
  color:#ffcc00;
  margin-bottom:10px;
}

.trimSlider{
  width:100%;
}

.trimValue{
  text-align:center;
  font-size:18px;
  color:#ffcc00;
  margin-top:5px;
}

</style>
</head>

<body>

<h1>🤖 EDGE DRIVE</h1>

<div class="status" id="status">
CONNECTED
</div>

<div class="dpad">

<button class="btn" id="up"
onpointerdown="sendCmd('forward')"
onpointerup="sendCmd('stop')">▲</button>

<button class="btn" id="down"
onpointerdown="sendCmd('backward')"
onpointerup="sendCmd('stop')">▼</button>

<button class="btn" id="left"
onpointerdown="sendCmd('left')"
onpointerup="sendCmd('stop')">◄</button>

<button class="btn" id="right"
onpointerdown="sendCmd('right')"
onpointerup="sendCmd('stop')">►</button>

<button class="btn" id="stop"
onclick="sendCmd('stop')">STOP</button>

</div>

<div class="controls">

<input type="range"
min="70"
max="255"
value="150"
class="slider"
id="speedSlider"
oninput="updateSpeed(this.value)">

<div class="speedText">
Speed: <span id="speedValue">150</span>
</div>

</div>

<div class="trimSection">
<div class="trimTitle">⚙️ Right Trim</div>
<input type="range"
min="-50"
max="50"
value="-15"
class="trimSlider"
id="trimSlider"
oninput="updateTrim(this.value)">
<div class="trimValue">
Trim: <span id="trimValue">-15</span>
</div>
</div>

<script>

function sendCmd(cmd)
{
  fetch('/'+cmd);

  document.getElementById('status').innerHTML =
  'COMMAND : ' + cmd.toUpperCase();
}

function updateSpeed(val)
{
  fetch('/speed?value='+val);

  document.getElementById('speedValue').innerHTML = val;
}

function updateTrim(val)
{
  fetch('/trim?value='+val);

  document.getElementById('trimValue').innerHTML = val;
}

/************ KEYBOARD SUPPORT ************/

document.addEventListener('keydown', function(e){

  if(e.repeat) return;

  if(e.key === 'ArrowUp')
    sendCmd('forward');

  else if(e.key === 'ArrowDown')
    sendCmd('backward');

  else if(e.key === 'ArrowLeft')
    sendCmd('left');

  else if(e.key === 'ArrowRight')
    sendCmd('right');

  else if(e.key === ' ')
    sendCmd('stop');
});

document.addEventListener('keyup', function(e){

  if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key))
  {
    sendCmd('stop');
  }
});

window.onblur = function()
{
  sendCmd('stop');
};

</script>

</body>
</html>
)====";

/************ MOTOR FUNCTION ************/
void setMotor(int in1, int in2, int pwm, bool d1, bool d2, int pwmValue)
{
  digitalWrite(in1, d1);
  digitalWrite(in2, d2);

  pwmValue = constrain(pwmValue, 0, 255);

  ledcWrite(pwm, pwmValue);
}

/************ STOP MOTOR ************/
void stopMotor(int in1, int in2, int pwm)
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);

  ledcWrite(pwm, 0);
}

/************ FORWARD ************/
void moveForward()
{
  int leftSpeed  = speedValue + leftTrim;
  int rightSpeed = speedValue + rightTrim;

  setMotor(LF_IN1, LF_IN2, LF_PWM, HIGH, LOW, leftSpeed);
  setMotor(LR_IN1, LR_IN2, LR_PWM, HIGH, LOW, leftSpeed);

  setMotor(RF_IN1, RF_IN2, RF_PWM, HIGH, LOW, rightSpeed);
  setMotor(RR_IN1, RR_IN2, RR_PWM, HIGH, LOW, rightSpeed);
}

/************ BACKWARD ************/
void moveBackward()
{
  int leftSpeed  = speedValue + leftTrim;
  int rightSpeed = speedValue + rightTrim;

  setMotor(LF_IN1, LF_IN2, LF_PWM, LOW, HIGH, leftSpeed);
  setMotor(LR_IN1, LR_IN2, LR_PWM, LOW, HIGH, leftSpeed);

  setMotor(RF_IN1, RF_IN2, RF_PWM, LOW, HIGH, rightSpeed);
  setMotor(RR_IN1, RR_IN2, RR_PWM, LOW, HIGH, rightSpeed);
}

/************ LEFT ************/
void turnLeft()
{
  setMotor(LF_IN1, LF_IN2, LF_PWM, LOW, HIGH, speedValue);
  setMotor(LR_IN1, LR_IN2, LR_PWM, LOW, HIGH, speedValue);

  setMotor(RF_IN1, RF_IN2, RF_PWM, HIGH, LOW, speedValue);
  setMotor(RR_IN1, RR_IN2, RR_PWM, HIGH, LOW, speedValue);
}

/************ RIGHT ************/
void turnRight()
{
  setMotor(LF_IN1, LF_IN2, LF_PWM, HIGH, LOW, speedValue);
  setMotor(LR_IN1, LR_IN2, LR_PWM, HIGH, LOW, speedValue);

  setMotor(RF_IN1, RF_IN2, RF_PWM, LOW, HIGH, speedValue);
  setMotor(RR_IN1, RR_IN2, RR_PWM, LOW, HIGH, speedValue);
}

/************ STOP ************/
void stopMotors()
{
  stopMotor(LF_IN1, LF_IN2, LF_PWM);
  stopMotor(LR_IN1, LR_IN2, LR_PWM);

  stopMotor(RF_IN1, RF_IN2, RF_PWM);
  stopMotor(RR_IN1, RR_IN2, RR_PWM);
}

/************ SETUP ************/
void setup()
{
  Serial.begin(115200);

  pinMode(LF_IN1, OUTPUT);
  pinMode(LF_IN2, OUTPUT);

  pinMode(LR_IN1, OUTPUT);
  pinMode(LR_IN2, OUTPUT);

  pinMode(RF_IN1, OUTPUT);
  pinMode(RF_IN2, OUTPUT);

  pinMode(RR_IN1, OUTPUT);
  pinMode(RR_IN2, OUTPUT);

  pinMode(STBY1, OUTPUT);
  pinMode(STBY2, OUTPUT);

  /************ PWM ************/
  ledcAttach(LF_PWM, PWM_FREQ, PWM_RES);
  ledcAttach(LR_PWM, PWM_FREQ, PWM_RES);

  ledcAttach(RF_PWM, PWM_FREQ, PWM_RES);
  ledcAttach(RR_PWM, PWM_FREQ, PWM_RES);

  /************ SAFETY STOP ************/
  stopMotors();

  /************ ENABLE DRIVERS ************/
  digitalWrite(STBY1, HIGH);
  digitalWrite(STBY2, HIGH);

  /************ WIFI ************/
  WiFi.mode(WIFI_STA);

  WiFi.begin(ssid, password);

  Serial.print("Connecting");

  while(WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi Connected");
  Serial.println(WiFi.localIP());

  /************ ROUTES ************/
  server.on("/", []()
  {
    server.send(200, "text/html", webpage);
  });

  server.on("/forward", []()
  {
    lastCommandTime = millis();

    moveForward();

    server.send(200, "text/plain", "OK");
  });

  server.on("/backward", []()
  {
    lastCommandTime = millis();

    moveBackward();

    server.send(200, "text/plain", "OK");
  });

  server.on("/left", []()
  {
    lastCommandTime = millis();

    turnLeft();

    server.send(200, "text/plain", "OK");
  });

  server.on("/right", []()
  {
    lastCommandTime = millis();

    turnRight();

    server.send(200, "text/plain", "OK");
  });

  server.on("/stop", []()
  {
    stopMotors();

    server.send(200, "text/plain", "OK");
  });

  server.on("/speed", []()
  {
    if(server.hasArg("value"))
    {
      speedValue = server.arg("value").toInt();

      if(speedValue < 70)
        speedValue = 70;
    }

    server.send(200, "text/plain", "OK");
  });

  server.on("/trim", []()
  {
    if(server.hasArg("value"))
    {
      rightTrim = server.arg("value").toInt();
      Serial.print("Right Trim: ");
      Serial.println(rightTrim);
    }

    server.send(200, "text/plain", "OK");
  });

  server.begin();

  Serial.println("Server Started");
  Serial.println(WiFi.localIP());
}

/************ LOOP ************/
void loop()
{
  server.handleClient();

  /************ FAILSAFE ************/
  if(millis() - lastCommandTime > failsafeTimeout)
  {
    stopMotors();
  }

  /************ WIFI RECONNECT ************/
  if(WiFi.status() != WL_CONNECTED)
  {
    WiFi.reconnect();
  }
}