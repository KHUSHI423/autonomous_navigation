/*
 * ESP32 + MPU6050 + Rotary Encoder
 * Output: USB Serial → Laptop → Python
 *
 * Connections:
 * MPU6050:  SDA->GPIO21, SCL->GPIO22, VCC->3.3V, GND->GND
 * Encoder:  DT->GPIO12, CLK->GPIO14, SW->GPIO13, VCC->3.3V, GND->GND
 */

#include <Wire.h>

// ==================== Pin Definitions ====================
#define I2C_SDA 21
#define I2C_SCL 22
#define MPU_ADDR 0x68

#define ENC_CLK 14
#define ENC_DT 12
#define ENC_SW 13

// ==================== Global Variables ====================
float ax, ay, az, gx, gy, gz;
float roll, pitch, yaw;
bool mpuConnected = false;

volatile int encoderValue = 0;
volatile int lastEncoded = 0;

unsigned long lastSendTime = 0;
const long SEND_INTERVAL = 100;

// ==================== Function Declarations ====================
void initMPU6050();
void initEncoder();
void readMPU6050();
int getEncoderAngle();
void sendSensorData();
void IRAM_ATTR updateEncoder();

// ==================== Setup ====================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n✅ ESP32 Ready");
  Serial.println("================");

  Wire.begin(I2C_SDA, I2C_SCL);

  initMPU6050();
  initEncoder();

  Serial.println("\n📡 Sending data...\n");
}

// ==================== Main Loop ====================
void loop() {
  readMPU6050();

  if (millis() - lastSendTime > SEND_INTERVAL) {
    sendSensorData();
    lastSendTime = millis();
  }
}

// ==================== Sensor Data Transmission ====================
void sendSensorData() {
  if (!mpuConnected) {
    // Send encoder only
    Serial.printf("ENC,%d,%lu\n", getEncoderAngle(), millis());
  } else {
    // Format: type,roll,pitch,yaw,ax,ay,az,gx,gy,gz,encoder_angle,timestamp
    Serial.printf(
      "DATA,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%lu\n",
      roll, pitch, yaw,
      ax, ay, az,
      gx, gy, gz,
      getEncoderAngle(),
      millis()
    );
  }
}

// ==================== MPU6050 Functions ====================
void initMPU6050() {
  Serial.println("🔧 Scanning I2C...");
  
  // Scan for I2C devices
  byte count = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("   Found device at 0x%02X\n", addr);
      count++;
    }
    delay(10);
  }
  
  if (count == 0) {
    Serial.println("⚠️  No I2C devices found! Check wiring.");
    Serial.println("   Continuing without MPU6050...");
    return;
  }
  
  Serial.println("\n🔧 Initializing MPU6050...");

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  if (Wire.endTransmission(true) == 0) {
    Serial.println("✅ MPU6050 found!");
    mpuConnected = true;
  } else {
    Serial.println("⚠️  MPU6050 not responding!");
    mpuConnected = false;
    return;
  }
  delay(100);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1A);
  Wire.write(0x03);
  Wire.endTransmission(true);

  Serial.println("✅ MPU6050 initialized");
}

void readMPU6050() {
  if (!mpuConnected) return;
  
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  int16_t rawAx = Wire.read() << 8 | Wire.read();
  int16_t rawAy = Wire.read() << 8 | Wire.read();
  int16_t rawAz = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read();
  int16_t rawGx = Wire.read() << 8 | Wire.read();
  int16_t rawGy = Wire.read() << 8 | Wire.read();
  int16_t rawGz = Wire.read() << 8 | Wire.read();

  ax = rawAx / 16384.0;
  ay = rawAy / 16384.0;
  az = rawAz / 16384.0;

  gx = rawGx / 131.0;
  gy = rawGy / 131.0;
  gz = rawGz / 131.0;

  roll = atan2(ay, az) * 180.0 / PI;
  pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
  yaw += gz * 0.1;
  if (yaw > 180) yaw -= 360;
  if (yaw < -180) yaw += 360;
}

// ==================== Rotary Encoder Functions ====================
void initEncoder() {
  Serial.println("🔧 Initializing Encoder...");

  pinMode(ENC_CLK, INPUT_PULLUP);
  pinMode(ENC_DT, INPUT_PULLUP);
  pinMode(ENC_SW, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC_CLK), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_DT), updateEncoder, CHANGE);

  Serial.println("✅ Encoder initialized");
}

void IRAM_ATTR updateEncoder() {
  int MSB = digitalRead(ENC_CLK);
  int LSB = digitalRead(ENC_DT);

  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncoded << 2) | encoded;

  if (sum == 0b1101 || sum == 0b0100 || sum == 0b1011 || sum == 0b0010)
    encoderValue++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b1001 || sum == 0b0001)
    encoderValue--;

  lastEncoded = encoded;
}

int getEncoderAngle() {
  const int PULSES_PER_REV = 20;
  int angle = encoderValue % PULSES_PER_REV;
  angle = angle * (360 / PULSES_PER_REV);
  if (angle < 0) angle += 360;
  return angle;
}
