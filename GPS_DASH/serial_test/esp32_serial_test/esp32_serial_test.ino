/*
 * ESP32 → USB Serial Test
 * Simple data stream to test serial communication
 * 
 * Upload this code, then open Serial Monitor at 115200 baud
 */

int counter = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("");
  Serial.println("================================");
  Serial.println("✅ ESP32 Serial Test");
  Serial.println("================================");
  Serial.println("");
  Serial.println("Sending test data every 500ms...");
  Serial.println("Format: lat,lon,counter,value");
  Serial.println("");
}

void loop() {
  counter++;
  
  // Send test data: lat,lon,counter,value
  Serial.print("11.0245,");
  Serial.print("77.0021,");
  Serial.print(counter);
  Serial.print(",");
  Serial.println(counter * 0.1, 1);
  
  // LED blink (optional - if ESP32 has built-in LED)
  // digitalWrite(LED_BUILTIN, counter % 2);
  
  delay(500);
}
