#include <WiFi.h>

const char* ssid = "Khushi";
const char* password = "12345678";

WiFiServer server(80);

int counter = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n🚀 ESP32 WiFi Test");
  Serial.println("================");

  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("✅ WiFi connected!");
  Serial.print("📍 IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("\n📡 Server started on port 80");
  Serial.println("Waiting for client...\n");
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    Serial.println("🔗 Client connected!");

    while (client.connected()) {
      counter++;

      String data =
        String(11.0245) + "," +
        String(77.0021) + "," +
        String(counter) + "," +
        String(counter * 0.1) + "\n";

      client.print(data);
      Serial.print("📤 ");
      Serial.println(data);

      delay(500);
    }

    client.stop();
    Serial.println("❌ Client disconnected\n");
  }
}
