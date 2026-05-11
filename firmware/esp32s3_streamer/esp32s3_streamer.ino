/*
 * AI Smart Glasses - Phase 1
 * ESP32-S3 WebSocket JPEG streamer.
 *
 * Board: ESP32-S3 with OV2640 (default pins = XIAO ESP32-S3 Sense).
 * Lib deps:
 *   - esp32 by Espressif Systems (Arduino-ESP32 >= 2.0.14 / 3.x)
 *   - ArduinoWebsockets by Gil Maimon (>= 0.5.3)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include "esp_camera.h"

#include "wifi_config.h"
#include "camera_pins.h"

using namespace websockets;

static WebsocketsClient wsClient;

static uint32_t lastSendMs = 0;
static uint32_t lastReconnectMs = 0;
static uint32_t frameCount = 0;
static uint32_t statWindowStart = 0;

static bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size   = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] init failed: 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, CAM_VFLIP);
    s->set_hmirror(s, CAM_HMIRROR);
    s->set_brightness(s, 0);
    s->set_saturation(s, 0);
  }
  return true;
}

static void onWsMessage(WebsocketsMessage msg) {
  if (msg.isText()) {
    Serial.printf("[WS] text: %s\n", msg.c_str());
  }
}

static void onWsEvent(WebsocketsEvent event, String data) {
  switch (event) {
    case WebsocketsEvent::ConnectionOpened:
      Serial.printf("[WS] connected to %s:%u%s\n", WS_HOST, WS_PORT, WS_PATH);
      break;
    case WebsocketsEvent::ConnectionClosed:
      Serial.println("[WS] disconnected");
      break;
    case WebsocketsEvent::GotPing:
      Serial.println("[WS] ping");
      break;
    case WebsocketsEvent::GotPong:
      Serial.println("[WS] pong");
      break;
  }
}

static void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("[WiFi] connecting to %s", WIFI_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[WiFi] IP = ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[WiFi] connect timeout, will retry in loop()");
  }
}

static bool wsConnect() {
  char url[96];
  snprintf(url, sizeof(url), "ws://%s:%u%s", WS_HOST, WS_PORT, WS_PATH);
  Serial.printf("[WS] connecting %s\n", url);
  return wsClient.connect(url);
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n=== AI Smart Glasses / ESP32-S3 streamer ===");

  if (!initCamera()) {
    Serial.println("[CAM] halt.");
    while (true) delay(1000);
  }

  connectWifi();

  wsClient.onMessage(onWsMessage);
  wsClient.onEvent(onWsEvent);
  wsConnect();

  statWindowStart = millis();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    static uint32_t lastRetry = 0;
    if (millis() - lastRetry > 3000) {
      lastRetry = millis();
      WiFi.reconnect();
    }
    return;
  }

  if (wsClient.available()) {
    wsClient.poll();
  } else if (millis() - lastReconnectMs > 2000) {
    lastReconnectMs = millis();
    wsConnect();
    return;
  }

  const uint32_t now = millis();
  if (!wsClient.available() || now - lastSendMs < TARGET_FRAME_INTERVAL_MS) {
    return;
  }
  lastSendMs = now;

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[CAM] fb_get failed");
    return;
  }

  if (fb->format == PIXFORMAT_JPEG && fb->len > 0) {
    wsClient.sendBinary((const char*)fb->buf, fb->len);
    frameCount++;
  }
  esp_camera_fb_return(fb);

  if (now - statWindowStart >= 2000) {
    float fps = frameCount * 1000.0f / (now - statWindowStart);
    Serial.printf("[STAT] ~%.1f fps, heap=%u, psram=%u\n",
                  fps, (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());
    frameCount = 0;
    statWindowStart = now;
  }
}
