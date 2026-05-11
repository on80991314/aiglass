/*
 * AI Smart Glasses — Phase 3+ 整合韌體
 * 元件：XIAO ESP32-S3 Sense / OV5640 / ICM42688 / BNO055
 *        MAX98357A (I2S 喇叭) / PDM 板載麥克風 / MicroSD
 *
 * WebSocket 協定 (單一連線):
 *   TX  0x01 + JPEG          — 影像幀
 *   TX  0x02 + u32LE_sr + i16PCM — 麥克風音訊
 *   TX  text JSON:
 *       {"type":"imu","t":..,"ax":..,"ay":..,"az":..,"gx":..,"gy":..,"gz":..}
 *       {"type":"orientation","t":..,"heading":..,"roll":..,"pitch":..,"cal":..}
 *       {"type":"event","name":"fall","t":..}
 *   RX  0x03 + u32LE_sr + i16PCM — TTS 回放
 *   RX  text JSON {"type":"cmd",...}
 *
 * Arduino Library Manager 需安裝：
 *   - ArduinoWebsockets  by Gil Maimon
 *   - ArduinoJson        by Benoit Blanchon
 *   - Adafruit ICM42688
 *   - Adafruit BNO055 + Adafruit Unified Sensor
 *   (SD 和 driver/i2s 為 Arduino-ESP32 內建)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
// SD 已停用（GPIO 7/8/9 被 MAX98357 喇叭佔用），保留 include 但不啟用。
// #include <SD.h>
#include <SPI.h>
#include <ArduinoWebsockets.h>
#include <ArduinoJson.h>
// ICM42688 + BNO055 都改用純 Wire 暫存器讀取，避免 Adafruit_Sensor.h 與
// esp_camera.h 兩邊都定義 sensor_t 造成 typedef 衝突。
#include "esp_camera.h"
#include "driver/i2s.h"

#include "wifi_config.h"
#include "camera_pins.h"
#include "hw_config.h"

#define TAG_JPEG       0x01
#define TAG_AUDIO_UP   0x02
#define TAG_AUDIO_DOWN 0x03

using namespace websockets;

static WebsocketsClient ws;
static bool icmReady = false;
static bool bnoReady = false;
static bool sdReady  = false;
static uint32_t lastReconnectMs = 0;
// static File logFile;   // SD 已停用

// ====================================================================
// ICM42688 — 純 I2C 暫存器存取（無需 Adafruit library）
// ====================================================================
// Datasheet: ICM-42688-P bank0 暫存器
//  WHO_AM_I = 0x75 -> 0x47
//  PWR_MGMT0 = 0x4E：低 4-bit = 0xF (accel + gyro 都進 LN mode)
//  ACCEL_CONFIG0 = 0x50：bit 7-5 = FS_SEL（0=±16g, 1=±8g, 2=±4g, 3=±2g）
//                       bit 3-0 = ODR（6=1kHz, 8=100Hz, ...）
//  GYRO_CONFIG0 = 0x4F：bit 7-5 = FS_SEL（0=±2000dps, 1=±1000, 2=±500, ...）
//                       bit 3-0 = ODR
//  burst read 從 TEMP_DATA1 (0x1D) 開始，14 bytes：T(2) + ACC(6) + GYR(6)
#define ICM_REG_WHO_AM_I       0x75
#define ICM_REG_PWR_MGMT0      0x4E
#define ICM_REG_GYRO_CONFIG0   0x4F
#define ICM_REG_ACCEL_CONFIG0  0x50
#define ICM_REG_TEMP_DATA1     0x1D
#define ICM_BURST_LEN          14
// ±8g、±500dps、ODR 100Hz
#define ICM_ACC_LSB_PER_G      4096.0f   // ±8g 範圍：32768/8 = 4096
#define ICM_GYR_LSB_PER_DPS    65.5f     // ±500dps 範圍：32768/500 = 65.5
#define ICM_GRAVITY            9.80665f

static uint8_t icm_read8(uint8_t reg) {
  Wire.beginTransmission(ICM42688_I2C_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0;
  Wire.requestFrom((int)ICM42688_I2C_ADDR, 1);
  return Wire.available() ? Wire.read() : 0;
}

static void icm_write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ICM42688_I2C_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static bool icm_burst_read(uint8_t *dst, size_t n) {
  Wire.beginTransmission(ICM42688_I2C_ADDR);
  Wire.write(ICM_REG_TEMP_DATA1);
  if (Wire.endTransmission(false) != 0) return false;
  size_t got = Wire.requestFrom((int)ICM42688_I2C_ADDR, (int)n);
  if (got != n) return false;
  for (size_t i = 0; i < n; ++i) dst[i] = Wire.read();
  return true;
}

static bool initIcm() {
  uint8_t who = icm_read8(ICM_REG_WHO_AM_I);
  Serial.printf("[ICM] WHO_AM_I=0x%02X (expect 0x47)\n", who);
  if (who != 0x47) return false;
  // PWR_MGMT0：accel + gyro 都進 LN mode
  icm_write8(ICM_REG_PWR_MGMT0, 0x0F);
  delay(2);
  // ACCEL_CONFIG0：FS=±8g (1<<5), ODR=100Hz (8)
  icm_write8(ICM_REG_ACCEL_CONFIG0, (1 << 5) | 8);
  // GYRO_CONFIG0：FS=±500dps (2<<5), ODR=100Hz (8)
  icm_write8(ICM_REG_GYRO_CONFIG0,  (2 << 5) | 8);
  delay(2);
  return true;
}

// ====================================================================
// BNO055 — 純 I2C 暫存器存取（無需 Adafruit library）
// ====================================================================
// Datasheet: BNO055
//  CHIP_ID = 0x00 -> 0xA0
//  OPR_MODE = 0x3D：0x00=CONFIG, 0x0C=NDOF（融合輸出 Euler）
//  EULER_H_LSB = 0x1A：6 bytes (heading/roll/pitch, 各 16-bit, LSB first)
//                       單位 1/16 度
//  CALIB_STAT = 0x35：bit 7-6 sys, 5-4 gyr, 3-2 acc, 1-0 mag
#define BNO_REG_CHIP_ID        0x00
#define BNO_REG_OPR_MODE       0x3D
#define BNO_REG_EULER_H_LSB    0x1A
#define BNO_REG_CALIB_STAT     0x35
#define BNO_OPMODE_CONFIG      0x00
#define BNO_OPMODE_NDOF        0x0C

static uint8_t bno_read8(uint8_t reg) {
  Wire.beginTransmission(BNO055_I2C_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0;
  Wire.requestFrom((int)BNO055_I2C_ADDR, 1);
  return Wire.available() ? Wire.read() : 0;
}

static void bno_write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(BNO055_I2C_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static bool bno_read_n(uint8_t reg, uint8_t *dst, size_t n) {
  Wire.beginTransmission(BNO055_I2C_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  size_t got = Wire.requestFrom((int)BNO055_I2C_ADDR, (int)n);
  if (got != n) return false;
  for (size_t i = 0; i < n; ++i) dst[i] = Wire.read();
  return true;
}

static bool initBno() {
  uint8_t id = bno_read8(BNO_REG_CHIP_ID);
  Serial.printf("[BNO] CHIP_ID=0x%02X (expect 0xA0)\n", id);
  if (id != 0xA0) return false;
  // 進 CONFIG 模式才能改設定
  bno_write8(BNO_REG_OPR_MODE, BNO_OPMODE_CONFIG);
  delay(25);
  // 切到 NDOF（融合所有感測器，輸出絕對方向）
  bno_write8(BNO_REG_OPR_MODE, BNO_OPMODE_NDOF);
  delay(20);
  return true;
}

// ====================================================================
// 初始化
// ====================================================================

static bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0; c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0=Y2_GPIO_NUM; c.pin_d1=Y3_GPIO_NUM; c.pin_d2=Y4_GPIO_NUM; c.pin_d3=Y5_GPIO_NUM;
  c.pin_d4=Y6_GPIO_NUM; c.pin_d5=Y7_GPIO_NUM; c.pin_d6=Y8_GPIO_NUM; c.pin_d7=Y9_GPIO_NUM;
  c.pin_xclk=XCLK_GPIO_NUM; c.pin_pclk=PCLK_GPIO_NUM;
  c.pin_vsync=VSYNC_GPIO_NUM; c.pin_href=HREF_GPIO_NUM;
  c.pin_sccb_sda=SIOD_GPIO_NUM; c.pin_sccb_scl=SIOC_GPIO_NUM;
  c.pin_pwdn=PWDN_GPIO_NUM; c.pin_reset=RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  // OV5640 支援最高 UXGA，串流用 VGA 即可；改 FRAMESIZE_SVGA (800x600) 若影像較模糊
  if (psramFound()) {
    c.frame_size=FRAMESIZE_VGA; c.jpeg_quality=10; c.fb_count=2;
    c.fb_location=CAMERA_FB_IN_PSRAM; c.grab_mode=CAMERA_GRAB_LATEST;
  } else {
    c.frame_size=FRAMESIZE_QVGA; c.jpeg_quality=15; c.fb_count=1;
    c.fb_location=CAMERA_FB_IN_DRAM; c.grab_mode=CAMERA_GRAB_WHEN_EMPTY;
  }
  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t *s = esp_camera_sensor_get();
  if (s) { s->set_vflip(s, CAM_VFLIP); s->set_hmirror(s, CAM_HMIRROR); }
  return true;
}

// 麥克風 DC offset 跨呼叫保留（用 EMA 跟蹤），讓 sendMicChunk 持續校正。
static float g_mic_dc_offset = 0.0f;
static bool  g_mic_warmed_up = false;

static bool initPdmMic() {
  i2s_config_t cfg = {};
  cfg.mode             = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
  cfg.sample_rate      = MIC_SAMPLE_RATE;
  cfg.bits_per_sample  = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format   = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_PCM_SHORT;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count    = 6;
  cfg.dma_buf_len      = 256;
  cfg.use_apll         = false;
  if (i2s_driver_install(MIC_I2S_PORT, &cfg, 0, NULL) != ESP_OK) return false;
  i2s_pin_config_t pins = {};
  pins.ws_io_num      = MIC_PDM_CLK;
  pins.data_in_num    = MIC_PDM_DATA;
  pins.bck_io_num     = I2S_PIN_NO_CHANGE;
  pins.data_out_num   = I2S_PIN_NO_CHANGE;
  return i2s_set_pin(MIC_I2S_PORT, &pins) == ESP_OK;
}

// PDM 麥克風暖機 + DC offset 預估。drain 100ms 樣本，用 EMA 算出 DC bias，
// 後續 sendMicChunk() 會在這個基礎上繼續微調。沒做這一步的話，PDM 訊號
// 會帶大量直流偏移送到 Whisper，等同被嚴重削波，內容辨識不出來。
//
// 加 wall-clock timeout：若 i2s_read 一直回 0 bytes，仍會在 ~500ms 後跳出，
// 避免整個 setup() 卡死導致 cam frame buffer overflow。
static void warmupMic() {
  const size_t N = MIC_SAMPLE_RATE / 10;   // ~100ms
  int16_t buf[256];
  size_t collected = 0;
  float dc = 0.0f;
  uint32_t t0 = millis();
  while (collected < N && (millis() - t0) < 500) {
    size_t want = sizeof(buf);
    size_t read = 0;
    if (i2s_read(MIC_I2S_PORT, buf, want, &read, pdMS_TO_TICKS(50)) != ESP_OK) break;
    if (read == 0) continue;
    size_t samples = read / 2;
    for (size_t i = 0; i < samples; ++i) {
      float v = (float)buf[i];
      dc += 0.005f * (v - dc);
    }
    collected += samples;
  }
  g_mic_dc_offset = dc;
  g_mic_warmed_up = true;
  Serial.printf("[MIC] warmup done, collected=%u, DC=%.0f\n",
                (unsigned)collected, dc);
}

static bool initSpeaker() {
  // 對應 correct_pin.ino：32-bit STEREO，相容 MAX98357A 的 left-channel 行為。
  i2s_config_t cfg = {};
  cfg.mode             = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate      = SPK_SAMPLE_RATE;
  cfg.bits_per_sample  = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format   = I2S_CHANNEL_FMT_RIGHT_LEFT;       // STEREO
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count    = 6;
  cfg.dma_buf_len      = 256;
  cfg.use_apll         = false;
  if (i2s_driver_install(SPK_I2S_PORT, &cfg, 0, NULL) != ESP_OK) return false;
  i2s_pin_config_t pins = {};
  pins.bck_io_num     = SPK_BCLK_PIN;
  pins.ws_io_num      = SPK_LRC_PIN;
  pins.data_out_num   = SPK_DIN_PIN;
  pins.data_in_num    = I2S_PIN_NO_CHANGE;
  return i2s_set_pin(SPK_I2S_PORT, &pins) == ESP_OK;
}

// SD 卡因 GPIO 7/8/9 被 MAX98357 喇叭佔用而停用；hw_config.h 已把
// SD_*_PIN 註解掉。`logEvent` 改成只往 Serial 印，保留 callsite 不動。
static bool initSd() {
  return false;
}

static void logEvent(const char *evt) {
  Serial.printf("[EVT] %lu,%s\n", millis(), evt);
}

// ====================================================================
// 發送幀
// ====================================================================

static void sendJpeg() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;
  if (fb->format == PIXFORMAT_JPEG && fb->len > 0) {
    uint8_t *pkt = (uint8_t*)malloc(fb->len + 1);
    if (pkt) {
      pkt[0] = TAG_JPEG;
      memcpy(pkt + 1, fb->buf, fb->len);
      ws.sendBinary((const char*)pkt, fb->len + 1);
      free(pkt);
    }
  }
  esp_camera_fb_return(fb);
}

static void sendMicChunk() {
  const size_t samples = MIC_SAMPLE_RATE / 1000 * MIC_FRAME_MS;
  int16_t raw[samples];
  size_t read = 0;
  if (i2s_read(MIC_I2S_PORT, raw, sizeof(raw), &read, 0) != ESP_OK || read == 0) return;

  // ---- 訊號處理 (correct_pin.ino 風格) ----
  // 1. EMA 跟蹤 DC offset（PDM 麥克風固有偏移）
  // 2. 把樣本減去 DC、放大 2× 再 clip 到 int16 範圍
  // 沒做這一步的話 PDM 訊號的中心點不在 0，Whisper 拿到等同被劈掉
  // 半邊振幅的訊號，辨識率極差。
  size_t n = read / 2;
  for (size_t i = 0; i < n; ++i) {
    float v = (float)raw[i];
    g_mic_dc_offset += 0.005f * (v - g_mic_dc_offset);
    int32_t centered  = (int32_t)(v - g_mic_dc_offset);
    int32_t amplified = centered * 2;
    if (amplified >  32767) amplified =  32767;
    if (amplified < -32768) amplified = -32768;
    raw[i] = (int16_t)amplified;
  }

  uint8_t header[5] = {TAG_AUDIO_UP, 0, 0, 0, 0};
  const uint32_t sr = MIC_SAMPLE_RATE;
  memcpy(header + 1, &sr, 4);
  const size_t plen = 5 + read;
  uint8_t *pkt = (uint8_t*)malloc(plen);
  if (!pkt) return;
  memcpy(pkt, header, 5);
  memcpy(pkt + 5, raw, read);
  ws.sendBinary((const char*)pkt, plen);
  free(pkt);
}

static void sendIcm() {
  if (!icmReady) return;
  uint8_t raw[ICM_BURST_LEN];
  if (!icm_burst_read(raw, ICM_BURST_LEN)) return;
  // raw[0..1] = TEMP, [2..3] AX, [4..5] AY, [6..7] AZ, [8..9] GX, [10..11] GY, [12..13] GZ
  auto s16be = [&](int idx) -> int16_t {
    return (int16_t)((raw[idx] << 8) | raw[idx + 1]);
  };
  int16_t axr = s16be(2),  ayr = s16be(4),  azr = s16be(6);
  int16_t gxr = s16be(8),  gyr = s16be(10), gzr = s16be(12);
  StaticJsonDocument<192> doc;
  doc["type"] = "imu";
  doc["t"]  = millis() / 1000.0f;
  doc["ax"] = ((float)axr / ICM_ACC_LSB_PER_G) * ICM_GRAVITY;
  doc["ay"] = ((float)ayr / ICM_ACC_LSB_PER_G) * ICM_GRAVITY;
  doc["az"] = ((float)azr / ICM_ACC_LSB_PER_G) * ICM_GRAVITY;
  doc["gx"] = (float)gxr / ICM_GYR_LSB_PER_DPS;
  doc["gy"] = (float)gyr / ICM_GYR_LSB_PER_DPS;
  doc["gz"] = (float)gzr / ICM_GYR_LSB_PER_DPS;
  char out[192]; ws.send(out, serializeJson(doc, out, sizeof(out)));
}

static void sendBno() {
  if (!bnoReady) return;
  uint8_t e[6];
  if (!bno_read_n(BNO_REG_EULER_H_LSB, e, 6)) return;
  // Euler heading/roll/pitch：each int16_t LSB-first，單位 1/16 度
  int16_t hRaw = (int16_t)((e[1] << 8) | e[0]);
  int16_t rRaw = (int16_t)((e[3] << 8) | e[2]);
  int16_t pRaw = (int16_t)((e[5] << 8) | e[4]);
  uint8_t cstat = bno_read8(BNO_REG_CALIB_STAT);
  uint8_t sysCal = (cstat >> 6) & 0x03;
  StaticJsonDocument<160> doc;
  doc["type"]    = "orientation";
  doc["t"]       = millis() / 1000.0f;
  doc["heading"] = hRaw / 16.0f;
  doc["roll"]    = rRaw / 16.0f;
  doc["pitch"]   = pRaw / 16.0f;
  doc["cal"]     = sysCal;
  char out[160]; ws.send(out, serializeJson(doc, out, sizeof(out)));
}

// ====================================================================
// 接收
// ====================================================================

// MAX98357 走 32-bit STEREO，但 server 送過來是 mono 16-bit PCM，所以
// 要把每一個 16-bit sample 左移 16 位塞進 32-bit slot 的高位，並在
// L+R 兩個 slot 各放一份相同訊號（mono → fake stereo）。0.8× gain 是
// 為了預留 headroom 避免 clipping。
static void playAudioDown(const uint8_t *data, size_t len) {
  if (len < 5) return;
  const int16_t *in = (const int16_t*)(data + 5);
  size_t inSamp = (len - 5) / 2;

  static int32_t outLR[1024 * 2];   // 1024 stereo pairs = 2048 int32
  size_t pos = 0;
  while (pos < inSamp) {
    size_t batch = inSamp - pos;
    if (batch > 1024) batch = 1024;
    for (size_t i = 0; i < batch; ++i) {
      int32_t s = (int32_t)((float)in[pos + i] * 0.8f);
      int32_t v32 = s << 16;
      outLR[i*2 + 0] = v32;
      outLR[i*2 + 1] = v32;
    }
    size_t bytes = batch * 2 * sizeof(int32_t);
    size_t off = 0;
    while (off < bytes) {
      size_t wrote = 0;
      if (i2s_write(SPK_I2S_PORT, (uint8_t*)outLR + off, bytes - off, &wrote, 100) != ESP_OK) break;
      if (wrote == 0) break;
      off += wrote;
    }
    pos += batch;
  }
}

static void onWsMessage(WebsocketsMessage msg) {
  if (msg.isBinary()) {
    const auto &raw = msg.rawData();
    if (raw.size() > 0 && (uint8_t)raw[0] == TAG_AUDIO_DOWN)
      playAudioDown((const uint8_t*)raw.data(), raw.size());
  } else {
    StaticJsonDocument<128> doc;
    if (deserializeJson(doc, msg.c_str()) == DeserializationError::Ok) {
      const char *name = doc["name"] | "";
      if (strcmp(doc["type"] | "", "cmd") == 0) {
        Serial.printf("[CMD] %s\n", name);
        logEvent(name);
      }
    }
  }
}

static void onWsEvent(WebsocketsEvent ev, String) {
  if (ev == WebsocketsEvent::ConnectionOpened)  { Serial.println("[WS] connected"); logEvent("ws_open"); }
  if (ev == WebsocketsEvent::ConnectionClosed)  { Serial.println("[WS] disconnected"); }
}

// ====================================================================
// Setup / Loop
// ====================================================================

static void connectWifi() {
  WiFi.mode(WIFI_STA); WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi]");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) { delay(250); Serial.print('.'); }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) { Serial.print("[WiFi] IP "); Serial.println(WiFi.localIP()); }
}

static bool wsConnect() {
  char url[96];
  snprintf(url, sizeof(url), "ws://%s:%u%s", WS_HOST, WS_PORT, WS_PATH);
  return ws.connect(url);
}

void setup() {
  Serial.begin(115200); delay(200);
  Serial.println("\n=== AI Smart Glasses ===");

  // I2C
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

  // ICM42688（純 Wire 暫存器存取）
  if (initIcm()) {
    icmReady = true;
    Serial.println("[ICM] ICM42688 ready (raw I2C)");
  } else { Serial.println("[ICM] not found"); }

  // BNO055（純 Wire 暫存器存取）
  if (initBno()) {
    bnoReady = true;
    Serial.println("[BNO] BNO055 ready (raw I2C, NDOF mode)");
  } else { Serial.println("[BNO] not found"); }

  Serial.println("[STEP] init camera...");
  if (!initCamera())  { Serial.println("[CAM] init fail"); }
  else                { Serial.println("[CAM] ok"); }

  Serial.println("[STEP] init mic...");
  if (!initPdmMic())  { Serial.println("[MIC] PDM init fail"); }
  else                { Serial.println("[MIC] init ok, warming up..."); warmupMic(); }

  Serial.println("[STEP] gain pin...");
  pinMode(SPK_GAIN_PIN, OUTPUT);
  digitalWrite(SPK_GAIN_PIN, LOW);

  Serial.println("[STEP] init speaker...");
  if (!initSpeaker()) { Serial.println("[SPK] init fail"); }
  else                { Serial.println("[SPK] ok"); }

  Serial.println("[STEP] init sd...");
  sdReady = initSd();
  Serial.printf("[SD] %s\n", sdReady ? "ready" : "off (disabled)");

  Serial.println("[STEP] connect wifi...");
  connectWifi();
  Serial.println("[STEP] connect ws...");
  ws.onMessage(onWsMessage);
  ws.onEvent(onWsEvent);
  wsConnect();
  Serial.println("[STEP] setup done, entering loop");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) { delay(20); return; }
  if (ws.available()) {
    ws.poll();
  } else if (millis() - lastReconnectMs > 2000) {
    lastReconnectMs = millis();
    wsConnect(); return;
  }

  static uint32_t tVid=0, tMic=0, tIcm=0, tBno=0;
  const uint32_t now = millis();
  if (now - tVid >= TARGET_FRAME_INTERVAL_MS)        { tVid=now; sendJpeg(); }
  if (now - tMic >= MIC_FRAME_MS)                    { tMic=now; sendMicChunk(); }
  if (now - tIcm >= (1000 / ICM_SAMPLE_HZ))          { tIcm=now; sendIcm(); }
  if (now - tBno >= (1000 / BNO_SAMPLE_HZ))          { tBno=now; sendBno(); }
}
