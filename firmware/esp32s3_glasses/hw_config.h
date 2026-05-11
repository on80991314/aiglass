#pragma once

// ============================================================
// XIAO ESP32-S3 Sense 實際元件腳位
// 依照 correct_pin.ino (v2.4-SPIIMU) 修訂：PDM mic 改加 DC offset
// 校正 + 放大 + warmup；MAX98357 喇叭腳位改成 D7/D8/D9 + D10 GAIN。
// ============================================================

// -------- I2C 共享總線 (ICM42688 + BNO055) --------
#define I2C_SDA_PIN        5
#define I2C_SCL_PIN        6
#define ICM42688_I2C_ADDR  0x68   // AD0 接 GND
#define BNO055_I2C_ADDR    0x28   // ADR 接 GND

// -------- PDM 麥克風 (XIAO Sense 板載 MSM261) --------
// 腳位 42/41 不變；訊號處理改在 esp32s3_glasses.ino 加 DC offset 校
// 正 + 2× 放大 + clip + warmup（先前未做這些步驟，直流偏移把訊號
// 整個吃掉，Whisper 才會辨識不出）。
#define MIC_I2S_PORT       I2S_NUM_0
#define MIC_PDM_CLK        42     // PDM CLK
#define MIC_PDM_DATA       41     // PDM DATA
#define MIC_SAMPLE_RATE    16000
#define MIC_FRAME_MS       20     // correct_pin.ino 用 20ms 一包，較穩

// -------- I2S 喇叭 (MAX98357A) --------
// 依 correct_pin.ino 改用 D7/D8/D9 + D10 GAIN 並改成 32-bit STEREO 輸出。
// XIAO ESP32-S3 Sense D# -> GPIO 對照：
//   D7 = GPIO 44, D8 = GPIO 7, D9 = GPIO 8, D10 = GPIO 9
//
// ⚠️ 這組腳位會與 XIAO Sense 擴充板的 SD 卡 (SPI on GPIO 7/8/9) 衝
//    突。為了讓喇叭可用，本版本「停用 SD 卡」（esp32s3_glasses.ino
//    內的 initSd 直接 return false）。若日後要同時使用 SD + 喇叭，
//    需換用 I2C SD 模組或 GPIO mux。
#define SPK_I2S_PORT       I2S_NUM_1
#define SPK_BCLK_PIN       7      // D8
#define SPK_LRC_PIN        44     // D7
#define SPK_DIN_PIN        8      // D9
#define SPK_GAIN_PIN       9      // D10  懸空 = 9dB；接 GND = 12dB；接 VIN = 15dB
#define SPK_SAMPLE_RATE    16000

// -------- MicroSD（已停用：腳位被 MAX98357 佔用） --------
// 若你不接喇叭、想用 SD，請把上面 SPK_* 改回 GPIO 2/3/4 並重新啟
// 用下面這四個。
// #define SD_CS_PIN          21
// #define SD_SCK_PIN         7
// #define SD_MOSI_PIN        9
// #define SD_MISO_PIN        8

// -------- IMU 採樣頻率 --------
#define ICM_SAMPLE_HZ      100    // 落下偵測用 (ICM42688)
#define BNO_SAMPLE_HZ      20     // 頭部朝向用 (BNO055, 絕對 Euler)
