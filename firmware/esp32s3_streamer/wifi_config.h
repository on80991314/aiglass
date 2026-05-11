#pragma once

// ====== Wi-Fi ======
#define WIFI_SSID   "303-5G"
#define WIFI_PASS   "gfourg40"

// ====== WebSocket target (edge PC running edge/server/ws_server.py) ======
#define WS_HOST     "192.168.0.11"   // edge PC LAN IP
#define WS_PORT     8765
#define WS_PATH     "/stream"

// ====== Streaming cadence ======
// 33ms ~= 30 fps upper bound. The actual rate is limited by Wi-Fi / JPEG size.
#define TARGET_FRAME_INTERVAL_MS  33

// ====== Camera orientation (tweak if the picture is upside-down / mirrored) ======
// 0 = no flip, 1 = flip. A 180-degree rotation == both = 1.
#define CAM_VFLIP    0
#define CAM_HMIRROR  1
