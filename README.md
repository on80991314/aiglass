# AI Smart Glasses

高性價比 AI 智慧輔助眼鏡 — ESP32-S3 硬體 + Python edge + 雲端 LLM/Maps。

兩個獨立 app，共用同一份 vision/audio/state machine 程式碼：
- **find_grab** — 5-state 物件尋找與抓取流程，含麥克風中文 STT、LLM 意圖解析、語音提示、MediaPipe 手部追蹤。
- **full** — 多功能眼鏡 orchestrator：跌倒偵測、行人 NAV、物體位置描述、翻譯。

## 目錄結構

```
ai-smart-glasses/
├── README.md
├── requirements.txt          ← pip install -r requirements.txt
├── pyproject.toml            ← pytest 設定
├── .env / .env.example       ← API 金鑰（GROQ / GEMINI / GMAP）
├── .gitignore                ← 排除 *.pt、.venv、第三方資料夾
│
├── docs/                     ← 規劃文件
├── firmware/                 ← ESP32-S3 韌體 (Arduino IDE)
│   ├── esp32s3_streamer/    Phase 1：純 WebSocket JPEG streamer
│   └── esp32s3_glasses/     Phase 3：整合 IMU + 麥克風 + 喇叭 + SD
│
├── edge/                     ← Python 邊緣端
│   ├── main.py               統一 launcher（subcommand: find_grab | full）
│   ├── apps/
│   │   ├── find_grab.py     尋物 App
│   │   └── full.py          全功能 App
│   ├── audio/
│   │   ├── mic.py           麥克風擷取 + VAD（energy / webrtcvad）
│   │   ├── mic_gate.py      語音播放期靜音閘
│   │   ├── stt.py           本機 faster-whisper
│   │   ├── groq_stt.py      Groq cloud whisper-large-v3
│   │   ├── tts.py           pyttsx3 / edge-tts
│   │   ├── voice_cues.py    預錄 WAV 提示音
│   │   └── intent_router.py 兩階段意圖路由（regex → LLM）
│   ├── vision/
│   │   ├── detector.py      YOLOv8 + EMA hysteresis 穩定器
│   │   ├── hands.py         MediaPipe Hands + 1€ filter
│   │   ├── one_euro.py      1€ 適應性低通濾波
│   │   ├── labels.py        中⇄英物品標籤對照表 + 「想要找 XX」regex
│   │   ├── label_normalizer.py  CN→EN（local + LLM fallback）
│   │   ├── spatial.py       空間描述（左前方/紅綠燈顏色/斑馬線）
│   │   ├── fall_detect.py   IMU 跌倒偵測規則
│   │   └── text_renderer.py CJK 文字繪製到 OpenCV 畫面
│   ├── state_machine/
│   │   ├── find_grab.py    6-state 尋物 FSM（含 CONFIRM_GRAB 確認狀態）
│   │   ├── fsm.py          全功能眼鏡 FSM（IDLE/NAV/FIND/TRANSLATE/FALL）
│   │   └── events.py
│   ├── server/
│   │   ├── ws_server.py    Phase 1 純 JPEG 接收 + 預覽
│   │   └── hub.py          Phase 3 多通道 WebSocket（JPEG / audio / IMU JSON）
│   ├── utils/
│   │   ├── paths.py        REPO/EDGE/MODELS 路徑常數
│   │   ├── dotenv.py       輕量 .env 載入
│   │   ├── config.py       環境變數 → Config dataclass
│   │   ├── logging_setup.py
│   │   └── throttle.py     語音節流（避免重複播報）
│   └── tests/                Smoke tests（webcam_yolo / tts / fall_detect）
│
├── cloud/
│   ├── gemini/gemini.py    Gemini intent + chat
│   └── gmap/maps.py        Google Maps walking directions
│
├── models/                   ← 集中所有模型權重（.gitignored）
│   ├── yolov8n.pt / yolov8s.pt / yolov8m.pt
│   └── hand_landmarker.task
│
└── tests/                    ← 純 Python 單元測試（pytest）
    ├── test_labels.py
    ├── test_intent_router.py
    ├── test_one_euro.py
    └── test_fsm_transitions.py
```

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (bash 風: source .venv/Scripts/activate)
pip install -r requirements.txt
cp .env.example .env             # 填入 GROQ_API_KEY / GEMINI_API_KEY / GMAP_API_KEY
```

需要 ffmpeg 在 PATH 上（給 `edge_tts` mp3 解碼用）。

## 執行

### 尋物模式（最常用）

```bash
# 從 PC webcam，本機 Whisper STT
python edge/main.py find_grab --mic

# 從 ESP32-S3 串流，Groq cloud STT + LLM 意圖 + 語音提示
python edge/main.py find_grab --source ws --mic --stt-backend groq --llm-intent --voice-cues

# 用 webrtcvad 取代能量門檻（更穩，需先 pip install webrtcvad-wheels）
python edge/main.py find_grab --mic --mic-vad webrtc
```

預覽視窗熱鍵：`q`/`ESC` 離開，`t` 模擬說「想要找杯子」，`r` 重置 FSM。

語音指令範例：
- 「想要找杯子」 → 進入 SEARCHING_OBJECT
- 「改找電腦」 → 切換目標
- 「拿到了」 → CONFIRM_GRAB → GRAB_SUCCESS
- 「還沒拿到」 → CONFIRM_GRAB → 退回 GUIDING_HAND
- 「停下／取消」 → 重置

### 全功能模式

```bash
python edge/main.py full
# 從 .env 讀全部設定；ESP32-S3 連 WebSocket hub 後自動啟動
```

## 測試

```bash
pytest                         # 全部單元測試
pytest tests/test_labels.py    # 單一檔
```

## Phase 進度

- ✅ Phase 1 — ESP32-S3 → WebSocket JPEG → PC OpenCV 預覽
- ✅ Phase 2 — 邊緣 YOLOv8 + MediaPipe Hands
- ✅ Phase 3 — STT + TTS + Groq/Gemini intent + 語音提示
- ✅ Phase 3.5 — Find-and-grab FSM with CONFIRM_GRAB confirmation
- 🚧 Phase 4 — IMU 跌倒實機測試、Google Maps 導航整合
