# edge/server/hub.py
# -*- coding: utf-8 -*-
"""
多通道 WebSocket hub（Phase 3 整合版）。

整合變更（aiglass3 → aiglass）：
  - /ws_audio 端點改用 aiglass3 驗證的 ASRCallback + DashScope Paraformer 串流。
  - STOP 指令後，自動用 Groq Whisper 做二次辨識（aiglass3 的 process_voice_file）。
  - 指令路由改由 edge/audio/intent_router.py 統一處理。
  - FSM 事件注入介面保持 aiglass 原有風格（dispatch_intent）。

使用方式：
    uvicorn edge.server.hub:app --host 0.0.0.0 --port 8081
"""

import os
import sys
import time
import json
import asyncio
import base64
from typing import Any, Dict, List, Optional, Set
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# ── aiglass 模組 ──
from edge.utils.config import Config
from edge.utils.throttle import Throttle
from edge.audio.intent_router import route as route_intent
from edge.audio.asr_core import (
    ASRCallback,
    process_voice_file,
    set_current_recognition,
    stop_current_recognition,
)

# ── 環境 ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ───────────────────────────────── FastAPI ─────────────────────────────────
app = FastAPI(title="AIGlass Hub")

# ───────────────────────── DashScope ASR 設定 ─────────────────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
ASR_MODEL = "paraformer-realtime-v2"
SAMPLE_RATE = 16_000
AUDIO_FMT = "pcm"
CHUNK_MS = 20
BYTES_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000 * 2
SILENCE_20MS = bytes(BYTES_CHUNK)

# ───────────────────────────── 全域狀態 ──────────────────────────────────
ui_clients: Dict[int, WebSocket] = {}
camera_viewers: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None
esp32_audio_ws: Optional[WebSocket] = None

last_frames: deque = deque(maxlen=10)
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50

interrupt_lock = asyncio.Lock()

# ── 播報狀態（由 TTS/Omni 模組更新，這裡只需讀取） ──
_is_playing_flag = False


def is_playing_now() -> bool:
    return _is_playing_flag


def set_playing(val: bool) -> None:
    global _is_playing_flag
    _is_playing_flag = val


# ── 最新音檔路徑（由錄製器更新） ──
_latest_audio_path: Optional[str] = None


def set_latest_audio_path(path: str) -> None:
    global _latest_audio_path
    _latest_audio_path = path


def get_latest_audio_path() -> Optional[str]:
    return _latest_audio_path


# ───────────────────────── UI 廣播 helpers ─────────────────────────────

async def ui_broadcast_raw(msg: str) -> None:
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str) -> None:
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)


async def ui_broadcast_final(text: str) -> None:
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[FINAL] {text}", flush=True)


# ───────────────────────── 全域複位 ─────────────────────────────────────

async def full_system_reset(reason: str = "") -> None:
    """系統全清零（移植自 aiglass3/main.py::full_system_reset）"""
    print(f"[SYSTEM RESET] reason={reason}", flush=True)

    await stop_current_recognition()

    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []

    try:
        last_frames.clear()
    except Exception:
        pass

    # 通知 ESP32
    try:
        if esp32_audio_ws and esp32_audio_ws.client_state == WebSocketState.CONNECTED:
            await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass

    print("[SYSTEM RESET] done.", flush=True)


# ───────────────────────── Intent → FSM 分派 ─────────────────────────────

async def dispatch_intent(intent_result: dict) -> None:
    """
    將 intent_router 的結果轉換為 FSM 事件。
    請根據您的 aiglass FSM 實作調整這裡的呼叫方式。
    """
    intent = intent_result.get("intent", "CHAT")
    slot = intent_result.get("slot")
    raw = intent_result.get("raw", "")

    print(f"[DISPATCH] intent={intent} slot={slot}", flush=True)

    # ── 在這裡呼叫您的 FSM ──
    # 範例（請替換為實際的 FSM 模組路徑）：
    # from edge.state_machine.fsm import fsm_instance
    # fsm_instance.on_voice_event(intent, slot=slot, raw=raw)

    # 目前先廣播到 UI 讓您驗證路由是否正確
    from edge.state_machine.fsm import fsm_instance
    fsm_instance.on_voice_event(intent, slot=slot, raw=raw)


# ───────────────────────── 主 on_final_text 回調 ─────────────────────────

async def on_final_text(text: str) -> None:
    """ASRCallback 的 on_final_text_fn：呼叫 intent_router 再分派到 FSM"""
    intent_result = await route_intent(text)
    await dispatch_intent(intent_result)


# ───────────────────────── WebSocket：ESP32 音頻 ─────────────────────────

@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    """
    ESP32 音頻入口（整合 aiglass3 驗證的 ASRCallback）。

    協議：
      ESP32 → server : "START" → 開始 ASR
      ESP32 → server : binary  → PCM16 音頻幀
      ESP32 → server : "STOP"  → 結束 ASR，觸發 Whisper 二次辨識
      ESP32 → server : "PROMPT:<text>" → 直接送文字到 intent_router
    """
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    print("[AUDIO] ESP32 connected", flush=True)

    recognition = None
    streaming = False
    last_ts = time.monotonic()
    keepalive_task: Optional[asyncio.Task] = None
    cb: Optional[ASRCallback] = None  # 保持到 STOP 時可以引用

    async def stop_rec(send_notice: Optional[str] = None) -> None:
        nonlocal recognition, streaming, keepalive_task
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try:
                await keepalive_task
            except Exception:
                pass
        keepalive_task = None
        if recognition:
            try:
                recognition.stop()
            except Exception:
                pass
            recognition = None
        await set_current_recognition(None)
        streaming = False
        if send_notice:
            try:
                await ws.send_text(send_notice)
            except Exception:
                pass

    async def on_sdk_error(msg: str) -> None:
        await stop_rec(send_notice="RESTART")

    async def keepalive_loop() -> None:
        nonlocal last_ts, recognition, streaming
        try:
            while streaming and recognition is not None:
                idle = time.monotonic() - last_ts
                if idle > 0.35:
                    try:
                        for _ in range(30):  # ~600ms 靜音
                            recognition.send_audio_frame(SILENCE_20MS)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("keepalive failed")
                        return
                await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            return

    try:
        while True:
            if ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call" in str(e):
                    break
                raise

            # ── 文字指令 ──
            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    print("[AUDIO] START", flush=True)
                    await stop_rec()

                    loop = asyncio.get_running_loop()

                    def post(coro):
                        asyncio.run_coroutine_threadsafe(coro, loop)

                    # ── 建立 aiglass3 驗證的 ASRCallback ──
                    cb = ASRCallback(
                        on_sdk_error=lambda s: post(on_sdk_error(s)),
                        post=post,
                        ui_broadcast_partial=ui_broadcast_partial,
                        ui_broadcast_final=ui_broadcast_final,
                        is_playing_now_fn=is_playing_now,
                        on_final_text_fn=on_final_text,      # ← 接 intent_router
                        full_system_reset_fn=full_system_reset,
                        interrupt_lock=interrupt_lock,
                    )

                    # ── 啟動 DashScope Paraformer ──
                    try:
                        from dashscope import audio as dash_audio
                        recognition = dash_audio.asr.Recognition(
                            api_key=DASHSCOPE_API_KEY,
                            model=ASR_MODEL,
                            format=AUDIO_FMT,
                            sample_rate=SAMPLE_RATE,
                            callback=cb,
                        )
                        recognition.start()
                        await set_current_recognition(recognition)
                        streaming = True
                        last_ts = time.monotonic()
                        keepalive_task = asyncio.create_task(keepalive_loop())
                        await ui_broadcast_partial("（已開始接收音頻…）")
                        await ws.send_text("OK:STARTED")
                    except Exception as e:
                        print(f"[AUDIO] DashScope 啟動失敗: {e}")
                        # 若無 DashScope，仍保持連線，等待 STOP 後用 Whisper
                        await ws.send_text(f"WARN:NO_DASHSCOPE:{e}")

                elif cmd == "STOP":
                    # 送幾幀靜音讓 Paraformer flush
                    if recognition:
                        for _ in range(15):
                            try:
                                recognition.send_audio_frame(SILENCE_20MS)
                            except Exception:
                                break
                    await stop_rec(send_notice="OK:STOPPED")

                    # ── aiglass3 的 Whisper 二次辨識 ──
                    latest_wav = get_latest_audio_path()
                    if latest_wav and cb is not None:
                        asyncio.create_task(
                            process_voice_file(
                                latest_wav,
                                cb,
                                keyword_filter=None,  # 不過濾，讓 intent_router 處理
                            )
                        )

                elif raw.startswith("PROMPT:"):
                    # ESP32 直接送文字（測試用）
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await on_final_text(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            # ── 音頻幀 ──
            elif "bytes" in msg and msg["bytes"] is not None:
                if streaming and recognition:
                    try:
                        recognition.send_audio_frame(msg["bytes"])
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("send_audio_frame failed")

    except Exception as e:
        print(f"[WS AUDIO ERROR] {e}", flush=True)
    finally:
        await stop_rec()
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[AUDIO] ESP32 disconnected", flush=True)


# ───────────────────────── WebSocket：ESP32 相機 ─────────────────────────

@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    global esp32_camera_ws
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    print("[CAMERA] ESP32 connected", flush=True)

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                last_frames.append((time.time(), data))

                # 推送到瀏覽器觀看者
                dead = set()
                for viewer_ws in list(camera_viewers):
                    try:
                        await viewer_ws.send_bytes(data)
                    except Exception:
                        dead.add(viewer_ws)
                camera_viewers -= dead
    except WebSocketDisconnect:
        pass
    finally:
        if esp32_camera_ws is ws:
            esp32_camera_ws = None
        print("[CAMERA] ESP32 disconnected", flush=True)


# ───────────────────────── WebSocket：瀏覽器觀看 ──────────────────────────

@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        camera_viewers.discard(ws)


# ───────────────────────── WebSocket：UI 狀態推送 ────────────────────────

@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)


# ───────────────────────── HTTP ───────────────────────────────────────────

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"


# ───────────────────────── 啟動入口 ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("edge.server.hub:app", host="0.0.0.0", port=8081, reload=False)