r"""Full multi-mode glasses orchestrator (Phase 3+).

Wires together (with the original 5-state top-level FSM):
    Hub (WS)  ->  YOLO detector       ->  FSM (FIND/NAV) ->  TTS  ->  Hub
         \->   Fall detector  ->  FSM
         \->   STT        ->  Groq intent  ->  FSM

Safe to run with a Phase-1 ESP32-S3 streamer: only the video pipeline
activates; audio / IMU branches stay idle until the corresponding
firmware is in place.

This is the broad-coverage app. For pure find-and-grab use
`apps.find_grab` (5-state object grab FSM, more accurate prompts).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import json  # 🟢 引入 JSON 處理庫
from typing import Optional

import cv2
import numpy as np

import os
from groq import Groq

from audio.stt import WhisperSTT
from audio.tts import TTS
from audio.zh_normalizer import normalize_obj, to_traditional
from server.hub import Hub
from state_machine.events import Intent, State
from state_machine.fsm import GlassesFSM
from utils.config import Config, load
from utils.throttle import SpeechThrottle
from vision.detector import YoloDetector
from vision.fall_detect import FallDetector, ImuSample
from vision.spatial import (crosswalk_hint, describe_position,
                            traffic_light_color)

log = logging.getLogger("full")


class App:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.hub = Hub(on_frame=self.on_frame, on_audio=self.on_audio, on_json=self.on_json)
        self.detector = YoloDetector(cfg.yolo_weights, cfg.yolo_device, cfg.yolo_conf)
        self.fall = FallDetector()
        self.stt = WhisperSTT(cfg.stt_model, language=cfg.stt_language)
        self.tts = TTS(cfg.tts_backend, cfg.tts_voice)
        self.throttle = SpeechThrottle(cfg.speech_min_gap_s, cfg.speech_repeat_gap_s)
        self.fsm = GlassesFSM(say=self._queue_say)
        self._say_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._last_frame_t = 0.0
        self._audio_buf: dict[str, list[np.ndarray]] = {}
        self._last_audio_voice_t: dict[str, float] = {}
        self._gemini = None
        if cfg.gemini_api_key:
            from cloud.gemini.gemini import GeminiClient
            self._gemini = GeminiClient(cfg.gemini_api_key, cfg.gemini_model)
        
        # 🟢 新增：初始化 Groq 用戶端（直接從環境變數讀取金鑰）
        self.groq_client = None
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
        else:
            log.warning("環境變數中未偵測到 GROQ_API_KEY！")

        self._gmap = None
        if cfg.gmap_api_key:
            from cloud.gmap.maps import GmapClient
            self._gmap = GmapClient(cfg.gmap_api_key, cfg.gmap_language)

    # ========= run =========
    async def run(self) -> None:
        tasks = [
            asyncio.create_task(self.hub.serve(self.cfg.ws_host, self.cfg.ws_port)),
            asyncio.create_task(self._speak_loop()),
        ]
        await asyncio.gather(*tasks)

    # ========= callbacks =========
    async def on_frame(self, peer: str, frame: np.ndarray) -> None:
        # throttle YOLO to ~6 fps so a laptop CPU keeps up

        cv2.imshow("ESP32 Glasses View", frame)
        cv2.waitKey(1)

        now = time.monotonic()
        if now - self._last_frame_t < 0.15:
            return
        self._last_frame_t = now

        dets = await asyncio.to_thread(self.detector.infer, frame)
        h, w = frame.shape[:2]

        if self.fsm.state == State.FIND and self.fsm.ctx.target_object:
            best = self.detector.best(dets, self.fsm.ctx.target_object)
            if best is not None:
                msg = describe_position(best, w, h, self.fsm.ctx.target_object)
                self._queue_say(msg, key=f"find:{self.fsm.ctx.target_object}")
        elif self.fsm.state == State.NAV:
            for d in dets:
                if d.label == "traffic light":
                    color = traffic_light_color(frame, d)
                    if color == "red":
                        self._queue_say("紅燈，請稍候", key="tl:red")
                    elif color == "green":
                        self._queue_say("綠燈，請小心通過", key="tl:green")
            hint = crosswalk_hint(frame)
            if hint:
                self._queue_say(hint, key="crosswalk")

    async def on_audio(self, peer: str, pcm: np.ndarray, sr: int) -> None:
        buf = self._audio_buf.setdefault(peer, [])
        buf.append(pcm)

        # simple energy VAD - collect until ~1s of silence then transcribe
        energy = float(np.abs(pcm).mean())
        now = time.monotonic()
        if energy > 300:
            self._last_audio_voice_t[peer] = now

        total_samples = sum(len(x) for x in buf)
        silence_s = now - self._last_audio_voice_t.get(peer, now)
        if total_samples > sr * 10 or (total_samples > sr * 1 and silence_s > 1.0):
            audio = np.concatenate(buf)
            buf.clear()
            self._last_audio_voice_t.pop(peer, None)
            asyncio.create_task(self._handle_voice(audio, sr))

    async def on_json(self, peer: str, obj: dict) -> None:
        kind = obj.get("type")
        if kind == "imu":
            s = ImuSample(
                t=float(obj.get("t", time.monotonic())),
                ax=float(obj.get("ax", 0.0)),
                ay=float(obj.get("ay", 0.0)),
                az=float(obj.get("az", 9.8)),
                gx=float(obj.get("gx", 0.0)),
                gy=float(obj.get("gy", 0.0)),
                gz=float(obj.get("gz", 0.0)),
            )
            if self.fall.push(s):
                self.fsm.on_fall()
                await self._notify_emergency(peer)
        elif kind == "event":
            name = obj.get("name", "")
            log.info("device event: %s", name)

    # ========= handlers =========
    async def _handle_voice(self, pcm: np.ndarray, sr: int) -> None:
        try:
            text = await asyncio.to_thread(self.stt.transcribe_pcm, pcm, sr)
        except Exception as e:
            log.warning("stt failed: %s", e)
            return
        text = to_traditional(text.strip())
        if not text:
            return
        log.info("heard: %s", text)

        # 🟢 修改：改用 Groq 的 Llama-3.3 進行結構化意圖解析與閒聊回覆
        if self.groq_client is None:
            self._queue_say("尚未設定 Groq 金鑰，無法理解語音指令")
            return

        try:
            # 透過執行緒調用 Groq，避免阻塞 asyncio 異步主流程
            response = await asyncio.to_thread(
                self.groq_client.chat.completions.create,
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一個智慧眼鏡的意圖分析與對話助手。請分析使用者的語音輸入，並嚴格回傳以下格式的 JSON 物件：\n"
                            "{\n"
                            "  \"intent\": \"NAV_TO\" 或 \"FIND_OBJECT\" 或 \"TRANSLATE_TO\" 或 \"CANCEL\" 或 \"CHAT\",\n"
                            "  \"destination\": \"導航目的地(若無則為空字串)\",\n"
                            "  \"object\": \"尋找的物品(若無則為空字串)\",\n"
                            "  \"lang\": \"翻譯語言(預設為 'zh-TW')\",\n"
                            "  \"reply\": \"當意圖為 CHAT 時，請在此提供親切的回答（台灣繁體中文，且控制在 30 個字以內）。其他意圖時此欄位請留空。\"\n"
                            "}"
                        )
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            intent = json.loads(response.choices[0].message.content)
            intent = normalize_obj(intent)
        except Exception as e:
            log.warning("Groq 大腦調用或解析失敗: %s", e)
            self._queue_say("抱歉，我的大腦連線發生錯誤")
            return

        name = (intent.get("intent") or "CHAT").upper()

        if name == "NAV_TO":
            self.fsm.on_intent(Intent.NAV_TO, {"destination": intent.get("destination", "")})
            await self._start_nav(intent.get("destination", ""))
        elif name == "FIND_OBJECT":
            self.fsm.on_intent(Intent.FIND_OBJECT, {"object": intent.get("object", "")})
        elif name == "TRANSLATE_TO":
            self.fsm.on_intent(Intent.TRANSLATE_TO, {"lang": intent.get("lang", "zh-TW")})
        elif name == "CANCEL":
            self.fsm.on_intent(Intent.CANCEL, {})
        else:
            # 🟢 這裡承接了 Groq 即時產生的對話回覆
            reply = intent.get("reply")
            if reply:
                self._queue_say(reply[:200])

    async def _start_nav(self, destination: str) -> None:
        if not self._gmap or not destination:
            return
        # Origin must come from a GPS feed. Phase 3 placeholder uses IP geolocation
        # or a manually-set origin; swap to a real GPS source in Phase 4.
        origin = self.cfg.extras.get("origin") or "current location"
        try:
            steps = await self._gmap.walk_directions(origin, destination)
        except Exception as e:
            log.warning("gmap failed: %s", e)
            self._queue_say("路徑規劃失敗")
            return
        step = self._gmap.pick_next(steps)
        if step:
            self._queue_say(step.instruction_zh)

    async def _notify_emergency(self, peer: str) -> None:
        contact = self.cfg.emergency_contact
        if not contact:
            log.warning("EMERGENCY_CONTACT not set; skipping notify")
            return
        # TODO: integrate your carrier of choice (Line Notify, Telegram, Twilio).
        log.warning("EMERGENCY: would notify %s that %s fell.", contact, peer)

    # ========= TTS =========
    def _queue_say(self, text: str, key: Optional[str] = None) -> None:
        if not text:
            return
        if not self.throttle.allow(key or text):
            return
        self._say_queue.put_nowait((text, key or text))

    async def _speak_loop(self) -> None:
        while True:
            text, _key = await self._say_queue.get()
            try:
                pcm = await self.tts.synthesize(text)
            except Exception as e:
                log.warning("tts failed: %s", e)
                continue
            log.info("say: %s", text)
            for peer in list(self.hub.sessions.keys()):
                try:
                    await self.hub.send_tts_pcm(peer, pcm, self.tts.sample_rate)
                except Exception as e:
                    log.warning("send_tts_pcm to %s failed: %s", peer, e)


def add_args(_p: argparse.ArgumentParser) -> None:
    """Full app reads everything from .env via utils/config.py — no CLI flags."""


def run(_args: argparse.Namespace) -> None:
    cfg = load()
    app = App(cfg)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass