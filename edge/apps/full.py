r"""Full multi-mode glasses orchestrator (Phase 3+).

Wires together (with the original 5-state top-level FSM):
    Hub (WS)  ->  YOLO detector       ->  FSM (FIND/NAV/CROSS_STREET) ->  TTS  ->  Hub
         \->   Fall detector  ->  FSM
         \->   STT  ->  IntentRouter (regex+LLM)  ->  FSM

Safe to run with a Phase-1 ESP32-S3 streamer: only the video pipeline
activates; audio / IMU branches stay idle until the corresponding
firmware is in place.

This is the broad-coverage app. For pure find-and-grab use
`apps.find_grab` (5-state object grab FSM, more accurate prompts).

過馬路整合（aiglass3 CrossStreetNavigator 移植）:
  - 語音說「幫我過馬路」→ IntentRouter 回傳 START_CROSSING
  - FSM 切換到 CROSS_STREET 狀態
  - on_frame 每幀驅動 CrossStreetNavigator.process_frame()
  - 結果的 guidance_text 透過 TTS 播報給使用者
  - 語音說「停止」→ 回到 IDLE
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import Optional

import cv2
import numpy as np
import os

from audio.groq_stt import GroqWhisperSTT
from audio.intent_router import IntentRouter
from audio.tts import TTS
from audio.zh_normalizer import to_traditional
from server.hub import Hub
from state_machine.events import State
from state_machine.fsm import GlassesFSM
from utils.config import Config, load
from utils.throttle import SpeechThrottle
from vision.detector import YoloDetector
from vision.fall_detect import FallDetector, ImuSample
from vision.spatial import describe_position, traffic_light_color

log = logging.getLogger("full")

# ── CrossStreetNavigator（從 aiglass3 移植，選擇性載入）──────────────────────
_CrossStreetNavigator = None
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from crosswalk_awareness import CrossStreetNavigator as _CSNav  # type: ignore
    _CrossStreetNavigator = _CSNav
    log.info("CrossStreetNavigator 載入成功（crosswalk_awareness）")
except Exception:
    try:
        # 回退：嘗試直接從 aiglass3 同目錄的 workflow_crossstreet 載入
        from workflow_crossstreet import CrossStreetNavigator as _CSNav2  # type: ignore
        _CrossStreetNavigator = _CSNav2
        log.info("CrossStreetNavigator 載入成功（workflow_crossstreet）")
    except Exception as e:
        log.warning("CrossStreetNavigator 未能載入，過馬路功能降級為簡易模式: %s", e)


class App:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.hub = Hub(on_frame=self.on_frame, on_audio=self.on_audio, on_json=self.on_json)
        self.detector = YoloDetector(cfg.yolo_weights, cfg.yolo_device, cfg.yolo_conf)
        self.fall = FallDetector()
        self.stt = GroqWhisperSTT()
        self.tts = TTS(cfg.tts_backend, cfg.tts_voice)
        self.throttle = SpeechThrottle(cfg.speech_min_gap_s, cfg.speech_repeat_gap_s)
        self.fsm = GlassesFSM(say=self._queue_say)
        self._say_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._last_frame_t = 0.0
        self._audio_buf: dict[str, list[np.ndarray]] = {}
        self._last_audio_voice_t: dict[str, float] = {}

        # ── Gemini（選擇性）──
        self._gemini = None
        if cfg.gemini_api_key:
            from cloud.gemini.gemini import GeminiClient
            self._gemini = GeminiClient(cfg.gemini_api_key, cfg.gemini_model)

        # ── Google Maps（選擇性）──
        self._gmap = None
        if cfg.gmap_api_key:
            from cloud.gmap.maps import GmapClient
            self._gmap = GmapClient(cfg.gmap_api_key, cfg.gmap_language)

        # ── IntentRouter（regex 優先，LLM 選擇性）──
        groq_key = os.getenv("GROQ_API_KEY", "")
        self.router = IntentRouter(
            use_llm=bool(groq_key),
            llm_provider="groq",
            api_key=groq_key or None,
        )

        # ── 過馬路導航器（aiglass3 CrossStreetNavigator）──
        self._cross_nav: Optional[object] = None
        if _CrossStreetNavigator is not None:
            try:
                # seg_model 使用 cfg 中的 yolo_weights（若有獨立斑馬線模型可另設）
                seg_model_path = getattr(cfg, "crosswalk_model", None) or getattr(cfg, "yolo_weights", None)
                if seg_model_path and os.path.exists(seg_model_path):
                    from ultralytics import YOLO  # type: ignore
                    seg_model = YOLO(seg_model_path)
                    self._cross_nav = _CrossStreetNavigator(seg_model=seg_model)
                    log.info("CrossStreetNavigator 初始化完成，模型: %s", seg_model_path)
                else:
                    self._cross_nav = _CrossStreetNavigator()
                    log.info("CrossStreetNavigator 初始化完成（無 seg_model）")
            except Exception as e:
                log.warning("CrossStreetNavigator 初始化失敗，降級為簡易模式: %s", e)
        else:
            log.info("使用簡易過馬路模式（僅紅綠燈 + 斑馬線提示）")

        # 過馬路：上次播報導引的時間（節流）
        self._last_cross_guidance_t: float = 0.0
        self._cross_guidance_interval: float = float(os.getenv("CROSS_GUIDANCE_INTERVAL_S", "2.5"))

    # ========= run =========
    async def run(self) -> None:
        tasks = [
            asyncio.create_task(self.hub.serve(self.cfg.ws_host, self.cfg.ws_port)),
            asyncio.create_task(self._speak_loop()),
        ]
        await asyncio.gather(*tasks)

    # ========= callbacks =========
    async def on_frame(self, peer: str, frame: np.ndarray) -> None:
        cv2.imshow("ESP32 Glasses View", frame)
        cv2.waitKey(1)

        # 節流：~6 fps
        now = time.monotonic()
        if now - self._last_frame_t < 0.15:
            return
        self._last_frame_t = now

        h, w = frame.shape[:2]

        # ── CROSS_STREET：每幀驅動 CrossStreetNavigator ──────────────────────
        if self.fsm.state == State.CROSS_STREET:
            await self._handle_cross_frame(frame, now)
            return

        # ── 其他狀態：使用原有 YOLO 偵測 ──────────────────────────────────────
        dets = await asyncio.to_thread(self.detector.infer, frame)

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

    async def _handle_cross_frame(self, frame: np.ndarray, now: float) -> None:
        """處理過馬路狀態下的每一幀。

        有 CrossStreetNavigator → 走完整斑馬線對齊 + 紅綠燈流程。
        沒有 → 簡易模式：用 YOLO 偵測紅綠燈 + 原有 crosswalk_hint。
        """
        if self._cross_nav is not None:
            # ── 完整模式（aiglass3 移植）────────────────────────────────────
            try:
                result = await asyncio.to_thread(self._cross_nav.process_frame, frame)
            except Exception as e:
                log.warning("cross_nav.process_frame 失敗: %s", e)
                return

            guidance = getattr(result, "guidance_text", "") or ""
            if guidance and (now - self._last_cross_guidance_t) >= self._cross_guidance_interval:
                self._queue_say(guidance, key=f"cross:{guidance}")
                self._last_cross_guidance_t = now

        else:
            # ── 簡易模式（fallback）────────────────────────────────────────
            dets = await asyncio.to_thread(self.detector.infer, frame)
            for d in dets:
                if d.label == "traffic light":
                    color = traffic_light_color(frame, d)
                    if color == "red":
                        self._queue_say("紅燈，請稍候", key="tl:red")
                    elif color == "green":
                        self._queue_say("綠燈，可以通過", key="tl:green")

            # crosswalk_hint（原有簡易斑馬線偵測）
            try:
                from vision.spatial import crosswalk_hint
                hint = crosswalk_hint(frame)
                if hint:
                    self._queue_say(hint, key="crosswalk")
            except Exception:
                pass

    async def on_audio(self, peer: str, pcm: np.ndarray, sr: int) -> None:
        buf = self._audio_buf.setdefault(peer, [])
        buf.append(pcm)

        energy = float(np.abs(pcm).mean())
        now = time.monotonic()
        if energy > 400:
            self._last_audio_voice_t[peer] = now

        total_samples = sum(len(x) for x in buf)
        silence_s = now - self._last_audio_voice_t.get(peer, now)
        if total_samples > sr * 10 or (total_samples > sr * 0.5 and silence_s > 0.8):
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
            log.info("device event: %s", obj.get("name", ""))

    # ========= voice handler =========
    async def _handle_voice(self, pcm: np.ndarray, sr: int) -> None:
        try:
            text = await asyncio.to_thread(self.stt.transcribe_pcm, pcm, sr)
        except Exception as e:
            log.warning("stt 失敗: %s", e)
            return

        text = to_traditional(text.strip())
        if not text:
            return
        log.info("heard: %s", text)

        intent = self.router.route(text)
        log.info("intent: %s", intent)

        kind = intent.kind

        # ── 緊急停止 ─────────────────────────────────────────────────────────
        if kind == "HOTWORD":
            if self.fsm.state == State.CROSS_STREET and self._cross_nav is not None:
                try:
                    self._cross_nav.reset()
                except Exception:
                    pass
            self.fsm._goto(State.IDLE, "已停止")
            return

        # ── 取消 ─────────────────────────────────────────────────────────────
        if kind in ("CANCEL", "STOP_NAV", "STOP_CROSSING", "STOP_TRAFFIC_LIGHT"):
            if self.fsm.state == State.CROSS_STREET and self._cross_nav is not None:
                try:
                    self._cross_nav.reset()
                except Exception:
                    pass
            self.fsm._goto(State.IDLE, "已取消")
            return

        # ── 開始過馬路 ───────────────────────────────────────────────────────
        if kind == "START_CROSSING":
            if self._cross_nav is not None:
                try:
                    self._cross_nav.reset()
                except Exception:
                    pass
                self.fsm._goto(State.CROSS_STREET, "過馬路模式已啟動，正在尋找斑馬線")
            else:
                # 簡易模式：直接切到 CROSS_STREET，由 fallback 處理
                self.fsm._goto(State.CROSS_STREET, "過馬路模式已啟動")
            return

        # ── 找物品 ───────────────────────────────────────────────────────────
        if kind == "FIND":
            obj_zh = intent.target_zh or ""
            self.fsm.ctx.target_object = intent.target_en or obj_zh
            self.fsm._goto(State.FIND, f"正在尋找{obj_zh}" if obj_zh else "正在尋找目標")
            return

        # ── 抓到了 ───────────────────────────────────────────────────────────
        if kind == "FOUND":
            self.fsm._goto(State.IDLE, "好的，已找到")
            return

        # ── 導航 ─────────────────────────────────────────────────────────────
        if kind == "START_BLINDPATH_NAV":
            self.fsm._goto(State.NAV, "開始導航")
            return

        # ── 閒聊回覆（LLM）──────────────────────────────────────────────────
        if kind == "NONE":
            await self._llm_chat(text)
            return

    async def _llm_chat(self, text: str) -> None:
        """使用 Groq Llama 回答非指令的自然對話。"""
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            return
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            import json as _json
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是智慧眼鏡的對話助手，回答請簡短（30 字以內），使用台灣繁體中文。"
                            "只輸出純文字，不要任何 JSON 或 Markdown。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=80,
            )
            reply = (response.choices[0].message.content or "").strip()
            if reply:
                self._queue_say(reply[:200])
        except Exception as e:
            log.warning("LLM chat 失敗: %s", e)

    # ========= nav =========
    async def _start_nav(self, destination: str) -> None:
        if not self._gmap or not destination:
            return
        origin = self.cfg.extras.get("origin") or "current location"
        try:
            steps = await self._gmap.walk_directions(origin, destination)
        except Exception as e:
            log.warning("gmap 失敗: %s", e)
            self._queue_say("路徑規劃失敗")
            return
        step = self._gmap.pick_next(steps)
        if step:
            self._queue_say(step.instruction_zh)

    async def _notify_emergency(self, peer: str) -> None:
        contact = self.cfg.emergency_contact
        if not contact:
            return
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
                log.warning("tts 失敗: %s", e)
                continue
            log.info("say: %s", text)
            for peer in list(self.hub.sessions.keys()):
                try:
                    await self.hub.send_tts_pcm(peer, pcm, self.tts.sample_rate)
                except Exception as e:
                    log.warning("send_tts_pcm to %s 失敗: %s", peer, e)


def add_args(_p: argparse.ArgumentParser) -> None:
    """Full app reads everything from .env via utils/config.py — no CLI flags."""


def run(_args: argparse.Namespace) -> None:
    cfg = load()
    app = App(cfg)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
