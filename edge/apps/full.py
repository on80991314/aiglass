r"""Full multi-mode glasses orchestrator (Phase 3+).

Wires together (with the extended 7-state top-level FSM):
    Hub (WS)  ->  YOLO detector       ->  FSM (FIND/NAV/CROSS_STREET/TRAFFIC_LIGHT) ->  TTS  ->  Hub
         \->   Fall detector  ->  FSM
         \->   STT  ->  IntentRouter (regex+LLM)  ->  FSM

Safe to run with a Phase-1 ESP32-S3 streamer: only the video pipeline
activates; audio / IMU branches stay idle until the corresponding
firmware is in place.

This is the broad-coverage app. For pure find-and-grab use
`apps.find_grab` (5-state object grab FSM, more accurate prompts).

功能整合（aiglass3）:
  過馬路（CROSS_STREET）:
    - 語音說「幫我過馬路」→ IntentRouter 回傳 START_CROSSING
    - FSM 切換到 CROSS_STREET 狀態
    - on_frame 每幀驅動 CrosswalkAwarenessMonitor.process_frame(crosswalk_mask)
    - 回傳 dict 的 voice_text 透過 TTS 播報給使用者
    - 語音說「停止」→ 回到 IDLE

  紅綠燈偵測（TRAFFIC_LIGHT）:
    - 語音說「幫我看紅綠燈」→ IntentRouter 回傳 START_TRAFFIC_LIGHT
    - FSM 切換到 TRAFFIC_LIGHT 狀態
    - 持續偵測紅/黃/綠燈並播報，節流間隔 3 秒
    - 語音說「停止」→ 回到 IDLE
    - 整合自 aiglass3 trafficlight_detection.py 的邏輯

STT 噪音過濾（Bug fix v2）:
  - Whisper 在環境音（電視/廣播/歌詞/字幕聲）下會辨識出不相干文字
  - 加入關鍵字白名單：只有包含指令關鍵字的辨識結果才會進入意圖路由
  - 超過 STT_MAX_CHARS(預設40) 字的文字直接丟棄（通常是歌詞/字幕）
  - 可透過環境變數 STT_STRICT_FILTER=0 關閉過濾（除錯用）
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

# ─────────────────────────────────────────────────────────────────────────────
# STT 噪音過濾設定
# 只有包含以下任一關鍵字的辨識結果才送入意圖路由器。
# 這樣可以過濾掉 Whisper 辨識到的電視聲、歌詞、字幕等不相干文字。
# 設定 STT_STRICT_FILTER=0 可停用過濾（除錯時使用）。
# ─────────────────────────────────────────────────────────────────────────────
_STT_STRICT_FILTER = os.getenv("STT_STRICT_FILTER", "1") == "1"

# 指令關鍵字白名單（繁簡體兼容）
_ALLOWED_KEYWORDS = [
    # 過馬路 / 斑馬線 / 紅綠燈
    "馬路", "马路", "斑馬", "斑马", "過馬", "过马",
    "過馬路", "过马路", "斑馬線", "斑马线",
    "紅綠燈", "红绿灯",
    "紅燈", "红灯",     # ← Bug fix: 原本與「綠燈」間漏逗號導致拼接
    "綠燈", "绿灯",     # ← 已修正
    "紅燈停", "绿灯走", "可以過", "可以通過",
    "看紅綠燈", "看红绿灯", "檢測紅綠燈", "检测红绿灯",
    # 導航 / 盲道
    "開始導航", "开始导航", "幫我導航", "帮我导航", "盲道導航",
    # 找東西
    "幫我找", "帮我找", "尋找", "寻找", "想找",
    # 識別
    "識別", "识别", "幫我看", "帮我看", "這是什麼", "这是什么",
    # 停止 / 取消
    "停止", "停下來", "取消", "結束任務", "算了不", "不用了",
    "別說了", "别说了", "閉嘴", "闭嘴",
    # 確認抓取
    "拿到了", "拿到啦", "抓到了", "找到了", "找到啦", "好了可以",
    # 否定回應
    "還沒拿", "还没拿", "沒拿到", "没拿到",
]

# 單句字數上限：超過此長度通常是環境音/歌詞，直接丟棄
_STT_MAX_CHARS = int(os.getenv("STT_MAX_CHARS", "40"))


def _is_command_text(text: str) -> bool:
    """判斷辨識出的文字是否包含有效指令關鍵字。"""
    if not _STT_STRICT_FILTER:
        return True
    if len(text) > _STT_MAX_CHARS:
        log.info("stt noise filter: too long (%d chars), dropped: %r", len(text), text[:50])
        return False
    for kw in _ALLOWED_KEYWORDS:
        if kw in text:
            return True
    log.info("stt noise filter: no keyword matched, dropped: %r", text)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CrosswalkAwarenessMonitor（從 aiglass3 移植，選擇性載入）
# ─────────────────────────────────────────────────────────────────────────────
_CrosswalkMonitor = None
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from crosswalk_awareness import CrosswalkAwarenessMonitor as _CWM  # type: ignore
    _CrosswalkMonitor = _CWM
    log.info("CrosswalkAwarenessMonitor 載入成功")
except Exception as _e1:
    log.warning("CrosswalkAwarenessMonitor 未能載入，過馬路功能降級為簡易模式: %s", _e1)


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

        # ── 過馬路感知器（aiglass3 CrosswalkAwarenessMonitor）──
        self._cross_nav: Optional[object] = None
        if _CrosswalkMonitor is not None:
            try:
                self._cross_nav = _CrosswalkMonitor()
                log.info("CrosswalkAwarenessMonitor 初始化完成")
            except Exception as e:
                log.warning("CrosswalkAwarenessMonitor 初始化失敗，降級為簡易模式: %s", e)
        else:
            log.info("使用簡易過馬路模式（僅紅綠燈 + 斑馬線提示）")

        # 過馬路：上次播報導引的時間（節流）
        self._last_cross_guidance_t: float = 0.0
        self._cross_guidance_interval: float = float(os.getenv("CROSS_GUIDANCE_INTERVAL_S", "2.5"))

        # ── 紅綠燈偵測（TRAFFIC_LIGHT 狀態，整合自 aiglass3）──────────────────
        # 持續偵測並播報，不走完整過馬路流程。
        # 節流間隔透過 TRAFFIC_LIGHT_INTERVAL_S 環境變數調整（預設 3 秒）。
        self._last_tl_t: float = 0.0
        self._tl_interval: float = float(os.getenv("TRAFFIC_LIGHT_INTERVAL_S", "3.0"))
        # 記錄上次偵測到的顏色，避免連續重複播報同一顏色
        self._last_tl_color: str = ""

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

        # ── CROSS_STREET：每幀驅動 CrosswalkAwarenessMonitor ─────────────────
        if self.fsm.state == State.CROSS_STREET:
            await self._handle_cross_frame(frame, now)
            return

        # ── TRAFFIC_LIGHT：獨立紅綠燈偵測模式（aiglass3 整合）────────────────
        if self.fsm.state == State.TRAFFIC_LIGHT:
            await self._handle_traffic_light_frame(frame, now)
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

    async def _handle_traffic_light_frame(self, frame: np.ndarray, now: float) -> None:
        """TRAFFIC_LIGHT 狀態：持續偵測紅綠燈並播報。
        
        整合自 aiglass3 trafficlight_detection.py 的邏輯：
        - 偵測 red / yellow / green 三種狀態
        - 節流（_tl_interval，預設 3 秒）避免過度播報
        - 同一顏色連續出現不重複播報，直到顏色變化
        """
        if (now - self._last_tl_t) < self._tl_interval:
            return  # 節流

        dets = await asyncio.to_thread(self.detector.infer, frame)

        detected_color = ""
        for d in dets:
            if d.label == "traffic light":
                color = traffic_light_color(frame, d)
                if color in ("red", "green", "yellow"):
                    detected_color = color
                    break  # 取第一個偵測到的燈

        if not detected_color:
            return  # 沒有偵測到任何燈，不播報

        # 顏色沒有變化則不重複播報
        if detected_color == self._last_tl_color:
            return

        self._last_tl_color = detected_color
        self._last_tl_t = now

        if detected_color == "red":
            self._queue_say("紅燈，請停下等候", key="tl:red")
        elif detected_color == "green":
            self._queue_say("綠燈，可以通行", key="tl:green")
        elif detected_color == "yellow":
            self._queue_say("黃燈，請注意減速", key="tl:yellow")

    async def _handle_cross_frame(self, frame: np.ndarray, now: float) -> None:
        """處理過馬路狀態下的每一幀。

        有 CrosswalkAwarenessMonitor → 走完整斑馬線感知流程。
        沒有 → 簡易模式：用 YOLO 偵測紅綠燈 + crosswalk_hint。
        """
        dets = await asyncio.to_thread(self.detector.infer, frame)
        h, w = frame.shape[:2]

        if self._cross_nav is not None:
            try:
                crosswalk_mask: Optional[np.ndarray] = None
                for d in dets:
                    if d.label in ("crosswalk", "zebra crossing", "zebra_crossing"):
                        x1 = max(0, int(d.box[0]))
                        y1 = max(0, int(d.box[1]))
                        x2 = min(w, int(d.box[2]))
                        y2 = min(h, int(d.box[3]))
                        if crosswalk_mask is None:
                            crosswalk_mask = np.zeros((h, w), dtype=np.uint8)
                        crosswalk_mask[y1:y2, x1:x2] = 255
                        break

                result = await asyncio.to_thread(
                    self._cross_nav.process_frame, crosswalk_mask
                )
            except Exception as e:
                log.warning("cross_nav.process_frame 失敗: %s", e)
                result = None

            if result and result.get("should_broadcast"):
                guidance = result.get("voice_text", "") or ""
                guidance = to_traditional(guidance)
                if guidance and (now - self._last_cross_guidance_t) >= self._cross_guidance_interval:
                    self._queue_say(guidance, key=f"cross:{guidance}")
                    self._last_cross_guidance_t = now

        else:
            try:
                from vision.spatial import crosswalk_hint
                hint = crosswalk_hint(frame)
                if hint:
                    self._queue_say(hint, key="crosswalk")
            except Exception:
                pass

        # 不管哪個模式，都偵測紅綠燈
        for d in dets:
            if d.label == "traffic light":
                color = traffic_light_color(frame, d)
                if color == "red":
                    self._queue_say("紅燈，請稍候", key="tl:red")
                elif color == "green":
                    self._queue_say("綠燈，可以通過", key="tl:green")

    async def on_audio(self, peer: str, pcm: np.ndarray, sr: int) -> None:
        energy = float(np.abs(pcm).mean())
        now = time.monotonic()

        if not hasattr(self, "_vad"):
            self._vad: dict = {}
        state = self._vad.setdefault(peer, {
            "recording": False,
            "buf": [],
            "last_voice_t": 0.0,
            "start_t": 0.0,
        })

        ON  = int(os.getenv("VOICE_ENERGY_ON",  "600"))
        OFF = int(os.getenv("VOICE_ENERGY_OFF", "200"))
        MIN_S     = float(os.getenv("VOICE_MIN_S",     "0.4"))
        MAX_S     = float(os.getenv("VOICE_MAX_S",     "6.0"))
        SILENCE_S = float(os.getenv("VOICE_SILENCE_S", "0.5"))

        has_voice = energy >= ON
        is_silent = energy < OFF

        if has_voice:
            if not state["recording"]:
                state["recording"] = True
                state["buf"] = []
                state["start_t"] = now
                log.debug("vad: start recording (energy=%.0f)", energy)
            state["buf"].append(pcm)
            state["last_voice_t"] = now

        elif state["recording"]:
            state["buf"].append(pcm)

            silence_s = now - state["last_voice_t"]
            duration_s = now - state["start_t"]

            should_send = (
                (is_silent and silence_s >= SILENCE_S and duration_s >= MIN_S)
                or duration_s >= MAX_S
            )

            if should_send:
                audio = np.concatenate(state["buf"])
                state["recording"] = False
                state["buf"] = []
                log.debug("vad: send %.1fs audio (energy=%.0f)", duration_s, energy)
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

        if not _is_command_text(text):
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

        # ── 取消（含各功能專屬停止指令）────────────────────────────────────
        if kind in ("CANCEL", "STOP_NAV", "STOP_CROSSING", "STOP_TRAFFIC_LIGHT"):
            if self.fsm.state == State.CROSS_STREET and self._cross_nav is not None:
                try:
                    self._cross_nav.reset()
                except Exception:
                    pass
            if self.fsm.state == State.TRAFFIC_LIGHT:
                self._last_tl_color = ""  # 重置偵測記錄
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
                self.fsm._goto(State.CROSS_STREET, "過馬路模式已啟動")
            return

        # ── 啟動紅綠燈偵測（aiglass3 整合）────────────────────────────────
        if kind == "START_TRAFFIC_LIGHT":
            self._last_tl_color = ""  # 重置，確保首次偵測一定播報
            self._last_tl_t = 0.0
            self.fsm._goto(State.TRAFFIC_LIGHT, "已啟動紅綠燈偵測")
            return

        # ── 找物品 ───────────────────────────────────────────────────────────
        if kind == "FIND":
            obj_zh = intent.target_zh or ""
            self.fsm.ctx.target_object = intent.target_en or obj_zh
            self.fsm._goto(State.FIND, f"正在尋找{obj_zh}" if obj_zh else "正在尋找目標")
            return

        if kind == "FOUND":
            self.fsm._goto(State.IDLE, "好的，已找到")
            return

        # ── 導航 ─────────────────────────────────────────────────────────────
        if kind == "START_BLINDPATH_NAV":
            self.fsm._goto(State.NAV, "開始導航")
            return

        # ── 視覺問答（aiglass3 整合）────────────────────────────────────────
        if kind == "VISUAL_QUERY":
            await self._visual_query()
            return

        # ── 閒聊回覆（LLM）──────────────────────────────────────────────────
        if kind == "NONE":
            await self._llm_chat(text)
            return

    async def _visual_query(self) -> None:
        """視覺問答：拍一幀，送 VLM（Gemini）描述畫面內容。
        
        整合自 aiglass3 的視覺識別功能。
        需要 GEMINI_API_KEY 設定。
        """
        if self._gemini is None:
            self._queue_say("視覺識別功能未啟用，請設定 Gemini API Key")
            return
        try:
            # 從最新收到的 frame 取一幀（如果有的話）
            # 注意：這裡需要 App 層保存最後一幀，目前以簡單版實現
            if not hasattr(self, "_latest_frame") or self._latest_frame is None:
                self._queue_say("目前沒有影像畫面")
                return
            frame = self._latest_frame
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_bytes = buf.tobytes()
            description = await asyncio.to_thread(
                self._gemini.describe_image, img_bytes,
                "請用繁體中文簡短描述這張圖片中最重要的內容（30字以內）"
            )
            if description:
                self._queue_say(description[:200], key="visual_query")
        except Exception as e:
            log.warning("visual_query 失敗: %s", e)
            self._queue_say("無法識別畫面")

    async def on_frame(self, peer: str, frame: np.ndarray) -> None:  # type: ignore[override]
        # 保存最新幀供視覺問答使用
        self._latest_frame = frame
        await super().on_frame(peer, frame)  # type: ignore[misc]

    async def _llm_chat(self, text: str) -> None:
        """使用 Groq Llama 回答非指令的自然對話。"""
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            return
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
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
