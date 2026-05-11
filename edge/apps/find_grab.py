"""Find-and-grab pipeline (extracted as an importable App).

Two video sources:
   --source webcam         (default, cv2.VideoCapture(0))
   --source ws             (receive JPEG frames from ESP32-S3 over WebSocket)

The FSM, YOLO and MediaPipe always run on the edge.
STT is from the PC microphone (or simulated by pressing 't' in the window).

Hotkeys in the preview window:
   q / ESC   quit
   t         simulate STT: 「想要找杯子」
   r         hard reset back to WAITING_FOR_COMMAND
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from typing import List, Optional

import cv2
import numpy as np

from audio.mic import MicListener
from audio.mic_gate import MicGate
from audio.zh_normalizer import to_traditional
from state_machine.find_grab import FGState, FindGrabConfig, FindGrabFSM
from vision.detector import Detection, YoloDetector
from vision.hands import HandResult, HandsDetector
from vision.text_renderer import put_text as cjk_text

log = logging.getLogger("find_grab")


# ---- module-level state shared between video producer + main loop ----
_latest_frame: Optional[np.ndarray] = None
_latest_frame_lock = threading.Lock()
_stop_flag = threading.Event()


def _set_latest(frame: np.ndarray) -> None:
    global _latest_frame
    with _latest_frame_lock:
        _latest_frame = frame


def _get_latest() -> Optional[np.ndarray]:
    with _latest_frame_lock:
        return None if _latest_frame is None else _latest_frame.copy()


# ============================================================
# Video sources
# ============================================================
def _webcam_producer(index: int = 0) -> None:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log.error("webcam open failed (index=%d)", index)
        _stop_flag.set()
        return
    try:
        while not _stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            _set_latest(frame)
    finally:
        cap.release()


async def _ws_handler(ws) -> None:
    log.info("ESP32 connected: %s", ws.remote_address)
    try:
        async for msg in ws:
            if not isinstance(msg, (bytes, bytearray)):
                continue
            data = bytes(msg)
            if data[:2] == b"\xff\xd8":     # phase-1 raw JPEG
                jpeg = data
            elif data and data[0] == 0x01:  # phase-3 tagged JPEG
                jpeg = data[1:]
            else:
                continue
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                _set_latest(frame)
    finally:
        log.info("ESP32 disconnected")


def _ws_producer(host: str, port: int) -> None:
    import websockets

    async def _serve():
        async def _h(ws):
            await _ws_handler(ws)
        server = await websockets.serve(_h, host=host, port=port,
                                        max_size=8 * 1024 * 1024,
                                        ping_interval=20, ping_timeout=20)
        log.info("WebSocket source listening on ws://%s:%d", host, port)
        try:
            while not _stop_flag.is_set():
                await asyncio.sleep(0.2)
        finally:
            server.close()
            await server.wait_closed()

    try:
        asyncio.run(_serve())
    except Exception as e:
        log.error("ws producer crashed: %s: %s", type(e).__name__, e)
        _stop_flag.set()


# ============================================================
# Overlay
# ============================================================
def _draw_overlay(frame: np.ndarray,
                  fsm: FindGrabFSM,
                  dets: List[Detection],
                  target_det: Optional[Detection],
                  hand: Optional[HandResult]) -> np.ndarray:
    h, w = frame.shape[:2]
    for d in dets:
        color = (0, 255, 255) if d is target_det else (100, 200, 100)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        cv2.putText(frame, f"{d.label} {d.conf:.2f}", (d.x1, max(18, d.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if hand is not None:
        hx1 = int(hand.x1 * w); hy1 = int(hand.y1 * h)
        hx2 = int(hand.x2 * w); hy2 = int(hand.y2 * h)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
        tip = (int(hand.tip_x * w), int(hand.tip_y * h))
        cv2.circle(frame, tip, 6, (255, 0, 255), -1)
        palm = (int(hand.palm_x * w), int(hand.palm_y * h))
        cv2.circle(frame, palm, 4, (180, 0, 180), -1)
        cv2.putText(frame, "hand", (hx1, max(18, hy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (int(0.35 * w), int(0.35 * h)),
                  (int(0.65 * w), int(0.65 * h)), (80, 80, 80), 1)

    # Top banner — uses CJK renderer so 「目標: 杯子」 actually shows.
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    banner = (f"STATE: {fsm.state.name}  目標: {fsm.target_zh or '-'}"
              f"  streak: {fsm.center_streak}")
    frame = cjk_text(frame, banner, (8, 4), font_size=18, color_bgr=(0, 255, 0))

    # Bottom subtitle — recently heard STT text, ~2 s
    sub = fsm.get_subtitle()
    if sub:
        bar_h = 36
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        frame = cjk_text(frame, sub, (8, h - bar_h + 6),
                         font_size=22, color_bgr=(255, 255, 0))
    return frame


# ============================================================
# CLI argument schema
# ============================================================
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=["webcam", "ws"], default="webcam")
    p.add_argument("--webcam-index", type=int, default=0)
    p.add_argument("--ws-host", default="0.0.0.0")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--yolo-weights", default="yolov8s.pt")
    p.add_argument("--yolo-device", default="cpu")
    # Low raw conf lets the stabilizer see near-threshold frames so it can
    # smooth flicker via EMA + hysteresis. Bypass with --no-stabilize and you
    # should raise this back toward 0.35.
    p.add_argument("--yolo-conf", type=float, default=0.20)
    p.add_argument("--yolo-imgsz", type=int, default=960,
                   help="YOLO inference input size; larger = better on small objects, slower")
    p.add_argument("--no-stabilize", action="store_true",
                   help="disable the EMA+hysteresis detection stabilizer")
    p.add_argument("--mic", action="store_true", help="enable microphone STT")
    p.add_argument("--mic-vad", choices=["energy", "webrtc"], default="energy",
                   help="energy threshold (default) or webrtcvad if installed")
    p.add_argument("--stt-backend", choices=["whisper", "groq"], default="whisper",
                   help="local faster-whisper, or Groq cloud whisper-large-v3 (faster)")
    p.add_argument("--stt-model", default="tiny")
    p.add_argument("--llm-intent", action="store_true",
                   help="route ambiguous STT through an LLM (set GROQ_API_KEY or GEMINI_API_KEY)")
    p.add_argument("--llm-provider", choices=["groq", "openai", "gemini"], default="groq")
    p.add_argument("--voice-cues", action="store_true",
                   help="play pre-recorded WAV cues (向左/向右/找到啦…) for instant feedback")


# ============================================================
# Run
# ============================================================
def run(args: argparse.Namespace) -> None:
    detector = YoloDetector(args.yolo_weights, args.yolo_device, args.yolo_conf,
                            imgsz=args.yolo_imgsz,
                            stabilize=not args.no_stabilize)
    hands = HandsDetector(max_num_hands=1)
    fsm = FindGrabFSM(FindGrabConfig())

    if args.source == "webcam":
        t_src = threading.Thread(target=_webcam_producer, args=(args.webcam_index,),
                                 daemon=True, name="webcam")
    else:
        t_src = threading.Thread(target=_ws_producer, args=(args.ws_host, args.ws_port),
                                 daemon=True, name="ws")
    t_src.start()

    # ---- intent + label routing (regex first, LLM optional) ----
    from audio.intent_router import IntentRouter
    from vision.label_normalizer import LabelNormalizer
    router = IntentRouter(use_llm=args.llm_intent, llm_provider=args.llm_provider)
    normalizer = LabelNormalizer(use_llm=args.llm_intent, llm_provider=args.llm_provider)

    # Shared mic gate so the speaker (voice cues) can mute the microphone
    # while playing — stops the cues from being re-transcribed.
    mic_gate = MicGate()

    cues = None
    if args.voice_cues:
        from audio.voice_cues import VoiceCues
        cues = VoiceCues(gate=mic_gate); cues.start()

    def on_speech_text(text: str) -> None:
        # Last-mile s2t in case any upstream path slipped — idempotent.
        text = to_traditional(text)
        # Always show what we heard, regardless of how it routes.
        fsm.set_subtitle(f"🎤 {text}", duration_s=2.0)
        intent = router.route(text)
        # `target_zh` from the LLM is already s2t-normalised in
        # intent_router._llm_extract, but normalise here too as belt-and-braces.
        if intent.target_zh:
            intent.target_zh = to_traditional(intent.target_zh)
        log.info("STT='%s' -> intent=%s target=%s",
                 text, intent.kind, intent.target_zh)
        if intent.kind == "HOTWORD":
            fsm.hotword_reset()
            if cues: cues.play("cancel")
        elif intent.kind == "CANCEL":
            fsm.cancel()
            if cues: cues.play("cancel")
        elif intent.kind == "FOUND":
            if fsm.state == FGState.CONFIRM_GRAB:
                fsm.confirm_grab(True)
                if cues: cues.play("grabbed")
            elif fsm.state == FGState.GUIDING_HAND:
                fsm._goto(FGState.GRAB_SUCCESS)
                if cues: cues.play("grabbed")
        elif intent.kind == "NOT_YET":
            if fsm.state == FGState.CONFIRM_GRAB:
                fsm.confirm_grab(False)
        elif intent.kind == "FIND" and intent.target_zh:
            # Intent router may have already given us the YOLO label
            # (regex via labels.to_en, OR a single LLM call). Only fall back
            # to a separate normalize() round-trip when both paths missed.
            if intent.target_en:
                en, src = intent.target_en, "intent"
            else:
                en, src = normalizer.normalize(intent.target_zh)
            log.info("label '%s' -> '%s' (%s)", intent.target_zh, en, src)
            fsm.set_target(intent.target_zh, en)
            if cues: cues.play("searching")

    mic: Optional[MicListener] = None
    if args.mic:
        if args.stt_backend == "groq":
            from audio.groq_stt import GroqWhisperSTT
            stt = GroqWhisperSTT(language="zh")
        else:
            from audio.stt import WhisperSTT
            stt = WhisperSTT(model_size=args.stt_model, language="zh")
        mic = MicListener(on_text=on_speech_text, stt=stt,
                          vad=args.mic_vad, gate=mic_gate)
        mic.start()
        log.info("mic enabled (backend=%s, vad=%s, llm-intent=%s) — try saying 「幫我找我的杯子」",
                 args.stt_backend, args.mic_vad, args.llm_intent)
    else:
        log.info("mic disabled — press 't' in the window to simulate 「想要找杯子」")

    window = "Find & Grab (q to quit)"
    fsm._emit("[FSM] WAITING_FOR_COMMAND — 請說「想要找 XX」")

    # placeholder shown before first frame arrives
    _placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(_placeholder, "Waiting for ESP32...", (120, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(_placeholder, "ws source: --source ws", (140, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    try:
        while not _stop_flag.is_set():
            frame = _get_latest()
            if frame is None:
                cv2.imshow(window, _placeholder)
                cv2.waitKey(30)
                continue

            run_hands = fsm.state in (FGState.GUIDING_HAND, FGState.CONFIRM_GRAB)
            dets = detector.infer(frame)
            hand = hands.detect(frame) if run_hands else None

            fsm.step((frame.shape[1], frame.shape[0]), dets, hand)

            target_det = fsm._find_target(dets) if fsm.target_en else None
            overlay = _draw_overlay(frame, fsm, dets, target_det, hand)
            cv2.imshow(window, overlay)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            if k == ord('t'):
                fsm.on_speech("想要找杯子")
            elif k == ord('r'):
                fsm._reset()
    finally:
        _stop_flag.set()
        if mic:
            mic.stop()
        if cues:
            cues.stop()
        hands.close()
        cv2.destroyAllWindows()
