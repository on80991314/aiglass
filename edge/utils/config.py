"""Runtime configuration loaded from env / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


@dataclass
class Config:
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765

    yolo_weights: str = "yolov8n.pt"
    yolo_device: str = "cpu"          # "cpu" | "cuda:0"
    yolo_conf: float = 0.35

    stt_model: str = "tiny"           # faster-whisper model size
    stt_language: str = "zh"

    tts_voice: str = "zh-TW-HsiaoChenNeural"
    tts_backend: str = "pyttsx3"      # "pyttsx3" | "edge_tts"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    gmap_api_key: str = ""
    gmap_language: str = "zh-TW"

    # Speech pacing — matches plan §4 "語音節流 3-7 秒"
    speech_min_gap_s: float = 3.0
    speech_repeat_gap_s: float = 7.0

    # Fall-alert contact (optional, for SMS / Line Notify / Telegram)
    emergency_contact: str = ""
    notify_token: str = ""

    extras: dict = field(default_factory=dict)


def load(dotenv_path: str | os.PathLike | None = None) -> Config:
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    _load_dotenv(Path(dotenv_path))

    def g(name: str, default: str = "") -> str:
        return os.environ.get(name, default)

    return Config(
        ws_host=g("WS_HOST", "0.0.0.0"),
        ws_port=int(g("WS_PORT", "8765")),
        yolo_weights=g("YOLO_WEIGHTS", "yolov8n.pt"),
        yolo_device=g("YOLO_DEVICE", "cpu"),
        yolo_conf=float(g("YOLO_CONF", "0.35")),
        stt_model=g("STT_MODEL", "tiny"),
        stt_language=g("STT_LANGUAGE", "zh"),
        tts_voice=g("TTS_VOICE", "zh-TW-HsiaoChenNeural"),
        tts_backend=g("TTS_BACKEND", "pyttsx3"),
        gemini_api_key=g("GEMINI_API_KEY", ""),
        gemini_model=g("GEMINI_MODEL", "gemini-2.5-flash"),
        gmap_api_key=g("GMAP_API_KEY", ""),
        gmap_language=g("GMAP_LANGUAGE", "zh-TW"),
        speech_min_gap_s=float(g("SPEECH_MIN_GAP_S", "3.0")),
        speech_repeat_gap_s=float(g("SPEECH_REPEAT_GAP_S", "7.0")),
        emergency_contact=g("EMERGENCY_CONTACT", ""),
        notify_token=g("NOTIFY_TOKEN", ""),
    )
