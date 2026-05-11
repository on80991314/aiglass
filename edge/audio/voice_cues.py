"""Pre-recorded WAV voice cues — instant audio feedback (no TTS latency).

Borrowed from OpenAIglasses_for_Navigation-main/voice/. We point at that
folder by default if it exists; otherwise the user can set VOICE_DIR.

Plays asynchronously via a single background thread so the FSM never
blocks waiting for a clip to finish.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from utils.paths import VOICE_DIRS
except ImportError:
    from ..utils.paths import VOICE_DIRS

from .mic_gate import MicGate

log = logging.getLogger("cues")


_DEFAULT_DIRS = [os.environ.get("VOICE_DIR"), *VOICE_DIRS]


# Phrase -> set of candidate filenames (try each, use first that exists).
# Phrases match the FSM emit text in spirit, not character-for-character.
_PHRASE_FILES = {
    "left":      ["向左.wav", "请向左转动。.wav", "左移.wav", "音频4.WAV"],
    "right":     ["向右.wav", "请向右转动。.wav", "右移.wav", "音频5.WAV"],
    "front":     ["向前.wav", "目标就在前方，请慢慢靠近。.wav", "音频7.WAV"],
    "up":        ["向上.wav"],
    "down":      ["向下.wav"],
    "center":    ["已对中.wav", "方向已对正！现在校准位置。.wav"],
    "found":     ["找到啦.wav", "目标就在前方，请慢慢靠近。.wav"],
    "grabbed":   ["拿到啦.wav", "已到达目标，引导结束。.wav", "寻物任务完成。.wav"],
    "searching": ["正在接近斑马线，为您对准方向。.wav"],   # generic "正在處理" fallback
    "lost":      ["目标消失，请原地小幅转动。.wav", "目标消失，请原地等待。.wav"],
    "cancel":    ["导航已被取消。.wav", "已停止导航。.wav"],
}


def _resolve_dirs() -> list[Path]:
    out: list[Path] = []
    for d in _DEFAULT_DIRS:
        if d is None:
            continue
        p = Path(d)
        if p.exists() and p.is_dir():
            out.append(p)
    return out


class VoiceCues:
    def __init__(self, gate: Optional[MicGate] = None) -> None:
        self.dirs = _resolve_dirs()
        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        self._q: queue.Queue[str] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # When set, suspend the mic listener for the clip's duration so the
        # speaker output is not picked up by the microphone and re-fed into STT.
        self._gate = gate
        if not self.dirs:
            log.warning("voice_cues: no voice dir found; cues disabled")
        else:
            log.info("voice_cues: searching in %s", [str(d) for d in self.dirs])
        self._sd_ok = False
        try:
            import sounddevice  # noqa: F401
            self._sd_ok = True
        except ImportError:
            log.warning("sounddevice missing; voice cues will not be audible")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="voice_cues", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait("__exit__")
        except queue.Full:
            pass

    def play(self, phrase: str) -> None:
        if not self.dirs or not self._sd_ok:
            return
        try:
            self._q.put_nowait(phrase)
        except queue.Full:
            pass  # backpressure — never block FSM

    # ------------ internals ------------
    def _find(self, phrase: str) -> Optional[Path]:
        names = _PHRASE_FILES.get(phrase, [])
        for d in self.dirs:
            for n in names:
                p = d / n
                if p.exists():
                    return p
        return None

    def _load(self, path: Path) -> Optional[tuple[np.ndarray, int]]:
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        try:
            with wave.open(str(path), "rb") as w:
                sr = w.getframerate()
                ch = w.getnchannels()
                sw = w.getsampwidth()
                raw = w.readframes(w.getnframes())
            if sw != 2:
                return None
            pcm = np.frombuffer(raw, dtype=np.int16)
            if ch > 1:
                pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
            self._cache[key] = (pcm, sr)
            return self._cache[key]
        except Exception as e:
            log.warning("voice_cues load fail %s: %s", path, e)
            return None

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            return
        while not self._stop.is_set():
            try:
                phrase = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if phrase == "__exit__":
                break
            p = self._find(phrase)
            if p is None:
                continue
            data = self._load(p)
            if data is None:
                continue
            pcm, sr = data
            # Mute the mic for the full clip duration so the playback
            # does not get re-transcribed as a user command.
            if self._gate is not None and sr > 0:
                self._gate.suspend(len(pcm) / float(sr))
            try:
                sd.play(pcm, sr, blocking=True)
            except Exception as e:
                log.warning("sd.play fail: %s", e)
