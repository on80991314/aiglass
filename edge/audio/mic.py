"""PC microphone capture with VAD + Whisper STT.

Runs in a background thread. Whenever it detects a completed utterance
(brief silence after speech), it transcribes and fires the callback.
"""

from __future__ import annotations

import logging
import threading
import time
import queue  # 引入標準庫佇列
from typing import Callable, Optional

import numpy as np

from .mic_gate import MicGate
from .zh_normalizer import to_traditional

log = logging.getLogger("mic")


class _EnergyVad:
    def __init__(self, thresh: int) -> None:
        self.thresh = thresh

    def is_speech(self, pcm: np.ndarray, sample_rate: int) -> bool:
        return int(np.abs(pcm).mean()) > self.thresh


class _WebRtcVad:
    """webrtcvad accepts only 10/20/30 ms frames at 8/16/32/48 kHz int16."""

    _ACCEPTED_RATES = {8000, 16000, 32000, 48000}
    _FRAME_MS = 20

    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, pcm: np.ndarray, sample_rate: int) -> bool:
        if sample_rate not in self._ACCEPTED_RATES:
            return False
        frame_samples = int(sample_rate * self._FRAME_MS / 1000)
        b = pcm.astype(np.int16).tobytes()
        any_speech = False
        for i in range(0, len(pcm) - frame_samples + 1, frame_samples):
            frame = b[i * 2:(i + frame_samples) * 2]
            try:
                if self._vad.is_speech(frame, sample_rate):
                    any_speech = True
                    break
            except Exception:
                return False
        return any_speech


class MicListener:
    def __init__(self,
                 on_text: Callable[[str], None],
                 sample_rate: int = 16000,
                 chunk_ms: int = 100,
                 silence_after_speech_s: float = 0.8,
                 max_utt_s: float = 8.0,
                 energy_thresh: int = 500,
                 vad: str = "energy",
                 preroll_s: float = 0.4,
                 stt=None,
                 gate: Optional[MicGate] = None) -> None:
        self.on_text = on_text
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.silence_after_speech_s = silence_after_speech_s
        self.max_utt_s = max_utt_s
        self.energy_thresh = energy_thresh
        
        chunk_s = self.chunk_samples / self.sample_rate
        self._preroll_chunks = max(1, int(round(preroll_s / chunk_s)))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stt = stt  
        self._gate = gate
        self._vad_kind = vad
        self._vad = self._make_vad(vad, energy_thresh)

    def _make_vad(self, kind: str, energy_thresh: int):
        if kind == "webrtc":
            try:
                v = _WebRtcVad(aggressiveness=2)
                log.info("mic VAD: webrtcvad")
                return v
            except ImportError:
                log.warning("webrtcvad not installed; falling back to energy VAD. "
                            "pip install webrtcvad-wheels")
        log.info("mic VAD: energy threshold=%d", energy_thresh)
        return _EnergyVad(energy_thresh)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _ensure_stt(self):
        if self._stt is None:
            from audio.stt import WhisperSTT
            self._stt = WhisperSTT(model_size="tiny", language="zh")
        return self._stt

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            log.error("sounddevice not installed; mic disabled. pip install sounddevice")
            return

        stt = self._ensure_stt()
        buf: list[np.ndarray] = []
        preroll: list[np.ndarray] = []
        speaking = False
        last_voice_t = 0.0
        utt_start = 0.0

        # 實作理由：使用執行緒安全的 Queue 作為底層硬體與 Python 邏輯之間的隔離緩衝區
        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            """PortAudio 原生底層 C 執行緒回呼，完全繞過 Python GIL 鎖限制。"""
            if status:
                log.debug("Sounddevice status: %s", status)
            # 將採集到的 Mono 數據複製並安全壓入佇列，確保主執行緒卡頓重推理時，音訊絕不丟失
            audio_queue.put(indata[:, 0].copy() if indata.ndim == 2 else indata.copy())

        log.info("mic listening at %d Hz (preroll=%d chunks)",
                 self.sample_rate, self._preroll_chunks)
        try:
            # 使用非阻塞式 InputStream 並傳入 callback
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=1, dtype="int16",
                                blocksize=self.chunk_samples,
                                callback=audio_callback):
                
                while not self._stop.is_set():
                    try:
                        # 從佇列非阻塞獲取音訊段，超時設定為 50ms 確保迴圈能定期響應 stop 事件
                        pcm = audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    # 揚聲器播報期隔離閘
                    if self._gate is not None and self._gate.is_suspended():
                        if speaking:
                            buf.clear()
                            speaking = False
                        preroll.clear()
                        continue

                    is_speech = self._vad.is_speech(pcm, self.sample_rate)
                    now = time.monotonic()

                    if is_speech:
                        if not speaking:
                            speaking = True
                            utt_start = now
                            buf = list(preroll)
                            preroll.clear()
                        last_voice_t = now
                        buf.append(pcm)
                    elif speaking:
                        buf.append(pcm)
                        silence = now - last_voice_t
                        too_long = now - utt_start > self.max_utt_s
                        if silence >= self.silence_after_speech_s or too_long:
                            audio = np.concatenate(buf) if buf else np.zeros(0, dtype=np.int16)
                            buf.clear()
                            speaking = False
                            if len(audio) > self.sample_rate * 0.3:
                                try:
                                    text = stt.transcribe_pcm(audio, self.sample_rate)
                                except Exception as e:
                                    log.warning("stt failed: %s", e)
                                    text = ""
                                
                                text = to_traditional((text or "").strip())
                                if text:
                                    log.info("heard: %s", text)
                                    try:
                                        self.on_text(text)
                                    except Exception:
                                        log.exception("on_text callback crashed")

                    if not speaking:
                        preroll.append(pcm)
                        if len(preroll) > self._preroll_chunks:
                            preroll.pop(0)
        except Exception:
            log.exception("mic thread crashed")