"""Text-to-speech. Two backends:

  * pyttsx3  - offline, uses Windows SAPI voices. Zero cloud cost.
  * edge_tts - Microsoft Edge neural voices, much better zh-TW, needs internet.

Both produce 16-bit mono PCM ready to ship to the ESP32-S3 speaker.
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Optional

import numpy as np


class TTS:
    def __init__(self, backend: str = "pyttsx3",
                 voice: str = "zh-TW-HsiaoChenNeural",
                 sample_rate: int = 16000) -> None:
        self.backend = backend
        self.voice = voice
        self.sample_rate = sample_rate
        self._pyttsx3 = None

    # ------------ public ------------
    async def synthesize(self, text: str) -> np.ndarray:
        if self.backend == "edge_tts":
            return await self._edge(text)
        return await asyncio.to_thread(self._offline, text)

    # ------------ offline ------------
    def _offline(self, text: str) -> np.ndarray:
        if self._pyttsx3 is None:
            import pyttsx3
            self._pyttsx3 = pyttsx3.init()
            self._pyttsx3.setProperty("rate", 180)
        # pyttsx3 can't easily return raw bytes on Windows;
        # render to temp wav then read back.
        import tempfile
        import os
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            self._pyttsx3.save_to_file(text, tmp.name)
            self._pyttsx3.runAndWait()
            with wave.open(tmp.name, "rb") as w:
                sr = w.getframerate()
                n = w.getnframes()
                raw = w.readframes(n)
                ch = w.getnchannels()
                sw = w.getsampwidth()
            pcm = np.frombuffer(raw, dtype=np.int16)
            if ch > 1:
                pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
            if sr != self.sample_rate:
                pcm = _resample_linear(pcm, sr, self.sample_rate)
            return pcm
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # ------------ edge_tts ------------
    async def _edge(self, text: str) -> np.ndarray:
        import edge_tts
        comm = edge_tts.Communicate(text, voice=self.voice)
        buf = io.BytesIO()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        # edge-tts returns mp3; decode via pydub if available, else via soundfile
        return _decode_mp3(buf.getvalue(), self.sample_rate)


# -------- helpers --------
def _resample_linear(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return pcm
    ratio = dst / src
    n_out = int(len(pcm) * ratio)
    x_old = np.arange(len(pcm))
    x_new = np.linspace(0, len(pcm) - 1, n_out)
    return np.interp(x_new, x_old, pcm).astype(np.int16)


def _decode_mp3(mp3_bytes: bytes, target_sr: int) -> np.ndarray:
    try:
        from pydub import AudioSegment  # requires ffmpeg
        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        return np.frombuffer(seg.raw_data, dtype=np.int16)
    except Exception as e:
        raise RuntimeError(f"mp3 decode failed, install pydub+ffmpeg: {e}")
