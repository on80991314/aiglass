"""Groq Whisper STT client.

Drop-in replacement for `audio.stt.WhisperSTT` that pushes audio to the
Groq cloud (whisper-large-v3) instead of running the model locally.
Latency is typically 0.3-0.8s for short utterances, vs 3-8s on CPU.

Same interface as WhisperSTT:
    stt = GroqWhisperSTT(api_key=...)
    text = stt.transcribe_pcm(int16_pcm, sample_rate=16000)
"""

from __future__ import annotations

import io
import logging
import os
import wave
from typing import Optional

import numpy as np

from .zh_normalizer import to_traditional

log = logging.getLogger("groq_stt")


class GroqWhisperSTT:
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "whisper-large-v3",
                 language: str = "zh") -> None:
        api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY is empty (set env var or pass api_key=...)")
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key,
                              base_url="https://api.groq.com/openai/v1")
        self.model = model
        self.language = language

    @staticmethod
    def _pcm_to_wav(pcm16_mono: np.ndarray, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm16_mono.astype(np.int16).tobytes())
        return buf.getvalue()

    def transcribe_pcm(self, pcm16_mono: np.ndarray, sample_rate: int = 16000) -> str:
        wav_bytes = self._pcm_to_wav(pcm16_mono, sample_rate)
        try:
            resp = self._client.audio.transcriptions.create(
                model=self.model,
                file=("utt.wav", wav_bytes, "audio/wav"),
                language=self.language,
                response_format="json",
            )
            text = (resp.text or "").strip()
            return to_traditional(text) if self.language.startswith("zh") else text
        except Exception as e:
            log.warning("groq stt error: %s", e)
            return ""
