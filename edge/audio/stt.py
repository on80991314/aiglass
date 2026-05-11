"""Speech-to-text wrapper using faster-whisper."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .zh_normalizer import to_traditional


class WhisperSTT:
    def __init__(self, model_size: str = "tiny", device: str = "cpu",
                 compute_type: str = "int8", language: str = "zh") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device,
                                       compute_type=self.compute_type)
        return self._model

    def transcribe_pcm(self, pcm16_mono: np.ndarray, sample_rate: int = 16000) -> str:
        """pcm16_mono: int16 numpy array at `sample_rate` Hz."""
        model = self._ensure_loaded()
        audio_f32 = pcm16_mono.astype(np.float32) / 32768.0
        if sample_rate != 16000:
            # Faster-whisper accepts any sample rate via resampling internally,
            # but keep this explicit for reproducibility.
            import librosa
            audio_f32 = librosa.resample(audio_f32, orig_sr=sample_rate, target_sr=16000)
        segments, _ = model.transcribe(
            audio_f32, language=self.language,
            vad_filter=True, beam_size=1,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return to_traditional(text) if self.language.startswith("zh") else text
