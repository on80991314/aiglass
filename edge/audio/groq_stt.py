"""Groq Whisper STT client — v3（改用實體檔案上傳，參考 aiglass3 驗證路徑）

問題根因（由 aiglass3 對比分析確認）：
  aiglass v2 使用 in-memory bytes 上傳：
      file=("utt.wav", wav_bytes, "audio/wav")
  → Groq API 對此方式辨識率不穩定，Whisper 有時無法正確讀取 WAV header，
    導致音訊被當作靜音，fallback 輸出 prompt 或空字串。

  aiglass3 驗證有效的方式：
      with open(file_path, "rb") as f:
          transcriptions.create(file=f, ...)
  → 以實體檔案 file handle 上傳，Groq API 能正確讀取 WAV。

修正策略（v3）：
  1. 先把 PCM 寫入系統暫存目錄的 .wav 檔（mkstemp 保證唯一）
  2. 改用 open(path, "rb") 以實體檔案方式上傳到 Groq
  3. 上傳後立刻刪除暫存檔
  4. 保留 peak normalization（避免靜音誤判）
  5. 保留 prompt bleed 偵測（過濾無效輸出）
  6. 保留優化後的中文 prompt 格式
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import wave
from typing import Optional

import numpy as np

log = logging.getLogger("groq_stt")

# ── Prompt（不以動詞/助詞結尾，避免 Whisper prompt bleed）──────────────────
_ZH_PROMPT = (
    "繁體中文語音。常見詞：幫我找、過馬路、開始導航、"
    "停止、拿到了、找到了、想要找、物品、斑馬線、紅綠燈。"
)

_MIN_DURATION_S = 0.5
_TARGET_PEAK = 0.7  # peak normalize 目標（int16 最大值的 70%）


class GroqWhisperSTT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3",  # v3: 改用 whisper-large-v3（aiglass3 驗證版本）
        language: str = "zh",
        prompt: Optional[str] = None,
    ) -> None:
        api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY is empty (set env var or pass api_key=...)")

        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model
        self.language = language
        self._prompt = prompt if prompt is not None else _ZH_PROMPT
        log.info("GroqWhisperSTT ready  model=%s  language=%s", model, language)

    # ── Peak Normalization ───────────────────────────────────────────────────
    @staticmethod
    def _normalize(pcm: np.ndarray) -> np.ndarray:
        """把最大振幅拉到 int16 最大值的 70%，確保 Whisper 能聽到聲音。"""
        pcm = pcm.astype(np.float32)
        peak = np.abs(pcm).max()
        if peak < 1.0:
            return pcm.astype(np.int16)  # 全靜音，不處理
        target = _TARGET_PEAK * 32767.0
        pcm = pcm * (target / peak)
        return np.clip(pcm, -32768, 32767).astype(np.int16)

    # ── 寫入暫存 WAV 檔 ──────────────────────────────────────────────────────
    @staticmethod
    def _write_temp_wav(pcm16_mono: np.ndarray, sample_rate: int) -> str:
        """將 PCM 資料寫入暫存 .wav 檔，回傳檔案路徑。"""
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="asr_utt_")
        try:
            os.close(fd)
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(pcm16_mono.tobytes())
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass
            raise
        return path

    # ── Prompt Bleed 偵測 ────────────────────────────────────────────────────
    def _strip_prompt_bleed(self, text: str) -> str:
        """若輸出文字 60% 以上都是 prompt 的詞，視為無效辨識，回傳空字串。"""
        if not text:
            return text
        prompt_tokens = set(
            self._prompt.replace("、", " ").replace("，", " ").replace("。", " ").split()
        )
        text_tokens = set(text.replace("，", " ").replace("。", " ").split())
        if not text_tokens:
            return text
        overlap = len(prompt_tokens & text_tokens) / len(text_tokens)
        if overlap > 0.6:
            log.warning(
                "prompt bleed detected (%.0f%% overlap) — discarding: %r",
                overlap * 100,
                text,
            )
            return ""
        return text

    # ── 主要辨識（v3 核心修正：實體檔案上傳）────────────────────────────────
    def transcribe_pcm(self, pcm16_mono: np.ndarray, sample_rate: int = 16000) -> str:
        duration_s = len(pcm16_mono) / max(sample_rate, 1)
        if duration_s < _MIN_DURATION_S:
            log.debug("clip too short (%.2fs), skip", duration_s)
            return ""

        # 1. Peak normalize
        pcm_norm = self._normalize(pcm16_mono)

        energy = int(np.abs(pcm_norm).mean())
        log.debug("audio energy after normalize: %d  duration: %.2fs", energy, duration_s)
        if energy < 100:
            log.warning("audio energy still low (%d) after normalize — check mic", energy)

        # 2. 寫入暫存檔（aiglass3 驗證路徑：實體檔案上傳）
        tmp_path: Optional[str] = None
        try:
            tmp_path = self._write_temp_wav(pcm_norm, sample_rate)
            log.debug("temp wav written: %s", tmp_path)

            # 3. 以實體檔案 open() 方式上傳（關鍵修正）
            with open(tmp_path, "rb") as audio_file:
                resp = self._client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language,
                    response_format="json",
                    prompt=self._prompt,
                    temperature=0.0,
                )

            text = (resp.text or "").strip()
            log.debug("groq raw: %r", text)

            # 4. Prompt bleed 過濾
            text = self._strip_prompt_bleed(text)

            # 5. 繁體中文正規化
            if self.language.startswith("zh") and text:
                from .zh_normalizer import to_traditional
                text = to_traditional(text)

            return text

        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status:
                log.warning("groq stt HTTP %s: %s", status, e)
            else:
                log.warning("groq stt error: %s", e)
            return ""

        finally:
            # 6. 清理暫存檔
            if tmp_path:
                try:
                    os.remove(tmp_path)
                    log.debug("temp wav removed: %s", tmp_path)
                except Exception:
                    pass

    # ── 從音檔路徑直接辨識（整合 aiglass3 的 recognize_speech_whisper）────────
    def transcribe_file(self, wav_path: str) -> str:
        """
        直接從 .wav 檔路徑辨識（對應 aiglass3 的 recognize_speech_whisper）。
        用於 ESP32 錄音後直接送檔案辨識的場景。
        """
        if not os.path.exists(wav_path):
            log.warning("transcribe_file: file not found: %s", wav_path)
            return ""
        try:
            log.info("[Groq Whisper] 上傳音檔進行辨識: %s", wav_path)
            with open(wav_path, "rb") as audio_file:
                resp = self._client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language,
                    response_format="json",
                    prompt=self._prompt,
                    temperature=0.0,
                )
            text = (resp.text or "").strip()
            log.debug("groq raw (file): %r", text)

            text = self._strip_prompt_bleed(text)

            if self.language.startswith("zh") and text:
                from .zh_normalizer import to_traditional
                text = to_traditional(text)

            return text
        except Exception as e:
            log.warning("groq stt (file) error: %s", e)
            return ""