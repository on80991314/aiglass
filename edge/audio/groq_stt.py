"""Groq Whisper STT client — v2（修正 prompt 滲漏 + 音量正規化）
 
問題根因（由診斷腳本確認）：
  1. Prompt 結尾含「請」字 → Whisper 把它接到辨識結果後面
     （測試 D 輸出「幫我過馬路，請」就是這個原因）
  2. In-memory WAV 送出後模型直接複製 prompt → 音量太低，
     Whisper 聽不到語音，fallback 到 prompt 繼續生成
     （測試 C 輸出整段 prompt 就是這個原因）
 
修正：
  1. Prompt 改用「指令詞列表」格式，不以動詞/助詞結尾，避免滲漏
  2. 送出前先做 peak normalization，確保音量足夠
  3. 加入 prompt bleed 偵測，把直接複製 prompt 的輸出過濾掉
  4. 保留 whisper-large-v3-turbo（測試 D 驗證可用）
"""
 
from __future__ import annotations
 
import io
import logging
import os
import wave
from typing import Optional
 
import numpy as np
 
log = logging.getLogger("groq_stt")
 
# ── Prompt 修正重點 ──────────────────────────────────────────────────────────
# 舊版結尾：「…請以繁體中文輸出，不要使用拼音或其他語言。」
#   → 結尾「請」字讓 Whisper 在輸出後補了一個「請」（prompt bleed）
#
# 新版：改用「常用指令詞列表」格式
#   → 不以動詞結尾，Whisper 不會「繼續說話」
#   → 把領域詞彙放進 vocabulary context，提升命中率
_ZH_PROMPT = (
    "繁體中文語音。常見詞：幫我找、過馬路、開始導航、"
    "停止、拿到了、找到了、想要找、物品、斑馬線、紅綠燈。"
)
 
_MIN_DURATION_S = 0.5
_TARGET_PEAK = 0.7   # peak normalize 目標（int16 最大值的 70%）
 
 
class GroqWhisperSTT:
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "whisper-large-v3-turbo",
                 language: str = "zh",
                 prompt: Optional[str] = None) -> None:
        api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY is empty (set env var or pass api_key=...)")
 
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key,
                              base_url="https://api.groq.com/openai/v1")
        self.model = model
        self.language = language
        self._prompt = prompt if prompt is not None else _ZH_PROMPT
        log.info("GroqWhisperSTT ready  model=%s  language=%s", model, language)
 
    # ── 修正 2：音量正規化 ────────────────────────────────────────────────────
    @staticmethod
    def _normalize(pcm: np.ndarray) -> np.ndarray:
        """Peak normalization：把最大振幅拉到 int16 最大值的 70%。
 
        診斷顯示 in-memory 音訊送到 Groq 時 Whisper 完全聽不到聲音，
        直接 fallback 輸出 prompt 內容（測試 C）。
        根本原因是麥克風收到的音量本身很低（-60dBFS 以下），
        不做正規化的話 Whisper 會把它當靜音處理。
        """
        pcm = pcm.astype(np.float32)
        peak = np.abs(pcm).max()
        if peak < 1.0:
            return pcm.astype(np.int16)   # 全靜音，不處理
        target = _TARGET_PEAK * 32767.0
        pcm = pcm * (target / peak)
        return np.clip(pcm, -32768, 32767).astype(np.int16)
 
    @staticmethod
    def _pcm_to_wav(pcm16_mono: np.ndarray, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm16_mono.tobytes())
        return buf.getvalue()
 
    # ── 修正 3：prompt bleed 偵測 ────────────────────────────────────────────
    def _strip_prompt_bleed(self, text: str) -> str:
        """如果輸出文字 60% 以上都是 prompt 的詞，視為無效辨識回傳空字串。
 
        測試 C 顯示：音量不足時模型直接輸出 prompt 文字。
        """
        if not text:
            return text
        prompt_tokens = set(
            self._prompt.replace("、", " ").replace("，", " ")
                        .replace("。", " ").split()
        )
        text_tokens = set(
            text.replace("，", " ").replace("。", " ").split()
        )
        if not text_tokens:
            return text
        overlap = len(prompt_tokens & text_tokens) / len(text_tokens)
        if overlap > 0.6:
            log.warning("prompt bleed detected (%.0f%% overlap) — discarding: %r",
                        overlap * 100, text)
            return ""
        return text
 
    # ── 主要辨識 ──────────────────────────────────────────────────────────────
    def transcribe_pcm(self, pcm16_mono: np.ndarray, sample_rate: int = 16000) -> str:
        duration_s = len(pcm16_mono) / max(sample_rate, 1)
        if duration_s < _MIN_DURATION_S:
            log.debug("clip too short (%.2fs), skip", duration_s)
            return ""
 
        # 音量正規化後再打包成 WAV
        pcm_norm = self._normalize(pcm16_mono)
        wav_bytes = self._pcm_to_wav(pcm_norm, sample_rate)
 
        energy = int(np.abs(pcm_norm).mean())
        log.debug("audio energy after normalize: %d  duration: %.2fs", energy, duration_s)
        if energy < 100:
            log.warning("audio energy still low (%d) after normalize — check mic", energy)
 
        try:
            resp = self._client.audio.transcriptions.create(
                model=self.model,
                file=("utt.wav", wav_bytes, "audio/wav"),
                language=self.language,
                response_format="json",
                prompt=self._prompt,
                temperature=0.0,
            )
            text = (resp.text or "").strip()
            log.debug("groq raw: %r", text)
 
            text = self._strip_prompt_bleed(text)
 
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