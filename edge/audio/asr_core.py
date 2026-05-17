# edge/audio/asr_core.py
# -*- coding: utf-8 -*-
"""
從 aiglass3/asr_core.py 整合而來（已驗證語音辨識正確）。
 
提供兩種 ASR 後端：
  1. DashScope Paraformer 串流（即時，低延遲）
  2. Groq Whisper（離線音檔，高準確度）
 
並提供統一的 ASRCallback，負責：
  - 熱詞中斷（全清零複位）
  - partial 字幕推送到 UI
  - final 句子→觸發 intent_router → FSM 事件
"""
 
import os
import json
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple
 
from openai import OpenAI  # Groq 使用 openai-compatible API
 
# ───────────────────────── 環境變數 ─────────────────────────
ASR_DEBUG_RAW = os.getenv("ASR_DEBUG_RAW", "0") == "1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
 
# Groq client（用於 Whisper 離線辨識）
_groq_client: Optional[OpenAI] = None
if GROQ_API_KEY:
    _groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
 
# ───────────────────────── 熱詞設定 ─────────────────────────
INTERRUPT_KEYWORDS: set = set(
    os.getenv("INTERRUPT_KEYWORDS", "停下,別說了,停止,取消").split(",")
)
 
 
def _normalize_cn(s: str) -> str:
    """正規化中文字串（去空白、轉小寫）"""
    try:
        import unicodedata
        s = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in s)
    except Exception:
        pass
    return (s or "").strip().lower()
 
 
def _shorten(s: str, limit: int = 200) -> str:
    return s if len(s) <= limit else (s[:limit] + "…")
 
 
# ───────────────────────── DashScope 事件解析 ─────────────────────────
 
def _safe_to_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    for attr in ("to_dict", "model_dump", "__dict__"):
        try:
            v = getattr(x, attr, None)
        except Exception:
            v = None
        if callable(v):
            try:
                d = v()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        elif isinstance(v, dict):
            return v
    try:
        s = str(x)
        if s and s.lstrip().startswith("{") and s.rstrip().endswith("}"):
            return json.loads(s)
    except Exception:
        pass
    return {"_raw": str(x)}
 
 
def _extract_sentence(event_obj: Any) -> Tuple[Optional[str], Optional[bool]]:
    """從 DashScope 事件中提取 (text, is_sentence_end)"""
    d = _safe_to_dict(event_obj)
    cands: List[Dict[str, Any]] = [d]
    for k in ("output", "data", "result"):
        v = d.get(k)
        if isinstance(v, dict):
            cands.append(v)
    for obj in cands:
        sent = obj.get("sentence")
        if isinstance(sent, dict):
            text = sent.get("text")
            is_end = sent.get("sentence_end")
            if is_end is not None:
                is_end = bool(is_end)
            return text, is_end
    for obj in cands:
        if "text" in obj and isinstance(obj.get("text"), str):
            return obj.get("text"), None
    return None, None
 
 
# ───────────────────────── 全域 ASR 閘門 ─────────────────────────
_current_recognition: Optional[object] = None
_rec_lock = asyncio.Lock()
 
 
async def set_current_recognition(r: Any) -> None:
    global _current_recognition
    async with _rec_lock:
        _current_recognition = r
 
 
async def stop_current_recognition() -> None:
    global _current_recognition
    async with _rec_lock:
        r = _current_recognition
        _current_recognition = None
    if r:
        try:
            r.stop()
        except Exception:
            pass
 
 
# ───────────────────────── ASRCallback ─────────────────────────
 
class ASRCallback:
    """
    統一 ASR 回調。
 
    設計原則（移植自 aiglass3，已驗證可用）：
      1. 熱詞命中 → 立刻全清零複位，不送 intent_router。
      2. AI 播報中 → partial 僅做 UI 展示，不觸發新一輪。
      3. 只有 sentence_end==True 的 final 才驅動 intent_router→FSM。
    """
 
    def __init__(
        self,
        *,
        on_sdk_error: Callable[[str], None],
        post: Callable[[asyncio.Future], None],
        ui_broadcast_partial: Callable[[str], asyncio.Future],
        ui_broadcast_final: Callable[[str], asyncio.Future],
        is_playing_now_fn: Callable[[], bool],
        on_final_text_fn: Callable[[str], asyncio.Future],   # ← 新：對應 aiglass 的 intent_router
        full_system_reset_fn: Callable[[str], asyncio.Future],
        interrupt_lock: asyncio.Lock,
    ):
        self._on_sdk_error = on_sdk_error
        self._post = post
        self._ui_partial = ui_broadcast_partial
        self._ui_final = ui_broadcast_final
        self._is_playing = is_playing_now_fn
        self._on_final_text = on_final_text_fn
        self._full_reset = full_system_reset_fn
        self._interrupt_lock = interrupt_lock
 
        self._last_partial: str = ""
        self._hot_interrupted: bool = False
 
    # ── DashScope SDK 回調 ──
    def on_open(self) -> None:
        pass
 
    def on_close(self) -> None:
        pass
 
    def on_complete(self) -> None:
        pass
 
    def on_error(self, err: Any) -> None:
        try:
            self._post(self._ui_partial(""))
            self._on_sdk_error(str(err))
        except Exception:
            pass
 
    def on_result(self, result: Any) -> None:
        self._handle(result)
 
    def on_event(self, event: Any) -> None:
        self._handle(event)
 
    # ── 內部 ──
    def _has_hotword(self, text: str) -> bool:
        t = _normalize_cn(text)
        if not t:
            return False
        return any(_normalize_cn(w) in t for w in INTERRUPT_KEYWORDS if w)
 
    def _handle(self, event: Any) -> None:
        if ASR_DEBUG_RAW:
            try:
                print("[ASR EVENT RAW]", json.dumps(_safe_to_dict(event), ensure_ascii=False), flush=True)
            except Exception:
                pass
 
        text, is_end = _extract_sentence(event)
        if text is None:
            return
        text = text.strip()
        if not text:
            return
 
        # ① 熱詞優先
        if not self._hot_interrupted and self._has_hotword(text):
            self._hot_interrupted = True
 
            async def _hot_reset() -> None:
                async with self._interrupt_lock:
                    print(f"[ASR HOTWORD] '{text}' → FULL RESET", flush=True)
                    await self._full_reset("Hotword interrupt")
 
            try:
                self._post(_hot_reset())
            except Exception:
                pass
            return
 
        # ② partial → UI only
        self._last_partial = text
        try:
            print(f"[ASR PARTIAL] '{_shorten(text)}'", flush=True)
            self._post(self._ui_partial(text))
        except Exception:
            pass
 
        # ③ final → intent_router（若未播報中）
        if is_end is True:
            final_text = text
            try:
                print(f"[ASR FINAL] '{final_text}'", flush=True)
                self._post(self._ui_final(final_text))
            except Exception:
                pass
 
            if not self._is_playing() and final_text:
                async def _run_final() -> None:
                    async with self._interrupt_lock:
                        print(f"[INTENT INPUT] {final_text}", flush=True)
                        await self._on_final_text(final_text)
 
                try:
                    self._post(_run_final())
                except Exception:
                    pass
 
            # 重置進入下一句
            self._last_partial = ""
            self._hot_interrupted = False
 
 
# ───────────────────────── Groq Whisper（離線） ─────────────────────────
 
def recognize_speech_whisper(audio_file_path: str) -> str:
    """
    呼叫 Groq Whisper API 將音檔轉為文字。
    （移植自 aiglass3/asr_core.py，已驗證）
    """
    if not _groq_client:
        print("[ASR WHISPER] 未設定 GROQ_API_KEY，跳過 Whisper 辨識")
        return ""
    if not os.path.exists(audio_file_path):
        print(f"[ASR WHISPER] 找不到音檔: {audio_file_path}")
        return ""
    try:
        with open(audio_file_path, "rb") as f:
            print(f"[Groq Whisper] 上傳音檔進行辨識: {audio_file_path}")
            transcription = _groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="zh",
                response_format="json",
            )
        return transcription.text or ""
    except Exception as e:
        print(f"[Groq API 錯誤] {e}")
        return ""
 
 
async def process_voice_file(
    wav_path: str,
    callback: ASRCallback,
    *,
    keyword_filter: Optional[List[str]] = None,
) -> None:
    """
    1. 用 Groq Whisper 辨識音檔。
    2. 可選關鍵字過濾（不符合就丟棄）。
    3. 通過熱詞/播報檢查後，呼叫 callback._on_final_text 觸發 intent_router。
 
    （移植自 aiglass3/asr_core.py::process_voice_file_to_ai）
 
    Args:
        wav_path:        ESP32 傳來的錄音路徑。
        callback:        已初始化的 ASRCallback 實例。
        keyword_filter:  若提供，文字必須包含其中一個關鍵字才繼續處理；
                         None 表示不過濾（處理所有語音）。
    """
    try:
        print(f"[ASR FILE] 開始處理音檔（Whisper）: {wav_path}")
        text = recognize_speech_whisper(wav_path)
        if not text:
            print("[ASR FILE] Whisper 辨識結果為空，跳過。")
            return
        text = text.strip()
 
        # 可選關鍵字過濾
        if keyword_filter is not None:
            if not any(kw in text for kw in keyword_filter):
                print(f"[ASR FILE] 未命中關鍵字，丟棄: {text}")
                return
 
        print(f"[ASR FILE] 辨識成功: {text}")
 
        # 熱詞中斷
        if callback._has_hotword(text):
            print(f"[ASR FILE HOTWORD] '{text}' → FULL RESET")
            async with callback._interrupt_lock:
                await callback._full_reset("Hotword interrupt from Whisper")
            return
 
        # UI 更新
        try:
            callback._post(callback._ui_final(text))
        except Exception as e:
            print(f"[ASR FILE] UI 更新失敗: {e}")
 
        # 觸發 intent_router（若未播報中）
        if not callback._is_playing():
            async with callback._interrupt_lock:
                print(f"[INTENT INPUT via Whisper] {text}", flush=True)
                await callback._on_final_text(text)
        else:
            print("[ASR FILE] AI 播報中，忽略此指令。")
 
    except Exception as e:
        print(f"[ASR FILE] 處理流程錯誤: {e}")