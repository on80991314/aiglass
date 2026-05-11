"""Gemini wrapper: intent parsing, translation, free-form chat.

Every method funnels its output through `to_traditional()` so that even when
Gemini ignores the 「請使用台灣繁體中文」 prompt and emits simplified
characters, the downstream FSM / TTS only ever sees traditional. Translate
output is exempt when the caller asks for a non-zh-TW target language.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# `cloud/` lives alongside `edge/` rather than inside it, so when this
# module is imported by edge code we need to make `audio` reachable.
_EDGE = Path(__file__).resolve().parents[2] / "edge"
if str(_EDGE) not in sys.path:
    sys.path.insert(0, str(_EDGE))

from audio.zh_normalizer import normalize_obj, to_traditional  # type: ignore[import-not-found]  # noqa: E402

log = logging.getLogger("gemini")


_INTENT_SYSTEM = (
    "你是一組智慧眼鏡的語意路由器。"
    "**所有自然語言輸出（object、destination 等）必須使用台灣繁體中文，"
    "禁止輸出簡體字。**"
    "讀取使用者的中文句子，回傳 JSON (不加 markdown)："
    '{"intent": "NAV_TO|FIND_OBJECT|TRANSLATE_TO|CANCEL|CHAT", '
    '"object": "...", "destination": "...", "lang": "..."}。'
    "intent 必填；其餘欄位只在需要時出現。"
)

_ZHTW_HINT = "（請使用台灣繁體中文回答，不要使用簡體字。）"


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        import google.generativeai as genai
        self._genai = genai
        self._genai.configure(api_key=api_key)
        self.model_name = model
        self._model = self._genai.GenerativeModel(model)

    async def parse_intent(self, text: str) -> dict[str, Any]:
        prompt = f"{_INTENT_SYSTEM}\n\n使用者: {text}\nJSON:"
        resp = await asyncio.to_thread(self._model.generate_content, prompt)
        raw = (resp.text or "").strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        try:
            obj = json.loads(raw)
        except Exception:
            log.warning("gemini intent parse failed: %r", raw)
            return {"intent": "CHAT"}
        # s2t every string field — LLMs ignore zh-TW instructions ~10% of
        # the time and emit simplified.
        return normalize_obj(obj)

    async def translate(self, text: str, target_lang: str = "zh-TW") -> str:
        prompt = (f"Translate the following to {target_lang}. "
                  f"Reply with the translation only, no quotes, no commentary.\n\n{text}")
        resp = await asyncio.to_thread(self._model.generate_content, prompt)
        out = (resp.text or "").strip()
        # Only force traditional when the user actually asked for zh / zh-TW.
        if target_lang.lower().replace("_", "-") in {"zh", "zh-tw", "zh-hant", "繁體中文", "zh-hk"}:
            out = to_traditional(out)
        return out

    async def chat(self, text: str) -> str:
        prompt = f"{text}\n\n{_ZHTW_HINT}"
        resp = await asyncio.to_thread(self._model.generate_content, prompt)
        return to_traditional((resp.text or "").strip())

    async def describe_scene(self, jpeg_bytes: bytes,
                             question: str = "用一句話描述前方場景。") -> str:
        """Ask Gemini about a frame — useful for fall-check, landmark lookup."""
        full_q = f"{question} {_ZHTW_HINT}"
        resp = await asyncio.to_thread(
            self._model.generate_content,
            [{"mime_type": "image/jpeg", "data": jpeg_bytes}, full_q],
        )
        return to_traditional((resp.text or "").strip())
