"""Two-stage intent router for the find-grab pipeline.

Stage 1: regex (parse_find_command + hotword/cancel/found/not_yet patterns)
         — instant, handles 「想要找 XX」 / 「拿到了」 / 「還沒」 / 「停下」.

Stage 2: LLM fallback (Groq Llama / Gemini) — handles natural phrasing like
         「我的鑰匙不見了」、「能幫我看看我的杯子在哪嗎」 etc.

The LLM call returns BOTH the Mandarin object and the YOLO English label
in a single round-trip, replacing the previous two-call sequence
(intent -> normalize). Use intent.target_en when set; only fall back to
LabelNormalizer when target_en is empty.

Intents:
    FIND       (target_zh, target_en) — start a new search
    CANCEL                              — stop / reset
    HOTWORD                             — emergency stop
    FOUND                               — user confirms they grabbed it
    NOT_YET                             — user says they did NOT grab it yet
    NONE                                — irrelevant chatter, ignore
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from audio.zh_normalizer import to_traditional
from vision.labels import parse_find_command, to_en

log = logging.getLogger("intent")

HOTWORDS = {"停下", "停止", "別說了", "别说了", "不要說了", "不要说了", "閉嘴", "闭嘴"}
CANCEL_RE = re.compile(r"(取消|算了|不找了|結束|结束|不用了|停止)")
# NOT_YET is checked BEFORE FOUND so 「還沒拿到」 doesn't match 「拿到」.
NOT_YET_RE = re.compile(r"(還沒|还没|沒拿到|没拿到|沒有拿到|没有拿到|沒抓到|没抓到|尚未|不對|不对|錯了|错了)")
FOUND_RE  = re.compile(r"(拿到了|拿到啦|抓到了|抓到啦|找到了|找到啦|有了|可以了|好了|OK|ok)")

_LLM_PROMPT = (
    "你是智慧眼鏡的指令路由器。讀取使用者中文，回 JSON（不加 markdown）。\n"
    "**所有 object 欄位必須使用台灣繁體中文，不可輸出簡體字。**\n"
    "intent 可選: FIND（找東西）、CANCEL（取消）、CHAT（其他）。\n"
    "FIND 時必須附 object（物品繁體中文名詞，去掉「我的/那個」等修飾）以及 "
    "label（對應的 YOLO/COCO 英文 class，全小寫；常見如 cup, bottle, "
    "cell phone, laptop, keys, wallet, book, chair, bag。沒對應就用簡短 "
    "lowercase 英文，多字用底線連接，例如 red_bull）。\n"
    "範例：\n"
    '"想要找我的杯子" -> {"intent":"FIND","object":"杯子","label":"cup"}\n'
    '"我的鑰匙不見了" -> {"intent":"FIND","object":"鑰匙","label":"keys"}\n'
    '"幫我找紅牛" -> {"intent":"FIND","object":"紅牛","label":"red_bull"}\n'
    '"算了不找了" -> {"intent":"CANCEL"}\n'
    '"今天天氣不錯" -> {"intent":"CHAT"}\n'
)


@dataclass
class Intent:
    kind: str                          # FIND | CANCEL | HOTWORD | FOUND | NOT_YET | NONE
    target_zh: Optional[str] = None
    target_en: Optional[str] = None    # set by regex (via labels.to_en) or LLM
    raw: str = ""


class IntentRouter:
    """LLM is optional. With no LLM, only regex + hotwords work."""

    def __init__(self,
                 use_llm: bool = False,
                 llm_provider: str = "groq",       # "groq" | "openai" | "gemini"
                 api_key: Optional[str] = None,
                 model: Optional[str] = None) -> None:
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self._client = None
        self._model = model

        if not use_llm:
            return

        if llm_provider in ("groq", "openai"):
            from openai import OpenAI
            if llm_provider == "groq":
                key = api_key or os.environ.get("GROQ_API_KEY", "")
                base = "https://api.groq.com/openai/v1"
                self._model = model or "llama-3.1-8b-instant"
            else:
                key = api_key or os.environ.get("OPENAI_API_KEY", "")
                base = None
                self._model = model or "gpt-4o-mini"
            if not key:
                log.warning("LLM key empty; LLM router disabled")
                self.use_llm = False
                return
            self._client = OpenAI(api_key=key, base_url=base)
        elif llm_provider == "gemini":
            key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not key:
                log.warning("GEMINI_API_KEY empty; LLM router disabled")
                self.use_llm = False
                return
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._model = model or "gemini-2.5-flash"
            self._client = genai.GenerativeModel(self._model)

    # ------------ public ------------
    def route(self, text: str) -> Intent:
        text = (text or "").strip()
        if not text:
            return Intent(kind="NONE", raw=text)

        # 1. hotwords first — cheap and authoritative
        for w in HOTWORDS:
            if w in text:
                return Intent(kind="HOTWORD", raw=text)

        # 2. NOT_YET checked BEFORE FOUND (「還沒拿到」 contains 「拿到」)
        if NOT_YET_RE.search(text):
            return Intent(kind="NOT_YET", raw=text)

        # 3. user confirms grab
        if FOUND_RE.search(text):
            return Intent(kind="FOUND", raw=text)

        # 4. cancel / abort
        if CANCEL_RE.search(text):
            return Intent(kind="CANCEL", raw=text)

        # 5. fast-path regex find — fill target_en from local map if possible
        parsed = parse_find_command(text)
        if parsed:
            zh, en = parsed
            return Intent(kind="FIND", target_zh=zh, target_en=en, raw=text)

        # 6. LLM fallback (one round-trip — gives object AND label)
        if self.use_llm and self._client is not None:
            kind, zh, en = self._llm_extract(text)
            if kind == "FIND" and zh:
                # Prefer LLM-provided label, but if it looks bogus, try local map.
                en = en or to_en(zh) or zh
                return Intent(kind="FIND", target_zh=zh, target_en=en, raw=text)
            if kind == "CANCEL":
                return Intent(kind="CANCEL", raw=text)

        return Intent(kind="NONE", raw=text)

    # ------------ LLM ------------
    def _llm_extract(self, text: str) -> tuple[str, Optional[str], Optional[str]]:
        """Returns (intent_kind, target_zh, target_en). Empty fields when unknown."""
        try:
            if self.llm_provider == "gemini":
                resp = self._client.generate_content(_LLM_PROMPT + f"\n使用者：「{text}」\nJSON：")
                raw = (resp.text or "").strip()
            else:
                rsp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _LLM_PROMPT},
                        {"role": "user",   "content": text},
                    ],
                    temperature=0.0,
                )
                raw = (rsp.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning("LLM intent error: %s", e)
            return ("NONE", None, None)

        raw = raw.strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        try:
            obj = json.loads(raw)
        except Exception:
            log.warning("LLM intent non-JSON: %r", raw[:120])
            return ("NONE", None, None)

        kind = (obj.get("intent") or "").upper()
        if kind == "FIND":
            # LLMs sometimes ignore the "繁體" instruction — force s2t here.
            target = to_traditional((obj.get("object") or "").strip())
            label = (obj.get("label") or "").strip().lower() or None
            if target:
                return ("FIND", target, label)
        if kind == "CANCEL":
            return ("CANCEL", None, None)
        return ("NONE", None, None)
