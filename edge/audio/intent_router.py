"""Two-stage intent router for the find-grab pipeline.

Stage 1: regex (parse_find_command + hotword/cancel/found/not_yet patterns,
         以及 aiglass3 驗證的導航指令)
         — instant, handles 「想要找 XX」 / 「拿到了」 / 「還沒」 / 「停下」
           / 「開始導航」 / 「過馬路」 / 「紅綠燈」 etc.

Stage 2: LLM fallback (Groq Llama / Gemini) — handles natural phrasing like
         「我的鑰匙不見了」、「能幫我看看我的杯子在哪嗎」 etc.

The LLM call returns BOTH the Mandarin object and the YOLO English label
in a single round-trip, replacing the previous two-call sequence
(intent -> normalize). Use intent.target_en when set; only fall back to
LabelNormalizer when target_en is empty.

Intents:
    FIND                    (target_zh, target_en) — start a new search
    CANCEL                                          — stop / reset
    HOTWORD                                         — emergency stop
    FOUND                                           — user confirms they grabbed it
    NOT_YET                                         — user says they did NOT grab it yet
    NONE                                            — irrelevant chatter, ignore

    # ── 以下為 aiglass3 整合新增 ──
    START_BLINDPATH_NAV                             — 開始盲道導航
    STOP_NAV                                        — 停止導航
    START_CROSSING                                  — 開始過馬路
    STOP_CROSSING                                   — 結束過馬路
    START_TRAFFIC_LIGHT                             — 啟動紅綠燈偵測
    STOP_TRAFFIC_LIGHT                              — 停止紅綠燈偵測
    VISUAL_QUERY                                    — 幫我看看這是什麼
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
CANCEL_RE  = re.compile(r"(取消|算了|不找了|結束|结束|不用了|停止)")
# NOT_YET is checked BEFORE FOUND so 「還沒拿到」 doesn't match 「拿到」.
NOT_YET_RE = re.compile(r"(還沒|还没|沒拿到|没拿到|沒有拿到|没有拿到|沒抓到|没抓到|尚未|不對|不对|錯了|错了)")
FOUND_RE   = re.compile(r"(拿到了|拿到啦|抓到了|抓到啦|找到了|找到啦|有了|可以了|好了|OK|ok)")

# ── aiglass3 整合：導航指令 regex（繁簡體兼容，停止類在啟動類之前） ──
_NAV_STOP_RE        = re.compile(r"停止導航|停止导航|結束導航|结束导航")
_NAV_BLINDPATH_RE   = re.compile(r"開始導航|开始导航|盲道導航|盲道导航|幫我導航|帮我导航")
_CROSSING_STOP_RE   = re.compile(r"過馬路結束|过马路结束|結束過馬路|结束过马路")
_CROSSING_START_RE  = re.compile(r"開始過馬路|开始过马路|幫我過馬路|帮我过马路")
_TRAFFIC_STOP_RE    = re.compile(r"停止檢測|停止检测|停止紅綠燈|停止红绿灯")
_TRAFFIC_START_RE   = re.compile(r"檢測紅綠燈|检测红绿灯|看紅綠燈|看红绿灯")
_VISUAL_QUERY_RE    = re.compile(r"幫我看看|帮我看看|幫我看下|帮我看下|這是什麼|这是什么|識別一下|识别一下")

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
                                       # | START_BLINDPATH_NAV | STOP_NAV
                                       # | START_CROSSING | STOP_CROSSING
                                       # | START_TRAFFIC_LIGHT | STOP_TRAFFIC_LIGHT
                                       # | VISUAL_QUERY
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

        # ── aiglass3 整合：導航指令（在 FIND regex 之前，停止類優先於啟動類） ──

        # 5a. 停止導航
        if _NAV_STOP_RE.search(text):
            return Intent(kind="STOP_NAV", raw=text)

        # 5b. 啟動盲道導航
        if _NAV_BLINDPATH_RE.search(text):
            return Intent(kind="START_BLINDPATH_NAV", raw=text)

        # 5c. 結束過馬路
        if _CROSSING_STOP_RE.search(text):
            return Intent(kind="STOP_CROSSING", raw=text)

        # 5d. 開始過馬路
        if _CROSSING_START_RE.search(text):
            return Intent(kind="START_CROSSING", raw=text)

        # 5e. 停止紅綠燈偵測
        if _TRAFFIC_STOP_RE.search(text):
            return Intent(kind="STOP_TRAFFIC_LIGHT", raw=text)

        # 5f. 啟動紅綠燈偵測
        if _TRAFFIC_START_RE.search(text):
            return Intent(kind="START_TRAFFIC_LIGHT", raw=text)

        # 5g. 視覺問答
        if _VISUAL_QUERY_RE.search(text):
            return Intent(kind="VISUAL_QUERY", raw=text)

        # 6. fast-path regex find — fill target_en from local map if possible
        parsed = parse_find_command(text)
        if parsed:
            zh, en = parsed
            return Intent(kind="FIND", target_zh=zh, target_en=en, raw=text)

        # 7. LLM fallback (one round-trip — gives object AND label)
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