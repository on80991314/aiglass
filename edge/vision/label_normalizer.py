"""Chinese object name -> YOLO English class name.

Two stages:
  1. Local map (vision/labels.py ZH_TO_COCO + custom vocab)
  2. LLM fallback for unseen Chinese names ("我的紅牛" -> "Red_Bull"-ish)

Returns (label_en, source) where source in {'local', 'llm', 'fallback'}.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .labels import CN_TO_EN, to_en

log = logging.getLogger("label_norm")


_LLM_PROMPT = (
    "Convert the Chinese object name into a short lowercase English label "
    "suitable as a YOLO class name (1~3 words, no punctuation, snake_case "
    "if needed). Output ONLY the label."
)


class LabelNormalizer:
    def __init__(self,
                 use_llm: bool = False,
                 llm_provider: str = "groq",
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
                self.use_llm = False
                return
            self._client = OpenAI(api_key=key, base_url=base)
        elif llm_provider == "gemini":
            key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not key:
                self.use_llm = False
                return
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._model = model or "gemini-2.5-flash"
            self._client = genai.GenerativeModel(self._model)

    def normalize(self, query_cn: str) -> tuple[str, str]:
        q = (query_cn or "").strip()
        if not q:
            return ("", "fallback")

        # 1+2. local map (exact + containment), via labels.to_en
        en = to_en(q)
        if en:
            return (en, "local")

        # 3. LLM
        if self.use_llm and self._client is not None:
            label = self._llm_normalize(q)
            if label:
                return (label, "llm")

        # 4. fallback — return raw input
        return (q, "fallback")

    def _llm_normalize(self, q: str) -> Optional[str]:
        try:
            if self.llm_provider == "gemini":
                resp = self._client.generate_content(f"{_LLM_PROMPT}\nInput: {q}\nLabel:")
                raw = (resp.text or "").strip()
            else:
                rsp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _LLM_PROMPT},
                        {"role": "user",   "content": q},
                    ],
                    temperature=0.0,
                )
                raw = (rsp.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning("label-normalize LLM error: %s", e)
            return None
        # clean
        raw = raw.strip().strip(".,!?\"' ").lower().replace("  ", " ")
        if not raw or len(raw.split()) > 4:
            return None
        return raw
